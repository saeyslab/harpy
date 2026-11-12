"""Prototype of a single-pass replacement for :meth:`harpy.utils.RasterAggregator.aggregate_stats`.

Two changes compared to :class:`~harpy.utils.RasterAggregator`:

1. All statistics in one pass. Each chunk returns, per label present in it, the count, the
   mean and the centered sums ``M2, M3, M4`` of every channel, plus the minimum, the maximum
   and the sum of the pixel coordinates when requested. Partial results are combined exactly.
   Minimum, maximum and coordinate sums combine trivially; the moments combine with the
   binomial expansion of ``sum((x - mean)**p)``: for partitions ``i`` with
   ``delta_i = mean_i - mean``,

   - ``M2 = sum(M2_i + n_i * delta_i**2)``
   - ``M3 = sum(M3_i + 3 * delta_i * M2_i + n_i * delta_i**3)``
   - ``M4 = sum(M4_i + 4 * delta_i * M3_i + 6 * delta_i**2 * M2_i + n_i * delta_i**4)``

   Every term is centered, so this is as numerically stable as the two-pass algorithm, but
   the image is read once instead of once per statistic.
2. Cheap and sparse per-chunk work. Labels are grouped with a lookup table over the label
   range of the chunk (``O(n)``) instead of a sort, and each chunk only returns the labels
   present in it, instead of a row for every label in the mask. A task handles one spatial
   chunk and as many of its channel chunks as fit in ``max_task_bytes``. The per-pixel kernels
   and the merge of partial results are compiled loops that release the GIL; the merge adds
   every partial straight into the output arrays, without temporaries of the output size.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Literal

import dask
import dask.array as da
import numba
import numpy as np
import pandas as pd
from dask.utils import parse_bytes
from loguru import logger as log
from numpy.typing import NDArray

from harpy.utils._keys import _INSTANCE_KEY

_MOMENT_ORDER = {"count": 0, "sum": 1, "mean": 1, "var": 2, "skew": 3, "kurtosis": 4}
SUPPORTED_STATS = (*_MOMENT_ORDER, "min", "max", "center_of_mass")
_MASK_STATS = ("count", "center_of_mass")  # computed from the mask alone


@dataclass(frozen=True)
class _Request:
    """What every chunk computes."""

    order: int  # highest centered moment; 0 means only pixel counts
    minimum: bool
    maximum: bool
    center_of_mass: bool

    @classmethod
    def from_stats(cls, stats_funcs: Sequence[str]) -> _Request:
        return cls(
            order=max(_MOMENT_ORDER.get(s, 0) for s in stats_funcs),
            minimum="min" in stats_funcs,
            maximum="max" in stats_funcs,
            center_of_mass="center_of_mass" in stats_funcs,
        )

    @property
    def reads_image(self) -> bool:
        return self.order > 0 or self.minimum or self.maximum


@dataclass
class _Partial:
    """Statistics of the labels present in a part of the mask, one row per label (sorted)."""

    labels: NDArray
    counts: NDArray
    mean: NDArray | None = None  # (k, c), float64
    m2: NDArray | None = None  # centered sums of powers, (k, c), float64
    m3: NDArray | None = None
    m4: NDArray | None = None
    minimum: NDArray | None = None  # (k, c), image dtype
    maximum: NDArray | None = None
    coordinate_sum: NDArray | None = None  # (k, 3): sum of global z, y, x, float64


_CHANNEL_FIELDS = ("mean", "m2", "m3", "m4", "minimum", "maximum")


class MomentAggregator:
    """
    Prototype: per-label statistics of :meth:`harpy.utils.RasterAggregator.aggregate_stats`, plus minimum, maximum and center of mass, in one pass over the image.

    Same arguments and output as :meth:`harpy.utils.RasterAggregator.aggregate_stats`. Runs on
    CPU. Without an image, only the mask statistics ``"count"`` and ``"center_of_mass"`` are
    available.

    Each task reads the mask of one spatial chunk and at most `max_task_bytes` of image data:
    the channel chunks of a spatial chunk are split into batches of consecutive chunks, and
    each batch groups the labels again. A channel chunk larger than `max_task_bytes` forms a
    batch on its own. The image is never rechunked, so its chunks are best aligned with the
    chunks on disk.

    Parameters
    ----------
    mask_dask_array
        Integer labels of shape ``('z', 'y', 'x')``.
    image_dask_array
        Image of shape ``('c', 'z', 'y', 'x')``, with the same spatial chunks as the mask. If
        `None`, only ``"count"`` and ``"center_of_mass"`` can be computed.
    instance_key
        Name of the label id column in the returned DataFrames.
    split_every
        Number of partial results merged per task in the tree reduction.
    engine
        Per-chunk kernels and merge of partial results. ``"numba"`` (default) runs compiled
        loops that release the GIL, so Dask threads run them in parallel. ``"numpy"`` uses
        :func:`numpy.bincount` and :func:`numpy.minimum.at`, which hold the GIL and allocate
        temporaries per channel, and merges with numpy expressions, which allocate temporaries
        of the size of the merged result.
    max_task_bytes
        Most image data one task reads, as a number of bytes or a string such as ``"256MiB"``.
        Smaller values lower the memory per task when a spatial chunk holds many channels, but
        every extra batch groups the labels of the spatial chunk again, which costs time.
    """

    def __init__(
        self,
        mask_dask_array: da.Array,
        image_dask_array: da.Array | None = None,
        instance_key: str = _INSTANCE_KEY,
        split_every: int = 8,
        engine: Literal["numba", "numpy"] = "numba",
        max_task_bytes: int | str = "256MiB",
    ):
        if engine not in ("numba", "numpy"):
            raise ValueError(f"'engine' is '{engine}'; choose 'numba' or 'numpy'.")
        task_bytes = parse_bytes(max_task_bytes)
        if task_bytes <= 0:
            raise ValueError(f"'max_task_bytes' is {max_task_bytes!r}; pass a positive size, e.g. '256MiB'.")
        if not np.issubdtype(mask_dask_array.dtype, np.integer):
            raise ValueError(
                f"'mask_dask_array' has dtype '{mask_dask_array.dtype}', but labels need an integer dtype. "
                "Cast it, e.g. 'mask_dask_array.astype(np.int32)'."
            )
        if mask_dask_array.ndim != 3:
            raise ValueError(f"Expected a mask of shape ('z', 'y', 'x'), got shape {mask_dask_array.shape}.")
        if image_dask_array is not None:
            _check_image(image_dask_array, mask_dask_array)
        self._mask = mask_dask_array
        self._image = image_dask_array
        self._instance_key = instance_key
        self._split_every = split_every
        self._engine = engine
        self._max_task_bytes = task_bytes

    def aggregate_stats(
        self,
        stats_funcs: str | Sequence[str] = ("sum", "mean", "count", "var", "kurtosis", "skew"),
        index: NDArray | None = None,
    ) -> list[pd.DataFrame]:
        """
        Compute the requested statistics for every label in one pass over the image.

        Parameters
        ----------
        stats_funcs
            Any of ``"sum"``, ``"mean"``, ``"count"``, ``"var"``, ``"kurtosis"``, ``"skew"``,
            ``"min"``, ``"max"`` and ``"center_of_mass"``.
        index
            Labels to return. If `None`, every label in the mask, including background.

        Returns
        -------
        One DataFrame per statistic, in the order of `stats_funcs`, as returned by
        :meth:`harpy.utils.RasterAggregator.aggregate_stats`, with a column per channel and a
        column with the label id. Exceptions:

        - ``"count"`` has one column with the number of pixels.
        - ``"min"`` and ``"max"`` keep the image dtype. As in
          :meth:`harpy.utils.RasterAggregator.aggregate_min`, a label of `index` that is absent
          from the mask gets the largest (``"min"``) or smallest (``"max"``) value of that dtype.
        - ``"center_of_mass"`` has columns ``0, 1, 2`` with the mean ``z, y, x`` pixel
          coordinate in float64, as :meth:`harpy.utils.RasterAggregator.center_of_mass`.
        """
        if isinstance(stats_funcs, str):
            stats_funcs = (stats_funcs,)
        invalid = [s for s in stats_funcs if s not in SUPPORTED_STATS]
        if invalid:
            raise ValueError(
                f"Unsupported statistic(s) {invalid} in 'stats_funcs'. Choose from {list(SUPPORTED_STATS)}."
            )
        request = _Request.from_stats(stats_funcs)
        if request.reads_image and self._image is None:
            image_stats = [s for s in stats_funcs if s not in _MASK_STATS]
            raise ValueError(
                f"Statistic(s) {image_stats} in 'stats_funcs' need an image, but this MomentAggregator has none. "
                f"Pass 'image_dask_array' when creating it, or request only {list(_MASK_STATS)}."
            )

        mask_blocks = self._mask.to_delayed().ravel()
        # (channel chunks, spatial chunks), same spatial order as mask_blocks
        image_blocks = np.empty((0, mask_blocks.size), dtype=object)
        batches = [[]]
        if request.reads_image:
            image_blocks = self._image.to_delayed().reshape(self._image.numblocks[0], -1)
            channel_nbytes = int(np.prod(self._image.chunksize[1:])) * self._image.dtype.itemsize
            batches = _channel_batches(self._image.chunks[0], channel_nbytes, self._max_task_bytes)
        starts = [np.cumsum((0, *sizes[:-1])) for sizes in self._mask.chunks]
        partials = []
        for i, block_index in enumerate(np.ndindex(*self._mask.numblocks)):
            offset = None
            if request.center_of_mass:
                offset = tuple(int(starts[axis][b]) for axis, b in enumerate(block_index))
            # One task per spatial chunk and batch of channel chunks groups the labels and handles
            # the channel chunks of its batch, so no label grouping is passed between tasks. The
            # image blocks go in as separate arguments: for a list argument holding Delayed objects,
            # dask culls the whole graph on every call. The batches of a spatial chunk are joined
            # right away, so dask runs them together and can release the mask block.
            parts = [
                dask.delayed(_chunk_partial)(
                    mask_blocks[i], offset if b == 0 else None, request, self._engine, *image_blocks[batch, i]
                )
                for b, batch in enumerate(batches)
            ]
            partials.append(parts[0] if len(parts) == 1 else dask.delayed(_join_channels)(*parts))
        # Every task is needed, so skip graph optimization: each delayed call is its own graph
        # layer, and culling thousands of layers takes seconds. `to_delayed` above has already
        # optimized the graphs of the mask and the image.
        merge = _merge_numba if self._engine == "numba" else _merge
        (result,) = dask.compute(_tree_reduce(partials, self._split_every, merge), optimize_graph=False)

        if index is None:
            index = result.labels
        index = np.asarray(index)
        # every pixel carries a label (background included), so `labels` is never empty
        position = np.minimum(np.searchsorted(result.labels, index), result.labels.size - 1)
        found = result.labels[position] == index
        n = np.where(found, result.counts[position], 0).astype(np.float64)[:, None]

        def take(values: NDArray, fill: float | int) -> NDArray:
            return np.where(found[:, None], values[position], np.asarray(fill, dtype=values.dtype))

        with np.errstate(divide="ignore", invalid="ignore"):
            out = {"count": n}
            if request.order >= 1:
                mean = take(result.mean, np.nan)
                out["mean"] = mean
                out["sum"] = np.where(n > 0, mean * n, 0.0)
            if request.order >= 2:
                var = take(result.m2, np.nan) / n
                out["var"] = var
            if request.order >= 3:
                out["skew"] = _replace_nan(take(result.m3, np.nan) / n / var**1.5, "skew")
            if request.order >= 4:
                out["kurtosis"] = _replace_nan(take(result.m4, np.nan) / n / var**2 - 3, "kurtosis")
            if request.minimum:
                out["min"] = take(result.minimum, _dtype_extremes(self._image.dtype)[1])
            if request.maximum:
                out["max"] = take(result.maximum, _dtype_extremes(self._image.dtype)[0])
            if request.center_of_mass:
                out["center_of_mass"] = take(result.coordinate_sum, np.nan) / n

        dfs = []
        for stat in stats_funcs:
            values = out[stat]
            if stat not in ("min", "max", "center_of_mass"):
                values = values.astype(np.float32)
            df = pd.DataFrame(values)
            df[self._instance_key] = index
            dfs.append(df)
        return dfs


def _replace_nan(values: NDArray, name: str) -> NDArray:
    # same behaviour as RasterAggregator: zero variance gives NaN, which is replaced by 0
    if np.isnan(values).any():
        log.warning(f"Replacing NaN values in '{name}' with 0 for affected instances.")
        values = np.nan_to_num(values, nan=0)
    return values


def _check_image(image: da.Array, mask: da.Array) -> None:
    if image.ndim != 4:
        raise ValueError(f"Expected an image of shape ('c', 'z', 'y', 'x'), got shape {image.shape}.")
    if image.shape[1:] != mask.shape:
        raise ValueError(
            f"The spatial shape of 'image_dask_array' {image.shape[1:]} differs from the shape of "
            f"'mask_dask_array' {mask.shape}. Pass arrays that cover the same pixels."
        )
    if image.chunks[1:] != mask.chunks:
        if image.chunksize[1:] != mask.chunksize:
            difference = f"chunk sizes, {image.chunksize[1:]} and {mask.chunksize}"
        else:
            difference = f"chunk boundaries, with chunk size {mask.chunksize} for both"
        raise ValueError(
            f"'image_dask_array' and 'mask_dask_array' have different spatial {difference}. Rechunk the mask "
            "to the spatial chunks of the image: 'mask_dask_array.rechunk(image_dask_array.chunks[1:])'."
        )


def _dtype_extremes(dtype: np.dtype) -> tuple[float | int, float | int]:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return info.min, info.max
    return -np.inf, np.inf


def _channel_batches(channel_chunks: Sequence[int], channel_nbytes: int, budget: int) -> list[list[int]]:
    """Split channel chunks into batches of consecutive chunks of at most `budget` bytes per spatial chunk.

    `channel_nbytes` is the size of one channel of a spatial chunk. A channel chunk larger
    than `budget` forms a batch on its own.
    """
    batches, batch, size = [], [], 0
    for j, n_channels in enumerate(channel_chunks):
        nbytes = n_channels * channel_nbytes
        if batch and size + nbytes > budget:
            batches.append(batch)
            batch, size = [], 0
        batch.append(j)
        size += nbytes
    batches.append(batch)
    return batches


def _group_labels(block: NDArray, engine: str = "numba") -> tuple[NDArray, NDArray[np.intp], NDArray[np.intp]]:
    """Labels present in `block`, their pixel counts, and the dense label index of every pixel (C order)."""
    flat = block.reshape(-1)
    low, high = int(flat.min()), int(flat.max())
    span = high - low + 1
    if span <= max(2 * flat.size, 1 << 16) and not (flat.dtype.kind == "u" and flat.dtype.itemsize == 8):
        if engine == "numba":
            return _lookup_group_kernel(flat, low, span)
        # np.bincount holds the GIL, so this path scales poorly over Dask threads
        relative = flat.astype(np.intp) - low
        counts = np.bincount(relative, minlength=span)
        present = np.flatnonzero(counts)
        lookup = np.empty(span, dtype=np.intp)
        lookup[present] = np.arange(present.size)
        return (present + low).astype(block.dtype), counts[present], lookup[relative]
    labels, segments, counts = np.unique(flat, return_inverse=True, return_counts=True)
    return labels, counts, segments.reshape(-1)


@numba.njit(nogil=True, cache=True)
def _lookup_group_kernel(flat, low, span):  # pragma: no cover (compiled)
    # same result as the np.bincount lookup-table path of _group_labels, without holding the GIL
    counts = np.zeros(span, dtype=np.int64)
    for i in range(flat.size):
        counts[flat[i] - low] += 1
    lookup = np.empty(span, dtype=np.intp)
    k = 0
    for j in range(span):
        if counts[j] > 0:
            lookup[j] = k
            k += 1
    labels = np.empty(k, dtype=flat.dtype)
    present = np.empty(k, dtype=np.int64)
    k = 0
    for j in range(span):
        if counts[j] > 0:
            labels[k] = j + low
            present[k] = counts[j]
            k += 1
    segments = np.empty(flat.size, dtype=np.intp)
    for i in range(flat.size):
        segments[i] = lookup[flat[i] - low]
    return labels, present, segments


def _chunk_partial(
    mask_block: NDArray,
    offset: tuple[int, ...] | None,
    request: _Request,
    engine: str,
    *image_blocks: NDArray,
) -> _Partial:
    """Statistics of one spatial chunk for the channel chunks in `image_blocks`, and its coordinate sums if `offset` is given."""
    group = _group_labels(mask_block, engine)
    labels, counts, segments = group
    channel_partials = [_chunk_image_stats(group, block, request, engine) for block in image_blocks]
    partial = _join_channels(*channel_partials) if channel_partials else _Partial(labels, counts)
    if offset is not None:
        partial.coordinate_sum = _coordinate_sums(segments, mask_block.shape, labels.size, engine)
        partial.coordinate_sum += counts[:, None] * np.asarray(offset, dtype=np.float64)
    return partial


@numba.njit(nogil=True, cache=True)
def _coordinate_kernel(segments, nz, ny, nx, k):  # pragma: no cover (compiled)
    out = np.zeros((k, 3))
    i = 0
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                s = segments[i]
                out[s, 0] += z
                out[s, 1] += y
                out[s, 2] += x
                i += 1
    return out


def _coordinate_sums(segments: NDArray, shape: tuple[int, ...], k: int, engine: str) -> NDArray:
    """Sum of the block-local ``z, y, x`` coordinates per label, shape ``(k, 3)``."""
    if engine == "numba":
        return _coordinate_kernel(segments, *shape, k)
    out = np.empty((k, 3))
    for axis, coordinate in enumerate(np.indices(shape, sparse=True)):
        weights = np.broadcast_to(coordinate, shape).reshape(-1).astype(np.float64)
        out[:, axis] = np.bincount(segments, weights=weights, minlength=k)
    return out


@numba.njit(nogil=True, cache=True)
def _image_kernel(segments, values, counts, order, minimum, maximum):  # pragma: no cover (compiled)
    # values: (c, n). Returns mean, M2, M3, M4 of shape (c, k); moments above `order` stay zero.
    # `minimum` and `maximum` have shape (c, k) and are updated in place; pass shape (0, k) to skip.
    #
    # Neighbouring pixels mostly share a label, so sums, minima and maxima are accumulated in
    # registers over each run of equal labels and written once per run. Updating `mean[c, s]` for
    # every pixel would make each pixel wait for the store of the previous one to the same address.
    # `v != v` is true for NaN only: NaN propagates into minimum and maximum, as in np.minimum.
    n_channels, n = values.shape
    k = counts.size
    do_min = minimum.shape[0] > 0
    do_max = maximum.shape[0] > 0
    mean = np.zeros((n_channels, k))
    m2 = np.zeros((n_channels, k))
    m3 = np.zeros((n_channels, k))
    m4 = np.zeros((n_channels, k))
    if n == 0:
        return mean, m2, m3, m4
    for c in range(n_channels):
        current = segments[0]
        total = 0.0
        low = values[c, 0]
        high = values[c, 0]
        for i in range(n):
            s = segments[i]
            v = values[c, i]
            if s != current:
                mean[c, current] += total
                if do_min and (low < minimum[c, current] or low != low):
                    minimum[c, current] = low
                if do_max and (high > maximum[c, current] or high != high):
                    maximum[c, current] = high
                current = s
                total = 0.0
                low = v
                high = v
            total += v
            if do_min and (v < low or v != v):
                low = v
            if do_max and (v > high or v != v):
                high = v
        mean[c, current] += total
        if do_min and (low < minimum[c, current] or low != low):
            minimum[c, current] = low
        if do_max and (high > maximum[c, current] or high != high):
            maximum[c, current] = high
        for j in range(k):
            mean[c, j] /= counts[j]
        if order >= 2:
            current = segments[0]
            center = mean[c, current]
            s2 = 0.0
            s3 = 0.0
            s4 = 0.0
            for i in range(n):
                s = segments[i]
                if s != current:
                    m2[c, current] += s2
                    m3[c, current] += s3
                    m4[c, current] += s4
                    s2 = 0.0
                    s3 = 0.0
                    s4 = 0.0
                    current = s
                    center = mean[c, s]
                d = values[c, i] - center
                d2 = d * d
                s2 += d2
                if order >= 3:
                    s3 += d2 * d
                if order >= 4:
                    s4 += d2 * d2
            m2[c, current] += s2
            m3[c, current] += s3
            m4[c, current] += s4
    return mean, m2, m3, m4


def _chunk_image_stats(
    group: tuple[NDArray, NDArray, NDArray],
    image_block: NDArray,
    request: _Request,
    engine: str = "numba",
) -> _Partial:
    """Mean, centered sums ``M2..M{order}``, minimum and maximum per label present in the chunk, for every channel."""
    labels, counts, segments = group
    image = image_block.reshape(image_block.shape[0], -1)
    k, n_channels = labels.size, image.shape[0]
    low, high = _dtype_extremes(image.dtype)
    minimum = np.full((n_channels if request.minimum else 0, k), high, dtype=image.dtype)
    maximum = np.full((n_channels if request.maximum else 0, k), low, dtype=image.dtype)
    partial = _Partial(labels, counts)
    if engine == "numba":
        mean, *moments = _image_kernel(segments, image, counts.astype(np.float64), request.order, minimum, maximum)
        if request.order >= 1:
            partial.mean = mean.T
        partial.m2, partial.m3, partial.m4 = (
            m.T if p <= request.order else None for p, m in zip((2, 3, 4), moments, strict=True)
        )
    else:
        n = counts.astype(np.float64)
        moments = {name: np.empty((k, n_channels)) for name in ("mean", "m2", "m3", "m4")[: request.order]}
        for c in range(n_channels):
            if request.minimum:
                np.minimum.at(minimum[c], segments, image[c])
            if request.maximum:
                np.maximum.at(maximum[c], segments, image[c])
            if request.order == 0:
                continue
            values = image[c].astype(np.float64)
            moments["mean"][:, c] = np.bincount(segments, weights=values, minlength=k) / n
            if request.order >= 2:
                d = values - moments["mean"][segments, c]
                d2 = d * d
                moments["m2"][:, c] = np.bincount(segments, weights=d2, minlength=k)
                if request.order >= 3:
                    moments["m3"][:, c] = np.bincount(segments, weights=d2 * d, minlength=k)
                if request.order >= 4:
                    moments["m4"][:, c] = np.bincount(segments, weights=d2 * d2, minlength=k)
        for name, values in moments.items():
            setattr(partial, name, values)
    if request.minimum:
        partial.minimum = minimum.T
    if request.maximum:
        partial.maximum = maximum.T
    return partial


def _join_channels(*partials: _Partial) -> _Partial:
    """Join partial results of the same labels that cover consecutive channels, in channel order."""
    first = partials[0]
    if len(partials) == 1:
        return first
    channels = {
        name: np.concatenate([getattr(p, name) for p in partials], axis=1)
        for name in _CHANNEL_FIELDS
        if getattr(first, name) is not None
    }
    # labels, counts and coordinate sums do not depend on the channels: keep those of the first
    return replace(first, **channels)


def _merge(*partials: _Partial) -> _Partial:
    """Combine partial results of several chunks exactly (see module docstring), with numpy expressions."""
    if len(partials) == 1:
        return partials[0]
    labels = np.concatenate([p.labels for p in partials])
    sort = np.argsort(labels, kind="stable")
    labels = labels[sort]
    first = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
    sizes = np.diff(np.r_[first, labels.size])
    counts = np.concatenate([p.counts for p in partials])[sort]
    merged = _Partial(labels[first], np.add.reduceat(counts, first))

    def gather(name: str) -> NDArray | None:
        if getattr(partials[0], name) is None:
            return None
        return np.concatenate([getattr(p, name) for p in partials], axis=0)[sort]

    if (minimum := gather("minimum")) is not None:
        merged.minimum = np.minimum.reduceat(minimum, first, axis=0)
    if (maximum := gather("maximum")) is not None:
        merged.maximum = np.maximum.reduceat(maximum, first, axis=0)
    if (coordinate_sum := gather("coordinate_sum")) is not None:
        merged.coordinate_sum = np.add.reduceat(coordinate_sum, first, axis=0)
    if (mean := gather("mean")) is None:
        return merged

    n = counts.astype(np.float64)[:, None]
    merged.mean = np.add.reduceat(n * mean, first, axis=0) / merged.counts[:, None]
    if (m2 := gather("m2")) is None:
        return merged
    delta = mean - np.repeat(merged.mean, sizes, axis=0)
    delta2 = delta * delta
    merged.m2 = np.add.reduceat(m2 + n * delta2, first, axis=0)
    if (m3 := gather("m3")) is None:
        return merged
    merged.m3 = np.add.reduceat(m3 + 3 * delta * m2 + n * delta2 * delta, first, axis=0)
    if (m4 := gather("m4")) is None:
        return merged
    merged.m4 = np.add.reduceat(m4 + 4 * delta * m3 + 6 * delta2 * m2 + n * delta2 * delta2, first, axis=0)
    return merged


@numba.njit(nogil=True, cache=True)
def _merge_sums_kernel(
    rows, counts, mean, minimum, maximum, coordinate_sum, out_counts, out_total, out_min, out_max, out_coordinates
):  # pragma: no cover (compiled)
    # pass 1 of _merge_numba for one partial: add its counts, count-weighted means and coordinate
    # sums at its output rows `rows`, and update the minimum and maximum. Fields that were not
    # requested have zero columns. `v != v` is true for NaN only: NaN propagates, as in np.minimum.
    for r in range(rows.size):
        t = rows[r]
        n = counts[r]
        out_counts[t] += n
        for j in range(mean.shape[1]):
            out_total[t, j] += n * mean[r, j]
        for j in range(minimum.shape[1]):
            v = minimum[r, j]
            if v < out_min[t, j] or v != v:
                out_min[t, j] = v
        for j in range(maximum.shape[1]):
            v = maximum[r, j]
            if v > out_max[t, j] or v != v:
                out_max[t, j] = v
        for j in range(coordinate_sum.shape[1]):
            out_coordinates[t, j] += coordinate_sum[r, j]


@numba.njit(nogil=True, cache=True)
def _merge_centered_kernel(rows, counts, mean, m2, m3, m4, merged_mean, out_m2, out_m3, out_m4):  # pragma: no cover
    # pass 2 of _merge_numba for one partial: the binomial combination of the module docstring,
    # around the merged mean. Moments above the requested order have zero columns.
    do_m3 = m3.shape[1] > 0
    do_m4 = m4.shape[1] > 0
    for r in range(rows.size):
        t = rows[r]
        n = counts[r]
        for j in range(m2.shape[1]):
            d = mean[r, j] - merged_mean[t, j]
            d2 = d * d
            out_m2[t, j] += m2[r, j] + n * d2
            if do_m3:
                out_m3[t, j] += m3[r, j] + 3 * d * m2[r, j] + n * d2 * d
            if do_m4:
                out_m4[t, j] += m4[r, j] + 4 * d * m3[r, j] + 6 * d2 * m2[r, j] + n * d2 * d2


def _merge_numba(*partials: _Partial) -> _Partial:
    """Same result as :func:`_merge` up to rounding, accumulated in place by compiled loops.

    Only the labels are sorted. The values of every partial are added straight into the output
    arrays, so no sorted copies of the inputs and no temporaries of the output size are made.
    """
    first = partials[0]
    if len(partials) == 1:
        return first
    labels = np.unique(np.concatenate([p.labels for p in partials]))
    rows = [np.searchsorted(labels, p.labels) for p in partials]  # output row of every input row

    def columns(p: _Partial, name: str) -> NDArray:
        value = getattr(p, name)
        return np.empty((p.labels.size, 0)) if value is None else value

    def output(name: str, fill: Callable[[np.dtype], float | int] = lambda dtype: 0) -> NDArray:
        value = getattr(first, name)
        if value is None:
            return np.empty((labels.size, 0))
        return np.full((labels.size, value.shape[1]), fill(value.dtype), dtype=value.dtype)

    counts = np.zeros(labels.size, dtype=first.counts.dtype)
    out = {
        "mean": output("mean"),  # count-weighted sums until divided by the counts below
        "minimum": output("minimum", lambda dtype: _dtype_extremes(dtype)[1]),
        "maximum": output("maximum", lambda dtype: _dtype_extremes(dtype)[0]),
        "coordinate_sum": output("coordinate_sum"),
    }
    for p, r in zip(partials, rows, strict=True):
        _merge_sums_kernel(
            r,
            p.counts,
            *(columns(p, name) for name in ("mean", "minimum", "maximum", "coordinate_sum")),
            counts,
            *out.values(),
        )
    if first.mean is not None:
        out["mean"] /= counts[:, None]
    if first.m2 is not None:
        out.update({name: output(name) for name in ("m2", "m3", "m4")})
        for p, r in zip(partials, rows, strict=True):
            _merge_centered_kernel(
                r,
                p.counts,
                p.mean,
                p.m2,
                columns(p, "m3"),
                columns(p, "m4"),
                out["mean"],
                out["m2"],
                out["m3"],
                out["m4"],
            )
    return _Partial(labels, counts, **{name: value for name, value in out.items() if getattr(first, name) is not None})


def _tree_reduce(partials: list, split_every: int, merge: Callable[..., _Partial]):
    while len(partials) > 1:
        partials = [dask.delayed(merge)(*partials[i : i + split_every]) for i in range(0, len(partials), split_every)]
    return partials[0]
