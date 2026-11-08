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
   present in it, instead of a row for every label in the mask.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import dask
import dask.array as da
import numba
import numpy as np
import pandas as pd
from loguru import logger as log
from numpy.typing import NDArray

from harpy.utils._keys import _INSTANCE_KEY

_MOMENT_ORDER = {"count": 0, "sum": 1, "mean": 1, "var": 2, "skew": 3, "kurtosis": 4}
SUPPORTED_STATS = (*_MOMENT_ORDER, "min", "max", "center_of_mass")


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
    CPU.

    Parameters
    ----------
    mask_dask_array
        Integer labels of shape ``('z', 'y', 'x')``.
    image_dask_array
        Image of shape ``('c', 'z', 'y', 'x')``, with the same spatial chunks as the mask.
    instance_key
        Name of the label id column in the returned DataFrames.
    split_every
        Number of partial results merged per task in the tree reduction.
    engine
        Per-chunk kernels. ``"numba"`` (default) runs compiled loops that release the GIL, so
        Dask threads run them in parallel. ``"numpy"`` uses :func:`numpy.bincount` and
        :func:`numpy.minimum.at`, which hold the GIL and allocate temporaries per channel.
    """

    def __init__(
        self,
        mask_dask_array: da.Array,
        image_dask_array: da.Array,
        instance_key: str = _INSTANCE_KEY,
        split_every: int = 8,
        engine: Literal["numba", "numpy"] = "numba",
    ):
        if engine not in ("numba", "numpy"):
            raise ValueError(f"'engine' is '{engine}'; choose 'numba' or 'numpy'.")
        if not np.issubdtype(mask_dask_array.dtype, np.integer):
            raise ValueError(
                f"'mask_dask_array' has dtype '{mask_dask_array.dtype}', but labels need an integer dtype. "
                "Cast it, e.g. 'mask_dask_array.astype(np.int32)'."
            )
        if mask_dask_array.ndim != 3 or image_dask_array.ndim != 4:
            raise ValueError(
                f"Expected a mask of shape ('z', 'y', 'x') and an image of shape ('c', 'z', 'y', 'x'), "
                f"got {mask_dask_array.shape} and {image_dask_array.shape}."
            )
        if image_dask_array.shape[1:] != mask_dask_array.shape:
            raise ValueError(
                f"The spatial shape of 'image_dask_array' {image_dask_array.shape[1:]} differs from the shape of "
                f"'mask_dask_array' {mask_dask_array.shape}. Pass arrays that cover the same pixels."
            )
        if image_dask_array.chunks[1:] != mask_dask_array.chunks:
            raise ValueError(
                f"The spatial chunks of 'image_dask_array' {image_dask_array.chunksize[1:]} differ from the chunks of "
                f"'mask_dask_array' {mask_dask_array.chunksize}. Rechunk one of them so the chunks match."
            )
        self._mask = mask_dask_array
        self._image = image_dask_array
        self._instance_key = instance_key
        self._split_every = split_every
        self._engine = engine

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

        mask_blocks = self._mask.to_delayed().ravel()
        # (channel chunks, spatial chunks), same spatial order as mask_blocks
        image_blocks = self._image.to_delayed().reshape(self._image.numblocks[0], -1)
        starts = [np.cumsum((0, *sizes[:-1])) for sizes in self._mask.chunks]
        partials = []
        for i, block_index in enumerate(np.ndindex(*self._mask.numblocks)):
            group = dask.delayed(_group_labels)(mask_blocks[i], self._engine)
            channel_partials = []
            if request.reads_image:
                channel_partials = [
                    dask.delayed(_chunk_image_stats)(group, block, request, self._engine)
                    for block in image_blocks[:, i]
                ]
            offset = None
            if request.center_of_mass:
                offset = tuple(int(starts[axis][b]) for axis, b in enumerate(block_index))
            # join the channel chunks of this spatial chunk right away, so its label grouping
            # is released once this chunk is done, instead of living until the whole reduction ends.
            # The channel partials go in as separate arguments: for a list argument holding Delayed
            # objects, dask culls the whole graph on every call.
            partials.append(dask.delayed(_chunk_partial)(group, offset, self._engine, *channel_partials))
        (result,) = dask.compute(_tree_reduce(partials, self._split_every))

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
            low, high = _dtype_extremes(self._image.dtype)
            if request.minimum:
                out["min"] = take(result.minimum, high)
            if request.maximum:
                out["max"] = take(result.maximum, low)
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


def _dtype_extremes(dtype: np.dtype) -> tuple[float | int, float | int]:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return info.min, info.max
    return -np.inf, np.inf


def _group_labels(
    block: NDArray, engine: str = "numba"
) -> tuple[NDArray, NDArray[np.intp], NDArray[np.intp], tuple[int, ...]]:
    """Labels present in `block`, their pixel counts, the dense label index of every pixel, and the block shape."""
    flat = block.reshape(-1)
    low, high = int(flat.min()), int(flat.max())
    span = high - low + 1
    if span <= max(2 * flat.size, 1 << 16) and not (flat.dtype.kind == "u" and flat.dtype.itemsize == 8):
        if engine == "numba":
            return (*_lookup_group_kernel(flat, low, span), block.shape)
        # np.bincount holds the GIL, so this path scales poorly over Dask threads
        relative = flat.astype(np.intp) - low
        counts = np.bincount(relative, minlength=span)
        present = np.flatnonzero(counts)
        lookup = np.empty(span, dtype=np.intp)
        lookup[present] = np.arange(present.size)
        return (present + low).astype(block.dtype), counts[present], lookup[relative], block.shape
    labels, segments, counts = np.unique(flat, return_inverse=True, return_counts=True)
    return labels, counts, segments.reshape(-1), block.shape


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
    group: tuple[NDArray, NDArray, NDArray, tuple[int, ...]],
    offset: tuple[int, ...] | None,
    engine: str,
    *channel_partials: _Partial,
) -> _Partial:
    """Join the channel chunks of one spatial chunk (same labels, same order) and add the coordinate sums."""
    labels, counts, segments, shape = group
    partial = _Partial(labels, counts)
    for name in _CHANNEL_FIELDS:
        parts = [getattr(p, name) for p in channel_partials]
        if parts and parts[0] is not None:
            setattr(partial, name, parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1))
    if offset is not None:
        partial.coordinate_sum = _coordinate_sums(segments, shape, labels.size, engine)
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
    n_channels, n = values.shape
    k = counts.size
    do_min = minimum.shape[0] > 0
    do_max = maximum.shape[0] > 0
    mean = np.zeros((n_channels, k))
    m2 = np.zeros((n_channels, k))
    m3 = np.zeros((n_channels, k))
    m4 = np.zeros((n_channels, k))
    for c in range(n_channels):
        for i in range(n):
            s = segments[i]
            v = values[c, i]
            mean[c, s] += v
            # `v != v` is true for NaN only: NaN propagates, as in np.minimum
            if do_min and (v < minimum[c, s] or v != v):
                minimum[c, s] = v
            if do_max and (v > maximum[c, s] or v != v):
                maximum[c, s] = v
        for j in range(k):
            mean[c, j] /= counts[j]
        if order >= 2:
            for i in range(n):
                s = segments[i]
                d = values[c, i] - mean[c, s]
                d2 = d * d
                m2[c, s] += d2
                if order >= 3:
                    m3[c, s] += d2 * d
                if order >= 4:
                    m4[c, s] += d2 * d2
    return mean, m2, m3, m4


def _chunk_image_stats(
    group: tuple[NDArray, NDArray, NDArray, tuple[int, ...]],
    image_block: NDArray,
    request: _Request,
    engine: str = "numba",
) -> _Partial:
    """Mean, centered sums ``M2..M{order}``, minimum and maximum per label present in the chunk, for every channel."""
    labels, counts, segments, _ = group
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


def _merge(*partials: _Partial) -> _Partial:
    """Combine partial results of several chunks exactly (see module docstring)."""
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


def _tree_reduce(partials: list, split_every: int):
    while len(partials) > 1:
        partials = [dask.delayed(_merge)(*partials[i : i + split_every]) for i in range(0, len(partials), split_every)]
    return partials[0]
