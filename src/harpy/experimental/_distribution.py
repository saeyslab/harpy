from __future__ import annotations

from collections.abc import Sequence

import dask
import dask.array as da
import numpy as np
import pandas as pd
from loguru import logger as log
from numpy.typing import NDArray

from harpy.experimental._layout import (
    InstanceBoxes,
    TilePlan,
    assemble_region,
    assign_owner_tiles,
    compute_instance_boxes,
    plan_tiles,
    region_pieces,
    required_depth,
)
from harpy.experimental._order_statistics import StatSpec, build_stat_spec, compute_segment_stats, sort_within_segments
from harpy.utils._keys import _INSTANCE_KEY

_AXIS_NAMES = ("z", "y", "x")


class DistributionAggregator:
    """
    Per-instance distribution statistics of image intensities (median, quantiles, IQR, MAD), computed out of core.

    Unlike the statistics of :class:`harpy.utils.MomentAggregator`, quantiles cannot be
    combined from per-chunk partial results: they need all pixel values of an instance at
    once. This class computes them tile by tile, without extracting instance windows first.

    Parameters
    ----------
    mask_dask_array
        Integer labels of shape ``('z', 'y', 'x')`` or ``('y', 'x')``. ``0`` is background.
        Its chunks define the tiles.
    image_dask_array
        Image of shape ``('c', *mask_dask_array.shape)``. Its chunks may differ from the
        chunks of the mask. Chunking along ``c`` bounds the number of channels loaded per task.
    depth
        Largest halo, in pixels, that a tile may read around itself, per spatial axis (an
        ``int`` applies to every axis). If `None` (default), each tile reads exactly the
        region its instances need, and all results are exact. Set `depth` to bound memory
        for labels that contain a few huge instances; instances that do not fit in their
        tile plus `depth` use only the pixels inside it (see :attr:`instance_table`).
    instance_key
        Name of the instance id column in the returned DataFrames.

    Notes
    -----
    The computation has three steps:

    1. One pass over the mask computes each instance's area and bounding box.
    2. Each instance is owned by one tile: the chunk of the mask containing the midpoint
       of its bounding box. A tile reads the union of the bounding boxes of the instances
       it owns. This is a halo of exactly the size those instances need, so instances that
       cross chunk borders are complete without choosing a `depth`.
    3. Per tile and per chunk of channels, one task gathers the pixel values of the owned
       instances, sorts them per instance in one vectorized sort, and reads the quantiles at
       their positions. The mask region is grouped once per tile and shared by the channel
       chunks.

    Peak memory per task is about one read region of the mask plus one read region of a
    channel chunk, plus about 24 bytes per owned pixel. Results match
    :func:`numpy.nanquantile` (``method="linear"``): NaN pixel values are ignored.

    Runs on CPU (NumPy).

    Examples
    --------
    .. code-block:: python

        import harpy as hp
        from harpy.experimental import DistributionAggregator

        sdata = hp.datasets.pixie_example()
        image = sdata["raw_image_fov0"].data[:, None, ...]  # (c, z, y, x)
        mask = sdata["label_whole_fov0"].data[None, ...]  # (z, y, x)

        aggregator = DistributionAggregator(mask_dask_array=mask, image_dask_array=image)
        dfs = aggregator.aggregate_stats(stats_funcs=("median", "quantiles", "iqr", "mad"), q=(0.1, 0.9))
        dfs["median"]  # one row per instance, one column per channel
    """

    def __init__(
        self,
        mask_dask_array: da.Array,
        image_dask_array: da.Array,
        depth: int | Sequence[int] | None = None,
        instance_key: str = _INSTANCE_KEY,
    ):
        if not np.issubdtype(mask_dask_array.dtype, np.integer):
            raise ValueError(
                f"'mask_dask_array' has dtype '{mask_dask_array.dtype}', but labels need an integer dtype. "
                "Cast it, e.g. 'mask_dask_array.astype(np.int32)'."
            )
        if mask_dask_array.ndim not in (2, 3):
            raise ValueError(
                f"'mask_dask_array' has {mask_dask_array.ndim} dimensions {mask_dask_array.shape}; "
                "expected ('z', 'y', 'x') or ('y', 'x')."
            )
        if image_dask_array.ndim != mask_dask_array.ndim + 1 or image_dask_array.shape[1:] != mask_dask_array.shape:
            raise ValueError(
                f"'image_dask_array' has shape {image_dask_array.shape}, but the mask has shape "
                f"{mask_dask_array.shape}; expected ('c', *{mask_dask_array.shape}). "
                "Add a channel axis, or a 'z' axis to both, so the spatial shapes match."
            )
        self._mask = mask_dask_array
        self._image = image_dask_array
        self._depth = _normalize_depth(depth, mask_dask_array.ndim)
        self._instance_key = instance_key
        self._boxes: InstanceBoxes | None = None

    @property
    def instance_table(self) -> pd.DataFrame:
        """
        One row per instance: area, bounding box, owner tile, and whether `depth` truncates it.

        Instances with ``truncated == True`` extend beyond their owner tile plus `depth`; their
        statistics use only the pixels inside that region.
        """
        boxes = self._instance_boxes()
        owner = assign_owner_tiles(boxes, self._mask.chunks)
        rows = np.arange(boxes.ids.size)
        truncated = np.zeros(boxes.ids.size, dtype=bool)
        for plan in plan_tiles(boxes, owner, rows, self._mask.chunks, self._depth):
            truncated[plan.rows] = plan.truncated
        axes = _AXIS_NAMES[-self._mask.ndim :]
        table = {self._instance_key: boxes.ids, "area": boxes.area}
        for a, name in enumerate(axes):
            table[f"{name}_min"] = boxes.bbox_min[:, a]
            table[f"{name}_max"] = boxes.bbox_max[:, a]
        for a, name in enumerate(axes):
            table[f"tile_{name}"] = owner[:, a]
        table["truncated"] = truncated
        return pd.DataFrame(table)

    def aggregate_stats(
        self,
        stats_funcs: str | Sequence[str] = ("median",),
        q: float | Sequence[float] | None = None,
        index: NDArray | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Compute distribution statistics for every instance and channel.

        Parameters
        ----------
        stats_funcs
            Statistics to compute: ``"median"``, ``"quantiles"`` (the quantiles in `q`),
            ``"iqr"`` (interquartile range, ``q0.75 - q0.25``) and ``"mad"`` (median
            absolute deviation from the median, unscaled). All are computed from one
            sort per channel.
        q
            Quantiles in ``[0, 1]``, required if `stats_funcs` contains ``"quantiles"``.
        index
            Instance ids to compute. If `None`, all non-zero instances in the mask.

        Returns
        -------
        A dict mapping each statistic name (``"median"``, ``"q0.1"``, ``"iqr"``, ``"mad"``, ...)
        to a DataFrame with one row per instance, sorted by instance id, one column per
        channel, and a column `instance_key`.
        """
        instance_ids, stats, names = self.aggregate_stats_lazy(stats_funcs=stats_funcs, q=q, index=index)
        stats = stats.compute()
        order = np.argsort(instance_ids, kind="stable")
        instance_ids = instance_ids[order]
        stats = stats[order]
        dfs = {}
        for s, name in enumerate(names):
            df = pd.DataFrame(stats[:, :, s])
            df[self._instance_key] = instance_ids
            dfs[name] = df
        return dfs

    def aggregate_stats_lazy(
        self,
        stats_funcs: str | Sequence[str] = ("median",),
        q: float | Sequence[float] | None = None,
        index: NDArray | None = None,
    ) -> tuple[NDArray, da.Array, tuple[str, ...]]:
        """
        Build the Dask graph for :meth:`aggregate_stats` without computing it.

        Use this to write large results to disk, e.g. with :func:`dask.array.to_zarr` after a
        rechunk to regular chunks. The bounding-box pass (step 1) is computed eagerly.

        Returns
        -------
        A 3-tuple ``(instance_ids, stats, names)``: instance ids of shape ``(i,)`` in tile order
        (not sorted), a lazy float32 array of shape ``(i, c, len(names))``, and the statistic
        name of each entry along the last axis.
        """
        spec = build_stat_spec(stats_funcs, q)
        boxes = self._instance_boxes()
        owner = assign_owner_tiles(boxes, self._mask.chunks)
        rows = self._rows_for_index(boxes, index)
        plans = plan_tiles(boxes, owner, rows, self._mask.chunks, self._depth)
        self._log_plans(plans, boxes, owner, rows)

        n_channels = self._image.shape[0]
        if not plans:
            return (
                np.empty(0, dtype=self._mask.dtype),
                da.zeros((0, n_channels, len(spec.names)), dtype=np.float32),
                spec.names,
            )

        mask_blocks = self._mask.to_delayed()
        image_blocks = self._image.to_delayed()
        channel_chunks = self._image.chunks[0]
        parts: list[list[da.Array]] = [[] for _ in channel_chunks]
        for plan in plans:
            owned_ids = boxes.ids[plan.rows]
            shape = tuple(window.stop - window.start for window in plan.region)
            mask_pieces = region_pieces(self._mask.chunks, plan.region)
            # blocks go in as separate arguments: for a list argument holding Delayed objects, dask
            # culls the whole graph on every call, which dominates graph construction for many tiles
            group = dask.delayed(_group_tile)(
                [(source, target) for _, source, target in mask_pieces],
                shape,
                self._mask.dtype,
                owned_ids,
                *[mask_blocks[block] for block, _, _ in mask_pieces],
            )
            image_pieces = region_pieces(self._image.chunks[1:], plan.region)
            for c, channel_chunk in enumerate(channel_chunks):
                stats = dask.delayed(_tile_statistics)(
                    group,
                    [(source, target) for _, source, target in image_pieces],
                    (channel_chunk, *shape),
                    self._image.dtype,
                    spec,
                    *[image_blocks[(c, *block)] for block, _, _ in image_pieces],
                )
                parts[c].append(
                    da.from_delayed(stats, shape=(owned_ids.size, channel_chunk, len(spec.names)), dtype=np.float32)
                )

        instance_ids = np.concatenate([boxes.ids[plan.rows] for plan in plans])
        stats = da.concatenate([da.concatenate(p, axis=0) for p in parts], axis=1)
        return instance_ids, stats, spec.names

    def _instance_boxes(self) -> InstanceBoxes:
        if self._boxes is None:
            log.info("Computing area and bounding box of every instance (one pass over the mask).")
            self._boxes = compute_instance_boxes(self._mask)
            log.info(f"Found {self._boxes.ids.size} instances.")
        return self._boxes

    def _rows_for_index(self, boxes: InstanceBoxes, index: NDArray | None) -> NDArray[np.intp]:
        if index is None:
            return np.arange(boxes.ids.size)
        index = np.unique(np.asarray(index))
        if np.any(index == 0):
            raise ValueError("'index' contains 0, which is background, not an instance. Remove 0 from 'index'.")
        rows = np.searchsorted(boxes.ids, index)
        found = (rows < boxes.ids.size) & (boxes.ids[np.minimum(rows, boxes.ids.size - 1)] == index)
        if not found.all():
            missing = index[~found]
            raise ValueError(
                f"{missing.size} id(s) in 'index' are not in the mask, e.g. {missing[:10].tolist()}. "
                "Remove them from 'index', or pass 'index=None' to use every instance in the mask."
            )
        return rows

    def _log_plans(
        self, plans: list[TilePlan], boxes: InstanceBoxes, owner: NDArray[np.intp], rows: NDArray[np.intp]
    ) -> None:
        if not plans:
            return
        tile_pixels = np.prod([max(c) for c in self._mask.chunks])
        largest = max(plans, key=lambda p: np.prod([w.stop - w.start for w in p.region]))
        largest_shape = tuple(w.stop - w.start for w in largest.region)
        log.info(
            f"{len(plans)} tiles own {rows.size} instances. Largest read region: {largest_shape} "
            f"({np.prod(largest_shape) / tile_pixels:.2f}x a full tile), owned by tile {largest.tile}."
        )
        n_truncated = int(sum(p.truncated.sum() for p in plans))
        if n_truncated:
            truncated_ids = np.concatenate([boxes.ids[p.rows[p.truncated]] for p in plans])
            log.warning(
                f"{n_truncated} of {rows.size} instances extend beyond their owner tile plus depth={self._depth}, "
                f"e.g. ids {truncated_ids[:10].tolist()}. Their statistics use only the pixels inside that region. "
                f"For exact results, set 'depth' to at least "
                f"{required_depth(boxes, owner, self._mask.chunks, rows)} or to None; "
                "see 'instance_table' for the affected instances."
            )


def _normalize_depth(depth: int | Sequence[int] | None, ndim: int) -> tuple[int, ...] | None:
    if depth is None:
        return None
    depth = (int(depth),) * ndim if np.isscalar(depth) else tuple(int(d) for d in depth)
    if len(depth) != ndim or any(d < 0 for d in depth):
        raise ValueError(
            f"'depth' is {depth}, but it needs one non-negative value per spatial axis of the mask ({ndim}), "
            "or a single non-negative int."
        )
    return depth


def _group_tile(
    placements: list[tuple[tuple[slice, ...], tuple[slice, ...]]],
    shape: tuple[int, ...],
    dtype: np.dtype,
    owned_ids: NDArray,
    *mask_blocks: NDArray,
) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.intp]]:
    """Pixels (flat index in the region) of the owned instances, their dense instance index, and the count per instance."""
    flat = assemble_region(shape, dtype, mask_blocks, placements).reshape(-1)
    n_owned = owned_ids.size
    low = int(owned_ids[0])
    span = int(owned_ids[-1]) - low + 1
    if (dtype.kind == "u" and dtype.itemsize == 8) or span > max(4 * flat.size, 1 << 20):
        position = np.minimum(np.searchsorted(owned_ids, flat), n_owned - 1)
        pixels = np.flatnonzero(owned_ids[position] == flat)
        segments = position[pixels]
    else:
        # dense lookup table over the owned id range: ~3x faster than searchsorted
        lookup = np.full(span, -1, dtype=np.intp)
        lookup[owned_ids.astype(np.int64) - low] = np.arange(n_owned)
        relative = flat.astype(np.int64) - low
        candidates = np.flatnonzero((relative >= 0) & (relative < span))
        segments = lookup[relative[candidates]]
        keep = segments >= 0
        pixels = candidates[keep]
        segments = segments[keep]
    return pixels, segments, np.bincount(segments, minlength=n_owned)


def _tile_statistics(
    group: tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.intp]],
    placements: list[tuple[tuple[slice, ...], tuple[slice, ...]]],
    shape: tuple[int, ...],
    dtype: np.dtype,
    spec: StatSpec,
    *image_blocks: NDArray,
) -> NDArray[np.float32]:
    """Statistics of shape ``(owned instances, channels in this chunk, len(spec.names))``."""
    pixels, segments, counts = group
    n_channels = shape[0]
    out = np.full((counts.size, n_channels, len(spec.names)), np.nan, dtype=np.float32)
    if pixels.size == 0:
        return out
    image = assemble_region(shape, dtype, image_blocks, placements).reshape(n_channels, -1)
    offsets = np.zeros(counts.size, dtype=np.intp)
    np.cumsum(counts[:-1], out=offsets[1:])
    for c in range(n_channels):
        values = image[c][pixels]
        n_valid = counts
        if values.dtype.kind == "f":
            is_nan = np.isnan(values)
            if is_nan.any():
                n_valid = counts - np.bincount(segments[is_nan], minlength=counts.size)
        sorted_values = sort_within_segments(segments, values)
        out[:, c, :] = compute_segment_stats(sorted_values, offsets, counts, n_valid, spec)
    return out
