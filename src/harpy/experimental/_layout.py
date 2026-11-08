"""Owner-tile layout for tiled per-instance computations.

One pass over the labels finds the area and bounding box of every instance. Each instance
is then owned by exactly one tile (a chunk of the labels array): the tile that contains the
midpoint of its bounding box. A tile reads the union of the bounding boxes of the instances
it owns, so every owned instance is complete in the region the tile reads.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import dask
import dask.array as da
import numpy as np
from numpy.typing import NDArray

_LOW_32_BITS = np.uint64(0xFFFFFFFF)


@dataclass(frozen=True)
class InstanceBoxes:
    """Area and inclusive bounding box of every non-zero instance, sorted by instance id."""

    ids: NDArray
    area: NDArray[np.int64]
    bbox_min: NDArray[np.int64]  # (i, ndim)
    bbox_max: NDArray[np.int64]  # (i, ndim), inclusive


@dataclass(frozen=True)
class TilePlan:
    """The instances one tile owns and the region it reads."""

    tile: tuple[int, ...]
    rows: NDArray[np.intp]  # rows into InstanceBoxes, sorted by instance id
    region: tuple[slice, ...]  # global region read by this tile
    truncated: NDArray[np.bool_]  # owned instances not complete inside `region`


def _chunk_starts(chunks: tuple[tuple[int, ...], ...]) -> list[NDArray[np.int64]]:
    return [np.r_[0, np.cumsum(c)[:-1]].astype(np.int64) for c in chunks]


def _sort_labels_with_positions(labels: NDArray, positions: NDArray[np.intp]) -> tuple[NDArray, NDArray[np.intp]]:
    """Sort by label, keeping raster order within a label."""
    low = labels.min()
    span = int(labels.max()) - int(low)
    if span < 2**32 and positions.size < 2**32:
        keys = (labels - low).astype(np.uint64) << np.uint64(32)
        keys |= positions.astype(np.uint64)
        keys.sort()
        return (keys >> np.uint64(32)).astype(labels.dtype) + low, (keys & _LOW_32_BITS).astype(np.intp)
    order = np.argsort(labels, kind="stable")
    return labels[order], positions[order]


def _block_instance_boxes(block: NDArray, offset: tuple[int, ...]) -> NDArray[np.int64]:
    """Per instance in `block`: ``[id, area, min_0..min_n, max_0..max_n]`` in global coordinates."""
    ndim = block.ndim
    flat = block.reshape(-1)
    positions = np.flatnonzero(flat)
    if positions.size == 0:
        return np.empty((0, 2 + 2 * ndim), dtype=np.int64)
    labels, positions = _sort_labels_with_positions(flat[positions], positions)
    starts = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
    out = np.empty((starts.size, 2 + 2 * ndim), dtype=np.int64)
    out[:, 0] = labels[starts]
    out[:, 1] = np.diff(np.r_[starts, labels.size])
    for axis, coordinate in enumerate(np.unravel_index(positions, block.shape)):
        out[:, 2 + axis] = np.minimum.reduceat(coordinate, starts) + offset[axis]
        out[:, 2 + ndim + axis] = np.maximum.reduceat(coordinate, starts) + offset[axis]
    return out


def compute_instance_boxes(mask: da.Array) -> InstanceBoxes:
    """One pass over `mask`: the area and bounding box of every non-zero instance."""
    ndim = mask.ndim
    starts = _chunk_starts(mask.chunks)
    blocks = mask.to_delayed()
    tasks = [
        dask.delayed(_block_instance_boxes)(blocks[index], tuple(int(starts[a][i]) for a, i in enumerate(index)))
        for index in np.ndindex(mask.numblocks)
    ]
    table = np.concatenate(dask.compute(*tasks), axis=0)
    if table.shape[0] == 0:
        empty = np.empty((0, ndim), dtype=np.int64)
        return InstanceBoxes(np.empty(0, dtype=mask.dtype), np.empty(0, dtype=np.int64), empty, empty)
    table = table[np.argsort(table[:, 0], kind="stable")]
    first = np.flatnonzero(np.r_[True, table[1:, 0] != table[:-1, 0]])
    return InstanceBoxes(
        ids=table[first, 0].astype(mask.dtype),
        area=np.add.reduceat(table[:, 1], first),
        bbox_min=np.minimum.reduceat(table[:, 2 : 2 + ndim], first, axis=0),
        bbox_max=np.maximum.reduceat(table[:, 2 + ndim :], first, axis=0),
    )


def assign_owner_tiles(boxes: InstanceBoxes, chunks: tuple[tuple[int, ...], ...]) -> NDArray[np.intp]:
    """Owner tile per instance: the chunk that contains the midpoint of its bounding box. Shape ``(i, ndim)``."""
    midpoint = (boxes.bbox_min + boxes.bbox_max) / 2.0
    owner = np.empty(boxes.bbox_min.shape, dtype=np.intp)
    for axis, axis_starts in enumerate(_chunk_starts(chunks)):
        index = np.searchsorted(axis_starts, midpoint[:, axis], side="right") - 1
        owner[:, axis] = np.clip(index, 0, axis_starts.size - 1)
    return owner


def required_depth(
    boxes: InstanceBoxes, owner: NDArray[np.intp], chunks: tuple[tuple[int, ...], ...], rows: NDArray[np.intp]
) -> tuple[int, ...]:
    """Smallest halo per axis around the owner tile that contains every instance in `rows` completely."""
    depth = []
    for axis, axis_chunks in enumerate(chunks):
        starts = _chunk_starts(chunks)[axis]
        ends = starts + np.asarray(axis_chunks)
        tile = owner[rows, axis]
        below = starts[tile] - boxes.bbox_min[rows, axis]
        above = boxes.bbox_max[rows, axis] + 1 - ends[tile]
        depth.append(int(max(0, below.max(initial=0), above.max(initial=0))))
    return tuple(depth)


def plan_tiles(
    boxes: InstanceBoxes,
    owner: NDArray[np.intp],
    rows: NDArray[np.intp],
    chunks: tuple[tuple[int, ...], ...],
    depth: tuple[int, ...] | None,
) -> list[TilePlan]:
    """
    Group the instances in `rows` by owner tile, and compute the region each tile reads.

    The region is the union of the owned bounding boxes. If `depth` is given, the region is
    clipped to the owner tile grown by `depth` on every side; instances that do not fit are
    marked as truncated.
    """
    if rows.size == 0:
        return []
    numblocks = tuple(len(c) for c in chunks)
    starts = _chunk_starts(chunks)
    flat_tile = np.ravel_multi_index(tuple(owner[rows].T), numblocks)
    order = np.lexsort((boxes.ids[rows], flat_tile))
    rows = rows[order]
    flat_tile = flat_tile[order]
    first = np.flatnonzero(np.r_[True, flat_tile[1:] != flat_tile[:-1]])
    last = np.r_[first[1:], rows.size]

    plans = []
    for begin, end in zip(first, last, strict=True):
        owned = rows[begin:end]
        tile = tuple(int(i) for i in np.unravel_index(flat_tile[begin], numblocks))
        low = boxes.bbox_min[owned].min(axis=0)
        high = boxes.bbox_max[owned].max(axis=0) + 1
        if depth is not None:
            tile_low = np.array([starts[a][i] for a, i in enumerate(tile)]) - np.asarray(depth)
            tile_high = np.array([starts[a][i] + chunks[a][i] for a, i in enumerate(tile)]) + np.asarray(depth)
            low = np.maximum(low, tile_low)
            high = np.minimum(high, tile_high)
        truncated = np.any(boxes.bbox_min[owned] < low, axis=1) | np.any(boxes.bbox_max[owned] + 1 > high, axis=1)
        region = tuple(slice(int(lo), int(hi)) for lo, hi in zip(low, high, strict=True))
        plans.append(TilePlan(tile=tile, rows=owned, region=region, truncated=truncated))
    return plans


def region_pieces(
    chunks: tuple[tuple[int, ...], ...], region: tuple[slice, ...]
) -> list[tuple[tuple[int, ...], tuple[slice, ...], tuple[slice, ...]]]:
    """Chunks intersecting `region`, as ``(chunk index, slice within the chunk, slice within the region)``."""
    per_axis = []
    for axis_chunks, window in zip(chunks, region, strict=True):
        ends = np.cumsum(axis_chunks)
        starts = ends - np.asarray(axis_chunks)
        first = int(np.searchsorted(ends, window.start, side="right"))
        last = int(np.searchsorted(starts, window.stop, side="left"))
        entries = []
        for i in range(first, last):
            low = max(window.start, int(starts[i]))
            high = min(window.stop, int(ends[i]))
            entries.append(
                (i, slice(low - int(starts[i]), high - int(starts[i])), slice(low - window.start, high - window.start))
            )
        per_axis.append(entries)
    return [
        (
            tuple(e[0] for e in combination),
            tuple(e[1] for e in combination),
            tuple(e[2] for e in combination),
        )
        for combination in itertools.product(*per_axis)
    ]


def assemble_region(
    shape: tuple[int, ...],
    dtype: np.dtype,
    blocks: Sequence[NDArray],
    placements: list[tuple[tuple[slice, ...], tuple[slice, ...]]],
) -> NDArray:
    """Stitch a region from the chunks that intersect it. Leading (e.g. channel) axes are copied whole."""
    if len(blocks) == 1:
        source, _ = placements[0]
        return blocks[0][(Ellipsis, *source)]
    out = np.empty(shape, dtype=dtype)
    for block, (source, target) in zip(blocks, placements, strict=True):
        out[(Ellipsis, *target)] = block[(Ellipsis, *source)]
    return out
