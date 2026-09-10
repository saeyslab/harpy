from __future__ import annotations

from dataclasses import dataclass

import dask.array as da
import numpy as np

from harpy.io._cosmx._models import _CosmxMosaicGeometry, _CosmxPreview


@dataclass(frozen=True)
class _CosmxMosaicBlock:
    shape: tuple[int, int]
    fov: int | None = None
    source_y: slice | None = None
    source_x: slice | None = None


def _mosaic_placements(
    preview: _CosmxPreview,
    mosaic: _CosmxMosaicGeometry,
) -> dict[int, tuple[int, int]]:
    """Derive mosaic-local ``(y, x)`` tile offsets from a valid preview."""
    positions = preview.manifest.positions_by_fov
    return {
        fov: (
            positions[fov].y_px - mosaic.origin_y_px,
            positions[fov].x_px - mosaic.origin_x_px,
        )
        for fov in mosaic.fovs
    }


def _mosaic_block_grid(
    mosaic: _CosmxMosaicGeometry,
    *,
    placements: dict[int, tuple[int, int]],
    tile_shape: tuple[int, int],
) -> tuple[tuple[_CosmxMosaicBlock, ...], ...]:
    """Partition a mosaic canvas into disjoint rectangular assembly blocks.

    The function collects the start and end coordinates of every placed FOV,
    together with the mosaic bounds, along both axes. Consecutive coordinates
    define a rectangular grid. Because every FOV edge is a grid boundary, each
    resulting block is either wholly covered by one FOV or wholly uncovered.

    For example, two four-pixel-wide FOVs separated by one pixel produce::

        x: 0        4 5        9
           +--------+ +--------+
           | FOV 1  | | FOV 2  |
           +--------+ +--------+
           | covered|g| covered|
                     ^
                    gap

    Covered blocks store the owning FOV and the corresponding FOV-local source
    slices. Uncovered blocks store only their shape and are later represented
    by lazy zero arrays. The returned blocks are assembly metadata, not Dask or
    Zarr chunks; :func:`_assemble_raster` concatenates them before the completed
    mosaic is rechunked.

    Parameters
    ----------
    mosaic
        Validated mosaic geometry defining the output canvas.
    placements
        Mapping from FOV number to its mosaic-local ``(y, x)`` tile origin.
    tile_shape
        Common source-FOV shape as ``(height, width)``.

    Returns
    -------
    tuple of tuple of _CosmxMosaicBlock
        Rows of blocks ordered from top to bottom and left to right.

    Raises
    ------
    RuntimeError
        If a block has multiple FOV owners, violating the validated no-overlap
        invariant.
    """
    tile_height, tile_width = tile_shape
    y_boundaries = sorted(
        {0, mosaic.shape[0]} | {y for y, _ in placements.values()} | {y + tile_height for y, _ in placements.values()}
    )
    x_boundaries = sorted(
        {0, mosaic.shape[1]} | {x for _, x in placements.values()} | {x + tile_width for _, x in placements.values()}
    )

    rows = []
    for y0, y1 in zip(y_boundaries[:-1], y_boundaries[1:], strict=True):
        row = []
        for x0, x1 in zip(x_boundaries[:-1], x_boundaries[1:], strict=True):
            owners = [
                fov
                for fov, (tile_y, tile_x) in placements.items()
                if tile_y <= y0 and y1 <= tile_y + tile_height and tile_x <= x0 and x1 <= tile_x + tile_width
            ]
            if len(owners) > 1:
                raise RuntimeError(f"CosMx mosaic block {(y0, y1, x0, x1)} has multiple FOV owners {owners}.")
            if not owners:
                row.append(_CosmxMosaicBlock(shape=(y1 - y0, x1 - x0)))
                continue
            fov = owners[0]
            tile_y, tile_x = placements[fov]
            row.append(
                _CosmxMosaicBlock(
                    shape=(y1 - y0, x1 - x0),
                    fov=fov,
                    source_y=slice(y0 - tile_y, y1 - tile_y),
                    source_x=slice(x0 - tile_x, x1 - tile_x),
                )
            )
        rows.append(tuple(row))
    return tuple(rows)


def _assemble_raster(
    block_grid: tuple[tuple[_CosmxMosaicBlock, ...], ...],
    *,
    planes: dict[int, da.Array],
    dtype: np.dtype,
) -> da.Array:
    """Assemble FOV planes and zero-filled gaps into one lazy 2D mosaic."""
    rows = []
    for row in block_grid:
        blocks = []
        for block in row:
            if block.fov is None:
                blocks.append(da.zeros(block.shape, chunks=block.shape, dtype=dtype))
            else:
                assert block.source_y is not None
                assert block.source_x is not None
                blocks.append(planes[block.fov][block.source_y, block.source_x])
        rows.append(blocks[0] if len(blocks) == 1 else da.concatenate(blocks, axis=1))
    return rows[0] if len(rows) == 1 else da.concatenate(rows, axis=0)


def _pixel_coordinate_system(base: str, mosaic: int) -> str:
    """Return the independent pixel coordinate system for one mosaic."""
    return f"{base}_{mosaic}"
