from __future__ import annotations

from dataclasses import dataclass

import dask.array as da
import numpy as np

from harpy.io._cosmx._models import _CosmxMosaicGeometry


@dataclass(frozen=True)
class _CosmxMosaicBlock:
    shape: tuple[int, int]
    fov: int | None = None
    source_y: slice | None = None
    source_x: slice | None = None


def _mosaic_block_grid(mosaic: _CosmxMosaicGeometry) -> tuple[tuple[_CosmxMosaicBlock, ...], ...]:
    """Partition a mosaic canvas into a grid of rectangular assembly blocks.

    The start and end of every FOV's tile, together with the canvas bounds,
    give the grid's boundaries along each axis. Because every tile edge is a
    boundary, each block lies wholly inside one FOV's tile or wholly in a gap.

    For example, take three FOVs with 4 x 4 pixel tiles at tile origins
    ``(0, 0)``, ``(2, 5)`` and ``(4, 1)``. Each character below is one pixel,
    marked with the number of the FOV whose tile covers it, or ``.`` for a
    gap::

        x   012345678
        y=0 1111.....
        y=1 1111.....
        y=2 1111.2222
        y=3 1111.2222
        y=4 .33332222
        y=5 .33332222
        y=6 .3333....
        y=7 .3333....

    Every start and end of a tile is a boundary: y = 0, 2, 4, 6, 8 and
    x = 0, 1, 4, 5, 9. The cuts run across the whole canvas and give a
    4 x 4 grid of 16 blocks, separated here by spaces and dashes::

        x   0 123 4 5678
        y=0 1 111 . ....
        y=1 1 111 . ....
            - --- - ----
        y=2 1 111 . 2222
        y=3 1 111 . 2222
            - --- - ----
        y=4 . 333 3 2222
        y=5 . 333 3 2222
            - --- - ----
        y=6 . 333 3 ....
        y=7 . 333 3 ....

    FOV 1's tile is cut into four blocks, also at x = 1, where FOV 3's tile
    starts although the two are not side by side; FOV 3's tile is cut into
    four and FOV 2's into two. The other six blocks are gaps.

    A covered block stores its FOV and the slices of that FOV's tile it
    covers. A gap block stores only its shape and becomes a lazy array of
    zeros. Blocks depend only on where the tiles are, not on the output
    chunks. :func:`_assemble_raster` turns each block into one Dask chunk and
    concatenates them; the writers then rechunk the mosaic to the requested
    chunks.

    Parameters
    ----------
    mosaic
        Validated mosaic geometry defining the output canvas, the
        mosaic-local ``(y, x)`` tile origin of every FOV and the common tile
        shape.

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
    tile_origins = mosaic.tile_origins
    tile_height, tile_width = mosaic.tile_shape
    y_boundaries = sorted(
        {0, mosaic.shape[0]}
        | {y for y, _ in tile_origins.values()}
        | {y + tile_height for y, _ in tile_origins.values()}
    )
    x_boundaries = sorted(
        {0, mosaic.shape[1]}
        | {x for _, x in tile_origins.values()}
        | {x + tile_width for _, x in tile_origins.values()}
    )

    rows = []
    for y0, y1 in zip(y_boundaries[:-1], y_boundaries[1:], strict=True):
        row = []
        for x0, x1 in zip(x_boundaries[:-1], x_boundaries[1:], strict=True):
            owners = [
                fov
                for fov, (tile_y, tile_x) in tile_origins.items()
                if tile_y <= y0 and y1 <= tile_y + tile_height and tile_x <= x0 and x1 <= tile_x + tile_width
            ]
            if len(owners) > 1:
                raise RuntimeError(f"CosMx mosaic block {(y0, y1, x0, x1)} has multiple FOV owners {owners}.")
            if not owners:
                row.append(_CosmxMosaicBlock(shape=(y1 - y0, x1 - x0)))
                continue
            fov = owners[0]
            tile_y, tile_x = tile_origins[fov]
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
    mosaic: _CosmxMosaicGeometry,
    *,
    planes: dict[int, da.Array],
    dtype: np.dtype,
) -> da.Array:
    """Assemble FOV planes and zero-filled gaps into one lazy 2D mosaic.

    ``planes`` maps each FOV of ``mosaic`` to its lazy, tile-shaped plane.
    The planes are cut along the blocks of :func:`_mosaic_block_grid` and
    concatenated, so no pixels are read.
    """
    rows = []
    for row in _mosaic_block_grid(mosaic):
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
