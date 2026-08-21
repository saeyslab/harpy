from __future__ import annotations

from pathlib import Path

import numpy as np

from harpy.io._cosmx._discovery import _discover_cosmx
from harpy.io._cosmx._models import _CosmxFovPosition
from harpy.io._cosmx._preview import _default_adjacency_tolerance_px, _mosaic_geometries, _preview_cosmx


def test_mosaic_geometry_only_uses_positions_and_tile_shape() -> None:
    positions = {
        1: _CosmxFovPosition(fov=1, x_px=0, y_px=0, x_mm=0.0, y_mm=0.0),
        2: _CosmxFovPosition(fov=2, x_px=8, y_px=0, x_mm=0.008, y_mm=0.0),
        3: _CosmxFovPosition(fov=3, x_px=20, y_px=0, x_mm=0.020, y_mm=0.0),
    }

    mosaics = _mosaic_geometries(positions=positions, tile_shape=(8, 8))

    assert [mosaic.fovs for mosaic in mosaics] == [(1, 2), (3,)]
    assert [mosaic.shape for mosaic in mosaics] == [(8, 16), (8, 8)]


def test_mosaic_geometry_bridges_small_axis_aligned_gap() -> None:
    positions = {
        1: _CosmxFovPosition(fov=1, x_px=0, y_px=0, x_mm=0.0, y_mm=0.0),
        2: _CosmxFovPosition(fov=2, x_px=101, y_px=0, x_mm=0.101, y_mm=0.0),
        3: _CosmxFovPosition(fov=3, x_px=300, y_px=0, x_mm=0.300, y_mm=0.0),
    }

    mosaics = _mosaic_geometries(positions=positions, tile_shape=(100, 100), adjacency_tolerance_px=2)

    assert [mosaic.fovs for mosaic in mosaics] == [(1, 2), (3,)]


def test_mosaic_geometry_does_not_bridge_corner_only_gap() -> None:
    positions = {
        1: _CosmxFovPosition(fov=1, x_px=0, y_px=0, x_mm=0.0, y_mm=0.0),
        2: _CosmxFovPosition(fov=2, x_px=101, y_px=101, x_mm=0.101, y_mm=0.101),
    }

    mosaics = _mosaic_geometries(positions=positions, tile_shape=(100, 100), adjacency_tolerance_px=2)

    assert [mosaic.fovs for mosaic in mosaics] == [(1,), (2,)]


def test_default_adjacency_tolerance_is_two_percent_of_tile() -> None:
    assert _default_adjacency_tolerance_px((4256, 4256)) == 85


def test_preview_selects_common_positioned_fovs(decoded_cosmx_path: Path) -> None:
    preview = _preview_cosmx(_discover_cosmx(decoded_cosmx_path))

    assert preview.included_fovs == (1, 2, 3)
    assert preview.excluded_fovs == (4,)
    assert preview.unpositioned_fovs == (4,)
    assert [mosaic.fovs for mosaic in preview.mosaics] == [(1, 2), (3,)]
    assert [mosaic.shape for mosaic in preview.mosaics] == [(8, 16), (8, 8)]
    assert [estimate.mosaic for estimate in preview.estimates] == [1, 2]
    pixels = 8 * 16 + 8 * 8
    assert preview.estimated_image_nbytes == pixels * 5 * np.dtype(np.uint16).itemsize
    assert preview.estimated_cell_labels_nbytes == pixels * np.dtype(np.uint32).itemsize
    assert preview.estimated_compartment_labels_nbytes == pixels * np.dtype(np.uint8).itemsize
    assert any("adjacency tolerance of 1 pixel" in message for message in preview.diagnostics)
    assert any("without reading transcript contents" in message for message in preview.diagnostics)
