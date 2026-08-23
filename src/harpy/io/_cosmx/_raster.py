from __future__ import annotations

from harpy.io._cosmx._models import _CosmxMosaicGeometry, _CosmxPreview


def _validate_orientation(*, flip_x: bool, flip_y: bool) -> None:
    """Require explicit booleans for the two dataset-wide axis flips."""
    for name, value in (("flip_x", flip_x), ("flip_y", flip_y)):
        if not isinstance(value, bool):
            raise ValueError(f"CosMx {name} must be a bool, found {value!r}.")


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


def _pixel_coordinate_system(base: str, mosaic: int) -> str:
    """Return the independent pixel coordinate system for one mosaic."""
    return f"{base}_{mosaic}"
