from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from harpy.io._cosmx._models import (
    _CosmxChannel,
    _CosmxFovFiles,
    _CosmxFovPosition,
    _CosmxManifest,
    _CosmxMosaicGeometry,
    _CosmxMosaicSizeEstimate,
    _CosmxPreview,
    _CosmxRunMetadata,
    _MorphologyPosition,
)


def _run_metadata(**updates: object) -> _CosmxRunMetadata:
    values = {
        "declared_fov_count": 2,
        "channels": (_CosmxChannel(channel_id="U", name="DNA"),),
        "pixel_size_um": 1.0,
        "tile_shape": (8, 8),
        "morphology_dtype": "uint16",
        "instance_labels_dtype": "uint16",
        "compartment_labels_dtype": "uint8",
    }
    values.update(updates)
    return _CosmxRunMetadata(**values)


def _manifest() -> _CosmxManifest:
    return _CosmxManifest(
        root=Path("/dataset"),
        fovs=(_CosmxFovFiles(fov=1), _CosmxFovFiles(fov=2)),
        positions=(_CosmxFovPosition(fov=1, x_px=0, y_px=0, x_mm=0.0, y_mm=0.0),),
        run=_run_metadata(),
        diagnostics=(),
    )


def _preview() -> _CosmxPreview:
    return _CosmxPreview(
        manifest=_manifest(),
        included_fovs=(1,),
        excluded_fovs=(2,),
        unpositioned_fovs=(2,),
        mosaics=(_CosmxMosaicGeometry(mosaic=1, fovs=(1,), origin_x_px=0, origin_y_px=0, shape=(8, 8)),),
        estimates=(
            _CosmxMosaicSizeEstimate(
                mosaic=1,
                image_nbytes=640,
                instance_labels_nbytes=256,
                compartment_labels_nbytes=64,
            ),
        ),
        diagnostics=(),
    )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"declared_fov_count": 0}, "Declared CosMx FOV count must be positive"),
        ({"channels": ()}, "at least one channel"),
        (
            {
                "channels": (
                    _CosmxChannel(channel_id="U", name="DNA"),
                    _CosmxChannel(channel_id="U", name="DNA duplicate"),
                )
            },
            "channel IDs must be unique",
        ),
        ({"pixel_size_um": float("nan")}, "pixel size must be finite and positive"),
        ({"tile_shape": (0, 8)}, "tile shape must contain two positive dimensions"),
        ({"morphology_dtype": "not-a-dtype"}, "Invalid CosMx morphology dtype"),
        ({"instance_labels_dtype": "int16"}, "instance labels dtype must be unsigned integer"),
        ({"compartment_labels_dtype": "float32"}, "compartment labels dtype must be unsigned integer"),
    ],
)
def test_run_metadata_rejects_invalid_structure(updates: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _run_metadata(**updates)


def test_leaf_models_reject_invalid_identifiers_and_coordinates() -> None:
    with pytest.raises(ValueError, match="FOV number must be positive"):
        _CosmxFovFiles(fov=0)
    with pytest.raises(ValueError, match="stage position.*must be finite"):
        _CosmxFovPosition(fov=1, x_px=0, y_px=0, x_mm=float("nan"), y_mm=0.0)
    with pytest.raises(ValueError, match="stage position.*must be finite"):
        _MorphologyPosition(fov=1, x_mm=0.0, y_mm=float("inf"))
    with pytest.raises(ValueError, match="channel ID must not be empty"):
        _CosmxChannel(channel_id="", name="DNA")


def test_manifest_rejects_inconsistent_fov_records() -> None:
    run_without_declared_count = _run_metadata(declared_fov_count=None)
    with pytest.raises(ValueError, match="manifest FOVs must be sorted and unique"):
        _CosmxManifest(
            root=Path("/dataset"),
            fovs=(_CosmxFovFiles(fov=1), _CosmxFovFiles(fov=1)),
            positions=(),
            run=run_without_declared_count,
            diagnostics=(),
        )
    with pytest.raises(ValueError, match="positions reference unknown FOVs"):
        _CosmxManifest(
            root=Path("/dataset"),
            fovs=(_CosmxFovFiles(fov=1),),
            positions=(_CosmxFovPosition(fov=2, x_px=0, y_px=0, x_mm=0.0, y_mm=0.0),),
            run=run_without_declared_count,
            diagnostics=(),
        )
    with pytest.raises(ValueError, match="must match the declared range"):
        _CosmxManifest(
            root=Path("/dataset"),
            fovs=(_CosmxFovFiles(fov=1),),
            positions=(),
            run=_run_metadata(),
            diagnostics=(),
        )


def test_mosaic_models_reject_invalid_geometry_and_estimates() -> None:
    with pytest.raises(ValueError, match="at least one FOV"):
        _CosmxMosaicGeometry(mosaic=1, fovs=(), origin_x_px=0, origin_y_px=0, shape=(8, 8))
    with pytest.raises(ValueError, match="must be sorted and unique"):
        _CosmxMosaicGeometry(mosaic=1, fovs=(2, 1), origin_x_px=0, origin_y_px=0, shape=(8, 8))
    with pytest.raises(ValueError, match="shape must contain two positive dimensions"):
        _CosmxMosaicGeometry(mosaic=1, fovs=(1,), origin_x_px=0, origin_y_px=0, shape=(0, 8))
    with pytest.raises(ValueError, match="byte estimates must be nonnegative"):
        _CosmxMosaicSizeEstimate(
            mosaic=1,
            image_nbytes=-1,
            instance_labels_nbytes=0,
            compartment_labels_nbytes=0,
        )


def test_preview_rejects_inconsistent_fov_and_mosaic_relationships() -> None:
    preview = _preview()

    with pytest.raises(ValueError, match="disjoint partition"):
        replace(preview, excluded_fovs=(1, 2))
    with pytest.raises(ValueError, match="unpositioned FOVs must be"):
        replace(preview, unpositioned_fovs=())
    with pytest.raises(ValueError, match="exactly one mosaic"):
        replace(
            preview,
            mosaics=(_CosmxMosaicGeometry(mosaic=1, fovs=(2,), origin_x_px=0, origin_y_px=0, shape=(8, 8)),),
        )
    with pytest.raises(ValueError, match="size-estimate mosaic IDs"):
        replace(
            preview,
            estimates=(
                _CosmxMosaicSizeEstimate(
                    mosaic=2,
                    image_nbytes=640,
                    instance_labels_nbytes=256,
                    compartment_labels_nbytes=64,
                ),
            ),
        )
