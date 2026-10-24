from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

from loguru import logger as log
from spatialdata import SpatialData

from harpy._metadata import (
    _FEATURE_PANELS_METADATA_KEY,
    _PROVENANCE_METADATA_KEY,
    _harpy_metadata,
    _metadata_registry,
)
from harpy.io._cosmx._constants import _FOV_BOUNDS_KEY, _SAMPLE_ID_KEY
from harpy.io._cosmx._models import _CosmxMosaicGeometry, _CosmxPreview


def _common_element_record(
    preview: _CosmxPreview,
    mosaic: _CosmxMosaicGeometry,
    *,
    sample_id: str,
    flip_x: bool,
    flip_y: bool,
) -> dict[str, object]:
    """Return the record fields shared by the image, labels and points elements of one mosaic."""
    record: dict[str, object] = {
        "fovs": list(mosaic.fovs),
        _SAMPLE_ID_KEY: sample_id,
        "mosaic": {
            "mode": preview.mosaic_mode,
            "adjacency_tolerance_px": preview.adjacency_tolerance_px,
        },
        "source_origin_px": {"x": mosaic.origin_x_px, "y": mosaic.origin_y_px},
        _FOV_BOUNDS_KEY: mosaic.fov_bounds_px,
        "orientation": {"flip_x": flip_x, "flip_y": flip_y},
        "pixel_size_um": preview.manifest.run.pixel_size_um,
    }
    if preview.manifest.run.acquisition_timestamp is not None:
        record["acquisition_timestamp"] = preview.manifest.run.acquisition_timestamp
    return record


def _commit_element_metadata(
    sdata: SpatialData,
    *,
    registry: str,
    element_name: str,
    record: Mapping[str, object],
    reader_version: str | None,
    feature_panel: tuple[str, Mapping[str, object]] | None = None,
) -> None:
    """Persist one element record after its payload has been written successfully.

    The element record, optional shared feature panel, and reader version are
    committed in one root-attributes write. If that write fails, restore the
    preceding attributes and make a best-effort attempt to delete only the newly
    written element before re-raising the original error.

    Callers validate existing panels and check for key collisions beforehand.
    Insert a panel only if absent; otherwise retain the validated shared record,
    including any additional metadata fields.
    """
    previous_attrs = deepcopy(sdata.attrs)
    try:
        attrs = deepcopy(previous_attrs)
        _metadata_registry(attrs, registry)[element_name] = dict(record)
        if feature_panel is not None:
            panel_name, panel_record = feature_panel
            _metadata_registry(attrs, _FEATURE_PANELS_METADATA_KEY).setdefault(panel_name, dict(panel_record))
        if reader_version is not None:
            _harpy_metadata(attrs)[_PROVENANCE_METADATA_KEY] = {
                "reader": "cosmx",
                "reader_version": reader_version,
            }
        sdata.attrs = attrs
        sdata.write_attrs()
    except Exception:
        if sdata.get(element_name) is not None:
            del sdata[element_name]
        try:
            sdata.delete_element_from_disk(element_name)
        except Exception as cleanup_error:  # noqa: BLE001
            log.warning(
                f"Best-effort cleanup failed for CosMx element {element_name!r} after its metadata "
                f"commit failed: {cleanup_error}"
            )
        sdata.attrs = previous_attrs
        try:
            sdata.write_attrs()
        except Exception as restore_error:  # noqa: BLE001
            log.warning(
                f"Best-effort restoration of SpatialData attributes failed after the metadata commit for "
                f"CosMx element {element_name!r}: {restore_error}"
            )
        raise
