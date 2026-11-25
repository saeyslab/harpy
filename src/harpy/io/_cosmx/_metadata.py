from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

from loguru import logger as log
from spatialdata import SpatialData

from harpy.io._cosmx._constants import _FOV_BOUNDS_KEY, _MICRONS_PER_UNIT_KEY, _MOSAIC_KEY, _SAMPLE_ID_KEY
from harpy.io._cosmx._models import _CosmxMosaicGeometry, _CosmxPreview
from harpy.metadata._registry import (
    _COORDINATE_SYSTEMS_METADATA_KEY,
    _FEATURE_PANELS_METADATA_KEY,
    _PROVENANCE_METADATA_KEY,
    _harpy_metadata,
    _metadata_registry,
)


def _coordinate_system_record(
    preview: _CosmxPreview,
    mosaic: _CosmxMosaicGeometry,
    *,
    sample_id: str,
    flip_x: bool,
    flip_y: bool,
) -> dict[str, object]:
    """Return the record of one mosaic's coordinate system: the facts of the mosaic and of its run.

    :func:`harpy.io.cosmx` writes the image, labels and points elements of a
    mosaic unscaled in the mosaic's pixel coordinate system, so the record
    describes all of them; ``microns_per_unit`` is the pixel size. The facts of the run
    (``microns_per_unit``, ``orientation``, ``mosaic_grouping`` and
    ``acquisition_timestamp``) are the same in each coordinate system of a
    sample.
    """
    record: dict[str, object] = {
        _SAMPLE_ID_KEY: sample_id,
        _MOSAIC_KEY: mosaic.mosaic,
        _MICRONS_PER_UNIT_KEY: preview.manifest.run.pixel_size_um,
        "fovs": list(mosaic.fovs),
        _FOV_BOUNDS_KEY: mosaic.fov_bounds_px,
        "source_origin_px": {"x": mosaic.origin_x_px, "y": mosaic.origin_y_px},
        "orientation": {"flip_x": flip_x, "flip_y": flip_y},
        "mosaic_grouping": {
            "mode": preview.mosaic_mode,
            "adjacency_tolerance_px": preview.adjacency_tolerance_px,
        },
    }
    if preview.manifest.run.acquisition_timestamp is not None:
        record["acquisition_timestamp"] = preview.manifest.run.acquisition_timestamp
    return record


def _commit_element_metadata(
    sdata: SpatialData,
    *,
    coordinate_system: tuple[str, Mapping[str, object]],
    registry: str,
    element_name: str,
    record: Mapping[str, object] | None,
    reader_version: str | None,
    feature_panel: tuple[str, Mapping[str, object]] | None = None,
) -> None:
    """Persist the metadata of one element after its payload has been written successfully.

    The record of the element's coordinate system, the element record (if
    any), the optional shared feature panel, and the reader version are
    committed in one root-attributes write. If that write fails, restore the
    preceding attributes and make a best-effort attempt to delete only the newly
    written element before re-raising the original error.

    The coordinate-system record is inserted if absent. The elements of one
    mosaic share it, so a present record must equal the new one; a different
    one raises, as the element would claim a coordinate system that another
    mosaic describes. Without fields of its own, such as points without a
    feature panel, an element has no record (``record=None``).

    Callers validate existing panels and check for key collisions beforehand.
    Insert a panel only if absent; otherwise retain the validated shared record,
    including any additional metadata fields.
    """
    previous_attrs = deepcopy(sdata.attrs)
    try:
        attrs = deepcopy(previous_attrs)
        coordinate_system_name, coordinate_system_record = coordinate_system
        coordinate_system_records = _metadata_registry(attrs, _COORDINATE_SYSTEMS_METADATA_KEY)
        existing = coordinate_system_records.setdefault(coordinate_system_name, dict(coordinate_system_record))
        if existing != coordinate_system_record:
            raise ValueError(
                f"CosMx coordinate system {coordinate_system_name!r} of element {element_name!r} already has a "
                f"different record in sdata.attrs['harpy']['{_COORDINATE_SYSTEMS_METADATA_KEY}']: {existing!r}. "
                "Each mosaic needs a coordinate system of its own; choose another coordinate_system base name."
            )
        if record is not None:
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
