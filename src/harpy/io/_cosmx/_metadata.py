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


def _commit_element_metadata(
    sdata: SpatialData,
    *,
    registry: str,
    element_name: str,
    record: Mapping[str, object],
    reader_version: str | None,
    feature_panel: tuple[str, Mapping[str, object]] | None = None,
    cleanup_element_on_failure: bool = True,
) -> None:
    """Persist one element record after its payload has been written successfully.

    The element record, optional shared feature panel, and reader version are
    committed in one root-attributes write. If that write fails, restore the
    preceding attributes and make a best-effort attempt to delete only the newly
    written element before re-raising the original error.
    """
    previous_attrs = deepcopy(sdata.attrs)
    try:
        attrs = deepcopy(previous_attrs)
        _metadata_registry(attrs, registry)[element_name] = dict(record)
        if feature_panel is not None:
            panel_name, panel_record = feature_panel
            _metadata_registry(attrs, _FEATURE_PANELS_METADATA_KEY)[panel_name] = dict(panel_record)
        if reader_version is not None:
            _harpy_metadata(attrs)[_PROVENANCE_METADATA_KEY] = {
                "reader": "cosmx",
                "reader_version": reader_version,
            }
        sdata.attrs = attrs
        sdata.write_attrs()
    except Exception:
        if cleanup_element_on_failure:
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
