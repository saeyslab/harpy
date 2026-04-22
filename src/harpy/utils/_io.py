from __future__ import annotations

import uuid
import warnings
from typing import Literal

from anndata import AnnData
from dask.dataframe import DataFrame
from geopandas import GeoDataFrame
from loguru import logger as log
from spatialdata import SpatialData, read_zarr
from xarray import DataArray, DataTree


def _write_element_with_cleanup(sdata: SpatialData, element_name: str) -> None:
    """Write an already-attached element and remove partial state if the write fails."""
    try:
        sdata.write_element(element_name)
    except Exception as e:
        log.warning(
            f"Writing element '{element_name}' failed with error: {e}. Attempting best-effort cleanup before re-raising."
        )
        if sdata.get(element_name) is not None:
            del sdata[element_name]
        try:
            sdata.delete_element_from_disk(element_name)
        except Exception as e:  # noqa: BLE001
            log.warning(f"Best-effort cleanup failed for element '{element_name}': {e}")
        raise


def _read_zarr_with_annotating_table_warning_suppressed(
    path: str,
    selection: list[str],
) -> SpatialData:
    """Read a partial SpatialData selection without surfacing expected table-target warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="The table is annotating",
            module="spatialdata._core.spatialdata",
        )
        return read_zarr(path, selection=selection)


def _incremental_io_on_disk(
    sdata: SpatialData,
    element_name: str,
    element: DataArray | DataTree | DataFrame | GeoDataFrame | AnnData,
    element_type: Literal["images", "labels", "shapes", "tables", "points"] = "images",
) -> SpatialData:
    assert element_type in [
        "images",
        "labels",
        "shapes",
        "tables",
        "points",
    ], "'element_type' should be one of [ 'images', 'labels', 'shapes', 'tables', 'points' ]"
    temporary_element_name = f"{element_name}_{uuid.uuid4()}"
    # a. write a backup copy of the data
    sdata[temporary_element_name] = element
    try:
        sdata.write_element(temporary_element_name)
    except Exception as e:
        log.warning(
            f"Writing temporary element '{temporary_element_name}' failed with error: {e}. "
            "Attempting best-effort cleanup before re-raising."
        )
        if sdata.get(temporary_element_name) is not None:
            del sdata[temporary_element_name]
        try:
            sdata.delete_element_from_disk(temporary_element_name)
        except Exception as e:  # noqa: BLE001
            log.warning(f"Best-effort cleanup failed for temporary element '{temporary_element_name}': {e}")
        raise
    # a2. remove the in-memory copy from the SpatialData object (note,
    # at this point the backup copy still exists on-disk)
    del sdata[temporary_element_name]
    del sdata[element_name]
    # a3 load the backup copy into memory
    sdata_copy = _read_zarr_with_annotating_table_warning_suppressed(sdata.path, selection=[element_type])
    # b1. rewrite the original data
    sdata.delete_element_from_disk(element_name)
    sdata[element_name] = sdata_copy[temporary_element_name]
    log.warning(f"Element with name '{element_name}' already exists. Overwriting...")
    _write_element_with_cleanup(sdata, element_name)
    # b2. reload the new data into memory (because it has been written but in-memory it still points
    # from the backup location)
    del sdata[element_name]
    sdata_materialized = _read_zarr_with_annotating_table_warning_suppressed(sdata.path, selection=[element_type])
    # to make sdata point to layer that is materialized, and keep object id.
    sdata[element_name] = sdata_materialized[element_name]
    # c. remove the backup copy
    del sdata_materialized[temporary_element_name]
    sdata_materialized.delete_element_from_disk(temporary_element_name)
    del sdata_materialized

    return sdata
