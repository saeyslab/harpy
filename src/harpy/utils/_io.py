from __future__ import annotations

import uuid
import warnings
from typing import Literal

from anndata import AnnData
from dask.dataframe import DataFrame
from geopandas import GeoDataFrame
from spatialdata import SpatialData, read_zarr
from xarray import DataArray, DataTree

from harpy.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def _incremental_io_on_disk(
    sdata: SpatialData,
    output_layer: str,
    element: DataArray | DataTree | DataFrame | GeoDataFrame | AnnData,
    element_type: str = Literal["images", "labels", "shapes", "tables", "points"],
) -> SpatialData:
    assert element_type in [
        "images",
        "labels",
        "shapes",
        "tables",
        "points",
    ], "'element_type' should be one of [ 'images', 'labels', 'shapes', 'tables', 'points' ]"
    new_output_layer = f"{output_layer}_{uuid.uuid4()}"
    # a. write a backup copy of the data
    sdata[new_output_layer] = element
    try:
        sdata.write_element(new_output_layer)
    except Exception as e:
        if new_output_layer in sdata[new_output_layer]:
            del sdata[new_output_layer]
        raise e
    # a2. remove the in-memory copy from the SpatialData object (note,
    # at this point the backup copy still exists on-disk)
    del sdata[new_output_layer]
    del sdata[output_layer]
    # a3 load the backup copy into memory
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="The table is annotating",
            module="spatialdata._core.spatialdata",
        )
        sdata_copy = read_zarr(sdata.path, selection=[element_type])
    # b1. rewrite the original data
    sdata.delete_element_from_disk(output_layer)
    sdata[output_layer] = sdata_copy[new_output_layer]
    log.warning(f"layer with name '{output_layer}' already exists. Overwriting...")
    sdata.write_element(output_layer)
    # b2. reload the new data into memory (because it has been written but in-memory it still points
    # from the backup location)
    del sdata[output_layer]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="The table is annotating",
            module="spatialdata._core.spatialdata",
        )
        sdata_materialized = read_zarr(sdata.path, selection=[element_type])
    # to make sdata point to layer that is materialized, and keep object id.
    sdata[output_layer] = sdata_materialized[output_layer]
    # c. remove the backup copy
    del sdata_materialized[new_output_layer]
    sdata_materialized.delete_element_from_disk(new_output_layer)
    del sdata_materialized

    return sdata
