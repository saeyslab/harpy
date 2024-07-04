import uuid

from anndata import AnnData
from dask.dataframe import DataFrame
from datatree import DataTree
from geopandas import GeoDataFrame
from spatialdata import SpatialData, read_zarr
from xarray import DataArray

from sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def _incremental_io_on_disk(
    sdata: SpatialData,
    output_layer: str,
    element: DataArray | DataTree | DataFrame | GeoDataFrame | AnnData,
) -> SpatialData:
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
    sdata_copy = read_zarr(sdata.path)
    # b1. rewrite the original data
    sdata.delete_element_from_disk(output_layer)
    sdata[output_layer] = sdata_copy[new_output_layer]
    log.warning(f"layer with name '{output_layer}' already exists. Overwriting...")
    sdata.write_element(output_layer)
    # b2. reload the new data into memory (because it has been written but in-memory it still points
    # from the backup location)
    sdata = read_zarr(sdata.path)
    # c. remove the backup copy
    del sdata[new_output_layer]
    sdata.delete_element_from_disk(new_output_layer)

    return sdata
