import spatialdata
from dask.dataframe.core import DataFrame as DaskDataFrame
from spatialdata import SpatialData, read_zarr
from spatialdata.models._utils import MappingToCoordinateSystem_t

from harpy.utils._io import _incremental_io_on_disk
from harpy.utils.pylogger import get_pylogger
from harpy.utils.utils import _self_contained_warning_message

log = get_pylogger(__name__)


def add_points_layer(
    sdata: SpatialData,
    ddf: DaskDataFrame,
    output_layer: str,
    coordinates: dict[str, str],
    transformations: MappingToCoordinateSystem_t | None = None,
    overwrite: bool = True,
) -> SpatialData:
    """
    Add a points layer to a SpatialData object.

    This function allows you to add a points layer to `sdata`.
    The points layer is derived from a `Dask` `DataFrame`.
    If `sdata` is backed by a zarr store, the resulting points layer will be backed to the zarr store, otherwise `ddf` will be persisted in memory.

    Parameters
    ----------
    sdata
        The SpatialData object to which the new points layer will be added.
    ddf
        The DaskDataFrame containing the points data to be added.
    output_layer
        The name of the output layer where the points data will be stored.
    coordinates
        A dictionary specifying the coordinate mappings for the points data (e.g., {"x": "x_column", "y": "y_column"}).
    transformations
        Transformations that will be added to the resulting `output_layer`. Currently `harpy` only supports the Identity transformation.
    overwrite
        If True, overwrites `output_layer` if it already exists in `sdata`.

    Returns
    -------
    The `sdata` object with the points layer added.
    """
    points = spatialdata.models.PointsModel.parse(ddf, coordinates=coordinates, transformations=transformations)

    # we persist points if sdata is not backed.
    if not sdata.is_backed():
        points = points.persist()

    if output_layer in [*sdata.points]:
        if sdata.is_backed():
            if overwrite:
                sdata = _incremental_io_on_disk(sdata, output_layer=output_layer, element=points)
            else:
                raise ValueError(
                    f"Attempting to overwrite 'sdata.points[\"{output_layer}\"]', but overwrite is set to False. Set overwrite to True to overwrite the .zarr store."
                )
        else:
            sdata[output_layer] = points

    else:
        sdata[output_layer] = points
        if sdata.is_backed():
            sdata.write_element(output_layer)
            if warning_message := _self_contained_warning_message(sdata, output_layer):
                log.warning(warning_message)
            if len(sdata[output_layer].dask) != 1:
                log.warning(
                    f"Element '{output_layer}' is Dask-backed, but not all tasks have been executed in the in-memory SpatialData object.\n"
                    "This may lead to unexpected behavior if lazy computations are not triggered.\n"
                    "To resolve this, ensure that you assign the result of operations:\n"
                    "    sdata = harpy.{operation}(sdata, ...)\n"
                    "Alternatively, manually reload from the Zarr store:\n"
                    f"    sdata = spatialdata.read_zarr('{sdata.path}')\n"
                    "For more details, see the discussion at:\n"
                    "    https://github.com/saeyslab/harpy/issues/90"
                )
            sdata = read_zarr(sdata.path)

    return sdata
