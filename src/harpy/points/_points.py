import spatialdata
from dask.dataframe import DataFrame as DaskDataFrame
from spatialdata import SpatialData, read_zarr
from spatialdata.models._utils import MappingToCoordinateSystem_t

from harpy.utils._io import _incremental_io_on_disk, _write_element_with_cleanup


def add_points(
    sdata: SpatialData,
    ddf: DaskDataFrame,
    output_points_name: str,
    coordinates: dict[str, str],
    transformations: MappingToCoordinateSystem_t | None = None,
    overwrite: bool = True,
) -> SpatialData:
    """
    Add a points element to a SpatialData object.

    This function allows you to add a points element to `sdata`.
    The points element is derived from a `Dask` `DataFrame`.
    If `sdata` is backed by a zarr store, the resulting points element will be backed to the zarr store, otherwise `ddf` will be persisted in memory.

    Parameters
    ----------
    sdata
        The SpatialData object to which the new points element will be added.
    ddf
        The DaskDataFrame containing the points data to be added.
    output_points_name
        The name of the output points element where the points data will be stored.
    coordinates
        A dictionary specifying the coordinate mappings for the points data (e.g., {"x": "x_column", "y": "y_column"}).
    transformations
        Transformations that will be added to the resulting `output_points_name`. Currently `harpy` only supports the Identity transformation.
    overwrite
        If True, overwrites `output_points_name` if it already exists in `sdata`.

    Returns
    -------
    The `sdata` object with the points element added.
    """
    points = spatialdata.models.PointsModel.parse(ddf, coordinates=coordinates, transformations=transformations)

    # We persist points if sdata is not backed, but Dask does not carry .attrs through persist().
    # Keep a copy of attrs (includes SpatialData transform metadata) and restore it safely after persist.
    if not sdata.is_backed():
        attrs = dict(points.attrs)
        points = points.persist()
        points.attrs.update(attrs)

    if output_points_name in [*sdata.points]:
        if sdata.is_backed():
            if overwrite:
                sdata = _incremental_io_on_disk(
                    sdata, element_name=output_points_name, element=points, element_type="points"
                )
            else:
                raise ValueError(
                    f"Attempting to overwrite 'sdata.points[\"{output_points_name}\"]', but overwrite is set to False. Set overwrite to True to overwrite the .zarr store."
                )
        else:
            sdata[output_points_name] = points

    else:
        sdata[output_points_name] = points
        if sdata.is_backed():
            _write_element_with_cleanup(sdata, output_points_name)
            del sdata[output_points_name]
            sdata_temp = read_zarr(sdata.path, selection=["points"])
            sdata[output_points_name] = sdata_temp[output_points_name]
            del sdata_temp
    return sdata
