from __future__ import annotations

from dask.array import Array
from geopandas import GeoDataFrame
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from spatialdata import SpatialData
from spatialdata.models._utils import MappingToCoordinateSystem_t
from spatialdata.transformations import get_transformation

from harpy.image._image import _get_spatial_element
from harpy.shape._manager import ShapesLayerManager
from harpy.utils._keys import _INSTANCE_KEY


def vectorize(
    sdata,
    labels_name: str,
    output_shapes_name: str,
    instance_key: str = _INSTANCE_KEY,
    overwrite: bool = False,
) -> SpatialData:
    """
    Vectorize a labels element.

    Convert a labels element to a shapes element with name `output_shapes_name`.
    If the `rasterio` library is installed will use implementation based on `rasterio`, else will use implementation based on `skimage`.
    We recommend installing `rasterio` for increased performance and more precise vectorization.

    For optimal performance it is recommended to configure `Dask` so it uses "processes" instead of "threads", e.g. via:

    >>> import dask
    >>> dask.config.set(scheduler='processes')

    Parameters
    ----------
    sdata
        The SpatialData object to which the new shapes element will be added.
    labels_name
        The labels element to vectorize.
    output_shapes_name
        The name of the output shapes element where the shapes data will be stored.
    instance_key
        Name of the resulting index of the GeoDataFrame.
        The user can set this to any value, it will only be used to name the index.
    overwrite
        If True, overwrites `output_shapes_name` if it already exists in `sdata`.

    Returns
    -------
    The `sdata` object with the shapes element added.
    """
    se = _get_spatial_element(sdata, layer=labels_name)
    sdata = add_shapes(
        sdata,
        input=se.data,
        output_shapes_name=output_shapes_name,
        transformations=get_transformation(sdata[labels_name], get_all=True),
        instance_key=instance_key,
        overwrite=overwrite,
    )
    return sdata


def add_shapes(
    sdata: SpatialData,
    input: Array | GeoDataFrame,
    output_shapes_name: str,
    transformations: MappingToCoordinateSystem_t = None,
    instance_key: str = _INSTANCE_KEY,
    overwrite: bool = False,
) -> SpatialData:
    """
    Add a shapes element to a SpatialData object.

    This function allows you to add a shapes element to `sdata`.
    The shapes element can be derived from a Dask array or a GeoDataFrame.
    If `sdata` is backed by a zarr store, the resulting shapes element will be backed to the zarr store.

    Parameters
    ----------
    sdata
        The SpatialData object to which the new shapes element will be added.
    input
        The input data containing the shapes, either as an array (i.e. segmentation masks) or a GeoDataFrame.
    output_shapes_name
        The name of the output shapes element where the shapes data will be stored.
    transformations
        Transformations that will be added to the resulting `output_shapes_name`.
    instance_key
        Name of the resulting index of the GeoDataFrame.
        The user can set this to any value, it will only be used to name the index.
    overwrite
        If True, overwrites `output_shapes_name` if it already exists in `sdata`.

    Returns
    -------
    The `sdata` object with the shapes element added.
    """
    manager = ShapesLayerManager()
    sdata = manager.add_shapes(
        sdata,
        input=input,
        output_shapes_name=output_shapes_name,
        transformations=transformations,
        instance_key=instance_key,
        overwrite=overwrite,
    )

    return sdata


def filter_shapes_layer(
    sdata: SpatialData,
    table_name: str,
    labels_name: str,
    prefix_filtered_shapes_name: str,
) -> SpatialData:
    """
    Filter shapes in a SpatialData object.

    Instances that do not appear in `table_name` (with region key equal to `labels_name`) will be removed from the shapes elements, via the instance key of `sdata.tables[table_name].obs`) and the index of the shapes elements in the `sdata` object.
    Only shapes elements of `sdata` in same coordinate system as the `labels_name` will be considered.
    Polygons that are filtered out from a shapes element (e.g. with name "shapes_example") will be added as a new shapes element with name `prefix_filtered_shapes_name` + "_" + "shapes_example".

    Parameters
    ----------
    sdata
        The SpatialData object,
    table_name
        The name of the table element.
    labels_name
        The name of the labels element.
    prefix_filtered_shapes_name
        The prefix for the name of the new shapes element consisting of the polygons that where filtered out from a shapes element.

    Returns
    -------
    The updated `sdata` object.
    """
    manager = ShapesLayerManager()

    sdata = manager.filter_shapes(
        sdata,
        table_name=table_name,
        labels_name=labels_name,
        prefix_filtered_shapes_name=prefix_filtered_shapes_name,
    )
    return sdata


def _extract_boundaries_from_geometry_collection(geometry):
    if isinstance(geometry, Polygon):
        return [geometry.boundary]
    elif isinstance(geometry, MultiPolygon):
        return [polygon.boundary for polygon in geometry.geoms]
    elif isinstance(geometry, GeometryCollection):
        boundaries = []
        for geom in geometry.geoms:
            boundaries.extend(_extract_boundaries_from_geometry_collection(geom))
        return boundaries
    else:
        return []


def intersect_rectangles(rect1: list[int | float], rect2: list[int | float]) -> list[int | float] | None:
    """
    Calculate the intersection of two (axis aligned) rectangles.

    Parameters
    ----------
    rect1 : List[int | float]
        List representing the first rectangle [x_min, x_max, y_min, y_max].
    rect2 : List[int | float]
        List representing the second rectangle [x_min, x_max, y_min, y_max].

    Returns
    -------
    Optional[List[int | float]]
        List representing the intersection rectangle [x_min, x_max, y_min, y_max],
        or None if the rectangles do not overlap.
    """
    overlap_x = not (rect1[1] <= rect2[0] or rect2[1] <= rect1[0])
    overlap_y = not (rect1[3] <= rect2[2] or rect2[3] <= rect1[2])

    if overlap_x and overlap_y:
        x_min = max(rect1[0], rect2[0])
        x_max = min(rect1[1], rect2[1])
        y_min = max(rect1[2], rect2[2])
        y_max = min(rect1[3], rect2[3])
        return [x_min, x_max, y_min, y_max]
    else:
        return None
