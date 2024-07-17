from __future__ import annotations

from dask.array import Array
from geopandas import GeoDataFrame
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from spatialdata import SpatialData
from spatialdata.models._utils import MappingToCoordinateSystem_t

from sparrow.shape._manager import ShapesLayerManager


def add_shapes_layer(
    sdata: SpatialData,
    input: Array | GeoDataFrame,
    output_layer: str,
    transformations: MappingToCoordinateSystem_t = None,
    overwrite: bool = False,
) -> SpatialData:
    """
    Add a shapes layer to a SpatialData object.

    This function allows you to add a shapes layer to `sdata`.
    The shapes layer can be derived from a Dask array or a GeoDataFrame.
    If `sdata` is backed by a zarr store, the resulting shapes layer will be backed to the zarr store.

    Parameters
    ----------
    sdata
        The SpatialData object to which the new shapes layer will be added.
    input
        The input data containing the shapes, either as an array (i.e. segmentation masks) or a GeoDataFrame.
    output_layer
        The name of the output layer where the shapes data will be stored.
    transformations
        Transformations that will be added to the resulting `output_layer`.
    overwrite
        If True, overwrites the output layer if it already exists in `sdata`.

    Returns
    -------
    The `sdata` object with the shapes layer added.
    """
    manager = ShapesLayerManager()
    sdata = manager.add_shapes(
        sdata,
        input=input,
        output_layer=output_layer,
        transformations=transformations,
        overwrite=overwrite,
    )

    return sdata


def filter_shapes_layer(
    sdata: SpatialData,
    table_layer: str,
    labels_layer: str,
    prefix_filtered_shapes_layer: str,
) -> SpatialData:
    """
    Filter shapes in a SpatialData object.

    Cells that do not appear in `table_layer` (with `_REGION_KEY` equal to `labels_layer`) will be removed from the shapes layers, via the `_INSTANCE_KEY` of `sdata.tables[table_layer].obs`) and the index of the shapes layers in the `sdata` object.
    Only shapes layers of `sdata` in same coordinate system as the `labels_layer` will be considered.
    Polygons that are filtered out from a shapes layer (e.g. with name "shapes_example") will be added as a new shapes layer with name `prefix_filtered_shapes_layer` + "_" + "shapes_example".

    Parameters
    ----------
    sdata
        The SpatialData object,
    table_layer
        The name of the table layer.
    labels_layer
        The name of the labels layer.
    prefix_filtered_shapes_layer
        The prefix for the name of the new shapes layer consisting of the polygons that where filtered out from a shapes layer.

    Returns
    -------
    The updated `sdata` object.
    """
    manager = ShapesLayerManager()

    sdata = manager.filter_shapes(
        sdata,
        table_layer=table_layer,
        labels_layer=labels_layer,
        prefix_filtered_shapes_layer=prefix_filtered_shapes_layer,
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
