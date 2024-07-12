from __future__ import annotations

from dask.array import Array
from geopandas import GeoDataFrame
from numpy.typing import NDArray
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from spatialdata import SpatialData
from spatialdata.models._utils import MappingToCoordinateSystem_t

from sparrow.shape._manager import ShapesLayerManager


def _add_shapes_layer(
    sdata: SpatialData,
    input: Array | GeoDataFrame,
    output_layer: str,
    transformations: MappingToCoordinateSystem_t = None,
    overwrite: bool = False,
) -> SpatialData:
    manager = ShapesLayerManager()
    sdata = manager.add_shapes(
        sdata,
        input=input,
        output_layer=output_layer,
        transformations=transformations,
        overwrite=overwrite,
    )

    return sdata


def _filter_shapes_layer(
    sdata: SpatialData,
    indexes_to_keep: NDArray,
    prefix_filtered_shapes_layer: str,
) -> SpatialData:
    manager = ShapesLayerManager()
    sdata = manager.filter_shapes(
        sdata,
        indexes_to_keep=indexes_to_keep,
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
