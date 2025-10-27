from __future__ import annotations

from dask.array import Array
from geopandas import GeoDataFrame
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, Point, MultiPoint, LineString, MultiLineString
from spatialdata import SpatialData
from spatialdata.models._utils import MappingToCoordinateSystem_t
from spatialdata.transformations import get_transformation

from harpy.image._image import _get_spatial_element
from harpy.shape._manager import ShapesLayerManager

import numpy as np

def vectorize(
    sdata,
    labels_layer: str,
    output_layer: str,
    overwrite: bool = False,
) -> SpatialData:
    """
    Vectorize a labels layer.

    Convert a labels layer to a shapes layer with name `output_layer`.
    If the `rasterio` library is installed will use implementation based on `rasterio`, else will use implementation based on `skimage`.
    We recommend installing `rasterio` for increased perforamnce and more precise vectorization.

    For optimal performance it is recommended to configure `Dask` so it uses "processes" instead of "threads", e.g. via:

    >>> import dask
    >>> dask.config.set(scheduler='processes')

    Parameters
    ----------
    sdata
        The SpatialData object to which the new shapes layer will be added.
    labels_layer
        The labels layer to vectorize.
    output_layer
        The name of the output layer where the shapes data will be stored.
    overwrite
        If True, overwrites the output layer if it already exists in `sdata`.

    Returns
    -------
    The `sdata` object with the shapes layer added.
    """
    se = _get_spatial_element(sdata, layer=labels_layer)
    sdata = add_shapes_layer(
        sdata,
        input=se.data,
        output_layer=output_layer,
        transformations=get_transformation(sdata[labels_layer], get_all=True),
        overwrite=overwrite,
    )
    return sdata


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
    
    
def prep_region_annotations(
    sdata: SpatialData,
    shapes_layer: str,
    output_shapes_layer: str,
    shape_names_column: str = "name",
    unnamed: str = "unnamed",
    unique_shape_names_column: str = "name-unique",
    erosion: float = 0.5,
    overwrite: bool = False,
):
    """
    Prepares region annotations in a shapes layer for `hp.sh.filter_by_morphology`, `hp.tb.assign_cells_to_shapes` and `hp.tb.compute_distance_to_shapes`.
    Operations performed:
        - Ensures a shape name column exists and fills missing names.
        - Converts Points with a `radius` column into circular polygons. Points without a `radius` column will be preserved as Points.
        - Slightly erodes polygons to avoid shared edges.
        - Explodes multipolygons into separate single polygons.
        - Generates unique names for shapes with duplicate base names.

    Parameters
    ----------
    sdata
        The SpatialData object containing the input shapes layer.
    shapes_layer
        The shapes layer in `sdata.shapes` to use as input.
    output_shapes_layer
        The output shapes layer in `sdata.tables` to which the updated shapes layer will be written.
    shape_names_column
        Column name in shapes layer containing geometry names. If not present, new names will be generated.
    unnamed
        Name to be assigned to any unnamed geometries in `shape_names_column`. Defaults to 'unnamed'.
    unique_shape_names_column
        Column name in which unique names will be created for single polygons by appending a counter to the original name in `shape_names_column` for polygons with the same name. Note 
        that multipolygons will be split in individual polygons and each will get a unique name based on the original name of the multipolygon. Unique names will be stored in 
        `{shape_names_column}-unique` in the updated shapes layer.
    erosion
        Number of pixels to erode polygons by. This can avoid problems with overlapping edges of geometries when calculating distances. Default is 0.5 (i.e. erosion by 0.5 pixels).    
    overwrite
        If True, overwrites the `output_shapes_layer` if it already exists in `sdata`.
        
    Returns
    -------
    Modified `sdata` object with updated updated shapes layer.
    """
    
    # Create copy of shapes layer
    gdf = sdata.shapes[shapes_layer].copy()
    print(f"Found {len(gdf)} geometries in {shapes_layer}.")
    
    # Ensure shape_names_column exist
    if shape_names_column not in gdf.columns:
        gdf[shape_names_column] = unnamed
        
    gdf[shape_names_column] = gdf[shape_names_column].fillna("").astype(str)
    unnamed_mask = gdf[shape_names_column] == ""
    for i, idx in enumerate(gdf[unnamed_mask].index):
        gdf.at[idx, shape_names_column] = unnamed
    
    # Convert Points with a radius column to circular polygons
    def _point_to_circle(geom, radius=None):
        if isinstance(geom, Point) and radius is not None:
            return geom.buffer(radius, resolution=16)
        return geom
    
    if "radius" in gdf.columns:
        mask_points_with_radius = gdf.geometry.geom_type.eq("Point") & gdf["radius"].notna()
        n_converted = mask_points_with_radius.sum()

        if n_converted > 0:
            print(f"Converting {n_converted} Point geometries with 'radius' to circular polygons.")
            gdf.loc[mask_points_with_radius, "geometry"] = gdf.loc[mask_points_with_radius].apply(
                lambda row: _point_to_circle(row.geometry, getattr(row, "radius", None)),
                axis=1
            )
        
    # Slightly erode all polygons (this avoids any shared borders between polygons)
    gdf["geometry"] = gdf.geometry.apply(
        lambda geom: geom.buffer(
            -erosion, 
            join_style=2, 
            resolution=16)
        if geom.geom_type in ["Polygon", "MultiPolygon"] else geom
    )
    polygon_mask = ~gdf.geometry.is_empty 
    removed = len(gdf) - polygon_mask.sum() 
    gdf = gdf[polygon_mask] # drop any polygons that collapsed to empty 
    if removed > 0: 
        print(f"Removed {removed} polygons that collapsed to empty after erosion.")
        
    # Explode multipolygons into single polygons (this allows us to treat multipolygons as unique polygons)
    n_multipolygons = gdf.geometry.apply(lambda g: g.geom_type == "MultiPolygon").sum()
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)
    n_after = len(gdf)
    print(f"Split {n_multipolygons} multipolygons into individual polygons. Total number of geometries after splitting multipolgons is {n_after}.")
    
    # Create unique names
    sizes = gdf.groupby(shape_names_column)[shape_names_column].transform("size")
    counter = gdf.groupby(shape_names_column).cumcount() + 1
    gdf[unique_shape_names_column] = np.where(
        sizes.eq(1),
        gdf[shape_names_column],
        gdf[shape_names_column] + counter.astype(str)
    )

    # Add filtered shapes layer
    sdata = add_shapes_layer(
        sdata,
        input=gdf,
        output_layer=output_shapes_layer,
        overwrite=overwrite,
    )

    return sdata
        
