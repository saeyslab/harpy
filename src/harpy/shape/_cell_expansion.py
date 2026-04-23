from __future__ import annotations

import geopandas
import numpy as np
from geopandas import GeoDataFrame
from loguru import logger as log
from longsgis import voronoiDiagram4plg
from shapely.geometry import Polygon
from spatialdata import SpatialData
from spatialdata.transformations import get_transformation

from harpy.shape._shape import add_shapes


def create_voronoi_boundaries(
    sdata: SpatialData,
    shapes_name: str = "segmentation_mask_boundaries",
    output_shapes_name: str | None = None,
    radius: int = 0,
    overwrite: bool = False,
) -> SpatialData:
    """
    Create Voronoi boundaries from the shapes element of the provided SpatialData object.

    Given a SpatialData object and a radius, this function calculates Voronoi boundaries
    and expands these boundaries based on the radius.

    Parameters
    ----------
    sdata
        The spatial data object on which Voronoi boundaries will be created.
    shapes_name
        The name of the shapes element in `sdata` used to derive
        Voronoi boundaries. Default is "segmentation_mask_boundaries".
    output_shapes_name
        Name of the resulting shapes element that will be added to `sdata`.
    radius
        The expansion radius for the Voronoi boundaries, by default 0.
        If provided, Voronoi boundaries will be expanded by this radius.
        Must be non-negative.
    overwrite
        If True, overwrites the `output_shapes_name` if it already exists in `sdata`.

    Returns
    -------
    Modified `sdata` object with the Voronoi boundaries created and
    possibly expanded.

    Raises
    ------
    ValueError
        If the provided radius is negative.
    """
    if radius < 0:
        raise ValueError(f"radius should be >0, provided value for radius is '{radius}'")

    if output_shapes_name is None:
        output_shapes_name = f"expanded_cells{radius}"
        log.info(f"Name of the output element is not provided. Setting to '{output_shapes_name}'.")

    x_min, y_min, x_max, y_max = sdata[shapes_name].geometry.total_bounds

    boundary = Polygon(
        [
            (x_min, y_min),
            (x_max, y_min),
            (x_max, y_max),
            (x_min, y_max),
        ]
    )

    gdf = sdata[shapes_name].copy()
    gdf["geometry"] = gdf.simplify(2)

    vd = voronoiDiagram4plg(gdf, boundary)
    voronoi = geopandas.sjoin(vd, gdf, predicate="contains", how="left")
    voronoi.index = voronoi.index_right
    voronoi = voronoi[~voronoi.index.duplicated(keep="first")]
    voronoi = _delete_overlap(voronoi, gdf)

    buffered = gdf.buffer(distance=radius)
    intersected = voronoi.sort_index().intersection(buffered.sort_index())

    gdf.geometry = intersected

    # sanity check. If this sanity check would fail in spatialdata at some point, then pass transformation to transformations parameter of add_shapes.
    assert get_transformation(gdf, get_all=True) == get_transformation(sdata[shapes_name], get_all=True)

    sdata = add_shapes(
        sdata,
        input=gdf,
        output_shapes_name=output_shapes_name,
        transformations=None,
        overwrite=overwrite,
    )

    return sdata


def _delete_overlap(voronoi: GeoDataFrame, polygons: GeoDataFrame) -> GeoDataFrame:
    I1, I2 = voronoi.sindex.query(voronoi["geometry"], predicate="overlaps")
    voronoi2 = voronoi.copy()

    geometry_loc = voronoi.columns.get_loc("geometry")

    for cell1, cell2 in zip(I1, I2, strict=True):
        voronoi.iloc[cell1, geometry_loc] = voronoi.iloc[cell1].geometry.intersection(
            voronoi2.iloc[cell1].geometry.difference(voronoi2.iloc[cell2].geometry)
        )
        voronoi.iloc[cell2, geometry_loc] = voronoi.iloc[cell2].geometry.intersection(
            voronoi2.iloc[cell2].geometry.difference(voronoi2.iloc[cell1].geometry)
        )
    assert np.array_equal(np.sort(voronoi.index), np.sort(polygons.index)), (
        "Indices of voronoi and polygons do not match"
    )
    polygons = polygons.reindex(voronoi.index)
    voronoi["geometry"] = voronoi.geometry.union(polygons.geometry)
    polygons = polygons.buffer(distance=0)
    voronoi = voronoi.buffer(distance=0)
    return voronoi
