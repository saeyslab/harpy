from __future__ import annotations

import geopandas
import numpy as np
from geopandas import GeoDataFrame
from longsgis import voronoiDiagram4plg
from shapely.geometry import Polygon
from spatialdata import SpatialData
from spatialdata.transformations import get_transformation

from sparrow.shape._shape import add_shapes_layer
from sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def create_voronoi_boundaries(
    sdata: SpatialData,
    shapes_layer: str = "segmentation_mask_boundaries",
    output_layer: str | None = None,
    radius: int = 0,
    overwrite: bool = False,
) -> SpatialData:
    """
    Create Voronoi boundaries from the shapes layer of the provided SpatialData object.

    Given a SpatialData object and a radius, this function calculates Voronoi boundaries
    and expands these boundaries based on the radius.

    Parameters
    ----------
    sdata
        The spatial data object on which Voronoi boundaries will be created.
    shapes_layer
        The name of the layer in `sdata` representing shapes used to derive
        Voronoi boundaries. Default is "segmentation_mask_boundaries".
    output_layer
        Name of the resulting shapes layer that will be added to `sdata`.
    radius
        The expansion radius for the Voronoi boundaries, by default 0.
        If provided, Voronoi boundaries will be expanded by this radius.
        Must be non-negative.
    overwrite
        If True, overwrites the `output_layer` if it already exists in `sdata`.

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

    if output_layer is None:
        output_layer = f"expanded_cells{radius}"
        log.info(f"Name of the output layer is not provided. Setting to '{output_layer}'.")

    x_min, y_min, x_max, y_max = sdata[shapes_layer].geometry.total_bounds

    boundary = Polygon(
        [
            (x_min, y_min),
            (x_max, y_min),
            (x_max, y_max),
            (x_min, y_max),
        ]
    )

    gdf = sdata[shapes_layer].copy()
    gdf["geometry"] = gdf.simplify(2)

    vd = voronoiDiagram4plg(gdf, boundary)
    voronoi = geopandas.sjoin(vd, gdf, predicate="contains", how="left")
    voronoi.index = voronoi.index_right
    voronoi = voronoi[~voronoi.index.duplicated(keep="first")]
    voronoi = _delete_overlap(voronoi, gdf)

    buffered = gdf.buffer(distance=radius)
    intersected = voronoi.sort_index().intersection(buffered.sort_index())

    gdf.geometry = intersected

    # sanity check. If this sanity check would fail in spatialdata at some point, then pass transformation to transformations parameter of add_shapes_layer.
    assert get_transformation(gdf, get_all=True) == get_transformation(sdata[shapes_layer], get_all=True)

    sdata = add_shapes_layer(
        sdata,
        input=gdf,
        output_layer=output_layer,
        transformations=None,
        overwrite=overwrite,
    )

    return sdata


def _delete_overlap(voronoi: GeoDataFrame, polygons: GeoDataFrame) -> GeoDataFrame:
    I1, I2 = voronoi.sindex.query(voronoi["geometry"], predicate="overlaps")
    voronoi2 = voronoi.copy()

    geometry_loc = voronoi.columns.get_loc("geometry")

    for cell1, cell2 in zip(I1, I2):
        voronoi.iloc[cell1, geometry_loc] = voronoi.iloc[cell1].geometry.intersection(
            voronoi2.iloc[cell1].geometry.difference(voronoi2.iloc[cell2].geometry)
        )
        voronoi.iloc[cell2, geometry_loc] = voronoi.iloc[cell2].geometry.intersection(
            voronoi2.iloc[cell2].geometry.difference(voronoi2.iloc[cell1].geometry)
        )
    assert np.array_equal(
        np.sort(voronoi.index), np.sort(polygons.index)
    ), "Indices of voronoi and polygons do not match"
    polygons = polygons.reindex(voronoi.index)
    voronoi["geometry"] = voronoi.geometry.union(polygons.geometry)
    polygons = polygons.buffer(distance=0)
    voronoi = voronoi.buffer(distance=0)
    return voronoi
