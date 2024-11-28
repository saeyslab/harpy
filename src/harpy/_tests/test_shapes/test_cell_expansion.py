from geopandas import GeoDataFrame

from harpy.shape import create_voronoi_boundaries


def test_cell_expansion(sdata_multi_c_no_backed):
    sdata_multi_c_no_backed = create_voronoi_boundaries(
        sdata_multi_c_no_backed,
        shapes_layer="masks_cellpose_boundaries",
        output_layer="masks_cellpose_boundaries_voronoi",
        radius=10,
        overwrite=True,
    )

    assert "masks_cellpose_boundaries_voronoi" in [*sdata_multi_c_no_backed.shapes]
    assert isinstance(sdata_multi_c_no_backed.shapes["masks_cellpose_boundaries_voronoi"], GeoDataFrame)
