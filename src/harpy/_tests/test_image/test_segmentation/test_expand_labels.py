from spatialdata import SpatialData

from harpy.image.segmentation._expand_masks import expand_labels


def test_expand_labels(sdata_multi_c_no_backed: SpatialData):
    sdata_multi_c_no_backed = expand_labels(
        sdata_multi_c_no_backed,
        labels_name="masks_cellpose_3D",
        distance=10,
        depth=100,
        chunks=256,
        output_labels_name="masks_cellpose_3D_expanded",
        output_shapes_name="masks_cellpose_3D_expanded_boundaries",
        overwrite=True,
    )

    assert "masks_cellpose_3D_expanded" in sdata_multi_c_no_backed.labels
    assert "masks_cellpose_3D_expanded_boundaries" in sdata_multi_c_no_backed.shapes

    assert isinstance(sdata_multi_c_no_backed, SpatialData)

    sdata_multi_c_no_backed = expand_labels(
        sdata_multi_c_no_backed,
        labels_name="masks_nuclear",
        distance=10,
        depth=100,
        chunks=256,
        output_labels_name="masks_nuclear_expanded",
        output_shapes_name="masks_nuclear_expanded_boundaries",
        overwrite=True,
    )

    assert "masks_nuclear_expanded" in sdata_multi_c_no_backed.labels
    assert "masks_nuclear_expanded_boundaries" in sdata_multi_c_no_backed.shapes

    assert isinstance(sdata_multi_c_no_backed, SpatialData)
