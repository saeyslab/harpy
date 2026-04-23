from spatialdata import SpatialData

from harpy.image.segmentation._align_masks import align_labels


def test_align_labels(sdata_multi_c_no_backed: SpatialData):
    sdata_multi_c_no_backed = align_labels(
        sdata_multi_c_no_backed,
        labels_name_1="masks_nuclear",
        labels_name_2="masks_whole",
        output_labels_name="masks_nuclear_aligned",
        output_shapes_name=None,
        overwrite=True,
        chunks=256,
        depth=100,
    )

    assert "masks_nuclear_aligned" in sdata_multi_c_no_backed.labels
    assert isinstance(sdata_multi_c_no_backed, SpatialData)
