from spatialdata import SpatialData

from harpy.image.segmentation._align_masks import align_labels_layers


def test_align_labels_layers(sdata_multi_c_no_backed: SpatialData):
    sdata_multi_c_no_backed = align_labels_layers(
        sdata_multi_c_no_backed,
        labels_layer_1="masks_nuclear",
        labels_layer_2="masks_whole",
        output_labels_layer="masks_nuclear_aligned",
        output_shapes_layer=None,
        overwrite=True,
        chunks=256,
        depth=100,
    )

    assert "masks_nuclear_aligned" in sdata_multi_c_no_backed.labels
    assert isinstance(sdata_multi_c_no_backed, SpatialData)
