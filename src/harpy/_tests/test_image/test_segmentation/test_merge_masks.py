from pandas import DataFrame
from spatialdata import SpatialData

from sparrow.image.segmentation._merge_masks import (
    mask_to_original,
    merge_labels_layers,
    merge_labels_layers_nuclei,
)


def test_merge_labels_layers(sdata_multi_c_no_backed: SpatialData):
    sdata_multi_c_no_backed = merge_labels_layers(
        sdata_multi_c_no_backed,
        labels_layer_1="masks_nuclear",
        labels_layer_2="masks_whole",
        output_labels_layer="masks_merged",
        output_shapes_layer=None,
        overwrite=True,
        chunks=256,
        depth=100,
    )

    assert "masks_merged" in sdata_multi_c_no_backed.labels
    assert isinstance(sdata_multi_c_no_backed, SpatialData)


def test_merge_labels_layers_nuclei(sdata_multi_c_no_backed: SpatialData):
    sdata_multi_c_no_backed = merge_labels_layers_nuclei(
        sdata_multi_c_no_backed,
        labels_layer="masks_whole",
        labels_layer_nuclei_expanded="masks_nuclear",
        labels_layer_nuclei="masks_nuclear",
        output_labels_layer="masks_merged_nuclear",
        output_shapes_layer=None,
        overwrite=True,
        chunks=256,
        depth=100,
    )

    assert "masks_merged_nuclear" in sdata_multi_c_no_backed.labels
    assert isinstance(sdata_multi_c_no_backed, SpatialData)


def test_mask_to_original(sdata_multi_c_no_backed: SpatialData):
    df = mask_to_original(
        sdata_multi_c_no_backed,
        labels_layer="masks_whole",
        original_labels_layers=["masks_nuclear"],
        depth=100,
        chunks=256,
    )

    assert df.shape == (674, 1)
    assert isinstance(df, DataFrame)
