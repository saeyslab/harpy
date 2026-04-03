import dask.array as da
import numpy as np
from pandas import DataFrame
from spatialdata import SpatialData

from harpy.image._image import add_labels_layer
from harpy.image.segmentation._merge_masks import (
    _map_mask_ids_to_original_labels,
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


def test_map_mask_ids_to_original_labels_dask_matches_numpy() -> None:
    mask = np.array(
        [
            [0, 1, 1, 2],
            [3, 3, 2, 2],
            [4, 4, 4, 2],
        ]
    )
    original = np.array(
        [
            [9, 5, 5, 8],
            [1, 1, 8, 7],
            [2, 2, 3, 7],
        ]
    )

    result = _map_mask_ids_to_original_labels(
        mask=da.from_array(mask, chunks=(2, 2)),
        original=original,
    )

    assert result == {1: 5, 2: 7, 3: 1, 4: 2}


def test_map_mask_ids_to_original_labels_ignores_background_only_overlap() -> None:
    mask = da.from_array(np.array([[1, 1, 2, 2]]), chunks=(1, 2))
    original = da.from_array(np.array([[0, 0, 5, 5]]), chunks=(1, 2))

    result = _map_mask_ids_to_original_labels(mask=mask, original=original)

    assert result == {2: 5}


def test_map_mask_ids_to_original_labels_breaks_ties_by_smallest_label() -> None:
    mask = da.from_array(np.array([[1, 1, 1, 1]]), chunks=(1, 2))
    original = da.from_array(np.array([[7, 2, 2, 7]]), chunks=(1, 2))

    result = _map_mask_ids_to_original_labels(mask=mask, original=original)

    assert result == {1: 2}


def test_mask_to_original_small_sdata() -> None:
    sdata = SpatialData()
    mask = da.from_array(np.array([[1, 1, 2, 2], [1, 0, 2, 3]], dtype=np.uint32), chunks=(1, 2))
    original_1 = da.from_array(np.array([[5, 5, 7, 7], [0, 0, 8, 0]], dtype=np.uint32), chunks=(1, 2))
    original_2 = da.from_array(np.array([[10, 10, 0, 0], [10, 0, 0, 20]], dtype=np.uint32), chunks=(1, 2))

    sdata = add_labels_layer(sdata, arr=mask, output_layer="mask", overwrite=True)
    sdata = add_labels_layer(sdata, arr=original_1, output_layer="original_1", overwrite=True)
    sdata = add_labels_layer(sdata, arr=original_2, output_layer="original_2", overwrite=True)

    result = mask_to_original(
        sdata,
        labels_layer="mask",
        original_labels_layers=["original_1", "original_2"],
        chunks=2,
    )

    expected = DataFrame(
        np.array([[5, 10], [7, 0], [0, 20]], dtype=np.uint32),
        index=["1", "2", "3"],
        columns=["original_1", "original_2"],
    )
    assert result.equals(expected)
