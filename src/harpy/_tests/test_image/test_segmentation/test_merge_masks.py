import dask.array as da
import numpy as np
import pytest
from pandas import DataFrame
from spatialdata import SpatialData

from harpy.image._image import add_labels_layer
from harpy.image.segmentation._merge_masks import (
    _get_mask_ids_to_original_overlap_counts,
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
        chunks=256,
    )

    assert df.shape == (674, 1)
    assert isinstance(df, DataFrame)


def test_get_mask_ids_to_original_overlap_counts() -> None:
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

    result = _get_mask_ids_to_original_overlap_counts(
        mask=da.from_array(mask, chunks=(2, 2)),
        original=original,
    )

    assert result == {1: {5: 2}, 2: {7: 2, 8: 2}, 3: {1: 2}, 4: {2: 2, 3: 1}}


def test_get_mask_ids_to_original_overlap_counts_ignores_background_only_overlap() -> None:
    mask = da.from_array(np.array([[1, 1, 2, 2]]), chunks=(1, 2))
    original = da.from_array(np.array([[0, 0, 5, 5]]), chunks=(1, 2))

    result = _get_mask_ids_to_original_overlap_counts(mask=mask, original=original)

    assert result == {2: {5: 2}}


def test_get_mask_ids_to_original_overlap_counts_keeps_tied_candidates() -> None:
    mask = da.from_array(np.array([[1, 1, 1, 1]]), chunks=(1, 2))
    original = da.from_array(np.array([[7, 2, 2, 7]]), chunks=(1, 2))

    result = _get_mask_ids_to_original_overlap_counts(mask=mask, original=original)

    assert result == {1: {2: 2, 7: 2}}


def test_get_mask_ids_to_original_overlap_counts_uses_subset_filters() -> None:
    mask = da.from_array(np.array([[1, 1, 2, 2], [3, 3, 2, 2]]), chunks=(1, 2))
    original = da.from_array(np.array([[5, 5, 7, 7], [9, 9, 7, 8]]), chunks=(1, 2))

    result = _get_mask_ids_to_original_overlap_counts(
        mask=mask,
        original=original,
        mask_ids=np.array([2, 99]),
        original_ids=np.array([7, 8]),
    )

    assert result == {2: {7: 3, 8: 1}}


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


def test_mask_to_original_small_sdata_supports_overlap_metrics() -> None:
    sdata = SpatialData()
    mask = da.from_array(np.array([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=np.uint32), chunks=(1, 2))
    original = da.from_array(np.array([[5, 5, 5, 5], [0, 0, 5, 0]], dtype=np.uint32), chunks=(1, 2))

    # For mask label 1 and original label 5:
    # - mask area = 3
    # - original area = 5
    # - non-background overlap = 2
    # The pixel where mask == 1 and original == 0 contributes to the mask area
    # but not to the overlap. This gives:
    # - mask_fraction = 2 / 3 = 0.666...
    # - original_fraction = 2 / 5 = 0.4
    # - iou = 2 / (3 + 5 - 2) = 2 / 6 = 0.333...
    # With threshold = 0.5, only mask_fraction keeps label 5.
    sdata = add_labels_layer(sdata, arr=mask, output_layer="mask", overwrite=True)
    sdata = add_labels_layer(sdata, arr=original, output_layer="original", overwrite=True)

    result_mask_fraction = mask_to_original(
        sdata,
        labels_layer="mask",
        original_labels_layers=["original"],
        chunks=2,
        threshold=0.5,
        overlap_metric="mask_fraction",
    )
    result_original_fraction = mask_to_original(
        sdata,
        labels_layer="mask",
        original_labels_layers=["original"],
        chunks=2,
        threshold=0.5,
        overlap_metric="original_fraction",
    )
    result_iou = mask_to_original(
        sdata,
        labels_layer="mask",
        original_labels_layers=["original"],
        chunks=2,
        threshold=0.5,
        overlap_metric="iou",
    )

    assert result_mask_fraction.equals(DataFrame(np.array([[5]], dtype=np.uint32), index=["1"], columns=["original"]))
    assert result_original_fraction.equals(
        DataFrame(np.array([[0]], dtype=np.uint32), index=["1"], columns=["original"])
    )
    assert result_iou.equals(DataFrame(np.array([[0]], dtype=np.uint32), index=["1"], columns=["original"]))


def test_mask_to_original_selects_winner_using_overlap_metric() -> None:
    sdata = SpatialData()
    mask = da.from_array(
        np.array(
            [
                [1, 1, 1, 0],
                [1, 1, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.uint32,
        ),
        chunks=(2, 2),
    )
    original = da.from_array(
        np.array(
            [
                [5, 5, 5, 5],
                [5, 6, 6, 5],
                [5, 5, 5, 5],
                [5, 5, 0, 0],
            ],
            dtype=np.uint32,
        ),
        chunks=(2, 2),
    )

    sdata = add_labels_layer(sdata, arr=mask, output_layer="mask", overwrite=True)
    sdata = add_labels_layer(sdata, arr=original, output_layer="original", overwrite=True)

    # Mask label 1 covers 6 pixels. Within that mask:
    # - original label 5 overlaps in 4 pixels
    # - original label 6 overlaps in 2 pixels
    #
    # mask_fraction picks label 5:
    # - mask_fraction(5) = 4 / 6
    # - mask_fraction(6) = 2 / 6
    #
    # But original label 5 is much larger overall:
    # - area_original(5) = 12
    # - area_original(6) = 2
    #
    # Therefore the winner flips for the other metrics:
    # - original_fraction(5) = 4 / 12
    # - original_fraction(6) = 2 / 2
    # - iou(5) = 4 / (6 + 12 - 4)
    # - iou(6) = 2 / (6 + 2 - 2)
    #
    # So mask_fraction keeps label 5, while original_fraction and iou select
    # label 6.
    result_mask_fraction = mask_to_original(
        sdata,
        labels_layer="mask",
        original_labels_layers=["original"],
        chunks=2,
        overlap_metric="mask_fraction",
    )
    result_original_fraction = mask_to_original(
        sdata,
        labels_layer="mask",
        original_labels_layers=["original"],
        chunks=2,
        overlap_metric="original_fraction",
    )
    result_iou = mask_to_original(
        sdata,
        labels_layer="mask",
        original_labels_layers=["original"],
        chunks=2,
        overlap_metric="iou",
    )

    assert result_mask_fraction.equals(DataFrame(np.array([[5]], dtype=np.uint32), index=["1"], columns=["original"]))
    assert result_original_fraction.equals(
        DataFrame(np.array([[6]], dtype=np.uint32), index=["1"], columns=["original"])
    )
    assert result_iou.equals(DataFrame(np.array([[6]], dtype=np.uint32), index=["1"], columns=["original"]))


def test_mask_to_original_raises_for_invalid_threshold() -> None:
    sdata = SpatialData()
    mask = da.from_array(np.array([[1, 1]], dtype=np.uint32), chunks=(1, 2))
    original = da.from_array(np.array([[5, 5]], dtype=np.uint32), chunks=(1, 2))

    sdata = add_labels_layer(sdata, arr=mask, output_layer="mask", overwrite=True)
    sdata = add_labels_layer(sdata, arr=original, output_layer="original", overwrite=True)

    with pytest.raises(ValueError, match="threshold"):
        mask_to_original(
            sdata,
            labels_layer="mask",
            original_labels_layers=["original"],
            threshold=1.5,
        )


def test_mask_to_original_raises_for_invalid_overlap_metric() -> None:
    sdata = SpatialData()
    mask = da.from_array(np.array([[1, 1]], dtype=np.uint32), chunks=(1, 2))
    original = da.from_array(np.array([[5, 5]], dtype=np.uint32), chunks=(1, 2))

    sdata = add_labels_layer(sdata, arr=mask, output_layer="mask", overwrite=True)
    sdata = add_labels_layer(sdata, arr=original, output_layer="original", overwrite=True)

    with pytest.raises(ValueError, match="overlap_metric"):
        mask_to_original(
            sdata,
            labels_layer="mask",
            original_labels_layers=["original"],
            overlap_metric="dice",  # type: ignore[arg-type]
        )
