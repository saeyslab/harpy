import dask.array as da
import numpy as np
import pytest
from pandas import DataFrame
from spatialdata import SpatialData

from harpy.image._image import add_labels_layer, get_dataarray
from harpy.image.segmentation._merge_masks import (
    _get_source_ids_to_reference_overlap_counts,
    match_labels_to_reference_layers,
    merge_labels_layers,
    merge_labels_layers_nuclei,
)


def test_merge_labels_layers(sdata_multi_c_no_backed: SpatialData):
    sdata_multi_c_no_backed = merge_labels_layers(
        sdata_multi_c_no_backed,
        candidate_labels_name="masks_whole",
        priority_labels_name="masks_nuclear",
        output_labels_name="masks_merged",
        output_shapes_name=None,
        overwrite=True,
        chunks=256,
    )

    assert "masks_merged" in sdata_multi_c_no_backed.labels
    assert isinstance(sdata_multi_c_no_backed, SpatialData)


def test_merge_labels_layers_nuclei(sdata_multi_c_no_backed: SpatialData):
    sdata_multi_c_no_backed = merge_labels_layers_nuclei(
        sdata_multi_c_no_backed,
        labels_name="masks_whole",
        labels_name_nuclei_expanded="masks_nuclear",
        labels_name_nuclei="masks_nuclear",
        output_labels_name="masks_merged_nuclear",
        output_shapes_name=None,
        overwrite=True,
        chunks=256,
        depth=100,
    )

    assert "masks_merged_nuclear" in sdata_multi_c_no_backed.labels
    assert isinstance(sdata_multi_c_no_backed, SpatialData)


def test_merge_labels_layers_small_sdata_keeps_no_overlap_candidates() -> None:
    sdata = SpatialData()
    candidate = da.from_array(np.array([[1, 1, 2, 2], [1, 1, 2, 0]], dtype=np.uint32), chunks=(1, 2))
    priority = da.from_array(np.array([[5, 5, 0, 0], [5, 5, 0, 0]], dtype=np.uint32), chunks=(1, 2))

    sdata = add_labels_layer(sdata, arr=candidate, output_labels_name="candidate", overwrite=True)
    sdata = add_labels_layer(sdata, arr=priority, output_labels_name="priority", overwrite=True)

    sdata = merge_labels_layers(
        sdata,
        candidate_labels_name="candidate",
        priority_labels_name="priority",
        output_labels_name="merged",
        overwrite=True,
        chunks=2,
        threshold=0.5,
    )

    result = get_dataarray(sdata, layer="merged").data.compute()
    expected = np.array([[5, 5, 6, 6], [5, 5, 6, 0]], dtype=np.uint32)
    assert np.array_equal(result, expected)


def test_merge_labels_layers_uses_global_candidate_fraction_across_chunks() -> None:
    sdata = SpatialData()
    candidate = da.from_array(np.array([[1, 1, 1, 1, 2, 2]], dtype=np.uint32), chunks=(1, 2))
    priority = da.from_array(np.array([[5, 5, 0, 0, 0, 0]], dtype=np.uint32), chunks=(1, 2))

    sdata = add_labels_layer(sdata, arr=candidate, output_labels_name="candidate", overwrite=True)
    sdata = add_labels_layer(sdata, arr=priority, output_labels_name="priority", overwrite=True)

    sdata = merge_labels_layers(
        sdata,
        candidate_labels_name="candidate",
        priority_labels_name="priority",
        output_labels_name="merged",
        overwrite=True,
        chunks=2,
        threshold=0.4,
    )

    # Candidate label 1 spans two chunks and overlaps the priority layer in 2 of
    # its 4 pixels, so its global candidate_fraction is 2 / 4 = 0.5 and it is
    # rejected for threshold 0.4. Candidate label 2 has no overlap and is kept.
    result = get_dataarray(sdata, layer="merged").data.compute()
    expected = np.array([[5, 5, 0, 0, 6, 6]], dtype=np.uint32)
    assert np.array_equal(result, expected)


def test_match_labels_to_reference_layers(sdata_multi_c_no_backed: SpatialData):
    df = match_labels_to_reference_layers(
        sdata_multi_c_no_backed,
        source_labels_name="masks_whole",
        reference_labels_layers=["masks_nuclear"],
        chunks=256,
    )

    assert df.shape == (674, 1)
    assert isinstance(df, DataFrame)


def test_get_source_ids_to_reference_overlap_counts() -> None:
    source = np.array(
        [
            [0, 1, 1, 2],
            [3, 3, 2, 2],
            [4, 4, 4, 2],
        ]
    )
    reference = np.array(
        [
            [9, 5, 5, 8],
            [1, 1, 8, 7],
            [2, 2, 3, 7],
        ]
    )

    result = _get_source_ids_to_reference_overlap_counts(
        source=da.from_array(source, chunks=(2, 2)),
        reference=reference,
    )

    assert result == {1: {5: 2}, 2: {7: 2, 8: 2}, 3: {1: 2}, 4: {2: 2, 3: 1}}


def test_get_source_ids_to_reference_overlap_counts_ignores_background_only_overlap() -> None:
    source = da.from_array(np.array([[1, 1, 2, 2]]), chunks=(1, 2))
    reference = da.from_array(np.array([[0, 0, 5, 5]]), chunks=(1, 2))

    result = _get_source_ids_to_reference_overlap_counts(source=source, reference=reference)

    assert result == {2: {5: 2}}


def test_get_source_ids_to_reference_overlap_counts_keeps_tied_candidates() -> None:
    source = da.from_array(np.array([[1, 1, 1, 1]]), chunks=(1, 2))
    reference = da.from_array(np.array([[7, 2, 2, 7]]), chunks=(1, 2))

    result = _get_source_ids_to_reference_overlap_counts(source=source, reference=reference)

    assert result == {1: {2: 2, 7: 2}}


def test_get_source_ids_to_reference_overlap_counts_uses_subset_filters() -> None:
    source = da.from_array(np.array([[1, 1, 2, 2], [3, 3, 2, 2]]), chunks=(1, 2))
    reference = da.from_array(np.array([[5, 5, 7, 7], [9, 9, 7, 8]]), chunks=(1, 2))

    result = _get_source_ids_to_reference_overlap_counts(
        source=source,
        reference=reference,
        source_ids=np.array([2, 99]),
        reference_ids=np.array([7, 8]),
    )

    assert result == {2: {7: 3, 8: 1}}


def test_match_labels_to_reference_layers_small_sdata() -> None:
    sdata = SpatialData()
    mask = da.from_array(np.array([[1, 1, 2, 2], [1, 0, 2, 3]], dtype=np.uint32), chunks=(1, 2))
    original_1 = da.from_array(np.array([[5, 5, 7, 7], [0, 0, 8, 0]], dtype=np.uint32), chunks=(1, 2))
    original_2 = da.from_array(np.array([[10, 10, 0, 0], [10, 0, 0, 20]], dtype=np.uint32), chunks=(1, 2))

    sdata = add_labels_layer(sdata, arr=mask, output_labels_name="mask", overwrite=True)
    sdata = add_labels_layer(sdata, arr=original_1, output_labels_name="original_1", overwrite=True)
    sdata = add_labels_layer(sdata, arr=original_2, output_labels_name="original_2", overwrite=True)

    result = match_labels_to_reference_layers(
        sdata,
        source_labels_name="mask",
        reference_labels_layers=["original_1", "original_2"],
        chunks=2,
    )

    expected = DataFrame(
        np.array([[5, 10], [7, 0], [0, 20]], dtype=np.uint32),
        index=["1", "2", "3"],
        columns=["original_1", "original_2"],
    )
    assert result.equals(expected)


@pytest.mark.parametrize(
    ("overlap_metric", "threshold"),
    [
        ("source_fraction", 0.5),
        ("iou", 0.5),
    ],
)
def test_match_labels_to_reference_layers_empty_source_returns_empty_dataframe(
    overlap_metric: str, threshold: float
) -> None:
    sdata = SpatialData()
    source = da.from_array(np.zeros((2, 2), dtype=np.uint32), chunks=(1, 2))
    reference = da.from_array(np.array([[5, 5], [0, 0]], dtype=np.uint32), chunks=(1, 2))

    sdata = add_labels_layer(sdata, arr=source, output_labels_name="source", overwrite=True)
    sdata = add_labels_layer(sdata, arr=reference, output_labels_name="reference", overwrite=True)

    result = match_labels_to_reference_layers(
        sdata,
        source_labels_name="source",
        reference_labels_layers=["reference"],
        chunks=2,
        threshold=threshold,
        overlap_metric=overlap_metric,  # type: ignore[arg-type]
    )

    assert result.empty
    assert result.shape == (0, 1)
    assert list(result.columns) == ["reference"]


def test_match_labels_to_reference_layers_supports_overlap_metrics() -> None:
    sdata = SpatialData()
    mask = da.from_array(np.array([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=np.uint32), chunks=(1, 2))
    original = da.from_array(np.array([[5, 5, 5, 5], [0, 0, 5, 0]], dtype=np.uint32), chunks=(1, 2))

    # For mask label 1 and original label 5:
    # - mask area = 3
    # - original area = 5
    # - non-background overlap = 2
    # The pixel where mask == 1 and original == 0 contributes to the mask area
    # but not to the overlap. This gives:
    # - source_fraction = 2 / 3 = 0.666...
    # - reference_fraction = 2 / 5 = 0.4
    # - iou = 2 / (3 + 5 - 2) = 2 / 6 = 0.333...
    # With threshold = 0.5, only source_fraction keeps label 5.
    sdata = add_labels_layer(sdata, arr=mask, output_labels_name="mask", overwrite=True)
    sdata = add_labels_layer(sdata, arr=original, output_labels_name="original", overwrite=True)

    result_source_fraction = match_labels_to_reference_layers(
        sdata,
        source_labels_name="mask",
        reference_labels_layers=["original"],
        chunks=2,
        threshold=0.5,
        overlap_metric="source_fraction",
    )
    result_reference_fraction = match_labels_to_reference_layers(
        sdata,
        source_labels_name="mask",
        reference_labels_layers=["original"],
        chunks=2,
        threshold=0.5,
        overlap_metric="reference_fraction",
    )
    result_iou = match_labels_to_reference_layers(
        sdata,
        source_labels_name="mask",
        reference_labels_layers=["original"],
        chunks=2,
        threshold=0.5,
        overlap_metric="iou",
    )

    assert result_source_fraction.equals(DataFrame(np.array([[5]], dtype=np.uint32), index=["1"], columns=["original"]))
    assert result_reference_fraction.equals(
        DataFrame(np.array([[0]], dtype=np.uint32), index=["1"], columns=["original"])
    )
    assert result_iou.equals(DataFrame(np.array([[0]], dtype=np.uint32), index=["1"], columns=["original"]))


def test_match_labels_to_reference_layers_selects_winner_using_overlap_metric() -> None:
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

    sdata = add_labels_layer(sdata, arr=mask, output_labels_name="mask", overwrite=True)
    sdata = add_labels_layer(sdata, arr=original, output_labels_name="original", overwrite=True)

    # Mask label 1 covers 6 pixels. Within that mask:
    # - original label 5 overlaps in 4 pixels
    # - original label 6 overlaps in 2 pixels
    #
    # source_fraction picks label 5:
    # - source_fraction(5) = 4 / 6
    # - source_fraction(6) = 2 / 6
    #
    # But original label 5 is much larger overall:
    # - area_original(5) = 12
    # - area_original(6) = 2
    #
    # Therefore the winner flips for the other metrics:
    # - reference_fraction(5) = 4 / 12
    # - reference_fraction(6) = 2 / 2
    # - iou(5) = 4 / (6 + 12 - 4)
    # - iou(6) = 2 / (6 + 2 - 2)
    #
    # So source_fraction keeps label 5, while reference_fraction and iou select
    # label 6.
    result_source_fraction = match_labels_to_reference_layers(
        sdata,
        source_labels_name="mask",
        reference_labels_layers=["original"],
        chunks=2,
        overlap_metric="source_fraction",
    )
    result_reference_fraction = match_labels_to_reference_layers(
        sdata,
        source_labels_name="mask",
        reference_labels_layers=["original"],
        chunks=2,
        overlap_metric="reference_fraction",
    )
    result_iou = match_labels_to_reference_layers(
        sdata,
        source_labels_name="mask",
        reference_labels_layers=["original"],
        chunks=2,
        overlap_metric="iou",
    )

    assert result_source_fraction.equals(DataFrame(np.array([[5]], dtype=np.uint32), index=["1"], columns=["original"]))
    assert result_reference_fraction.equals(
        DataFrame(np.array([[6]], dtype=np.uint32), index=["1"], columns=["original"])
    )
    assert result_iou.equals(DataFrame(np.array([[6]], dtype=np.uint32), index=["1"], columns=["original"]))


def test_match_labels_to_reference_layers_raises_for_invalid_threshold() -> None:
    sdata = SpatialData()
    mask = da.from_array(np.array([[1, 1]], dtype=np.uint32), chunks=(1, 2))
    original = da.from_array(np.array([[5, 5]], dtype=np.uint32), chunks=(1, 2))

    sdata = add_labels_layer(sdata, arr=mask, output_labels_name="mask", overwrite=True)
    sdata = add_labels_layer(sdata, arr=original, output_labels_name="original", overwrite=True)

    with pytest.raises(ValueError, match="threshold"):
        match_labels_to_reference_layers(
            sdata,
            source_labels_name="mask",
            reference_labels_layers=["original"],
            threshold=1.5,
        )


def test_match_labels_to_reference_layers_raises_for_invalid_overlap_metric() -> None:
    sdata = SpatialData()
    mask = da.from_array(np.array([[1, 1]], dtype=np.uint32), chunks=(1, 2))
    original = da.from_array(np.array([[5, 5]], dtype=np.uint32), chunks=(1, 2))

    sdata = add_labels_layer(sdata, arr=mask, output_labels_name="mask", overwrite=True)
    sdata = add_labels_layer(sdata, arr=original, output_labels_name="original", overwrite=True)

    with pytest.raises(ValueError, match="overlap_metric"):
        match_labels_to_reference_layers(
            sdata,
            source_labels_name="mask",
            reference_labels_layers=["original"],
            overlap_metric="dice",  # type: ignore[arg-type]
        )
