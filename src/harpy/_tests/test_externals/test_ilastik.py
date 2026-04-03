from __future__ import annotations

import dask.array as da
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from harpy.externals.ilastik import _map_instance_ids_to_prediction_label, _map_instance_predictions_to_obs


def test_map_instance_ids_to_prediction_label_numpy() -> None:
    segmentation = np.array(
        [
            [0, 1, 1, 2],
            [3, 3, 2, 2],
        ]
    )
    predictions = np.array(
        [
            [9, 5, 5, 8],
            [1, 1, 8, 7],
        ]
    )

    result = _map_instance_ids_to_prediction_label(segmentation=segmentation, predictions=predictions)

    assert result == {1: 5, 2: 8, 3: 1}


def test_map_instance_ids_to_prediction_label_dask_matches_numpy() -> None:
    segmentation = np.array(
        [
            [0, 1, 1, 2],
            [3, 3, 2, 2],
            [4, 4, 4, 2],
        ]
    )
    predictions = np.array(
        [
            [9, 5, 5, 8],
            [1, 1, 8, 7],
            [2, 2, 3, 7],
        ]
    )

    result = _map_instance_ids_to_prediction_label(
        segmentation=da.from_array(segmentation, chunks=(2, 2)),
        predictions=predictions,
    )

    assert result == {1: 5, 2: 7, 3: 1, 4: 2}


def test_map_instance_ids_to_prediction_label_breaks_ties_by_smallest_label() -> None:
    segmentation = da.from_array(np.array([[1, 1, 1, 1]]), chunks=(1, 2))
    predictions = da.from_array(np.array([[3, 2, 2, 3]]), chunks=(1, 2))

    result = _map_instance_ids_to_prediction_label(segmentation=segmentation, predictions=predictions)

    assert result == {1: 2}


def test_map_instance_ids_to_prediction_label_uses_precomputed_global_indices() -> None:
    segmentation = da.from_array(np.array([[1, 2, 2], [3, 3, 0]]), chunks=(1, 2))
    predictions = da.from_array(np.array([[4, 5, 5], [7, 7, 9]]), chunks=(1, 2))

    result = _map_instance_ids_to_prediction_label(
        segmentation=segmentation,
        predictions=predictions,
        instance_ids=np.array([99, 3, 2]),
        prediction_labels=np.array([9, 7, 5, 4]),
    )

    assert result == {2: 5, 3: 7}


def test_map_instance_ids_to_prediction_label_raises_for_shape_mismatch() -> None:
    segmentation = da.from_array(np.array([[1, 2], [3, 4]]), chunks=(1, 2))
    predictions = np.array([[1, 2, 3]])

    with pytest.raises(ValueError, match="Segmentation shape"):
        _map_instance_ids_to_prediction_label(segmentation=segmentation, predictions=predictions)


def test_map_instance_predictions_to_obs_uses_custom_unmapped_label() -> None:
    adata = AnnData(obs=pd.DataFrame({"cell_id": np.array([1, 2, 3], dtype=np.int64)}))

    result = _map_instance_predictions_to_obs(
        adata=adata,
        instance_key="cell_id",
        instance_to_prediction={1: 5, 3: 7},
        unmapped_label=-1,
    )

    assert result.astype("int64").tolist() == [5, -1, 7]
    assert str(result.dtype) == "category"


def test_map_instance_predictions_to_obs_raises_for_unmapped_instance() -> None:
    adata = AnnData(obs=pd.DataFrame({"cell_id": np.array([1, 2, 3], dtype=np.int64)}))

    with pytest.raises(ValueError, match="\\[2\\]") as exc_info:
        _map_instance_predictions_to_obs(
            adata=adata,
            instance_key="cell_id",
            instance_to_prediction={1: 5, 3: 7},
            raise_on_unmapped_instance=True,
        )

    assert "not in the segmentation mask" in str(exc_info.value)
