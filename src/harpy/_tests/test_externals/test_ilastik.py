from __future__ import annotations

import dask.array as da
import numpy as np
import pytest

import harpy.externals.ilastik as ilastik
from harpy.externals.ilastik import _map_instance_ids_to_prediction_label


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


def test_map_instance_ids_to_prediction_label_computes_chunk_graph_once(monkeypatch: pytest.MonkeyPatch) -> None:
    segmentation = da.from_array(np.array([[1, 1, 2, 2], [3, 3, 4, 4]]), chunks=(1, 2))
    predictions = da.from_array(np.array([[5, 5, 6, 6], [7, 7, 8, 8]]), chunks=(1, 2))
    original_compute = ilastik.dask.compute
    compute_calls = 0

    def _counting_compute(*args: object, **kwargs: object) -> tuple[object, ...]:
        nonlocal compute_calls
        compute_calls += 1
        return original_compute(*args, **kwargs)

    monkeypatch.setattr(ilastik.dask, "compute", _counting_compute)

    result = ilastik._map_instance_ids_to_prediction_label(
        segmentation=segmentation,
        predictions=predictions,
        instance_ids=np.array([1, 2, 3, 4]),
        prediction_labels=np.array([5, 6, 7, 8]),
    )

    assert result == {1: 5, 2: 6, 3: 7, 4: 8}
    assert compute_calls == 1


def test_map_instance_ids_to_prediction_label_raises_for_shape_mismatch() -> None:
    segmentation = da.from_array(np.array([[1, 2], [3, 4]]), chunks=(1, 2))
    predictions = np.array([[1, 2, 3]])

    with pytest.raises(ValueError, match="Segmentation shape"):
        _map_instance_ids_to_prediction_label(segmentation=segmentation, predictions=predictions)
