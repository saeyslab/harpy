from copy import deepcopy
from types import SimpleNamespace

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from anndata.abc import CSRDataset
from scipy import sparse
from spatialdata import SpatialData, read_zarr
from spatialdata._io.format import SpatialDataContainerFormatV01
from spatialdata.models import Labels2DModel, Labels3DModel, PointsModel, TableModel
from spatialdata.transformations import Identity, Translation

import harpy.table._aggregation_checkpoint as checkpoint_module
import harpy.table._aggregation_writer as writer_module
import harpy.table._allocation as aggregation_module
from harpy.table import validate_table
from harpy.table._allocation import aggregate_points, bin_counts
from harpy.utils._keys import _INSTANCE_KEY, _REGION_KEY, _SPATIAL

_PANEL = {
    "feature_key": "gene",
    "feature_class_key": "code_class",
    "classes": ["Endogenous", "Negative", "SystemControl"],
    "features_by_class": {
        "Endogenous": ["GeneA", "GeneB", "GeneZero"],
        "Negative": ["Negative01", "Negative02", "NegativeZero"],
        "SystemControl": ["SystemControl1"],
    },
}


def test_allocate_import_warns_and_resolves_to_aggregate_points(monkeypatch: pytest.MonkeyPatch):
    messages: list[str] = []
    monkeypatch.setattr("harpy.table._allocation.log", SimpleNamespace(warning=messages.append))
    monkeypatch.setattr("harpy.table._allocation._WARNED_DEPRECATED_ATTRIBUTES", set())

    alias = aggregation_module.__getattr__("allocate")
    repeated_alias = aggregation_module.__getattr__("allocate")

    assert alias is repeated_alias is aggregate_points
    assert messages == ["`harpy.tb.allocate` is deprecated. Import and use `harpy.tb.aggregate_points` instead."]


def test_aggregation_checkpoint_merges_partial_counts_and_records_manifests(tmp_path):
    assigned = dd.from_pandas(
        pd.DataFrame(
            {
                "cells": [42, 42, 51, 42],
                "gene": ["EPCAM", "EPCAM", "VIM", "EPCAM"],
            }
        ),
        npartitions=2,
    )
    local = checkpoint_module._local_feature_counts(
        assigned,
        pair_ordinal=0,
        instance_key="cells",
        feature_key="gene",
    )
    checkpoint = checkpoint_module._stage_aggregation_checkpoint(
        [local],
        path=tmp_path / "counts",
        pairs=(checkpoint_module._CheckpointPair(0, "labels", "points", "global", ("x", "y")),),
        discover_features=True,
    )

    result = dd.read_parquet(checkpoint.path).compute().sort_values(["instance_id", "feature"])
    assert result.to_dict("records") == [
        {"aggregation_pair": 0, "instance_id": 42, "feature": "EPCAM", "count": 3},
        {"aggregation_pair": 0, "instance_id": 51, "feature": "VIM", "count": 1},
    ]
    assert result.dtypes[["aggregation_pair", "instance_id", "count"]].to_dict() == {
        "aggregation_pair": np.dtype(np.int64),
        "instance_id": np.dtype(np.uint64),
        "count": np.dtype(np.uint64),
    }
    assert pd.api.types.is_string_dtype(result["feature"].dtype)
    assert checkpoint.observed_features == ("EPCAM", "VIM")
    assert sorted(identity for part in checkpoint.partitions for identity in part.identities) == [(0, 42), (0, 51)]
    row_stops = np.cumsum([len(part.identities) for part in checkpoint.partitions])
    assert [part.row_start for part in checkpoint.partitions] == [0, *row_stops[:-1]]
    assert [part.row_stop for part in checkpoint.partitions] == row_stops.tolist()


def test_checkpoint_partition_to_csr_aligns_the_requested_feature_axis(tmp_path):
    path = tmp_path / "part.parquet"
    pd.DataFrame(
        {
            "aggregation_pair": np.array([0, 0, 0], dtype=np.int64),
            "instance_id": np.array([42, 42, 51], dtype=np.uint64),
            "feature": pd.Series(["VIM", "EPCAM", "KRT8"], dtype="string"),
            "count": np.array([3, 7, 5], dtype=np.uint64),
        }
    ).to_parquet(path, index=False)
    partition = checkpoint_module._CheckpointPartition(
        ordinal=0,
        path=path,
        identities=((0, 42), (0, 51)),
        row_count=3,
    )

    matrix = writer_module._checkpoint_partition_to_csr(
        partition,
        feature_axis=("EPCAM", "KRT8", "VIM"),
        feature_axis_hash=writer_module._feature_axis_hash(("EPCAM", "KRT8", "VIM")),
    )

    assert sparse.isspmatrix_csr(matrix)
    assert matrix.indices.dtype == np.dtype(np.int64)
    assert matrix.indptr.dtype == np.dtype(np.int64)
    np.testing.assert_array_equal(matrix.toarray(), [[7, 0, 3], [0, 5, 0]])


def test_aggregation_checkpoint_executes_each_source_partition_once(tmp_path):
    reads: list[int] = []

    @dask.delayed
    def read_partition(ordinal: int) -> pd.DataFrame:
        reads.append(ordinal)
        return pd.DataFrame(
            {
                "cells": pd.Series([ordinal + 1], dtype=np.uint32),
                "gene": pd.Series([f"Gene{ordinal}"], dtype="string"),
            }
        )

    assigned = dd.from_delayed(
        [read_partition(0), read_partition(1)],
        meta=pd.DataFrame({"cells": pd.Series(dtype=np.uint32), "gene": pd.Series(dtype="string")}),
    )
    local = checkpoint_module._local_feature_counts(
        assigned,
        pair_ordinal=0,
        instance_key="cells",
        feature_key="gene",
    )

    checkpoint_module._stage_aggregation_checkpoint(
        [local],
        path=tmp_path / "counts",
        pairs=(checkpoint_module._CheckpointPair(0, "labels", "points", "global", ("x", "y")),),
        discover_features=True,
    )

    assert sorted(reads) == [0, 1]


def test_aggregation_checkpoint_rejects_a_pair_without_assigned_instances(tmp_path):
    populated = dd.from_pandas(pd.DataFrame({"cells": [1], "gene": ["GeneA"]}), npartitions=1)
    empty = dd.from_pandas(
        pd.DataFrame({"cells": pd.Series(dtype=np.uint32), "gene": pd.Series(dtype="string")}),
        npartitions=1,
    )
    partial_counts = [
        checkpoint_module._local_feature_counts(
            points,
            pair_ordinal=ordinal,
            instance_key="cells",
            feature_key="gene",
        )
        for ordinal, points in enumerate((populated, empty))
    ]

    with pytest.raises(ValueError, match=r"labels_empty.*points_empty"):
        checkpoint_module._stage_aggregation_checkpoint(
            partial_counts,
            path=tmp_path / "counts",
            pairs=(
                checkpoint_module._CheckpointPair(0, "labels", "points", "global", ("x", "y")),
                checkpoint_module._CheckpointPair(1, "labels_empty", "points_empty", "global", ("x", "y")),
            ),
            discover_features=True,
        )


def test_checkpoint_partition_to_csr_rejects_uint32_overflow(tmp_path):
    path = tmp_path / "part.parquet"
    pd.DataFrame(
        {
            "aggregation_pair": np.array([0], dtype=np.int64),
            "instance_id": np.array([1], dtype=np.uint64),
            "feature": pd.Series(["GeneA"], dtype="string"),
            "count": np.array([np.iinfo(np.uint32).max + 1], dtype=np.uint64),
        }
    ).to_parquet(path, index=False)
    partition = checkpoint_module._CheckpointPartition(
        ordinal=0,
        path=path,
        identities=((0, 1),),
        row_count=1,
    )

    with pytest.raises(ValueError, match="exceed the uint32"):
        writer_module._checkpoint_partition_to_csr(
            partition,
            feature_axis=("GeneA",),
            feature_axis_hash=writer_module._feature_axis_hash(("GeneA",)),
        )


def test_aggregate_points(sdata_transcripts: SpatialData):
    assert sdata_transcripts.is_backed()

    sdata_transcripts = aggregate_points(
        sdata_transcripts,
        labels_name="segmentation_mask",
        output_table_name="table_transcriptomics_recompute",
        chunks=1000,
        overwrite=True,
    )

    assert "table_transcriptomics_recompute" in [*sdata_transcripts.tables]
    recomputed = sdata_transcripts["table_transcriptomics_recompute"]
    expected = sdata_transcripts["table_transcriptomics"]
    assert recomputed.shape == (649, 96)
    recomputed_order = np.argsort(recomputed.obs[_INSTANCE_KEY].to_numpy())
    expected_order = np.argsort(expected.obs[_INSTANCE_KEY].to_numpy())

    assert np.array_equal(
        _to_memory(recomputed.X)[recomputed_order].toarray(),
        expected.X[expected_order].toarray(),
    )


def test_aggregate_points_multiple_pairs_create_one_table(sdata_transcripts: SpatialData):
    assert sdata_transcripts.is_backed()

    sdata_transcripts = aggregate_points(
        sdata_transcripts,
        labels_name=["segmentation_mask", "segmentation_mask_expanded"],
        output_table_name="table_transcriptomics",
        chunks=20000,
        overwrite=True,
    )

    assert "table_transcriptomics" in [*sdata_transcripts.tables]
    # The multi-region ordinary path uses the union of observed features;
    # unlike the removed append path, it does not silently inner-join them.
    assert sdata_transcripts["table_transcriptomics"].shape == (1302, 98)
    assert sdata_transcripts["table_transcriptomics"].obs[_REGION_KEY].cat.categories.to_list() == [
        "segmentation_mask",
        "segmentation_mask_expanded",
    ]


@pytest.mark.parametrize(
    ("labels_name", "points_name", "message"),
    [
        (["segmentation_mask", "segmentation_mask"], "transcripts", "Duplicate labels"),
        (
            ["segmentation_mask", "segmentation_mask_expanded"],
            ["transcripts", "transcripts", "transcripts"],
            "length 1",
        ),
    ],
)
def test_aggregate_points_rejects_invalid_pairs(
    sdata_transcripts: SpatialData,
    labels_name: list[str],
    points_name: str | list[str],
    message: str,
):
    with pytest.raises(ValueError, match=message):
        aggregate_points(
            sdata_transcripts,
            labels_name=labels_name,
            points_name=points_name,
            output_table_name="new_table",
        )


def test_aggregate_points_rejects_unbacked_input_before_pair_validation(sdata_transcripts_no_backed: SpatialData):
    with pytest.raises(ValueError, match="requires a SpatialData object backed") as error:
        aggregate_points(
            sdata_transcripts_no_backed,
            labels_name=["missing", "missing"],
            output_table_name="new_table",
        )
    assert 'sdata.write("sdata.zarr")' in str(error.value)


def test_aggregate_points_overwrite(sdata_transcripts: SpatialData):
    with pytest.raises(
        ValueError,
        match="already exists.*overwrite=True",
    ):
        sdata_transcripts = aggregate_points(
            sdata_transcripts,
            labels_name="segmentation_mask",
            output_table_name="table_transcriptomics",
            chunks=20000,
            overwrite=False,
        )


def test_aggregation_write_failure_preserves_existing_table_and_cleans_workspace(monkeypatch, tmp_path):
    sdata = _backed(_class_aware_sdata(), tmp_path)
    sdata = aggregate_points(
        sdata,
        labels_name="labels_a",
        points_name="points_a",
        to_coordinate_system="sample_a",
        output_table_name="table",
    )
    expected = sdata.tables["table"].X.to_memory()

    def fail(*args, **kwargs):
        raise RuntimeError("injected metadata failure")

    monkeypatch.setattr(writer_module, "_set_spatialdata_table_attrs", fail)
    with pytest.raises(RuntimeError, match="injected metadata failure"):
        aggregate_points(
            sdata,
            labels_name="labels_a",
            points_name="points_a",
            to_coordinate_system="sample_a",
            output_table_name="table",
            overwrite=True,
        )

    reopened = read_zarr(sdata.path)
    assert (reopened.tables["table"].X != expected).nnz == 0
    assert not list((tmp_path / "input.zarr" / "tables").glob(".harpy-aggregate-*"))


def test_class_aware_aggregation_uses_panel_axis_and_adds_auxiliary_summaries(tmp_path):
    sdata = _class_aware_sdata()
    output = tmp_path / "class-aware.zarr"
    sdata.write(output)
    sdata = read_zarr(output)
    assert sdata.is_backed()

    sdata = aggregate_points(
        sdata,
        labels_name=["labels_a", "labels_b"],
        points_name=["points_a", "points_b"],
        to_coordinate_system=["sample_a", "sample_b"],
        output_table_name="table",
        expression_class="Endogenous",
    )

    adata = sdata.tables["table"]
    assert isinstance(adata.X, CSRDataset)
    assert adata.var_names.to_list() == ["GeneA", "GeneB", "GeneZero"]
    assert adata.obs[_REGION_KEY].cat.categories.to_list() == ["labels_a", "labels_b"]
    order = sorted(
        range(adata.n_obs),
        key=lambda row: (
            adata.obs[_REGION_KEY].cat.categories.get_loc(adata.obs[_REGION_KEY].iloc[row]),
            adata.obs[_INSTANCE_KEY].iloc[row],
        ),
    )
    obs = adata.obs.iloc[order]
    expression = adata.X.to_memory()[order]
    assert obs[_INSTANCE_KEY].to_list() == [1, 2, 3, 1]
    assert obs["n_endogenous_points"].to_list() == [2, 1, 0, 1]
    assert obs["n_negative_points"].to_list() == [1, 1, 1, 0]
    assert obs["n_system_control_points"].to_list() == [1, 0, 0, 2]
    assert np.allclose(obs["auxiliary_points_fraction"], [0.5, 0.5, 1.0, 2 / 3])
    assert np.array_equal(np.asarray(expression.sum(axis=1)).ravel(), obs["n_endogenous_points"])
    assert not {"negative_points_per_feature", "system_control_points_per_feature"} & set(adata.obs)
    assert np.allclose(
        adata.obsm[_SPATIAL][order],
        [[0.5, 0.5], [3.5, 0.5], [0.5, 2.0], [0.5, 0.5]],
    )

    auxiliary = adata.obsm["auxiliary_feature_counts"]
    assert isinstance(auxiliary, CSRDataset)
    assert auxiliary.dtype == np.dtype(np.uint32)
    assert np.array_equal(
        auxiliary.to_memory()[order].toarray(),
        np.array(
            [
                [1, 0, 0, 1],
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 2],
            ],
            dtype=np.uint32,
        ),
    )
    feature_matrix_metadata = adata.uns["feature_matrices"]["auxiliary_feature_counts"]
    assert {key: value for key, value in feature_matrix_metadata.items() if key != "feature_columns"} == {
        "schema_version": 1,
        "source_kind": "harpy_aggregate_points",
    }
    assert list(feature_matrix_metadata["feature_columns"]) == [
        "Negative01",
        "Negative02",
        "NegativeZero",
        "SystemControl1",
    ]

    metadata = adata.uns["feature_class_aggregation"]
    assert metadata["source_kind"] == "harpy_aggregate_points"
    assert metadata["auxiliary_class_feature_counts"] == {"Negative": 3, "SystemControl": 1}
    assert metadata["auxiliary_points_fraction_column"] == "auxiliary_points_fraction"
    assert metadata["auxiliary_feature_matrix_key"] == "auxiliary_feature_counts"
    assert metadata["count_columns"] == {
        "Endogenous": "n_endogenous_points",
        "Negative": "n_negative_points",
        "SystemControl": "n_system_control_points",
    }
    assert metadata["regions"] == {
        "labels_a": {"points_element": "points_a", "coordinate_system": "sample_a"},
        "labels_b": {"points_element": "points_b", "coordinate_system": "sample_b"},
    }

    roundtripped_sdata = read_zarr(output)
    roundtripped = roundtripped_sdata.tables["table"]
    roundtripped_metadata = roundtripped.uns["feature_class_aggregation"]
    assert roundtripped_metadata["expression_class"] == "Endogenous"
    assert roundtripped_metadata["count_columns"] == metadata["count_columns"]
    assert roundtripped_metadata["auxiliary_class_feature_counts"] == metadata["auxiliary_class_feature_counts"]
    assert roundtripped_metadata["regions"] == metadata["regions"]
    assert sparse.isspmatrix_csr(roundtripped.obsm["auxiliary_feature_counts"])
    assert roundtripped.obsm["auxiliary_feature_counts"].dtype == np.dtype(np.uint32)
    assert list(roundtripped.uns["feature_matrices"]["auxiliary_feature_counts"]["feature_columns"]) == [
        "Negative01",
        "Negative02",
        "NegativeZero",
        "SystemControl1",
    ]
    validate_table(roundtripped_sdata, "table")


def test_aggregate_points_writes_a_reopenable_zarr_v2_table(tmp_path):
    path = tmp_path / "zarr-v2.zarr"
    _class_aware_sdata().write(path, sdata_formats=SpatialDataContainerFormatV01())
    sdata = aggregate_points(
        read_zarr(path),
        labels_name="labels_a",
        points_name="points_a",
        to_coordinate_system="sample_a",
        output_table_name="table",
        expression_class="Endogenous",
    )

    assert isinstance(sdata.tables["table"].X, CSRDataset)
    reopened = read_zarr(path)
    assert reopened.tables["table"].shape == (3, 3)
    assert reopened.tables["table"].uns[TableModel.ATTRS_KEY] == {
        "region": ["labels_a"],
        "region_key": _REGION_KEY,
        "instance_key": _INSTANCE_KEY,
    }


def test_aggregate_points_publishes_without_spatialdata_write_element(monkeypatch, tmp_path):
    sdata = _backed(_class_aware_sdata(), tmp_path)

    def fail(*args, **kwargs):
        raise AssertionError("SpatialData.write_element() must not write the assembled table.")

    monkeypatch.setattr(sdata, "write_element", fail)
    result = aggregate_points(
        sdata,
        labels_name="labels_a",
        points_name="points_a",
        to_coordinate_system="sample_a",
        output_table_name="table",
    )

    assert isinstance(result.tables["table"].X, CSRDataset)


def test_ordinary_aggregation_uses_label_centers_without_class_metadata(tmp_path):
    result = aggregate_points(
        _backed(_class_aware_sdata(), tmp_path),
        labels_name="labels_a",
        points_name="points_a",
        to_coordinate_system="sample_a",
        output_table_name="table",
    )

    adata = result.tables["table"]
    order = np.argsort(adata.obs[_INSTANCE_KEY].to_numpy())
    assert adata.obs[_INSTANCE_KEY].iloc[order].to_list() == [1, 2, 3]
    assert np.allclose(adata.obsm[_SPATIAL][order], [[0.5, 0.5], [3.5, 0.5], [0.5, 2.0]])
    assert "auxiliary_feature_counts" not in adata.obsm
    assert "feature_class_aggregation" not in adata.uns
    assert "feature_matrices" not in adata.uns
    validate_table(result, "table")


def test_validate_table_accepts_a_registered_generic_feature_matrix(tmp_path):
    result = aggregate_points(
        _backed(_class_aware_sdata(), tmp_path),
        labels_name="labels_a",
        points_name="points_a",
        to_coordinate_system="sample_a",
        output_table_name="table",
    )
    adata = result.tables["table"]
    adata.obsm["derived_features"] = np.ones((adata.n_obs, 2), dtype=np.float64)
    adata.uns["feature_matrices"] = {
        "derived_features": {
            "schema_version": 1,
            "source_kind": "test_features",
            "backend": "numpy",
            "dtype": "float64",
            "feature_columns": ["first", "second"],
        }
    }

    validate_table(result, "table")


def test_validate_table_accepts_filtered_and_reordered_feature_axes(tmp_path):
    result = aggregate_points(
        _backed(_class_aware_sdata(), tmp_path),
        labels_name="labels_a",
        points_name="points_a",
        to_coordinate_system="sample_a",
        output_table_name="table",
        expression_class="Endogenous",
    )
    adata = result.tables["table"][:, ["GeneZero", "GeneB"]].copy()
    adata.obsm["auxiliary_feature_counts"] = adata.obsm["auxiliary_feature_counts"][:, [3, 0]]
    adata.uns["feature_matrices"]["auxiliary_feature_counts"]["feature_columns"] = [
        "SystemControl1",
        "Negative01",
    ]
    result.tables["table"] = adata

    # The summaries describe the original complete aggregation and therefore
    # need not equal row sums after legitimate feature filtering.
    assert not np.array_equal(
        np.asarray(adata.X.sum(axis=1)).ravel(),
        adata.obs["n_endogenous_points"].to_numpy(),
    )
    validate_table(result, "table")


def test_validate_table_accepts_recalculated_summary_columns(tmp_path):
    result = aggregate_points(
        _backed(_class_aware_sdata(), tmp_path),
        labels_name="labels_a",
        points_name="points_a",
        to_coordinate_system="sample_a",
        output_table_name="table",
        expression_class="Endogenous",
    )
    adata = result.tables["table"]
    adata.obs["n_endogenous_points"] = adata.obs["n_endogenous_points"].astype(np.float64) / 2
    adata.obs["auxiliary_points_fraction"] = np.linspace(0.1, 0.9, adata.n_obs)

    validate_table(result, "table")


def test_validate_table_accepts_preprocessed_matrix_representations(tmp_path):
    result = aggregate_points(
        _backed(_class_aware_sdata(), tmp_path),
        labels_name="labels_a",
        points_name="points_a",
        to_coordinate_system="sample_a",
        output_table_name="table",
        expression_class="Endogenous",
    )
    adata = result.tables["table"]
    adata.X = _to_memory(adata.X).toarray().astype(np.float32)
    auxiliary = adata.obsm["auxiliary_feature_counts"]
    adata.obsm["auxiliary_feature_counts"] = _to_memory(auxiliary).toarray().astype(np.float64)

    assert set(adata.uns["feature_matrices"]["auxiliary_feature_counts"]) == {
        "schema_version",
        "source_kind",
        "feature_columns",
    }
    validate_table(result, "table")


def test_aggregate_points_uses_irregular_label_center_instead_of_point_position(tmp_path):
    labels = Labels2DModel.parse(
        np.array(
            [
                [1, 1, 0],
                [1, 0, 0],
                [0, 0, 0],
            ],
            dtype=np.uint32,
        ),
        dims=("y", "x"),
        transformations={"sample": Identity()},
    )
    points = PointsModel.parse(
        pd.DataFrame({"x": [1], "y": [0], "gene": ["GeneA"]}),
        transformations={"sample": Identity()},
    )

    sdata = _backed(SpatialData(labels={"labels": labels}, points={"points": points}), tmp_path)
    result = aggregate_points(
        sdata,
        labels_name="labels",
        points_name="points",
        to_coordinate_system="sample",
        output_table_name="table",
    )

    assert np.allclose(result.tables["table"].obsm[_SPATIAL], [[1 / 3, 1 / 3]])


def test_label_centers_apply_pair_translation(tmp_path):
    sdata = _class_aware_sdata()
    labels = sdata.labels["labels_a"]
    sdata.labels["labels_a"] = Labels2DModel.parse(
        labels.data,
        dims=("y", "x"),
        transformations={"translated": Translation([10, 20], axes=("x", "y"))},
    )
    points = sdata.points["points_a"].compute()
    points["x"] += 10
    points["y"] += 20
    sdata.points["points_a"] = PointsModel.parse(
        dd.from_pandas(points, npartitions=2),
        transformations={"translated": Identity()},
    )
    sdata = _backed(sdata, tmp_path)

    result = aggregate_points(
        sdata,
        labels_name="labels_a",
        points_name="points_a",
        to_coordinate_system="translated",
        output_table_name="table",
        expression_class="Endogenous",
    )

    assert np.allclose(result.tables["table"].obsm[_SPATIAL], [[10.5, 20.5], [13.5, 20.5], [10.5, 22.0]])


def test_aggregate_points_uses_xyz_label_center_order(tmp_path):
    labels = Labels3DModel.parse(
        np.ones((2, 2, 2), dtype=np.uint32),
        dims=("z", "y", "x"),
        transformations={"volume": Identity()},
    )
    points = PointsModel.parse(
        pd.DataFrame({"x": [0], "y": [0], "z": [0], "gene": ["GeneA"]}),
        transformations={"volume": Identity()},
    )

    sdata = _backed(SpatialData(labels={"labels": labels}, points={"points": points}), tmp_path)
    result = aggregate_points(
        sdata,
        labels_name="labels",
        points_name="points",
        to_coordinate_system="volume",
        output_table_name="table",
    )

    assert np.allclose(result.tables["table"].obsm[_SPATIAL], [[0.5, 0.5, 0.5]])


def test_assign_points_to_labels_routes_once_across_irregular_chunks():
    labels_data = np.arange(1, 36, dtype=np.uint32).reshape(5, 7)
    labels_data[1, 1] = 0
    labels = Labels2DModel.parse(
        da.from_array(labels_data, chunks=((2, 3), (3, 2, 2))),
        dims=("y", "x"),
        transformations={"sample": Identity()},
    )
    categories = pd.CategoricalDtype(categories=["A", "B", "C"])
    points = PointsModel.parse(
        dd.from_pandas(
            pd.DataFrame(
                {
                    "point_id": np.arange(10),
                    "x": [0, 3, 5, 7, -1, 1, 0.5, 1.5, 4, 6],
                    "y": [0, 0, 2, 0, 0, 1, 0.5, 1.5, 4, 4],
                    "feature": pd.Series(["A", "B", "C", "A", "B", "C", "A", "B", "C", "A"], dtype=categories),
                }
            ),
            npartitions=3,
        ),
        transformations={"sample": Identity()},
    )

    assigned = aggregation_module._assign_points_to_labels(
        labels,
        points,
        value_key=["point_id", "feature"],
        chunks=None,
        to_coordinate_system="sample",
    )
    result = assigned.compute().sort_values("point_id").reset_index(drop=True)

    assert assigned.npartitions == 6
    assert result.columns.to_list() == ["x", "y", "point_id", "feature", _INSTANCE_KEY]
    assert result["point_id"].to_list() == [0, 1, 2, 6, 7, 8, 9]
    assert result[_INSTANCE_KEY].to_list() == [1, 4, 20, 1, 17, 33, 35]
    assert result[["x", "y"]].to_numpy().tolist() == [[0, 0], [3, 0], [5, 2], [0, 0], [2, 2], [4, 4], [6, 4]]
    assert result["feature"].dtype == categories
    assert not any("block_id" in column for column in result.columns)


def test_classify_points_by_label_block_uses_half_open_irregular_grid():
    feature_dtype = pd.CategoricalDtype(categories=["A", "B", "C"])
    partition = pd.DataFrame(
        {
            "point_id": np.arange(7),
            "x": [10, 12.5, 13, 15, 17, 10, 16],
            "y": [20, 21.5, 20, 24, 20, 19, 21],
            "feature": pd.Series(["A", "B", "C", "A", "B", "C", "A"], dtype=feature_dtype),
        }
    )

    result = aggregation_module._classify_points_by_label_block(
        partition,
        coordinate_keys=("y", "x"),
        boundaries=((0, 2, 5), (0, 3, 5, 7)),
        translations=(20, 10),
        grid_shape=(2, 3),
        block_id_key="block_id",
    )

    assert result["point_id"].to_list() == [0, 1, 2, 3, 6]
    assert result[["x", "y"]].to_numpy().tolist() == [[10, 20], [12, 22], [13, 20], [15, 24], [16, 21]]
    assert result["block_id"].to_list() == [0, 3, 1, 5, 2]
    assert result["block_id"].dtype == np.dtype(np.int64)
    assert result["feature"].dtype == feature_dtype


def test_classify_points_by_label_block_preserves_empty_schema():
    partition = pd.DataFrame(
        {
            "x": pd.Series([-1, 4], dtype=np.float64),
            "y": pd.Series([0, 0], dtype=np.float64),
            "feature": pd.Series(["A", "B"], dtype="string"),
        }
    )

    result = aggregation_module._classify_points_by_label_block(
        partition,
        coordinate_keys=("y", "x"),
        boundaries=((0, 2), (0, 4)),
        translations=(0, 0),
        grid_shape=(1, 1),
        block_id_key="block_id",
    )

    assert result.empty
    assert result.columns.to_list() == ["x", "y", "feature", "block_id"]
    assert result.dtypes.to_dict() == {
        "x": np.dtype(np.int64),
        "y": np.dtype(np.int64),
        "feature": pd.StringDtype(),
        "block_id": np.dtype(np.int64),
    }


def test_assign_points_to_labels_applies_integer_translation():
    labels = Labels2DModel.parse(
        da.from_array(np.array([[1, 2], [3, 4]], dtype=np.uint32), chunks=(1, 1)),
        dims=("y", "x"),
        transformations={"translated": Translation([10, 20], axes=("x", "y"))},
    )
    points = PointsModel.parse(
        pd.DataFrame({"x": [10, 11], "y": [20, 21], "feature": ["A", "B"]}),
        transformations={"translated": Identity()},
    )

    result = aggregation_module._assign_points_to_labels(
        labels,
        points,
        value_key="feature",
        to_coordinate_system="translated",
    ).compute()

    assert result.sort_values("feature")[_INSTANCE_KEY].to_list() == [1, 4]


def test_assign_points_to_labels_routes_three_dimensional_blocks():
    labels_data = np.arange(1, 25, dtype=np.uint32).reshape(2, 3, 4)
    labels = Labels3DModel.parse(
        da.from_array(labels_data, chunks=(1, 2, 2)),
        dims=("z", "y", "x"),
        transformations={"volume": Identity()},
    )
    points = PointsModel.parse(
        pd.DataFrame(
            {
                "point_id": [0, 1, 2],
                "x": [0, 2, 3],
                "y": [0, 2, 2],
                "z": [0, 0, 1],
            }
        ),
        transformations={"volume": Identity()},
    )

    assigned = aggregation_module._assign_points_to_labels(
        labels,
        points,
        value_key="point_id",
        to_coordinate_system="volume",
    )
    result = assigned.compute().sort_values("point_id")

    assert assigned.npartitions == 8
    assert result[_INSTANCE_KEY].to_list() == [1, 11, 24]


def test_assign_points_to_labels_rejects_fractional_translation():
    labels = Labels2DModel.parse(
        np.ones((2, 2), dtype=np.uint32),
        dims=("y", "x"),
        transformations={"translated": Translation([0.25, 0], axes=("x", "y"))},
    )
    points = PointsModel.parse(
        pd.DataFrame({"x": [0], "y": [0], "feature": ["A"]}),
        transformations={"translated": Identity()},
    )

    with pytest.raises(ValueError, match="translation along 'x' must be pixel-aligned"):
        aggregation_module._assign_points_to_labels(
            labels,
            points,
            value_key="feature",
            to_coordinate_system="translated",
        )


@pytest.mark.parametrize("labels_ndim", [2, 3])
def test_assign_points_to_labels_rejects_dimension_mismatch(labels_ndim: int):
    if labels_ndim == 2:
        labels = Labels2DModel.parse(
            np.ones((2, 2), dtype=np.uint32),
            dims=("y", "x"),
            transformations={"sample": Identity()},
        )
        frame = pd.DataFrame({"x": [0], "y": [0], "z": [0], "feature": ["A"]})
        message = "Two-dimensional labels require only"
    else:
        labels = Labels3DModel.parse(
            np.ones((2, 2, 2), dtype=np.uint32),
            dims=("z", "y", "x"),
            transformations={"sample": Identity()},
        )
        frame = pd.DataFrame({"x": [0], "y": [0], "feature": ["A"]})
        message = "Three-dimensional labels require"
    points = PointsModel.parse(frame, transformations={"sample": Identity()})

    with pytest.raises(ValueError, match=message):
        aggregation_module._assign_points_to_labels(
            labels,
            points,
            value_key="feature",
            to_coordinate_system="sample",
        )


def test_assign_points_to_labels_graph_construction_does_not_read_sources():
    reads: list[str] = []

    @dask.delayed
    def read_labels():
        reads.append("labels")
        return np.array([[1, 0], [0, 2]], dtype=np.uint32)

    @dask.delayed
    def read_points():
        reads.append("points")
        return pd.DataFrame(
            {
                "x": [0, 1],
                "y": [0, 1],
                "feature": pd.Series(["A", "B"], dtype="string"),
            }
        )

    labels = Labels2DModel.parse(
        da.from_delayed(read_labels(), shape=(2, 2), dtype=np.uint32),
        dims=("y", "x"),
        transformations={"sample": Identity()},
    )
    points = dd.from_delayed(
        [read_points()],
        meta=pd.DataFrame(
            {
                "x": pd.Series(dtype=np.int64),
                "y": pd.Series(dtype=np.int64),
                "feature": pd.Series(dtype="string"),
            }
        ),
    )
    points.attrs["transform"] = {"sample": Identity()}

    assigned = aggregation_module._assign_points_to_labels(
        labels,
        points,
        value_key="feature",
        to_coordinate_system="sample",
    )

    assert reads == []
    assert assigned.compute()[_INSTANCE_KEY].to_list() == [1, 2]
    assert sorted(reads) == ["labels", "points"]


def test_class_aware_aggregation_assigns_points_once(monkeypatch: pytest.MonkeyPatch, tmp_path):
    calls = 0
    assign_points_to_labels = aggregation_module._assign_points_to_labels

    def wrapped_assign_points_to_labels(*args, **kwargs):
        nonlocal calls
        calls += 1
        return assign_points_to_labels(*args, **kwargs)

    monkeypatch.setattr(aggregation_module, "_assign_points_to_labels", wrapped_assign_points_to_labels)
    aggregate_points(
        _backed(_class_aware_sdata(), tmp_path),
        labels_name="labels_a",
        points_name="points_a",
        to_coordinate_system="sample_a",
        output_table_name="table",
        expression_class="Endogenous",
    )

    assert calls == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "columns",
        "pointer",
        "shape",
        "missing_summary",
        "panel",
        "expression_class",
        "auxiliary_class",
    ],
)
def test_validate_table_rejects_class_aware_inconsistency(mutation: str, tmp_path):
    result = aggregate_points(
        _backed(_class_aware_sdata(), tmp_path),
        labels_name="labels_a",
        points_name="points_a",
        to_coordinate_system="sample_a",
        output_table_name="table",
        expression_class="Endogenous",
    )
    adata = result.tables["table"]
    if mutation == "columns":
        adata.uns["feature_matrices"]["auxiliary_feature_counts"]["feature_columns"] = ["Negative01"]
    elif mutation == "pointer":
        adata.uns["feature_class_aggregation"]["auxiliary_feature_matrix_key"] = "missing"
    elif mutation == "shape":
        adata.obsm["auxiliary_feature_counts"] = adata.obsm["auxiliary_feature_counts"][:, :-1]
    elif mutation == "missing_summary":
        adata.obs.drop(columns="n_negative_points", inplace=True)
    elif mutation == "panel":
        result.attrs["harpy"]["feature_panels"]["feature_panel_test"]["features_by_class"]["Negative"].append(
            "NegativeAddedLater"
        )
    elif mutation == "expression_class":
        adata.var_names = pd.Index(["Negative01", "GeneB", "GeneZero"], name="gene")
    else:
        adata.uns["feature_matrices"]["auxiliary_feature_counts"]["feature_columns"] = [
            "GeneA",
            "Negative02",
            "NegativeZero",
            "SystemControl1",
        ]

    with pytest.raises(ValueError):
        validate_table(result, "table")


def test_class_aware_contract_requires_an_auxiliary_class():
    panel = deepcopy(_PANEL)
    panel["classes"] = ["Endogenous"]
    panel["features_by_class"] = {"Endogenous": panel["features_by_class"]["Endogenous"]}

    with pytest.raises(ValueError, match="at least one non-expression"):
        aggregation_module._FeatureClassAggregationContract(
            panel=aggregation_module._parse_feature_panel(panel, panel_name="expression_only"),
            expression_class="Endogenous",
        )


def test_class_aware_aggregation_rejects_source_points_that_disagree_with_panel(tmp_path):
    sdata = _class_aware_sdata()
    points = sdata.points["points_a"].compute()
    points.loc[len(points)] = [4, 4, "Unknown", "Endogenous"]
    points["code_class"] = points["code_class"].astype(pd.CategoricalDtype(categories=_PANEL["classes"]))
    sdata.points["points_a"] = PointsModel.parse(points, transformations={"sample_a": Identity()})
    sdata = _backed(sdata, tmp_path)

    with pytest.raises(ValueError, match="feature 'Unknown' is absent from the panel"):
        aggregate_points(
            sdata,
            labels_name="labels_a",
            points_name="points_a",
            to_coordinate_system="sample_a",
            output_table_name="table",
            expression_class="Endogenous",
        )


def test_class_aware_aggregation_restores_unknown_dask_categories(tmp_path):
    sdata = _backed(_class_aware_sdata(), tmp_path)
    points = sdata.points["points_a"]
    points = points.assign(code_class=points["code_class"].cat.as_unknown())
    sdata.points["points_a"] = PointsModel.parse(points, transformations={"sample_a": Identity()})

    result = aggregate_points(
        sdata,
        labels_name="labels_a",
        points_name="points_a",
        to_coordinate_system="sample_a",
        output_table_name="table",
        expression_class="Endogenous",
    )

    assert sorted(result.tables["table"].obs["n_negative_points"].to_list()) == [1, 1, 1]


def test_aggregation_preserves_custom_table_annotation_keys(tmp_path):
    result = aggregate_points(
        _backed(_class_aware_sdata(), tmp_path),
        labels_name="labels_a",
        points_name="points_a",
        to_coordinate_system="sample_a",
        output_table_name="table",
        region_key="region",
        instance_key="instance",
    )

    attrs = result.tables["table"].uns[TableModel.ATTRS_KEY]
    assert attrs[TableModel.REGION_KEY_KEY] == "region"
    assert attrs[TableModel.INSTANCE_KEY] == "instance"
    assert result.tables["table"].obs.index.name == "instance_index"


def test_aggregation_accepts_an_explicit_table_index_name(tmp_path):
    result = aggregate_points(
        _backed(_class_aware_sdata(), tmp_path),
        labels_name="labels_a",
        points_name="points_a",
        to_coordinate_system="sample_a",
        output_table_name="table",
        table_index_name="observations",
    )

    assert result.tables["table"].obs.index.name == "observations"
    assert read_zarr(result.path).tables["table"].obs.index.name == "observations"


@pytest.mark.parametrize("table_index_name", ["", _INSTANCE_KEY, _REGION_KEY, "auxiliary_points_fraction"])
def test_aggregation_rejects_an_invalid_table_index_name(tmp_path, table_index_name: str):
    with pytest.raises(ValueError, match="table_index_name"):
        aggregate_points(
            _backed(_class_aware_sdata(), tmp_path),
            labels_name="labels_a",
            points_name="points_a",
            to_coordinate_system="sample_a",
            output_table_name="table",
            table_index_name=table_index_name,
        )


def test_class_aware_aggregation_rejects_a_table_index_name_colliding_with_a_summary(tmp_path):
    with pytest.raises(ValueError, match="Generated feature-class columns collide"):
        aggregate_points(
            _backed(_class_aware_sdata(), tmp_path),
            labels_name="labels_a",
            points_name="points_a",
            to_coordinate_system="sample_a",
            output_table_name="table",
            expression_class="Endogenous",
            table_index_name="n_negative_points",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_reference", "feature_panel"),
        ("missing_panel", "must be a mapping"),
        ("non_categorical", "must be categorical"),
        ("incompatible", "compatible panels"),
    ],
)
def test_class_aware_aggregation_rejects_invalid_panel_contract_before_assignment(
    mutation: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    sdata = _class_aware_sdata()
    if mutation == "missing_reference":
        del sdata.attrs["harpy"]["points"]["points_a"]["feature_panel"]
    elif mutation == "missing_panel":
        sdata.attrs["harpy"]["points"]["points_a"]["feature_panel"] = "missing"
    elif mutation == "non_categorical":
        points = sdata.points["points_a"].compute()
        points["code_class"] = points["code_class"].astype(str)
        sdata.points["points_a"] = PointsModel.parse(points, transformations={"sample_a": Identity()})
    else:
        incompatible = deepcopy(_PANEL)
        incompatible["features_by_class"]["Endogenous"].append("GeneC")
        sdata.attrs["harpy"]["feature_panels"]["feature_panel_other"] = incompatible
        sdata.attrs["harpy"]["points"]["points_b"]["feature_panel"] = "feature_panel_other"
    sdata = _backed(sdata, tmp_path)

    def fail(*args, **kwargs):
        raise AssertionError("Spatial assignment started before panel validation completed.")

    monkeypatch.setattr("harpy.table._allocation._assign_points_to_labels", fail)
    with pytest.raises(ValueError, match=message):
        aggregate_points(
            sdata,
            labels_name=["labels_a", "labels_b"] if mutation == "incompatible" else "labels_a",
            points_name=["points_a", "points_b"] if mutation == "incompatible" else "points_a",
            to_coordinate_system=["sample_a", "sample_b"] if mutation == "incompatible" else "sample_a",
            output_table_name="table",
            expression_class="Endogenous",
        )


def _backed(sdata: SpatialData, tmp_path, *, name: str = "input.zarr") -> SpatialData:
    path = tmp_path / name
    sdata.write(path)
    return read_zarr(path)


def _to_memory(matrix):
    return matrix.to_memory() if isinstance(matrix, CSRDataset) else matrix


def _class_aware_sdata() -> SpatialData:
    labels_a = Labels2DModel.parse(
        np.array(
            [
                [1, 1, 0, 2, 2],
                [1, 1, 0, 2, 2],
                [3, 3, 0, 4, 4],
            ],
            dtype=np.uint32,
        ),
        dims=("y", "x"),
        transformations={"sample_a": Identity()},
    )
    labels_b = Labels2DModel.parse(
        np.array([[1, 1], [1, 1]], dtype=np.uint32),
        dims=("y", "x"),
        transformations={"sample_b": Identity()},
    )
    categories = pd.CategoricalDtype(categories=_PANEL["classes"])
    points_a = pd.DataFrame(
        {
            "x": [0, 1, 1, 0, 3, 4, 0],
            "y": [0, 1, 0, 1, 0, 1, 2],
            "gene": [
                "GeneA",
                "GeneA",
                "Negative01",
                "SystemControl1",
                "GeneB",
                "Negative02",
                "Negative01",
            ],
            "code_class": pd.Series(
                [
                    "Endogenous",
                    "Endogenous",
                    "Negative",
                    "SystemControl",
                    "Endogenous",
                    "Negative",
                    "Negative",
                ],
                dtype=categories,
            ),
        }
    )
    points_b = pd.DataFrame(
        {
            "x": [0, 1, 1],
            "y": [0, 0, 1],
            "gene": ["GeneB", "SystemControl1", "SystemControl1"],
            "code_class": pd.Series(["Endogenous", "SystemControl", "SystemControl"], dtype=categories),
        }
    )
    panel_name = "feature_panel_test"
    return SpatialData(
        labels={"labels_a": labels_a, "labels_b": labels_b},
        points={
            "points_a": PointsModel.parse(
                dd.from_pandas(points_a, npartitions=2),
                transformations={"sample_a": Identity()},
            ),
            "points_b": PointsModel.parse(
                dd.from_pandas(points_b, npartitions=1),
                transformations={"sample_b": Identity()},
            ),
        },
        attrs={
            "harpy": {
                "metadata_version": 1,
                "points": {
                    "points_a": {"feature_panel": panel_name},
                    "points_b": {"feature_panel": panel_name},
                },
                "feature_panels": {panel_name: deepcopy(_PANEL)},
            }
        },
    )


def test_bin_counts(
    sdata_bin,
):
    table_name_bins = "square_002um"
    labels_name = "square_labels_32"  # custom grid to bin the counts of table_name_bins, can be any segmentation mask.
    table_name = "table_custom_bin_32"
    output_table_name = f"{table_name}_reproduce"

    # check that barcodes are unique in table_name_bins of sdata_bin
    assert sdata_bin.tables[table_name_bins].obs.index.is_unique

    sdata_bin = bin_counts(
        sdata_bin,
        table_name=table_name_bins,
        labels_name=labels_name,
        output_table_name=output_table_name,
        overwrite=True,
        region_key=_REGION_KEY,
        instance_key=_INSTANCE_KEY,
        spatial_key=_SPATIAL,
        append=False,
    )

    assert np.array_equal(
        sdata_bin[table_name].obs[_INSTANCE_KEY].values, sdata_bin[output_table_name].obs[_INSTANCE_KEY].values
    )

    assert np.array_equal(sdata_bin[table_name].var_names, sdata_bin[output_table_name].var_names)

    matrix1 = sdata_bin[table_name].X
    matrix2 = sdata_bin[output_table_name].X

    assert (matrix1 != matrix2).nnz == 0

    assert np.array_equal(sdata_bin[table_name].obsm[_SPATIAL], sdata_bin[output_table_name].obsm[_SPATIAL])
