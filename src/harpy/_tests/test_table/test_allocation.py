from copy import deepcopy
from types import SimpleNamespace

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from spatialdata import SpatialData, read_zarr
from spatialdata.models import Labels2DModel, Labels3DModel, PointsModel, TableModel
from spatialdata.transformations import Identity, Translation

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
    monkeypatch.setattr("harpy.table._allocation._DEPRECATED_ATTRIBUTES_WARNED", set())

    alias = aggregation_module.__getattr__("allocate")
    repeated_alias = aggregation_module.__getattr__("allocate")

    assert alias is repeated_alias is aggregate_points
    assert messages == ["`harpy.tb.allocate` is deprecated. Import and use `harpy.tb.aggregate_points` instead."]


@pytest.mark.parametrize("coordinate_columns", [("x", "y"), ("x", "y", "z")])
def test_pair_reductions_derives_coordinate_columns(coordinate_columns: tuple[str, ...]):
    reductions = aggregation_module._PairReductions(**_pair_reduction_inputs(coordinate_columns=coordinate_columns))

    assert reductions.coordinate_columns == coordinate_columns
    np.testing.assert_array_equal(reductions.instance_ids, [1, 2])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("coordinate_columns", "must use columns"),
        ("duplicate_centers", "instance IDs must be unique"),
        ("center_instances", "must use the same instances"),
        ("feature_index", "feature_counts must use a two-level MultiIndex"),
        ("feature_dtype", "feature_counts must use uint32"),
        ("feature_instance_name", "must match center index"),
    ],
)
def test_pair_reductions_rejects_structural_inconsistency(mutation: str, message: str):
    values = _pair_reduction_inputs()
    centers = values["centers"]
    feature_counts = values["feature_counts"]
    assert isinstance(centers, pd.DataFrame)
    assert isinstance(feature_counts, pd.Series)

    if mutation == "coordinate_columns":
        centers.columns = ["y", "x"]
    elif mutation == "duplicate_centers":
        centers.index = pd.Index([1, 1], name="cell_id")
    elif mutation == "center_instances":
        centers.index = pd.Index([1, 3], name="cell_id")
    elif mutation == "feature_index":
        feature_counts.index = pd.Index([1, 2], name="cell_id")
    elif mutation == "feature_dtype":
        values["feature_counts"] = feature_counts.astype(np.uint64)
    elif mutation == "feature_instance_name":
        feature_counts.index = feature_counts.index.set_names(["instance", "gene"])

    with pytest.raises(ValueError, match=message):
        aggregation_module._PairReductions(**values)


def test_counts_to_sparse_aligns_permuted_feature_levels_to_output_axis():
    counts = pd.Series(
        np.array([3, 7, 5], dtype=np.uint32),
        index=pd.MultiIndex(
            levels=[[1, 2], ["VIM", "EPCAM", "KRT8", "Unused"]],
            codes=[[0, 0, 1], [0, 1, 2]],
            names=["cell_id", "gene"],
        ),
    )

    matrix = aggregation_module._counts_to_sparse(
        counts,
        instance_ids=np.array([1, 2]),
        feature_axis=("EPCAM", "KRT8", "VIM"),
        cell_index_name="cell_id",
        feature_key="gene",
    )

    assert sparse.isspmatrix_csr(matrix)
    np.testing.assert_array_equal(
        matrix.toarray(),
        np.array(
            [
                [7, 0, 3],
                [0, 5, 0],
            ],
            dtype=np.uint32,
        ),
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
    assert sdata_transcripts["table_transcriptomics_recompute"].shape == (649, 96)

    assert np.array_equal(
        sdata_transcripts["table_transcriptomics_recompute"].X.toarray(),
        sdata_transcripts["table_transcriptomics"].X.toarray(),
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
    sdata_transcripts_no_backed: SpatialData,
    labels_name: list[str],
    points_name: str | list[str],
    message: str,
):
    with pytest.raises(ValueError, match=message):
        aggregate_points(
            sdata_transcripts_no_backed,
            labels_name=labels_name,
            points_name=points_name,
            output_table_name="new_table",
        )


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
    assert adata.var_names.to_list() == ["GeneA", "GeneB", "GeneZero"]
    assert adata.obs[_REGION_KEY].cat.categories.to_list() == ["labels_a", "labels_b"]
    assert adata.obs[_INSTANCE_KEY].to_list() == [1, 2, 3, 1]
    assert adata.obs["n_endogenous_points"].to_list() == [2, 1, 0, 1]
    assert adata.obs["n_negative_points"].to_list() == [1, 1, 1, 0]
    assert adata.obs["n_system_control_points"].to_list() == [1, 0, 0, 2]
    assert np.allclose(adata.obs["auxiliary_points_fraction"], [0.5, 0.5, 1.0, 2 / 3])
    assert np.array_equal(np.asarray(adata.X.sum(axis=1)).ravel(), adata.obs["n_endogenous_points"])
    assert not {"negative_points_per_feature", "system_control_points_per_feature"} & set(adata.obs)
    assert np.allclose(adata.obsm[_SPATIAL], [[0.5, 0.5], [3.5, 0.5], [0.5, 2.0], [0.5, 0.5]])

    auxiliary = adata.obsm["auxiliary_feature_counts"]
    assert sparse.isspmatrix_csr(auxiliary)
    assert auxiliary.dtype == np.dtype(np.uint32)
    assert np.array_equal(
        auxiliary.toarray(),
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


def test_ordinary_aggregation_uses_label_centers_without_class_metadata():
    result = aggregate_points(
        _class_aware_sdata(),
        labels_name="labels_a",
        points_name="points_a",
        to_coordinate_system="sample_a",
        output_table_name="table",
    )

    adata = result.tables["table"]
    assert adata.obs[_INSTANCE_KEY].to_list() == [1, 2, 3]
    assert np.allclose(adata.obsm[_SPATIAL], [[0.5, 0.5], [3.5, 0.5], [0.5, 2.0]])
    assert "auxiliary_feature_counts" not in adata.obsm
    assert "feature_class_aggregation" not in adata.uns
    assert "feature_matrices" not in adata.uns
    validate_table(result, "table")


def test_validate_table_accepts_a_registered_generic_feature_matrix():
    result = aggregate_points(
        _class_aware_sdata(),
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


def test_validate_table_accepts_filtered_and_reordered_feature_axes():
    result = aggregate_points(
        _class_aware_sdata(),
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


def test_validate_table_accepts_recalculated_summary_columns():
    result = aggregate_points(
        _class_aware_sdata(),
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


def test_validate_table_accepts_preprocessed_matrix_representations():
    result = aggregate_points(
        _class_aware_sdata(),
        labels_name="labels_a",
        points_name="points_a",
        to_coordinate_system="sample_a",
        output_table_name="table",
        expression_class="Endogenous",
    )
    adata = result.tables["table"]
    adata.X = adata.X.toarray().astype(np.float32)
    auxiliary = adata.obsm["auxiliary_feature_counts"]
    adata.obsm["auxiliary_feature_counts"] = auxiliary.toarray().astype(np.float64)

    assert set(adata.uns["feature_matrices"]["auxiliary_feature_counts"]) == {
        "schema_version",
        "source_kind",
        "feature_columns",
    }
    validate_table(result, "table")


def test_aggregate_points_uses_irregular_label_center_instead_of_point_position():
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

    result = aggregate_points(
        SpatialData(labels={"labels": labels}, points={"points": points}),
        labels_name="labels",
        points_name="points",
        to_coordinate_system="sample",
        output_table_name="table",
    )

    assert np.allclose(result.tables["table"].obsm[_SPATIAL], [[1 / 3, 1 / 3]])


def test_label_centers_apply_pair_translation():
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

    result = aggregate_points(
        sdata,
        labels_name="labels_a",
        points_name="points_a",
        to_coordinate_system="translated",
        output_table_name="table",
        expression_class="Endogenous",
    )

    assert np.allclose(result.tables["table"].obsm[_SPATIAL], [[10.5, 20.5], [13.5, 20.5], [10.5, 22.0]])


def test_aggregate_points_uses_xyz_label_center_order():
    labels = Labels3DModel.parse(
        np.ones((2, 2, 2), dtype=np.uint32),
        dims=("z", "y", "x"),
        transformations={"volume": Identity()},
    )
    points = PointsModel.parse(
        pd.DataFrame({"x": [0], "y": [0], "z": [0], "gene": ["GeneA"]}),
        transformations={"volume": Identity()},
    )

    result = aggregate_points(
        SpatialData(labels={"labels": labels}, points={"points": points}),
        labels_name="labels",
        points_name="points",
        to_coordinate_system="volume",
        output_table_name="table",
    )

    assert np.allclose(result.tables["table"].obsm[_SPATIAL], [[0.5, 0.5, 0.5]])


def test_class_aware_aggregation_assigns_points_once(monkeypatch: pytest.MonkeyPatch):
    calls = 0
    assign_points_to_labels = aggregation_module._assign_points_to_labels

    def wrapped_assign_points_to_labels(*args, **kwargs):
        nonlocal calls
        calls += 1
        return assign_points_to_labels(*args, **kwargs)

    monkeypatch.setattr(aggregation_module, "_assign_points_to_labels", wrapped_assign_points_to_labels)
    aggregate_points(
        _class_aware_sdata(),
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
def test_validate_table_rejects_class_aware_inconsistency(mutation: str):
    result = aggregate_points(
        _class_aware_sdata(),
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


def test_class_aware_aggregation_rejects_source_points_that_disagree_with_panel():
    sdata = _class_aware_sdata()
    points = sdata.points["points_a"].compute()
    points.loc[len(points)] = [4, 4, "Unknown", "Endogenous"]
    points["code_class"] = points["code_class"].astype(pd.CategoricalDtype(categories=_PANEL["classes"]))
    sdata.points["points_a"] = PointsModel.parse(points, transformations={"sample_a": Identity()})

    with pytest.raises(ValueError, match="feature 'Unknown' is absent from the panel"):
        aggregate_points(
            sdata,
            labels_name="labels_a",
            points_name="points_a",
            to_coordinate_system="sample_a",
            output_table_name="table",
            expression_class="Endogenous",
        )


def test_class_aware_aggregation_restores_unknown_dask_categories():
    sdata = _class_aware_sdata()
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

    assert result.tables["table"].obs["n_negative_points"].to_list() == [1, 1, 1]


def test_aggregation_preserves_custom_table_annotation_keys():
    result = aggregate_points(
        _class_aware_sdata(),
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


def _pair_reduction_inputs(*, coordinate_columns: tuple[str, ...] = ("x", "y")) -> dict[str, object]:
    centers = pd.DataFrame(
        np.arange(2 * len(coordinate_columns), dtype=float).reshape(2, len(coordinate_columns)),
        index=pd.Index([1, 2], name="cell_id"),
        columns=coordinate_columns,
    )
    feature_counts = pd.Series(
        np.array([2, 3], dtype=np.uint32),
        index=pd.MultiIndex.from_tuples(
            [(1, "GeneA"), (2, "GeneB")],
            names=["cell_id", "gene"],
        ),
    )
    return {
        "pair": aggregation_module._AggregationPair(
            labels_name="labels",
            points_name="points",
            coordinate_system="global",
        ),
        "centers": centers,
        "feature_counts": feature_counts,
    }


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
