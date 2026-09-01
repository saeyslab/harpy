from copy import deepcopy
from types import SimpleNamespace

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from spatialdata import SpatialData, read_zarr
from spatialdata.models import Labels2DModel, PointsModel, TableModel
from spatialdata.transformations import Identity

import harpy.table._allocation as aggregation_module
from harpy.table._allocation import aggregate_points, bin_counts
from harpy.utils._keys import _INSTANCE_KEY, _REGION_KEY, _SPATIAL

_PANEL = {
    "feature_key": "gene",
    "feature_class_key": "code_class",
    "classes": ["Endogenous", "Negative", "SystemControl"],
    "features_by_class": {
        "Endogenous": ["GeneA", "GeneB", "GeneZero"],
        "Negative": ["Negative01", "Negative02"],
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
    reductions = aggregation_module._PairReductions(
        **_pair_reduction_inputs(coordinate_columns=coordinate_columns)
    )

    assert reductions.coordinate_columns == coordinate_columns


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("coordinate_columns", "must use columns"),
        ("duplicate_coordinates", "instance IDs must be unique"),
        ("coordinate_instances", "must occur in feature_counts"),
        ("feature_index", "feature_counts must use a two-level MultiIndex"),
        ("feature_dtype", "feature_counts must use uint32"),
        ("feature_instance_name", "must match coordinate index"),
        ("class_index", "class_counts must use a two-level MultiIndex"),
        ("class_dtype", "class_counts must use uint32"),
        ("class_instances", "must reference compatible instance levels"),
    ],
)
def test_pair_reductions_rejects_structural_inconsistency(mutation: str, message: str):
    values = _pair_reduction_inputs()
    coordinates = values["coordinates"]
    feature_counts = values["feature_counts"]
    class_counts = values["class_counts"]
    assert isinstance(coordinates, pd.DataFrame)
    assert isinstance(feature_counts, pd.Series)
    assert isinstance(class_counts, pd.Series)

    if mutation == "coordinate_columns":
        coordinates.columns = ["y", "x"]
    elif mutation == "duplicate_coordinates":
        coordinates.index = pd.Index([1, 1], name="cell_id")
    elif mutation == "coordinate_instances":
        coordinates.index = pd.Index([1, 3], name="cell_id")
    elif mutation == "feature_index":
        feature_counts.index = pd.Index([1, 2], name="cell_id")
    elif mutation == "feature_dtype":
        values["feature_counts"] = feature_counts.astype(np.uint64)
    elif mutation == "feature_instance_name":
        feature_counts.index = feature_counts.index.set_names(["instance", "gene"])
    elif mutation == "class_index":
        class_counts.index = pd.Index([1, 2], name="cell_id")
    elif mutation == "class_dtype":
        values["class_counts"] = class_counts.astype(np.uint64)
    else:
        class_counts.index = pd.MultiIndex.from_tuples(
            [(1, "Endogenous"), (3, "Endogenous")],
            names=["cell_id", "code_class"],
        )

    with pytest.raises(ValueError, match=message):
        aggregation_module._PairReductions(**values)


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


def test_class_aware_aggregation_uses_panel_axis_and_adds_control_summaries(tmp_path):
    sdata = _class_aware_sdata()

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
    assert adata.obs[_INSTANCE_KEY].to_list() == [1, 2, 1]
    assert adata.obs["n_endogenous_points"].to_list() == [2, 1, 1]
    assert adata.obs["n_negative_points"].to_list() == [1, 1, 0]
    assert adata.obs["n_system_control_points"].to_list() == [1, 0, 2]
    assert np.allclose(adata.obs["control_fraction"], [0.5, 0.5, 2 / 3])
    assert np.array_equal(np.asarray(adata.X.sum(axis=1)).ravel(), adata.obs["n_endogenous_points"])
    assert not {"negative_points_per_feature", "system_control_points_per_feature"} & set(adata.obs)
    assert np.allclose(adata.obsm[_SPATIAL], [[0.5, 0.5], [3.0, 0.0], [0.0, 0.0]])

    metadata = adata.uns["feature_class_aggregation"]
    assert metadata["source_kind"] == "harpy_aggregate_points"
    assert metadata["control_class_denominators"] == {"Negative": 2, "SystemControl": 1}
    assert metadata["count_columns"] == {
        "Endogenous": "n_endogenous_points",
        "Negative": "n_negative_points",
        "SystemControl": "n_system_control_points",
    }
    assert metadata["regions"] == {
        "labels_a": {"points_element": "points_a", "coordinate_system": "sample_a"},
        "labels_b": {"points_element": "points_b", "coordinate_system": "sample_b"},
    }

    output = tmp_path / "class-aware.zarr"
    sdata.write(output)
    roundtripped = read_zarr(output).tables["table"]
    roundtripped_metadata = roundtripped.uns["feature_class_aggregation"]
    assert roundtripped_metadata["expression_class"] == "Endogenous"
    assert roundtripped_metadata["count_columns"] == metadata["count_columns"]
    assert roundtripped_metadata["control_class_denominators"] == metadata["control_class_denominators"]
    assert roundtripped_metadata["regions"] == metadata["regions"]


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

    assert result.tables["table"].obs["n_negative_points"].to_list() == [1, 1]


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
    coordinates = pd.DataFrame(
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
    class_counts = pd.Series(
        np.array([2, 3], dtype=np.uint32),
        index=pd.MultiIndex.from_tuples(
            [(1, "Endogenous"), (2, "Endogenous")],
            names=["cell_id", "code_class"],
        ),
    )
    return {
        "pair": aggregation_module._AggregationPair(
            labels_name="labels",
            points_name="points",
            coordinate_system="global",
        ),
        "coordinates": coordinates,
        "feature_counts": feature_counts,
        "class_counts": class_counts,
    }


def _class_aware_sdata() -> SpatialData:
    labels_a = Labels2DModel.parse(
        np.array(
            [
                [1, 1, 0, 2, 2],
                [1, 1, 0, 2, 2],
                [0, 0, 0, 0, 0],
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
            "x": [0, 1, 1, 0, 3, 4],
            "y": [0, 1, 0, 1, 0, 1],
            "gene": ["GeneA", "GeneA", "Negative01", "SystemControl1", "GeneB", "Negative02"],
            "code_class": pd.Series(
                ["Endogenous", "Endogenous", "Negative", "SystemControl", "Endogenous", "Negative"],
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
