"""hp.qc.summarize_table on dummy class-aware tables: in memory, lazy and backed."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse
from spatialdata import SpatialData
from spatialdata.models import TableModel

import harpy as hp
from harpy._tests.test_table.test_io.test_updates import _CountComputes
from harpy.qc import TableSummary, summarize_table

_COUNT_COLUMNS = {
    "Endogenous": "n_endogenous_points",
    "Negative": "n_negative_points",
    "SystemControl": "n_system_control_points",
}
_PANEL_FEATURE_COUNTS = {"Endogenous": 2, "Negative": 4, "SystemControl": 5}


def _table():
    """Two regions of three instances. cell_1 has no points at all; cell_4 has only SystemControl points."""
    counts = pd.DataFrame(
        {
            "n_endogenous_points": [10, 0, 5, 8, 0, 2],
            "n_negative_points": [1, 0, 0, 2, 0, 1],
            "n_system_control_points": [0, 0, 1, 0, 3, 0],
        },
        index=[f"cell_{i}" for i in range(6)],
    )
    obs = pd.DataFrame(
        {"region": pd.Categorical(["a"] * 3 + ["b"] * 3), "cell_ID": [1, 2, 3, 1, 2, 3]}, index=counts.index
    ).join(counts)
    auxiliary = counts["n_negative_points"] + counts["n_system_control_points"]
    obs["auxiliary_points_fraction"] = auxiliary / counts.sum(axis=1).replace(0, np.nan)
    adata = AnnData(
        X=sparse.csr_matrix(np.arange(12, dtype=np.float32).reshape(6, 2)),
        obs=obs,
        var=pd.DataFrame(index=["GeneA", "GeneB"]),
    )
    adata.uns["point_aggregation"] = {
        "schema_version": 1,
        "source_kind": "harpy_aggregate_points",
        "feature_key": "gene",
        "feature_class_key": "code_class",
        "expression_class": "Endogenous",
        "classes": list(_COUNT_COLUMNS),
        "panel_feature_counts": dict(_PANEL_FEATURE_COUNTS),
        "count_columns": dict(_COUNT_COLUMNS),
        "auxiliary_points_fraction_column": "auxiliary_points_fraction",
        "auxiliary_feature_matrix_key": "auxiliary_feature_counts",
        "regions": {
            "a": {"points_element": "points_a", "coordinate_system": "a"},
            "b": {"points_element": "points_b", "coordinate_system": "b"},
        },
    }
    return TableModel.parse(adata, region=["a", "b"], region_key="region", instance_key="cell_ID")


def _sdata(table=None):
    # Only the table: summarize_table needs neither the annotated labels nor the source points.
    return SpatialData(tables={"counts": _table() if table is None else table})


def test_per_instance_holds_the_retained_instances_aligned_with_obs():
    sdata = _sdata()
    obs, uns = sdata.tables["counts"].obs.copy(), deepcopy(dict(sdata.tables["counts"].uns))
    summary = summarize_table(sdata, "counts")
    assert isinstance(summary, TableSummary)
    per_instance = summary.per_instance
    # cell_1 has no points across the selected classes, so it is excluded.
    assert list(per_instance.index) == ["cell_0", "cell_2", "cell_3", "cell_4", "cell_5"]
    assert list(per_instance.columns) == [
        "region",
        "cell_ID",
        "n_endogenous_points",
        "n_negative_points",
        "n_system_control_points",
        "auxiliary_points_fraction",
        "n_endogenous_points_per_panel_feature",
        "n_negative_points_per_panel_feature",
        "n_system_control_points_per_panel_feature",
    ]
    pd.testing.assert_frame_equal(per_instance[list(obs.columns)], obs.loc[per_instance.index], check_categorical=False)
    for name, column in _COUNT_COLUMNS.items():
        np.testing.assert_allclose(
            per_instance[f"{column}_per_panel_feature"], per_instance[column] / _PANEL_FEATURE_COUNTS[name]
        )
    # Nothing is written to the table.
    pd.testing.assert_frame_equal(sdata.tables["counts"].obs, obs)
    assert sdata.tables["counts"].uns.keys() == uns.keys()


def test_metadata_records_the_resolved_selection_and_the_sources():
    metadata = summarize_table(_sdata(), "counts").metadata
    assert metadata.table_name == "counts"
    assert metadata.region == ("a", "b")
    assert metadata.feature_classes == ("Endogenous", "Negative", "SystemControl")
    assert metadata.expression_class == "Endogenous"
    assert metadata.points_name_by_region == {"a": "points_a", "b": "points_b"}
    assert metadata.panel_feature_counts == _PANEL_FEATURE_COUNTS

    # Selections keep the recorded order; the expression class is recorded also when it is not selected.
    metadata = summarize_table(_sdata(), "counts", region=["b"], feature_classes=["SystemControl", "Negative"]).metadata
    assert metadata.region == ("b",)
    assert metadata.feature_classes == ("Negative", "SystemControl")
    assert metadata.expression_class == "Endogenous"
    assert metadata.points_name_by_region == {"b": "points_b"}
    assert metadata.panel_feature_counts == {"Negative": 4, "SystemControl": 5}


def test_the_selected_classes_decide_the_population():
    everything = summarize_table(_sdata(), "counts").per_class
    assert everything[["n_total_instances", "n_retained_instances", "n_excluded_instances"]].iloc[0].tolist() == [
        6,
        5,
        1,
    ]
    # Without SystemControl, cell_4 has no points across the selected classes either.
    summary = summarize_table(_sdata(), "counts", feature_classes=["Endogenous", "Negative"])
    assert "cell_4" not in summary.per_instance.index
    population = summary.per_class[["n_total_instances", "n_retained_instances", "n_excluded_instances"]]
    assert (population == [6, 4, 2]).all(axis=None)
    assert summary.per_class["pct_excluded_instances"].tolist() == pytest.approx([100 * 2 / 6] * 2)


def test_class_statistics_agree_with_the_retained_instances():
    summary = summarize_table(_sdata(), "counts")
    per_instance = summary.per_instance
    assert summary.per_class["feature_class"].tolist() == ["Endogenous", "Negative", "SystemControl"]
    for row in summary.per_class.itertuples():
        counts = per_instance[_COUNT_COLUMNS[row.feature_class]].to_numpy()
        n_features = _PANEL_FEATURE_COUNTS[row.feature_class]
        assert row.n_points == counts.sum()
        assert row.n_retained_instances_without_class == np.count_nonzero(counts == 0)
        expected = [counts.mean(), np.median(counts), counts.std(ddof=1), np.percentile(counts, 95)]
        actual = [
            row.mean_points_per_instance,
            row.median_points_per_instance,
            row.std_points_per_instance,
            row.p95_points_per_instance,
        ]
        np.testing.assert_allclose(actual, expected)
        assert row.n_features == n_features
        rates = [
            row.mean_points_per_panel_feature_per_instance,
            row.median_points_per_panel_feature_per_instance,
            row.std_points_per_panel_feature_per_instance,
            row.p95_points_per_panel_feature_per_instance,
        ]
        np.testing.assert_allclose(rates, np.asarray(expected) / n_features)


def test_per_region_holds_the_statistics_within_each_region():
    summary = summarize_table(_sdata(), "counts")
    assert summary.per_region["region"].tolist() == ["a"] * 3 + ["b"] * 3
    for name in ["a", "b"]:
        within = summarize_table(_sdata(), "counts", region=name)
        # With one selected region, per_class and per_region hold the same values.
        pd.testing.assert_frame_equal(
            within.per_region.drop(columns="region"), within.per_class, check_categorical=False
        )
        pd.testing.assert_frame_equal(
            summary.per_region.loc[summary.per_region["region"] == name].drop(columns="region").reset_index(drop=True),
            within.per_class,
            check_categorical=False,
        )
        assert set(within.per_instance["region"]) == {name}


@pytest.mark.parametrize("mode", ["lazy", "backed"])
def test_lazy_and_backed_tables_give_the_in_memory_result_without_reading_matrices(tmp_path, monkeypatch, mode):
    _sdata().write(tmp_path / "sdata.zarr")
    sdata = hp.io.read_zarr(tmp_path / "sdata.zarr", table_mode=mode)
    expected = summarize_table(_sdata(), "counts")
    if mode == "backed":
        # A storage-backed matrix runs no Dask graph: fail on any read instead.
        def fail(*args, **kwargs):
            raise AssertionError("summarize_table read a matrix")

        monkeypatch.setattr(type(sdata.tables["counts"].X), "__getitem__", fail)
        monkeypatch.setattr(type(sdata.tables["counts"].X), "to_memory", fail)
    with _CountComputes() as computes:
        summary = summarize_table(sdata, "counts")
    assert computes.count == 0
    assert summary.metadata == expected.metadata
    for name in ["per_instance", "per_class", "per_region"]:
        pd.testing.assert_frame_equal(
            getattr(summary, name), getattr(expected, name), check_categorical=False, check_dtype=False
        )


def _without_record(table):
    del table.uns["point_aggregation"]


def _without_column(column):
    def change(table):
        del table.obs[column]

    return change


def _with_zero_panel_count(table):
    table.uns["point_aggregation"]["panel_feature_counts"]["Negative"] = 0


def _without_recorded_region(table):
    del table.uns["point_aggregation"]["regions"]["b"]


@pytest.mark.parametrize(
    ("change", "kwargs", "message"),
    [
        (None, {"table_name": "absent"}, "not in sdata"),
        (_without_record, {}, "not class-aware"),
        (_without_column("n_negative_points"), {}, "missing aggregation columns"),
        (_without_column("auxiliary_points_fraction"), {}, "missing aggregation columns"),
        (_with_zero_panel_count, {}, "no positive panel feature count"),
        (_without_recorded_region, {}, "has no regions"),
        (None, {"feature_classes": ["Blank"]}, "Unknown feature classes"),
        (None, {"feature_classes": ["Negative", "Negative"]}, "duplicate"),
        (None, {"feature_classes": []}, "one or more exact names"),
        (None, {"region": "c"}, "Unknown regions"),
    ],
)
def test_invalid_tables_and_selections_raise(change, kwargs, message):
    table = _table()
    if change is not None:
        change(table)
    arguments = {"table_name": "counts", **kwargs}
    with pytest.raises(ValueError, match=message):
        summarize_table(_sdata(table), arguments.pop("table_name"), **arguments)
