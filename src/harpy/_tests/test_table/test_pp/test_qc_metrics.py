"""harpy.tb.pp.calculate_qc_metrics on the dummy tables of the tests of summarize_table."""

import numpy as np
import pandas as pd
import pytest
from spatialdata import SpatialData

import harpy as hp
from harpy._tests.test_qc.test_table.test_summarize_table import _OBSM_KEYS, _X, _assert_summaries_equal, _table
from harpy.qc import summarize_table
from harpy.table.pp import calculate_qc_metrics

_CLASSES = ("x", "negative", "system_control")


def test_obs_holds_the_measures_per_instance_and_their_rates():
    table = _table()
    summary = calculate_qc_metrics(table, obsm_keys=_OBSM_KEYS)
    _assert_summaries_equal(summary, summarize_table(_table(), obsm_keys=_OBSM_KEYS))

    measures = [name for name in summary.per_instance.columns if name not in ("region", "cell_ID")]
    pd.testing.assert_frame_equal(table.obs[measures], summary.per_instance[measures])
    assert table.obs["n_points_x"].dtype == np.uint64
    # cell_1 has no points: 0 points and a missing share.
    assert table.obs.loc["cell_1", "n_points_x"] == 0
    assert table.obs.loc["cell_1", ["pct_points_x", "pct_points_negative"]].isna().all()

    for label in _CLASSES:
        rate = table.obs[f"n_points_per_panel_feature_{label}"]
        assert rate.dtype == np.float64
        np.testing.assert_allclose(rate, table.obs[f"n_points_{label}"] / summary.panel_feature_counts[label])
    assert table.obs.loc["cell_1", "n_points_per_panel_feature_x"] == 0


def test_var_holds_the_statistics_per_feature_of_x_labelled_with_its_class():
    table = _table()
    summary = calculate_qc_metrics(table, obsm_keys=_OBSM_KEYS)
    rows = summary.per_feature[summary.per_feature["feature_class"] == "x"]
    columns = [
        "n_points",
        "within_class_fraction",
        "n_instances_without_feature",
        "pct_instances_without_feature",
        "mean_points_per_instance",
        "std_points_per_instance",
    ]
    assert list(table.var.columns) == [f"{column}_x" for column in columns]
    expected = rows.set_index("feature")[columns].rename(columns=lambda column: f"{column}_x")
    pd.testing.assert_frame_equal(table.var, expected, check_names=False)


def test_one_class_adds_no_share_and_a_table_without_x_nothing_to_var():
    table = _table()
    calculate_qc_metrics(table)
    assert "pct_points_x" not in table.obs

    without_x = _table(x=False)
    calculate_qc_metrics(without_x, obsm_keys=_OBSM_KEYS)
    assert without_x.var.columns.empty
    assert "n_points_per_panel_feature_negative" in without_x.obs


def test_an_obsm_matrix_named_x_of_a_table_without_x_adds_nothing_to_var():
    # Its class label is "x", as that of X, but it has no .var axis: only X or a layer has one.
    table = _table(x=False)
    table.obsm["x"] = table.obsm["negative"].copy()
    calculate_qc_metrics(table, obsm_keys=["x"], feature_names={"x": ["A", "B", "C", "D"]})
    assert table.var.columns.empty
    assert "n_points_x" in table.obs


def test_a_second_call_replaces_the_columns():
    table = _table()
    calculate_qc_metrics(table)
    table.X = table.X * 2
    calculate_qc_metrics(table)
    np.testing.assert_array_equal(table.obs["n_points_x"], 2 * np.asarray(_X).sum(axis=1))


def test_calls_with_other_matrices_keep_both_sets_of_columns_each_labelled():
    table = _table()
    table.layers["counts"] = table.X * 2
    calculate_qc_metrics(table)
    calculate_qc_metrics(table, layer="counts")
    np.testing.assert_array_equal(table.obs["n_points_x"], np.asarray(_X).sum(axis=1))
    np.testing.assert_array_equal(table.obs["n_points_counts"], 2 * np.asarray(_X).sum(axis=1))
    np.testing.assert_array_equal(table.var["n_points_counts"], 2 * table.var["n_points_x"])


@pytest.mark.parametrize(
    ("change", "kwargs", "message"),
    [
        (None, {"obsm_keys": ["absent"]}, "not an .obsm key"),
        ("unannotated_region", {}, r"such as \['c'\]; every row needs an annotated region"),
    ],
)
def test_a_failed_check_leaves_the_table_unchanged(change, kwargs, message):
    table = _table()
    if change == "unannotated_region":
        # A region that the annotation does not list, which a table changed after parsing can have.
        table.obs["region"] = table.obs["region"].cat.add_categories("c")
        table.obs.loc["cell_5", "region"] = "c"
    obs, var = table.obs.copy(), table.var.copy()
    with pytest.raises(ValueError, match=message):
        calculate_qc_metrics(table, **kwargs)
    pd.testing.assert_frame_equal(table.obs, obs)
    pd.testing.assert_frame_equal(table.var, var)


def test_the_columns_persist_with_add_table_updates(tmp_path):
    path = tmp_path / "sdata.zarr"
    SpatialData(tables={"counts": _table()}).write(path)
    sdata = hp.io.read_zarr(path)
    calculate_qc_metrics(sdata.tables["counts"], obsm_keys=_OBSM_KEYS)
    hp.tb.io.add_table_updates(sdata, table_name="counts", overwrite=True)

    stored = hp.io.read_zarr(path, table_mode="eager").tables["counts"]
    pd.testing.assert_frame_equal(stored.obs, sdata.tables["counts"].obs)
    pd.testing.assert_frame_equal(stored.var, sdata.tables["counts"].var)
