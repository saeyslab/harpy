"""hp.qc.summarize_table_by_feature and hp.qc.instance_histogram_by_feature on the dummy tables of summarize_table."""

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from anndata.abc import CSRDataset
from scipy import sparse

import harpy as hp
import harpy.qc._histogram as histogram_module
from harpy._tests.test_qc.test_table.test_instance_plots import _capture
from harpy._tests.test_qc.test_table.test_summarize_table import (
    _NEGATIVE,
    _OBSM_KEYS,
    _SYSTEM_CONTROL,
    _X,
    _sdata,
    _table,
)
from harpy._tests.test_table.test_io.test_updates import _CountComputes
from harpy.qc import FeatureTableSummary, summarize_table, summarize_table_by_feature

# The retained instances of a summary of the three classes: cell_1 has no points at all.
_RETAINED = [0, 2, 3, 4, 5]
# Requested in another order than the summary's: x, negative, then system_control.
_REQUESTED = ["SystemControl01", "Negative04", "GeneA"]
_FEATURES = ["GeneA", "Negative04", "SystemControl01"]
# Each requested feature's column, in the order of _FEATURES.
_COLUMNS = [np.asarray(_X)[:, 0], np.asarray(_NEGATIVE)[:, 3], np.asarray(_SYSTEM_CONTROL)[:, 0]]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _summaries(sdata=None):
    sdata = _sdata() if sdata is None else sdata
    summary = summarize_table(sdata, "counts", obsm_keys=_OBSM_KEYS)
    return summary, summarize_table_by_feature(sdata, summary=summary, features=_REQUESTED)


def _assert_feature_summaries_equal(result, expected):
    assert result.metadata == expected.metadata
    for name in ["per_instance", "per_feature", "per_region_feature"]:
        pd.testing.assert_frame_equal(getattr(result, name), getattr(expected, name))


def test_per_instance_holds_each_requested_feature_per_retained_instance():
    summary, result = _summaries()
    assert isinstance(result, FeatureTableSummary)
    assert result.metadata == summary.metadata
    per_instance = result.per_instance
    assert list(per_instance.columns) == ["region", "cell_ID", "feature", "n_points"]
    # The summary's order of classes and columns, not the request order.
    assert per_instance["feature"].cat.categories.tolist() == _FEATURES
    assert per_instance["feature"].tolist() == [name for name in _FEATURES for _ in _RETAINED]
    assert per_instance["n_points"].dtype == np.uint64
    np.testing.assert_array_equal(per_instance["n_points"], np.concatenate([column[_RETAINED] for column in _COLUMNS]))
    identity = summary.per_instance[["region", "cell_ID"]].reset_index(drop=True)
    for position in range(len(_FEATURES)):
        rows = per_instance.iloc[position * len(_RETAINED) : (position + 1) * len(_RETAINED)]
        pd.testing.assert_frame_equal(rows[["region", "cell_ID"]].reset_index(drop=True), identity)


def test_an_instance_retained_for_another_class_and_an_undetected_feature_keep_their_zeros():
    summary = summarize_table(_sdata(), "counts", obsm_keys=_OBSM_KEYS)
    result = summarize_table_by_feature(_sdata(), summary=summary, features=["GeneA", "Negative03"])
    per_instance = result.per_instance.set_index(["feature", "cell_ID", "region"])["n_points"]
    # cell_4, in region b, is retained for its system_control points: it holds no GeneA.
    assert per_instance.loc[("GeneA", 2, "b")] == 0
    negative03 = result.per_feature.set_index("feature").loc["Negative03"]
    assert negative03["n_points"] == 0
    assert negative03["n_retained_instances_without_feature"] == len(_RETAINED)
    assert negative03["pct_retained_instances_without_feature"] == 100
    assert negative03["mean_points_per_instance"] == negative03["median_points_per_instance"] == 0


def test_per_feature_statistics_agree_with_the_values_and_the_table_summary():
    summary, result = _summaries()
    per_feature = result.per_feature
    assert list(per_feature.columns) == [
        "feature",
        "feature_class",
        "n_total_instances",
        "n_retained_instances",
        "n_excluded_instances",
        "pct_excluded_instances",
        "n_points",
        "n_retained_instances_without_feature",
        "pct_retained_instances_without_feature",
        "mean_points_per_instance",
        "median_points_per_instance",
        "std_points_per_instance",
        "p95_points_per_instance",
    ]
    assert per_feature["feature"].tolist() == _FEATURES
    assert per_feature["feature_class"].tolist() == ["x", "negative", "system_control"]
    population = ["n_total_instances", "n_retained_instances", "n_excluded_instances", "pct_excluded_instances"]
    for column in population:
        assert (per_feature[column] == summary.per_class[column].iloc[0]).all()
    for name, column in zip(_FEATURES, _COLUMNS, strict=True):
        values = column[_RETAINED]
        row = per_feature.set_index("feature").loc[name]
        assert row["n_points"] == values.sum()
        assert row["n_retained_instances_without_feature"] == np.count_nonzero(values == 0)
        assert row["mean_points_per_instance"] == pytest.approx(values.mean())
        assert row["median_points_per_instance"] == pytest.approx(np.median(values))
        assert row["std_points_per_instance"] == pytest.approx(values.std(ddof=1))
        assert row["p95_points_per_instance"] == pytest.approx(np.percentile(values, 95))
        # The totals, mean and SD of the table summary, which has no median or percentile.
        reference = summary.per_feature.set_index("feature").loc[name]
        for statistic in ["n_points", "mean_points_per_instance", "std_points_per_instance"]:
            assert row[statistic] == pytest.approx(reference[statistic])


def test_per_region_feature_holds_the_statistics_within_each_region():
    summary, result = _summaries()
    per_region_feature = result.per_region_feature
    assert per_region_feature.columns[0] == "region"
    assert per_region_feature["region"].cat.categories.tolist() == ["a", "b"]
    regions = np.asarray(["a"] * 3 + ["b"] * 3)
    for region, n_total in [("a", 3), ("b", 3)]:
        retained = [row for row in _RETAINED if regions[row] == region]
        rows = per_region_feature[per_region_feature["region"] == region].set_index("feature")
        assert (rows["n_total_instances"] == n_total).all()
        assert (rows["n_retained_instances"] == len(retained)).all()
        for name, column in zip(_FEATURES, _COLUMNS, strict=True):
            values = column[retained]
            assert rows.loc[name, "n_points"] == values.sum()
            assert rows.loc[name, "median_points_per_instance"] == pytest.approx(np.median(values))
    pd.testing.assert_series_equal(
        per_region_feature.groupby("feature", observed=True)["n_points"].sum().astype(np.uint64),
        result.per_feature.set_index("feature")["n_points"],
    )


def test_the_classes_without_a_requested_feature_stay_categories():
    summary = summarize_table(_sdata(), "counts", obsm_keys=_OBSM_KEYS)
    result = summarize_table_by_feature(_sdata(), summary=summary, features="Negative01")
    for frame in [result.per_feature, result.per_region_feature]:
        assert frame["feature_class"].tolist() == ["negative"] * len(frame)
        assert frame["feature_class"].cat.categories.tolist() == ["x", "negative", "system_control"]
        assert frame["feature"].cat.categories.tolist() == ["Negative01"]


def test_a_table_reordered_since_the_summary_gives_the_same_result():
    sdata = _sdata()
    summary, expected = _summaries(sdata)
    # The rows move with their .obs names, so the instances are found by name.
    sdata.tables["counts"] = sdata.tables["counts"][::-1].copy()
    result = summarize_table_by_feature(sdata, summary=summary, features=_REQUESTED)
    _assert_feature_summaries_equal(result, expected)


@pytest.mark.parametrize("mode", ["lazy", "backed"])
def test_lazy_and_backed_tables_give_the_in_memory_result(tmp_path, mode):
    _, expected = _summaries()
    _sdata().write(tmp_path / "sdata.zarr")
    sdata = hp.io.read_zarr(tmp_path / "sdata.zarr", table_mode=mode, sparse_chunks=2)
    matrix_type = da.Array if mode == "lazy" else CSRDataset
    assert isinstance(sdata.tables["counts"].X, matrix_type)
    summary = summarize_table(sdata, "counts", obsm_keys=_OBSM_KEYS)
    with _CountComputes() as computes:
        result = summarize_table_by_feature(sdata, summary=summary, features=_REQUESTED)
    # A lazy table runs one Dask compute for all its matrices; a backed table runs none.
    assert computes.count == (1 if mode == "lazy" else 0)
    _assert_feature_summaries_equal(result, expected)


def _with_ambiguous_name(table):
    table.uns["feature_matrices"]["negative"]["feature_columns"][0] = "GeneA"


def _reordered_var_names(table):
    table.var_names = ["GeneB", "GeneA"]


def _narrower_obsm(table):
    table.obsm["negative"] = sparse.csr_matrix(np.zeros((6, 3), dtype=np.uint32))


def _reordered_registration(table):
    # Same width, other names: only the registration shows the change.
    table.uns["feature_matrices"]["negative"]["feature_columns"].reverse()


def _renamed_instances(table):
    table.obs_names = [f"other_{i}" for i in range(6)]


def _permuted_instance_names(table):
    # Every name is still in the table, but now labels another instance.
    table.obs_names = [f"cell_{i}" for i in reversed(range(6))]


def _normalized_x(table):
    table.X = sparse.csr_matrix(np.asarray(_X, dtype=np.float64) / 2)


@pytest.mark.parametrize(
    ("before", "after", "features", "message"),
    [
        (None, None, "Missing", r"Unknown features: \['Missing'\]"),
        (None, None, [], "one or more exact feature names"),
        (None, None, ["GeneA", "GeneA"], "duplicate"),
        (_with_ambiguous_name, None, "GeneA", r"More than one column .* \{'GeneA': \['x', 'negative'\]\}"),
        (None, _reordered_var_names, "GeneA", r"adata.X .* no longer has the columns"),
        (None, _narrower_obsm, "GeneA", r"adata.obsm\['negative'\] .* no longer has the columns"),
        (None, _reordered_registration, "GeneA", r"adata.obsm\['negative'\] .* no longer has the columns"),
        (None, _renamed_instances, "GeneA", "no longer in table"),
        (None, _permuted_instance_names, "GeneA", "now label other instances"),
        (None, _normalized_x, "GeneA", "not counts"),
    ],
)
def test_invalid_features_and_tables_changed_since_the_summary_raise(before, after, features, message):
    table = _table()
    if before is not None:
        before(table)
    sdata = _sdata(table)
    summary = summarize_table(sdata, "counts", obsm_keys=_OBSM_KEYS)
    if after is not None:
        after(sdata.tables["counts"])
    with pytest.raises(ValueError, match=message):
        summarize_table_by_feature(sdata, summary=summary, features=features)


def test_summarize_table_by_feature_takes_a_table_summary():
    with pytest.raises(TypeError, match="summary must be a TableSummary"):
        summarize_table_by_feature(_sdata(), summary=None, features="GeneA")


def test_instance_histogram_by_feature_plots_one_feature_annotated_with_its_statistics(monkeypatch):
    _, result = _summaries()
    calls = _capture(monkeypatch, histogram_module, "_plot_histogram")
    ax = hp.qc.instance_histogram_by_feature(result, feature="GeneA")
    (values,), options = calls[0]
    np.testing.assert_array_equal(values, np.asarray(_X)[_RETAINED, 0])
    row = result.per_feature.set_index("feature").loc["GeneA"]
    assert options["median"] == pytest.approx(row["median_points_per_instance"])
    assert options["std"] == pytest.approx(row["std_points_per_instance"])
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("GeneA points per instance", "Number of instances")


def test_instance_histogram_by_feature_rejects_unknown_features_and_other_summaries():
    summary, result = _summaries()
    with pytest.raises(
        ValueError, match=r"Unknown feature 'GeneB'; choose from \['GeneA', 'Negative04', 'SystemControl01'\]"
    ):
        hp.qc.instance_histogram_by_feature(result, feature="GeneB")
    with pytest.raises(TypeError, match="summary must be a FeatureTableSummary"):
        hp.qc.instance_histogram_by_feature(summary, feature="GeneA")
