"""hp.qc.summarize_table on dummy tables with CSR count matrices: in memory, lazy and backed."""

from copy import deepcopy

import dask.array as da
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from anndata.abc import CSRDataset
from scipy import sparse
from spatialdata import SpatialData
from spatialdata.models import TableModel

import harpy as hp
from harpy._tests.test_table.test_io.test_updates import _CountComputes
from harpy.qc import TableSummary, TableSummaryMetadata, summarize_table

# Six instances in two regions, a and b. cell_1 has no points at all; cell_4 has
# points of system_control only.
_X = [[7, 3], [0, 0], [5, 0], [4, 4], [0, 0], [0, 2]]
_NEGATIVE = [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]
_SYSTEM_CONTROL = [[0] * 5, [0] * 5, [0, 1, 0, 0, 0], [0] * 5, [2, 0, 1, 0, 0], [0] * 5]
_CLASSES = {"x": _X, "negative": _NEGATIVE, "system_control": _SYSTEM_CONTROL}
_OBSM_KEYS = ["negative", "system_control"]


def _counts(dense, *, explicit_zero=None):
    """Return uint32 CSR counts, optionally with one explicitly stored zero at ``(row, column)``."""
    matrix = sparse.csr_matrix(np.asarray(dense, dtype=np.uint32))
    if explicit_zero is None:
        return matrix
    coo = matrix.tocoo()
    row, column = explicit_zero
    return sparse.csr_matrix(
        (np.append(coo.data, np.uint32(0)), (np.append(coo.row, row), np.append(coo.col, column))), shape=matrix.shape
    )


def _table(*, csr=sparse.csr_matrix, x=True):
    obs = pd.DataFrame(
        {"region": pd.Categorical(["a"] * 3 + ["b"] * 3), "cell_ID": [1, 2, 3, 1, 2, 3]},
        index=[f"cell_{i}" for i in range(6)],
    )
    adata = AnnData(
        # cell_2 stores an explicit zero for GeneB, which is no detection.
        X=csr(_counts(_X, explicit_zero=(2, 1))) if x else None,
        obs=obs,
        var=pd.DataFrame(index=["GeneA", "GeneB"]),
        obsm={"negative": csr(_counts(_NEGATIVE)), "system_control": csr(_counts(_SYSTEM_CONTROL))},
    )
    return TableModel.parse(adata, region=["a", "b"], region_key="region", instance_key="cell_ID")


def _sdata(table=None):
    # Only the table: summarize_table needs neither the annotated labels nor the source points.
    return SpatialData(tables={"counts": _table() if table is None else table})


def _assert_summaries_equal(summary, expected):
    assert summary.metadata == expected.metadata
    for name in ["per_instance", "per_class", "per_region"]:
        pd.testing.assert_frame_equal(getattr(summary, name), getattr(expected, name))


def test_per_instance_holds_each_class_per_retained_instance_aligned_with_obs():
    sdata = _sdata()
    table = sdata.tables["counts"]
    assert table.X.nnz == np.count_nonzero(_X) + 1
    obs, uns = table.obs.copy(), deepcopy(dict(table.uns))
    summary = summarize_table(sdata, "counts", obsm_keys=_OBSM_KEYS)
    assert isinstance(summary, TableSummary)
    per_instance = summary.per_instance
    # cell_1 has no points in any class read, so it is excluded.
    assert list(per_instance.index) == ["cell_0", "cell_2", "cell_3", "cell_4", "cell_5"]
    assert list(per_instance.columns) == [
        "region",
        "cell_ID",
        "n_x_points",
        "n_negative_points",
        "n_system_control_points",
        "n_x_detected_features",
        "n_negative_detected_features",
        "n_system_control_detected_features",
        "pct_x_points",
        "pct_negative_points",
        "pct_system_control_points",
    ]
    pd.testing.assert_frame_equal(
        per_instance[["region", "cell_ID"]], obs.loc[per_instance.index, ["region", "cell_ID"]]
    )
    retained = [0, 2, 3, 4, 5]
    totals = np.asarray(_X).sum(axis=1) + np.asarray(_NEGATIVE).sum(axis=1) + np.asarray(_SYSTEM_CONTROL).sum(axis=1)
    for label, dense in _CLASSES.items():
        dense = np.asarray(dense)[retained]
        assert per_instance[f"n_{label}_points"].dtype == np.uint64
        np.testing.assert_array_equal(per_instance[f"n_{label}_points"], dense.sum(axis=1))
        # The stored zero of cell_2 in X is not a detection.
        np.testing.assert_array_equal(per_instance[f"n_{label}_detected_features"], (dense > 0).sum(axis=1))
        np.testing.assert_allclose(per_instance[f"pct_{label}_points"], 100 * dense.sum(axis=1) / totals[retained])
    np.testing.assert_allclose(per_instance.filter(like="pct_").sum(axis=1), 100)
    # Nothing is written to the table.
    pd.testing.assert_frame_equal(table.obs, obs)
    assert table.uns.keys() == uns.keys()


def test_the_matrices_read_decide_the_classes_and_the_population():
    sdata = _sdata()
    alone = summarize_table(sdata, "counts")
    assert alone.feature_classes == ("x",)
    # With one class there are no percentages; cell_1 and cell_4 have no points in X.
    assert list(alone.per_instance.columns) == ["region", "cell_ID", "n_x_points", "n_x_detected_features"]
    assert list(alone.per_instance.index) == ["cell_0", "cell_2", "cell_3", "cell_5"]

    # A class matrix adds the instances with points of that class only.
    summary = summarize_table(sdata, "counts", obsm_keys="system_control")
    assert summary.feature_classes == ("x", "system_control")
    assert "cell_4" in summary.per_instance.index
    population = summary.per_class[["n_total_instances", "n_retained_instances", "n_excluded_instances"]]
    assert (population == [6, 5, 1]).all(axis=None)
    assert summary.per_class["pct_excluded_instances"].tolist() == pytest.approx([100 / 6] * 2)


def test_a_table_without_x_is_summarized_from_its_obsm_matrices():
    summary = summarize_table(_sdata(_table(x=False)), "counts", obsm_keys=_OBSM_KEYS)
    assert summary.feature_classes == ("negative", "system_control")
    assert list(summary.per_instance.index) == ["cell_0", "cell_2", "cell_3", "cell_4", "cell_5"]


def _sdata_with_normalized_x():
    """Return the table with its raw counts in the layer "RawCounts" and a normalized X."""
    table = _table()
    table.layers["RawCounts"] = table.X.copy()
    # A normalized X holds fractions, which are not counts.
    table.X = table.X.astype(np.float64) / 3
    return _sdata(table)


def test_a_normalized_x_raises_and_asks_for_the_raw_count_layer():
    with pytest.raises(ValueError, match=r"adata\.X of table 'counts' holds values that are not counts.*layer="):
        summarize_table(_sdata_with_normalized_x(), "counts")


def test_a_layer_is_read_in_place_of_x():
    summary = summarize_table(_sdata_with_normalized_x(), "counts", layer="RawCounts", obsm_keys=["negative"])
    assert summary.metadata.layer == "RawCounts"
    # The layer's label is its name in snake case, and X is not read as well.
    assert summary.feature_classes == ("raw_counts", "negative")
    # cell_4 has points in neither class read, so the retained instances are cells 0, 2, 3 and 5.
    np.testing.assert_array_equal(summary.per_instance["n_raw_counts_points"], [10, 5, 8, 2])


def test_coinciding_class_labels_raise():
    table = _table()
    # The layer's label, "negative", is also the label of obsm["negative"].
    table.layers["Negative"] = table.X.copy()
    with pytest.raises(ValueError, match="labels coincide"):
        summarize_table(_sdata(table), "counts", layer="Negative", obsm_keys=["negative"])


def test_class_statistics_agree_with_the_retained_instances_in_the_column_order_of_the_bin_summary():
    summary = summarize_table(_sdata(), "counts", obsm_keys=_OBSM_KEYS)
    per_class = summary.per_class
    # The columns of SpatialBinSummary.per_class, instances in place of bins, then those only the table has.
    assert list(per_class.columns) == [
        "feature_class",
        "n_total_instances",
        "n_retained_instances",
        "n_excluded_instances",
        "pct_excluded_instances",
        "n_points",
        "n_retained_instances_without_class",
        "pct_retained_instances_without_class",
        "mean_points_per_instance",
        "median_points_per_instance",
        "std_points_per_instance",
        "p95_points_per_instance",
        "mean_detected_features_per_instance",
        "median_detected_features_per_instance",
        "std_detected_features_per_instance",
        "p95_detected_features_per_instance",
        "n_features",
    ]
    assert per_class["n_points"].dtype == np.uint64
    assert per_class["feature_class"].tolist() == ["x", "negative", "system_control"]
    for row in per_class.itertuples():
        for measure, column in [("points", "points"), ("detected_features", "detected_features")]:
            values = summary.per_instance[f"n_{row.feature_class}_{column}"].to_numpy()
            expected = [values.mean(), np.median(values), values.std(ddof=1), np.percentile(values, 95)]
            actual = [
                getattr(row, f"{statistic}_{measure}_per_instance") for statistic in ["mean", "median", "std", "p95"]
            ]
            np.testing.assert_allclose(actual, expected)
        counts = summary.per_instance[f"n_{row.feature_class}_points"].to_numpy()
        assert row.n_points == counts.sum()
        assert row.n_retained_instances_without_class == np.count_nonzero(counts == 0)
    # The denominators of a class's points per panel feature, from per_class, as on the points side.
    assert summary.panel_feature_counts == {"x": 2, "negative": 4, "system_control": 5}


def test_the_denominator_follows_a_subset_of_x():
    table = _table()[:, ["GeneA"]].copy()
    summary = summarize_table(_sdata(table), "counts")
    assert summary.panel_feature_counts == {"x": 1}
    np.testing.assert_array_equal(summary.per_instance["n_x_points"], [7, 5, 4])


def test_per_region_holds_the_statistics_within_each_region():
    summary = summarize_table(_sdata(), "counts", obsm_keys=_OBSM_KEYS)
    assert summary.per_region["region"].tolist() == ["a"] * 3 + ["b"] * 3
    assert list(summary.per_region.columns) == ["region", *summary.per_class.columns]
    for name in ["a", "b"]:
        within = summarize_table(_sdata(), "counts", obsm_keys=_OBSM_KEYS, region=name)
        assert set(within.per_instance["region"]) == {name}
        # With one selected region, per_class and per_region hold the same values.
        pd.testing.assert_frame_equal(within.per_region.drop(columns="region"), within.per_class)
        pd.testing.assert_frame_equal(
            summary.per_region.loc[summary.per_region["region"] == name].drop(columns="region").reset_index(drop=True),
            within.per_class,
        )


def test_metadata_records_the_inputs():
    metadata = summarize_table(_sdata(), "counts", obsm_keys="negative").metadata
    assert metadata == TableSummaryMetadata(
        table_name="counts",
        region=("a", "b"),
        layer=None,
        obsm_keys=("negative",),
    )
    assert summarize_table(_sdata(), "counts", region=["b"]).metadata.region == ("b",)


def test_csr_arrays_give_the_result_of_csr_matrices():
    expected = summarize_table(_sdata(), "counts", obsm_keys=_OBSM_KEYS)
    summary = summarize_table(_sdata(_table(csr=sparse.csr_array)), "counts", obsm_keys=_OBSM_KEYS)
    _assert_summaries_equal(summary, expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
def test_counts_stored_as_whole_floats_or_signed_integers_give_the_result_of_uint32(dtype):
    expected = summarize_table(_sdata(), "counts", obsm_keys=_OBSM_KEYS)
    summary = summarize_table(_sdata(_table(csr=lambda matrix: matrix.astype(dtype))), "counts", obsm_keys=_OBSM_KEYS)
    _assert_summaries_equal(summary, expected)


@pytest.mark.parametrize("region", [None, "b"])
@pytest.mark.parametrize("mode", ["lazy", "backed"])
def test_lazy_and_backed_tables_give_the_in_memory_result(tmp_path, mode, region):
    _sdata().write(tmp_path / "sdata.zarr")
    sdata = hp.io.read_zarr(tmp_path / "sdata.zarr", table_mode=mode, sparse_chunks=2)
    if mode == "lazy":
        assert isinstance(sdata.tables["counts"].X, da.Array)
        assert sdata.tables["counts"].X.numblocks == (3, 1)
    else:
        assert isinstance(sdata.tables["counts"].X, CSRDataset)
    expected = summarize_table(_sdata(), "counts", obsm_keys=_OBSM_KEYS, region=region)
    with _CountComputes() as computes:
        summary = summarize_table(sdata, "counts", obsm_keys=_OBSM_KEYS, region=region)
    # A lazy table runs one Dask compute for all its matrices; a backed table runs none.
    assert computes.count == (1 if mode == "lazy" else 0)
    _assert_summaries_equal(summary, expected)


def test_lazy_matrices_with_different_row_chunks_run_one_compute():
    table = _table()
    table.X = da.from_array(table.X, chunks=(2, -1), asarray=False)
    table.obsm["negative"] = da.from_array(table.obsm["negative"], chunks=(4, -1), asarray=False)
    # Column blocks are joined into whole rows.
    table.obsm["system_control"] = da.from_array(table.obsm["system_control"], chunks=(3, 2), asarray=False)
    expected = summarize_table(_sdata(), "counts", obsm_keys=_OBSM_KEYS)
    with _CountComputes() as computes:
        summary = summarize_table(_sdata(table), "counts", obsm_keys=_OBSM_KEYS)
    assert computes.count == 1
    _assert_summaries_equal(summary, expected)


@pytest.mark.parametrize(
    ("key", "convert", "kind"),
    [
        (None, lambda matrix: matrix.toarray(), "ndarray"),
        ("negative", lambda matrix: matrix.tocsc(), "csc_matrix"),
        ("negative", lambda matrix: da.from_array(matrix.toarray(), chunks=(3, -1)), "Dask array of ndarray blocks"),
    ],
)
def test_matrices_that_are_not_csr_raise_before_any_value_is_read(key, convert, kind):
    table = _table()
    if key is None:
        table.X = convert(table.X)
    else:
        table.obsm[key] = convert(table.obsm[key])
    with _CountComputes() as computes, pytest.raises(ValueError, match=f"is a {kind}, not a CSR matrix"):
        summarize_table(_sdata(table), "counts", obsm_keys=_OBSM_KEYS)
    assert computes.count == 0


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("value", [0.5, -1.0])
def test_matrices_whose_values_are_not_counts_raise(value, lazy):
    table = _table()
    matrix = table.obsm["negative"].astype(np.float64)
    matrix.data[0] = value
    table.obsm["negative"] = da.from_array(matrix, chunks=(2, -1), asarray=False) if lazy else matrix
    with pytest.raises(ValueError, match=r"adata\.obsm\['negative'\] of table 'counts' holds values that are not"):
        summarize_table(_sdata(table), "counts", obsm_keys=_OBSM_KEYS)


def _without_annotation(table):
    del table.uns["spatialdata_attrs"]


def _without_x(table):
    table.X = None


@pytest.mark.parametrize(
    ("change", "kwargs", "message"),
    [
        (None, {"table_name": "absent"}, "not in sdata"),
        (_without_annotation, {}, "no SpatialData annotation"),
        (None, {"layer": "absent"}, "not a layer of table"),
        (None, {"obsm_keys": ["absent"]}, "not an .obsm key of table"),
        (None, {"obsm_keys": ["negative", "negative"]}, "duplicate keys"),
        (None, {"obsm_keys": [1]}, "sequence of .obsm keys"),
        (_without_x, {}, "neither layer nor obsm_keys"),
        (None, {"region": "c"}, "Unknown regions"),
        (None, {"region": ["a", "a"]}, "duplicate names"),
        (None, {"region": []}, "one or more exact names"),
    ],
)
def test_invalid_tables_and_arguments_raise(change, kwargs, message):
    table = _table()
    if change is not None:
        change(table)
    arguments = {"table_name": "counts", **kwargs}
    with pytest.raises(ValueError, match=message):
        summarize_table(_sdata(table), arguments.pop("table_name"), **arguments)
