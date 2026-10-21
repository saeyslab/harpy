"""hp.qc.summarize_table on dummy tables with CSR count matrices: in memory, lazy and backed."""

import warnings
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
_FEATURE_NAMES = {
    "x": ("GeneA", "GeneB"),
    "negative": tuple(f"Negative0{i}" for i in range(1, 5)),
    "system_control": tuple(f"SystemControl0{i}" for i in range(1, 6)),
}


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
    # The .obsm matrices register their column names, as aggregate_points writes them.
    adata.uns["feature_matrices"] = {
        key: {"schema_version": 1, "source_kind": "test", "feature_columns": list(_FEATURE_NAMES[key])}
        for key in _OBSM_KEYS
    }
    # Only the table, outside any SpatialData object: summarize_table needs neither the annotated labels nor
    # the source points.
    return TableModel.parse(adata, region=["a", "b"], region_key="region", instance_key="cell_ID")


def _assert_summaries_equal(summary, expected):
    assert summary.metadata == expected.metadata
    for name in ["per_instance", "per_class", "per_region_class", "per_feature", "per_region_feature"]:
        pd.testing.assert_frame_equal(getattr(summary, name), getattr(expected, name))


def test_per_instance_holds_each_class_per_row_aligned_with_obs():
    table = _table()
    assert table.X.nnz == np.count_nonzero(_X) + 1
    obs, uns = table.obs.copy(), deepcopy(dict(table.uns))
    with warnings.catch_warnings():
        # The missing share of cell_1 raises no warning.
        warnings.simplefilter("error", RuntimeWarning)
        summary = summarize_table(table, obsm_keys=_OBSM_KEYS)
    assert isinstance(summary, TableSummary)
    per_instance = summary.per_instance
    # Every row is an instance, also cell_1 without points in any class read.
    assert list(per_instance.index) == list(obs.index)
    assert list(per_instance.columns) == [
        "region",
        "cell_ID",
        "n_points_x",
        "n_points_negative",
        "n_points_system_control",
        "n_detected_features_x",
        "n_detected_features_negative",
        "n_detected_features_system_control",
        "pct_points_x",
        "pct_points_negative",
        "pct_points_system_control",
    ]
    pd.testing.assert_frame_equal(per_instance[["region", "cell_ID"]], obs[["region", "cell_ID"]])
    totals = np.asarray(_X).sum(axis=1) + np.asarray(_NEGATIVE).sum(axis=1) + np.asarray(_SYSTEM_CONTROL).sum(axis=1)
    with_points = totals > 0
    for label, dense in _CLASSES.items():
        dense = np.asarray(dense)
        assert per_instance[f"n_points_{label}"].dtype == np.uint64
        np.testing.assert_array_equal(per_instance[f"n_points_{label}"], dense.sum(axis=1))
        # The stored zero of cell_2 in X is not a detection.
        np.testing.assert_array_equal(per_instance[f"n_detected_features_{label}"], (dense > 0).sum(axis=1))
        share = per_instance[f"pct_points_{label}"].to_numpy()
        np.testing.assert_allclose(share[with_points], 100 * dense.sum(axis=1)[with_points] / totals[with_points])
    # cell_1 has 0 points and 0 detected features in each class, and a missing share.
    assert per_instance.loc["cell_1"].filter(like="pct_").isna().all()
    np.testing.assert_allclose(per_instance.filter(like="pct_").loc[with_points].sum(axis=1), 100)
    # Nothing is written to the table.
    pd.testing.assert_frame_equal(table.obs, obs)
    assert table.uns.keys() == uns.keys()


def test_the_matrices_read_decide_the_classes_not_the_population():
    table = _table()
    alone = summarize_table(table)
    assert alone.feature_classes == ("x",)
    # With one class there are no shares; cell_1 and cell_4, without points in X, count as zeros.
    assert list(alone.per_instance.columns) == ["region", "cell_ID", "n_points_x", "n_detected_features_x"]
    np.testing.assert_array_equal(alone.per_instance["n_points_x"], [10, 0, 5, 8, 0, 2])

    summary = summarize_table(table, obsm_keys="system_control")
    assert summary.feature_classes == ("x", "system_control")
    assert summary.per_class["n_instances"].tolist() == [6, 6]
    assert alone.per_class["n_instances"].tolist() == [6]


def test_a_table_without_x_is_summarized_from_its_obsm_matrices():
    summary = summarize_table(_table(x=False), obsm_keys=_OBSM_KEYS)
    assert summary.feature_classes == ("negative", "system_control")
    np.testing.assert_array_equal(summary.per_instance["n_points_negative"], [1, 0, 0, 2, 0, 1])


def test_an_x_without_columns_raises():
    # Every gene filtered away: X has no feature, so its points per panel feature would be undefined.
    with pytest.raises(ValueError, match=r"adata\.X has no columns, so its class would have no features"):
        summarize_table(_table()[:, []].copy(), obsm_keys=_OBSM_KEYS)


def _table_with_normalized_x():
    """Return the table with its raw counts in the layer "RawCounts" and a normalized X."""
    table = _table()
    table.layers["RawCounts"] = table.X.copy()
    # A normalized X holds fractions, which are not counts.
    table.X = table.X.astype(np.float64) / 3
    return table


def test_a_normalized_x_raises_and_asks_for_the_raw_count_layer():
    with pytest.raises(ValueError, match=r"adata\.X holds values that are not counts.*layer="):
        summarize_table(_table_with_normalized_x())


def test_a_layer_is_read_in_place_of_x():
    summary = summarize_table(_table_with_normalized_x(), layer="RawCounts", obsm_keys=["negative"])
    assert summary.metadata.layer == "RawCounts"
    # The layer's label is its name in snake case, and X is not read as well.
    assert summary.feature_classes == ("raw_counts", "negative")
    np.testing.assert_array_equal(summary.per_instance["n_points_raw_counts"], [10, 0, 5, 8, 0, 2])


def test_coinciding_class_labels_raise():
    table = _table()
    # The layer's label, "negative", is also the label of obsm["negative"].
    table.layers["Negative"] = table.X.copy()
    with pytest.raises(ValueError, match="labels coincide"):
        summarize_table(table, layer="Negative", obsm_keys=["negative"])


def test_per_class_columns_and_their_order():
    per_class = summarize_table(_table(), obsm_keys=_OBSM_KEYS).per_class
    # The columns of SpatialBinSummary.per_class, instances in place of bins and one population column in place
    # of four, then those only the table has.
    assert list(per_class.columns) == [
        "feature_class",
        "n_instances",
        "n_points",
        "n_instances_without_class",
        "pct_instances_without_class",
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


def test_class_statistics_agree_with_every_row():
    summary = summarize_table(_table(), obsm_keys=_OBSM_KEYS)
    per_class = summary.per_class
    assert per_class["feature_class"].tolist() == ["x", "negative", "system_control"]
    for row in per_class.itertuples():
        for measure in ["points", "detected_features"]:
            values = summary.per_instance[f"n_{measure}_{row.feature_class}"].to_numpy()
            expected = [values.mean(), np.median(values), values.std(ddof=1), np.percentile(values, 95)]
            actual = [
                getattr(row, f"{statistic}_{measure}_per_instance") for statistic in ["mean", "median", "std", "p95"]
            ]
            np.testing.assert_allclose(actual, expected)
        counts = summary.per_instance[f"n_points_{row.feature_class}"].to_numpy()
        assert row.n_points == counts.sum()
        assert row.n_instances_without_class == np.count_nonzero(counts == 0)
        assert row.pct_instances_without_class == pytest.approx(100 * np.count_nonzero(counts == 0) / 6)
    # The denominators of a class's points per panel feature, from per_class, as on the points side.
    assert summary.panel_feature_counts == {"x": 2, "negative": 4, "system_control": 5}


def test_the_denominator_follows_a_subset_of_x():
    table = _table()[:, ["GeneA"]].copy()
    summary = summarize_table(table)
    assert summary.panel_feature_counts == {"x": 1}
    np.testing.assert_array_equal(summary.per_instance["n_points_x"], [7, 0, 5, 4, 0, 0])


def test_per_region_class_holds_the_class_statistics_within_each_region():
    summary = summarize_table(_table(), obsm_keys=_OBSM_KEYS)
    per_region_class = summary.per_region_class
    assert per_region_class["region"].tolist() == ["a"] * 3 + ["b"] * 3
    assert list(per_region_class.columns) == ["region", *summary.per_class.columns]
    for name in ["a", "b"]:
        within = summarize_table(_table(), obsm_keys=_OBSM_KEYS, region=name)
        assert set(within.per_instance["region"]) == {name}
        # With one selected region, per_class and per_region_class hold the same values.
        pd.testing.assert_frame_equal(within.per_region_class.drop(columns="region"), within.per_class)
        pd.testing.assert_frame_equal(
            per_region_class.loc[per_region_class["region"] == name].drop(columns="region").reset_index(drop=True),
            within.per_class,
        )


def test_metadata_records_the_inputs():
    metadata = summarize_table(_table(), obsm_keys="negative").metadata
    assert metadata == TableSummaryMetadata(
        region=("a", "b"),
        layer=None,
        obsm_keys=("negative",),
        feature_matrices_key="feature_matrices",
    )
    assert summarize_table(_table(), region=["b"]).metadata.region == ("b",)


def test_per_feature_columns_and_their_order():
    per_feature = summarize_table(_table(), obsm_keys=_OBSM_KEYS).per_feature
    # Like FeatureSpatialBinSummary.per_feature, instances in place of bins, with one population column in place of
    # four and only the mean and SD; with within_class_fraction of PointsSummary.per_feature after n_points.
    assert list(per_feature.columns) == [
        "feature",
        "feature_class",
        "n_instances",
        "n_points",
        "within_class_fraction",
        "n_instances_without_feature",
        "pct_instances_without_feature",
        "mean_points_per_instance",
        "std_points_per_instance",
    ]
    assert per_feature["n_points"].dtype == np.uint64


def test_feature_statistics_agree_with_every_row():
    summary = summarize_table(_table(), obsm_keys=_OBSM_KEYS)
    per_feature = summary.per_feature
    # Every feature of each class read, in class and then column order, undetected ones included.
    assert per_feature["feature"].tolist() == [name for names in _FEATURE_NAMES.values() for name in names]
    assert per_feature["feature_class"].tolist() == [label for label, names in _FEATURE_NAMES.items() for _ in names]
    assert summary.feature_names == _FEATURE_NAMES

    for label, dense in _CLASSES.items():
        dense = np.asarray(dense)
        rows = per_feature[per_feature["feature_class"] == label]
        np.testing.assert_array_equal(rows["n_points"], dense.sum(axis=0))
        np.testing.assert_allclose(rows["within_class_fraction"], dense.sum(axis=0) / dense.sum())
        # The stored zero of cell_2 for GeneB is not a detection.
        without = (dense == 0).sum(axis=0)
        np.testing.assert_array_equal(rows["n_instances_without_feature"], without)
        np.testing.assert_allclose(rows["pct_instances_without_feature"], 100 * without / len(dense))
        np.testing.assert_allclose(rows["mean_points_per_instance"], dense.mean(axis=0))
        np.testing.assert_allclose(rows["std_points_per_instance"], dense.std(axis=0, ddof=1))
    # A class's feature points add up to its points in per_class.
    class_points = per_feature.groupby("feature_class", observed=True)["n_points"].sum()
    np.testing.assert_array_equal(class_points.to_numpy(), summary.per_class["n_points"].to_numpy())
    # Every feature holds the population of per_class: the 6 rows.
    assert (per_feature["n_instances"] == 6).all()


def test_per_region_feature_holds_the_feature_statistics_within_each_region():
    summary = summarize_table(_table(), obsm_keys=_OBSM_KEYS)
    per_region_feature = summary.per_region_feature
    assert list(per_region_feature.columns) == ["region", *summary.per_feature.columns]
    n_features = len(summary.per_feature)
    assert per_region_feature["region"].tolist() == ["a"] * n_features + ["b"] * n_features
    for name in ["a", "b"]:
        within = summarize_table(_table(), obsm_keys=_OBSM_KEYS, region=name)
        # With one selected region, per_feature and per_region_feature hold the same values.
        pd.testing.assert_frame_equal(within.per_region_feature.drop(columns="region"), within.per_feature)
        pd.testing.assert_frame_equal(
            per_region_feature.loc[per_region_feature["region"] == name].drop(columns="region").reset_index(drop=True),
            within.per_feature,
        )
    # The regions' points add up to the pooled points.
    by_region = per_region_feature["n_points"].to_numpy().reshape(2, n_features)
    np.testing.assert_array_equal(by_region.sum(axis=0), summary.per_feature["n_points"].to_numpy())


def test_an_unregistered_obsm_matrix_takes_the_names_passed():
    table = _table()
    del table.uns["feature_matrices"]["negative"]
    names = ["N1", "N2", "N3", "N4"]
    summary = summarize_table(table, obsm_keys=_OBSM_KEYS, feature_names={"negative": names})
    assert summary.feature_names == {**_FEATURE_NAMES, "negative": tuple(names)}


def test_a_registry_under_a_custom_key_names_its_matrices():
    table = _table()
    table.uns["custom_feature_matrices"] = table.uns.pop("feature_matrices")
    # Under the default key, the matrices are unregistered.
    with pytest.raises(ValueError, match=r"not registered in uns\['feature_matrices'\]"):
        summarize_table(table, obsm_keys=_OBSM_KEYS)
    summary = summarize_table(table, obsm_keys=_OBSM_KEYS, feature_matrices_key="custom_feature_matrices")
    assert summary.metadata.feature_matrices_key == "custom_feature_matrices"
    assert summary.feature_names == _FEATURE_NAMES


def test_only_the_registrations_of_the_matrices_read_are_checked():
    table = _table()
    # A broken registration of a matrix that is not read does not stop the summary.
    table.uns["feature_matrices"]["unrelated"] = {"feature_columns": []}
    summary = summarize_table(table, obsm_keys=_OBSM_KEYS)
    assert summary.feature_names == _FEATURE_NAMES


def _without_registration(table):
    del table.uns["feature_matrices"]["negative"]


def _with_short_registration(table):
    table.uns["feature_matrices"]["negative"]["feature_columns"] = ["N1", "N2", "N3"]


@pytest.mark.parametrize(
    ("change", "kwargs", "message"),
    [
        (None, {"feature_names": {"negative": list(_FEATURE_NAMES["negative"])}}, "is registered in uns"),
        (None, {"obsm_keys": ["negative"], "feature_names": {"system_control": ["S"] * 5}}, "not read"),
        (_without_registration, {}, "has no column names"),
        (_without_registration, {"feature_names": {"negative": ["N1", "N2", "N3"]}}, "has 3 names, but"),
        (_without_registration, {"feature_names": {"negative": ["N1", "N1", "N2", "N3"]}}, "unique non-empty"),
        (_without_registration, {"feature_names": {"negative": []}}, "non-empty sequence"),
        (_with_short_registration, {}, "declares 3 feature columns"),
        (None, {"feature_names": ["negative"]}, "feature_names must map"),
        (None, {"feature_matrices_key": ""}, "Parameter 'feature_matrices_key'"),
        (None, {"feature_matrices_key": None}, "Parameter 'feature_matrices_key'"),
    ],
)
def test_invalid_feature_names_raise(change, kwargs, message):
    table = _table()
    if change is not None:
        change(table)
    arguments = {"obsm_keys": _OBSM_KEYS, **kwargs}
    with pytest.raises(ValueError, match=message):
        summarize_table(table, **arguments)


def test_csr_arrays_give_the_result_of_csr_matrices():
    expected = summarize_table(_table(), obsm_keys=_OBSM_KEYS)
    summary = summarize_table(_table(csr=sparse.csr_array), obsm_keys=_OBSM_KEYS)
    _assert_summaries_equal(summary, expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
def test_counts_stored_as_whole_floats_or_signed_integers_give_the_result_of_uint32(dtype):
    expected = summarize_table(_table(), obsm_keys=_OBSM_KEYS)
    summary = summarize_table(_table(csr=lambda matrix: matrix.astype(dtype)), obsm_keys=_OBSM_KEYS)
    _assert_summaries_equal(summary, expected)


@pytest.mark.parametrize("region", [None, "b"])
@pytest.mark.parametrize("mode", ["lazy", "backed"])
def test_lazy_and_backed_tables_give_the_in_memory_result(tmp_path, mode, region):
    SpatialData(tables={"counts": _table()}).write(tmp_path / "sdata.zarr")
    table = hp.io.read_zarr(tmp_path / "sdata.zarr", table_mode=mode, sparse_chunks=2).tables["counts"]
    if mode == "lazy":
        assert isinstance(table.X, da.Array)
        assert table.X.numblocks == (3, 1)
    else:
        assert isinstance(table.X, CSRDataset)
    expected = summarize_table(_table(), obsm_keys=_OBSM_KEYS, region=region)
    with _CountComputes() as computes:
        summary = summarize_table(table, obsm_keys=_OBSM_KEYS, region=region)
    # A lazy table runs one Dask compute for all its matrices; a backed table runs none.
    assert computes.count == (1 if mode == "lazy" else 0)
    _assert_summaries_equal(summary, expected)


def test_lazy_matrices_with_different_row_chunks_run_one_compute():
    table = _table()
    table.X = da.from_array(table.X, chunks=(2, -1), asarray=False)
    table.obsm["negative"] = da.from_array(table.obsm["negative"], chunks=(4, -1), asarray=False)
    # Column blocks are joined into whole rows.
    table.obsm["system_control"] = da.from_array(table.obsm["system_control"], chunks=(3, 2), asarray=False)
    expected = summarize_table(_table(), obsm_keys=_OBSM_KEYS)
    with _CountComputes() as computes:
        summary = summarize_table(table, obsm_keys=_OBSM_KEYS)
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
        summarize_table(table, obsm_keys=_OBSM_KEYS)
    assert computes.count == 0


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("value", [0.5, -1.0])
def test_matrices_whose_values_are_not_counts_raise(value, lazy):
    table = _table()
    matrix = table.obsm["negative"].astype(np.float64)
    matrix.data[0] = value
    table.obsm["negative"] = da.from_array(matrix, chunks=(2, -1), asarray=False) if lazy else matrix
    with pytest.raises(ValueError, match=r"adata\.obsm\['negative'\] holds values that are not"):
        summarize_table(table, obsm_keys=_OBSM_KEYS)


def _without_annotation(table):
    del table.uns["spatialdata_attrs"]


def _without_x(table):
    table.X = None


@pytest.mark.parametrize(
    ("change", "kwargs", "message"),
    [
        (_without_annotation, {}, "no SpatialData annotation"),
        (None, {"layer": "absent"}, "not a layer of the table"),
        (None, {"obsm_keys": ["absent"]}, "not an .obsm key of the table"),
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
    with pytest.raises(ValueError, match=message):
        summarize_table(table, **kwargs)


def test_summarize_table_takes_the_table():
    with pytest.raises(TypeError, match=r"summarize_table takes the table, an AnnData such as sdata\.tables\[name\]"):
        summarize_table(SpatialData(tables={"counts": _table()}))
