from contextlib import contextmanager

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import zarr
from anndata.abc import CSRDataset
from loguru import logger
from scipy import sparse
from spatialdata import read_zarr

from harpy.table._add_feature_matrix import add_feature_matrix
from harpy.table._allocation import aggregate_points
from harpy.table._concat import concat
from harpy.table._metadata import _merged_metadata, _metadata_difference
from harpy.table._validation import validate_table
from harpy.table.io import add_table, read_table
from harpy.utils._keys import _INSTANCE_KEY, _REGION_KEY

_PAIRS = (
    ("labels_a", "points_a", "sample_a", "table_a"),
    ("labels_b", "points_b", "sample_b", "table_b"),
)


@pytest.fixture
def aggregated_sdata(backed_class_aware_sdata):
    """Return ``backed_class_aware_sdata`` with one backed table per labels and points pair, in a store of its own.

    ``aggregate_points`` aggregates each pair class-aware into ``table_a`` (three
    instances) and ``table_b`` (one instance), which it attaches backed, and
    ``add_feature_matrix`` adds a morphology matrix to each, which it attaches
    lazily. Each test gets its own store under ``tmp_path`` and aggregates again,
    which is slow: use this fixture only in tests that write to the store, such
    as with ``add_table``, or that need the tables as ``aggregate_points``
    attaches them, backed. A copy of one shared store would not do for the
    latter: reopened, its tables load differently. Other tests use
    ``in_memory_tables``, which aggregates once per module.
    """
    return _aggregated(backed_class_aware_sdata)


@pytest.fixture(scope="module")
def aggregated_tables(make_class_aware_sdata, tmp_path_factory):
    """Return ``table_a`` and ``table_b`` of ``aggregated_sdata``, loaded into memory, aggregated once per module."""
    path = tmp_path_factory.mktemp("concat") / "input.zarr"
    make_class_aware_sdata().write(path)
    sdata = _aggregated(read_zarr(path))
    return [sdata.tables[name].to_memory() for name in ("table_a", "table_b")]


@pytest.fixture
def in_memory_tables(aggregated_tables):
    """Return copies of ``table_a`` and ``table_b``: ``AnnData.copy`` deep-copies ``uns``, so tests can change them."""
    return [table.copy() for table in aggregated_tables]


def _aggregated(sdata):
    """Aggregate each labels and points pair of ``sdata`` into its table, and add a morphology matrix to it."""
    for labels_name, points_name, coordinate_system, table_name in _PAIRS:
        sdata = aggregate_points(
            sdata,
            labels_name=labels_name,
            points_name=points_name,
            to_coordinate_system=coordinate_system,
            output_table_name=table_name,
            expression_class="Endogenous",
        )
        sdata = add_feature_matrix(
            sdata,
            labels_name=labels_name,
            image_name=None,
            table_name=table_name,
            feature_key="morphology",
            features=["area"],
            to_coordinate_system=coordinate_system,
        )
    return sdata


@contextmanager
def _concat_logs():
    """Collect the messages that ``harpy.table._concat`` logs."""
    records = []
    sink_id = logger.add(
        records.append, format="{message}", filter=lambda record: record["name"] == "harpy.table._concat"
    )
    try:
        yield records
    finally:
        logger.remove(sink_id)


def _messages(records) -> list[str]:
    return [record.record["message"] for record in records]


def test_concat_round_trip_after_per_cell_qc(aggregated_sdata):
    """Concatenated tables are stored with add_table and pass validate_table, their records merged."""
    sdata = aggregated_sdata
    # Per-cell QC subsets table_a, here in memory. Subsetting loads backed matrices but keeps lazy ones lazy,
    # such as the morphology matrix that add_feature_matrix attaches, so both tables are loaded first.
    table_a = sdata.tables["table_a"].to_memory()
    table_a = table_a[table_a.obs[_INSTANCE_KEY] != 3].copy()
    table_a.obsp["connectivities"] = sparse.csr_matrix((table_a.n_obs, table_a.n_obs))
    table_b = sdata.tables["table_b"].to_memory()

    with _concat_logs() as records:
        adata = concat({"a": table_a, "b": table_b}, table_key="sample")

    assert list(adata.obs_names) == [*table_a.obs_names, *table_b.obs_names]
    assert list(adata.obs["sample"]) == ["a", "a", "b"]
    assert adata.uns["spatialdata_attrs"] == {
        "region": ["labels_a", "labels_b"],
        "region_key": _REGION_KEY,
        "instance_key": _INSTANCE_KEY,
    }
    assert set(adata.uns["feature_matrices"]) == {"negative", "system_control", "morphology"}
    # Read from the store, as aggregated_sdata is backed: the morphology matrix has no image, which stays None.
    assert adata.uns["feature_matrices"]["morphology"]["regions"] == {
        "labels_a": {"image_element": None, "coordinate_system": "sample_a"},
        "labels_b": {"image_element": None, "coordinate_system": "sample_b"},
    }
    assert set(adata.uns["spatial_coordinates"]["spatial_canonical"]["regions"]) == {"labels_a", "labels_b"}
    assert adata.uns["point_aggregation"]["regions"] == {
        "labels_a": {"points_element": "points_a", "coordinate_system": "sample_a"},
        "labels_b": {"points_element": "points_b", "coordinate_system": "sample_b"},
    }
    assert sparse.issparse(adata.X)
    assert not adata.obsp
    assert any("drops obsp['connectivities'] of tables['a']" in message for message in _messages(records))

    sdata = add_table(sdata, adata, output_table_name="table", region=adata.uns["spatialdata_attrs"]["region"])
    validate_table(sdata, "table")


@pytest.mark.parametrize("inputs", ["backed", "lazy", "backed and lazy"])
def test_concat_keeps_lazy_tables_lazy(aggregated_sdata, inputs):
    """Backed and lazy matrices give lazy ones; a matrix in memory in every table stays in memory."""
    sdata = aggregated_sdata
    lazy = [read_table(sdata.path, table_name=name, mode="lazy") for name in ("table_a", "table_b")]
    backed = [sdata.tables["table_a"], sdata.tables["table_b"]]
    tables = {"backed": backed, "lazy": lazy, "backed and lazy": [backed[0], lazy[1]]}[inputs]
    for table in tables:
        table.obsm["embedding"] = np.ones((table.n_obs, 2))

    adata = concat(tables)

    assert isinstance(adata.X, da.Array)
    assert sparse.issparse(adata.X._meta) and adata.X._meta.format == "csr"
    for key in ("negative", "system_control", "spatial_canonical", "morphology"):
        assert isinstance(adata.obsm[key], da.Array)
    assert isinstance(adata.obsm["embedding"], np.ndarray)
    expected = [table.to_memory() for table in tables]
    np.testing.assert_array_equal(adata.X.compute().toarray(), np.vstack([table.X.toarray() for table in expected]))
    np.testing.assert_array_equal(
        adata.obsm["spatial_canonical"].compute(), np.vstack([table.obsm["spatial_canonical"] for table in expected])
    )
    assert isinstance(sdata.tables["table_a"].X, CSRDataset)
    assert isinstance(sdata.tables["table_a"].obsm["spatial_canonical"], zarr.Array)


def test_concat_raises_for_a_mismatched_feature_matrix_and_drops_it_on_request(in_memory_tables):
    table_a, table_b = in_memory_tables
    # For example, a per-sample selection of negative probes keeps two probes in table_b.
    table_b.obsm["negative"] = table_b.obsm["negative"][:, :2]
    table_b.uns["feature_matrices"]["negative"]["feature_columns"] = ["Negative01", "Negative02"]

    with pytest.raises(
        ValueError,
        match=r"'negative' has metadata that differs in \['feature_columns'\] between tables\[0\] and tables\[1\]\. "
        r".*drop_mismatched_feature_matrices=True",
    ):
        concat([table_a, table_b])

    with _concat_logs() as records:
        adata = concat([table_a, table_b], drop_mismatched_feature_matrices=True)

    assert "negative" not in adata.obsm
    assert set(adata.uns["feature_matrices"]) == {"system_control", "morphology"}
    assert any("drops the feature matrices ['negative']" in message for message in _messages(records))


def test_concat_merges_harpys_records_whatever_uns_merge(in_memory_tables):
    """``uns_merge`` merges only the entries Harpy does not own; those it drops are logged."""
    table_a, table_b = in_memory_tables
    table_a.uns["qc"] = {"a": {"threshold": 1}}
    table_b.uns["qc"] = {"b": {"threshold": 2}}

    adata = concat({"a": table_a, "b": table_b}, uns_merge="unique")

    assert adata.uns["qc"] == {"a": {"threshold": 1}, "b": {"threshold": 2}}
    # "unique" would drop the declared regions and the metadata of each region, on which the tables disagree.
    assert adata.uns["spatialdata_attrs"]["region"] == ["labels_a", "labels_b"]
    assert set(adata.uns["feature_matrices"]["morphology"]["regions"]) == {"labels_a", "labels_b"}

    with _concat_logs() as records:
        adata = concat({"a": table_a, "b": table_b})

    assert "qc" not in adata.uns
    assert any("drops the uns entries ['qc']" in message for message in _messages(records))


def test_concat_leaves_out_the_entries_of_regions_a_table_does_not_declare(in_memory_tables):
    """A subset to fewer regions keeps their entries in its records; concat leaves them out and logs it."""
    merged = concat(in_memory_tables)
    subset = merged[merged.obs[_REGION_KEY] == "labels_a"].copy()
    subset.uns["spatialdata_attrs"]["region"] = ["labels_a"]
    assert "labels_b" in subset.uns["point_aggregation"]["regions"]

    with _concat_logs() as records:
        adata = concat({"subset": subset})

    assert list(adata.uns["spatial_coordinates"]["spatial_canonical"]["regions"]) == ["labels_a"]
    assert list(adata.uns["point_aggregation"]["regions"]) == ["labels_a"]
    assert list(adata.uns["feature_matrices"]["morphology"]["regions"]) == ["labels_a"]
    assert any(
        "leaves out the entries of the regions ['labels_b'] in uns['spatial_coordinates']['spatial_canonical'], "
        "uns['point_aggregation'], uns['feature_matrices']['morphology'] of tables['subset']" in message
        for message in _messages(records)
    )


def test_merged_metadata_takes_the_entries_of_the_declared_regions():
    """Each table contributes the entries of the regions it declares; the other fields must be equal."""
    # The first table was subset to cells_b, but its metadata still holds cells_a; read from storage, it holds arrays.
    subset = {
        "feature_columns": np.array(["mean__0"]),
        "regions": {"cells_a": {"image_element": "image_a"}, "cells_b": {"image_element": "image_b"}},
    }
    other = {"feature_columns": ["mean__0"], "regions": {"cells_c": {"image_element": None}}}

    assert _metadata_difference([subset, other]) is None
    merged = _merged_metadata(
        [subset, other], declared_regions=[("cells_b",), ("cells_c",)], region_order=["cells_c", "cells_b"]
    )

    assert merged["regions"] == {"cells_b": {"image_element": "image_b"}, "cells_c": {"image_element": None}}
    assert set(subset["regions"]) == {"cells_a", "cells_b"}
    assert _metadata_difference([subset, other, {**other, "feature_columns": ["mean__1"]}]) == (2, ["feature_columns"])
    # Metadata without regions, such as that of a class matrix of aggregate_points, is compared and kept whole.
    whole = {"feature_columns": ["Negative01"]}
    assert _merged_metadata([whole, whole], declared_regions=[("cells_b",), ("cells_c",)], region_order=[]) == whole


@pytest.mark.parametrize("uns_merge", [None, "same", "unique", "first", "only"])
def test_concat_merges_colors_by_category_name(in_memory_tables, uns_merge):
    table_a, table_b = in_memory_tables
    # Categories differ between the tables; "T" is unused in table_b.
    table_a.obs["cell_type"] = pd.Categorical(["B", "T", "B"], categories=["B", "T"])
    table_b.obs["cell_type"] = pd.Categorical(["M"], categories=["M", "T"])
    table_a.uns["cell_type_colors"] = np.array(["red", "blue"], dtype=object)
    table_b.uns["cell_type_colors"] = np.array(["green", "blue"], dtype=object)

    adata = concat([table_a, table_b], uns_merge=uns_merge)

    categories = list(adata.obs["cell_type"].cat.categories)
    assert sorted(categories) == ["B", "M", "T"]
    assert dict(zip(categories, adata.uns["cell_type_colors"], strict=True)) == {"B": "red", "M": "green", "T": "blue"}


def test_concat_warns_about_conflicting_and_missing_colors(in_memory_tables):
    table_a, table_b = in_memory_tables
    table_a.obs["cell_type"] = pd.Categorical(["B", "T", "B"], categories=["B", "T"])
    table_b.obs["cell_type"] = pd.Categorical(["M"], categories=["M", "T"])
    table_a.uns["cell_type_colors"] = ["red", "blue"]
    table_b.uns["cell_type_colors"] = ["green", "yellow"]

    with _concat_logs() as records:
        adata = concat([table_a, table_b])

    categories = list(adata.obs["cell_type"].cat.categories)
    assert dict(zip(categories, adata.uns["cell_type_colors"], strict=True))["T"] == "blue"
    assert any("'T' is 'blue' in tables[0] and 'yellow' in tables[1]" in message for message in _messages(records))

    # table_b holds "M" without a palette, so "M" gets no color.
    del table_b.uns["cell_type_colors"]
    with _concat_logs() as records:
        adata = concat([table_a, table_b])

    assert "cell_type_colors" not in adata.uns
    assert any("the categories ['M'] of obs['cell_type'] have no color" in message for message in _messages(records))


def _zarr_array_without_encoding(table_a, table_b):
    table_a.obsm["plain"] = zarr.create_array(store=zarr.storage.MemoryStore(), shape=(table_a.n_obs, 2), dtype="f8")
    table_b.obsm["plain"] = np.zeros((table_b.n_obs, 2))
    return [table_a, table_b], {}


def _shared_region(table_a, table_b):
    return [table_a, table_a], {}


def _other_var_names(table_a, table_b):
    return [table_a, table_b[:, ["GeneA", "GeneB"]].copy()], {}


def _other_obs_columns(table_a, table_b):
    table_a.obs["n_counts"] = 1
    return [table_a, table_b], {}


def _other_layers(table_a, table_b):
    table_a.layers["counts"] = table_a.X.copy()
    return [table_a, table_b], {}


def _other_obsm_keys(table_a, table_b):
    table_a.obsm["X_pca"] = np.zeros((table_a.n_obs, 2))
    return [table_a, table_b], {}


def _other_obsm_widths(table_a, table_b):
    table_a.obsm["X_pca"] = np.zeros((table_a.n_obs, 2))
    table_b.obsm["X_pca"] = np.zeros((table_b.n_obs, 3))
    return [table_a, table_b], {}


def _signed_instance_ids(table_a, table_b):
    table_a.obs[_INSTANCE_KEY] = table_a.obs[_INSTANCE_KEY].astype("int64")
    return [table_a, table_b], {}


def _bool_and_int_column(table_a, table_b):
    table_a.obs["flag"] = True
    table_b.obs["flag"] = 1
    return [table_a, table_b], {}


def _csr_and_csc(table_a, table_b):
    table_b.X = table_b.X.tocsc()
    return [table_a, table_b], {}


def _lazy_and_in_memory(table_a, table_b):
    table_b.obsm["spatial_canonical"] = da.from_array(table_b.obsm["spatial_canonical"])
    return [table_a, table_b], {}


def _table_key_is_an_obs_column(table_a, table_b):
    return [table_a, table_b], {"table_key": _INSTANCE_KEY}


def _duplicate_obs_names(table_a, table_b):
    table_b.obs_names = [table_a.obs_names[0]]
    return [table_a, table_b], {}


def _canonical_centers_in_one_table(table_a, table_b):
    del table_b.obsm["spatial_canonical"]
    del table_b.uns["spatial_coordinates"]
    return [table_a, table_b], {}


def _canonical_matrix_without_metadata(table_a, table_b):
    del table_b.uns["spatial_coordinates"]
    return [table_a, table_b], {}


def _feature_metadata_of_version_1(table_a, table_b):
    # Metadata as Harpy 0.5 wrote it: schema version 1, with parallel lists in place of regions.
    record = table_b.uns["feature_matrices"]["morphology"]
    del record["regions"]
    record.update(schema_version=1, source_label=["labels_b"], source_image=[None], coordinate_system=["sample_b"])
    return [table_a, table_b], {}


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (_zarr_array_without_encoding, r"obsm\['plain'\] of tables\[0\] is a Zarr array without AnnData encoding"),
        (_shared_region, r"Each region must be in one table only.*'labels_a' in tables\[0\], tables\[1\]"),
        (_other_var_names, r"var_names differ: tables\[0\] has \['GeneZero'\] that tables\[1\] lacks"),
        (_other_obs_columns, r"obs columns must be the same in every table: tables\[1\] lacks \['n_counts'\]"),
        (_other_layers, r"layers must be the same in every table: tables\[1\] lacks \['counts'\]"),
        (_other_obsm_keys, r"obsm keys must be the same in every table: tables\[1\] lacks \['X_pca'\]"),
        (
            _other_obsm_widths,
            r"obsm\['X_pca'\] has shape\[1:\] \(2,\) in tables\[0\] and shape\[1:\] \(3,\) in tables\[1\]",
        ),
        (_signed_instance_ids, r"obs column 'cell_ID' is int64 in tables\[0\] and uint64 in tables\[1\]; cast it"),
        (_bool_and_int_column, r"obs column 'flag' is bool in tables\[0\] and int64 in tables\[1\]; cast it"),
        (_csr_and_csc, r"X is CSR in tables\[0\] and CSC in tables\[1\]"),
        (
            _lazy_and_in_memory,
            r"obsm\['spatial_canonical'\] is lazy in tables\[1\] and in memory in tables\[0\].*to_memory\(\)",
        ),
        (_table_key_is_an_obs_column, r"table_key 'cell_ID' is an obs column of tables\[0\], tables\[1\]"),
        (_duplicate_obs_names, r"obs_names must be unique across the tables"),
        (_canonical_centers_in_one_table, r"Canonical centers are in tables\[0\] but not in tables\[1\]"),
        (
            _canonical_matrix_without_metadata,
            r"tables\[1\]: The table has obsm\['spatial_canonical'\] but not "
            r"uns\['spatial_coordinates'\]\['spatial_canonical'\]; canonical centers need both\. Rebuild them",
        ),
        (
            _feature_metadata_of_version_1,
            r"tables\[1\]: Feature-matrix metadata feature_matrices\.morphology\.schema_version must equal 2, found 1\. "
            r"Compute the matrix again",
        ),
    ],
)
def test_concat_raises_for_tables_of_different_structure(in_memory_tables, change, match):
    tables, kwargs = change(*in_memory_tables)

    with pytest.raises(ValueError, match=match):
        concat(tables, **kwargs)
