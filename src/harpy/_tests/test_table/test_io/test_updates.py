"""The comparison helpers that decide whether a component of a table changed against its store."""

import dask
import numpy as np
import pandas as pd
import pytest
import zarr
from anndata import AnnData
from anndata.io import sparse_dataset, write_elem
from dask.callbacks import Callback
from scipy import sparse
from spatialdata import SpatialData
from spatialdata.models import TableModel
from zarr.storage import MemoryStore

from harpy._storage._anndata import _decode_anndata_element, _read_anndata_element
from harpy.table.io import read_table, write_table
from harpy.table.io._read import _open_table_group
from harpy.table.io._updates import _component_changed, _matrix_equals_stored, _same_dataframe, _same_uns_value

_VALUES = np.arange(24, dtype=np.float32).reshape(6, 4)
_FORMATS = {"dense": np.asarray, "csr": sparse.csr_matrix, "csc": sparse.csc_matrix}


class _CountComputes(Callback):
    """Count Dask computes; the matrix comparison computes one stored block per compute."""

    def __init__(self):
        super().__init__()
        self.count = 0

    def _start(self, dsk):
        self.count += 1


@pytest.fixture
def table_store(tmp_path):
    """A SpatialData store with one table, ``counts``, holding every kind of component."""
    path = tmp_path / "sdata.zarr"
    SpatialData().write(path)
    write_table(path, table_name="counts", adata=_table())
    return path


def _table():
    obs_names = [str(i) for i in range(6)]
    obs = pd.DataFrame(
        {
            "region": pd.Categorical(["cells"] * 6),
            "instance": np.arange(1, 7),
            "cluster": pd.Categorical(["a", "b", "a", "b", "a", "b"], categories=["a", "b"]),
            "score": np.array([0.5, np.nan, 1.0, 2.0, np.nan, 3.0]),
        },
        index=obs_names,
    )
    adata = AnnData(
        X=sparse.csr_matrix(_VALUES),
        obs=obs,
        var=pd.DataFrame({"highly_variable": [True, False, True, False]}, index=list("abcd")),
        layers={"counts": sparse.csr_matrix(_VALUES), "dense": _VALUES.copy()},
        obsm={"dense": _VALUES[:, :2].copy(), "frame": pd.DataFrame({"x": np.arange(6.0)}, index=obs_names)},
        uns={"pca": {"variance": np.array([1.5, np.nan]), "params": {"n_comps": 2, "solver": "arpack"}}},
    )
    return TableModel.parse(adata, region="cells", region_key="region", instance_key="instance")


def _stored_element(tmp_path, matrix, *, dense_row_chunks=None):
    """Write one matrix to its own local Zarr store and return its element."""
    group = zarr.open_group(str(tmp_path / "matrix.zarr"), mode="w")
    dataset_kwargs = {} if dense_row_chunks is None else {"chunks": (dense_row_chunks, matrix.shape[1])}
    write_elem(group, "matrix", matrix, dataset_kwargs=dataset_kwargs)
    return group["matrix"]


# The dataframe rule: obs, var, raw.var and DataFrame-valued obsm/varm entries.


def _frame():
    return pd.DataFrame(
        {"cluster": pd.Categorical(["a", "b", "a"], categories=["a", "b"]), "score": [0.5, np.nan, 1.0]},
        index=["x", "y", "z"],
    )


def test_dataframes_with_equal_values_and_nan_are_the_same():
    assert _same_dataframe(_frame(), _frame())


def test_dataframes_without_columns_are_the_same_whatever_the_class_of_their_columns_index():
    """AnnData turns the empty RangeIndex of such a dataframe into an empty object index."""
    stored = pd.DataFrame(index=["a", "b"])
    value = AnnData(var=stored.copy()).var
    assert type(value.columns) is not type(stored.columns)
    assert _same_dataframe(value, stored)


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(lambda frame: frame.assign(extra=1), id="new column"),
        pytest.param(lambda frame: frame[["score", "cluster"]], id="column order"),
        pytest.param(lambda frame: frame.assign(score=frame["score"].astype("float32")), id="dtype"),
        pytest.param(
            lambda frame: frame.assign(cluster=frame["cluster"].cat.reorder_categories(["b", "a"])),
            id="category order",
        ),
        pytest.param(lambda frame: frame.assign(score=[0.5, np.nan, 1.5]), id="value"),
        pytest.param(lambda frame: frame.set_axis(["x", "y", "w"]), id="index"),
    ],
)
def test_dataframes_differ_on_any_column_dtype_category_value_or_index_change(change):
    assert not _same_dataframe(change(_frame()), _frame())


def test_obs_var_and_dataframes_in_obsm_read_back_from_the_store_are_unchanged(table_store):
    group = _open_table_group(table_store, table_name="counts")
    adata = read_table(table_store, table_name="counts", mode="lazy")
    assert not _component_changed(group, ("obs",), adata.obs)
    assert not _component_changed(group, ("var",), adata.var)
    assert not _component_changed(group, ("obsm", "frame"), adata.obsm["frame"])


def test_changed_dataframes_are_changed(table_store):
    group = _open_table_group(table_store, table_name="counts")
    adata = read_table(table_store, table_name="counts", mode="lazy")
    adata.obs["leiden"] = pd.Categorical(["0", "1", "0", "1", "0", "1"])
    assert _component_changed(group, ("obs",), adata.obs)
    assert _component_changed(group, ("obsm", "frame"), adata.obsm["frame"] * 2)
    # A dataframe replacing a matrix, and the reverse, are changed.
    assert _component_changed(group, ("obsm", "dense"), pd.DataFrame(_VALUES[:, :2], index=adata.obs_names))
    assert _component_changed(group, ("obsm", "frame"), adata.obsm["frame"].to_numpy())


# The uns comparator, per top-level key.


@pytest.mark.parametrize(
    ("value", "stored"),
    [
        pytest.param({"params": {"n": 2, "zero": True}}, {"params": {"n": np.int64(2), "zero": np.True_}}, id="nested"),
        pytest.param(np.array([1.0, np.nan]), np.array([1.0, np.nan]), id="nan"),
        pytest.param([1, 2, 3], np.array([1, 2, 3]), id="list and stored array"),
        pytest.param(["a", "b"], np.array(["a", "b"], dtype=object), id="strings"),
        pytest.param(None, None, id="none"),
        pytest.param(_frame(), _frame(), id="dataframe"),
    ],
)
def test_uns_values_equal_by_value_are_the_same(value, stored):
    assert _same_uns_value(value, stored)


@pytest.mark.parametrize(
    ("value", "stored"),
    [
        pytest.param({"params": {"n": 3}}, {"params": {"n": 2}}, id="nested value"),
        pytest.param({"params": {"n": 2, "extra": 1}}, {"params": {"n": 2}}, id="nested key added"),
        pytest.param({"params": {}}, {"params": {"n": 2}}, id="nested key removed"),
        pytest.param({"n": 2}, 2, id="mapping and scalar"),
        pytest.param([1, 2], [1, 2, 3], id="shape"),
        pytest.param(None, 0, id="none and zero"),
        pytest.param(_frame().assign(extra=1), _frame(), id="dataframe"),
        pytest.param(_frame(), _frame().to_numpy(), id="dataframe and array"),
        pytest.param([[1], [1, 2]], [[1], [1, 2]], id="ragged, comparison raises"),
    ],
)
def test_uns_values_that_differ_or_cannot_be_compared_are_different(value, stored):
    assert not _same_uns_value(value, stored)


def test_uns_entries_read_back_from_the_store_are_unchanged_until_a_nested_value_changes(table_store):
    group = _open_table_group(table_store, table_name="counts")
    adata = read_table(table_store, table_name="counts", mode="lazy")
    assert not _component_changed(group, ("uns", "pca"), adata.uns["pca"])
    assert not _component_changed(group, ("uns", TableModel.ATTRS_KEY), adata.uns[TableModel.ATTRS_KEY])
    adata.uns["pca"]["params"]["n_comps"] = 3
    assert _component_changed(group, ("uns", "pca"), adata.uns["pca"])


# Identity: registered lazy reads and backed handles, decided without reading values.


def test_a_lazy_read_of_the_element_itself_is_unchanged(table_store):
    group = _open_table_group(table_store, table_name="counts")
    adata = read_table(table_store, table_name="counts", mode="lazy")
    with _CountComputes() as computes:
        assert not _component_changed(group, ("X",), adata.X)
        assert not _component_changed(group, ("layers", "dense"), adata.layers["dense"])
    assert computes.count == 0


def test_a_lazy_read_of_another_element_is_changed_even_with_equal_values(table_store):
    """layers["counts"] holds the same values as X, but a read of X is not a read of the layer."""
    group = _open_table_group(table_store, table_name="counts")
    adata = read_table(table_store, table_name="counts", mode="lazy")
    assert _component_changed(group, ("layers", "counts"), adata.X)


def test_derived_dask_arrays_are_changed_without_computing_them(table_store):
    group = _open_table_group(table_store, table_name="counts")
    adata = read_table(table_store, table_name="counts", mode="lazy")
    with _CountComputes() as computes:
        assert _component_changed(group, ("X",), adata.X * 1)
        assert _component_changed(group, ("layers", "dense"), adata.layers["dense"].rechunk({0: 1}))
    assert computes.count == 0


def test_a_backed_handle_is_unchanged_only_for_its_own_element(table_store):
    group = _open_table_group(table_store, table_name="counts")
    adata = read_table(table_store, table_name="counts", mode="backed")
    assert not _component_changed(group, ("X",), adata.X)
    assert not _component_changed(group, ("layers", "dense"), adata.layers["dense"])
    assert _component_changed(group, ("layers", "counts"), adata.X)


def test_a_backed_handle_is_recognised_wherever_its_store_was_opened(table_store):
    """Handles opened at the table or at the element itself point to the same files as Harpy's reads."""
    group = _open_table_group(table_store, table_name="counts")
    table_folder = table_store / "tables" / "counts"
    from_table = sparse_dataset(zarr.open_group(str(table_folder), mode="r")["X"])
    from_element = sparse_dataset(zarr.open_group(str(table_folder / "X"), mode="r"))
    dense_from_element = zarr.open_array(str(table_folder / "layers" / "dense"), mode="r")
    assert not _component_changed(group, ("X",), from_table)
    assert not _component_changed(group, ("X",), from_element)
    assert not _component_changed(group, ("layers", "dense"), dense_from_element)
    assert _component_changed(group, ("layers", "counts"), from_element)


def test_reads_and_handles_of_another_store_are_changed(table_store, tmp_path):
    other = tmp_path / "other.zarr"
    SpatialData().write(other)
    write_table(other, table_name="counts", adata=_table())
    group = _open_table_group(table_store, table_name="counts")
    assert _component_changed(group, ("X",), read_table(other, table_name="counts", mode="lazy").X)
    assert _component_changed(group, ("X",), read_table(other, table_name="counts", mode="backed").X)


def test_elements_in_stores_other_than_a_local_store_have_no_identity():
    """Reads and handles count as changed there, while in-memory values are still compared."""
    group = zarr.open_group(MemoryStore(), mode="w")
    write_elem(group, "table", AnnData(X=sparse.csr_matrix(_VALUES)))
    table = group["table"]
    assert _component_changed(table, ("X",), _read_anndata_element(table, ("X",), mode="lazy"))
    assert _component_changed(table, ("X",), _read_anndata_element(table, ("X",), mode="backed"))
    assert not _component_changed(table, ("X",), sparse.csr_matrix(_VALUES))


# In-memory matrices: metadata first, then block by block.


@pytest.fixture
def small_blocks():
    """Lazy reads of the small test matrices in blocks of one row (dense, CSR) or one column (CSC)."""
    with dask.config.set({"array.chunk-size": "16B"}):
        yield


def _element_and_blocks(tmp_path, kind):
    element = _stored_element(tmp_path, _FORMATS[kind](_VALUES), dense_row_chunks=1 if kind == "dense" else None)
    axis = 1 if kind == "csc" else 0
    return element, _decode_anndata_element(element, mode="lazy").numblocks[axis]


@pytest.mark.usefixtures("small_blocks")
@pytest.mark.parametrize("kind", list(_FORMATS))
def test_equal_matrices_are_unchanged_after_reading_every_block(tmp_path, kind):
    element, n_blocks = _element_and_blocks(tmp_path, kind)
    assert n_blocks > 2
    with _CountComputes() as computes:
        assert _matrix_equals_stored(_FORMATS[kind](_VALUES), element)
    assert computes.count == n_blocks


@pytest.mark.usefixtures("small_blocks")
@pytest.mark.parametrize("kind", list(_FORMATS))
def test_a_value_change_is_found_at_its_block_without_reading_further(tmp_path, kind):
    element, n_blocks = _element_and_blocks(tmp_path, kind)
    value = _FORMATS[kind](_VALUES.copy())
    # A stored value in block 2: row 2 for dense and CSR, column 2 for CSC.
    row, column = (1, 2) if kind == "csc" else (2, 1)
    value[row, column] += 1
    with _CountComputes() as computes:
        assert not _matrix_equals_stored(value, element)
    assert computes.count == 3 < n_blocks


@pytest.mark.parametrize(
    ("stored", "value"),
    [
        pytest.param(_VALUES, _VALUES.astype(np.float64), id="dense dtype"),
        pytest.param(_VALUES, _VALUES[:5], id="dense shape"),
        pytest.param(sparse.csr_matrix(_VALUES), sparse.csr_matrix(_VALUES.astype(np.float64)), id="sparse dtype"),
        pytest.param(sparse.csr_matrix(_VALUES), sparse.csr_matrix(_VALUES[:, :3]), id="sparse shape"),
        pytest.param(sparse.csr_matrix(_VALUES), sparse.csc_matrix(_VALUES), id="csr and csc"),
        pytest.param(sparse.csr_matrix(_VALUES), _VALUES, id="sparse and dense"),
        pytest.param(_VALUES, sparse.csr_matrix(_VALUES), id="dense and sparse"),
        pytest.param(sparse.csr_matrix(_VALUES), sparse.csr_matrix(_VALUES + 1), id="number of stored values"),
    ],
)
def test_metadata_changes_are_found_without_reading_values(tmp_path, stored, value):
    element = _stored_element(tmp_path, stored)
    with _CountComputes() as computes:
        assert not _matrix_equals_stored(value, element)
    assert computes.count == 0


@pytest.mark.parametrize("kind", list(_FORMATS))
def test_nan_equals_nan(tmp_path, kind):
    values = _VALUES.copy()
    values[1, 1] = np.nan
    element = _stored_element(tmp_path, _FORMATS[kind](values))
    assert _matrix_equals_stored(_FORMATS[kind](values), element)


def test_sparse_matrices_are_compared_in_canonical_form(tmp_path):
    """The same entries in another order within a row are equal."""
    element = _stored_element(tmp_path, sparse.csr_matrix(np.array([[1.0, 2.0], [0.0, 3.0]])))
    unsorted = sparse.csr_matrix((np.array([2.0, 1.0, 3.0]), np.array([1, 0, 1]), np.array([0, 2, 3])), shape=(2, 2))
    assert not unsorted.has_sorted_indices
    assert _matrix_equals_stored(unsorted, element)


def test_a_difference_in_explicit_zeros_only_is_changed(tmp_path):
    """Equal as dense matrices, with as many stored values, but an explicit zero at another position."""
    stored = sparse.csr_matrix((np.array([1.0, 0.0, 2.0]), np.array([0, 1, 1]), np.array([0, 2, 3])), shape=(2, 2))
    value = sparse.csr_matrix((np.array([1.0, 0.0, 2.0]), np.array([0, 0, 1]), np.array([0, 1, 3])), shape=(2, 2))
    np.testing.assert_array_equal(stored.toarray(), value.toarray())
    element = _stored_element(tmp_path, stored)
    # The metadata check passes, so the blocks are what differ.
    assert element["data"].shape[0] == value.nnz
    assert not _matrix_equals_stored(value, element)


@pytest.mark.parametrize(
    "value",
    [pytest.param(_VALUES.tolist(), id="list"), pytest.param(sparse.coo_matrix(_VALUES), id="coo")],
)
def test_values_the_comparison_does_not_handle_are_changed_even_with_equal_values(table_store, value):
    group = _open_table_group(table_store, table_name="counts")
    assert _component_changed(group, ("X",), value)


@pytest.mark.parametrize(
    "stored",
    [pytest.param(_VALUES.astype(str), id="stored strings"), pytest.param(_VALUES, id="stored numbers")],
)
def test_string_arrays_differ_at_the_metadata_step(tmp_path, stored):
    element = _stored_element(tmp_path, stored)
    with _CountComputes() as computes:
        assert not _matrix_equals_stored(_VALUES.astype(str), element)
    assert computes.count == 0


def test_an_eager_table_read_back_from_the_store_is_unchanged(table_store):
    group = _open_table_group(table_store, table_name="counts")
    adata = read_table(table_store, table_name="counts", mode="eager")
    for path, value in [
        (("X",), adata.X),
        (("layers", "counts"), adata.layers["counts"]),
        (("layers", "dense"), adata.layers["dense"]),
        (("obsm", "dense"), adata.obsm["dense"]),
    ]:
        assert not _component_changed(group, path, value), path
    changed = adata.layers["dense"].copy()
    changed[0, 0] += 1
    assert _component_changed(group, ("layers", "dense"), changed)
