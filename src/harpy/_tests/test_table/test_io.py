from contextlib import contextmanager

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import zarr
from anndata import AnnData, read_zarr
from anndata.abc import CSCDataset, CSRDataset
from anndata.io import write_elem
from dask.callbacks import Callback
from scipy import sparse
from spatialdata import SpatialData
from spatialdata.models import TableModel
from zarr.storage import LocalStore

from harpy.table import read_table, read_table_components


@contextmanager
def _guard_reads(monkeypatch, *, annotations_only=False):
    """Reject matrix payload reads (even outside Dask) and any store writes.

    Complete lazy/backed reads may inspect matrix metadata. Annotation-only reads
    must not even open those matrix nodes or construct their Dask graphs.
    """
    original_get = LocalStore.get
    original_get_partial = LocalStore.get_partial_values
    matrix_roots = ("X", "layers", "obsm/embedding", "varm", "obsp", "varp", "raw/X", "raw/varm")

    def check_key(key):
        assert not key.startswith("tables/unrelated/")
        if any(key.startswith(f"tables/counts/{slot}/") for slot in matrix_roots):
            assert not annotations_only, f"Unrequested matrix metadata was read: {key}"
            assert key.rsplit("/", 1)[-1] in {".zarray", ".zattrs", ".zgroup", ".zmetadata", "zarr.json"}, key

    async def guarded_get(self, key, *args, **kwargs):
        assert self.read_only
        # Check the key before reading: matrix metadata is allowed (unless
        # annotations_only), but chunks containing matrix values are rejected.
        check_key(key)
        return await original_get(self, key, *args, **kwargs)

    async def guarded_get_partial(self, prototype, key_ranges):
        assert self.read_only
        key_ranges = list(key_ranges)
        for key, _ in key_ranges:
            check_key(key)
        return await original_get_partial(self, prototype, key_ranges)

    def unexpected_compute(graph):
        pytest.fail("Reading must not compute a Dask graph.")

    # Temporarily replace Zarr's read methods to reject matrix-payload reads;
    # exiting this context restores the original methods.
    with monkeypatch.context() as patch:
        patch.setattr(LocalStore, "get", guarded_get)
        patch.setattr(LocalStore, "get_partial_values", guarded_get_partial)
        # Register a callback that fails the test when a local Dask scheduler
        # starts computing. Constructing lazy graphs is allowed; executing
        # them invokes unexpected_compute() and fails the test.
        with Callback(start=unexpected_compute):
            # Both guards stay active while the caller executes its with-body.
            yield


def _assert_value(actual, expected):
    if isinstance(actual, da.Array):
        actual = actual.compute()
    elif isinstance(actual, (CSRDataset, CSCDataset)):
        actual = actual.to_memory()
    elif isinstance(actual, zarr.Array):
        actual = actual[:]
    if sparse.issparse(expected):
        assert sparse.issparse(actual)
        assert actual.format == expected.format
        np.testing.assert_array_equal(actual.toarray(), expected.toarray())
    elif isinstance(expected, pd.DataFrame):
        pd.testing.assert_frame_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_value(actual[key], expected[key])
    else:
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("matrix_kind", ["dense", "csr", "csc"])
@pytest.mark.parametrize("mode", ["lazy", "backed", "eager"])
def test_complete_read_preserves_all_slots_without_lazy_matrix_reads(
    make_table_io_store, monkeypatch, zarr_format, matrix_kind, mode
):
    path = make_table_io_store(zarr_format=zarr_format, matrix_kind=matrix_kind)
    expected = read_zarr(path / "tables" / "counts")
    if mode != "eager":
        with _guard_reads(monkeypatch):
            actual = read_table(path, table_name="counts", mode=mode, sparse_chunk_size=1)
    else:
        actual = read_table(path, table_name="counts", mode=mode, sparse_chunk_size=1)

    matrix_types = {
        "lazy": (da.Array,),
        "backed": (zarr.Array, CSRDataset, CSCDataset),
        "eager": (np.ndarray, sparse.spmatrix),
    }[mode]

    for slot in ("X", "layers", "obsm", "varm", "obsp", "varp", "obs", "var", "uns"):
        actual_value, expected_value = getattr(actual, slot), getattr(expected, slot)
        if slot in {"layers", "obsm", "varm", "obsp", "varp"}:
            actual_value, expected_value = dict(actual_value), dict(expected_value)
            assert all(isinstance(value, (*matrix_types, pd.DataFrame)) for value in actual_value.values())
        _assert_value(actual_value, expected_value)
    assert isinstance(actual.X, matrix_types)
    assert actual.raw.n_vars == 5 and actual.n_vars == 3
    assert isinstance(actual.raw.X, matrix_types)
    _assert_value(actual.raw.X, expected.raw.X)
    _assert_value(actual.raw.var, expected.raw.var)
    _assert_value(dict(actual.raw.varm), dict(expected.raw.varm))
    # Ordinary AnnData selection remains usable without first computing all X.
    subset = actual[[1], [0, 2]].copy()
    assert isinstance(subset.X, da.Array) == (mode == "lazy")
    _assert_value(subset.X, expected[[1], [0, 2]].X)


@pytest.mark.parametrize("matrix_kind", ["csr", "csc"])
@pytest.mark.parametrize("options, size", [({}, 1000), ({"sparse_chunk_size": 128}, 128)])
def test_sparse_chunk_size_controls_the_compressed_axis(tmp_path, monkeypatch, matrix_kind, options, size):
    """Both readers use Harpy's default or override, retaining the other axis whole."""
    path = tmp_path / "sdata.zarr"
    root = zarr.open_group(str(path), mode="w")
    matrix = getattr(sparse, f"{matrix_kind}_matrix")(([1, 2, 3], ([0, 128, 1002], [0, 129, 1004])), shape=(1003, 1005))
    write_elem(root.require_group("tables"), "counts", AnnData(X=matrix))
    with _guard_reads(monkeypatch):
        table = read_table(path, table_name="counts", **options)
        components = read_table_components(path, table_name="counts", components=[("X",)], **options)
    if matrix_kind == "csr":
        expected_chunks = ((size,) * (1003 // size) + (1003 % size,), (1005,))
    else:
        expected_chunks = ((1003,), (size,) * (1005 // size) + (1005 % size,))
    for value in (table.X, components[("X",)]):
        assert value.chunks == expected_chunks
        _assert_value(value, matrix)


def test_sparse_chunk_size_reaches_mapping_entries_and_raw(make_table_io_store, monkeypatch):
    path = make_table_io_store()
    with _guard_reads(monkeypatch):
        table = read_table(path, table_name="counts", sparse_chunk_size=np.int64(1))
        components = read_table_components(
            path, table_name="counts", components=[("layers",), ("varm", "loadings"), ("raw", "X")], sparse_chunk_size=1
        )
    assert table.layers["counts"].chunks == components[("layers",)]["counts"].chunks == ((1, 1), (3,))
    assert table.varm["loadings"].chunks == components[("varm", "loadings")].chunks == ((3,), (1, 1))
    assert table.raw.X.chunks == components[("raw", "X")].chunks == ((1, 1), (5,))
    assert table.obsp["neighbors"].chunks == ((1, 1), (2,))


def test_sparse_chunk_size_leaves_dense_disk_chunks_unchanged(make_table_io_store, monkeypatch):
    path = make_table_io_store(matrix_kind="dense")
    matrix = np.arange(6).reshape(2, 3)
    group = zarr.open_group(str(path), mode="r+")["tables/counts"]
    write_elem(group, "X", matrix, dataset_kwargs={"chunks": (1, 2)})
    with _guard_reads(monkeypatch):
        table = read_table(path, table_name="counts", sparse_chunk_size=1)
        components = read_table_components(path, table_name="counts", components=[("X",)], sparse_chunk_size=1)
    for value in (table.X, components[("X",)]):
        assert value.chunks == ((1, 1), (2, 1))
        _assert_value(value, matrix)
    assert group["X"].chunks == (1, 2)


@pytest.mark.parametrize("reader", [read_table, read_table_components])
@pytest.mark.parametrize(
    "size, error", [(True, TypeError), (1.5, TypeError), (None, TypeError), (0, ValueError), (-1, ValueError)]
)
def test_invalid_sparse_chunk_size_fails_before_opening_store(tmp_path, reader, size, error):
    options = {"components": [("X",)]} if reader is read_table_components else {}
    with pytest.raises(error, match="sparse_chunk_size must be a positive integer"):
        reader(tmp_path / "absent.zarr", table_name="counts", sparse_chunk_size=size, **options)


@pytest.mark.parametrize("mode", ["lazy", "backed", "eager"])
def test_annotation_only_read_is_selective_and_independently_owned(make_table_io_store, monkeypatch, mode):
    path = make_table_io_store()
    with _guard_reads(monkeypatch, annotations_only=True):
        values = read_table_components(
            str(path), table_name="counts", components=[("obs",), ("uns", "analysis"), ("obsm", "frame")], mode=mode
        )
        second = read_table_components(path, table_name="counts", components=[("obs",), ("uns",)])
    assert list(values) == [("obs",), ("uns", "analysis"), ("obsm", "frame")]
    assert isinstance(values[("uns", "analysis")]["values"], np.ndarray)
    values[("obs",)].loc["c1", "instance"] = 99
    values[("uns", "analysis")]["values"][0] = 99
    assert second[("obs",)].loc["c1", "instance"] == 1
    assert second[("uns",)]["analysis"]["values"][0] == 0
    fresh = read_table(path, table_name="counts")
    assert fresh.obs.loc["c1", "instance"] == 1
    assert fresh.uns["analysis"]["values"][0] == 0


@pytest.mark.parametrize("mode", ["lazy", "backed", "eager"])
def test_components_cover_matrix_mappings_and_raw(make_table_io_store, monkeypatch, mode):
    path = make_table_io_store()
    paths = [
        ("X",),
        ("var",),
        ("layers",),
        ("obsm",),
        ("varm",),
        ("obsp",),
        ("varp",),
        ("raw", "X"),
        ("raw", "var"),
        ("raw", "varm"),
        ("uns", "analysis", "values"),
    ]
    if mode != "eager":
        with _guard_reads(monkeypatch):
            values = read_table_components(path, table_name="counts", components=paths, mode=mode, sparse_chunk_size=1)
    else:
        values = read_table_components(path, table_name="counts", components=paths, mode=mode, sparse_chunk_size=1)
    expected = read_table(path, table_name="counts", mode="eager")
    for key in paths:
        target = expected.raw if key[0] == "raw" else expected
        slot = key[1] if key[0] == "raw" else key[0]
        if slot == "uns":
            _assert_value(values[key], expected.uns["analysis"]["values"])
        else:
            expected_value = getattr(target, slot)
            if slot in {"layers", "obsm", "varm", "obsp", "varp"}:
                expected_value = dict(expected_value)
            _assert_value(values[key], expected_value)
    matrix_type = {"lazy": da.Array, "backed": CSRDataset, "eager": sparse.csr_matrix}[mode]
    assert isinstance(values[("X",)], matrix_type)
    assert isinstance(values[("layers",)]["counts"], matrix_type)
    entries = read_table_components(
        path, table_name="counts", components=[("layers", "counts"), ("raw", "varm", "loadings")], mode=mode
    )
    _assert_value(entries[("layers", "counts")], expected.layers["counts"])
    _assert_value(entries[("raw", "varm", "loadings")], expected.raw.varm["loadings"])


def test_spatialdata_written_table_preserves_annotation_without_mutating_attached_table(tmp_path):
    """Read a SpatialData-written table with its region/instance annotation intact.

    Editing the result's obs/uns leaves the original attached table unchanged;
    assigning a new lazy X expression leaves the stored matrix unchanged.
    """
    table = TableModel.parse(
        AnnData(
            X=np.ones((2, 1)),
            obs=pd.DataFrame({"region": pd.Categorical(["cells", "cells"]), "instance": [1, 2]}, index=["c1", "c2"]),
        ),
        region=["cells"],
        region_key="region",
        instance_key="instance",
    )
    sdata = SpatialData(tables={"counts": table})
    path = tmp_path / "sdata.zarr"
    sdata.write(path)
    result = read_table(path, table_name="counts")
    TableModel.validate(result)
    assert result.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY] == ["cells"]
    result.obs.loc["c1", "instance"] = 9
    result.uns["note"] = "local"
    result.X = result.X * 2
    assert table.obs.loc["c1", "instance"] == 1
    assert "note" not in table.uns
    _assert_value(read_table(path, table_name="counts", mode="eager").X, np.ones((2, 1)))


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("mode", ["lazy", "backed", "eager"])
def test_missing_optional_slots_and_encoded_none_are_distinct(make_table_io_store, zarr_format, mode):
    path = make_table_io_store(zarr_format=zarr_format)
    group = zarr.open_group(str(path), mode="r+")["tables/counts"]
    for key in ("X", "raw", "layers", "obsm", "varm", "obsp", "varp", "uns"):
        del group[key]
    result = read_table(path, table_name="counts", mode=mode)
    assert result.shape == (2, 3) and result.X is None and result.raw is None
    assert not result.uns and not result.layers
    paths = [("X",), ("raw", "X"), ("uns", "absent")]
    assert read_table_components(path, table_name="counts", components=paths, mode=mode, missing="omit") == {}
    with pytest.raises(KeyError):
        read_table_components(path, table_name="counts", components=[("X",)], mode=mode)
    write_elem(group, "X", None)
    write_elem(group, "raw", None)
    assert read_table_components(path, table_name="counts", components=[("X",)], mode=mode, missing="omit") == {
        ("X",): None
    }
    result = read_table(path, table_name="counts", mode=mode)
    assert result.X is None and result.raw is None


@pytest.mark.parametrize(
    "components",
    [
        [],
        [("raw",)],
        [("obs", "column")],
        [("X", "data")],
        [("obsm", "embedding", "0")],
        [("uns", "../bad")],
        [("uns", "zarr.json")],
        [("uns", "")],
        [("obs",), ("obs",)],
        [("uns", "analysis"), ("uns",)],
    ],
)
def test_invalid_logical_paths_are_rejected(tmp_path, components):
    # Validation precedes storage access; no store is needed for an invalid request.
    with pytest.raises(ValueError):
        read_table_components(tmp_path / "absent.zarr", table_name="counts", components=components, missing="omit")


@pytest.mark.parametrize("path", [("uns", "analysis", "values", "0"), ("uns", "frame", "column")])
def test_missing_omit_does_not_traverse_arrays_or_dataframes(make_table_io_store, path):
    store = make_table_io_store()
    group = zarr.open_group(str(store), mode="r+")["tables/counts/uns"]
    write_elem(group, "frame", pd.DataFrame({"column": [1]}))
    with pytest.raises(ValueError, match="not a mapping"):
        read_table_components(store, table_name="counts", components=[path], missing="omit")


def test_missing_omit_does_not_hide_decoding_errors(make_table_io_store):
    store = make_table_io_store()
    group = zarr.open_group(str(store), mode="r+")["tables/counts/obs"]
    group.attrs["_index"] = "missing_index"
    with pytest.raises(KeyError, match="missing_index"):
        read_table_components(store, table_name="counts", components=[("obs",)], missing="omit")


def test_missing_stores_and_tables_are_not_created(tmp_path, make_table_io_store):
    absent = tmp_path / "absent.zarr"
    with pytest.raises(FileNotFoundError):
        read_table(absent, table_name="counts")
    assert not absent.exists()
    store = make_table_io_store()
    with pytest.raises(FileNotFoundError):
        read_table_components(store, table_name="missing", components=[("obs",)], missing="omit")
    with pytest.raises(ValueError, match="root"):
        read_table(store / "tables" / "counts", table_name="counts")
    with pytest.raises(ValueError, match="local"):
        read_table("s3://bucket/sdata.zarr", table_name="counts")
    with pytest.raises(ValueError, match="Invalid table name"):
        read_table(store, table_name="../counts")
    zarr.open_group(str(store), mode="r+").require_group("labels/counts")
    with pytest.raises(ValueError, match="collides"):
        read_table(store, table_name="counts")


@pytest.mark.parametrize("options", [{"components": "obs"}, {"components": [("obs", 1)]}, {"missing": None}])
def test_invalid_argument_types_raise_type_error(tmp_path, options):
    request = {"components": [("obs",)], **options}
    with pytest.raises(TypeError):
        read_table_components(tmp_path / "absent.zarr", table_name="counts", **request)


@pytest.mark.parametrize("shape", [(0, 3), (2, 0), (0, 0)])
@pytest.mark.parametrize("matrix_kind", ["dense", "csr", "csc"])
def test_empty_matrix_axes_remain_lazy(tmp_path, shape, matrix_kind):
    path = tmp_path / "sdata.zarr"
    root = zarr.open_group(str(path), mode="w")
    matrix = np.zeros(shape) if matrix_kind == "dense" else getattr(sparse, f"{matrix_kind}_matrix")(shape)
    write_elem(root.require_group("tables"), "empty", AnnData(X=matrix))
    table = read_table(path, table_name="empty")
    assert table.shape == shape
    assert isinstance(table.X, da.Array)
    _assert_value(table.X, matrix)


@pytest.mark.parametrize("mode", ["lazy", "backed", "eager"])
@pytest.mark.parametrize("matrix_kind", ["csr", "dense"])
@pytest.mark.parametrize("attribute, value", [("encoding-type", "unsupported"), ("encoding-version", "unsupported")])
def test_unsupported_matrix_encoding_fails_without_fallback(make_table_io_store, mode, matrix_kind, attribute, value):
    path = make_table_io_store(matrix_kind=matrix_kind)
    group = zarr.open_group(str(path), mode="r+")["tables/counts/X"]
    group.attrs[attribute] = value
    # obs is still readable without inspecting the broken expression encoding.
    assert len(read_table_components(path, table_name="counts", components=[("obs",)])[("obs",)]) == 2
    with pytest.raises(Exception, match="unsupported"):
        read_table(path, table_name="counts", mode=mode)


@pytest.mark.parametrize("reader", [read_table, read_table_components])
@pytest.mark.parametrize("mode, error", [(None, TypeError), (True, TypeError), ("unknown", ValueError)])
def test_invalid_mode_fails_before_opening_store(tmp_path, reader, mode, error):
    options = {"components": [("X",)]} if reader is read_table_components else {}
    with pytest.raises(error, match="mode must be"):
        reader(tmp_path / "absent.zarr", table_name="counts", mode=mode, **options)


@pytest.mark.parametrize("reader", [read_table, read_table_components])
@pytest.mark.parametrize("matrix_kind", ["dense", "csr"])
def test_backed_handles_reject_direct_writes(make_table_io_store, reader, matrix_kind):
    """Backed selects a representation, not write permission or AnnData file-backed mode."""
    path = make_table_io_store(matrix_kind=matrix_kind)
    if reader is read_table:
        table = reader(path, table_name="counts", mode="backed")
        assert not table.isbacked
        matrix = table.X
    else:
        matrix = reader(path, table_name="counts", components=[("X",)], mode="backed")[("X",)]
    with pytest.raises(ValueError, match="read.only"):
        matrix[0, 1] = 99
    expected = np.array([[0, 1, 2], [3, 0, 4]])
    if matrix_kind == "csr":
        expected = sparse.csr_matrix(expected)
    _assert_value(read_table(path, table_name="counts", mode="eager").X, expected)
