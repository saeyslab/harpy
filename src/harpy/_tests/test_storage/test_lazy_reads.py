"""The registry of lazy reads, which recognises a stored matrix not changed since it was read."""

import pickle

import numpy as np
import pandas as pd
import pytest
import zarr
from anndata import AnnData
from anndata.experimental import read_elem_lazy
from anndata.io import write_elem
from scipy import sparse
from spatialdata import SpatialData
from spatialdata.models import TableModel
from zarr.storage import MemoryStore

import harpy._storage._anndata as anndata_storage
from harpy._storage._anndata import _decode_anndata_element, _lazy_read_source
from harpy.table.io import read_table, write_table

_VALUES = np.arange(24, dtype=np.float32).reshape(6, 4)
_MATRICES = {
    "csr": sparse.csr_matrix(_VALUES),
    "csc": sparse.csc_matrix(_VALUES),
    "dense": _VALUES,
    # An empty compressed axis takes the decoder's from_delayed path.
    "empty_csc": sparse.csc_matrix((6, 0), dtype=np.float32),
}


def _stored_element(tmp_path, matrix, name="store.zarr"):
    """Write one matrix to its own local Zarr store and return its element."""
    group = zarr.open_group(str(tmp_path / name), mode="w")
    write_elem(group, "matrix", matrix)
    return group["matrix"]


def _table_store(tmp_path, name="sdata.zarr"):
    """Write a small annotated table with a sparse X and a dense obsm entry to a SpatialData store."""
    path = tmp_path / name
    SpatialData().write(path)
    obs = pd.DataFrame(
        {"region": pd.Categorical(["cells"] * 6), "instance": np.arange(1, 7)}, index=[str(i) for i in range(6)]
    )
    table = TableModel.parse(
        AnnData(X=sparse.csr_matrix(_VALUES), obs=obs, obsm={"dense": _VALUES[:, :2].copy()}),
        region="cells",
        region_key="region",
        instance_key="instance",
    )
    write_table(path, table_name="counts", adata=table)
    return path


@pytest.mark.parametrize("kind", list(_MATRICES))
def test_lazy_reads_register_their_stored_element(tmp_path, kind):
    element = _stored_element(tmp_path, _MATRICES[kind])
    array = _decode_anndata_element(element, mode="lazy")
    assert _lazy_read_source(array) == (tmp_path / "store.zarr" / "matrix").resolve()


def test_read_table_registers_each_matrix_with_its_path_in_the_store(tmp_path):
    path = _table_store(tmp_path)
    adata = read_table(path, table_name="counts", mode="lazy")
    assert _lazy_read_source(adata.X) == (path / "tables" / "counts" / "X").resolve()
    assert _lazy_read_source(adata.obsm["dense"]) == (path / "tables" / "counts" / "obsm" / "dense").resolve()


def test_reads_from_two_stores_with_the_same_layout_are_not_mixed_up(tmp_path):
    first, second = (_table_store(tmp_path, name) for name in ("first.zarr", "second.zarr"))
    first_x = read_table(first, table_name="counts", mode="lazy").X
    second_x = read_table(second, table_name="counts", mode="lazy").X
    assert first_x.name != second_x.name
    assert _lazy_read_source(first_x) == (first / "tables" / "counts" / "X").resolve()
    assert _lazy_read_source(second_x) == (second / "tables" / "counts" / "X").resolve()


@pytest.mark.parametrize("kind", ["csr", "dense"])
def test_copies_pickling_and_persist_keep_the_registration(tmp_path, kind):
    """These keep the Dask name, so a read the user has not changed is still recognised."""
    array = _decode_anndata_element(_stored_element(tmp_path, _MATRICES[kind]), mode="lazy")
    source = _lazy_read_source(array)
    assert source is not None
    assert _lazy_read_source(array.copy()) == source
    assert _lazy_read_source(AnnData(X=array).copy().X) == source
    assert _lazy_read_source(pickle.loads(pickle.dumps(array))) == source
    # persist() holds computed values under the same name; the registry
    # identifies the read, not the current state of the store.
    assert _lazy_read_source(array.persist()) == source


def test_anndata_read_elem_lazy_with_the_same_chunks_maps_to_the_same_element(tmp_path):
    """Harpy's lazy read is AnnData's read_elem_lazy, so the same call gives the same name."""
    element = _stored_element(tmp_path, _MATRICES["dense"])
    array = _decode_anndata_element(element, mode="lazy")
    external = read_elem_lazy(element, chunks=(array.chunksize[0], -1))
    assert external.name == array.name
    assert _lazy_read_source(external) == _lazy_read_source(array)


@pytest.mark.parametrize("kind", ["csr", "dense"])
def test_operations_that_change_the_graph_are_not_registered(tmp_path, kind):
    array = _decode_anndata_element(_stored_element(tmp_path, _MATRICES[kind]), mode="lazy")
    assert _lazy_read_source(array * 2) is None
    assert _lazy_read_source(array[:3]) is None
    assert _lazy_read_source(array.rechunk({0: 2})) is None
    # A rechunk to the same chunks returns the same array, which is still the read.
    assert _lazy_read_source(array.rechunk(array.chunks)) == _lazy_read_source(array)


def test_reads_from_other_stores_are_not_registered():
    group = zarr.open_group(MemoryStore(), mode="w")
    write_elem(group, "matrix", _MATRICES["csr"])
    array = _decode_anndata_element(group["matrix"], mode="lazy")
    assert _lazy_read_source(array) is None


@pytest.mark.parametrize("mode", ["backed", "eager"])
def test_backed_and_eager_reads_are_not_dask_arrays_and_have_no_source(tmp_path, mode):
    value = _decode_anndata_element(_stored_element(tmp_path, _MATRICES["csr"]), mode=mode)
    assert _lazy_read_source(value) is None


def test_lazy_reads_of_staged_components_are_registered_too(tmp_path, monkeypatch):
    """write_table reads the staged table lazily to validate it; those reads are registered as well.

    Their store paths are temporary staging folders next to the store, so their
    names never match a user's arrays.
    """
    registered = {}
    monkeypatch.setattr(anndata_storage, "_LAZY_READS", registered)
    path = _table_store(tmp_path)
    staged_reads = [
        source
        for source in registered.values()
        if any(part.startswith(f".{path.name}.harpy-") for part in source.parts)
    ]
    assert staged_reads, "write_table's lazy validation reads of the staged table were not registered."
