"""add_table_updates: the write-back of a table attached to backed SpatialData, with reinstallation."""

import numpy as np
import pandas as pd
import pytest
import zarr
from scipy import sparse
from spatialdata import SpatialData
from spatialdata.models import TableModel

import harpy.table.io._components as components_module
import harpy.table.io._updates as updates_module
from harpy._storage._anndata import _lazy_read_source
from harpy._tests.test_table.test_io.test_write import _store_bytes
from harpy._tests.test_table.test_io.test_write_table_updates import _dense, _store, _stored, _table
from harpy.table import add_table_updates, read_table, write_table_updates


def _attach(path, *, backed=True):
    """Attach the stored table, read lazily, to SpatialData that is backed by the store or not."""
    table = read_table(path, table_name="counts", mode="lazy")
    sdata = SpatialData(tables={"counts": table})
    if backed:
        sdata.path = path
    return sdata, table


def _process(adata):
    """Changes as a scanpy run makes them: a derived X, a new obs column and a new embedding."""
    adata.X = adata.X * 2
    adata.obs["leiden"] = pd.Categorical(["0", "1"] * (adata.n_obs // 2))
    adata.obsm["embedding"] = np.ones((adata.n_obs, 2))


def _element(path, *keys):
    return path.joinpath("tables", "counts", *keys).resolve()


@pytest.fixture
def store(tmp_path):
    return _store(tmp_path)


@pytest.fixture
def written(monkeypatch):
    """The components passed to _update_table_components, one mapping per call."""
    calls = []
    original_update = updates_module._update_table_components

    def record(*args, **kwargs):
        calls.append(dict(kwargs["components"]))
        return original_update(*args, **kwargs)

    monkeypatch.setattr(updates_module, "_update_table_components", record)
    return calls


def test_the_adapter_writes_what_the_store_path_function_writes_and_reinstalls_it(
    store, tmp_path, monkeypatch, written
):
    """add_table_updates writes what write_table_updates writes, then makes the attached table match the store.

    The same changes go to two identical stores: one through the adapter, one
    through the store-path function, which must write the same components. The
    adapter then reinstalls them as lazy reads of the store, and reinstalls X too,
    which x_to left as the stored counts (reopen_also). The attached table then
    matches the store, so a second call finds nothing changed and writes nothing.
    """
    counts = _stored(store).X.toarray()
    sdata, table = _attach(store)
    _process(table)
    returned = add_table_updates(sdata, table_name="counts", x_to=("layers", "doubled"), overwrite=True)

    # The same changes, written back by the store-path function to a second store.
    reference_path = _store(tmp_path, name="reference.zarr")
    reference = read_table(reference_path, table_name="counts", mode="lazy")
    _process(reference)
    reference_writes = []
    original_write = updates_module.write_table_components

    def record(*args, **kwargs):
        reference_writes.append(set(kwargs["components"]))
        return original_write(*args, **kwargs)

    monkeypatch.setattr(updates_module, "write_table_components", record)
    write_table_updates(
        reference_path, table_name="counts", adata=reference, x_to=("layers", "doubled"), overwrite=True
    )

    assert set(written[0]) == reference_writes[0] == {("layers", "doubled"), ("obs",), ("obsm", "embedding")}
    assert returned is sdata and sdata.tables["counts"] is table
    # The written components are reinstalled as lazy reads of the store, and so
    # is X, which x_to left as it was: the counts.
    assert _lazy_read_source(table.layers["doubled"]) == _element(store, "layers", "doubled")
    assert _lazy_read_source(table.obsm["embedding"]) == _element(store, "obsm", "embedding")
    assert _lazy_read_source(table.X) == _element(store, "X")
    np.testing.assert_array_equal(table.X.compute().toarray(), counts)
    np.testing.assert_array_equal(_dense(table.layers["doubled"].compute()), counts * 2)
    assert "leiden" in table.obs

    # The attached table now matches the store: a second call writes nothing.
    add_table_updates(sdata, table_name="counts", x_to=("layers", "doubled"))
    assert len(written) == 1


def test_new_components_need_no_overwrite_and_stored_ones_do(store, written):
    """overwrite concerns the store: a component that is attached but not stored is new."""
    sdata, table = _attach(store)
    table.obsm["embedding"] = np.ones((table.n_obs, 2))
    add_table_updates(sdata, table_name="counts")
    assert set(written[0]) == {("obsm", "embedding")}

    table.obs["leiden"] = pd.Categorical(["0", "1"] * (table.n_obs // 2))
    obs = table.obs
    with pytest.raises(FileExistsError, match=r"\('obs',\)"):
        add_table_updates(sdata, table_name="counts")
    assert len(written) == 1 and table.obs is obs
    add_table_updates(sdata, table_name="counts", overwrite=True)
    assert set(written[1]) == {("obs",)}


def test_unbacked_spatialdata_raises_naming_the_ways_out(store):
    sdata, table = _attach(store, backed=False)
    table.obsm["embedding"] = np.ones((table.n_obs, 2))
    with pytest.raises(ValueError, match="sdata.path is None") as error:
        add_table_updates(sdata, table_name="counts")
    for way_out in ("sdata.write(path)", "hp.tb.write_table", "hp.tb.write_table_updates(store"):
        assert way_out in str(error.value)

    # The store-path function, with the store the table was read from, writes only its changes.
    write_table_updates(store, table_name="counts", adata=sdata.tables["counts"])
    stored = _stored(store)
    np.testing.assert_array_equal(stored.obsm["embedding"], np.ones((table.n_obs, 2)))


def test_a_missing_attached_annotation_raises_before_writing(store):
    """add_table_updates raises, before writing, when the attached table lacks the store's annotation.

    After writing, the adapter puts the written components back into the
    attached table, so that table and the store must describe the same
    SpatialData linkage; add_table_components raises the same way.
    write_table_updates only writes to the store, and leaves the stored
    annotation as it is.
    """
    sdata, table = _attach(store)
    del table.uns[TableModel.ATTRS_KEY]
    table.obsm["embedding"] = np.ones((table.n_obs, 2))
    before = _store_bytes(store)
    with pytest.raises(ValueError, match="attached table has no SpatialData annotation"):
        add_table_updates(sdata, table_name="counts")
    assert _store_bytes(store) == before


def test_a_table_without_a_stored_counterpart_raises(store):
    sdata, _ = _attach(store)
    sdata.tables["unsaved"] = _table()
    with pytest.raises(FileNotFoundError, match="hp.tb.add_table"):
        add_table_updates(sdata, table_name="unsaved")


def test_x_to_a_layer_of_a_table_stored_without_x_leaves_the_attached_x_none(tmp_path, written):
    table = _table()
    table.X = None
    path = _store(tmp_path, table)
    sdata, attached = _attach(path)
    attached.X = sparse.csr_matrix(attached.layers["dense"].compute())
    add_table_updates(sdata, table_name="counts", x_to=("layers", "counts"))
    assert set(written[0]) == {("layers", "counts")}
    assert attached.X is None
    assert _lazy_read_source(attached.layers["counts"]) == _element(path, "layers", "counts")


@pytest.mark.parametrize("failure", ["installation", "finalization"])
def test_a_failure_restores_the_attached_table_and_the_store(store, monkeypatch, failure):
    """Also after x_to: the processed X and the layers without the destination come back."""
    sdata, table = _attach(store)
    table.X = table.X * 2
    zarr.consolidate_metadata(str(store))
    before = _store_bytes(store)
    previous = {slot: getattr(table, f"_{slot}") for slot in ("X", "layers")}
    original_install = components_module._install_memory_updates
    original_consolidate = zarr.consolidate_metadata

    def failed_install(*args, **kwargs):
        original_install(*args, **kwargs)
        raise RuntimeError("installation failed")

    def failed_consolidate(*args, **kwargs):
        original_consolidate(*args, **kwargs)
        raise RuntimeError("finalization failed")

    if failure == "installation":
        monkeypatch.setattr(components_module, "_install_memory_updates", failed_install)
    else:
        monkeypatch.setattr(zarr, "consolidate_metadata", failed_consolidate)
    with pytest.raises(RuntimeError, match=failure):
        add_table_updates(sdata, table_name="counts", x_to=("layers", "log1p"))
    assert all(getattr(table, f"_{slot}") is value for slot, value in previous.items())
    assert "log1p" not in table.layers
    assert _store_bytes(store) == before
    assert not list(store.parent.glob(f".{store.name}.harpy-*"))
