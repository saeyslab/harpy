from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import zarr
from anndata import read_h5ad
from dask.callbacks import Callback
from spatialdata import SpatialData
from spatialdata.models import TableModel
from zarr.storage import LocalStore

import harpy.table.io._components as component_adapter
import harpy.table.io._write as table_writer
from harpy._tests.test_table.test_io.test_read import _assert_value
from harpy._tests.test_table.test_io.test_write import _annotated_store, _store_bytes
from harpy.table import (
    add_table_components,
    delete_table_components,
    read_table,
    remove_table_components,
)


def _attach(path, *, backed, annotated=False):
    if annotated:
        _annotated_store(path)
    table = read_table(path, table_name="counts", mode="lazy")
    sdata = SpatialData(tables={"counts": table})
    if backed:
        sdata.path = path
    return sdata, table


def _unexpected(*args, **kwargs):
    raise AssertionError("Unexpected computation or storage operation")


@pytest.mark.parametrize("backed", [False, True])
def test_mixed_update_preserves_unrequested_live_state(make_table_io_store, backed):
    """Replace embedding/metadata and remove loadings, without refreshing unrelated local edits."""
    path = make_table_io_store()
    sdata, table = _attach(path, backed=backed, annotated=True)
    table.obs["local_note"] = ["keep", "me"]
    table.uns["analysis"]["local"] = object()
    original_obs, original_X, original_raw = table.obs, table.X, table.raw
    original_frame = table.obsm["frame"]
    original_analysis = table.uns["analysis"]
    original_table_bytes = _store_bytes(path / "tables/counts")
    embedding = da.from_array(np.arange(6).reshape(2, 3), chunks=(1, 3))
    result = add_table_components(
        sdata,
        table_name="counts",
        components={("obsm", "embedding"): embedding, ("uns", "analysis", "method"): "new"},
        delete=[("varm", "loadings")],
        obs_identity=table.obs[["region", "instance"]],
        overwrite=backed,
    )
    assert result is sdata and sdata.tables["counts"] is table
    assert table.obs is original_obs and table.X is original_X and table.raw is original_raw
    assert table.obsm["frame"] is original_frame
    assert table.uns["analysis"]["local"] is original_analysis["local"]
    assert original_analysis["method"] == "test"
    assert table.uns["analysis"]["method"] == "new"
    assert "loadings" not in table.varm
    assert isinstance(table.obsm["embedding"], da.Array)
    np.testing.assert_array_equal(table.obsm["embedding"].compute(), embedding.compute())
    if backed:
        stored = read_table(path, table_name="counts", mode="eager")
        assert "local_note" not in stored.obs and "local" not in stored.uns["analysis"]
        assert "loadings" not in stored.varm and stored.uns["analysis"]["method"] == "new"
        np.testing.assert_array_equal(stored.obsm["embedding"], embedding.compute())
    else:
        assert table.obsm["embedding"] is embedding
        assert _store_bytes(path / "tables/counts") == original_table_bytes


@pytest.mark.parametrize("annotated", [False, True])
@pytest.mark.parametrize("observation_aligned", [False, True])
def test_destination_identity_contains_only_needed_columns(
    make_table_io_store, monkeypatch, annotated, observation_aligned
):
    """Prepare only spatial identity columns, and only for annotated observation checks."""
    path = make_table_io_store()
    sdata, table = _attach(path, backed=False, annotated=annotated)
    table.obs["local_note"] = ["keep", "me"]
    original_obs = table.obs.copy()
    validate = component_adapter._validate_component_values_against_axes

    def check_identity(*args, **kwargs):
        identity = kwargs["expected_region_instance_identity"]
        if annotated and observation_aligned:
            pd.testing.assert_frame_equal(identity, table.obs[["region", "instance"]])
        else:
            assert identity is None
        return validate(*args, **kwargs)

    monkeypatch.setattr(component_adapter, "_validate_component_values_against_axes", check_identity)
    options = {}
    if observation_aligned:
        components = {("obsm", "new"): np.ones((2, 1))}
        options["obs_identity"] = table.obs[["region", "instance"]] if annotated else table.obs_names
    else:
        components = {("uns", "new"): "value"}
    add_table_components(sdata, table_name="counts", components=components, **options)
    pd.testing.assert_frame_equal(table.obs, original_obs)


@pytest.mark.parametrize("mode", ["lazy", "backed"])
def test_unbacked_updates_do_not_compute_copy_or_access_storage(make_table_io_store, monkeypatch, mode):
    """Unbacked refers to SpatialData: supplied matrices may still be storage-backed."""
    path = make_table_io_store(matrix_kind="dense")
    table = read_table(path, table_name="counts", mode=mode)
    sdata = SpatialData(tables={"counts": table})
    replacement = table.X
    old_obs = table.obs
    updated_obs = old_obs.assign(score=[1, 2])
    with monkeypatch.context() as patch:
        for method in ("get", "set", "delete"):
            patch.setattr(LocalStore, method, _unexpected)
        with Callback(start=_unexpected):
            add_table_components(
                sdata,
                table_name="counts",
                components={("layers", "counts"): replacement, ("obs",): updated_obs},
                var_names=table.var_names,
            )
            remove_table_components(sdata, table_name="counts", components=[("X",), ("raw",)])
    assert table.layers["counts"] is replacement
    pd.testing.assert_frame_equal(table.obs, updated_obs)
    assert "score" not in old_obs and table.X is None and table.raw is None


@pytest.mark.parametrize("backed", [False, True])
def test_axis_frame_update_does_not_modify_unrequested_dataframe_indices(make_table_io_store, backed):
    path = make_table_io_store()
    sdata, table = _attach(path, backed=backed)
    frame = table.obsm["frame"]
    old_index = frame.index
    obs = table.obs.copy()
    obs.index = obs.index.rename("new_name")
    supplied_frame = pd.DataFrame({"a": [1, 2]}, index=obs.index.copy())
    add_table_components(
        sdata,
        table_name="counts",
        components={("obs",): obs, ("obsm", "new_frame"): supplied_frame},
        overwrite=backed,
    )
    # Inspect without accessing table.obsm: AnnData's mapping getter itself
    # normalizes dataframe index names against the current observation index.
    assert table._obsm["frame"] is frame and frame.index is old_index
    assert frame.index.name is None
    assert supplied_frame.index.name == "new_name"
    assert table.obs.index.name == "new_name"


@pytest.mark.parametrize("backed", [False, True])
def test_none_replacements_and_whole_uns_preserve_annotation(make_table_io_store, backed):
    path = make_table_io_store()
    sdata, table = _attach(path, backed=backed, annotated=True)
    original_raw = table.raw
    attrs = table.uns[TableModel.ATTRS_KEY]
    add_table_components(
        sdata,
        table_name="counts",
        components={("X",): None, ("uns",): {TableModel.ATTRS_KEY: attrs, "optional": None}},
        obs_identity=table.obs[["region", "instance"]],
        var_names=table.var_names,
        overwrite=backed,
    )
    assert table.X is None and table.raw is original_raw and "counts" in table.layers
    assert table.uns["optional"] is None and "analysis" not in table.uns
    TableModel.validate(table)
    assert isinstance(attrs[TableModel.REGION_KEY], list)
    if backed:
        root = zarr.open_group(str(path), mode="r", use_consolidated=False)
        # None is an encoded replacement, not removal of the X path.
        assert root["tables/counts/X"].attrs["encoding-type"] == "null"
        assert root["tables/counts/uns/optional"].attrs["encoding-type"] == "null"


@pytest.mark.parametrize("presence", ["both", "memory", "disk", "neither"])
def test_backed_overwrite_concerns_the_store_only(make_table_io_store, presence):
    """overwrite guards stored targets; a target present only in memory is replaced without it."""
    path = make_table_io_store()
    if presence in {"memory", "neither"}:
        delete_table_components(path, table_name="counts", components=[("obsm", "embedding")])
    sdata, table = _attach(path, backed=True)
    if presence == "memory":
        table.obsm["embedding"] = np.ones((2, 2))
    elif presence == "disk":
        del table.obsm["embedding"]
    before = _store_bytes(path)
    options = {
        "table_name": "counts",
        "components": {("obsm", "embedding"): np.full((2, 2), 7)},
        "obs_identity": table.obs_names,
    }
    stored = presence in {"both", "disk"}
    if stored:
        old = table.obsm.get("embedding")
        with pytest.raises(FileExistsError, match="overwrite"):
            add_table_components(sdata, **options)
        assert table.obsm.get("embedding") is old and _store_bytes(path) == before
    add_table_components(sdata, **options, overwrite=stored)
    np.testing.assert_array_equal(table.obsm["embedding"].compute(), np.full((2, 2), 7))


@pytest.mark.parametrize("presence", ["both", "memory", "disk", "neither"])
def test_removal_resolves_both_locations_without_unnecessary_writes(make_table_io_store, monkeypatch, presence):
    """Memory-only and absent targets require neither a workspace nor consolidation."""
    path = make_table_io_store()
    if presence in {"memory", "neither"}:
        delete_table_components(path, table_name="counts", components=[("obsm", "embedding")])
    sdata, table = _attach(path, backed=True)
    if presence == "memory":
        table.obsm["embedding"] = np.ones((2, 2))
    elif presence == "disk":
        del table.obsm["embedding"]
    original_X = table.X
    before = _store_bytes(path)
    messages = []
    monkeypatch.setattr(table_writer.log, "info", messages.append)
    if presence in {"memory", "neither"}:
        monkeypatch.setattr(table_writer.tempfile, "mkdtemp", _unexpected)
        monkeypatch.setattr(zarr, "consolidate_metadata", _unexpected)
        monkeypatch.setattr(LocalStore, "set", _unexpected)
        monkeypatch.setattr(LocalStore, "delete", _unexpected)
    with Callback(start=_unexpected):
        result = remove_table_components(sdata, table_name="counts", components=[("obsm", "embedding")])
    assert result is sdata and table.X is original_X and "embedding" not in table.obsm
    assert "embedding" not in read_table(path, table_name="counts", mode="lazy").obsm
    if presence in {"memory", "neither"}:
        assert _store_bytes(path) == before
    if presence == "neither":
        assert any("already absent" in message for message in messages)


@pytest.mark.parametrize("operation", ["add", "remove"])
@pytest.mark.parametrize(
    ("problem", "backed"),
    [("view", False), ("view", True), ("missing_live", False), ("missing_live", True), ("missing_stored", True)],
)
def test_destination_requirements_fail_before_staging(make_table_io_store, monkeypatch, backed, operation, problem):
    path = make_table_io_store()
    sdata, table = _attach(path, backed=backed)
    name = "counts"
    if problem == "view":
        sdata.tables[name] = table[:1]
    elif problem == "missing_live":
        del sdata.tables[name]
    else:
        name = "not_stored"
        sdata.tables[name] = table
    before = _store_bytes(path)
    monkeypatch.setattr(table_writer.tempfile, "mkdtemp", _unexpected)
    with Callback(start=_unexpected), pytest.raises((ValueError, FileNotFoundError)):
        if operation == "add":
            add_table_components(sdata, table_name=name, components={("uns", "new"): 1})
        else:
            remove_table_components(sdata, table_name=name, components=[("X",)])
    assert _store_bytes(path) == before
    assert "new" not in table.uns and table.X is not None


def test_hdf5_destination_cannot_write_outside_the_spatialdata_operation(make_table_io_store):
    path = make_table_io_store()
    table = read_table(path, table_name="counts", mode="eager")
    hdf5_path = path.parent / "table.h5ad"
    table.write_h5ad(hdf5_path)
    backed_table = read_h5ad(hdf5_path, backed="r+")
    try:
        sdata = SpatialData(tables={"counts": backed_table})
        before = hdf5_path.read_bytes()
        with pytest.raises(ValueError, match="HDF5-backed"):
            remove_table_components(sdata, table_name="counts", components=[("X",)])
        assert backed_table.X is not None and hdf5_path.read_bytes() == before
    finally:
        backed_table.file.close()


@pytest.mark.parametrize("backed", [False, True])
@pytest.mark.parametrize("problem", ["multiindex_columns", "index_name"])
def test_axis_frames_keep_anndata_metadata_requirements(make_table_io_store, backed, problem):
    path = make_table_io_store()
    sdata, table = _attach(path, backed=backed)
    obs = table.obs.copy()
    if problem == "multiindex_columns":
        obs.columns = pd.MultiIndex.from_tuples([("a", "b"), ("c", "d")])
    else:
        obs.index.name = 17
    previous = table.obs
    before = _store_bytes(path)
    with pytest.raises(ValueError, match="MultiIndex|index name"):
        add_table_components(sdata, table_name="counts", components={("obs",): obs}, overwrite=True)
    assert table.obs is previous and _store_bytes(path) == before


@pytest.mark.parametrize("backed", [False, True])
def test_malformed_live_deletion_parent_is_not_a_missing_target(make_table_io_store, backed):
    path = make_table_io_store()
    sdata, table = _attach(path, backed=backed)
    table.uns["bad"] = 2
    before = _store_bytes(path)
    with pytest.raises(ValueError, match="not a mapping"):
        remove_table_components(sdata, table_name="counts", components=[("uns", "bad", "child")])
    assert table.uns["bad"] == 2 and _store_bytes(path) == before


@pytest.mark.parametrize("axis", ["obs", "annotated_obs", "var", "raw_var"])
def test_backed_updates_reject_live_axis_order_mismatch(make_table_io_store, monkeypatch, axis):
    """Supplied identities can match memory yet still be unsafe to attach against storage."""
    path = make_table_io_store()
    sdata, table = _attach(path, backed=True, annotated=axis == "annotated_obs")
    if axis == "annotated_obs":
        table.obs["region"] = pd.Categorical(table.obs["region"].iloc[::-1].to_numpy())
        options = {"obs_identity": table.obs[["region", "instance"]]}
        target, shape = ("obsm", "new"), (2, 1)
    elif axis == "obs":
        table.obs_names = table.obs_names[::-1]
        options = {"obs_identity": table.obs_names}
        target, shape = ("obsm", "new"), (2, 1)
    elif axis == "var":
        table.var_names = table.var_names[::-1]
        options = {"var_names": table.var_names}
        target, shape = ("varm", "new"), (3, 1)
    else:
        table.raw.var.index = table.raw.var.index[::-1]
        options = {"raw_var_names": table.raw.var_names}
        target, shape = ("raw", "varm", "new"), (5, 1)
    before = _store_bytes(path)
    monkeypatch.setattr(table_writer.tempfile, "mkdtemp", _unexpected)
    with pytest.raises(ValueError, match="differ in order"):
        add_table_components(sdata, table_name="counts", components={target: np.ones(shape)}, **options)
    assert _store_bytes(path) == before
    # An independent uns update does not attach any observation/feature-aligned matrix.
    monkeypatch.undo()
    add_table_components(sdata, table_name="counts", components={("uns", "new"): 1})
    assert table.uns["new"] == 1


@pytest.mark.parametrize("backed", [False, True])
@pytest.mark.parametrize(
    "problem",
    [
        "identity_missing",
        "explicit_conflict",
        "shape",
        "annotation",
        "annotation_type",
        "missing_identity_column",
        "duplicate_obs_columns",
        "overlap",
        "parent",
    ],
)
def test_adapters_share_component_validation(make_table_io_store, backed, problem):
    path = make_table_io_store()
    sdata, table = _attach(path, backed=backed, annotated=True)
    components = {("obsm", "new"): np.ones((2, 2))}
    options = {"obs_identity": table.obs[["region", "instance"]]}
    if problem == "identity_missing":
        options = {}
    elif problem == "explicit_conflict":
        components[("obs",)] = table.obs.copy()
        options["obs_identity"] = options["obs_identity"].iloc[::-1]
    elif problem == "shape":
        components[("obsm", "new")] = np.ones((3, 2))
    elif problem == "annotation":
        components = {("uns", TableModel.ATTRS_KEY): {}}
    elif problem == "annotation_type":
        table.uns[TableModel.ATTRS_KEY] = []
    elif problem == "missing_identity_column":
        table.obs.drop(columns="region", inplace=True)
    elif problem == "duplicate_obs_columns":
        # Duplicates outside the identity columns must not disappear unnoticed
        # when preparation selects just the region and instance columns.
        table.obs["extra_a"] = 0
        table.obs["extra_b"] = 1
        table.obs.rename(columns={"extra_a": "extra", "extra_b": "extra"}, inplace=True)
    elif problem == "overlap":
        options["delete"] = [("obsm", "new")]
    else:
        table.uns["bad"] = 2
        components = {("uns", "bad", "nested"): 3}
    before = _store_bytes(path)
    original_obs = table.obs
    with pytest.raises(ValueError):
        add_table_components(sdata, table_name="counts", components=components, overwrite=True, **options)
    assert table.obs is original_obs and "new" not in table.obsm
    assert _store_bytes(path) == before


@pytest.mark.parametrize("backed", [False, True])
@pytest.mark.parametrize("supplied_frame", [False, True])
def test_raw_creation_and_selective_updates(make_table_io_store, backed, supplied_frame):
    """raw has its own var axis; replacing raw.X/varm must not refresh the main table or raw.var."""
    path = make_table_io_store()
    delete_table_components(path, table_name="counts", components=[("raw",)])
    sdata, table = _attach(path, backed=backed)
    original_X, original_var = table.X, table.var
    raw_var = pd.DataFrame({"raw_note": ["a", "b"]}, index=["r1", "r2"])
    raw_X = da.ones((2, 2), chunks=(1, 2))
    components = {("raw", "X"): raw_X, ("raw", "varm", "new"): np.ones((2, 1))}
    if supplied_frame:
        components[("raw", "var")] = raw_var
    add_table_components(
        sdata, table_name="counts", components=components, obs_identity=table.obs_names, raw_var_names=raw_var.index
    )
    assert table.X is original_X and table.var is original_var
    assert table.raw.var_names.tolist() == ["r1", "r2"]
    np.testing.assert_array_equal(table.raw.X.compute(), np.ones((2, 2)))
    if supplied_frame:
        pd.testing.assert_frame_equal(table.raw.var, raw_var)
    old_raw = table.raw
    old_raw.var["local"] = 1
    add_table_components(
        sdata,
        table_name="counts",
        components={("raw", "X"): raw_X * 3},
        delete=[("raw", "varm", "new")],
        obs_identity=table.obs_names,
        raw_var_names=raw_var.index,
        overwrite=backed,
    )
    assert table.raw.var is old_raw.var and "new" in old_raw.varm and "new" not in table.raw.varm
    np.testing.assert_array_equal(table.raw.X.compute(), np.full((2, 2), 3))
    if backed:
        stored = read_table(path, table_name="counts", mode="eager")
        assert "local" not in stored.raw.var and "new" not in stored.raw.varm
        np.testing.assert_array_equal(stored.raw.X, np.full((2, 2), 3))
    remove_table_components(sdata, table_name="counts", components=[("raw",)])
    assert table.raw is None


def test_metadata_update_neither_reads_matrix_payloads_nor_computes(make_table_io_store, monkeypatch):
    path = make_table_io_store()
    sdata, table = _attach(path, backed=True)
    original_get = LocalStore.get

    async def guarded_get(self, key, *args, **kwargs):
        # Metadata lookup is allowed; numerical chunk access is not, including
        # reads outside Dask. Serialization only writes the requested uns record.
        matrix_roots = ("X/", "layers/", "obsm/", "varm/", "obsp/", "varp/", "raw/X/", "raw/varm/")
        if key.startswith(tuple(f"tables/counts/{root}" for root in matrix_roots)):
            assert key.rsplit("/", 1)[-1] in {"zarr.json", ".zarray", ".zattrs", ".zgroup"}, key
        return await original_get(self, key, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(LocalStore, "get", guarded_get)
        # Independently reject Dask execution during reading, validation and installation.
        with Callback(start=_unexpected):
            add_table_components(sdata, table_name="counts", components={("uns", "note"): "new"})
            remove_table_components(sdata, table_name="counts", components=[("obsm", "embedding")])
    assert table.uns["note"] == "new" and "embedding" not in table.obsm


@pytest.mark.parametrize("backed", [False, True])
def test_lazy_replacement_can_depend_on_deleted_component(make_table_io_store, backed):
    path = make_table_io_store()
    sdata, table = _attach(path, backed=backed)
    original_layer = table.layers["counts"]
    expected = original_layer.compute() * 2
    add_table_components(
        sdata,
        table_name="counts",
        components={("X",): original_layer * 2},
        delete=[("layers", "counts")],
        obs_identity=table.obs_names,
        var_names=table.var_names,
        overwrite=backed,
    )
    assert "counts" not in table.layers
    _assert_value(table.X.compute(), expected)
    if backed:
        stored = read_table(path, table_name="counts", mode="eager")
        _assert_value(stored.X, expected)
        assert "counts" not in stored.layers


@pytest.mark.parametrize("failure", ["staging", "publication", "installation", "finalization"])
def test_backed_failures_restore_disk_and_exact_live_references(make_table_io_store, monkeypatch, failure):
    """The writer owns disk rollback; the adapter restores live slots, even after successful installation."""
    path = make_table_io_store()
    sdata, table = _attach(path, backed=True)
    table.uns["local"] = object()
    zarr.consolidate_metadata(str(path))
    before = _store_bytes(path)
    old_slots = {slot: getattr(table, f"_{slot}") for slot in ("obsm", "uns", "raw")}
    original_write = table_writer._write_anndata_element
    original_rename = Path.rename
    original_install = component_adapter._install_memory_updates
    original_consolidate = zarr.consolidate_metadata

    def failed_write(*args, **kwargs):
        original_write(*args, **kwargs)
        raise RuntimeError("staging failure")

    def failed_rename(self, destination):
        if "staging-" in str(self) and self.name == "component-1":
            raise RuntimeError("publication failure")
        return original_rename(self, destination)

    def failed_install(*args, **kwargs):
        original_install(*args, **kwargs)
        raise RuntimeError("installation failure")

    def failed_consolidate(*args, **kwargs):
        original_consolidate(*args, **kwargs)
        raise RuntimeError("finalization failure")

    if failure == "staging":
        monkeypatch.setattr(table_writer, "_write_anndata_element", failed_write)
    elif failure == "publication":
        monkeypatch.setattr(Path, "rename", failed_rename)
    elif failure == "installation":
        monkeypatch.setattr(component_adapter, "_install_memory_updates", failed_install)
    else:
        monkeypatch.setattr(zarr, "consolidate_metadata", failed_consolidate)
    with pytest.raises(RuntimeError, match=failure):
        add_table_components(
            sdata,
            table_name="counts",
            components={("obsm", "embedding"): np.ones((2, 2)), ("uns", "analysis", "method"): "new"},
            delete=[("raw",)],
            obs_identity=table.obs_names,
            overwrite=True,
        )
    assert sdata.tables["counts"] is table
    assert all(getattr(table, f"_{slot}") is value for slot, value in old_slots.items())
    assert _store_bytes(path) == before
    assert not list(path.parent.glob(f".{path.name}.harpy-*"))


@pytest.mark.parametrize("backed", [False, True])
def test_deletion_only_installation_failure_restores_memory_and_disk(make_table_io_store, monkeypatch, backed):
    path = make_table_io_store()
    sdata, table = _attach(path, backed=backed)
    before = _store_bytes(path)
    previous = table._obsm, table._raw
    original_install = component_adapter._install_memory_updates

    def fail(table, updates):
        original_install(table, updates)
        raise RuntimeError("installation failure")

    monkeypatch.setattr(component_adapter, "_install_memory_updates", fail)
    with pytest.raises(RuntimeError, match="installation failure"):
        remove_table_components(sdata, table_name="counts", components=[("obsm", "embedding"), ("raw",)])
    assert table._obsm is previous[0] and table._raw is previous[1]
    assert _store_bytes(path) == before
