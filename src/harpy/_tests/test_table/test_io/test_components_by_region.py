import dask.array as da
import numpy as np
import pandas as pd
import pytest
import zarr
from anndata import read_h5ad
from anndata.io import write_elem
from dask import delayed
from dask.callbacks import Callback
from scipy import sparse
from spatialdata import SpatialData
from spatialdata.models import TableModel
from zarr.storage import LocalStore

import harpy.table.io._components_by_region as regional_adapter
import harpy.table.io._write as table_writer
import harpy.table.io._write_by_region as regional_writer
from harpy._storage._anndata import _read_backed_element
from harpy._tests.test_table.test_io.test_write import _store_bytes
from harpy.table import add_table_components_by_region, delete_table_components, read_table

# Settings for a 12 × 5 matrix, with the blocks they give it. Dense matrices in
# storage have stored chunks of (3, 2): an integer rounds down to whole stored
# chunks of 3 rows, and "storage" keeps them.
_STORAGE_BACKED_SETTINGS = [
    ("dense", {"dense_chunks": 4}, ((3, 3, 3, 3), (5,))),
    ("dense", {"dense_chunks": "storage"}, ((3, 3, 3, 3), (2, 2, 1))),
    ("csr", {"sparse_chunks": 5}, ((5, 5, 2), (5,))),
    ("csc", {"sparse_chunks": 2}, ((12,), (2, 2, 1))),
]


def _unexpected(*args, **kwargs):
    raise AssertionError("Unexpected storage access or computation")


def _matrix(values, matrix_format):
    return values if matrix_format == "dense" else getattr(sparse, f"{matrix_format}_matrix")(values)


def _dense(matrix):
    if isinstance(matrix, da.Array):
        matrix = matrix.compute()
    return matrix.toarray() if sparse.issparse(matrix) else matrix


@pytest.mark.parametrize("backed", [False, True])
@pytest.mark.parametrize("matrix_format", ["dense", "csr", "csc"])
@pytest.mark.parametrize("create", [False, True])
def test_regional_update_preserves_local_state_and_uses_destination_measurements(
    regional_store, monkeypatch, backed, matrix_format, create
):
    """Update A in place; retain B from disk when backed, from memory otherwise.

    The attached matrix deliberately differs from storage. Split sparse chunks
    exercise lazy CSR-column/CSC-row preparation; unrelated edits and original
    references survive, including the input array's chunks and measurements.
    """
    path, table, identity, old = regional_store(matrix_format)
    sdata = SpatialData(tables={"counts": table})
    if backed:
        sdata.path = path
    local_values = old + 1000
    original_matrix = da.from_array(_matrix(local_values, matrix_format), chunks=(3, 2), asarray=False)
    table.obsm["features"] = original_matrix
    table.obs["local_note"] = "keep"
    table.uns["analysis"]["local"] = object()
    original_obs, original_X = table.obs, table.X
    original_unrelated, original_analysis = table.obsm["unrelated"], table.uns["analysis"]
    original_chunks = original_matrix.chunks
    values = np.arange(20, dtype=np.float32).reshape(4, 5) * 10
    values[0] = 0
    payload = da.from_array(_matrix(values, matrix_format), chunks=(2, 2), asarray=False)
    component = ("obsm", "new" if create else "features")
    fill = np.nan if matrix_format == "dense" else 0
    expected = np.full_like(old, fill) if create else (old if backed else local_values).copy()
    expected[[1, 2, 3, 6]] = values
    before = _store_bytes(path)
    identity.index = ["ignored"] * len(identity)

    with monkeypatch.context() as patch:
        if not backed:
            # Unbacked updates must not consult storage, even with lazy matrices.
            for method in ("get", "get_partial_values", "set", "delete"):
                patch.setattr(LocalStore, method, _unexpected)
        # Separately guard against scheduling already-in-memory Dask inputs.
        with Callback(start=None if backed else _unexpected):
            result = add_table_components_by_region(
                sdata,
                table_name="counts",
                components={component: payload, ("uns", "analysis", "method"): "updated"},
                obs_identity=identity,
                fill_values={component: fill} if create else None,
                sparse_chunks=3,
                dense_chunks=3,
                overwrite=backed,
            )
    assert result is sdata and sdata.tables["counts"] is table
    assert table.obs is original_obs and table.X is original_X
    assert table.obsm["unrelated"] is original_unrelated
    assert table.uns["analysis"]["local"] is original_analysis["local"]
    assert original_analysis["method"] == "old" and table.uns["analysis"]["method"] == "updated"
    merged = table.obsm[component[1]]
    assert isinstance(merged, da.Array)
    computed = merged.compute()
    assert ("dense" if isinstance(computed, np.ndarray) else computed.format) == matrix_format
    np.testing.assert_array_equal(_dense(computed), expected)
    assert original_matrix.chunks == original_chunks and payload.chunks == ((2, 2), (2, 2, 1))
    np.testing.assert_array_equal(_dense(original_matrix), local_values)
    np.testing.assert_array_equal(_dense(payload), values)
    if not backed and not create:
        expected_chunks = {
            "dense": ((3, 3, 3, 3), (2, 2, 1)),
            "csr": ((3, 3, 3, 3), (5,)),
            "csc": ((12,), (2, 2, 1)),
        }[matrix_format]
        assert merged.chunks == expected_chunks
    if backed:
        stored = read_table(path, table_name="counts", mode="eager")
        np.testing.assert_array_equal(_dense(stored.obsm[component[1]]), expected)
        assert "local_note" not in stored.obs and "local" not in stored.uns["analysis"]
        assert stored.uns["analysis"]["method"] == "updated"
        after = _store_bytes(path)
        for prefix in ("tables/counts/X/", "tables/counts/obsm/unrelated/"):
            assert {k: v for k, v in before.items() if k.startswith(prefix)} == {
                k: v for k, v in after.items() if k.startswith(prefix)
            }
    else:
        assert _store_bytes(path) == before


@pytest.mark.parametrize("matrix_format", ["dense", "csr", "csc"])
@pytest.mark.parametrize("mode", ["eager", "backed"])
def test_unbacked_preparation_wraps_memory_and_storage_handles_without_computation(
    regional_store, monkeypatch, matrix_format, mode
):
    """Unbacked means sdata has no path, not that every matrix is already in RAM."""
    path, _, identity, old = regional_store(matrix_format)
    table = read_table(path, table_name="counts", mode=mode)
    sdata = SpatialData(tables={"counts": table})
    original = table.obsm["features"]
    values = _matrix(np.ones((4, 5), dtype=np.float32), matrix_format)
    before = _store_bytes(path)
    with monkeypatch.context() as patch:
        # Handle metadata may be inspected; no matrix payloads or writes are needed.
        original_get = LocalStore.get

        async def metadata_only(self, key, *args, **kwargs):
            assert key.rsplit("/", 1)[-1] in {"zarr.json", ".zarray", ".zattrs", ".zgroup"}, key
            return await original_get(self, key, *args, **kwargs)

        patch.setattr(LocalStore, "get", metadata_only)
        patch.setattr(LocalStore, "set", _unexpected)
        patch.setattr(LocalStore, "delete", _unexpected)
        with Callback(start=_unexpected):
            add_table_components_by_region(
                sdata, table_name="counts", components={("obsm", "features"): values}, obs_identity=identity
            )
    assert isinstance(table.obsm["features"], da.Array)
    expected = old.copy()
    expected[[1, 2, 3, 6]] = 1
    np.testing.assert_array_equal(_dense(table.obsm["features"]), expected)
    assert table.obsm["features"] is not original and _store_bytes(path) == before


@pytest.mark.parametrize("matrix_format, settings, expected_chunks", _STORAGE_BACKED_SETTINGS)
def test_unbacked_update_reads_storage_backed_existing_matrices_with_the_settings(
    regional_store, matrix_format, settings, expected_chunks
):
    """An attached table read in backed mode holds a zarr.Array or a sparse dataset handle.

    The merge splits it as the readers would, and the output keeps those blocks.
    """
    path, _, identity, old = regional_store(matrix_format)
    table = read_table(path, table_name="counts", mode="backed")
    sdata = SpatialData(tables={"counts": table})
    add_table_components_by_region(
        sdata,
        table_name="counts",
        components={("obsm", "features"): _matrix(np.ones((4, 5), dtype=np.float32), matrix_format)},
        obs_identity=identity,
        **settings,
    )
    merged = table.obsm["features"]
    assert merged.chunks == expected_chunks
    expected = old.copy()
    expected[[1, 2, 3, 6]] = 1
    np.testing.assert_array_equal(_dense(merged), expected)


@pytest.mark.parametrize("backed", [False, True])
@pytest.mark.parametrize("matrix_format, settings, expected_chunks", _STORAGE_BACKED_SETTINGS)
def test_full_update_reads_storage_backed_inputs_with_the_settings(
    regional_store, tmp_path, monkeypatch, backed, matrix_format, settings, expected_chunks
):
    """A full update keeps the input's blocks, so a storage-backed input must be split by the settings.

    Holds for backed and unbacked SpatialData alike. The input is stored in its
    own Zarr group, dense in chunks of (3, 2).
    """
    path, table, _, _ = regional_store(matrix_format)
    sdata = SpatialData(tables={"counts": table})
    if backed:
        sdata.path = path
    values = np.arange(60, dtype=np.float32).reshape(12, 5) * 10
    group = zarr.open_group(str(tmp_path / "input.zarr"), mode="w")
    dataset_kwargs = {"chunks": (3, 2)} if matrix_format == "dense" else {}
    write_elem(group, "matrix", _matrix(values, matrix_format), dataset_kwargs=dataset_kwargs)
    payload = _read_backed_element(group["matrix"])
    original_regional_matrix = regional_writer._regional_matrix
    prepared_chunks = []

    def capture_chunks(*args, **kwargs):
        result = original_regional_matrix(*args, **kwargs)
        prepared_chunks.append(result.chunks)
        return result

    monkeypatch.setattr(regional_writer, "_regional_matrix", capture_chunks)
    add_table_components_by_region(
        sdata,
        table_name="counts",
        components={("obsm", "features"): payload},
        obs_identity=table.obs[["region", "instance"]],
        overwrite=True,
        **settings,
    )
    assert prepared_chunks == [expected_chunks]
    np.testing.assert_array_equal(_dense(table.obsm["features"]), values)


@pytest.mark.parametrize("setting, value", [("sparse_chunks", "storage"), ("dense_chunks", 0)])
def test_invalid_chunk_settings_preserve_memory(regional_store, setting, value):
    """The adapter rejects invalid settings with the readers' messages, before any change."""
    _, table, identity, _ = regional_store()
    sdata = SpatialData(tables={"counts": table})
    original = table._obsm
    with pytest.raises((TypeError, ValueError), match=f"{setting} must be"):
        add_table_components_by_region(
            sdata,
            table_name="counts",
            components={("obsm", "features"): np.ones((4, 5), dtype=np.float32)},
            obs_identity=identity,
            **{setting: value},
        )
    assert table._obsm is original


@pytest.mark.parametrize("presence", ["both", "memory", "disk", "neither"])
def test_backed_component_presence_controls_overwrite_and_new_entry_fills(regional_store, presence):
    """Only disk presence requires overwrite, and only disk presence supplies retained rows.

    overwrite concerns the store: a target present only in memory is new on disk,
    so it needs no permission, and it follows the new-entry fill rules.
    """
    path, table, identity, old = regional_store()
    sdata = SpatialData(tables={"counts": table})
    sdata.path = path
    if presence in {"disk", "neither"}:
        del table.obsm["features"]
    if presence in {"memory", "neither"}:
        delete_table_components(path, table_name="counts", components=[("obsm", "features")])
    values = np.ones((4, 5), dtype=np.float32)
    options = {"table_name": "counts", "components": {("obsm", "features"): values}, "obs_identity": identity}
    before = _store_bytes(path)
    original = table._obsm
    stored = presence in {"both", "disk"}
    if stored:
        with pytest.raises(FileExistsError):
            add_table_components_by_region(sdata, **options)
        assert table._obsm is original and _store_bytes(path) == before
    else:
        with pytest.raises(ValueError, match="requires a fill"):
            add_table_components_by_region(sdata, **options)
        assert table._obsm is original and _store_bytes(path) == before
    add_table_components_by_region(sdata, **options, fill_values={("obsm", "features"): 0}, overwrite=stored)
    expected = old.copy() if presence in {"both", "disk"} else np.zeros_like(old)
    expected[[1, 2, 3, 6]] = 1
    np.testing.assert_array_equal(_dense(table.obsm["features"]), expected)


@pytest.mark.parametrize("change", ["unselected_order", "keys", "declared_regions"])
def test_backed_update_checks_all_attached_observations_before_staging(regional_store, monkeypatch, change):
    """An A-only payload cannot justify attaching unchanged B rows in the wrong order."""
    path, table, identity, _ = regional_store()
    sdata = SpatialData(tables={"counts": table})
    sdata.path = path
    if change == "unselected_order":
        table.obs.loc[["cell-0", "cell-4"], "instance"] = [2, 1]
    elif change == "keys":
        table.obs.rename(columns={"region": "sample"}, inplace=True)
        table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY] = "sample"
    else:
        table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY] = ["A"]
    original = table._obsm
    before = _store_bytes(path)
    monkeypatch.setattr(table_writer.tempfile, "mkdtemp", _unexpected)
    with Callback(start=_unexpected), pytest.raises(ValueError):
        add_table_components_by_region(
            sdata,
            table_name="counts",
            components={("obsm", "features"): np.ones((4, 5), dtype=np.float32)},
            obs_identity=identity,
            overwrite=True,
        )
    assert table._obsm is original and _store_bytes(path) == before


@pytest.mark.parametrize("backed", [False, True])
@pytest.mark.parametrize("problem", ["partial_region", "format", "annotation", "fill", "dataframe"])
def test_invalid_regional_update_preserves_memory_and_storage(regional_store, backed, problem):
    """Representative regional and annotation failures must not install accompanying metadata."""
    path, table, identity, _ = regional_store()
    sdata = SpatialData(tables={"counts": table})
    if backed:
        sdata.path = path
    values = np.ones((4, 5), dtype=np.float32)
    key = "features"
    components = {("uns", "analysis", "method"): "not installed"}
    if problem == "partial_region":
        identity, values = identity.iloc[:2], values[:2]
    elif problem == "format":
        values = sparse.csr_matrix(values)
    elif problem == "annotation":
        components[("uns", TableModel.ATTRS_KEY, TableModel.REGION_KEY)] = ["A"]
    elif problem == "fill":
        key = "new"
    else:
        values = pd.DataFrame(values)
    components[("obsm", key)] = values
    previous = {slot: getattr(table, f"_{slot}") for slot in ("obsm", "uns")}
    before = _store_bytes(path)
    with pytest.raises((ValueError, TypeError)):
        add_table_components_by_region(
            sdata, table_name="counts", components=components, obs_identity=identity, overwrite=True
        )
    assert all(getattr(table, f"_{slot}") is value for slot, value in previous.items())
    assert table.uns["analysis"]["method"] == "old" and _store_bytes(path) == before


@pytest.mark.parametrize("backed", [False, True])
@pytest.mark.parametrize("problem", ["missing_table", "view", "unannotated"])
def test_destination_requirements_reject_before_preparation(regional_store, monkeypatch, backed, problem):
    path, table, identity, _ = regional_store()
    sdata = SpatialData(tables={"counts": table})
    if backed:
        sdata.path = path
    if problem == "missing_table":
        del sdata.tables["counts"]
    elif problem == "view":
        sdata.tables["counts"] = table[:]
    else:
        del table.uns[TableModel.ATTRS_KEY]
    before = _store_bytes(path)
    original = table._obsm
    monkeypatch.setattr(table_writer.tempfile, "mkdtemp", _unexpected)
    with Callback(start=_unexpected), pytest.raises(ValueError):
        add_table_components_by_region(
            sdata,
            table_name="counts",
            components={("obsm", "new"): np.ones((4, 5))},
            obs_identity=identity,
            fill_values={("obsm", "new"): 0},
        )
    assert table._obsm is original and _store_bytes(path) == before


def test_backed_missing_stored_table_is_not_created(regional_store):
    path, table, identity, _ = regional_store()
    sdata = SpatialData(tables={"not_stored": table})
    sdata.path = path
    before = _store_bytes(path)
    with pytest.raises(FileNotFoundError, match="persist the table explicitly"):
        add_table_components_by_region(
            sdata,
            table_name="not_stored",
            components={("obsm", "features"): np.ones((4, 5))},
            obs_identity=identity,
            overwrite=True,
        )
    assert _store_bytes(path) == before


def test_hdf5_backed_destination_is_rejected(regional_store):
    path, table, identity, _ = regional_store()
    hdf5_path = path.parent / "table.h5ad"
    table.write_h5ad(hdf5_path)
    backed_table = read_h5ad(hdf5_path, backed="r+")
    try:
        # HDF5 reopens region lists as arrays. Normalize the in-memory annotation
        # so SpatialData accepts it and the test reaches Harpy's HDF5 guard.
        attrs = backed_table.uns[TableModel.ATTRS_KEY]
        attrs[TableModel.REGION_KEY] = attrs[TableModel.REGION_KEY].tolist()
        sdata = SpatialData(tables={"counts": backed_table})
        before = hdf5_path.read_bytes()
        with pytest.raises(ValueError, match="HDF5-backed"):
            add_table_components_by_region(
                sdata,
                table_name="counts",
                components={("obsm", "features"): np.ones((4, 5))},
                obs_identity=identity,
            )
        assert hdf5_path.read_bytes() == before
    finally:
        backed_table.file.close()


@pytest.mark.parametrize("matrix_format", ["dense", "csr", "csc"])
def test_unbacked_empty_feature_axis_keeps_format_during_chunk_preparation(regional_store, matrix_format):
    """Repartitioning an attached CSC's rows must not turn empty sparse blocks dense."""
    _, table, identity, _ = regional_store()
    table.obsm["features"] = da.from_array(
        _matrix(np.empty((12, 0), dtype=np.float32), matrix_format), chunks=(3, 2), asarray=False
    )
    sdata = SpatialData(tables={"counts": table})
    add_table_components_by_region(
        sdata,
        table_name="counts",
        components={("obsm", "features"): _matrix(np.empty((4, 0), dtype=np.float32), matrix_format)},
        obs_identity=identity,
    )
    merged = table.obsm["features"].compute()
    assert merged.shape == (12, 0)
    assert ("dense" if isinstance(merged, np.ndarray) else merged.format) == matrix_format


@pytest.mark.parametrize("backed", [False, True])
def test_full_region_coverage_needs_no_fill_and_can_replace_whole_uns(regional_store, backed):
    """Full identities remain in interleaved table order; metadata replacement preserves linkage."""
    path, table, _, _ = regional_store()
    sdata = SpatialData(tables={"counts": table})
    if backed:
        sdata.path = path
    values = np.arange(24).reshape(12, 2)
    metadata = {TableModel.ATTRS_KEY: table.uns[TableModel.ATTRS_KEY], "analysis": {"method": "all regions"}}
    add_table_components_by_region(
        sdata,
        table_name="counts",
        components={("obsm", "all_regions"): values, ("uns",): metadata},
        obs_identity=table.obs[["region", "instance"]],
        overwrite=backed,
    )
    np.testing.assert_array_equal(_dense(table.obsm["all_regions"]), values)
    assert table.uns["analysis"]["method"] == "all regions"
    TableModel.validate(table)


@pytest.mark.parametrize("backed, failure", [(False, "installation"), (True, "installation"), (True, "finalization")])
def test_failure_restores_exact_live_references_and_storage(regional_store, monkeypatch, backed, failure):
    """Failure after installation still restores old slots; disk backups survive through finalization."""
    path, table, identity, _ = regional_store()
    sdata = SpatialData(tables={"counts": table})
    if backed:
        sdata.path = path
    table.uns["analysis"]["local"] = object()
    zarr.consolidate_metadata(str(path))
    before = _store_bytes(path)
    previous = {slot: getattr(table, f"_{slot}") for slot in ("obsm", "uns")}
    original_install = regional_adapter._install_memory_updates
    original_consolidate = zarr.consolidate_metadata

    def failed_install(*args, **kwargs):
        original_install(*args, **kwargs)
        raise RuntimeError("installation failed")

    def failed_consolidate(*args, **kwargs):
        original_consolidate(*args, **kwargs)
        raise RuntimeError("finalization failed")

    if failure == "installation":
        monkeypatch.setattr(regional_adapter, "_install_memory_updates", failed_install)
    else:
        monkeypatch.setattr(zarr, "consolidate_metadata", failed_consolidate)
    with pytest.raises(RuntimeError, match=failure):
        add_table_components_by_region(
            sdata,
            table_name="counts",
            components={("obsm", "features"): np.ones((4, 5), dtype=np.float32), ("uns", "analysis", "method"): "new"},
            obs_identity=identity,
            overwrite=True,
        )
    assert sdata.tables["counts"] is table
    assert all(getattr(table, f"_{slot}") is value for slot, value in previous.items())
    assert table.uns["analysis"]["method"] == "old" and _store_bytes(path) == before
    assert not list(path.parent.glob(f".{path.name}.harpy-*"))


@pytest.mark.parametrize("backed", [False, True])
def test_lazy_self_update_installs_result_without_reading_published_data_early(regional_store, backed):
    """Old lazy matrix -> selected A rows * 2 -> complete replacement -> selective attachment."""
    path, _, identity, old = regional_store()
    table = read_table(path, table_name="counts", mode="lazy")
    sdata = SpatialData(tables={"counts": table})
    if backed:
        sdata.path = path
    payload = table.obsm["features"][[1, 2, 3, 6]] * 2
    add_table_components_by_region(
        sdata,
        table_name="counts",
        components={("obsm", "features"): payload},
        obs_identity=identity,
        overwrite=True,
    )
    expected = old.copy()
    expected[[1, 2, 3, 6]] *= 2
    np.testing.assert_array_equal(_dense(table.obsm["features"]), expected)


def test_backed_update_ignores_local_matrix_payloads_and_reopens_only_requested_paths(regional_store, monkeypatch):
    """Merge supplied regional values with stored measurements, without evaluating local matrices.

    The stored features matrix has shape (12, 5), but identity selects only
    region A's four observations. The supplied np.ones((4, 5)) therefore updates
    four rows, not the complete obsm entry::

        Region   Table row positions   Measurements in the result
        A        1, 2, 3, 6            Supplied ones, across all five columns
        B        The other eight rows  Retained from the stored matrix (old)

    Before the update, the test replaces the attached features and unrelated
    matrices locally, without changing their stored versions. These replacements
    raise if computed: retained B measurements must come from the stored features
    matrix, not its attached replacement. The result still has shape (12, 5) and
    replaces the entire attached features matrix. Thus unsaved edits within that
    matrix are not preserved, even for unselected B rows.

    Guards also reject payload reads from stored X and unrelated obsm entries,
    while allowing metadata reads. Only the requested features matrix and analysis
    method are reopened from permanent paths for attachment; unrelated live
    references remain unchanged.
    """
    path, table, identity, old = regional_store()
    for key in ("features", "unrelated"):
        shape = table.obsm[key].shape
        # Shape/dtype inspection is allowed, but computing either local matrix
        # runs _unexpected() and fails the test; retained feature values must come from disk.
        table.obsm[key] = da.from_delayed(delayed(_unexpected)(), shape=shape, dtype=np.float32)
    sdata = SpatialData(tables={"counts": table})
    sdata.path = path
    original_X, unrelated = table.X, table.obsm["unrelated"]
    original_get, original_partial = LocalStore.get, LocalStore.get_partial_values
    original_read = regional_adapter._read_anndata_element
    reopened = []

    def check_key(key):
        if key.startswith(("tables/counts/X/", "tables/counts/obsm/unrelated/")):
            assert key.rsplit("/", 1)[-1] in {"zarr.json", ".zarray", ".zattrs", ".zgroup"}, key

    async def guarded_get(self, key, *args, **kwargs):
        # Guard direct Zarr payload reads, not only reads made through Dask.
        check_key(key)
        return await original_get(self, key, *args, **kwargs)

    async def guarded_partial(self, prototype, key_ranges):
        key_ranges = list(key_ranges)
        for key, _ in key_ranges:
            check_key(key)
        return await original_partial(self, prototype, key_ranges)

    def read_published(group, component, **kwargs):
        assert group.name == "/tables/counts" and group.store.root == path
        reopened.append(component)
        return original_read(group, component, **kwargs)

    monkeypatch.setattr(LocalStore, "get", guarded_get)
    monkeypatch.setattr(LocalStore, "get_partial_values", guarded_partial)
    monkeypatch.setattr(regional_adapter, "_read_anndata_element", read_published)
    add_table_components_by_region(
        sdata,
        table_name="counts",
        components={("obsm", "features"): np.ones((4, 5), dtype=np.float32), ("uns", "analysis", "method"): "new"},
        obs_identity=identity,
        overwrite=True,
    )
    assert reopened == [("obsm", "features"), ("uns", "analysis", "method")]
    assert table.X is original_X and table.obsm["unrelated"] is unrelated
    expected = old.copy()
    expected[[1, 2, 3, 6]] = 1
    np.testing.assert_array_equal(_dense(table.obsm["features"]), expected)
