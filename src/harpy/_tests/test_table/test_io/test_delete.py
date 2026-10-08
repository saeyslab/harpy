"""Explicit removals share scoped writing's validation, publication and rollback."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import zarr
from anndata import read_zarr
from anndata.io import read_elem, write_elem
from dask.callbacks import Callback
from spatialdata import SpatialData
from spatialdata.models import TableModel
from zarr.storage import LocalStore

import harpy.table.io._write as table_writer
from harpy._storage._anndata import _read_anndata_table
from harpy._tests.test_table.test_io.test_read import _assert_value
from harpy._tests.test_table.test_io.test_write import _annotated_store, _store_bytes
from harpy.table.io import delete_table_components, read_table, write_table_components


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize(
    "component",
    [
        ("X",),
        ("raw",),
        ("layers", "counts"),
        ("obsm", "embedding"),
        ("obsm", "frame"),
        ("varm", "loadings"),
        ("obsp", "neighbors"),
        ("varp", "neighbors"),
        ("raw", "varm", "loadings"),
        ("uns", "analysis"),
        ("uns", "analysis", "optional"),
    ],
)
def test_deletion_preserves_unrequested_data_and_axes(make_table_io_store, zarr_format, component):
    """Remove only the named subtree, leaving all other table bytes and empty parents intact."""
    path = make_table_io_store(zarr_format=zarr_format)
    table_path = path / "tables/counts"
    original = read_table(path, table_name="counts", mode="eager")
    before = _store_bytes(table_path)

    delete_table_components(path, table_name="counts", components=[component])

    prefix = "/".join(component) + "/"
    assert _store_bytes(table_path) == {key: value for key, value in before.items() if not key.startswith(prefix)}
    # Both readers must see the removal, including through consolidated metadata.
    root = zarr.open_group(str(path), mode="r", use_consolidated=True)
    group = root["tables/counts"]
    assert "/".join(component) not in group
    if len(component) > 1:
        assert "/".join(component[:-1]) in group
    for reopened in (read_table(path, table_name="counts", mode="eager"), read_zarr(table_path)):
        assert reopened.shape == original.shape
        pd.testing.assert_frame_equal(reopened.obs, original.obs)
        pd.testing.assert_frame_equal(reopened.var, original.var)
        if component == ("X",):
            assert reopened.X is None
        if component == ("raw",):
            assert reopened.raw is None
        else:
            assert reopened.raw.shape == original.raw.shape
    assert not list(path.parent.glob(f".{path.name}.harpy-*"))


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("matrix_kind", ["dense", "csr", "csc"])
def test_deletion_reads_only_metadata(make_table_io_store, monkeypatch, zarr_format, matrix_kind):
    """Delete dense/sparse/DataFrame payloads without decoding them or computing Dask graphs."""
    path = make_table_io_store(zarr_format=zarr_format, matrix_kind=matrix_kind)
    components = [("X",), ("raw",), ("obsm", "frame"), ("uns", "analysis")]
    original_get, original_partial = LocalStore.get, LocalStore.get_partial_values
    x_bytes = _store_bytes(path / "tables/counts/layers/counts")

    def check_key(key):
        if key.startswith("tables/"):
            assert key.rsplit("/", 1)[-1] in {".zarray", ".zattrs", ".zgroup", "zarr.json"}, key

    async def guarded_get(self, key, *args, **kwargs):
        check_key(key)
        return await original_get(self, key, *args, **kwargs)

    async def guarded_partial(self, prototype, key_ranges):
        key_ranges = list(key_ranges)
        for key, _ in key_ranges:
            check_key(key)
        return await original_partial(self, prototype, key_ranges)

    def unexpected_compute(graph):
        pytest.fail("Deletion must not compute Dask graphs.")

    # The store guards catch payload reads even outside Dask; consolidation may
    # inspect metadata, including the fixture's unrelated unsupported table.
    with monkeypatch.context() as patch:
        patch.setattr(LocalStore, "get", guarded_get)
        patch.setattr(LocalStore, "get_partial_values", guarded_partial)
        # Independently reject any scheduler execution during deletion.
        with Callback(start=unexpected_compute):
            delete_table_components(path, table_name="counts", components=components)
    assert all(not (path / "tables/counts").joinpath(*component).exists() for component in components)
    assert _store_bytes(path / "tables/counts/layers/counts") == x_bytes


@pytest.mark.parametrize("mixed", [False, True])
def test_related_deletions_and_optional_obs_replacement(make_table_io_store, mixed):
    """Callers explicitly couple feature data/metadata; local AnnData references remain unchanged."""
    path = make_table_io_store()
    table = _annotated_store(path)
    root = zarr.open_group(str(path), mode="r+", use_consolidated=False)
    write_elem(root["tables/counts/uns"], "feature_matrices", {"embedding": {"columns": ["a", "b"]}})
    delete = [("obsm", "embedding"), ("uns", "feature_matrices", "embedding")]
    if mixed:
        updated_obs = table.obs.assign(score=[4, 5])
        write_table_components(
            path, table_name="counts", components={("obs",): updated_obs}, delete=delete, overwrite=True
        )
    else:
        delete_table_components(path, table_name="counts", components=delete)
    reopened = read_table(path, table_name="counts", mode="eager")
    assert "embedding" not in reopened.obsm
    assert reopened.uns["feature_matrices"] == {}
    assert "frame" in reopened.obsm
    assert "embedding" in table.obsm
    assert "score" not in table.obs
    pd.testing.assert_frame_equal(reopened.obs, updated_obs if mixed else table.obs)


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_all_missing_deletions_are_logged_without_writes(make_table_io_store, monkeypatch, zarr_format):
    path = make_table_io_store(zarr_format=zarr_format)
    before = _store_bytes(path)
    messages = []
    monkeypatch.setattr(table_writer.log, "info", messages.append)

    def unexpected(*args, **kwargs):
        pytest.fail("An all-missing deletion must not stage, publish or write metadata.")

    with monkeypatch.context() as patch:
        patch.setattr(table_writer.tempfile, "mkdtemp", unexpected)
        patch.setattr(LocalStore, "set", unexpected)
        patch.setattr(LocalStore, "delete", unexpected)
        delete_table_components(
            path, table_name="counts", components=[("obsm", "missing"), ("uns", "missing", "nested")]
        )
    assert _store_bytes(path) == before
    assert len(messages) == 2
    assert all("Table 'counts'" in message and "already absent" in message for message in messages)
    assert "('obsm', 'missing')" in messages[0]


def test_missing_deletions_do_not_skip_other_updates(make_table_io_store):
    path = make_table_io_store()
    # Explicit deletion needs no overwrite permission; creating a new uns value
    # with None still writes an encoded value rather than deleting that path.
    write_table_components(
        path,
        table_name="counts",
        components={("uns", "new"): None},
        delete=[("obsm", "missing"), ("obsm", "embedding")],
    )
    group = zarr.open_group(str(path), mode="r", use_consolidated=True)["tables/counts"]
    assert "uns/new" in group
    assert "obsm/embedding" not in group
    delete_table_components(path, table_name="counts", components=[("uns", "new")])
    assert not (path / "tables/counts/uns/new").exists()


@pytest.mark.parametrize(
    "components",
    [
        [],
        [("obs",)],
        [("var",)],
        [("raw", "X")],
        [("raw", "var")],
        [("raw", "varm")],
        [("uns",)],
        [("uns", TableModel.ATTRS_KEY)],
        [("uns", TableModel.ATTRS_KEY, "region")],
        *[[(slot,)] for slot in ("layers", "obsm", "varm", "obsp", "varp")],
        [("X", "data")],
        [("obsm", "embedding", "0")],
        [("uns", "analysis", "../unsafe")],
        [("uns", "analysis", 7)],
        [("uns", "analysis", "zarr.json")],
        [("obsm", "missing"), ("obsm", "missing")],
        [("uns", "missing"), ("uns", "missing", "child")],
        [("raw", "varm", "loadings"), ("raw",)],
        [()],
        [["obsm", "embedding"]],
        "obsm",
    ],
)
def test_invalid_deletions_fail_without_changing_storage(make_table_io_store, components):
    path = make_table_io_store()
    before = _store_bytes(path)
    with pytest.raises((ValueError, TypeError)):
        delete_table_components(path, table_name="counts", components=components)
    assert _store_bytes(path) == before
    assert not list(path.parent.glob(f".{path.name}.harpy-*"))


@pytest.mark.parametrize(
    "replacement,deletion",
    [
        (("obsm", "missing"), ("obsm", "missing")),
        (("uns", "missing"), ("uns", "missing", "child")),
        (("uns", "missing", "child"), ("uns", "missing")),
        (("raw", "X"), ("raw",)),
    ],
)
def test_mixed_conflicts_rejected_before_skipping_absent_targets(make_table_io_store, replacement, deletion):
    path = make_table_io_store()
    before = _store_bytes(path)
    with pytest.raises(ValueError, match="non-overlapping"):
        write_table_components(path, table_name="counts", components={replacement: None}, delete=[deletion])
    assert _store_bytes(path) == before


@pytest.mark.parametrize("case", ["empty", "overwrite", "identity"])
def test_mixed_updates_keep_replacement_requirements(make_table_io_store, case):
    path = make_table_io_store()
    before = _store_bytes(path)
    options = {"components": {("obsm", "embedding"): np.ones((2, 2))}}
    if case == "empty":
        options["components"] = {}
    elif case == "identity":
        options.update(overwrite=True, obs_identity=["c2", "c1"])
    with pytest.raises((ValueError, FileExistsError)):
        write_table_components(path, table_name="counts", delete=[("raw",)], **options)
    assert _store_bytes(path) == before


@pytest.mark.parametrize("case", ["mapping", "version", "unrecognized", "symlink", "io_error"])
def test_bad_deletion_parents_are_not_treated_as_missing(make_table_io_store, monkeypatch, case):
    path = make_table_io_store()
    root = zarr.open_group(str(path), mode="r+", use_consolidated=False)
    group = root["tables/counts"]
    target = path / "tables/counts/uns/bad"
    if case == "mapping":
        write_elem(group["uns"], "bad", np.ones(2))
    elif case == "version":
        write_elem(group["uns"], "bad", {})
        group["uns/bad"].attrs["encoding-version"] = "unsupported"
    elif case == "unrecognized":
        target.mkdir()
    elif case == "symlink":
        target.symlink_to(path.parent / "missing", target_is_directory=True)
    else:
        original_get = LocalStore.get

        async def failed_get(self, key, *args, **kwargs):
            if key.startswith("tables/counts/uns/bad/"):
                raise PermissionError("injected I/O error")
            return await original_get(self, key, *args, **kwargs)

        monkeypatch.setattr(LocalStore, "get", failed_get)
    before = _store_bytes(path)
    with pytest.raises((ValueError, PermissionError)):
        delete_table_components(path, table_name="counts", components=[("uns", "bad", "child")])
    assert _store_bytes(path) == before
    if case == "symlink":
        assert target.is_symlink()


@pytest.mark.parametrize("missing", ["store", "table"])
def test_deletion_requires_existing_store_and_table(make_table_io_store, missing):
    path = make_table_io_store()
    before = _store_bytes(path)
    with pytest.raises(FileNotFoundError):
        delete_table_components(
            path / "missing" if missing == "store" else path,
            table_name="missing" if missing == "table" else "counts",
            components=[("X",)],
        )
    assert _store_bytes(path) == before


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("failure", ["staging", "backup", "publication", "installation", "finalization"])
def test_mixed_failure_restores_deletions_replacements_and_attachment(
    make_table_io_store, monkeypatch, zarr_format, failure
):
    """The caller owns memory rollback; the shared operation restores disk and root metadata."""
    path = make_table_io_store(zarr_format=zarr_format)
    zarr.consolidate_metadata(str(path))
    before = _store_bytes(path)
    previous = read_table(path, table_name="counts")
    sdata = SpatialData(tables={"counts": previous})
    original_rename, original_consolidate = Path.rename, zarr.consolidate_metadata
    original_write = table_writer._write_anndata_element

    def failed_write(*args, **kwargs):
        original_write(*args, **kwargs)
        raise RuntimeError("staging failure")

    def failed_rename(self, destination):
        if failure == "backup" and self == path / "tables/counts/raw":
            raise RuntimeError("backup failure")
        if failure == "publication" and "staging-" in str(self) and self.name == "component-0":
            raise RuntimeError("publication failure")
        return original_rename(self, destination)

    def failed_consolidate(*args, **kwargs):
        original_consolidate(*args, **kwargs)
        raise RuntimeError("finalization failure")

    monkeypatch.setattr(Path, "rename", failed_rename)
    if failure == "staging":
        monkeypatch.setattr(table_writer, "_write_anndata_element", failed_write)
    elif failure == "finalization":
        monkeypatch.setattr(zarr, "consolidate_metadata", failed_consolidate)
    with pytest.raises(RuntimeError, match=failure):
        try:
            with table_writer._write_table_operation(
                path,
                table_name="counts",
                components={("uns", "new_parent", "record"): {"value": 1}},
                delete=[("obsm", "embedding"), ("raw",)],
            ) as published:
                sdata.tables["counts"] = _read_anndata_table(published, mode="lazy")
                assert "embedding" not in sdata.tables["counts"].obsm
                assert sdata.tables["counts"].raw is None
                if failure == "installation":
                    raise RuntimeError("installation failure")
        except BaseException:
            sdata.tables["counts"] = previous
            raise
    assert sdata.tables["counts"] is previous
    assert _store_bytes(path) == before
    assert not list(path.parent.glob(f".{path.name}.harpy-*"))


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_deletion_only_finalization_failure_restores_store(make_table_io_store, monkeypatch, zarr_format):
    path = make_table_io_store(zarr_format=zarr_format)
    zarr.consolidate_metadata(str(path))
    before = _store_bytes(path)
    consolidate = zarr.consolidate_metadata

    def fail(*args, **kwargs):
        consolidate(*args, **kwargs)
        raise RuntimeError("finalization failure")

    monkeypatch.setattr(zarr, "consolidate_metadata", fail)
    with pytest.raises(RuntimeError, match="finalization failure"):
        delete_table_components(path, table_name="counts", components=[("X",), ("raw",)])
    assert _store_bytes(path) == before
    assert not list(path.parent.glob(f".{path.name}.harpy-*"))


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("matrix_kind", ["dense", "csr", "csc"])
@pytest.mark.parametrize("fail", [False, True])
def test_lazy_replacement_finishes_before_its_source_is_deleted(
    make_table_io_store, monkeypatch, zarr_format, matrix_kind, fail
):
    """old layer -> lazy X * 2 -> fully staged X -> backup/remove layer -> publish X.

    A finalization failure must restore both the source layer and the old X.
    Numerical reads are permitted here for the replacement, not for deletion.
    """
    path = make_table_io_store(zarr_format=zarr_format, matrix_kind=matrix_kind)
    zarr.consolidate_metadata(str(path))
    source = read_table(path, table_name="counts", sparse_chunks=1)
    expected = read_zarr(path / "tables/counts").layers["counts"] * 2
    before = _store_bytes(path)
    original_rename, original_consolidate = Path.rename, zarr.consolidate_metadata
    checked = []

    def checked_rename(self, destination):
        if self == path / "tables/counts/layers/counts":
            # The first replacement was fully serialized before the source layer
            # moves. Decode that staged value independently to check its contents.
            workspace = next(path.parent.glob(f".{path.name}.harpy-table-staging-*"))
            _assert_value(read_elem(zarr.open_group(str(workspace), mode="r")["component-0"]), expected)
            checked.append(True)
        return original_rename(self, destination)

    def failed_consolidate(*args, **kwargs):
        original_consolidate(*args, **kwargs)
        raise RuntimeError("finalization failure")

    monkeypatch.setattr(Path, "rename", checked_rename)
    if fail:
        monkeypatch.setattr(zarr, "consolidate_metadata", failed_consolidate)
    options = {
        "table_name": "counts",
        "components": {("X",): source.layers["counts"] * 2},
        "delete": [("layers", "counts")],
        "obs_identity": source.obs_names,
        "var_names": source.var_names,
        "overwrite": True,
    }
    if fail:
        with pytest.raises(RuntimeError, match="finalization"):
            write_table_components(path, **options)
        assert _store_bytes(path) == before
    else:
        write_table_components(path, **options)
        reopened = read_zarr(path / "tables/counts")
        _assert_value(reopened.X, expected)
        assert "counts" not in reopened.layers
    assert checked == [True]
