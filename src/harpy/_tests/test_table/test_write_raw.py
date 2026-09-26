"""Raw creation preserves the parent table and shares component-write rollback."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import zarr
from anndata import read_zarr
from anndata.io import write_elem
from scipy import sparse
from zarr.storage import LocalStore

import harpy.table._write as table_writer
from harpy._storage._anndata import _read_anndata_element
from harpy._tests.test_table.test_io import _assert_value
from harpy._tests.test_table.test_write import _annotated_store, _store_bytes
from harpy.table import read_table, write_table_components


def _without_raw(path, state):
    """Prepare either physical absence or an encoded None in the temporary fixture."""
    group = zarr.open_group(str(path), mode="r+", use_consolidated=False)["tables/counts"]
    if state == "absent":
        del group["raw"]
    else:
        write_elem(group, "raw", None)
    return group


@pytest.mark.parametrize(
    "zarr_format,matrix_kind,mode,state,with_frame",
    [
        (2, "dense", "eager", "absent", False),
        (2, "csr", "lazy", "null", True),
        (2, "csc", "backed", "absent", True),
        (3, "dense", "backed", "null", False),
        (3, "csr", "eager", "absent", True),
        (3, "csc", "lazy", "null", False),
    ],
)
def test_raw_creation_roundtrip(make_table_io_store, zarr_format, matrix_kind, mode, state, with_frame):
    """New raw shares observations, but may have a different feature axis from main X."""
    path = make_table_io_store(zarr_format=zarr_format)
    parent = _annotated_store(path) if with_frame else read_table(path, table_name="counts")
    _without_raw(path, state)
    matrix = np.arange(8, dtype=np.uint32).reshape(2, 4)
    if matrix_kind != "dense":
        matrix = getattr(sparse, f"{matrix_kind}_matrix")(matrix)
    root = zarr.open_group(str(path), mode="r+", use_consolidated=False)
    write_elem(root, "source", matrix)
    payload = _read_anndata_element(root, ("source",), mode=mode)
    names = pd.Index(["raw_d", "raw_b", "raw_a", "raw_c"], name="gene")
    frame = pd.DataFrame({"kind": ["a", "a", "b", "b"]}, index=names)
    original_frame = frame.copy(deep=True)
    components = {("raw", "X"): payload, ("raw", "varm", "loadings"): np.ones((4, 2))}
    if with_frame:
        components[("raw", "var")] = frame
        # Supplying obs provides the row identity context, without a second argument.
        components[("obs",)] = parent.obs
        identities = {}
    else:
        identities = {"obs_identity": parent.obs_names, "raw_var_names": names}
    before = _store_bytes(path)
    write_table_components(
        path, table_name="counts", components=components, overwrite=state == "null" or with_frame, **identities
    )
    actual = read_zarr(path / "tables/counts")
    _assert_value(actual.raw.X, matrix)
    if with_frame:
        pd.testing.assert_frame_equal(actual.raw.var, original_frame)
    else:
        pd.testing.assert_index_equal(actual.raw.var_names, names)
        assert actual.raw.var.shape == (4, 0)
    _assert_value(actual.raw.varm["loadings"], np.ones((4, 2)))
    pd.testing.assert_frame_equal(frame, original_frame)
    after = _store_bytes(path)
    # Root consolidation and explicitly supplied obs may change; unrelated payloads must not.
    preserved = [
        key
        for key in before
        if key.startswith(("tables/", "source/")) and not key.startswith(("tables/counts/raw/", "tables/counts/obs/"))
    ]
    assert {key: after[key] for key in preserved} == {key: before[key] for key in preserved}
    assert (("raw", "var") in components) == with_frame
    reopened = read_table(path, table_name="counts")
    assert reopened.raw.shape == (2, 4)
    root = zarr.open_group(str(path), mode="r", use_consolidated=True)
    assert root["tables/counts/raw"].attrs["encoding-type"] == "raw"


@pytest.mark.parametrize(
    "case,match",
    [
        ("missing_matrix", "requires a non-None"),
        ("null_matrix", "requires a non-None"),
        ("varm_only", "requires a non-None"),
        ("missing_features", "requires raw_var_names"),
        ("duplicate_features", "duplicate"),
        ("null_feature", "non-null string"),
        ("conflicting_features", "differ in order"),
        ("missing_obs", "obs_identity"),
        ("reordered_obs", "differ in order"),
        ("rows", "shape"),
        ("columns", "shape"),
        ("varm_shape", "shape"),
    ],
)
def test_invalid_raw_creation_preserves_store(make_table_io_store, case, match):
    path = make_table_io_store()
    _without_raw(path, "absent")
    components = {("raw", "X"): np.ones((2, 4))}
    options = {"obs_identity": ["c1", "c2"], "raw_var_names": ["d", "b", "a", "c"]}
    if case == "missing_matrix":
        components = {("raw", "var"): pd.DataFrame(index=options["raw_var_names"])}
    elif case == "null_matrix":
        components[("raw", "X")] = None
    elif case == "varm_only":
        components = {("raw", "varm", "loadings"): np.ones((4, 2))}
    elif case == "missing_features":
        del options["raw_var_names"]
    elif case == "duplicate_features":
        options["raw_var_names"] = ["a"] * 4
    elif case == "null_feature":
        options["raw_var_names"] = ["a", "b", None, "d"]
    elif case == "conflicting_features":
        components[("raw", "var")] = pd.DataFrame(index=options["raw_var_names"][::-1])
    elif case == "missing_obs":
        del options["obs_identity"]
    elif case == "reordered_obs":
        options["obs_identity"] = ["c2", "c1"]
    elif case == "rows":
        components[("raw", "X")] = np.ones((3, 4))
    elif case == "columns":
        components[("raw", "X")] = np.ones((2, 3))
    else:
        components[("raw", "varm", "loadings")] = np.ones((3, 2))
    before = _store_bytes(path)
    with pytest.raises(ValueError, match=match):
        write_table_components(path, table_name="counts", components=components, **options)
    assert _store_bytes(path) == before


def test_raw_creation_rejects_staged_feature_index_mismatch(make_table_io_store, monkeypatch):
    """Read-back validation compares raw.var with the intended index, not with itself."""
    path = make_table_io_store()
    _without_raw(path, "absent")
    frame = pd.DataFrame(index=["a", "b", "c", "d"])
    original_frame = frame.copy(deep=True)
    before = _store_bytes(path)
    original_write = table_writer._write_anndata_element

    def write_with_reordered_raw_var(group, component_path, value, **kwargs):
        original_write(group, component_path, value, **kwargs)
        if component_path == ("raw",):
            # Change only the serialized index; the intended feature order stays intact.
            staged_var = _read_anndata_element(group, ("raw", "var"), mode="eager")
            write_elem(group["raw"], "var", staged_var.iloc[::-1])

    monkeypatch.setattr(table_writer, "_write_anndata_element", write_with_reordered_raw_var)
    with pytest.raises(ValueError, match="raw_var dataframe index identities match but differ in order"):
        write_table_components(
            path,
            table_name="counts",
            components={("raw", "X"): np.ones((2, 4)), ("raw", "var"): frame},
            obs_identity=["c1", "c2"],
        )
    assert _store_bytes(path) == before
    pd.testing.assert_frame_equal(frame, original_frame)


@pytest.mark.parametrize(
    "state", ["null", "wrong_type", "wrong_version", "malformed_null", "unrecognized_path", "existing_axis"]
)
def test_raw_creation_respects_existing_destination(make_table_io_store, state):
    path = make_table_io_store()
    group = zarr.open_group(str(path), mode="r+")["tables/counts"]
    if state == "null":
        write_elem(group, "raw", None)
    elif state == "wrong_type":
        write_elem(group, "raw", {})
    elif state == "wrong_version":
        group["raw"].attrs["encoding-version"] = "unsupported"
    elif state == "malformed_null":
        group["raw"].attrs["encoding-type"] = "null"
    elif state == "unrecognized_path":
        del group["raw"]
        raw_path = path / "tables/counts/raw"
        raw_path.mkdir()
        (raw_path / "user-file").write_text("keep")
    before = _store_bytes(path)
    with pytest.raises(FileExistsError if state == "null" else ValueError):
        write_table_components(
            path,
            table_name="counts",
            components={("raw", "X"): np.ones((2, 4))},
            obs_identity=["c1", "c2"],
            raw_var_names=["a", "b", "c", "d"],
            overwrite=state != "null",
        )
    assert _store_bytes(path) == before


def test_existing_raw_varm_update_preserves_other_raw_components(make_table_io_store):
    path = make_table_io_store()
    before = _store_bytes(path / "tables/counts/raw")
    values = np.ones((5, 3))
    write_table_components(
        path,
        table_name="counts",
        components={("raw", "varm", "new_loadings"): values},
        raw_var_names=[f"raw{i}" for i in range(5)],
    )
    after = _store_bytes(path / "tables/counts/raw")
    assert {key: after[key] for key in before} == before
    _assert_value(read_zarr(path / "tables/counts").raw.varm["new_loadings"], values)


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_raw_creation_does_not_access_unrelated_matrix_payloads(make_table_io_store, monkeypatch, zarr_format):
    path = make_table_io_store(zarr_format=zarr_format)
    _without_raw(path, "absent")
    original_get, original_partial, original_set = LocalStore.get, LocalStore.get_partial_values, LocalStore.set
    prefixes = tuple(f"tables/counts/{slot}/" for slot in ("X", "layers", "obsm", "varm", "obsp", "varp"))

    def check_read(key):
        if key.startswith(prefixes):
            assert key.rsplit("/", 1)[-1] in {".zarray", ".zattrs", ".zgroup", "zarr.json"}, key

    async def guarded_get(self, key, *args, **kwargs):
        check_read(key)
        return await original_get(self, key, *args, **kwargs)

    async def guarded_partial(self, prototype, key_ranges):
        key_ranges = list(key_ranges)
        for key, _ in key_ranges:
            check_read(key)
        return await original_partial(self, prototype, key_ranges)

    async def guarded_set(self, key, *args, **kwargs):
        assert not key.startswith(prefixes), key
        return await original_set(self, key, *args, **kwargs)

    # Guard physical reads as well as writes; metadata inspection for consolidation is allowed.
    with monkeypatch.context() as patch:
        patch.setattr(LocalStore, "get", guarded_get)
        patch.setattr(LocalStore, "get_partial_values", guarded_partial)
        patch.setattr(LocalStore, "set", guarded_set)
        write_table_components(
            path,
            table_name="counts",
            components={("raw", "X"): np.ones((2, 4))},
            obs_identity=["c1", "c2"],
            raw_var_names=["a", "b", "c", "d"],
        )


@pytest.mark.parametrize("state", ["absent", "null"])
@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("failure", ["staging", "publication", "finalization"])
def test_raw_creation_failure_restores_absence_or_null(make_table_io_store, monkeypatch, state, zarr_format, failure):
    """A new raw subtree and an uns replacement roll back together, including root consolidation."""
    path = make_table_io_store(zarr_format=zarr_format)
    _without_raw(path, state)
    zarr.consolidate_metadata(str(path))
    before = _store_bytes(path)
    original_write = table_writer._write_anndata_element
    original_rename = Path.rename
    original_consolidate = zarr.consolidate_metadata

    def failed_write(*args, **kwargs):
        original_write(*args, **kwargs)
        raise RuntimeError("staging failure")

    def failed_rename(self, destination):
        # Fail on the second payload, after raw has already been published.
        if "staging-" in str(self) and self.name == "component-1":
            raise RuntimeError("publication failure")
        return original_rename(self, destination)

    def failed_consolidate(*args, **kwargs):
        original_consolidate(*args, **kwargs)
        raise RuntimeError("finalization failure")

    if failure == "staging":
        monkeypatch.setattr(table_writer, "_write_anndata_element", failed_write)
    elif failure == "publication":
        monkeypatch.setattr(Path, "rename", failed_rename)
    else:
        monkeypatch.setattr(zarr, "consolidate_metadata", failed_consolidate)
    with pytest.raises(RuntimeError, match=failure):
        write_table_components(
            path,
            table_name="counts",
            components={("raw", "X"): np.ones((2, 4)), ("uns", "analysis"): {"method": "new"}},
            obs_identity=["c1", "c2"],
            raw_var_names=["a", "b", "c", "d"],
            overwrite=True,
        )
    assert _store_bytes(path) == before
    assert not list(path.parent.glob(f".{path.name}.harpy-*"))
