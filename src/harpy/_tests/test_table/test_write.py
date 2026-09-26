from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import zarr
from anndata import AnnData, read_zarr
from anndata._io.specs.registry import IORegistryError
from anndata.io import read_elem, write_elem
from dask import delayed
from dask.callbacks import Callback
from scipy import sparse
from spatialdata import SpatialData
from spatialdata.models import TableModel
from zarr.storage import LocalStore

import harpy.table._write as table_writer
from harpy._tests.test_table.test_io import _assert_value
from harpy.table import read_table, read_table_components, write_table, write_table_components


def _store_bytes(path):
    return {str(file.relative_to(path)): file.read_bytes() for file in path.rglob("*") if file.is_file()}


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("matrix_kind", ["dense", "csr", "csc"])
@pytest.mark.parametrize("mode", ["lazy", "backed"])
def test_complete_write_roundtrip_preserves_slots_and_input(make_table_io_store, zarr_format, matrix_kind, mode):
    """Round-trip a table through Harpy while preserving contents and inputs.

    counts (disk) --AnnData read--> original_table
    counts (disk) --Harpy read----> table_to_write --Harpy write--> copy (disk)
    copy   (disk) --AnnData read--> copied_table

    The Harpy read uses lazy/backed matrices; the AnnData reads are eager.
    Compare copied_table with original_table across all populated slots,
    including raw and sparse formats. Verify that table_to_write.obs and
    the original table's file bytes remain unchanged.
    """
    path = make_table_io_store(zarr_format=zarr_format, matrix_kind=matrix_kind)
    original_table = read_zarr(path / "tables" / "counts")
    original_table_bytes = _store_bytes(path / "tables" / "counts")
    table_to_write = read_table(path, table_name="counts", mode=mode, sparse_chunk_size=1)
    write_table(path, table_name="copy", adata=table_to_write)
    copied_table = read_zarr(path / "tables" / "copy")
    for slot in ("X", "obs", "var", "uns", "layers", "obsm", "varm", "obsp", "varp"):
        copied_value, original_value = getattr(copied_table, slot), getattr(original_table, slot)
        if slot in {"layers", "obsm", "varm", "obsp", "varp"}:
            copied_value, original_value = dict(copied_value), dict(original_value)
        _assert_value(copied_value, original_value)
    _assert_value(copied_table.raw.X, original_table.raw.X)
    _assert_value(copied_table.raw.var, original_table.raw.var)
    _assert_value(dict(copied_table.raw.varm), dict(original_table.raw.varm))
    pd.testing.assert_frame_equal(table_to_write.obs, original_table.obs)
    assert _store_bytes(path / "tables" / "counts") == original_table_bytes
    # Check that consolidated metadata includes the new table and these properties,
    # not that every consolidated entry matches its individual metadata file.
    root = zarr.open_group(str(path), mode="r", use_consolidated=True)
    assert root["tables/copy"].metadata.zarr_format == zarr_format
    assert root["tables/copy"].attrs["region"] is None


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("with_expression_matrix", [False, True])
def test_complete_write_creates_container_and_spatialdata_format(tmp_path, zarr_format, with_expression_matrix):
    """Write the first table into an empty on-disk SpatialData store.

    Start with only a Zarr root and SpatialData metadata, not an in-memory
    SpatialData object. The writer must create the missing tables container
    and write a table that SpatialData.read() can reopen, with or without .X.
    """
    path = tmp_path / "sdata.zarr"
    root = zarr.open_group(str(path), mode="w", zarr_format=zarr_format)
    root.attrs["spatialdata_attrs"] = {"version": "0.2"}
    root.attrs["custom"] = {"keep": True}
    obs = pd.DataFrame({"region": pd.Categorical(["cells", "cells"]), "instance": [1, 2]}, index=["a", "b"])
    matrix = np.array([[0, 1], [2, 0]], dtype=np.float32) if with_expression_matrix else None
    table = TableModel.parse(AnnData(X=matrix, obs=obs), region="cells", region_key="region", instance_key="instance")
    write_table(path, table_name="annotated", adata=table)
    reopened = SpatialData.read(path, selection=["tables"])
    pd.testing.assert_frame_equal(reopened.tables["annotated"].obs, table.obs)
    if with_expression_matrix:
        np.testing.assert_array_equal(reopened.tables["annotated"].X, matrix)
    else:
        assert reopened.tables["annotated"].X is None
    assert reopened.attrs["custom"] == {"keep": True}
    assert zarr.open_group(str(path), mode="r")["tables/annotated"].attrs["region"] == ["cells"]


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_complete_write_replaces_axes_and_removes_linkage(make_table_io_store, zarr_format):
    """A full replacement can change both axes and discards omitted slot contents and linkage."""
    path = make_table_io_store(zarr_format=zarr_format)
    _annotated_store(path)
    replacement = AnnData(
        X=np.array([[7]], dtype=np.float32),
        obs=pd.DataFrame(index=["new"]),
        var=pd.DataFrame(index=["new_gene"]),
    )
    write_table(path, table_name="counts", adata=replacement, overwrite=True)
    actual = read_table(path, table_name="counts")
    assert actual.obs_names.tolist() == ["new"]
    assert actual.var_names.tolist() == ["new_gene"]
    for slot in ("layers", "obsm", "varm", "obsp", "varp"):
        assert not getattr(actual, slot), slot
    assert actual.raw is None
    assert not actual.uns
    root = zarr.open_group(str(path), mode="r", use_consolidated=True)
    assert root["tables/counts"].metadata.zarr_format == zarr_format
    assert root["tables/counts"].attrs["region"] is None


@pytest.mark.parametrize("table_name", ["new", "counts"])
def test_complete_write_rejects_duplicate_observation_pairs_without_changing_store(make_table_io_store, table_name):
    """Repeated region/instance pairs must prevent both table creation and replacement."""
    path = make_table_io_store()
    table = read_table(path, table_name="counts")
    TableModel.parse(table, region="cells", region_key="region", instance_key="instance")
    # A valid table can acquire duplicate identities after it was parsed.
    table.obs["instance"] = [1, 1]
    before = _store_bytes(path)

    with pytest.raises(ValueError, match="region/instance pairs must be non-null and unique"):
        write_table(path, table_name=table_name, adata=table, overwrite=True)

    assert _store_bytes(path) == before


def test_complete_write_allows_instance_ids_shared_by_different_regions(make_table_io_store):
    """Uniqueness applies to region/instance pairs, not instance IDs alone."""
    path = make_table_io_store()
    table = _annotated_store(path)

    reopened = read_table(path, table_name="counts", mode="backed")
    pd.testing.assert_frame_equal(reopened.obs, table.obs)
    assert reopened.obs["instance"].tolist() == [1, 1]
    assert reopened.uns[TableModel.ATTRS_KEY]["region"] == ["cells_a", "cells_b"]


@pytest.mark.parametrize("scope", ["table", "components", "raw"])
@pytest.mark.parametrize("matrix_kind", ["dense", "csr", "csc"])
def test_lazy_self_overwrite_finishes_staging_before_publication(make_table_io_store, scope, matrix_kind):
    path = make_table_io_store(matrix_kind=matrix_kind)
    source = read_table(path, table_name="counts", sparse_chunk_size=1)
    expected = read_zarr(path / "tables/counts").X * 3
    source.X = source.X * 3
    if scope == "table":
        del source.layers["counts"]
        source.raw = None
        write_table(path, table_name="counts", adata=source, overwrite=True)
        actual = read_table(path, table_name="counts", mode="eager")
        assert not actual.layers and actual.raw is None
        _assert_value(actual.X, expected)
    elif scope == "raw":
        write_elem(zarr.open_group(str(path), mode="r+")["tables/counts"], "raw", None)
        write_table_components(
            path,
            table_name="counts",
            components={("raw", "X"): source.X, ("X",): source.X + source.X},
            obs_identity=source.obs_names,
            var_names=source.var_names,
            raw_var_names=source.var_names,
            overwrite=True,
        )
        actual = read_zarr(path / "tables/counts")
        _assert_value(actual.raw.X, expected)
        _assert_value(actual.X, expected * 2)
    else:
        old_siblings = _store_bytes(path / "tables/counts/layers")
        write_table_components(
            path,
            table_name="counts",
            components={("X",): source.X},
            obs_identity=source.obs_names,
            var_names=source.var_names,
            overwrite=True,
        )
        _assert_value(
            read_table_components(path, table_name="counts", components=[("X",)], mode="eager")[("X",)], expected
        )
        assert _store_bytes(path / "tables/counts/layers") == old_siblings


@pytest.mark.parametrize("matrix_kind", ["dense", "csr", "csc"])
@pytest.mark.parametrize("scope", ["X", "raw"])
def test_chunks_are_written_without_mutating_shared_input_blocks(make_table_io_store, monkeypatch, matrix_kind, scope):
    path = make_table_io_store()
    if scope == "raw":
        write_elem(zarr.open_group(str(path), mode="r+")["tables/counts"], "raw", None)
    expected = np.array([[1, 0, 2], [0, 3, 0]], dtype=np.uint32)
    axis = 1 if matrix_kind == "csc" else 0
    blocks = list(np.split(expected, expected.shape[axis], axis=axis))
    if matrix_kind != "dense":
        expected = getattr(sparse, f"{matrix_kind}_matrix")(expected)
        blocks = [getattr(sparse, f"{matrix_kind}_matrix")(block) for block in blocks]
    originals = [block.copy() for block in blocks]
    computed = []

    def load_block(index):
        computed.append(index)
        return blocks[index]

    matrix = da.concatenate(
        [
            da.from_delayed(delayed(load_block)(i), shape=block.shape, dtype=np.uint32, meta=block)
            for i, block in enumerate(blocks)
        ],
        axis=axis,
    )
    original_compute = da.Array.compute

    def guarded_compute(self, *args, **kwargs):
        assert self.shape != (2, 3), "The complete expression matrix must not be computed before writing."
        return original_compute(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(da.Array, "compute", guarded_compute)
        write_table_components(
            path,
            table_name="counts",
            components={("X",) if scope == "X" else ("raw", "X"): matrix},
            obs_identity=["c1", "c2"],
            **{"var_names" if scope == "X" else "raw_var_names": ["g1", "g2", "g3"]},
            overwrite=True,
        )
    assert sorted(computed) == list(range(len(blocks)))
    for block, original in zip(blocks, originals, strict=True):
        _assert_value(block, original)
        if sparse.issparse(block):
            assert block.indices.dtype == original.indices.dtype
    actual = read_zarr(path / "tables/counts")
    _assert_value(actual.X if scope == "X" else actual.raw.X, expected)


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_obs_update_never_reads_or_writes_expression_data(make_table_io_store, monkeypatch, zarr_format):
    """Guard direct Zarr payload access and Dask computation, not just the returned result."""
    path = make_table_io_store(zarr_format=zarr_format)
    table = read_table(path, table_name="counts")
    obs = table.obs.assign(score=[4, 5])
    before = _store_bytes(path / "tables/counts")
    original_get, original_partial, original_set = LocalStore.get, LocalStore.get_partial_values, LocalStore.set

    def check_read(key):
        if key.startswith("tables/counts/") and key.split("/")[2] in {
            "X",
            "layers",
            "obsm",
            "varm",
            "obsp",
            "varp",
            "raw",
        }:
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
        assert not key.startswith("tables/counts/X/"), key
        return await original_set(self, key, *args, **kwargs)

    def unexpected_compute(graph):
        pytest.fail("An annotation-only write must not compute matrix graphs.")

    with monkeypatch.context() as patch:
        patch.setattr(LocalStore, "get", guarded_get)
        patch.setattr(LocalStore, "get_partial_values", guarded_partial)
        patch.setattr(LocalStore, "set", guarded_set)
        with Callback(start=unexpected_compute):
            write_table_components(path, table_name="counts", components={("obs",): obs}, overwrite=True)
    after = _store_bytes(path / "tables/counts")
    assert {key: value for key, value in before.items() if not key.startswith("obs/")} == {
        key: value for key, value in after.items() if not key.startswith("obs/")
    }
    pd.testing.assert_frame_equal(read_zarr(path / "tables/counts").obs, obs)
    assert "score" not in table.obs


def test_component_writes_cover_matrix_axes_raw_and_mapping_replacement(make_table_io_store):
    path = make_table_io_store()
    table = read_table(path, table_name="counts", mode="eager")
    replacements = {
        ("X",): table.X * 2,
        ("layers", "counts"): table.layers["counts"] * 3,
        ("obsm", "embedding"): np.ones((2, 4)),
        ("varm", "loadings"): np.ones((3, 4)),
        ("obsp", "neighbors"): sparse.eye(2, format="csr"),
        ("varp", "neighbors"): np.zeros((3, 3)),
        ("obs",): table.obs.assign(score=[1, 2]),
        ("var",): table.var.assign(score=[1, 2, 3]),
        ("raw", "X"): table.raw.X * 4,
        ("raw", "var"): table.raw.var.assign(score=0),
        ("raw", "varm", "loadings"): np.ones((5, 4)),
        ("uns", "analysis"): {"replacement": True},
        ("uns", "new", "optional"): None,
    }
    # The supplied dataframes provide all three axes; no duplicate identity arguments.
    write_table_components(path, table_name="counts", components=replacements, overwrite=True)
    actual = read_table_components(path, table_name="counts", components=list(replacements), mode="eager")
    for key, value in replacements.items():
        _assert_value(actual[key], value)
    assert "method" not in read_zarr(path / "tables/counts").uns["analysis"]
    # None is serialized, not a deletion request.
    assert "tables/counts/uns/new/optional" in zarr.open_group(str(path), mode="r")
    write_table_components(
        path,
        table_name="counts",
        components={("X",): None},
        obs_identity=table.obs_names,
        var_names=table.var_names,
        overwrite=True,
    )
    assert read_table(path, table_name="counts").X is None


def _annotated_store(path):
    """Annotate two rows from different regions that intentionally share instance ID 1."""
    table = read_table(path, table_name="counts", mode="eager")
    table.obs["region"] = pd.Categorical(["cells_a", "cells_b"])
    table.obs["instance"] = [1, 1]
    TableModel.parse(table, region=["cells_a", "cells_b"], region_key="region", instance_key="instance")
    write_table(path, table_name="counts", adata=table, overwrite=True)
    return table


@pytest.mark.parametrize(
    "change", ["reordered", "duplicate", "missing", "float_ids", "string_ids", "obs_index", "conflicting_explicit"]
)
def test_annotation_identity_rejections_preserve_store(make_table_io_store, change):
    path = make_table_io_store()
    table = _annotated_store(path)
    identity = table.obs[["region", "instance"]].copy()
    values = {("obsm", "embedding"): np.ones((2, 2))}
    if change == "reordered":
        identity = identity.iloc[::-1]
    elif change == "duplicate":
        identity = identity.iloc[[0, 0]]
    elif change == "missing":
        identity = identity.iloc[:1]
    elif change == "float_ids":
        identity["instance"] = identity["instance"].astype(float)
    elif change == "string_ids":
        identity["instance"] = identity["instance"].astype(str)
    elif change == "obs_index":
        values[("obs",)] = table.obs.set_axis(["new1", "new2"])
    else:
        values[("obs",)] = table.obs
        identity = identity.iloc[::-1]
    before = _store_bytes(path)
    with pytest.raises((TypeError, ValueError)):
        write_table_components(path, table_name="counts", components=values, obs_identity=identity, overwrite=True)
    assert _store_bytes(path) == before


def test_annotated_pairs_not_identity_frame_index_define_alignment(make_table_io_store):
    path = make_table_io_store()
    table = _annotated_store(path)
    identity = table.obs[["region", "instance"]].set_axis(["ignored1", "ignored2"])
    write_table_components(
        path,
        table_name="counts",
        components={("obsm", "embedding"): np.ones((2, 2))},
        obs_identity=identity,
        overwrite=True,
    )
    # Duplicate observation names remain valid when region/instance pairs identify rows.
    table.obs_names = ["duplicate", "duplicate"]
    write_table(path, table_name="counts", adata=table, overwrite=True)
    write_table_components(
        path,
        table_name="counts",
        components={("obs",): table.obs.assign(score=[1, 2])},
        overwrite=True,
    )


@pytest.mark.parametrize(
    "components, identities, error",
    [
        ({("obsm", "embedding"): np.ones((2, 2))}, {}, "obs_identity"),
        ({("X",): np.ones((2, 3))}, {"obs_identity": ["c1", "c2"]}, "var_names"),
        (
            {("raw", "X"): np.ones((2, 5))},
            {"obs_identity": ["c1", "c2"], "var_names": ["g1", "g2", "g3"]},
            "raw_var_names",
        ),
        ({("obsm", "embedding"): np.ones((2, 2))}, {"obs_identity": ["c2", "c1"]}, "differ in order"),
        ({("obsm", "embedding"): np.ones((2, 2))}, {"obs_identity": ["c1", "c1"]}, "duplicate"),
        ({("varp", "neighbors"): np.ones((3, 2))}, {"var_names": ["g1", "g2", "g3"]}, "shape"),
        (
            {("layers", "counts"): np.ones((2, 4))},
            {"obs_identity": ["c1", "c2"], "var_names": ["g1", "g2", "g3"]},
            "shape",
        ),
        (
            {("obsm", "frame"): pd.DataFrame({"x": [1, 2]}, index=["c2", "c1"])},
            {"obs_identity": ["c1", "c2"]},
            "dataframe index",
        ),
        ({("uns", "test"): 1}, {"obs_identity": ["c2", "c1"]}, "differ in order"),
    ],
)
def test_component_identity_and_shape_contracts(make_table_io_store, components, identities, error):
    path = make_table_io_store()
    before = _store_bytes(path)
    with pytest.raises(ValueError, match=error):
        write_table_components(path, table_name="counts", components=components, overwrite=True, **identities)
    assert _store_bytes(path) == before


@pytest.mark.parametrize(
    "components",
    [
        {},
        {("obsm",): {}},
        {("obs", "score"): [1, 2]},
        {("X", "data"): [1]},
        {("uns", "analysis"): {}, ("uns", "analysis", "method"): "new"},
        {("uns", "analysis", "values", "child"): 1},
        {("uns", "../escape"): 1},
    ],
)
def test_invalid_component_scopes_are_rejected(make_table_io_store, components):
    path = make_table_io_store()
    before = _store_bytes(path)
    with pytest.raises(ValueError):
        write_table_components(path, table_name="counts", components=components, overwrite=True)
    assert _store_bytes(path) == before


@pytest.mark.parametrize("scope", ["table", "component"])
def test_overwrite_requires_permission(make_table_io_store, scope):
    path = make_table_io_store()
    before = _store_bytes(path)
    with pytest.raises(FileExistsError):
        if scope == "table":
            write_table(path, table_name="counts", adata=AnnData())
        else:
            write_table_components(path, table_name="counts", components={("uns", "analysis"): {}})
    assert _store_bytes(path) == before


@pytest.mark.parametrize("annotated", [False, True])
def test_spatial_annotation_cannot_be_added_removed_or_changed_by_components(make_table_io_store, annotated):
    path = make_table_io_store()
    table = _annotated_store(path) if annotated else read_table(path, table_name="counts", mode="eager")
    invalid = (
        ({("uns",): {}}, {("uns", TableModel.ATTRS_KEY, "region"): ["other"]})
        if annotated
        else (
            {("uns",): {TableModel.ATTRS_KEY: {}}},
            {("uns", TableModel.ATTRS_KEY): {}},
        )
    )
    before = _store_bytes(path)
    for components in invalid:
        with pytest.raises(ValueError, match="annotation"):
            write_table_components(path, table_name="counts", components=components, overwrite=True)
        assert _store_bytes(path) == before
    # Replacing uns is valid when the linkage is preserved.
    write_table_components(path, table_name="counts", components={("uns",): {**table.uns, "new": 1}}, overwrite=True)


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("scope", ["table", "components"])
@pytest.mark.parametrize("failure", ["staging", "parents", "publication", "finalization", "installation"])
def test_failed_update_restores_paths_parents_and_consolidation(
    make_table_io_store, monkeypatch, zarr_format, scope, failure
):
    """A coupled update restores exact store bytes, including consolidation and newly created parents."""
    path = make_table_io_store(zarr_format=zarr_format)
    zarr.consolidate_metadata(str(path))
    before = _store_bytes(path)
    original_write = table_writer._write_anndata_element
    original_rename = Path.rename
    original_consolidate = zarr.consolidate_metadata
    original_parents = table_writer._create_destination_parents

    def failed_write(*args, **kwargs):
        original_write(*args, **kwargs)
        raise RuntimeError("staging failure")

    def failed_rename(self, destination):
        if "staging-" in str(self) and (scope == "table" or "component-1" in str(self)):
            raise RuntimeError("publication failure")
        return original_rename(self, destination)

    def failed_consolidate(*args, **kwargs):
        original_consolidate(*args, **kwargs)
        raise RuntimeError("finalization failure")

    def failed_parents(*args, **kwargs):
        original_parents(*args, **kwargs)
        raise RuntimeError("parents failure")

    if failure == "staging":
        monkeypatch.setattr(table_writer, "_write_anndata_element", failed_write)
    elif failure == "parents":
        monkeypatch.setattr(table_writer, "_create_destination_parents", failed_parents)
    elif failure == "publication":
        monkeypatch.setattr(Path, "rename", failed_rename)
    elif failure == "finalization":
        monkeypatch.setattr(zarr, "consolidate_metadata", failed_consolidate)
    options = (
        {"adata": AnnData(obs=pd.DataFrame(index=["new"]))}
        if scope == "table"
        else {
            "components": {("obsm", "embedding"): np.zeros((2, 2)), ("uns", "new_parent", "record"): {"new": True}},
            "obs_identity": ["c1", "c2"],
        }
    )
    with pytest.raises(RuntimeError, match=failure):
        with table_writer._write_table_operation(path, table_name="counts", overwrite=True, **options) as group:
            assert group.name == "/tables/counts"
            if failure == "installation":
                raise RuntimeError("installation failure")
    assert _store_bytes(path) == before
    assert not list(path.parent.glob(f".{path.name}.harpy-*"))


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_failed_creation_removes_tables_container(tmp_path, monkeypatch, zarr_format):
    path = tmp_path / "sdata.zarr"
    zarr.open_group(str(path), mode="w", zarr_format=zarr_format)
    before = _store_bytes(path)

    def fail(*args, **kwargs):
        raise RuntimeError("consolidation failure")

    monkeypatch.setattr(zarr, "consolidate_metadata", fail)
    with pytest.raises(RuntimeError, match="consolidation"):
        write_table(path, table_name="new", adata=AnnData())
    assert _store_bytes(path) == before


def test_component_uns_write_does_not_validate_unrelated_axes(make_table_io_store):
    path = make_table_io_store()
    group = zarr.open_group(str(path), mode="r+")["tables/counts"]
    obs = read_elem(group["obs"])
    obs.index = ["duplicate", "duplicate"]
    write_elem(group, "obs", obs)
    write_table_components(path, table_name="counts", components={("uns", "new"): "allowed"})
    with pytest.raises(ValueError, match="duplicate"):
        write_table_components(
            path,
            table_name="counts",
            components={("obsm", "new"): np.ones((2, 2))},
            obs_identity=["duplicate", "duplicate"],
        )


@pytest.mark.parametrize("shape", [(0, 3), (2, 0)])
@pytest.mark.parametrize("matrix_kind", ["dense", "csr", "csc"])
def test_empty_matrix_axes_roundtrip(make_table_io_store, shape, matrix_kind):
    path = make_table_io_store()
    matrix = np.zeros(shape, dtype=np.uint32)
    if matrix_kind != "dense":
        matrix = getattr(sparse, f"{matrix_kind}_matrix")(matrix)
    table = AnnData(X=matrix)
    write_table(path, table_name="empty", adata=table)
    lazy = read_table(path, table_name="empty")
    write_table(path, table_name="empty", adata=lazy, overwrite=True)
    _assert_value(read_zarr(path / "tables/empty").X, matrix)


@pytest.mark.parametrize("case", ["missing_store", "missing_table", "collision", "symlink", "unsupported_value"])
def test_invalid_destinations_and_values_leave_store_unchanged(make_table_io_store, case):
    path = make_table_io_store()
    table_name = "counts"
    options = {"components": {("uns", "new"): 1}}
    error = ValueError
    if case == "missing_store":
        path = path.parent / "absent.zarr"
        error = FileNotFoundError
    elif case == "missing_table":
        table_name = "absent"
        error = FileNotFoundError
    elif case == "collision":
        zarr.open_group(str(path), mode="r+").create_group("images/counts")
    elif case == "symlink":
        (path / "tables/counts/uns/alias").symlink_to(path / "tables/counts/uns/analysis", target_is_directory=True)
        options["components"] = {("uns", "alias"): {"new": 1}}
    else:
        options["components"] = {("uns", "new"): object()}
        # AnnData rejects types with no registered encoding before publication.
        error = IORegistryError
    before = _store_bytes(path)
    with pytest.raises(error):
        write_table_components(path, table_name=table_name, overwrite=True, **options)
    assert _store_bytes(path) == before
    if case == "symlink":
        assert (path / "tables/counts/uns/alias").is_symlink()
    assert not list(path.parent.glob(f".{path.name}.harpy-*"))


def test_unrecognized_parent_directory_is_not_owned_by_writer(make_table_io_store):
    path = make_table_io_store()
    directory = path / "tables/counts/uns/external"
    directory.mkdir()
    (directory / "notes.txt").write_text("not a Zarr group")
    before = _store_bytes(path)
    with pytest.raises(ValueError, match="unrecognized path"):
        write_table_components(path, table_name="counts", components={("uns", "external", "record"): 1})
    assert _store_bytes(path) == before


def test_identity_index_name_and_equivalent_string_dtype_do_not_change_identity(make_table_io_store):
    path = make_table_io_store()
    write_table_components(
        path,
        table_name="counts",
        components={("X",): np.ones((2, 3))},
        obs_identity=pd.Index(["c1", "c2"], dtype="string", name="different_name"),
        var_names=pd.Index(["g1", "g2", "g3"], dtype="string"),
        overwrite=True,
    )
    np.testing.assert_array_equal(read_table(path, table_name="counts", mode="eager").X, np.ones((2, 3)))


def test_unrecognized_component_destination_still_requires_overwrite(make_table_io_store):
    path = make_table_io_store()
    destination = path / "tables/counts/uns/external"
    destination.mkdir()
    (destination / "notes.txt").write_text("not a Zarr value")
    before = _store_bytes(path)
    with pytest.raises(FileExistsError):
        write_table_components(path, table_name="counts", components={("uns", "external"): 1})
    assert _store_bytes(path) == before


def test_root_metadata_symlink_is_not_written_through(make_table_io_store):
    path = make_table_io_store()
    metadata = path / "zarr.json"
    original = path.parent / "external_metadata.json"
    metadata.rename(original)
    metadata.symlink_to(original)
    before = original.read_bytes()
    with pytest.raises(ValueError, match="symbolic-link metadata"):
        write_table(path, table_name="new", adata=AnnData())
    assert metadata.is_symlink() and original.read_bytes() == before
    assert not (path / "tables/new").exists()
