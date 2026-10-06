from copy import deepcopy
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import zarr
from anndata import AnnData
from dask.callbacks import Callback
from scipy import sparse
from spatialdata import SpatialData, read_zarr
from spatialdata.models import Labels2DModel, TableModel

import harpy.table.io._add_table as table_manager
import harpy.table.io._write as table_writer
from harpy._tests.test_table.test_io.test_read import _assert_value
from harpy._tests.test_table.test_io.test_write import _store_bytes
from harpy.table import read_table
from harpy.table.io._add_table import _cast_stringdtype_uns, add_table
from harpy.utils._keys import _INSTANCE_KEY, _REGION_KEY


@pytest.fixture
def annotated_sdata(tmp_path):
    """A small backed SpatialData: one labels element and a table annotating it.

    These tests only exercise one table, so they do not need the full
    transcriptomics example, whose images make it slow to build and write.
    """
    labels = Labels2DModel.parse(np.array([[0, 1, 1], [2, 2, 3]], dtype=np.uint32), dims=("y", "x"))
    obs = pd.DataFrame(
        {_REGION_KEY: pd.Categorical(["segmentation_mask"] * 3), _INSTANCE_KEY: [1, 2, 3]},
        index=["1", "2", "3"],
    )
    table = TableModel.parse(
        AnnData(X=np.arange(6, dtype=np.float32).reshape(3, 2), obs=obs),
        region="segmentation_mask",
        region_key=_REGION_KEY,
        instance_key=_INSTANCE_KEY,
    )
    path = tmp_path / "sdata.zarr"
    SpatialData(labels={"segmentation_mask": labels}, tables={"table_transcriptomics": table}).write(path)
    return read_zarr(path)


@pytest.mark.parametrize("is_backed", [True, False])
def test_add_table(annotated_sdata: SpatialData, recwarn, is_backed):
    assert annotated_sdata.is_backed()

    if not is_backed:
        annotated_sdata.path = None

    adata = annotated_sdata["table_transcriptomics"]

    annotated_sdata = add_table(
        annotated_sdata,
        adata=adata,
        output_table_name="table_transcriptomics",
        instance_key=_INSTANCE_KEY,
        region_key=_REGION_KEY,
        region=adata.obs[_REGION_KEY].cat.categories.to_list(),
        overwrite=True,
    )

    assert _INSTANCE_KEY == annotated_sdata["table_transcriptomics"].uns[TableModel.ATTRS_KEY][TableModel.INSTANCE_KEY]
    assert _REGION_KEY == annotated_sdata["table_transcriptomics"].uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY]

    assert ["segmentation_mask"] == annotated_sdata["table_transcriptomics"].uns[TableModel.ATTRS_KEY][
        TableModel.REGION_KEY
    ]

    userwarning_msg = f"The table is annotating {annotated_sdata['table_transcriptomics'].obs[_REGION_KEY].cat.categories.to_list()[0]}, which is not present in the SpatialData object."

    assert not any(isinstance(w.message, UserWarning) and str(w.message) == userwarning_msg for w in recwarn.list)


@pytest.mark.parametrize("is_backed", [True, False])
def test_add_table_change_region_instance_keys(annotated_sdata: SpatialData, recwarn, is_backed):
    assert annotated_sdata.is_backed()

    if not is_backed:
        annotated_sdata.path = None

    adata = annotated_sdata["table_transcriptomics"]

    # test if we can update the name of the instance and region keys.
    new_instance_key = "instance_key_test"
    new_region_key = "region_key_test"
    adata.obs.rename(
        columns={_REGION_KEY: new_region_key, _INSTANCE_KEY: new_instance_key},
        inplace=True,
    )
    # need to pop the spatialdata_attrs, otherwise harpy will complain that new region key does not match the old region key
    adata.uns.pop(TableModel.ATTRS_KEY, None)

    annotated_sdata = add_table(
        annotated_sdata,
        adata=adata,
        output_table_name="table_transcriptomics",
        instance_key=new_instance_key,
        region_key=new_region_key,
        region=adata.obs[new_region_key].cat.categories.to_list(),
        overwrite=True,
    )

    assert (
        new_instance_key == annotated_sdata["table_transcriptomics"].uns[TableModel.ATTRS_KEY][TableModel.INSTANCE_KEY]
    )
    assert (
        new_region_key == annotated_sdata["table_transcriptomics"].uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY]
    )

    assert ["segmentation_mask"] == annotated_sdata["table_transcriptomics"].uns[TableModel.ATTRS_KEY][
        TableModel.REGION_KEY
    ]

    userwarning_msg = f"The table is annotating {annotated_sdata['table_transcriptomics'].obs[new_region_key].cat.categories.to_list()[0]}, which is not present in the SpatialData object."

    assert not any(isinstance(w.message, UserWarning) and str(w.message) == userwarning_msg for w in recwarn.list)


@pytest.mark.parametrize("is_backed", [True, False])
def test_add_table_not_annotating(annotated_sdata: SpatialData, is_backed):
    assert annotated_sdata.is_backed()

    if not is_backed:
        annotated_sdata.path = None

    adata = annotated_sdata["table_transcriptomics"]

    annotated_sdata = add_table(
        annotated_sdata,
        adata=adata,
        output_table_name="table_transcriptomics",
        region=None,  # table is not annotating a region
        overwrite=True,
    )

    assert TableModel.ATTRS_KEY not in annotated_sdata["table_transcriptomics"].uns


def test_add_new_backed_table_does_not_warn_about_missing_regions(annotated_sdata: SpatialData, recwarn):
    adata = annotated_sdata["table_transcriptomics"].copy()

    annotated_sdata = add_table(
        annotated_sdata,
        adata=adata,
        output_table_name="table_transcriptomics_copy",
        instance_key=_INSTANCE_KEY,
        region_key=_REGION_KEY,
        region=adata.obs[_REGION_KEY].cat.categories.to_list(),
        overwrite=False,
    )

    assert "table_transcriptomics_copy" in annotated_sdata.tables

    userwarning_msg = (
        f"The table is annotating {adata.obs[_REGION_KEY].cat.categories.to_list()[0]!r}, "
        "which is not present in the SpatialData object."
    )
    assert not any(isinstance(w.message, UserWarning) and str(w.message) == userwarning_msg for w in recwarn.list)


def _string_dtype_array(values: list[str]) -> np.ndarray:
    if not hasattr(np, "dtypes") or not hasattr(np.dtypes, "StringDType"):
        pytest.skip("NumPy StringDType is not available in this NumPy version.")
    return np.asarray(values, dtype=np.dtypes.StringDType())


def test_cast_stringdtype_uns_keeps_u7_when_values_fit(sdata_transcripts_no_backed: SpatialData):
    key = "category_labels"
    sdata_transcripts_no_backed["table_transcriptomics"].uns[key] = _string_dtype_array(["#112233", "#aabbcc"])

    _cast_stringdtype_uns(sdata_transcripts_no_backed["table_transcriptomics"], target_dtype="U7")

    values = sdata_transcripts_no_backed["table_transcriptomics"].uns[key]
    assert isinstance(values, np.ndarray)
    assert values.dtype == np.dtype("U7")
    assert values.tolist() == ["#112233", "#aabbcc"]


def test_cast_stringdtype_uns_leaves_top_level_string_lists_untouched(sdata_transcripts_no_backed: SpatialData):
    key = "metadata"
    sdata_transcripts_no_backed["table_transcriptomics"].uns[key] = ["#11223344", "darkgreen"]

    _cast_stringdtype_uns(sdata_transcripts_no_backed["table_transcriptomics"], target_dtype="U7")

    values = sdata_transcripts_no_backed["table_transcriptomics"].uns[key]
    assert isinstance(values, list)
    assert values == ["#11223344", "darkgreen"]


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("matrix_kind", ["dense", "csr", "csc"])
@pytest.mark.parametrize("mode", ["eager", "lazy", "backed"])
@pytest.mark.parametrize("output_name", ["copy", "counts"])
def test_add_table_attaches_only_target_lazily_and_preserves_slots(
    make_table_io_store, monkeypatch, zarr_format, matrix_kind, mode, output_name
):
    """Create/replace one table, leaving both the caller's inputs and other tables alone.

    The store contains a malformed unrelated table, so reopening all tables would
    fail. The attached result must preserve every slot and use lazy matrices;
    reopening it must not execute a graph after the write has finished. Harpy's
    AnnData serializer must run exactly once into staging, and attachment must
    reopen the permanent table path.
    """
    path = make_table_io_store(zarr_format=zarr_format, matrix_kind=matrix_kind)
    original = read_table(path, table_name="counts", mode="eager")
    source = read_table(path, table_name="counts", mode=mode)
    source_matrix = source.X
    unrelated_bytes = _store_bytes(path / "tables" / "unrelated")
    untouched = AnnData(uns={"keep": {"value": 1}})
    sdata = SpatialData(tables={"untouched": untouched})
    sdata.path = path
    if output_name == "counts":
        sdata.tables[output_name] = source
    written_stores = []
    write_staged = table_writer._write_anndata_element
    read_published = table_manager._read_anndata_table

    def record_write(group, component_path, value, **kwargs):
        written_stores.append(Path(group.store.root))
        return write_staged(group, component_path, value, **kwargs)

    def guarded_reopen(group, **kwargs):
        assert group.name == f"/tables/{output_name}"

        def unexpected_compute(graph):
            pytest.fail("Reopening the published table must not compute its matrices.")

        with Callback(start=unexpected_compute):
            return read_published(group, **kwargs)

    monkeypatch.setattr(table_writer, "_write_anndata_element", record_write)
    monkeypatch.setattr(table_manager, "_read_anndata_table", guarded_reopen)
    result = add_table(sdata, source, output_name, region=None, overwrite=output_name == "counts")

    assert result is sdata
    assert len(written_stores) == 1
    assert not written_stores[0].is_relative_to(path)
    assert not list(path.parent.glob(f".{path.name}.harpy-*"))
    attached = sdata.tables[output_name]
    assert isinstance(attached.X, da.Array)
    assert isinstance(attached.raw.X, da.Array)
    for slot in ("layers", "obsm", "varm", "obsp", "varp"):
        for value in getattr(attached, slot).values():
            assert isinstance(value, (da.Array, pd.DataFrame))
        _assert_value(dict(getattr(attached, slot)), dict(getattr(original, slot)))
    for slot in ("X", "obs", "var", "uns"):
        _assert_value(getattr(attached, slot), getattr(original, slot))
    for slot in ("X", "var"):
        _assert_value(getattr(attached.raw, slot), getattr(original.raw, slot))
    _assert_value(dict(attached.raw.varm), dict(original.raw.varm))
    assert source.X is source_matrix
    pd.testing.assert_frame_equal(source.obs, original.obs)
    _assert_value(source.uns, original.uns)
    assert sdata.tables["untouched"] is untouched
    assert _store_bytes(path / "tables" / "unrelated") == unrelated_bytes
    root = zarr.open_group(str(path), mode="r", use_consolidated=True)
    assert root[f"tables/{output_name}"].metadata.zarr_format == zarr_format


def test_reopened_table_uses_the_readers_default_chunks(make_table_io_store):
    """add_table attaches its reopened table with the same lazy layout as read_table.

    With 2500 rows, the former fixed 1000-row sparse blocks would split X; the
    "auto" default reads it as one block. The dense obsm entry is read in whole rows.
    """
    path = make_table_io_store()
    sdata = SpatialData()
    sdata.path = path
    n_obs = 2500
    adata = AnnData(
        X=sparse.random(n_obs, 20, density=0.05, format="csr", dtype=np.float32, random_state=0),
        obsm={"features": np.arange(n_obs * 40, dtype=np.float32).reshape(n_obs, 40)},
    )

    add_table(sdata, adata, "large", region=None)

    attached = sdata.tables["large"]
    expected = read_table(path, table_name="large")
    assert attached.X.chunks == expected.X.chunks == ((n_obs,), (20,))
    assert attached.obsm["features"].chunks == expected.obsm["features"].chunks
    assert attached.obsm["features"].numblocks[1] == 1
    _assert_value(attached.X, adata.X)
    _assert_value(attached.obsm["features"], adata.obsm["features"])


@pytest.mark.parametrize("backed", [False, True])
@pytest.mark.parametrize("region", [None, ["cells"]])
def test_add_table_prepares_annotations_without_changing_input(make_table_io_store, backed, region):
    """Parsing/replacing linkage must not mutate supplied obs or nested uns."""
    path = make_table_io_store()
    source = read_table(path, table_name="counts")
    TableModel.parse(source, region="cells", region_key="region", instance_key="instance")
    # Parsing will convert this column back to categorical, but only in the result.
    source.obs["region"] = source.obs["region"].astype(object)
    original_obs = source.obs.copy(deep=True)
    original_uns = deepcopy(source.uns)
    sdata = SpatialData()
    if backed:
        sdata.path = path
    add_table(sdata, source, "prepared", region=region, region_key="region", instance_key="instance")
    attached = sdata.tables["prepared"]
    assert (TableModel.ATTRS_KEY in attached.uns) == (region is not None)
    if region is not None:
        assert isinstance(attached.obs["region"].dtype, pd.CategoricalDtype)
    pd.testing.assert_frame_equal(source.obs, original_obs)
    _assert_value(source.uns, original_uns)
    attached.uns["analysis"]["method"] = "changed locally"
    _assert_value(source.uns, original_uns)


def test_unbacked_add_table_preserves_lazy_view_without_writing(make_table_io_store, monkeypatch):
    """Attach a view's selected rows (including raw) without computing or writing."""
    path = make_table_io_store()
    source = read_table(path, table_name="counts")[:1, :2]
    before = _store_bytes(path)
    sdata = SpatialData(tables={"counts": AnnData()})

    def unexpected_write(*args, **kwargs):
        pytest.fail("Unbacked attachment must not invoke the writer.")

    def unexpected_compute(graph):
        pytest.fail("Unbacked attachment must not compute matrices.")

    monkeypatch.setattr(table_manager, "_write_table_operation", unexpected_write)
    with Callback(start=unexpected_compute):
        # Retain legacy replacement behavior even with overwrite=False.
        add_table(sdata, source, "counts", region=None, overwrite=False)
    attached = sdata.tables["counts"]
    assert attached.shape == (1, 2)
    assert attached.raw.shape == (1, 5)
    assert source.is_view
    _assert_value(attached.X, source.X.compute())
    _assert_value(attached.raw.X, source.raw.X.compute())
    assert _store_bytes(path) == before


@pytest.mark.parametrize("loaded", [False, True])
def test_backed_add_table_rejects_overwrite_even_if_table_not_loaded(make_table_io_store, loaded):
    path = make_table_io_store()
    sdata = SpatialData()
    sdata.path = path
    previous = read_table(path, table_name="counts") if loaded else None
    if loaded:
        sdata.tables["counts"] = previous
    before = _store_bytes(path)
    with pytest.raises((ValueError, FileExistsError), match="overwrite"):
        add_table(sdata, AnnData(), "counts", region=None)
    assert sdata.tables.get("counts") is previous
    assert _store_bytes(path) == before


def test_backed_add_table_replaces_a_table_attached_but_never_saved(make_table_io_store):
    """overwrite concerns the store: a table present only in memory is replaced without it."""
    path = make_table_io_store()
    sdata = SpatialData()
    sdata.path = path
    unsaved = read_table(path, table_name="counts", mode="eager")
    sdata.tables["unsaved"] = unsaved
    source = read_table(path, table_name="counts", mode="eager")
    add_table(sdata, source, "unsaved", region=None)
    assert sdata.tables["unsaved"] is not unsaved
    stored = read_table(path, table_name="unsaved", mode="eager")
    assert stored.shape == source.shape


@pytest.mark.parametrize("matrix_kind", ["dense", "csr", "csc"])
def test_add_table_lazy_self_overwrite_uses_old_values_and_attaches_new_result(make_table_io_store, matrix_kind):
    """Safely overwrite the table that supplies a lazy computation.

    Flow::

        counts/X (disk) --lazy * 3--> source.X
        source         --write-----> staging
        staging        --publish---> counts (disk)
        counts (disk)  --reopen-----> sdata.tables["counts"]

    Checks:
    - Attached and freshly read values: original values * 3.
    - Attached table: a new object, not source.
    - source.X: still the submitted matrix object.
    """
    path = make_table_io_store(matrix_kind=matrix_kind)
    source = read_table(path, table_name="counts")
    expected = source.X.compute() * 3
    source.X = source.X * 3
    submitted_matrix = source.X
    sdata = SpatialData(tables={"counts": source})
    sdata.path = path

    add_table(sdata, source, "counts", region=None, overwrite=True)

    assert sdata.tables["counts"] is not source
    assert source.X is submitted_matrix
    _assert_value(sdata.tables["counts"].X, expected)
    _assert_value(read_table(path, table_name="counts").X, expected)


@pytest.mark.parametrize("backed", [False, True])
def test_add_table_invalid_annotation_preserves_input_and_existing_entry(make_table_io_store, backed):
    path = make_table_io_store()
    source = read_table(path, table_name="counts")
    source.obs["instance"] = [1, 1]
    previous = AnnData()
    sdata = SpatialData(tables={"counts": previous})
    if backed:
        sdata.path = path
    before = _store_bytes(path)
    original_uns = deepcopy(source.uns)
    with pytest.raises(ValueError, match="unique"):
        add_table(
            sdata, source, "counts", region=["cells"], region_key="region", instance_key="instance", overwrite=True
        )
    assert sdata.tables["counts"] is previous
    _assert_value(source.uns, original_uns)
    assert _store_bytes(path) == before


@pytest.mark.parametrize("backed", [False, True])
def test_add_table_string_workaround_only_changes_prepared_target(make_table_io_store, monkeypatch, backed):
    path = make_table_io_store()
    values = _string_dtype_array(["longer than seven", "short"])
    source = AnnData(uns={"names": values})
    untouched = AnnData(uns={"names": values.copy()})
    sdata = SpatialData(tables={"untouched": untouched})
    if backed:
        sdata.path = path
    monkeypatch.setattr(table_manager, "_needs_stringdtype_copy_workaround", lambda: True)
    add_table(sdata, source, "new", region=None)
    converted = sdata.tables["new"].uns["names"]
    assert converted.dtype.kind == "U"
    assert converted.tolist() == values.tolist()
    assert source.uns["names"].dtype.kind == "T"
    assert untouched.uns["names"].dtype.kind == "T"


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("destination", ["missing_container", "new", "replace"])
@pytest.mark.parametrize("failure", ["staging", "parents", "publication", "reopening", "installation", "finalization"])
def test_add_table_failure_restores_disk_and_attached_table(
    tmp_path, make_table_io_store, monkeypatch, zarr_format, destination, failure
):
    """Keep disk bytes and live references unchanged across the entire adapter operation.

    Installation fails after assigning the replacement, and finalization fails
    after changing consolidated metadata. Both must restore the old in-memory
    entry as well as disk state; a failed first write must remove its new container.
    """
    if destination == "missing_container":
        path = tmp_path / "empty.zarr"
        root = zarr.open_group(str(path), mode="w", zarr_format=zarr_format)
        root.attrs["spatialdata_attrs"] = {"version": "0.2"}
    else:
        path = make_table_io_store(zarr_format=zarr_format)
    zarr.consolidate_metadata(str(path))
    before = _store_bytes(path)
    name = "counts" if destination == "replace" else "new"
    previous = read_table(path, table_name=name) if destination == "replace" else None
    untouched = AnnData(uns={"keep": True})
    sdata = SpatialData(tables={"untouched": untouched})
    sdata.path = path
    if previous is not None:
        sdata.tables[name] = previous
    source = AnnData(X=np.array([[2.0]]))
    original_write = table_writer._write_anndata_element
    original_parents = table_writer._create_destination_parents
    original_rename = Path.rename
    original_install = type(sdata.tables).__setitem__
    original_consolidate = zarr.consolidate_metadata
    consolidation_calls = []

    def failed_write(*args, **kwargs):
        original_write(*args, **kwargs)
        raise RuntimeError("staging failure")

    def failed_parents(*args, **kwargs):
        original_parents(*args, **kwargs)
        raise RuntimeError("parents failure")

    def failed_rename(self, target):
        if "staging-" in str(self) and self.name == "table":
            raise RuntimeError("publication failure")
        return original_rename(self, target)

    def failed_reopen(*args, **kwargs):
        raise RuntimeError("reopening failure")

    def failed_install(self, key, value):
        original_install(self, key, value)
        if self is sdata.tables and key == name and value is not previous:
            raise RuntimeError("installation failure")

    def failed_consolidate(*args, **kwargs):
        consolidation_calls.append(True)
        original_consolidate(*args, **kwargs)
        raise RuntimeError("finalization failure")

    if failure == "staging":
        monkeypatch.setattr(table_writer, "_write_anndata_element", failed_write)
    elif failure == "parents":
        monkeypatch.setattr(table_writer, "_create_destination_parents", failed_parents)
    elif failure == "publication":
        monkeypatch.setattr(Path, "rename", failed_rename)
    elif failure == "reopening":
        monkeypatch.setattr(table_manager, "_read_anndata_table", failed_reopen)
    elif failure == "installation":
        monkeypatch.setattr(type(sdata.tables), "__setitem__", failed_install)
    else:
        monkeypatch.setattr(zarr, "consolidate_metadata", failed_consolidate)

    with pytest.raises(RuntimeError, match=failure):
        add_table(sdata, source, name, region=None, overwrite=destination == "replace")
    assert _store_bytes(path) == before
    assert sdata.tables.get(name) is previous
    assert sdata.tables["untouched"] is untouched
    np.testing.assert_array_equal(source.X, [[2.0]])
    assert not list(path.parent.glob(f".{path.name}.harpy-*"))
    if failure == "finalization":
        assert len(consolidation_calls) == 1
