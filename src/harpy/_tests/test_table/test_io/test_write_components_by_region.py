from contextlib import contextmanager
from pathlib import Path

import dask
import dask.array as da
import numpy as np
import pandas as pd
import pytest
import zarr
from anndata import read_zarr, settings
from anndata.io import write_elem
from dask import delayed
from dask.callbacks import Callback
from scipy import sparse
from spatialdata.models import TableModel
from zarr.storage import LocalStore

import harpy.table.io._write as table_writer
import harpy.table.io._write_by_region as regional_writer
from harpy._tests.test_table.test_io.test_write import _store_bytes
from harpy.table.io import read_table_components, write_table_components, write_table_components_by_region


def _input_matrix(tmp_path, values, matrix_format, mode):
    matrix = values if matrix_format == "dense" else getattr(sparse, f"{matrix_format}_matrix")(values)
    if mode == "memory":
        return matrix
    if mode == "lazy":
        # Neither rows nor columns align with the stored three-unit chunks.
        return da.from_array(matrix, chunks=(2, 2), asarray=False)
    group = zarr.open_group(str(tmp_path / "input.zarr"), mode="w")
    # Dense stored chunks of (2, 2) split rows and columns, so the readers' row
    # blocks of whole stored chunks differ from the stored layout.
    write_elem(group, "matrix", matrix, dataset_kwargs={"chunks": (2, 2)} if matrix_format == "dense" else {})
    return regional_writer._decode_anndata_element(group["matrix"], mode="backed")


def _prepared_chunks(monkeypatch, path, *, component, payload, identity, create, **settings):
    """Return the blocks of the complete replacement matrix, captured before serialization."""
    original_operation = regional_writer._write_table_operation
    prepared_chunks = []

    @contextmanager
    def capture_chunks(*args, **kwargs):
        prepared_chunks.append(kwargs["components"][component].chunks)
        with original_operation(*args, **kwargs) as published:
            yield published

    monkeypatch.setattr(regional_writer, "_write_table_operation", capture_chunks)
    write_table_components_by_region(
        path,
        table_name="counts",
        components={component: payload},
        obs_identity=identity,
        fill_values={component: 0} if create else None,
        overwrite=not create,
        **settings,
    )
    (chunks,) = prepared_chunks
    return chunks


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("matrix_format", ["dense", "csr", "csc"])
@pytest.mark.parametrize("mode", ["memory", "lazy", "backed"])
@pytest.mark.parametrize("create", [False, True])
def test_regional_merge_is_independent_of_chunk_boundaries(
    regional_store, tmp_path, zarr_format, matrix_format, mode, create
):
    """Selected rows map by table position, never by corresponding source blocks.

    A crosses the boundary between stored rows 2 and 3, and reappears at row 6.
    Its four supplied rows have different chunk boundaries. Compare every output
    value against an independent dense reference, including B-only chunks, an
    existing NaN, and zeros that must replace old nonzero measurements.
    """
    path, table, identity, old = regional_store(matrix_format, zarr_format)
    values = np.arange(20, dtype=np.float32).reshape(4, 5) * 10
    values[0] = 0
    payload = _input_matrix(tmp_path, values, matrix_format, mode)
    component = ("obsm", "new" if create else "features")
    before = _store_bytes(path / "tables/counts")
    identity.index = ["ignored"] * len(identity)
    expected = np.zeros_like(old) if create else old.copy()
    expected[[1, 2, 3, 6]] = values
    write_table_components_by_region(
        path,
        table_name="counts",
        components={component: payload, ("uns", "analysis", "method"): "updated"},
        obs_identity=identity,
        fill_values={component: 0} if create else None,
        sparse_chunks=3,
        dense_chunks=3,
        overwrite=True,
    )
    reopened = read_zarr(path / "tables/counts")
    result = reopened.obsm[component[1]]
    assert ("dense" if isinstance(result, np.ndarray) else result.format) == matrix_format
    assert result.dtype == expected.dtype
    np.testing.assert_array_equal(result if matrix_format == "dense" else result.toarray(), expected)
    pd.testing.assert_frame_equal(reopened.obs, table.obs)
    assert reopened.uns["analysis"]["method"] == "updated"
    after = _store_bytes(path / "tables/counts")
    assert {key: value for key, value in before.items() if key.startswith(("X/", "obsm/unrelated/"))} == {
        key: value for key, value in after.items() if key.startswith(("X/", "obsm/unrelated/"))
    }
    original = payload.compute() if isinstance(payload, da.Array) else payload
    if mode == "backed":
        original = original[:] if matrix_format == "dense" else original.to_memory()
    np.testing.assert_array_equal(original if matrix_format == "dense" else original.toarray(), values)


@pytest.mark.parametrize("matrix_format", ["dense", "csr", "csc"])
@pytest.mark.parametrize("mode", ["memory", "lazy", "backed"])
def test_lazy_matrix_applies_integer_settings_to_all_but_dask_inputs(tmp_path, matrix_format, mode):
    """An integer sets rows per dense or CSR block and columns per CSC block.

    Dask inputs keep their blocks. A storage-backed dense input is split as the
    readers split it: 3 rows round down to one whole stored chunk of 2 rows.
    """
    values = np.ones((4, 5), dtype=np.float32)
    payload = _input_matrix(tmp_path, values, matrix_format, mode)
    lazy = regional_writer._lazy_matrix(payload, matrix_format, sparse_chunks=3, dense_chunks=3)
    if mode == "lazy":
        assert lazy is payload
    elif mode == "backed" and matrix_format == "dense":
        assert lazy.chunks == ((2, 2), (5,))
    else:
        assert lazy.chunks == (((4,), (3, 2)) if matrix_format == "csc" else ((3, 1), (5,)))


def test_lazy_matrix_storage_setting_keeps_stored_chunks_of_storage_backed_dense_inputs_only(tmp_path):
    """``"storage"`` keeps the stored chunks of a dense zarr.Array.

    An in-memory matrix has no stored chunks to keep, so it is split as with ``"auto"``.
    """
    values = np.ones((4, 5), dtype=np.float32)
    backed = _input_matrix(tmp_path, values, "dense", "backed")
    storage = {"sparse_chunks": "auto", "dense_chunks": "storage"}
    assert regional_writer._lazy_matrix(backed, "dense", **storage).chunks == ((2, 2), (2, 2, 1))
    # The readers' row blocks of whole stored chunks, for comparison.
    assert regional_writer._lazy_matrix(backed, "dense", sparse_chunks="auto", dense_chunks=3).chunks == (
        (2, 2),
        (5,),
    )
    auto = {"sparse_chunks": "auto", "dense_chunks": "auto"}
    assert (
        regional_writer._lazy_matrix(values, "dense", **storage).chunks
        == regional_writer._lazy_matrix(values, "dense", **auto).chunks
    )


def test_plain_zarr_arrays_are_rejected_and_work_when_wrapped_in_dask(regional_store, tmp_path):
    """A zarr.Array without AnnData encoding is rejected before anything is written.

    Storage-backed values are split by the readers' decoder, which needs the
    encoding. Wrapped in da.from_zarr, the array is a lazy input instead.
    """
    path, _, identity, old = regional_store()
    values = np.arange(20, dtype=np.float32).reshape(4, 5)
    plain = zarr.open_group(str(tmp_path / "plain.zarr"), mode="w").create_array(
        "matrix", shape=values.shape, chunks=(2, 2), dtype=values.dtype
    )
    plain[:] = values
    options = {"table_name": "counts", "obs_identity": identity, "overwrite": True}
    before = _store_bytes(path)
    with pytest.raises(TypeError, match="without AnnData encoding; wrap it with dask.array.from_zarr"):
        write_table_components_by_region(path, components={("obsm", "features"): plain}, **options)
    assert _store_bytes(path) == before
    write_table_components_by_region(path, components={("obsm", "features"): da.from_zarr(plain)}, **options)
    expected = old.copy()
    expected[[1, 2, 3, 6]] = values
    np.testing.assert_array_equal(read_zarr(path / "tables/counts").obsm["features"], expected)


@pytest.mark.parametrize(
    "matrix_format, expected", [("dense", ((4,), (5,))), ("csr", ((2, 2), (5,))), ("csc", ((4,), (2, 2, 1)))]
)
def test_lazy_matrix_auto_sizes_in_memory_values_from_array_chunk_size(tmp_path, matrix_format, expected):
    """``"auto"`` sizes in-memory blocks from Dask's ``array.chunk-size``, here 100 bytes.

    For a 4 × 5 float32 matrix of ones:

    - dense: 20 bytes per row, so 5 rows, capped at the 4 rows there are;
    - CSR: 5 values per row of 8 bytes (float32 data and int32 index) plus
      4 bytes of indptr, 44 bytes, so 2 rows;
    - CSC: 4 values per column, 36 bytes, so 2 columns.
    """
    values = _input_matrix(tmp_path, np.ones((4, 5), dtype=np.float32), matrix_format, "memory")
    with dask.config.set({"array.chunk-size": "100B"}):
        lazy = regional_writer._lazy_matrix(values, matrix_format, sparse_chunks="auto", dense_chunks="auto")
    assert lazy.chunks == expected


@pytest.mark.parametrize("matrix_format", ["dense", "csr", "csc"])
@pytest.mark.parametrize("create", [False, True])
def test_integer_settings_control_merge_layout_of_existing_and_new_entries(
    regional_store, monkeypatch, matrix_format, create
):
    """An integer applies to the existing matrix and new entries alike.

    With 2, CSR entries use two-row blocks spanning all columns, CSC entries
    two-column blocks spanning all rows, and a new dense entry two-row blocks.
    An existing dense matrix, stored in chunks of (3, 2), is read as
    ``read_table`` reads it: 2 rows round to one whole stored chunk of 3 rows,
    spanning all columns. Capture the complete replacement matrix's blocks
    before serialization, not its eventual on-disk chunks or the input's blocks.
    """
    path, _, identity, _ = regional_store(matrix_format)
    component = ("obsm", "new" if create else "features")
    values = np.ones((4, 5), dtype=np.float32)
    payload = values if matrix_format == "dense" else getattr(sparse, f"{matrix_format}_matrix")(values)
    chunks = _prepared_chunks(
        monkeypatch,
        path,
        component=component,
        payload=payload,
        identity=identity,
        create=create,
        sparse_chunks=np.int64(2),
        dense_chunks=np.int64(2),
    )
    if matrix_format == "dense" and not create:
        expected = ((3, 3, 3, 3), (5,))
    elif matrix_format == "csc":
        expected = ((12,), (2, 2, 1))
    else:
        expected = ((2, 2, 2, 2, 2, 2), (5,))
    assert chunks == expected


@pytest.mark.parametrize("matrix_format", ["dense", "csr", "csc"])
@pytest.mark.parametrize("create", [False, True])
def test_auto_settings_read_existing_matrices_as_read_table_and_size_new_entries(
    regional_store, monkeypatch, matrix_format, create
):
    """With the default ``"auto"``, an existing matrix keeps the blocks ``read_table`` gives it.

    New entries are sized from ``array.chunk-size``, here 64 bytes, for the
    full 12-row output of the 4 × 5 float32 input of ones:

    - dense: 20 bytes per row, so 3 rows;
    - CSR: 5 values per selected row, 8 bytes each (float32 data and int32
      index), plus 4 bytes of indptr, 44 bytes per row, so 1 row;
    - CSC: 4 values per column, 36 bytes, so 1 column.
    """
    path, _, identity, _ = regional_store(matrix_format)
    component = ("obsm", "new" if create else "features")
    values = np.ones((4, 5), dtype=np.float32)
    payload = values if matrix_format == "dense" else getattr(sparse, f"{matrix_format}_matrix")(values)
    with dask.config.set({"array.chunk-size": "64B"}):
        if create:
            expected = {
                "dense": ((3, 3, 3, 3), (5,)),
                "csr": ((1,) * 12, (5,)),
                "csc": ((12,), (1,) * 5),
            }[matrix_format]
        else:
            expected = read_table_components(path, table_name="counts", components=[component])[component].chunks
        chunks = _prepared_chunks(
            monkeypatch, path, component=component, payload=payload, identity=identity, create=create
        )
    assert chunks == expected


@pytest.mark.parametrize("mode", ["memory", "backed"])
def test_new_csr_entries_are_sized_from_the_input_rows_and_limited_by_the_table_rows(tmp_path, mode):
    """A new CSR entry's values sit in the selected rows only, so its bytes per row come from them.

    The 4 × 5 float32 input of ones has 5 values per row of 8 bytes (float32
    data and int32 index) plus 4 bytes of indptr: 44 bytes per row. Averaged
    over a 12-row table it would be 20 / 12 × 8 + 4, about 17 bytes per row.
    The blocks split all 12 table rows, so they may be longer than the input.
    """
    values = _input_matrix(tmp_path, np.ones((4, 5), dtype=np.float32), "csr", mode)
    options = {"blocked_axis_length": 12, "sparse_chunks": "auto", "dense_chunks": "auto"}
    with dask.config.set({"array.chunk-size": "88B"}):
        # 88 // 44 = 2 rows, not the 5 rows the table average would give.
        assert regional_writer._block_length(values, "csr", **options) == 2
    with dask.config.set({"array.chunk-size": "1MiB"}):
        # Limited by the 12 table rows, not by the 4 input rows.
        assert regional_writer._block_length(values, "csr", **options) == 12


@pytest.mark.parametrize("create", [False, True])
def test_storage_setting_keeps_stored_chunks_of_existing_dense_matrices_only(regional_store, monkeypatch, create):
    """``"storage"`` keeps the stored (3, 2) chunks of an existing dense matrix.

    A new entry has no stored chunks to keep, so it is sized as with
    ``"auto"``: 3 rows of 20 bytes for an ``array.chunk-size`` of 64 bytes.
    """
    path, _, identity, _ = regional_store()
    component = ("obsm", "new" if create else "features")
    with dask.config.set({"array.chunk-size": "64B"}):
        chunks = _prepared_chunks(
            monkeypatch,
            path,
            component=component,
            payload=np.ones((4, 5), dtype=np.float32),
            identity=identity,
            create=create,
            dense_chunks="storage",
        )
    assert chunks == (((3, 3, 3, 3), (5,)) if create else ((3, 3, 3, 3), (2, 2, 1)))


@pytest.mark.parametrize("matrix_format, expected", [("csr", ((2,) * 6, (5,))), ("csc", ((12,), (1,) * 5))])
def test_new_entries_from_lazy_sparse_inputs_fall_back_to_dask_auto(
    regional_store, monkeypatch, matrix_format, expected
):
    """A lazy sparse input does not know its number of non-zero values without computing it.

    Its new entry is then sized by Dask's own ``"auto"``, as if it were dense,
    along the rows (CSR) or columns (CSC) of the full 12 × 5 entry. With an
    ``array.chunk-size`` of 48 bytes and 4 bytes per float32 entry:

    - CSR: 5 × 4 = 20 bytes per row, so 2 rows;
    - CSC: 12 × 4 = 48 bytes per column, so 1 column.

    The input holds one value per row, so counting its non-zero values would
    give 4 rows (12 bytes per row) or 4 columns (about 10 bytes per column):
    the result shows which sizing was used.
    """
    path, _, identity, _ = regional_store(matrix_format)
    values = getattr(sparse, f"{matrix_format}_matrix")(np.eye(4, 5, dtype=np.float32))
    payload = da.from_array(values, chunks=(2, 2), asarray=False)
    with dask.config.set({"array.chunk-size": "48B"}):
        chunks = _prepared_chunks(
            monkeypatch, path, component=("obsm", "new"), payload=payload, identity=identity, create=True
        )
    assert chunks == expected


@pytest.mark.parametrize(
    "setting, value",
    [
        ("sparse_chunks", 0),
        ("sparse_chunks", "storage"),
        ("sparse_chunks", None),
        ("dense_chunks", -1),
        ("dense_chunks", True),
        ("dense_chunks", 1.5),
        ("dense_chunks", "3"),
    ],
)
def test_invalid_chunk_settings_leave_store_unchanged(regional_store, setting, value):
    """Invalid settings are rejected with the readers' messages, before anything is written."""
    path, _, identity, _ = regional_store()
    before = _store_bytes(path)
    with pytest.raises((TypeError, ValueError), match=f"{setting} must be"):
        write_table_components_by_region(
            path,
            table_name="counts",
            components={("obsm", "features"): np.ones((4, 5), dtype=np.float32)},
            obs_identity=identity,
            overwrite=True,
            **{setting: value},
        )
    assert _store_bytes(path) == before


@pytest.mark.parametrize("matrix_format", ["dense", "csr", "csc"])
@pytest.mark.parametrize("create", [False, True])
def test_all_regions_in_table_order_need_no_fill(regional_store, matrix_format, create):
    path, table, _, old = regional_store(matrix_format)
    values = np.nan_to_num(old) * 2
    payload = values if matrix_format == "dense" else getattr(sparse, f"{matrix_format}_matrix")(values)
    name = "new" if create else "features"
    write_table_components_by_region(
        path,
        table_name="counts",
        components={("obsm", name): payload},
        obs_identity=table.obs[["region", "instance"]],
        overwrite=not create,
    )
    result = read_zarr(path / "tables/counts").obsm[name]
    np.testing.assert_array_equal(result if matrix_format == "dense" else result.toarray(), values)


@pytest.mark.parametrize(
    "stored_format, supplied_format",
    [(old, new) for old in ("dense", "csr", "csc") for new in ("dense", "csr", "csc") if old != new],
)
@pytest.mark.parametrize("all_regions", [False, True])
def test_mismatched_formats_never_publish(regional_store, stored_format, supplied_format, all_regions):
    path, table, identity, _ = regional_store(stored_format)
    if all_regions:
        identity = table.obs[["region", "instance"]]
    values = np.ones((len(identity), 5), dtype=np.float32)
    payload = values if supplied_format == "dense" else getattr(sparse, f"{supplied_format}_matrix")(values)
    before = _store_bytes(path)
    with pytest.raises(ValueError, match="matrix format"):
        write_table_components_by_region(
            path,
            table_name="counts",
            components={("obsm", "features"): payload, ("uns", "new"): 7},
            obs_identity=identity,
            overwrite=True,
        )
    assert _store_bytes(path) == before


@pytest.mark.parametrize(
    "change",
    [
        "empty",
        "partial",
        "reversed",
        "duplicate",
        "grouped",
        "unknown",
        "float_ids",
        "missing_id",
        "plain_region",
        "extra_column",
    ],
)
def test_invalid_regional_identities_leave_store_unchanged(regional_store, change):
    path, table, identity, _ = regional_store()
    if change == "empty":
        identity = identity.iloc[:0]
    elif change == "partial":
        identity = identity.iloc[:2]
    elif change == "reversed":
        identity = identity.iloc[::-1]
    elif change == "duplicate":
        identity = identity.iloc[[0, 0, 2, 3]]
    elif change == "grouped":
        identity = table.obs[["region", "instance"]].sort_values("region")
    elif change == "unknown":
        identity["region"] = pd.Categorical(["C"] * len(identity))
    elif change == "float_ids":
        identity["instance"] = identity["instance"].astype(float)
    elif change == "missing_id":
        identity.loc[identity.index[0], "instance"] = np.nan
    elif change == "plain_region":
        identity["region"] = identity["region"].astype(str)
    else:
        identity["extra"] = 0
    before = _store_bytes(path)
    with pytest.raises((ValueError, TypeError)):
        write_table_components_by_region(
            path,
            table_name="counts",
            components={("obsm", "features"): np.ones((len(identity), 5), dtype=np.float32)},
            obs_identity=identity,
            overwrite=True,
        )
    assert _store_bytes(path) == before


@pytest.mark.parametrize(
    "change", ["undeclared", "unobserved", "missing_key", "missing_column", "duplicate", "float_ids", "plain_region"]
)
@pytest.mark.parametrize("regional", [False, True])
def test_shared_annotation_checks_preserve_existing_writer_safeguards(regional_store, change, regional):
    """Full stored annotation is checked even if the regional payload only names A."""
    path, table, identity, _ = regional_store()
    group = zarr.open_group(str(path), mode="r+")["tables/counts"]
    attrs = dict(table.uns[TableModel.ATTRS_KEY])
    if change == "undeclared":
        attrs["region"] = ["A"]
    elif change == "unobserved":
        attrs["region"] = ["A", "B", "C"]
    elif change == "missing_key":
        del attrs["instance_key"]
    elif change == "missing_column":
        attrs["instance_key"] = "absent"
    else:
        obs = table.obs.copy()
        if change == "duplicate":
            obs.loc["cell-2", "instance"] = obs.loc["cell-1", "instance"]
        elif change == "float_ids":
            obs["instance"] = obs["instance"].astype(float)
        else:
            obs["region"] = obs["region"].astype(str)
        write_elem(group, "obs", obs)
    write_elem(group["uns"], TableModel.ATTRS_KEY, attrs)
    before = _store_bytes(path)
    writer = write_table_components_by_region if regional else write_table_components
    supplied_identity = identity if regional else table.obs[["region", "instance"]]
    with pytest.raises((ValueError, TypeError)):
        writer(
            path,
            table_name="counts",
            components={("obsm", "features"): np.ones((len(supplied_identity), 5), dtype=np.float32)},
            obs_identity=supplied_identity,
            overwrite=True,
        )
    assert _store_bytes(path) == before


@pytest.mark.parametrize(
    "case",
    [
        "none",
        "dataframe",
        "rows",
        "columns",
        "lossy",
        "nonnumeric",
        "missing_fill",
        "bad_fill",
        "uns_none",
        "scope",
        "no_matrix",
        "unannotated",
        "overwrite",
        "metadata_overwrite",
        "fill_key",
    ],
)
def test_invalid_payloads_reject_the_whole_batch(regional_store, case):
    path, table, identity, _ = regional_store()
    components = {("obsm", "features"): np.ones((4, 5), dtype=np.float32), ("uns", "new"): None}
    fills = None
    overwrite = True
    if case == "none":
        components[("obsm", "features")] = None
    elif case == "dataframe":
        components[("obsm", "features")] = pd.DataFrame(np.ones((4, 5)))
    elif case == "rows":
        components[("obsm", "features")] = np.ones((2, 5), dtype=np.float32)
    elif case == "columns":
        components[("obsm", "features")] = np.ones((4, 6), dtype=np.float32)
    elif case == "lossy":
        components[("obsm", "features")] = np.ones((4, 5), dtype=np.float64)
    elif case == "nonnumeric":
        components[("obsm", "features")] = np.full((4, 5), "a")
    elif case in {"missing_fill", "bad_fill"}:
        components = {("obsm", "new"): np.ones((4, 2), dtype=np.int32)}
        fills = None if case == "missing_fill" else {("obsm", "new"): np.nan}
    elif case == "uns_none":
        components = {("obsm", "features"): components[("obsm", "features")], ("uns",): None}
    elif case == "scope":
        components[("X",)] = np.ones((4, 2))
    elif case == "no_matrix":
        components = {("uns", "new"): None}
    elif case == "unannotated":
        group = zarr.open_group(str(path), mode="r+")["tables/counts/uns"]
        del group[TableModel.ATTRS_KEY]
    elif case == "overwrite":
        overwrite = False
    elif case == "metadata_overwrite":
        components = {("obsm", "new"): np.ones((4, 2)), ("uns", "analysis"): {"method": "new"}}
        fills = {("obsm", "new"): 0}
        overwrite = False
    else:
        fills = {("obsm", "typo"): 0}
    before = _store_bytes(path)
    with pytest.raises((ValueError, TypeError, FileExistsError)):
        write_table_components_by_region(
            path,
            table_name="counts",
            components=components,
            obs_identity=identity,
            fill_values=fills,
            overwrite=overwrite,
        )
    assert _store_bytes(path) == before


@pytest.mark.parametrize("matrix_format", ["csr", "csc"])
@pytest.mark.parametrize("fill", [np.nan, 1])
def test_sparse_creation_rejects_nonzero_fills(regional_store, matrix_format, fill):
    path, _, identity, _ = regional_store(matrix_format)
    before = _store_bytes(path)
    with pytest.raises(ValueError, match="zero fill"):
        write_table_components_by_region(
            path,
            table_name="counts",
            components={("obsm", "new"): getattr(sparse, f"{matrix_format}_matrix")((4, 2), dtype=float)},
            obs_identity=identity,
            fill_values={("obsm", "new"): fill},
        )
    assert _store_bytes(path) == before


def test_dense_creation_nan_fill_safe_casts_and_nested_null_metadata(regional_store):
    path, _, identity, old = regional_store()
    write_table_components_by_region(
        path,
        table_name="counts",
        obs_identity=identity,
        overwrite=True,
        components={
            ("obsm", "features"): np.zeros((4, 5), dtype=np.int16),
            ("obsm", "new"): np.ones((4, 2), dtype=np.float32),
            ("uns", "analysis", "threshold"): None,
        },
        fill_values={("obsm", "features"): np.nan, ("obsm", "new"): np.nan},
    )
    result = read_zarr(path / "tables/counts")
    expected = old.copy()
    expected[[1, 2, 3, 6]] = 0
    np.testing.assert_array_equal(result.obsm["features"], expected)
    assert np.isnan(result.obsm["new"][[0, 4, 5, 7, 8, 9, 10, 11]]).all()
    np.testing.assert_array_equal(result.obsm["new"][[1, 2, 3, 6]], np.ones((4, 2)))
    assert "threshold" in result.uns["analysis"] and result.uns["analysis"]["threshold"] is None


@pytest.mark.parametrize("matrix_format", ["dense", "csr", "csc"])
def test_lazy_regional_self_overwrite(regional_store, matrix_format):
    path, _, identity, old = regional_store(matrix_format)
    source = read_table_components(path, table_name="counts", components=[("obsm", "features")], sparse_chunks=2)[
        ("obsm", "features")
    ]
    payload = source[[1, 2, 3, 6], :] * 2
    expected = old.copy()
    expected[[1, 2, 3, 6]] *= 2
    write_table_components_by_region(
        path,
        table_name="counts",
        components={("obsm", "features"): payload},
        obs_identity=identity,
        sparse_chunks=3,
        dense_chunks=3,
        overwrite=True,
    )
    result = read_zarr(path / "tables/counts").obsm["features"]
    np.testing.assert_array_equal(result if matrix_format == "dense" else result.toarray(), expected)


@pytest.mark.parametrize("matrix_format", ["dense", "csr", "csc"])
@pytest.mark.parametrize("create", [False, True])
def test_matrix_can_have_an_empty_feature_axis(regional_store, matrix_format, create):
    path, _, identity, _ = regional_store(matrix_format)
    values = np.empty((4, 0), dtype=np.float32)
    payload = values if matrix_format == "dense" else getattr(sparse, f"{matrix_format}_matrix")(values)
    if not create:
        existing = np.empty((12, 0), dtype=np.float32)
        if matrix_format != "dense":
            existing = getattr(sparse, f"{matrix_format}_matrix")(existing)
        write_elem(zarr.open_group(str(path), mode="r+")["tables/counts/obsm"], "empty", existing)
    write_table_components_by_region(
        path,
        table_name="counts",
        components={("obsm", "empty"): payload},
        obs_identity=identity,
        fill_values={("obsm", "empty"): 0},
        sparse_chunks=3,
        dense_chunks=3,
        overwrite=not create,
    )
    result = read_zarr(path / "tables/counts").obsm["empty"]
    assert result.shape == (12, 0)
    assert result.dtype == np.float32
    assert ("dense" if isinstance(result, np.ndarray) else result.format) == matrix_format


@pytest.mark.parametrize("matrix_format", ["dense", "csr", "csc"])
@pytest.mark.parametrize("create", [False, True])
def test_regional_graph_construction_does_not_read_or_compute(
    regional_store, tmp_path, monkeypatch, matrix_format, create
):
    """Preparing shared merge blocks builds a graph without loading their values.

    Single-row dense/CSR bands exercise leading and interior bands without
    updates. Existing dense layouts and CSC's full-row blocks stay unchanged.
    """
    path, _, _, old = regional_store(matrix_format)
    existing = None
    if not create:
        # The stored dense layout splits columns too, which the merge must keep.
        existing = read_table_components(
            path, table_name="counts", components=[("obsm", "features")], sparse_chunks=1, dense_chunks="storage"
        )[("obsm", "features")]
    values = np.arange(20, dtype=np.float32).reshape(4, 5)
    payload = _input_matrix(tmp_path, values, matrix_format, "lazy")

    async def unexpected_read(*args, **kwargs):
        pytest.fail("Graph construction read from Zarr.")

    def unexpected_compute(*args, **kwargs):
        pytest.fail("Graph construction started Dask computation.")

    with monkeypatch.context() as patch:
        # Guard storage access directly, including reads outside the Dask scheduler.
        patch.setattr(LocalStore, "get", unexpected_read)
        patch.setattr(LocalStore, "get_partial_values", unexpected_read)
        # Separately reject computation, even if all its inputs are already in memory.
        with Callback(start=unexpected_compute):
            result = regional_writer._regional_matrix(
                payload,
                existing=existing,
                table_row_positions=np.array([1, 2, 3, 6]),
                n_obs=len(old),
                matrix_format=matrix_format,
                fill=0 if create else None,
                new_entry_block_length=1,
            )
    computed = result.compute()
    expected = np.zeros_like(old) if create else old.copy()
    expected[[1, 2, 3, 6]] = values
    np.testing.assert_array_equal(computed if matrix_format == "dense" else computed.toarray(), expected)


@pytest.mark.parametrize("matrix_format", ["dense", "csr", "csc"])
def test_existing_block_alignment_preserves_values_and_inputs(matrix_format):
    """Reuse aligned blocks without changing measurements or input buffers.

    Uneven terminal chunks must retain the right measurements, including rows
    without updates. Sparse layouts keep the uncompressed axis whole, matching
    the lazy reader's contract.
    """
    old = np.arange(77, dtype=np.float32).reshape(11, 7)
    old[4, 1] = np.nan
    values = np.arange(28, dtype=np.float32).reshape(4, 7) * 10
    table_row_positions = np.array([1, 3, 9, 10])
    original = old.copy() if matrix_format == "dense" else getattr(sparse, f"{matrix_format}_matrix")(old)
    updates = values.copy() if matrix_format == "dense" else getattr(sparse, f"{matrix_format}_matrix")(values)
    chunks = ((4, 4, 3), (3, 3, 1))
    if matrix_format == "csr":
        expected_chunks = (chunks[0], (7,))
    elif matrix_format == "csc":
        expected_chunks = ((11,), chunks[1])
    else:
        expected_chunks = chunks
    existing = da.from_array(original, chunks=expected_chunks, asarray=False)
    regional_values = da.from_array(updates, chunks=(2, 3), asarray=False)

    result = regional_writer._regional_matrix(
        regional_values,
        existing=existing,
        table_row_positions=table_row_positions,
        n_obs=len(old),
        matrix_format=matrix_format,
        fill=None,
        new_entry_block_length=None,
    )
    assert result.chunks == expected_chunks
    computed = result.compute()
    assert ("dense" if isinstance(computed, np.ndarray) else computed.format) == matrix_format
    expected = old.copy()
    expected[table_row_positions] = values
    np.testing.assert_array_equal(computed if matrix_format == "dense" else computed.toarray(), expected)
    np.testing.assert_array_equal(original if matrix_format == "dense" else original.toarray(), old)
    np.testing.assert_array_equal(updates if matrix_format == "dense" else updates.toarray(), values)


@pytest.mark.parametrize("matrix_format", ["csr", "csc"])
def test_misaligned_existing_blocks_rejected_before_computation(matrix_format):
    """A split uncompressed axis violates the internal reader/merge contract.

    The public writer's lazy reader never supplies this layout. Construct it
    directly to check that a future reader change fails explicitly rather
    than silently pairing existing blocks with different output boundaries.
    """
    matrix_type = getattr(sparse, f"{matrix_format}_matrix")
    existing = da.from_array(matrix_type(np.ones((11, 7))), chunks=(4, 3), asarray=False)
    regional_values = da.from_array(matrix_type(np.ones((4, 7))), chunks=(2, 3), asarray=False)

    def unexpected_compute(*args, **kwargs):
        pytest.fail("Chunk-layout validation started Dask computation.")

    with Callback(start=unexpected_compute):
        with pytest.raises(RuntimeError, match="Existing matrix chunks do not match the finalized output layout"):
            regional_writer._regional_matrix(
                regional_values,
                existing=existing,
                table_row_positions=np.array([1, 3, 9, 10]),
                n_obs=11,
                matrix_format=matrix_format,
                fill=None,
                new_entry_block_length=None,
            )


def test_aligned_existing_blocks_share_source_reads(regional_store, monkeypatch):
    """A dense write shares existing source blocks across multiple column blocks.

    Each delayed loader returns one caller-owned three-row block. Splitting
    its columns gives the existing matrix the stored (3, 2) chunk layout.
    All output blocks align, but independently delaying their array slices
    can reload the shared source block. Count those loads during a full write
    and check that neither updated nor untouched source rows were modified.
    """
    path, _, identity, old = regional_store()
    blocks = [block.copy() for block in np.split(old, [3, 6, 9])]
    requested = []

    def load_block(index):
        requested.append(index)
        return blocks[index]

    existing = da.concatenate(
        [
            da.from_delayed(delayed(load_block)(i), shape=block.shape, dtype=block.dtype, meta=block)
            for i, block in enumerate(blocks)
        ],
        axis=0,
    ).rechunk((3, 2), method="tasks")
    original_read = regional_writer._read_anndata_element

    # Substitute counted source blocks only for the regional writer's opening of
    # the stored matrix; as a Dask array, it keeps its blocks in _lazy_matrix().
    # Validation and serialization still operate on the real table.
    def read_existing(group, component_path, **kwargs):
        if component_path == ("obsm", "features") and kwargs.get("mode") == "backed":
            return existing
        return original_read(group, component_path, **kwargs)

    monkeypatch.setattr(regional_writer, "_read_anndata_element", read_existing)
    values = np.arange(20, dtype=np.float32).reshape(4, 5) * 10
    write_table_components_by_region(
        path,
        table_name="counts",
        components={("obsm", "features"): values},
        obs_identity=identity,
        dense_chunks=3,
        overwrite=True,
    )
    assert sorted(requested) == list(range(len(blocks)))
    np.testing.assert_array_equal(np.concatenate(blocks), old)
    expected = old.copy()
    expected[[1, 2, 3, 6]] = values
    np.testing.assert_array_equal(read_zarr(path / "tables/counts").obsm["features"], expected)


@pytest.mark.parametrize("matrix_format", ["dense", "csr", "csc"])
@pytest.mark.parametrize("create", [False, True])
def test_merge_reads_only_affected_matrices_and_keeps_shared_blocks_unchanged(
    regional_store, monkeypatch, matrix_format, create
):
    """Guard Zarr reads independently of Dask, and inspect computed task results.

    No task may collect the full old/merged matrix. Selected input blocks are
    returned by reference, so sparse index-array mutations cannot hide behind
    a test-side copy. Dense writes must share those reads even when an input
    block supplies multiple destination blocks. Sparse writes may compute the
    same input again across AnnData's separate output-chunk computations.
    """
    path, _, identity, old = regional_store(matrix_format)
    values = np.arange(20, dtype=np.float32).reshape(4, 5)
    axis = 1 if matrix_format == "csc" else 0
    arrays = np.split(values, [2], axis=axis)
    blocks = (
        arrays if matrix_format == "dense" else [getattr(sparse, f"{matrix_format}_matrix")(part) for part in arrays]
    )
    originals = [block.copy() for block in blocks]
    requested = []

    def load_block(index):
        requested.append(index)
        return blocks[index]

    payload = da.concatenate(
        [
            da.from_delayed(delayed(load_block)(i), shape=block.shape, dtype=block.dtype, meta=block)
            for i, block in enumerate(blocks)
        ],
        axis=axis,
    )
    original_get, original_partial = LocalStore.get, LocalStore.get_partial_values

    def check_key(key):
        if key.startswith(("tables/counts/X/", "tables/counts/obsm/unrelated/")) or (
            create and key.startswith("tables/counts/obsm/features/")
        ):
            assert key.rsplit("/", 1)[-1] in {".zarray", ".zattrs", ".zgroup", "zarr.json"}, key

    async def guarded_get(self, key, *args, **kwargs):
        check_key(key)
        return await original_get(self, key, *args, **kwargs)

    async def guarded_partial(self, prototype, key_ranges):
        key_ranges = list(key_ranges)
        for key, _ in key_ranges:
            check_key(key)
        return await original_partial(self, prototype, key_ranges)

    def check_result(value):
        if isinstance(value, (tuple, list, dict)):
            for item in value.values() if isinstance(value, dict) else value:
                check_result(item)
        elif isinstance(value, np.ndarray) or sparse.issparse(value):
            assert value.shape != old.shape, "A task collected the complete matrix."

    def posttask(key, result, dsk, state, worker_id):
        check_result(result)

    component = ("obsm", "new" if create else "features")
    with monkeypatch.context() as patch:
        # Guard direct matrix-payload reads, including reads not scheduled by Dask.
        patch.setattr(LocalStore, "get", guarded_get)
        patch.setattr(LocalStore, "get_partial_values", guarded_partial)
        # Separately detect full-matrix intermediates in executed Dask tasks.
        with Callback(posttask=posttask):
            write_table_components_by_region(
                path,
                table_name="counts",
                components={component: payload},
                obs_identity=identity,
                fill_values={component: 0} if create else None,
                sparse_chunks=3,
                dense_chunks=3,
                overwrite=True,
            )
    if matrix_format == "dense":
        assert sorted(requested) == [0, 1]
    else:
        assert set(requested) == {0, 1}
    for block, original in zip(blocks, originals, strict=True):
        np.testing.assert_array_equal(
            block if matrix_format == "dense" else block.toarray(),
            original if matrix_format == "dense" else original.toarray(),
        )
        if matrix_format != "dense":
            assert block.indices.dtype == original.indices.dtype
            assert block.indptr.dtype == original.indptr.dtype
    expected = np.zeros_like(old) if create else old.copy()
    expected[[1, 2, 3, 6]] = values
    result = read_zarr(path / "tables/counts").obsm[component[1]]
    np.testing.assert_array_equal(result if matrix_format == "dense" else result.toarray(), expected)


def test_stored_dataframe_is_rejected_before_reading_its_values(regional_store, monkeypatch):
    path, table, identity, _ = regional_store()
    group = zarr.open_group(str(path), mode="r+")["tables/counts/obsm"]
    write_elem(group, "features", pd.DataFrame({"value": np.arange(12)}, index=table.obs.index))
    before = _store_bytes(path)
    original_get = LocalStore.get

    async def guarded_get(self, key, *args, **kwargs):
        if key.startswith("tables/counts/obsm/features/"):
            assert key.rsplit("/", 1)[-1] in {".zarray", ".zattrs", ".zgroup", "zarr.json"}, key
        return await original_get(self, key, *args, **kwargs)

    monkeypatch.setattr(LocalStore, "get", guarded_get)
    with pytest.raises(TypeError, match="DataFrame-valued"):
        write_table_components_by_region(
            path,
            table_name="counts",
            components={("obsm", "features"): np.ones((4, 1))},
            obs_identity=identity,
            overwrite=True,
        )
    assert _store_bytes(path) == before


@pytest.mark.parametrize("dtype", ["int16", "uint32", "string", "object", "category"])
@pytest.mark.parametrize("regional", [False, True])
def test_shared_identity_validation_accepts_supported_identifier_types(regional_store, dtype, regional):
    path, table, _, _ = regional_store()
    obs = table.obs.copy()
    obs["instance"] = (
        obs["instance"].astype(str).astype(dtype)
        if dtype in {"string", "object", "category"}
        else obs["instance"].astype(dtype)
    )
    with settings.override(allow_write_nullable_strings=True):
        write_elem(zarr.open_group(str(path), mode="r+")["tables/counts"], "obs", obs)
    identity = obs.loc[obs["region"] == "A" if regional else np.ones(len(obs), dtype=bool), ["region", "instance"]]
    writer = write_table_components_by_region if regional else write_table_components
    writer(
        path,
        table_name="counts",
        components={("obsm", "features"): np.zeros((len(identity), 5), dtype=np.float32)},
        obs_identity=identity,
        overwrite=True,
    )
    result = read_zarr(path / "tables/counts").obsm["features"]
    np.testing.assert_array_equal(result[[1, 2, 3, 6]], np.zeros((4, 5)))


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("failure", ["staging", "publication", "finalization", "installation"])
def test_regional_failure_restores_matrices_metadata_and_root(regional_store, monkeypatch, zarr_format, failure):
    path, _, identity, _ = regional_store(zarr_format=zarr_format)
    zarr.consolidate_metadata(str(path))
    before = _store_bytes(path)
    original_write = table_writer._write_anndata_element
    original_rename = Path.rename
    original_consolidate = zarr.consolidate_metadata

    def failed_write(*args, **kwargs):
        original_write(*args, **kwargs)
        raise RuntimeError("staging failure")

    def failed_rename(self, destination):
        if "staging-" in str(self) and "component-1" in str(self):
            raise RuntimeError("publication failure")
        return original_rename(self, destination)

    def failed_consolidate(*args, **kwargs):
        original_consolidate(*args, **kwargs)
        raise RuntimeError("finalization failure")

    if failure == "staging":
        monkeypatch.setattr(table_writer, "_write_anndata_element", failed_write)
    elif failure == "publication":
        monkeypatch.setattr(Path, "rename", failed_rename)
    elif failure == "finalization":
        monkeypatch.setattr(zarr, "consolidate_metadata", failed_consolidate)
    with pytest.raises(RuntimeError, match=failure):
        with regional_writer._write_table_components_by_region_operation(
            path,
            table_name="counts",
            components={
                ("obsm", "features"): np.zeros((4, 5), dtype=np.float32),
                ("uns", "new_parent", "record"): {"method": "new"},
            },
            obs_identity=identity,
            overwrite=True,
        ):
            if failure == "installation":
                raise RuntimeError("installation failure")
    assert _store_bytes(path) == before
    assert not list(path.parent.glob(f".{path.name}.harpy-*"))
