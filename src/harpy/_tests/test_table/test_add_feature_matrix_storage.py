from copy import deepcopy

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import zarr
from anndata import AnnData
from anndata.io import write_elem
from scipy import sparse
from spatialdata import SpatialData
from spatialdata.models import Labels2DModel, TableModel
from zarr.storage import LocalStore

import harpy.table._add_feature_matrix as feature_writer
import harpy.table._write as table_writer
from harpy._tests.test_table.test_io import _assert_value
from harpy._tests.test_table.test_write import _store_bytes
from harpy.table import read_table
from harpy.table._add_feature_matrix import add_feature_matrix


@pytest.fixture
def feature_sdata(tmp_path):
    """Small interleaved A/B table, with identical instance IDs in different regions.

    Stored rows: B1, A2, B2, A1, B3, A3. Label areas in A are 3, 2, 1,
    so calculating A must write [2, 3, 1] into table rows [1, 3, 5].
    The malformed unrelated table detects accidental whole-store table reads.
    """

    def make(*, backed=True, zarr_format=3, existing=True, mode="lazy", metadata_key="feature_matrices"):
        labels = {
            "A": Labels2DModel.parse(np.array([[1, 1, 1], [2, 2, 3]], dtype=np.uint32)),
            "B": Labels2DModel.parse(np.array([[1, 2, 2], [2, 3, 3]], dtype=np.uint32)),
        }
        table = TableModel.parse(
            AnnData(
                X=sparse.csr_matrix(np.arange(12).reshape(6, 2)),
                obs=pd.DataFrame(
                    {"sample": pd.Categorical(["B", "A", "B", "A", "B", "A"]), "object_id": [1, 2, 2, 1, 3, 3]},
                    index=[f"cell{i}" for i in range(6)],
                ),
                layers={"counts": np.arange(12).reshape(6, 2)},
                obsm={"unrelated": np.ones((6, 2))},
                uns={"unrelated": {"value": 7}},
            ),
            region=["A", "B"],
            region_key="sample",
            instance_key="object_id",
        )
        if existing:
            table.obsm["features"] = np.arange(10, 16, dtype=np.float64).reshape(6, 1)
            table.uns[metadata_key] = {
                "features": {
                    "feature_columns": ["area"],
                    "schema_version": 1,
                    "backend": "numpy",
                    "source_kind": "harpy_add_feature_matrix",
                    "dtype": "float64",
                    "source_label": ["A", "B"],
                    "source_image": [None, None],
                    "source_channels": None,
                    "coordinate_system": ["old_A", "old_B"],
                    "features": ["area"],
                    "extra": "keep",
                }
            }
        path = tmp_path / "sdata.zarr"
        if backed:
            root = zarr.open_group(str(path), mode="w", zarr_format=zarr_format)
            root.attrs["spatialdata_attrs"] = {"version": "0.2"}
            write_elem(root.require_group("tables"), "counts", table)
            root["tables"].create_group("unrelated").attrs["encoding-type"] = "unsupported"
            zarr.consolidate_metadata(str(path))
            table = read_table(path, table_name="counts", mode=mode)
        sdata = SpatialData(labels=labels, tables={"counts": table})
        if backed:
            sdata.path = path
        return sdata

    return make


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("mode", ["lazy", "backed"])
def test_backed_update_preserves_other_regions_and_unrelated_data(
    feature_sdata, monkeypatch, zarr_format, existing, mode
):
    """Update A in stored row order without reading/writing X or other matrices.

    New matrices fill B with NaN; existing matrices keep B's stored values and
    source description. Only the two updated entries are refreshed in memory,
    preserving unrelated local changes and matrix references.
    """
    sdata = feature_sdata(zarr_format=zarr_format, existing=existing, mode=mode, metadata_key="custom_features")
    table = sdata.tables["counts"]
    previous_x, previous_other = table.X, table.obsm["unrelated"]
    table.obs["local_note"] = "unsaved"
    table.uns["unrelated"]["value"] = 99
    table.uns.setdefault("custom_features", {})["other"] = {"local_note": "unsaved"}
    previous_metadata = table.uns["custom_features"]
    previous_metadata_values = deepcopy(previous_metadata)
    original_bytes = _store_bytes(sdata.path)
    original_get, original_partial = LocalStore.get, LocalStore.get_partial_values
    original_set, original_delete = LocalStore.set, LocalStore.delete
    original_write = table_writer._write_anndata_element
    original_read = feature_writer._read_anndata_element
    staging = False

    def check_read(key):
        if key.rsplit("/", 1)[-1] in {"zarr.json", ".zarray", ".zattrs", ".zgroup"}:
            return
        assert not key.startswith(("tables/counts/X/", "tables/counts/layers/", "tables/counts/obsm/unrelated/")), key
        if key.startswith("tables/counts/obsm/features/"):
            assert staging, "The existing matrix was materialized before serialization."

    async def guarded_get(self, key, *args, **kwargs):
        check_read(key)
        return await original_get(self, key, *args, **kwargs)

    async def guarded_partial(self, prototype, key_ranges):
        key_ranges = list(key_ranges)
        for key, _ in key_ranges:
            check_read(key)
        return await original_partial(self, prototype, key_ranges)

    async def guarded_set(self, key, *args, **kwargs):
        assert not key.startswith(("tables/counts/X/", "tables/counts/layers/", "tables/counts/obsm/unrelated/")), key
        return await original_set(self, key, *args, **kwargs)

    async def guarded_delete(self, key, *args, **kwargs):
        assert not key.startswith(("tables/counts/X/", "tables/counts/layers/", "tables/counts/obsm/unrelated/")), key
        return await original_delete(self, key, *args, **kwargs)

    def write_staged(*args, **kwargs):
        nonlocal staging
        staging = True
        return original_write(*args, **kwargs)

    def read_feature_component(group, path, **kwargs):
        result = original_read(group, path, **kwargs)
        if path == ("obsm", "features") and not staging:
            # Preparation needs only existence and dtype, not a second lazy graph.
            assert isinstance(result, zarr.Array)
        return result

    with monkeypatch.context() as patch:
        # Guard Zarr access, including reads outside Dask. Existing feature
        # values may be read only once the shared writer starts serialization.
        patch.setattr(LocalStore, "get", guarded_get)
        patch.setattr(LocalStore, "get_partial_values", guarded_partial)
        patch.setattr(LocalStore, "set", guarded_set)
        patch.setattr(LocalStore, "delete", guarded_delete)
        patch.setattr(table_writer, "_write_anndata_element", write_staged)
        patch.setattr(feature_writer, "_read_anndata_element", read_feature_component)
        add_feature_matrix(
            sdata,
            "A",
            None,
            table_name="counts",
            feature_key="features",
            features=["area"],
            feature_matrices_key="custom_features",
            overwrite_feature_key=existing,
        )

    expected = np.arange(10, 16, dtype=float).reshape(6, 1) if existing else np.full((6, 1), np.nan)
    expected[[1, 3, 5], 0] = [2, 3, 1]
    assert sdata.tables["counts"] is table
    assert table.X is previous_x and table.obsm["unrelated"] is previous_other
    assert isinstance(table.obsm["features"], da.Array)
    np.testing.assert_array_equal(table.obsm["features"].compute(), expected)
    _assert_value(previous_metadata, previous_metadata_values)
    assert table.obs["local_note"].tolist() == ["unsaved"] * 6
    assert table.uns["unrelated"]["value"] == 99
    assert table.uns["custom_features"]["other"] == {"local_note": "unsaved"}
    metadata = table.uns["custom_features"]["features"]
    np.testing.assert_array_equal(metadata["source_label"], ["B", "A"] if existing else ["A"])
    np.testing.assert_array_equal(metadata["coordinate_system"], ["old_B", "global"] if existing else ["global"])
    if existing:
        assert metadata["extra"] == "keep"
    reopened = read_table(sdata.path, table_name="counts", mode="eager")
    np.testing.assert_array_equal(reopened.obsm["features"], expected)
    _assert_value(reopened.uns["custom_features"]["features"], metadata)
    assert "local_note" not in reopened.obs and "other" not in reopened.uns["custom_features"]
    assert reopened.uns["unrelated"]["value"] == 7
    after = _store_bytes(sdata.path)
    # The whole affected matrix may be rewritten; unrelated payloads and their
    # metadata must remain byte-for-byte unchanged (root consolidation may change).
    for path, contents in original_bytes.items():
        if not path.startswith(
            ("tables/counts/obsm/features/", "tables/counts/uns/custom_features/features/")
        ) and path not in {"zarr.json", ".zmetadata"}:
            assert after[path] == contents, path


def test_unbacked_regional_updates_preserve_values_and_sources_without_writing(feature_sdata, monkeypatch):
    sdata = feature_sdata(backed=False, existing=False)

    def unexpected_write(*args, **kwargs):
        pytest.fail("An unbacked update must not write to disk.")

    monkeypatch.setattr(feature_writer, "_write_table_components_by_region_operation", unexpected_write)
    for region in ["B", "A"]:
        add_feature_matrix(
            sdata,
            region,
            None,
            table_name="counts",
            feature_key="features",
            features=["area"],
            overwrite_feature_key=True,
        )
    table = sdata.tables["counts"]
    assert isinstance(table.obsm["features"], np.ndarray)
    np.testing.assert_array_equal(table.obsm["features"][:, 0], [1, 2, 3, 3, 2, 1])
    assert table.uns["feature_matrices"]["features"]["source_label"] == ["B", "A"]
    assert table.uns["feature_matrices"]["features"]["source_image"] == [None, None]


@pytest.mark.parametrize("backed", [False, True])
@pytest.mark.parametrize("reason", ["overwrite", "columns", "column_order", "missing_metadata"])
def test_rejected_feature_updates_leave_matrix_and_metadata_unchanged(feature_sdata, backed, reason):
    sdata = feature_sdata(backed=backed)
    table = sdata.tables["counts"]
    features = ["area"]
    if reason != "overwrite":
        if reason == "columns":
            table.uns["feature_matrices"]["features"]["feature_columns"] = ["different_feature"]
        elif reason == "column_order":
            table.uns["feature_matrices"]["features"]["feature_columns"] = ["perimeter", "area"]
            table.obsm["features"] = np.zeros((6, 2))
            features = ["area", "perimeter"]
        else:
            del table.uns["feature_matrices"]["features"]
        if backed:
            root = zarr.open_group(str(sdata.path), mode="r+", use_consolidated=False)
            write_elem(root["tables/counts/uns"], "feature_matrices", table.uns["feature_matrices"])
            if reason == "column_order":
                write_elem(root["tables/counts/obsm"], "features", table.obsm["features"])
    previous_matrix, previous_metadata = table.obsm["features"], table.uns["feature_matrices"]
    before = _store_bytes(sdata.path) if backed else None
    with pytest.raises(ValueError, match="already exists" if reason == "overwrite" else "compatible schema"):
        add_feature_matrix(
            sdata,
            "A",
            None,
            table_name="counts",
            feature_key="features",
            features=features,
            overwrite_feature_key=reason != "overwrite",
        )
    assert table.obsm["features"] is previous_matrix
    assert table.uns["feature_matrices"] is previous_metadata
    if backed:
        assert _store_bytes(sdata.path) == before


@pytest.mark.parametrize("location", ["disk_only", "memory_only"])
def test_feature_overwrite_requires_permission_for_attached_or_stored_entries(feature_sdata, location):
    sdata = feature_sdata(existing=location == "disk_only")
    table = sdata.tables["counts"]
    if location == "disk_only":
        del table.obsm["features"]
        del table.uns["feature_matrices"]
    else:
        table.obsm["features"] = np.ones((6, 1))
    before = _store_bytes(sdata.path)
    with pytest.raises(ValueError, match="overwrite_feature_key=True"):
        add_feature_matrix(sdata, "A", None, table_name="counts", feature_key="features", features=["area"])
    assert _store_bytes(sdata.path) == before


@pytest.mark.parametrize("backed", [False, True])
@pytest.mark.parametrize(
    ("representation", "message"),
    [
        ("dataframe", "DataFrame"),
        ("csr", "dense feature matrix"),
        ("csc", "dense feature matrix"),
        ("shape", "shape"),
        ("dtype", "safely cast"),
    ],
)
def test_incompatible_feature_matrix_is_rejected_without_payload_reads(
    feature_sdata, monkeypatch, backed, representation, message
):
    """Feature-matrix compatibility is checked in both storage modes.

    Neither path may change the existing entries on failure. Backed checks use
    only matrix metadata, including when rejecting DataFrames and sparse formats.
    """
    sdata = feature_sdata(backed=backed)
    table = sdata.tables["counts"]
    matrix = np.ones((6, 1))
    if representation == "dataframe":
        replacement = pd.DataFrame(matrix, index=table.obs_names, columns=["area"])
    elif representation == "csr":
        replacement = sparse.csr_matrix(matrix)
    elif representation == "csc":
        replacement = sparse.csc_matrix(matrix)
    elif representation == "shape":
        replacement = np.ones((6, 2))
    else:
        replacement = matrix.astype(np.float32)
    if backed:
        root = zarr.open_group(str(sdata.path), mode="r+", use_consolidated=False)
        write_elem(root["tables/counts/obsm"], "features", replacement)
    else:
        table.obsm["features"] = replacement
    previous_matrix, previous_metadata = table.obsm["features"], table.uns["feature_matrices"]
    before = _store_bytes(sdata.path) if backed else None
    original_get = LocalStore.get

    async def guarded_get(self, key, *args, **kwargs):
        if key.startswith("tables/counts/obsm/features/"):
            assert key.rsplit("/", 1)[-1] in {"zarr.json", ".zarray", ".zattrs", ".zgroup"}, key
        return await original_get(self, key, *args, **kwargs)

    monkeypatch.setattr(LocalStore, "get", guarded_get)
    with pytest.raises((TypeError, ValueError), match=message):
        add_feature_matrix(
            sdata, "A", None, table_name="counts", feature_key="features", features=["area"], overwrite_feature_key=True
        )
    assert table.obsm["features"] is previous_matrix
    assert table.uns["feature_matrices"] is previous_metadata
    if backed:
        assert _store_bytes(sdata.path) == before


@pytest.mark.parametrize("backed", [False, True])
def test_feature_metadata_uses_existing_matrix_dtype(feature_sdata, backed):
    """Metadata describes the destination dtype, not the calculated float64 values.

    For backed updates, the stored matrix is authoritative even when the local
    matrix and its metadata still describe an older dtype.
    """
    sdata = feature_sdata(backed=backed)
    table = sdata.tables["counts"]
    existing = np.arange(10, 16, dtype=np.complex128).reshape(6, 1) + 2j
    if backed:
        root = zarr.open_group(str(sdata.path), mode="r+", use_consolidated=False)
        write_elem(root["tables/counts/obsm"], "features", existing)
    else:
        table.obsm["features"] = existing

    add_feature_matrix(
        sdata, "A", None, table_name="counts", feature_key="features", features=["area"], overwrite_feature_key=True
    )

    assert table.obsm["features"].dtype == np.dtype("complex128")
    assert table.uns["feature_matrices"]["features"]["dtype"] == "complex128"
    expected = existing.copy()
    expected[[1, 3, 5], 0] = [2, 3, 1]
    result = table.obsm["features"].compute() if backed else table.obsm["features"]
    np.testing.assert_array_equal(result, expected)


def test_backed_update_rejects_reordered_unselected_observations(feature_sdata):
    """Even unchanged B rows must agree: the installed matrix covers the full table."""
    sdata = feature_sdata()
    table = sdata.tables["counts"]
    table.obs = table.obs.iloc[[2, 1, 0, 3, 4, 5]].copy()
    before = _store_bytes(sdata.path)
    with pytest.raises(ValueError, match="In-memory observation"):
        add_feature_matrix(
            sdata, "A", None, table_name="counts", feature_key="features", features=["area"], overwrite_feature_key=True
        )
    assert _store_bytes(sdata.path) == before


@pytest.mark.parametrize("backed", [False, True])
@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing_table", "does not exist in sdata.tables"),
        ("missing_annotation", "must have SpatialData annotation"),
        ("malformed_annotation", "must have SpatialData annotation"),
        ("missing_identity_key", "distinct, nonempty region_key and instance_key"),
        ("missing_identity_column", "must contain the stored region and instance columns"),
        ("noncategorical_regions", "region column must be categorical"),
        ("duplicate_selected_identity", "region/instance pairs must be non-null and unique"),
        ("duplicate_unselected_identity", "region/instance pairs must be non-null and unique"),
        ("missing_labels", "does not exist in sdata.labels"),
        ("unobserved_labels", "has no observations in table"),
    ],
)
def test_invalid_feature_table_or_selection_fails_before_calculation(feature_sdata, monkeypatch, backed, case, message):
    """Both storage modes validate the whole table and the requested label sources.

    A valid A/B table cannot accept an A/C request merely because C exists in
    sdata.labels or in unused categories. Duplicate identities also fail when
    they occur in B, outside the selected region A. No calculation or mutation
    should occur after any of these invalid requests.
    """
    sdata = feature_sdata(backed=backed)
    table = sdata.tables["counts"]
    labels = ["A"]
    if case == "missing_table":
        del sdata.tables["counts"]
    elif case == "missing_annotation":
        del table.uns[TableModel.ATTRS_KEY]
    elif case == "malformed_annotation":
        table.uns[TableModel.ATTRS_KEY] = "not a mapping"
    elif case == "missing_identity_key":
        del table.uns[TableModel.ATTRS_KEY][TableModel.INSTANCE_KEY]
    elif case == "missing_identity_column":
        table.obs.drop(columns="object_id", inplace=True)
    elif case == "noncategorical_regions":
        table.obs["sample"] = table.obs["sample"].astype(object)
    elif case == "duplicate_selected_identity":
        table.obs.loc["cell5", "object_id"] = 2
    elif case == "duplicate_unselected_identity":
        table.obs.loc["cell4", "object_id"] = 1
    elif case == "missing_labels":
        del sdata.labels["A"]
    else:
        sdata.labels["C"] = sdata.labels["A"]
        table.obs["sample"] = table.obs["sample"].cat.add_categories(["C"])
        labels = ["A", "C"]

    previous_matrix = table.obsm["features"]
    previous_x = table.X
    previous_obs, previous_uns = table.obs.copy(deep=True), deepcopy(table.uns)
    before = _store_bytes(sdata.path) if backed else None

    def unexpected_calculation(*args, **kwargs):
        pytest.fail("Invalid feature requests must fail before raster computation.")

    monkeypatch.setattr(feature_writer, "_compute_pair_feature_frame", unexpected_calculation)
    with pytest.raises(ValueError, match=message):
        add_feature_matrix(
            sdata,
            labels,
            None,
            table_name="counts",
            feature_key="features",
            features=["area"],
            overwrite_feature_key=True,
        )
    assert sdata.tables.get("counts") is (None if case == "missing_table" else table)
    assert table.obsm["features"] is previous_matrix and table.X is previous_x
    pd.testing.assert_frame_equal(table.obs, previous_obs)
    _assert_value(table.uns, previous_uns)
    if backed:
        assert _store_bytes(sdata.path) == before


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("failure", ["matrix", "metadata", "installation", "finalization"])
def test_failed_feature_update_restores_disk_and_attached_entries(
    feature_sdata, monkeypatch, zarr_format, existing, failure
):
    """Both component writes and attachment belong to one rollback window.

    Fail after a staged component write, after installing the new matrix, or
    after consolidation changes root metadata. Restore old references (or
    remove newly created entries) as well as the exact original store bytes.
    """
    sdata = feature_sdata(zarr_format=zarr_format, existing=existing)
    table = sdata.tables["counts"]
    previous_matrix = table.obsm.get("features")
    previous_metadata = table.uns.get("feature_matrices")
    before = _store_bytes(sdata.path)
    original_write = table_writer._write_anndata_element
    original_install = type(table.obsm).__setitem__
    original_consolidate = zarr.consolidate_metadata

    def failed_write(group, path, *args, **kwargs):
        original_write(group, path, *args, **kwargs)
        if path == ("component-0",) and failure == "matrix" or path == ("component-1",) and failure == "metadata":
            raise RuntimeError(f"{failure} failure")

    def failed_install(self, key, value):
        original_install(self, key, value)
        if self.parent is table and key == "features" and value is not previous_matrix:
            raise RuntimeError("installation failure")

    def failed_consolidate(*args, **kwargs):
        original_consolidate(*args, **kwargs)
        raise RuntimeError("finalization failure")

    if failure in {"matrix", "metadata"}:
        monkeypatch.setattr(table_writer, "_write_anndata_element", failed_write)
    elif failure == "installation":
        monkeypatch.setattr(type(table.obsm), "__setitem__", failed_install)
    else:
        monkeypatch.setattr(zarr, "consolidate_metadata", failed_consolidate)
    with pytest.raises(RuntimeError, match=failure):
        add_feature_matrix(
            sdata,
            "A",
            None,
            table_name="counts",
            feature_key="features",
            features=["area"],
            overwrite_feature_key=existing,
        )
    assert _store_bytes(sdata.path) == before
    assert sdata.tables["counts"] is table
    assert table.obsm.get("features") is previous_matrix
    assert table.uns.get("feature_matrices") is previous_metadata
    assert not list(sdata.path.parent.glob(f".{sdata.path.name}.harpy-*"))


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("replace", [False, True])
def test_new_feature_table_publishes_complete_result_and_attaches_lazily(feature_sdata, zarr_format, replace):
    sdata = feature_sdata(zarr_format=zarr_format)
    output = "counts" if replace else "new"
    add_feature_matrix(
        sdata,
        ["A", "B"],
        None,
        output_table_name=output,
        feature_key="features",
        features=["area"],
        overwrite_output_table=replace,
    )
    attached = sdata.tables[output]
    assert attached.X is None and isinstance(attached.obsm["features"], da.Array)
    np.testing.assert_array_equal(attached.obsm["features"].compute()[:, 0], [3, 2, 1, 1, 3, 2])
    reopened = read_table(sdata.path, table_name=output, mode="eager")
    _assert_value(reopened.uns, attached.uns)
    np.testing.assert_array_equal(reopened.obsm["features"], attached.obsm["features"].compute())
    if replace:
        assert "unrelated" not in reopened.obsm and not reopened.layers
    assert (
        zarr.open_group(str(sdata.path), mode="r", use_consolidated=True)[f"tables/{output}"].metadata.zarr_format
        == zarr_format
    )


@pytest.mark.parametrize("backed", [False, True])
@pytest.mark.parametrize("replace", [False, True])
def test_new_feature_table_is_not_published_before_calculation(feature_sdata, monkeypatch, backed, replace):
    """A failed calculation must neither install an empty table nor replace an existing one."""
    sdata = feature_sdata(backed=backed)
    previous = sdata.tables["counts"]
    before = _store_bytes(sdata.path) if backed else None

    def failed_calculation(*args, **kwargs):
        raise RuntimeError("calculation failure")

    monkeypatch.setattr(feature_writer, "_compute_pair_feature_frame", failed_calculation)
    with pytest.raises(RuntimeError, match="calculation failure"):
        add_feature_matrix(
            sdata,
            "A",
            None,
            output_table_name="counts" if replace else "new",
            feature_key="features",
            features=["area"],
            overwrite_output_table=replace,
        )
    assert sdata.tables["counts"] is previous
    assert "new" not in sdata.tables
    if backed:
        assert _store_bytes(sdata.path) == before
