from contextlib import contextmanager

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr
from anndata import AnnData
from anndata import read_zarr as read_anndata_zarr
from anndata.abc import CSRDataset
from dask.callbacks import Callback
from geopandas.testing import assert_geodataframe_equal
from scipy import sparse
from spatialdata import SpatialData
from spatialdata import read_zarr as read_spatialdata_zarr
from spatialdata.models import Labels2DModel, PointsModel
from spatialdata.transformations import get_transformation
from zarr.storage import LocalStore

from harpy.io import read_zarr
from harpy.points import add_feature_panel
from harpy.table import aggregate_points


@contextmanager
def _guard_table_reads(monkeypatch, *, selected, allow_matrix_reads=False):
    """Reject writes, unselected table reads, and premature table-matrix reads."""
    original_get = LocalStore.get
    original_partial = LocalStore.get_partial_values
    metadata_keys = {".zarray", ".zattrs", ".zgroup", ".zmetadata", "zarr.json"}

    def check_key(key):
        parts = key.split("/")
        if len(parts) < 3 or parts[0] != "tables":
            return
        assert parts[1] in selected, f"Unselected table was read: {key}"
        if not allow_matrix_reads and parts[2] in {"X", "layers", "obsm"}:
            assert parts[-1] in metadata_keys, f"Matrix values were read: {key}"

    async def guarded_get(self, key, *args, **kwargs):
        assert self.read_only
        check_key(key)
        return await original_get(self, key, *args, **kwargs)

    async def guarded_partial(self, prototype, key_ranges):
        assert self.read_only
        key_ranges = list(key_ranges)
        for key, _ in key_ranges:
            check_key(key)
        return await original_partial(self, prototype, key_ranges)

    async def unexpected_write(*args, **kwargs):
        pytest.fail("Reading must not modify the store.")

    def unexpected_compute(graph):
        pytest.fail("Reading must not compute a Dask graph.")

    # Zarr guards catch payload reads even if they do not go through Dask.
    with monkeypatch.context() as patch:
        patch.setattr(LocalStore, "get", guarded_get)
        patch.setattr(LocalStore, "get_partial_values", guarded_partial)
        patch.setattr(LocalStore, "set", unexpected_write)
        patch.setattr(LocalStore, "delete", unexpected_write)
        # Graph construction is allowed; executing it during reading is not.
        with Callback(start=unexpected_compute):
            yield


@pytest.mark.parametrize(
    "table_name, expected",
    [(None, ["csc", "csr", "dense"]), ("csr", ["csr"]), (["dense", "csc"], ["dense", "csc"]), ([], [])],
)
def test_selection_preserves_non_table_elements(spatial_store, monkeypatch, table_name, expected):
    """Only selected tables are opened; all spatial elements and root context survive."""
    original = read_spatialdata_zarr(spatial_store)
    with _guard_table_reads(monkeypatch, selected=expected):
        result = read_zarr(spatial_store, table_name=table_name)
    assert list(result.tables) == expected
    assert result.path == original.path == spatial_store
    assert result.is_backed()
    assert result.attrs == original.attrs
    assert result.coordinate_systems == original.coordinate_systems
    for kind in ("images", "labels", "points", "shapes"):
        assert set(getattr(result, kind)) == set(getattr(original, kind))
    for name in ("image", "cells", "calls", "outlines"):
        assert get_transformation(result[name], get_all=True) == get_transformation(original[name], get_all=True)
    xr.testing.assert_equal(result.images["image"], original.images["image"])
    xr.testing.assert_equal(result.labels["cells"], original.labels["cells"])
    pd.testing.assert_frame_equal(result.points["calls"].compute(), original.points["calls"].compute())
    assert_geodataframe_equal(result.shapes["outlines"], original.shapes["outlines"])
    assert all(isinstance(table.X, da.Array) for table in result.tables.values())


@pytest.mark.parametrize("mode", ["lazy", "backed", "eager"])
def test_table_modes_survive_attachment(spatial_store, monkeypatch, mode):
    """The requested mode, sparse chunk size and table annotation survive attachment."""
    with _guard_table_reads(monkeypatch, selected={"csr"}, allow_matrix_reads=mode == "eager"):
        result = read_zarr(spatial_store, table_name="csr", table_mode=mode, sparse_chunk_size=1)
    table = result.tables["csr"]
    assert table.uns["spatialdata_attrs"] == {"region": "cells", "region_key": "region", "instance_key": "instance"}
    if mode == "lazy":
        assert isinstance(table.X, da.Array)
        assert table.X.chunks == ((1, 1), (2,))
        values = table.X.compute()
    elif mode == "backed":
        assert isinstance(table.X, CSRDataset)
        values = table.X.to_memory()
    else:
        assert sparse.isspmatrix_csr(table.X)
        values = table.X
    np.testing.assert_array_equal(values.toarray(), [[1, 0], [2, 3]])

    # The wrapper must not change SpatialData's own eager table-reading policy.
    ordinary = read_spatialdata_zarr(spatial_store)
    assert isinstance(ordinary.tables["dense"].X, np.ndarray)
    assert sparse.isspmatrix_csr(ordinary.tables["csr"].X)


def test_table_only_store_skips_broken_unselected_table(tmp_path, monkeypatch):
    """A nonempty element-type selection prevents SpatialData's eager all-table fallback."""
    path = tmp_path / "tables.zarr"
    SpatialData(tables={"counts": AnnData(X=sparse.csr_matrix([[1, 2]]))}).write(path)
    root = zarr.open_group(str(path), mode="r+")
    root["tables"].create_group("broken").attrs["encoding-type"] = "unsupported"
    for selection in ([], "counts"):
        names = set() if selection == [] else {"counts"}
        with _guard_table_reads(monkeypatch, selected=names):
            result = read_zarr(path, table_name=selection)
        assert set(result.tables) == names
    with pytest.raises(ValueError, match="Unsupported AnnData"):
        read_zarr(path)
    with pytest.raises(FileNotFoundError, match="missing"):
        read_zarr(path, table_name=["counts", "missing"])


def test_store_without_tables(tmp_path):
    path = tmp_path / "empty.zarr"
    SpatialData(attrs={"note": "empty"}).write(path)
    assert not read_zarr(path).tables
    with pytest.raises(FileNotFoundError, match="missing"):
        read_zarr(path, table_name="missing")


@pytest.mark.parametrize(
    "options, error, message",
    [
        ({"table_name": ["counts", "counts"]}, ValueError, "duplicate"),
        ({"table_name": "../counts"}, ValueError, "Invalid table name"),
        ({"table_name": [1]}, TypeError, "strings"),
        ({"table_name": {"counts"}}, TypeError, "table_name"),
        ({"table_mode": "invalid", "table_name": []}, ValueError, "mode"),
        ({"sparse_chunk_size": 0, "table_name": []}, ValueError, "positive"),
    ],
)
def test_invalid_requests_fail_before_store_access(tmp_path, options, error, message):
    with pytest.raises(error, match=message):
        read_zarr(tmp_path / "not-created.zarr", **options)


@pytest.mark.parametrize("class_aware", [False, True])
def test_reopens_aggregation_tables(tmp_path, monkeypatch, class_aware):
    """Real aggregation outputs reopen lazily, including auxiliary counts and canonical centers."""
    sdata = SpatialData(
        labels={"cells": Labels2DModel.parse(np.array([[1, 1], [2, 2]], dtype=np.uint32))},
        points={
            "calls": PointsModel.parse(
                pd.DataFrame({"x": [0.0, 1.0, 0.0], "y": [0.0, 1.0, 1.0], "gene": ["A", "B", "Neg"]})
            )
        },
    )
    if class_aware:
        add_feature_panel(
            sdata,
            "calls",
            feature_key="gene",
            feature_class_key="code_class",
            features_by_class={"Endogenous": ["A", "B"], "Negative": ["Neg"]},
        )
    path = tmp_path / "aggregation.zarr"
    sdata.write(path)
    options = {"expression_class": "Endogenous"} if class_aware else {"feature_key": "gene"}
    aggregate_points(sdata, labels_name="cells", points_name="calls", output_table_name="counts", **options)
    expected = read_anndata_zarr(path / "tables" / "counts")
    with _guard_table_reads(monkeypatch, selected={"counts"}):
        result = read_zarr(path, table_name="counts")
    table = result.tables["counts"]
    assert isinstance(table.X, da.Array)
    np.testing.assert_array_equal(table.X.compute().toarray(), expected.X.toarray())
    pd.testing.assert_frame_equal(table.obs, expected.obs)
    for key, matrix in table.obsm.items():
        assert isinstance(matrix, da.Array)
        values = matrix.compute()
        reference = expected.obsm[key]
        np.testing.assert_array_equal(
            values.toarray() if sparse.issparse(values) else values,
            reference.toarray() if sparse.issparse(reference) else reference,
        )
    if class_aware:
        assert table.uns["feature_class_aggregation"]["expression_class"] == "Endogenous"
