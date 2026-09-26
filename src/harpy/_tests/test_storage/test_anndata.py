from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import zarr
from anndata import AnnData
from anndata.abc import CSRDataset
from scipy import sparse
from spatialdata import SpatialData
from spatialdata.models import Labels2DModel, TableModel
from spatialdata.transformations import Identity

import harpy._storage._anndata as anndata_storage
from harpy._storage._anndata import (
    _read_anndata_element,
    _read_backed_element,
    _read_backed_table,
    _write_anndata_element,
    _write_spatialdata_table_attrs,
)
from harpy.utils._keys import _INSTANCE_KEY, _REGION_KEY, _SPATIAL


def test_read_backed_table_satisfies_anndata_and_spatialdata_contract(tmp_path):
    table_group = _write_test_regions_table(tmp_path / "table")

    table = _read_backed_table(table_group)

    TableModel.validate(table)
    labels = Labels2DModel.parse(
        np.array([[1, 2]], dtype=np.uint32),
        dims=("y", "x"),
        transformations={"global": Identity()},
    )
    container = SpatialData(labels={"labels": labels}, tables={"table": table})

    assert container.tables["table"] is table
    assert isinstance(table.X, CSRDataset)
    assert isinstance(table.obsm[_SPATIAL], zarr.Array)
    assert isinstance(table.obsm["auxiliary_feature_counts"], CSRDataset)
    assert isinstance(table.obs, pd.DataFrame)
    assert isinstance(table.var, pd.DataFrame)
    assert isinstance(table.uns, dict)
    assert table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY] == ["labels"]


def test_write_spatialdata_table_attrs_writes_regions_table_contract(tmp_path):
    group = zarr.open_group(store=str(tmp_path / "table"), mode="w")

    _write_spatialdata_table_attrs(
        group,
        regions=["labels_a", "labels_b"],
        region_key="region",
        instance_key="instance",
    )

    assert dict(group.attrs) == {
        "instance_key": "instance",
        "region": ["labels_a", "labels_b"],
        "region_key": "region",
        "spatialdata-encoding-type": "ngff:regions_table",
        "version": "0.2",
    }


def test_read_backed_element_uses_the_stored_encoding(tmp_path):
    group = zarr.open_group(store=str(tmp_path / "elements.zarr"), mode="w")
    _write_anndata_element(group, ("dense",), np.array([[1.0, 2.0]]), create_parents=False)
    _write_anndata_element(group, ("sparse",), sparse.csr_matrix([[0, 3]], dtype=np.uint32), create_parents=False)
    _write_anndata_element(group, ("frame",), pd.DataFrame({"value": [4]}), create_parents=False)
    _write_anndata_element(
        group,
        ("uns", "registry", "record"),
        {"value": 5},
        create_parents=True,
    )

    dense = _read_backed_element(group["dense"])
    sparse_matrix = _read_backed_element(group["sparse"])

    assert isinstance(dense, zarr.Array)
    assert isinstance(sparse_matrix, CSRDataset)
    assert np.array_equal(dense[:], [[1.0, 2.0]])
    assert np.array_equal(sparse_matrix.to_memory().toarray(), [[0, 3]])
    assert _read_anndata_element(group, ("frame",)).equals(pd.DataFrame({"value": [4]}))
    assert _read_anndata_element(group, ("uns", "registry", "record")) == {"value": 5}


@pytest.mark.parametrize(
    "mode, dense_type, sparse_type",
    [("lazy", da.Array, da.Array), ("backed", zarr.Array, CSRDataset), ("eager", np.ndarray, sparse.csr_matrix)],
)
def test_mapping_decoder_preserves_matrix_mode(tmp_path, mode, dense_type, sparse_type):
    """Dictionary nesting must preserve the requested matrix mode and sparse chunks."""
    group = zarr.open_group(str(tmp_path / "elements.zarr"), mode="w")
    _write_anndata_element(
        group,
        ("mapping",),
        {
            "dense": np.ones((2, 2)),
            "strings": np.array([["a", "b"], ["c", "d"]]),
            "nested": {"sparse": sparse.csr_matrix([[0, 3], [4, 0]])},
        },
        create_parents=False,
    )

    result = anndata_storage._decode_anndata_element(group["mapping"], mode=mode, sparse_chunk_size=1)

    assert isinstance(result["dense"], dense_type)
    assert isinstance(result["strings"], dense_type)
    assert isinstance(result["nested"]["sparse"], sparse_type)
    if mode == "lazy":
        assert result["nested"]["sparse"].chunks == ((1, 1), (2,))


@pytest.mark.parametrize("mode", ["lazy", "backed"])
def test_lazy_and_backed_reads_reject_eager_only_encoding(tmp_path, monkeypatch, mode):
    """Harpy rejects structured arrays before dispatch, even if AnnData can decode them."""
    group = zarr.open_group(str(tmp_path / "elements.zarr"), mode="w", zarr_format=2)
    values = np.array([(1.0, 2.0)], dtype=[("a", "f8"), ("b", "f8")])
    component_path = ("obsm", "structured")
    _write_anndata_element(group, component_path, values, create_parents=True)
    np.testing.assert_array_equal(_read_anndata_element(group, component_path, mode="eager"), values)

    def unexpected_decode(*args, **kwargs):
        pytest.fail("An unsupported matrix encoding must not reach an AnnData decoder.")

    for name in ("read_elem", "read_elem_lazy", "sparse_dataset"):
        monkeypatch.setattr(anndata_storage, name, unexpected_decode)
    with pytest.raises(ValueError, match=f"rec-array.*{mode}"):
        _read_anndata_element(group, component_path, mode=mode)


@pytest.mark.parametrize("mode", ["lazy", "backed"])
@pytest.mark.parametrize("version", ["0.3.0", None])
def test_dense_encoding_version_is_checked_before_decoding(tmp_path, monkeypatch, mode, version):
    """Harpy rejects unknown or missing dense versions before asking AnnData to read."""
    group = zarr.open_group(str(tmp_path / "elements.zarr"), mode="w")
    _write_anndata_element(group, ("X",), np.ones((2, 2)), create_parents=False)
    if version is None:
        del group["X"].attrs["encoding-version"]
    else:
        group["X"].attrs["encoding-version"] = version

    def unexpected_decode(*args, **kwargs):
        pytest.fail("An unsupported dense encoding must not reach an AnnData decoder.")

    for name in ("read_elem", "read_elem_lazy", "sparse_dataset"):
        monkeypatch.setattr(anndata_storage, name, unexpected_decode)
    with pytest.raises(ValueError, match=f"array.*{version}.*{mode}"):
        _read_anndata_element(group, ("X",), mode=mode)


@pytest.mark.parametrize("mode", ["lazy", "eager", "backed"])
@pytest.mark.parametrize("matrix_kind", ["csr", "csc"])
@pytest.mark.parametrize("version", ["0.2.0", None])
def test_sparse_encoding_version_is_checked_before_decoding(tmp_path, monkeypatch, mode, matrix_kind, version):
    """Unknown or missing sparse versions fail in Harpy, regardless of AnnData support."""
    group = zarr.open_group(str(tmp_path / "table.zarr"), mode="w")
    matrix = getattr(sparse, f"{matrix_kind}_matrix")([[0, 3]], dtype=np.uint32)
    _write_anndata_element(group, ("X",), matrix, create_parents=False)
    if version is None:
        del group["X"].attrs["encoding-version"]
    else:
        group["X"].attrs["encoding-version"] = version

    def unexpected_decode(*args, **kwargs):
        pytest.fail("An unsupported sparse encoding must not reach an AnnData decoder.")

    for name in ("read_elem", "read_elem_lazy", "sparse_dataset"):
        monkeypatch.setattr(anndata_storage, name, unexpected_decode)

    with pytest.raises(ValueError, match=f"Unsupported {matrix_kind}_matrix encoding version {version!r}; expected"):
        _read_anndata_element(group, ("X",), mode=mode)


def _write_test_regions_table(path: Path) -> zarr.Group:
    obs = pd.DataFrame(
        {
            _REGION_KEY: pd.Categorical(["labels", "labels"]),
            _INSTANCE_KEY: np.array([1, 2], dtype=np.uint32),
        },
        index=pd.Index(["labels_1", "labels_2"], name="observation"),
    )
    var = pd.DataFrame(index=pd.Index(["GeneA", "GeneB"], name="gene"))
    table = TableModel.parse(
        AnnData(
            X=sparse.csr_matrix(np.array([[1, 0], [0, 2]], dtype=np.uint32)),
            obs=obs,
            var=var,
            obsm={
                _SPATIAL: np.array([[0.0, 0.0], [1.0, 0.0]]),
                "auxiliary_feature_counts": sparse.csr_matrix(np.array([[0, 3], [4, 0]], dtype=np.uint32)),
            },
        ),
        region=["labels"],
        region_key=_REGION_KEY,
        instance_key=_INSTANCE_KEY,
    )
    table.write_zarr(path)
    group = zarr.open_group(store=str(path), mode="r+", use_consolidated=False)
    _write_spatialdata_table_attrs(
        group,
        regions=["labels"],
        region_key=_REGION_KEY,
        instance_key=_INSTANCE_KEY,
    )
    return group
