from pathlib import Path

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

from harpy.table._zarr import (
    _publish_staged_anndata_elements,
    _read_anndata_element,
    _read_backed_element,
    _read_backed_table,
    _StagedAnnDataElement,
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
    _write_anndata_element(group, ("dense",), np.array([[1.0, 2.0]]))
    _write_anndata_element(group, ("sparse",), sparse.csr_matrix([[0, 3]], dtype=np.uint32))
    _write_anndata_element(group, ("frame",), pd.DataFrame({"value": [4]}))
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


def test_publish_staged_anndata_elements_publishes_one_consistency_unit(tmp_path):
    root_path = tmp_path / "sdata.zarr"
    workspace = tmp_path / "workspace.zarr"
    root = zarr.open_group(store=str(root_path), mode="w")
    staging = zarr.open_group(store=str(workspace), mode="w")
    _write_anndata_element(root, ("matrix",), np.array([1], dtype=np.uint32))
    _write_anndata_element(root, ("metadata",), {"version": 0})
    _write_anndata_element(staging, ("matrix",), np.array([2], dtype=np.uint32))
    _write_anndata_element(staging, ("metadata",), {"version": 1})

    with _publish_staged_anndata_elements(
        root=root_path,
        workspace=workspace,
        elements=(
            _StagedAnnDataElement(staged=workspace / "matrix", destination=root_path / "matrix"),
            _StagedAnnDataElement(staged=workspace / "metadata", destination=root_path / "metadata"),
        ),
        operation="test",
    ) as published:
        assert np.array_equal(_read_anndata_element(published, ("matrix",))[:], [2])
        assert _read_anndata_element(published, ("metadata",)) == {"version": 1}

    assert not workspace.exists()
    assert not list(tmp_path.glob(".sdata.zarr.harpy-test-backup-*"))


def test_publish_staged_anndata_elements_restores_all_destinations_on_failure(tmp_path):
    root_path = tmp_path / "sdata.zarr"
    workspace = tmp_path / "workspace.zarr"
    root = zarr.open_group(store=str(root_path), mode="w")
    staging = zarr.open_group(store=str(workspace), mode="w")
    _write_anndata_element(root, ("matrix",), np.array([1], dtype=np.uint32))
    _write_anndata_element(root, ("metadata",), {"version": 0})
    _write_anndata_element(staging, ("matrix",), np.array([2], dtype=np.uint32))
    _write_anndata_element(staging, ("metadata",), {"version": 1})

    with pytest.raises(RuntimeError, match="injected attachment failure"):
        with _publish_staged_anndata_elements(
            root=root_path,
            workspace=workspace,
            elements=(
                _StagedAnnDataElement(staged=workspace / "matrix", destination=root_path / "matrix"),
                _StagedAnnDataElement(staged=workspace / "metadata", destination=root_path / "metadata"),
            ),
            operation="test",
        ) as published:
            assert np.array_equal(_read_anndata_element(published, ("matrix",))[:], [2])
            raise RuntimeError("injected attachment failure")

    restored = zarr.open_group(store=str(root_path), mode="r", use_consolidated=False)
    assert np.array_equal(_read_anndata_element(restored, ("matrix",))[:], [1])
    assert _read_anndata_element(restored, ("metadata",)) == {"version": 0}
    assert not workspace.exists()
    assert not list(tmp_path.glob(".sdata.zarr.harpy-test-backup-*"))


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
