from copy import deepcopy

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
import zarr
from anndata import AnnData
from anndata.io import write_elem
from scipy import sparse
from spatialdata import SpatialData, read_zarr
from spatialdata.models import Labels2DModel, PointsModel
from spatialdata.transformations import Identity

_PANEL = {
    "feature_key": "gene",
    "feature_class_key": "code_class",
    "classes": ["Endogenous", "Negative", "SystemControl"],
    "features_by_class": {
        "Endogenous": ["GeneA", "GeneB", "GeneZero"],
        "Negative": ["Negative01", "Negative02", "NegativeZero"],
        "SystemControl": ["SystemControl1"],
    },
}


@pytest.fixture
def class_aware_panel():
    """Return a fresh copy of the feature panel referenced by ``class_aware_sdata``."""
    return deepcopy(_PANEL)


@pytest.fixture
def class_aware_sdata():
    """Return an in-memory SpatialData with two labels/points pairs that reference one feature panel.

    ``points_a`` (``sample_a``) and ``points_b`` (``sample_b``) carry ``gene`` and
    categorical ``code_class`` columns; both reference ``feature_panel_test``, a
    copy of ``class_aware_panel``.
    """
    labels_a = Labels2DModel.parse(
        np.array(
            [
                [1, 1, 0, 2, 2],
                [1, 1, 0, 2, 2],
                [3, 3, 0, 4, 4],
            ],
            dtype=np.uint32,
        ),
        dims=("y", "x"),
        transformations={"sample_a": Identity()},
    )
    labels_b = Labels2DModel.parse(
        np.array([[1, 1], [1, 1]], dtype=np.uint32),
        dims=("y", "x"),
        transformations={"sample_b": Identity()},
    )
    categories = pd.CategoricalDtype(categories=_PANEL["classes"])
    points_a = pd.DataFrame(
        {
            "x": [0, 1, 1, 0, 3, 4, 0],
            "y": [0, 1, 0, 1, 0, 1, 2],
            "gene": [
                "GeneA",
                "GeneA",
                "Negative01",
                "SystemControl1",
                "GeneB",
                "Negative02",
                "Negative01",
            ],
            "code_class": pd.Series(
                [
                    "Endogenous",
                    "Endogenous",
                    "Negative",
                    "SystemControl",
                    "Endogenous",
                    "Negative",
                    "Negative",
                ],
                dtype=categories,
            ),
        }
    )
    points_b = pd.DataFrame(
        {
            "x": [0, 1, 1],
            "y": [0, 0, 1],
            "gene": ["GeneB", "SystemControl1", "SystemControl1"],
            "code_class": pd.Series(["Endogenous", "SystemControl", "SystemControl"], dtype=categories),
        }
    )
    panel_name = "feature_panel_test"
    return SpatialData(
        labels={"labels_a": labels_a, "labels_b": labels_b},
        points={
            "points_a": PointsModel.parse(
                dd.from_pandas(points_a, npartitions=2),
                transformations={"sample_a": Identity()},
            ),
            "points_b": PointsModel.parse(
                dd.from_pandas(points_b, npartitions=1),
                transformations={"sample_b": Identity()},
            ),
        },
        attrs={
            "harpy": {
                "metadata_version": 1,
                "points": {
                    "points_a": {"feature_panel": panel_name},
                    "points_b": {"feature_panel": panel_name},
                },
                "feature_panels": {panel_name: deepcopy(_PANEL)},
            }
        },
    )


@pytest.fixture
def backed_class_aware_sdata(class_aware_sdata, tmp_path):
    """Return ``class_aware_sdata`` written to ``tmp_path / "input.zarr"`` and reopened."""
    path = tmp_path / "input.zarr"
    class_aware_sdata.write(path)
    return read_zarr(path)


@pytest.fixture
def make_table_io_store(tmp_path):
    """Return a factory for a temporary store with a complete table and a malformed unrelated table.

    The complete table has two observations, three features and five raw features.
    The factory accepts ``matrix_kind`` (dense/CSR/CSC) and ``zarr_format``.
    """

    def make_store(*, zarr_format=3, matrix_kind="csr"):
        matrix = np.array([[0, 1, 2], [3, 0, 4]], dtype=np.float32)
        if matrix_kind != "dense":
            matrix = getattr(sparse, f"{matrix_kind}_matrix")(matrix)
        obs = pd.DataFrame({"region": pd.Categorical(["cells", "cells"]), "instance": [1, 2]}, index=["c1", "c2"])
        table = AnnData(
            X=matrix,
            obs=obs,
            var=pd.DataFrame({"selected": [True, False, True]}, index=["g1", "g2", "g3"]),
            layers={"counts": matrix.copy()},
            obsm={"embedding": np.arange(4).reshape(2, 2), "frame": pd.DataFrame({"score": [4, 5]}, index=obs.index)},
            varm={"loadings": sparse.csc_matrix(np.arange(6).reshape(3, 2))},
            obsp={"neighbors": sparse.csr_matrix(np.eye(2))},
            varp={"neighbors": np.eye(3)},
            uns={"analysis": {"values": np.arange(4), "optional": None, "method": "test"}},
        )
        table.raw = AnnData(
            X=sparse.csr_matrix(np.arange(10).reshape(2, 5)),
            obs=obs.copy(),
            var=pd.DataFrame({"selected": [True] * 5}, index=[f"raw{i}" for i in range(5)]),
            varm={"loadings": np.ones((5, 2))},
        )
        path = tmp_path / "sdata.zarr"
        root = zarr.open_group(str(path), mode="w", zarr_format=zarr_format)
        root.attrs["spatialdata_attrs"] = {"version": "0.2"}
        write_elem(root.require_group("tables"), "counts", table)
        # A broken, unrelated table must not be inspected by either public reader.
        root["tables"].create_group("unrelated").attrs["encoding-type"] = "unsupported"
        return path

    return make_store
