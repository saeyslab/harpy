import numpy as np
import pandas as pd
import pytest
import zarr
from anndata import AnnData
from anndata.io import write_elem
from scipy import sparse


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
