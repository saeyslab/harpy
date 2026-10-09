import numpy as np
import pandas as pd
import pytest
import zarr
from anndata import AnnData
from anndata.io import write_elem
from scipy import sparse
from spatialdata.models import TableModel


@pytest.fixture
def regional_store(tmp_path):
    """Make interleaved A/B tables: A occupies rows 1, 2, 3, 6 out of twelve.

    Both regions use instance ID 1. Three-row chunks straddle A, mix A/B,
    and include a final B-only chunk. C is an unused categorical category,
    not a declared region. Expression and unrelated obsm data are sentinels.
    """

    def make(matrix_format="dense", zarr_format=3):
        root_path = tmp_path / f"{matrix_format}-{zarr_format}.zarr"
        regions = np.array(["B", "A", "A", "A", "B", "B", "A", "B", "B", "B", "B", "B"])
        obs = pd.DataFrame({"region": pd.Categorical(regions, categories=["A", "B", "C"])})
        obs["instance"] = obs.groupby("region", observed=True).cumcount() + 1
        obs.index = [f"cell-{i}" for i in range(len(obs))]
        old = np.arange(60, dtype=np.float32).reshape(12, 5)
        old[4, 1] = np.nan
        matrix = old if matrix_format == "dense" else getattr(sparse, f"{matrix_format}_matrix")(old)
        table = TableModel.parse(
            AnnData(
                X=np.ones((12, 2)),
                obs=obs,
                obsm={"features": matrix, "unrelated": np.ones((12, 4))},
                uns={"analysis": {"method": "old"}},
            ),
            region=["A", "B"],
            region_key="region",
            instance_key="instance",
        )
        root = zarr.open_group(str(root_path), mode="w", zarr_format=zarr_format)
        root.attrs["spatialdata_attrs"] = {"version": "0.2"}
        write_elem(root.require_group("tables"), "counts", table)
        if matrix_format == "dense":
            write_elem(root["tables/counts/obsm"], "features", matrix, dataset_kwargs={"chunks": (3, 2)})
        identity = table.obs.loc[table.obs["region"] == "A", ["region", "instance"]].copy()
        return root_path, table, identity, old

    return make
