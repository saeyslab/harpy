"""Table functions with their own logic load lazy and backed tables into memory, so that they run on a backed sdata."""

import dask.array as da
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import harpy as hp
from harpy._tests.test_table.test_io.test_write_table_updates import _store
from harpy._tests.test_table.test_niches.test_clustering import _chain_graph
from harpy.table import nhood_kmeans
from harpy.table._table import _load_into_memory
from harpy.table.io import read_table
from harpy.utils._keys import _ANNOTATION_KEY


@pytest.mark.parametrize("mode", ["lazy", "backed", "eager"])
def test_load_into_memory_loads_lazy_and_backed_tables_and_keeps_in_memory_ones(tmp_path, mode):
    path = _store(tmp_path)
    adata = read_table(path, table_name="counts", mode=mode)
    loaded = _load_into_memory(adata, "counts")
    assert isinstance(loaded.X, sparse.csr_matrix)
    assert isinstance(loaded.layers["dense"], np.ndarray)
    np.testing.assert_array_equal(loaded.X.toarray(), read_table(path, table_name="counts", mode="eager").X.toarray())
    if mode == "eager":
        assert loaded is adata


def test_a_function_runs_on_a_backed_sdata_with_lazy_tables(tmp_path, sdata_blobs):
    """nhood_kmeans runs on a table that a backed sdata attaches with lazy matrices.

    It stands in for the other table functions that load their table into memory, such as
    score_genes, flowsom and nhood_lda: each calls _load_into_memory on the table it reads,
    in the same way, so this test checks that call once rather than once per function.
    """
    adata = sdata_blobs.tables["table"]
    adata.obs[_ANNOTATION_KEY] = pd.Categorical(np.where(np.arange(adata.n_obs) % 2 == 0, "even", "odd"))
    adata.obsp["radius_test_connectivities"] = _chain_graph(adata.n_obs)
    sdata_blobs.write(tmp_path / "sdata.zarr")
    sdata = hp.io.read_zarr(tmp_path / "sdata.zarr")
    assert isinstance(sdata.tables["table"].obsp["radius_test_connectivities"], da.Array)

    sdata = nhood_kmeans(
        sdata=sdata,
        labels_name="blobs_labels",
        table_name="table",
        output_table_name="table_niches",
        cluster_key=_ANNOTATION_KEY,
        connectivity_key="radius_test",
        n_clusters=2,
        overwrite=True,
    )
    assert isinstance(sdata.tables["table"].obsp["radius_test_connectivities"], da.Array)
    assert sdata.tables["table_niches"].obs["nhood_kmeans"].cat.categories.size == 2
