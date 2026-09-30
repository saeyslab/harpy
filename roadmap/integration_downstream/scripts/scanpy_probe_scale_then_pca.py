"""Show that sparse ``scale(zero_center=True)`` produces ``np.matrix`` blocks that break PCA.

Supports "Upstream limitations to document" in ``../scanpy_rapids_singlecell.md``.
Uses the synchronous scheduler to avoid the Numba crash described in
``numba_scale_threads.py``.
"""

import dask
import numpy as np
import scanpy as sc
from _common import describe, make_dask_adata, quiet, run_step

quiet()
dask.config.set(scheduler="synchronous")
adata = make_dask_adata("csr_matrix")
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=200)
sc.pp.scale(adata, max_value=10)
print("after scale X:", describe(adata.X), "| first block:", type(adata.X.blocks[0, 0].compute()).__name__)
run_step("pca, default solver, after sparse scale", sc.pp.pca, adata.copy(), n_comps=20)
run_step("pca, covariance_eigh, after sparse scale", sc.pp.pca, adata.copy(), n_comps=20, svd_solver="covariance_eigh")

converted = adata.copy()
converted.X = converted.X.map_blocks(np.asarray, meta=np.array([], dtype=converted.X.dtype))
print("after map_blocks(np.asarray):", describe(converted.X))
run_step(
    "pca, covariance_eigh, after converting blocks", sc.pp.pca, converted, n_comps=20, svd_solver="covariance_eigh"
)
