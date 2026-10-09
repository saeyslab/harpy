"""Dense Dask PCA with ``svd_solver="covariance_eigh"``: row-only versus column-split chunks.

Supports "Dense tables" and Gap 1 in ``../scanpy_rapids_singlecell.md``. Runs each
combination of chunk layout and Dask scheduler; none of them uses dask-ml.
"""

import anndata as ad
import dask
import dask.array as da
import numpy as np
import scanpy as sc
from _common import describe, quiet, run_step

quiet()
for scheduler in ("threads", "synchronous"):
    for column_chunk in (-1, 150):
        X = da.random.default_rng(0).random((4000, 600), chunks=(1000, column_chunk)).astype(np.float32)
        adata = ad.AnnData(X=X)
        with dask.config.set(scheduler=scheduler):
            run_step(
                f"pca covariance_eigh, scheduler={scheduler}, X {describe(X)}",
                sc.pp.pca,
                adata,
                n_comps=20,
                svd_solver="covariance_eigh",
            )
