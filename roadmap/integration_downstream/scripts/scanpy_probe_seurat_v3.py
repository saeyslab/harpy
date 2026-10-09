"""Compare ``highly_variable_genes(flavor="seurat_v3")`` on dense Dask and in-memory data.

Supports "Upstream limitations to document" in ``../scanpy_rapids_singlecell.md``.
Outliers in the first cell make the clipping step matter. The loess fit is a
stand-in (see ``_common.install_fake_skmisc``), identical for both inputs.
"""

import anndata as ad
import dask.array as da
import numpy as np
import scanpy as sc
from _common import install_fake_skmisc, make_counts, quiet

quiet()
install_fake_skmisc()
X = make_counts().toarray()
X[0, :50] = 5000.0
in_memory = ad.AnnData(X.copy())
lazy = ad.AnnData(da.from_array(X, chunks=(500, 500)))
sc.pp.highly_variable_genes(in_memory, flavor="seurat_v3", n_top_genes=100)
sc.pp.highly_variable_genes(lazy, flavor="seurat_v3", n_top_genes=100)
difference = np.abs(in_memory.var["variances_norm"].to_numpy() - lazy.var["variances_norm"].to_numpy())
agreement = (in_memory.var["highly_variable"].to_numpy() == lazy.var["highly_variable"].to_numpy()).mean()
print(f"max |variances_norm difference| = {difference.max():.4g}; highly_variable agreement = {agreement:.2%}")
