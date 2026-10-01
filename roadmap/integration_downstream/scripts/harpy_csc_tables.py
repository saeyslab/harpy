"""Lazily read CSC-stored tables with Harpy and run scanpy on them, before and after conversion to CSR.

Supports Gap 1 (CSC tables) in ``../scanpy_rapids_singlecell.md``. Harpy's Visium and
Visium HD readers store ``X`` as CSC. A lazy read then chunks the gene axis, with every
cell in each block. With 600 genes the whole matrix fits in one block and hides the
problem, so both gene counts are run.
"""

import numpy as np
import scanpy as sc
from _common import annotated_table, describe, new_store, quiet, run_step
from scipy import sparse

import harpy as hp

quiet()
rng = np.random.default_rng(0)
for n_vars in (600, 2500):
    counts = sparse.random(4000, n_vars, density=0.05, format="csc", random_state=0, dtype=np.float32)
    counts.data = rng.poisson(3, counts.nnz).astype(np.float32) + 1
    path = new_store({"counts": annotated_table(counts)})

    for converted in (False, True):
        adata = hp.tb.read_table(path, table_name="counts", mode="lazy")
        label = f"{n_vars} genes, {'converted to row-chunked CSR' if converted else 'as read'}"
        if converted:
            # Every row block of the result needs data from every gene block of the input.
            meta = sparse.csr_matrix((0, 0), dtype=adata.X.dtype)
            adata.X = adata.X.rechunk((1000, -1)).map_blocks(sparse.csr_matrix, meta=meta)
        accepted = adata.X.numblocks[1] == 1 and isinstance(adata.X._meta, sparse.csr_matrix)
        print(f"===== {label}: X {describe(adata.X)}; passes rapids-singlecell's layout check: {accepted}")
        run_step(
            "calculate_qc_metrics(percent_top=None)", sc.pp.calculate_qc_metrics, adata, percent_top=None, inplace=True
        )
        run_step("normalize_total", sc.pp.normalize_total, adata)
        run_step("log1p", sc.pp.log1p, adata)
        run_step("highly_variable_genes(flavor='seurat')", sc.pp.highly_variable_genes, adata, n_top_genes=200)
        run_step("pca", sc.pp.pca, adata, n_comps=20)
