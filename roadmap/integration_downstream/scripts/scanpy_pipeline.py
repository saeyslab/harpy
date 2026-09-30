"""Run a standard scanpy pipeline on a Dask-backed X and report where it computes.

Supports "Upstream limitations to document" in ``../scanpy_rapids_singlecell.md``.

Usage::

    scanpy_pipeline.py KIND [ROWS,COLUMNS]

``KIND`` is ``csr_matrix``, ``csr_array`` or ``dense``; the chunk shape defaults to
``500,500`` for a 2000 × 500 matrix.
"""

import sys

import scanpy as sc
from _common import describe, make_dask_adata, quiet, run_step

quiet()
kind = sys.argv[1] if len(sys.argv) > 1 else "csr_matrix"
chunks = tuple(int(size) for size in sys.argv[2].split(",")) if len(sys.argv) > 2 else (500, 500)
print(f"===== scanpy pipeline: kind={kind} chunks={chunks} =====")

adata = make_dask_adata(kind, chunks)
print("X:", describe(adata.X))
run_step("calculate_qc_metrics(qc_vars=['mt'])", sc.pp.calculate_qc_metrics, adata, qc_vars=["mt"], inplace=True)
adata.layers["counts"] = adata.X
run_step("normalize_total", sc.pp.normalize_total, adata)
run_step("log1p", sc.pp.log1p, adata)
print("   X:", describe(adata.X))
run_step("highly_variable_genes(flavor='seurat', n_top_genes=200)", sc.pp.highly_variable_genes, adata, n_top_genes=200)
_, ok = run_step("pca, default solver", sc.pp.pca, adata, n_comps=20)
if not ok:
    run_step("pca, svd_solver='covariance_eigh'", sc.pp.pca, adata, n_comps=20, svd_solver="covariance_eigh")
if "X_pca" in adata.obsm:
    print("   obsm['X_pca']:", describe(adata.obsm["X_pca"]), "| varm['PCs']:", describe(adata.varm["PCs"]))
    run_step("neighbors", sc.pp.neighbors, adata, n_neighbors=15)
    print("   obsp['connectivities']:", describe(adata.obsp.get("connectivities")))
    run_step("umap", sc.tl.umap, adata, maxiter=50)
    run_step("leiden", sc.tl.leiden, adata, flavor="igraph", n_iterations=2, directed=False)
print("final X:", describe(adata.X))
