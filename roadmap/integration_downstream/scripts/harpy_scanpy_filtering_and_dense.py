"""scanpy steps on lazy Harpy tables, writes after filtering, stale tables, and dense chunking.

Supports these parts of ``../scanpy_rapids_singlecell.md``:

- the step table in "Verified: scanpy on a lazily read Harpy table";
- "After filtering" and "Dense tables" in the same section;
- Gap 1 (dense chunks) and Gap 4 (stale lazy tables).
"""

import dask
import numpy as np
import scanpy as sc
from _common import annotated_table, describe, make_counts, new_store, quiet, run_step

import harpy as hp

quiet()
counts = make_counts(4000, 600, density=0.05)
path = new_store({"counts": annotated_table(counts)})
hp.tb.io.write_table(path, table_name="dense", adata=annotated_table(counts.toarray()))

# Sparse steps on a lazily read table.
adata = hp.tb.io.read_table(path, table_name="counts", mode="lazy")
print("lazy X:", describe(adata.X))
run_step("calculate_qc_metrics(percent_top=None)", sc.pp.calculate_qc_metrics, adata, percent_top=None, inplace=True)
run_step("filter_cells(min_genes=25)", sc.pp.filter_cells, adata, min_genes=25)
run_step("filter_genes(min_cells=150)", sc.pp.filter_genes, adata, min_cells=150)
print("shape after filtering:", adata.shape, "| X:", describe(adata.X))
run_step("normalize_total", sc.pp.normalize_total, adata)
run_step("log1p", sc.pp.log1p, adata)
run_step("highly_variable_genes(flavor='seurat')", sc.pp.highly_variable_genes, adata, n_top_genes=200)
run_step("scale (sparse, zero_center=True) on a copy", sc.pp.scale, adata.copy(), max_value=10)
run_step("pca (sparse Dask)", sc.pp.pca, adata, n_comps=20)

# The filtered table no longer matches the stored observations.
run_step(
    "write_table_components after filtering (expected to be refused)",
    hp.tb.io.write_table_components,
    path,
    table_name="counts",
    components={("obsm", "X_pca"): adata.obsm["X_pca"]},
    obs_identity=adata.obs[["region", "cell_id"]],
    overwrite=True,
)

# write_table replaces the whole table, while the lazy X still reads the table it replaces.
expected = adata.X.compute().toarray()
run_step(
    "write_table(overwrite=True) of the filtered lazy table",
    hp.tb.io.write_table,
    path,
    table_name="counts",
    adata=adata,
    overwrite=True,
)
back = hp.tb.io.read_table(path, table_name="counts", mode="lazy")
print("reopened shape:", back.shape, "| X:", describe(back.X), "| obsm:", list(back.obsm))
print("reopened X matches expected:", np.allclose(back.X.compute().toarray(), expected, atol=1e-5))

# The AnnData read before the overwrite still points at the replaced storage.
run_step("compute lazy X read before the overwrite (stale)", lambda: str(adata.X.compute().shape))

# Dense table written by Harpy from a NumPy X: lazy reads keep the on-disk chunks.
# The synchronous scheduler avoids a Numba crash on this machine; see numba_scale_threads.py.
dense = hp.tb.io.read_table(path, table_name="dense", mode="lazy")
print("dense lazy X:", describe(dense.X))
with dask.config.set(scheduler="synchronous"):
    run_step("dense normalize_total", sc.pp.normalize_total, dense)
    run_step("dense log1p", sc.pp.log1p, dense)
    run_step("dense highly_variable_genes(flavor='seurat')", sc.pp.highly_variable_genes, dense, n_top_genes=200)
    run_step("dense scale", sc.pp.scale, dense, max_value=10)
    run_step("dense pca, default solver", sc.pp.pca, dense, n_comps=20)
    run_step(
        "dense pca, covariance_eigh, column-split chunks", sc.pp.pca, dense, n_comps=20, svd_solver="covariance_eigh"
    )
    dense.X = dense.X.rechunk((1000, -1))
    print("after rechunk((1000, -1)):", describe(dense.X))
    run_step("dense pca, covariance_eigh, row-only chunks", sc.pp.pca, dense, n_comps=20, svd_solver="covariance_eigh")
