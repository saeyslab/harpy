"""Read a table lazily with Harpy, run scanpy on it, and write the results back.

Supports "Verified: scanpy on a lazily read Harpy table" (writing results back) in
``../scanpy_rapids_singlecell.md``. The stored counts in ``X`` are never rewritten;
every result is written as a separate component and checked after reopening.
"""

import numpy as np
import scanpy as sc
from _common import annotated_table, describe, make_counts, new_store, quiet, run_step

import harpy as hp

quiet()
counts = make_counts(4000, 600, density=0.05)
path = new_store({"counts": annotated_table(counts)})
adata = hp.tb.read_table(path, table_name="counts", mode="lazy")
print("lazy X:", describe(adata.X))

run_step("normalize_total", sc.pp.normalize_total, adata)
run_step("log1p", sc.pp.log1p, adata)
run_step("highly_variable_genes(flavor='seurat')", sc.pp.highly_variable_genes, adata, n_top_genes=100)
run_step("pca (sparse Dask)", sc.pp.pca, adata, n_comps=20)
print("X:", describe(adata.X), "| obsm['X_pca']:", describe(adata.obsm["X_pca"]))

# Expected values, computed before writing. Signs of PCA components are deterministic
# here because the same lazy graph is evaluated.
expected_log1p = adata.X.compute().toarray()
expected_pca = adata.obsm["X_pca"].compute()

run_step(
    "write_table_components: obs, var, lazy layer, lazy X_pca, PCs, uns",
    hp.tb.write_table_components,
    path,
    table_name="counts",
    components={
        ("obs",): adata.obs,
        ("var",): adata.var,
        ("layers", "log1p"): adata.X,
        ("obsm", "X_pca"): adata.obsm["X_pca"],
        ("varm", "PCs"): adata.varm["PCs"],
        ("uns", "log1p"): adata.uns["log1p"],
        ("uns", "hvg"): adata.uns["hvg"],
        ("uns", "pca"): adata.uns["pca"],
    },
    overwrite=True,
)

back = hp.tb.read_table(path, table_name="counts", mode="lazy")
print("reopened X:", describe(back.X))
print("reopened layers['log1p']:", describe(back.layers["log1p"]))
print("reopened obsm['X_pca']:", describe(back.obsm["X_pca"]))
print("reopened uns keys:", sorted(back.uns))
print("reopened var columns:", list(back.var.columns))
print("layers['log1p'] matches expected:", np.allclose(back.layers["log1p"].compute().toarray(), expected_log1p))
print("obsm['X_pca'] matches expected:", np.allclose(back.obsm["X_pca"].compute(), expected_pca, atol=1e-5))
print("X still holds the original counts:", np.array_equal(back.X.compute().toarray(), counts.toarray()))

run_step("neighbors on reopened lazy X_pca", sc.pp.neighbors, back, n_neighbors=10)
run_step("leiden", sc.tl.leiden, back, flavor="igraph", n_iterations=2)
run_step(
    "write_table_components: obs, obsp graphs, uns",
    hp.tb.write_table_components,
    path,
    table_name="counts",
    components={
        ("obs",): back.obs,
        ("obsp", "connectivities"): back.obsp["connectivities"],
        ("obsp", "distances"): back.obsp["distances"],
        ("uns", "neighbors"): back.uns["neighbors"],
        ("uns", "leiden"): back.uns["leiden"],
    },
    overwrite=True,
)

final = hp.tb.read_table(path, table_name="counts", mode="lazy")
print("final obsp keys:", sorted(final.obsp), "| obs has leiden:", "leiden" in final.obs.columns)
print(
    "connectivities match:",
    np.allclose(final.obsp["connectivities"].compute().toarray(), back.obsp["connectivities"].toarray()),
)
