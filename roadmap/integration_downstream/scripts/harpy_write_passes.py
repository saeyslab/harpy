"""Count how often the source X is read when one call writes two lazy results.

Supports Gap 2 in ``../scanpy_rapids_singlecell.md``. Relies on anndata 0.12
internals: ``make_dask_chunk`` reads one sparse block from storage per call.
"""

import anndata._io.specs.lazy_methods as lazy_methods
import scanpy as sc
from _common import ComputeCounter, annotated_table, make_counts, new_store, quiet

import harpy as hp

quiet()
path = new_store({"counts": annotated_table(make_counts(4000, 600, density=0.05))})

block_reads = 0
read_block = lazy_methods.make_dask_chunk


def counting_read_block(path_or_sparse_dataset, elem_name, block_info=None):
    """Count block reads, then delegate to AnnData's reader."""
    global block_reads
    block_reads += 1
    return read_block(path_or_sparse_dataset, elem_name, block_info=block_info)


# Patch before reading, so the lazy graph is built with the counting reader.
lazy_methods.make_dask_chunk = counting_read_block

adata = hp.tb.read_table(path, table_name="counts", mode="lazy")
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.pca(adata, n_comps=10, svd_solver="covariance_eigh")

block_reads = 0
with ComputeCounter() as counter:
    hp.tb.write_table_components(
        path,
        table_name="counts",
        components={("layers", "log1p"): adata.X, ("obsm", "X_pca"): adata.obsm["X_pca"]},
        obs_identity=adata.obs[["region", "cell_id"]],
        var_names=adata.var_names,
        overwrite=True,
    )
print(
    f"source X has {adata.X.numblocks[0]} row blocks; "
    f"write computes={counter.n}; source block reads during the write={block_reads}"
)
