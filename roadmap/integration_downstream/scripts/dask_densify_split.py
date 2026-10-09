"""Sparse blocks sized by non-zero values, and splitting rows before a step densifies them.

Supports Gap 1 ("Sparse block size": memory target and densifying steps) in
``../scanpy_rapids_singlecell.md``:

A. Dask's ``"auto"`` sizes a sparse-backed array as if it were dense, from
   ``array.chunk-size``, so ``X.rechunk({0: "auto"})`` gives dense-safe rows.
B. Splitting row blocks is a simple split: each smaller block depends on one block.
C. Peak memory of ``scale(zero_center=True)`` plus a reduction, for one large block
   versus split blocks. Uses the synchronous scheduler, so one block is processed at
   a time, and ``tracemalloc``, which tracks NumPy allocations.
"""

import tempfile
import tracemalloc
import warnings

import anndata as ad
import dask
import dask.array as da
import numpy as np
import scanpy as sc
import zarr
from anndata.experimental import read_elem_lazy
from anndata.io import write_elem
from scipy import sparse

warnings.filterwarnings("ignore")
dask.config.set(scheduler="synchronous")

n_obs, n_vars = 100_000, 2000
path = f"{tempfile.mkdtemp()}/table.zarr"
counts = sparse.random(n_obs, n_vars, density=0.02, format="csr", dtype=np.float32, random_state=0)
write_elem(zarr.open_group(path, mode="w"), "X", counts)
group = zarr.open_group(path, mode="r")
print(
    f"matrix {n_obs} x {n_vars}, {counts.nnz} non-zero values; dense float32 size {n_obs * n_vars * 4 / 2**20:.0f} MiB"
)

# A. Dask's "auto" for a sparse-backed array, and how array.chunk-size changes it.
lazy = read_elem_lazy(group["X"], chunks=(n_obs, -1))
for chunk_size in ("128MiB", "32MiB"):
    with dask.config.set({"array.chunk-size": chunk_size}):
        rows = lazy.rechunk({0: "auto", 1: -1}).chunks[0][0]
        dense_rows = da.core.normalize_chunks(("auto", -1), shape=(n_obs, n_vars), dtype=np.float32)[0][0]
    print(
        f"A. array.chunk-size={chunk_size}: rechunk({{0: 'auto'}}) on CSR blocks -> {rows} rows; dense float32 -> {dense_rows} rows"
    )

# B. Splitting 50,000-row blocks into 10,000-row blocks: input blocks per output block.
large = read_elem_lazy(group["X"], chunks=(50_000, -1))
split = large.rechunk((10_000, -1))
graph = dict(split.__dask_graph__())
input_keys = {key for key in graph if isinstance(key, tuple) and key[0] == large.name}
inputs_per_output = set()
for output_key in dask.core.flatten(split.__dask_keys__()):
    reached, stack = set(), [output_key]
    while stack:
        for dependency in dask.core.get_dependencies(graph, stack.pop()):
            if dependency in input_keys:
                reached.add(dependency)
            else:
                stack.append(dependency)
    inputs_per_output.add(len(reached))
print(
    f"B. split 50,000 -> 10,000 rows: {split.numblocks[0]} blocks; input blocks per block: {sorted(inputs_per_output)}"
)


def peak_mib(rows: int) -> float:
    """Peak traced memory of scale(zero_center=True) plus column sums, with the given rows per block."""
    adata = ad.AnnData(X=read_elem_lazy(group["X"], chunks=(n_obs, -1)))
    if rows != n_obs:
        adata.X = adata.X.rechunk((rows, -1))
    sc.pp.scale(adata, zero_center=True)
    # scanpy 1.11.1 returns np.matrix blocks here (fixed in 1.11.2); convert for the reduction.
    dense = adata.X.map_blocks(np.asarray, meta=np.array([], dtype=adata.X.dtype))
    tracemalloc.start()
    dense.sum(axis=0).compute()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 2**20


for rows in (n_obs, 10_000):
    print(
        f"C. scale(zero_center=True) + column sums, {rows} rows per block: peak traced memory {peak_mib(rows):.0f} MiB"
    )
