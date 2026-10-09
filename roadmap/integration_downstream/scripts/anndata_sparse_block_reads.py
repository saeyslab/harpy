"""How sparse matrices are chunked on disk, and how often lazy reads fetch each stored chunk.

Supports Gap 1 ("Sparse block size") in ``../scanpy_rapids_singlecell.md``:

- ``data`` and ``indices`` are chunked by number of non-zero values, not by rows;
- the total number of non-zero values is available from metadata alone;
- small row blocks fetch the same stored chunk from several tasks.

Reads go through a Zarr store that counts fetches of stored ``data`` chunks.
"""

import tempfile
from collections import Counter

import numpy as np
import zarr
from anndata.experimental import read_elem_lazy
from anndata.io import write_elem
from scipy import sparse
from zarr.storage import LocalStore


class CountingStore(LocalStore):
    """Local Zarr store that counts fetches of the sparse matrix's stored ``data`` chunks."""

    fetches: Counter = Counter()

    async def get(self, key, prototype, byte_range=None):
        """Count the fetch, then delegate to the local store."""
        if key.startswith("X/data/c"):
            CountingStore.fetches[key] += 1
        return await super().get(key, prototype, byte_range)


path = f"{tempfile.mkdtemp()}/table.zarr"
X = sparse.random(200_000, 2000, density=0.02, format="csr", dtype=np.float32, random_state=0)
write_elem(zarr.open_group(path, mode="w"), "X", X)

group = zarr.open_group(store=CountingStore(path), mode="r")
for name in ("data", "indices", "indptr"):
    array = group["X"][name]
    print(f"stored {name}: shape={array.shape} chunks={array.chunks} dtype={array.dtype}")
n_rows = group["X"].attrs["shape"][0]
nnz = group["X"]["data"].shape[0]
stored_chunks = -(-nnz // group["X"]["data"].chunks[0])
print(f"non-zero values from metadata: {nnz}; average per row: {nnz / n_rows:.1f}; stored data chunks: {stored_chunks}")

for rows in (1000, 50_000):
    CountingStore.fetches.clear()
    lazy = read_elem_lazy(group["X"], chunks=(rows, -1))
    lazy.compute()
    fetches = sum(CountingStore.fetches.values())
    print(
        f"{rows} rows per block: {lazy.numblocks[0]} blocks of about {rows * nnz // n_rows} non-zero values; "
        f"stored data chunk fetches={fetches} for {stored_chunks} stored chunks ({fetches / stored_chunks:.1f} per chunk)"
    )
