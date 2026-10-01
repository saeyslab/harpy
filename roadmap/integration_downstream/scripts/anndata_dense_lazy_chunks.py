"""Check that AnnData's lazy reader returns row-only chunks from column-split dense storage.

Supports Gap 1 (dense matrices, proposed change 2 and "Aligned reading, not a rechunk")
in ``../scanpy_rapids_singlecell.md``. Passing ``chunks`` to ``read_elem_lazy`` produces
the row-only layout directly, without a ``rechunk`` layer in the Dask graph.
"""

import tempfile

import numpy as np
import zarr
from anndata.experimental import read_elem_lazy
from anndata.io import write_elem


def layer_names(array) -> list[str]:
    """Return the Dask graph layer names without their unique token suffixes."""
    return [name.split("-")[0] for name in array.dask.layers]


group = zarr.open_group(f"{tempfile.mkdtemp()}/table.zarr", mode="w")
values = np.arange(4000 * 600, dtype=np.float32).reshape(4000, 600)
write_elem(group, "X", values, dataset_kwargs={"chunks": (1000, 150)})
print("stored chunks:", group["X"].chunks)

stored = read_elem_lazy(group["X"])
print("read_elem_lazy(element): chunks", stored.chunksize, "numblocks", stored.numblocks)

built_in = read_elem_lazy(group["X"], chunks=(1000, 600))
print("read_elem_lazy(element, chunks=(1000, 600)): chunks", built_in.chunksize, "numblocks", built_in.numblocks)
print("values equal:", bool((built_in.compute() == values).all()))

afterwards = read_elem_lazy(group["X"]).rechunk((1000, 600))
print("graph, chunks passed to read_elem_lazy:", layer_names(built_in), "tasks:", len(built_in.dask))
print("graph, rechunk afterwards:", layer_names(afterwards), "tasks:", len(afterwards.dask))
