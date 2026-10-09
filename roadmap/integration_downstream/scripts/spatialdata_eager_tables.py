"""Show that ``sd.read_zarr`` loads table matrices into memory, while labels stay lazy.

Supports "What Harpy's table I/O contributes" in ``../scanpy_rapids_singlecell.md``.
"""

import spatialdata as sd
from _common import annotated_table, make_counts, new_store, quiet

quiet()
path = new_store({"counts": annotated_table(make_counts(2000, 50))})
sdata = sd.read_zarr(path)
print("sd.read_zarr table X:", type(sdata.tables["counts"].X).__name__)
print("sd.read_zarr labels data:", type(sdata.labels["cells"].data).__name__)
