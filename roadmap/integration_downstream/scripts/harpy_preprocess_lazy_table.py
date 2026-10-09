"""Run ``hp.tb.preprocess_transcriptomics`` on a table read by ``hp.io.read_zarr``, which is lazy by default.

Supports the correction in Gap 5 of ``../scanpy_rapids_singlecell.md``: Harpy's own
wrappers already receive lazy tables when stores are opened with Harpy's reader, and
``preprocess_transcriptomics`` currently fails on them at ``sc.pp.scale``.
"""

import tempfile
import traceback

import dask
import dask.array as da
import numpy as np
from _common import annotated_table, quiet
from scipy import sparse
from spatialdata import SpatialData
from spatialdata.models import Labels2DModel

import harpy as hp

quiet()
dask.config.set(scheduler="synchronous")

# Every pixel is its own cell, so cell sizes exist for size normalization.
side, n_vars = 60, 300
labels = np.arange(1, side * side + 1, dtype=np.int32).reshape(side, side)
rng = np.random.default_rng(0)
counts = sparse.random(side * side, n_vars, density=0.2, format="csr", random_state=0, dtype=np.float32)
counts.data = rng.poisson(5, counts.nnz).astype(np.float32) + 1
path = f"{tempfile.mkdtemp()}/sdata.zarr"
SpatialData(
    labels={"cells": Labels2DModel.parse(labels, dims=("y", "x"))}, tables={"counts": annotated_table(counts)}
).write(path)

sdata = hp.io.read_zarr(path)
print("hp.io.read_zarr table X:", type(sdata.tables["counts"].X).__name__)
try:
    sdata = hp.tb.preprocess_transcriptomics(
        sdata,
        labels_name="cells",
        table_name="counts",
        output_table_name="preprocessed",
        min_counts=1,
        min_cells=1,
        n_comps=10,
        update_shapes_elements=False,
        overwrite=True,
    )
    X = sdata.tables["preprocessed"].X
    print("preprocess_transcriptomics on the lazy table: OK; output X:", type(X).__name__, isinstance(X, da.Array))
except Exception as error:  # noqa: BLE001 - report the failure and where it happened
    print("preprocess_transcriptomics on the lazy table: FAIL:", type(error).__name__, str(error).splitlines()[0])
    for frame in traceback.extract_tb(error.__traceback__):
        if "harpy/table" in frame.filename:
            print(f"   at {frame.filename.split('src/')[-1]}:{frame.lineno}: {frame.line}")
