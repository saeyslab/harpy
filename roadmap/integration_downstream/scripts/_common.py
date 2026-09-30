"""Shared helpers for the downstream-integration experiments.

Run the scripts in this folder from the repository root, for example::

    .venv/bin/python roadmap/integration_downstream/scripts/harpy_scanpy_write_back.py

Python puts the script's folder on ``sys.path``, so the scripts import this module
directly.
"""

import sys
import tempfile
import traceback
import types
import warnings
from collections.abc import Callable

import anndata as ad
import dask.array as da
import numpy as np
import pandas as pd
from dask.callbacks import Callback
from scipy import sparse

REGION = "cells"
REGION_KEY = "region"
INSTANCE_KEY = "cell_id"

_WARNING_KEYWORDS = ("Dask", "dask", "zero-center", "Rechunk", "chunk", "sparse", "densif", "non-integer")


class ComputeCounter(Callback):
    """Count Dask graph evaluations, and their tasks, inside a ``with`` block."""

    def __init__(self):
        super().__init__()
        self.n = 0
        self.ntasks = 0

    def _start(self, dsk):
        self.n += 1
        self.ntasks += len(dsk)


def quiet() -> None:
    """Hide deprecation noise and Harpy's publication log messages."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from loguru import logger

    logger.remove()


def describe(x: object) -> str:
    """Summarize a matrix: Dask chunking and block type, or its Python type."""
    if isinstance(x, da.Array):
        return f"dask[{type(x._meta).__name__}] chunks={x.chunksize} numblocks={x.numblocks}"
    return type(x).__name__


def run_step(name: str, fn: Callable, *args, verbose: bool = False, **kwargs) -> tuple[object, bool]:
    """Run one experiment step and print its outcome and number of Dask computes.

    A string result is printed after the outcome. Dask-related warnings raised by
    the step are printed below it. Returns the result and whether the step succeeded.
    """
    counter = ComputeCounter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            with counter:
                out = fn(*args, **kwargs)
            ok, message = True, ""
        except Exception as error:  # noqa: BLE001 - report every failure and continue with the next step
            out, ok = None, False
            message = f": {type(error).__name__}: {str(error).splitlines()[0][:220]}"
            if verbose:
                traceback.print_exc()
    detail = f" -> {out}" if isinstance(out, str) else ""
    print(f"[{'ok' if ok else 'FAIL'}] {name}: computes={counter.n}{message}{detail}", flush=True)
    for warning in sorted({f"{w.category.__name__}: {str(w.message)[:160]}" for w in caught}):
        if "deprecated" not in warning.lower() and any(key in warning for key in _WARNING_KEYWORDS):
            print(f"       warning: {warning}")
    return out, ok


def make_counts(n_obs: int = 2000, n_vars: int = 500, density: float = 0.1, seed: int = 0) -> sparse.csr_matrix:
    """Create Poisson-like sparse counts as a float32 CSR matrix."""
    rng = np.random.default_rng(seed)
    counts = sparse.random(
        n_obs, n_vars, density=density, format="csr", random_state=seed, data_rvs=lambda k: rng.poisson(3, k) + 1
    )
    return counts.astype(np.float32)


def make_dask_adata(
    kind: str = "csr_matrix", chunks: tuple[int, int] = (500, 500), n_obs: int = 2000, n_vars: int = 500
) -> ad.AnnData:
    """Create an in-memory AnnData whose ``X`` is a Dask array of the given block type.

    ``kind`` is ``"csr_matrix"``, ``"csr_array"`` or ``"dense"``. ``obs`` has a
    categorical ``group`` and a continuous ``cont`` column; ``var["mt"]`` marks the
    first ten genes.
    """
    dense = make_counts(n_obs, n_vars).toarray()
    blocks = da.from_array(dense, chunks=chunks)
    if kind == "dense":
        X = blocks
    elif kind in {"csr_matrix", "csr_array"}:
        block_type = getattr(sparse, kind)
        X = blocks.map_blocks(block_type, dtype=dense.dtype, meta=block_type((0, 0), dtype=dense.dtype))
    else:
        raise ValueError(f"Unknown kind {kind!r}.")
    rng = np.random.default_rng(1)
    obs = pd.DataFrame(index=[f"c{i}" for i in range(n_obs)])
    obs["group"] = pd.Categorical(rng.choice(["a", "b", "c", "d"], n_obs))
    obs["cont"] = rng.normal(size=n_obs)
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_vars)])
    var["mt"] = np.arange(n_vars) < 10
    return ad.AnnData(X=X, obs=obs, var=var)


def annotated_table(X: object) -> ad.AnnData:
    """Wrap a matrix in a SpatialData table annotating the labels element ``cells``."""
    from spatialdata.models import TableModel

    n_obs, n_vars = X.shape
    obs = pd.DataFrame(
        {REGION_KEY: pd.Categorical([REGION] * n_obs), INSTANCE_KEY: np.arange(1, n_obs + 1)},
        index=[str(i) for i in range(n_obs)],
    )
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_vars)])
    return TableModel.parse(
        ad.AnnData(X=X, obs=obs, var=var), region=REGION, region_key=REGION_KEY, instance_key=INSTANCE_KEY
    )


def new_store(tables: dict[str, ad.AnnData]) -> str:
    """Write a SpatialData store with one labels element and the given tables to a temporary folder."""
    from spatialdata import SpatialData
    from spatialdata.models import Labels2DModel

    path = f"{tempfile.mkdtemp()}/sdata.zarr"
    labels = Labels2DModel.parse(np.zeros((64, 64), dtype=np.int32), dims=("y", "x"))
    SpatialData(labels={REGION: labels}, tables=tables).write(path)
    return path


def install_fake_skmisc() -> None:
    """Install a stand-in for ``skmisc.loess``, which ``flavor="seurat_v3"`` needs.

    scikit-misc was not installed. The stand-in fits a quadratic instead of a loess
    curve. It is only used to reach scanpy's Dask code paths; results that depend
    on the fitted curve are not meaningful.
    """

    class _Outputs:
        pass

    class loess:  # mirrors the skmisc class name
        def __init__(self, x, y, span=0.3, degree=2):
            self.x = np.asarray(x, dtype=float)
            self.y = np.asarray(y, dtype=float)
            self.outputs = _Outputs()

        def fit(self):
            coefficients = np.polyfit(self.x, self.y, 2)
            self.outputs.fitted_values = np.polyval(coefficients, self.x)

    package = types.ModuleType("skmisc")
    module = types.ModuleType("skmisc.loess")
    module.loess = loess
    package.loess = module
    sys.modules["skmisc"] = package
    sys.modules["skmisc.loess"] = module
