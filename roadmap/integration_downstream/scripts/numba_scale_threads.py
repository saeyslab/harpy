"""Reproduce the Numba crash from ``sc.pp.scale`` on dense Dask data under Dask threads.

Supports "Numba and Dask threads" in ``../scanpy_rapids_singlecell.md``.

Without arguments, the script runs itself three times per configuration in child
processes, because the crash terminates the Python process, and reports each
outcome. With a scheduler argument (``threads`` or ``synchronous``), it runs one
child experiment.
"""

import os
import subprocess
import sys

CONFIGURATIONS = [
    ("threads", {}),
    ("synchronous", {}),
    ("threads", {"NUMBA_NUM_THREADS": "1"}),
]


def child(scheduler: str) -> None:
    """Scale a dense Dask matrix, then run PCA, with the given Dask scheduler."""
    import warnings

    warnings.filterwarnings("ignore")
    import anndata as ad
    import dask
    import dask.array as da
    import numpy as np
    import scanpy as sc

    X = da.random.default_rng(0).random((4000, 600), chunks=(1000, -1)).astype(np.float32)
    adata = ad.AnnData(X=X)
    with dask.config.set(scheduler=scheduler):
        sc.pp.scale(adata, max_value=10)
        sc.pp.pca(adata, n_comps=20, svd_solver="covariance_eigh")
        adata.obsm["X_pca"] = np.asarray(adata.obsm["X_pca"])
    print("completed", adata.obsm["X_pca"].shape, flush=True)


def main() -> None:
    """Run each configuration three times in child processes and report the outcomes."""
    for scheduler, extra_env in CONFIGURATIONS:
        label = f"scheduler={scheduler}" + "".join(f", {key}={value}" for key, value in extra_env.items())
        for attempt in range(1, 4):
            result = subprocess.run(
                [sys.executable, __file__, scheduler],
                env={**os.environ, **extra_env},
                capture_output=True,
                text=True,
                check=False,
            )
            output = result.stdout + result.stderr
            if result.returncode == 0 and "completed" in output:
                outcome = "completed"
            elif "Numba workqueue threading layer is terminating" in output:
                outcome = f"crashed (exit code {result.returncode}): Numba workqueue threading layer is terminating"
            else:
                last_line = output.strip().splitlines()[-1] if output.strip() else ""
                outcome = f"failed (exit code {result.returncode}): {last_line[:200]}"
            print(f"{label}, run {attempt}: {outcome}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        child(sys.argv[1])
    else:
        main()
