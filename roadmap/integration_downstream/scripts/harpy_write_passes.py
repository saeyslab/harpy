"""Count Dask computes, source reads, write time and memory when Harpy writes lazy results.

Supports gap 2 and Phase 4 in ``../scanpy_rapids_singlecell.md``. A lazily read
CSR table goes through ``normalize_total``, ``log1p`` and, where needed, PCA.
Each case then writes some of the results with ``write_table_components``:

- ``log1p``: the sparse ``log1p`` layer alone, after ``normalize_total`` with
  its default ``target_sum=None``, the median of all cells' totals;
- ``log1p_fixed_target``: the same with ``target_sum=1e4``, so that the graph
  has no global reduction;
- ``x_pca``: the dense ``X_pca`` alone;
- ``both``: ``log1p`` and ``X_pca`` in one call;
- ``one_compute``: a reference, not a write. One ``dask.compute`` of a total per
  block of ``log1p`` and ``X_pca``: the reads, time and memory of a single pass
  over both results, without writing them.

Each case runs in its own process, so that peak memory starts from the same
point. Memory is the resident memory of that process, sampled every 20 ms
during the write: its peak minus its value when the write starts.

Usage, from the repository root::

    .venv/bin/python roadmap/integration_downstream/scripts/harpy_write_passes.py
    .venv/bin/python roadmap/integration_downstream/scripts/harpy_write_passes.py 1000000

Without arguments: 8,000 × 600 counts at 5% density, read in 1000-row blocks
(8 blocks), for exact counts within seconds. With a number of cells: that many
cells × 2,000 genes with 100 counts per cell, read with the readers' default
``"auto"`` blocks. ``--sparse-chunks`` sets the rows per block instead.

Relies on anndata 0.12 internals: ``make_dask_chunk`` reads one sparse block
from storage per call.
"""

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil
from _common import INSTANCE_KEY, REGION_KEY, ComputeCounter, annotated_table, make_counts, new_store, quiet
from scipy import sparse

CASES = ("log1p", "log1p_fixed_target", "x_pca", "both", "one_compute")
WRITTEN = {"log1p": ["log1p"], "log1p_fixed_target": ["log1p"], "x_pca": ["x_pca"], "both": ["log1p", "x_pca"]}


def make_large_counts(n_obs: int, n_vars: int, values_per_row: int, seed: int = 0) -> sparse.csr_matrix:
    """Create float32 CSR counts with the same number of values per row, fast for millions of rows.

    Each row has one value in each of ``values_per_row`` equal column windows,
    at a random position within the window.
    """
    rng = np.random.default_rng(seed)
    window = n_vars // values_per_row
    indices = np.arange(values_per_row) * window + rng.integers(0, window, size=(n_obs, values_per_row))
    data = (rng.poisson(3, size=n_obs * values_per_row) + 1).astype(np.float32)
    indptr = np.arange(0, n_obs * values_per_row + 1, values_per_row)
    return sparse.csr_matrix((data, indices.ravel().astype(np.int32), indptr), shape=(n_obs, n_vars))


class PeakMemory:
    """Sample this process's resident memory in a thread while the ``with`` block runs."""

    def __init__(self, interval: float = 0.02):
        self.interval = interval
        self._process = psutil.Process()

    def __enter__(self):
        self.start = self.peak = self._process.memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def _sample(self):
        while not self._stop.wait(self.interval):
            self.peak = max(self.peak, self._process.memory_info().rss)

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self._process.memory_info().rss)

    @property
    def increase_mib(self) -> float:
        """Peak resident memory minus that at the start, in MiB."""
        return (self.peak - self.start) / 2**20


def _block_total(block):
    """Reduce one block to a 1 × 1 array, so a compute keeps no results in memory."""
    return np.array([[block.sum()]])


def _block_totals(x):
    chunks = tuple((1,) * n for n in x.numblocks)
    return x.map_blocks(_block_total, dtype=np.float64, meta=np.empty((0, 0)), chunks=chunks)


def run_case(store: str, case: str, sparse_chunks: str | int) -> dict:
    """Run one case in this process and return its measurements."""
    import anndata._io.specs.lazy_methods as lazy_methods
    import dask
    import scanpy as sc

    import harpy as hp

    quiet()
    reads = 0
    read_block = lazy_methods.make_dask_chunk

    def counting_read_block(path_or_sparse_dataset, elem_name, block_info=None):
        """Count block reads, then delegate to AnnData's reader."""
        nonlocal reads
        reads += 1
        return read_block(path_or_sparse_dataset, elem_name, block_info=block_info)

    # Patch before reading, so the lazy graph is built with the counting reader.
    lazy_methods.make_dask_chunk = counting_read_block

    adata = hp.tb.read_table(store, table_name="counts", mode="lazy", sparse_chunks=sparse_chunks)
    sc.pp.normalize_total(adata, target_sum=1e4 if case == "log1p_fixed_target" else None)
    sc.pp.log1p(adata)
    if case in {"x_pca", "both", "one_compute"}:
        # covariance_eigh fits eagerly; only the transform into X_pca stays lazy.
        sc.pp.pca(adata, n_comps=50, svd_solver="covariance_eigh")
    results = {"log1p": (("layers", "log1p"), adata.X)}
    if "X_pca" in adata.obsm:
        results["x_pca"] = (("obsm", "X_pca"), adata.obsm["X_pca"])

    reads = 0
    with ComputeCounter() as counter, PeakMemory() as memory:
        start = time.perf_counter()
        if case == "one_compute":
            dask.compute(*(_block_totals(value) for _, value in results.values()))
        else:
            hp.tb.write_table_components(
                store,
                table_name="counts",
                components=dict(results[name] for name in WRITTEN[case]),
                obs_identity=adata.obs[[REGION_KEY, INSTANCE_KEY]],
                var_names=adata.var_names,
                overwrite=True,
            )
        seconds = time.perf_counter() - start
    return {
        "row_blocks": adata.X.numblocks[0],
        "computes": counter.n,
        "reads": reads,
        "seconds": seconds,
        "memory_mib": memory.increase_mib,
    }


def make_store(n_obs: int | None) -> str:
    """Write the counts table with Harpy's writer, so it has Harpy's stored layout."""
    import harpy as hp

    quiet()
    counts = make_counts(8000, 600, density=0.05) if n_obs is None else make_large_counts(n_obs, 2000, 100)
    store = new_store({})
    hp.tb.write_table(store, table_name="counts", adata=annotated_table(counts))
    print(f"table: {counts.shape[0]:,} × {counts.shape[1]:,}, {counts.nnz:,} non-zero values", flush=True)
    return store


def main() -> None:
    """Write the table, run each case in its own process, and print one line per case."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("n_obs", nargs="?", type=int, help="number of cells of a large table")
    parser.add_argument("--sparse-chunks", help="rows per lazy block, or 'auto'")
    parser.add_argument("--store", help=argparse.SUPPRESS)
    parser.add_argument("--case", choices=CASES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    sparse_chunks = args.sparse_chunks or ("auto" if args.n_obs else "1000")
    sparse_chunks = sparse_chunks if sparse_chunks == "auto" else int(sparse_chunks)

    if args.case:
        # Child process: run one case and report it on stdout.
        print("RESULT " + json.dumps(run_case(args.store, args.case, sparse_chunks)), flush=True)
        return

    store = make_store(args.n_obs)
    try:
        for case in CASES:
            command = [sys.executable, __file__, "--store", store, "--case", case]
            command += ["--sparse-chunks", str(sparse_chunks)]
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            lines = [line for line in completed.stdout.splitlines() if line.startswith("RESULT ")]
            if completed.returncode != 0 or not lines:
                print(f"{case}: failed\n{completed.stderr[-2000:]}", flush=True)
                continue
            result = json.loads(lines[-1].removeprefix("RESULT "))
            print(
                f"{case}: {result['row_blocks']} row blocks; computes={result['computes']}; "
                f"source block reads={result['reads']}; {result['seconds']:.1f} s; "
                f"peak memory +{result['memory_mib']:,.0f} MiB",
                flush=True,
            )
    finally:
        shutil.rmtree(Path(store).parent, ignore_errors=True)


if __name__ == "__main__":
    main()
