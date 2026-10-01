# Experiment scripts

These scripts produced the verified results in
[`../scanpy_rapids_singlecell.md`](../scanpy_rapids_singlecell.md). Each one builds
synthetic data in a temporary folder and prints its results; none of them writes
inside the repository.

Run them from the repository root with the project environment:

```bash
.venv/bin/python roadmap/integration_downstream/scripts/harpy_scanpy_write_back.py
```

Shared helpers live in `_common.py`. Results were recorded on macOS (Apple
silicon) with Python 3.13, scanpy 1.11.1, anndata 0.12.10, dask 2026.7.1,
zarr 3.2.1 and spatialdata 0.8.0, without dask-ml, scikit-misc, TBB or OpenMP.
Some scripts rely on private scanpy or anndata functions of those versions and
may need updating for other versions.

## Harpy table I/O

| Script                                | Shows                                                                                                     | Expected output                                                                                                                                                                           |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `spatialdata_eager_tables.py`         | `sd.read_zarr` loads table matrices into memory; labels stay lazy.                                        | Table `X` is `csr_matrix`; labels data is a Dask `Array`.                                                                                                                                 |
| `harpy_scanpy_write_back.py`          | Lazy read → scanpy → `write_table_components` → reopen, then graphs written in a second call.             | All steps `ok`; written layer, `X_pca` and connectivities match; `X` still holds the counts.                                                                                              |
| `harpy_scanpy_filtering_and_dense.py` | The scanpy step table with compute counts, writes after filtering, stale lazy tables, and dense chunking. | Component write after filtering refused; `write_table` result matches; stale `X` fails with a reshape error; dense lazy `X` has `(1000, 150)` chunks and PCA works only after rechunking. |
| `harpy_write_passes.py`               | Source reads when one call writes two lazy results (Gap 2).                                               | 4 row blocks, 5 computes, 28 block reads.                                                                                                                                                 |
| `harpy_csc_tables.py`                 | CSC-stored tables read lazily, before and after conversion to row-chunked CSR (Gap 1, CSC matrices).      | As read, PCA is refused for 600 and 2500 genes, and QC fails for 2500 genes; after conversion all steps run.                                                                              |

## Chunking and threading

| Script                          | Shows                                                                                                                                                                                                                                               | Expected output                                                                                                                                                                                                                                                                                                          |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `anndata_dense_lazy_chunks.py`  | Which stored chunks AnnData chooses when writing NumPy and Dask arrays, and `read_elem_lazy` with explicit row-only chunks on column-split dense storage, compared with a `rechunk` afterwards (Gap 1, "Evidence on writes" and proposed change 2). | Writes: NumPy and Dask `(10000, 600)` both stored as `(5000, 75)`, explicit `dataset_kwargs` chunks as `(10000, 600)`. Reads: without `chunks`, `(1000, 150)` blocks, 4 × 4; with `chunks=(1000, 600)`, `(1000, 600)` blocks, 4 × 1, same values, 5 tasks and no `rechunk` layer; a `rechunk` afterwards needs 21 tasks. |
| `anndata_sparse_block_reads.py` | How sparse matrices are chunked on disk, and how often lazy reads fetch each stored `data` chunk for small and large row blocks (Gap 1, "Sparse block size").                                                                                       | `data`/`indices` stored in chunks of 250,000 non-zero values; 1000-row blocks fetch each stored chunk 7.2 times on average, 50,000-row blocks 1.1 times.                                                                                                                                                                 |
| `dask_densify_split.py`         | Dask's `"auto"` on sparse-backed arrays and its `array.chunk-size` setting, splitting rows before densifying, and peak memory of `scale(zero_center=True)` (Gap 1, "Memory target" and "Densifying steps").                                         | `"auto"` gives 16,777 rows at 128 MiB and 4,194 at 32 MiB, as for dense float32; each split block depends on one block; peak 3,083 MiB for one 100,000-row block, 336 MiB for 10,000-row blocks.                                                                                                                         |
| `numba_dense_pca.py`            | Dense Dask PCA with `covariance_eigh` for row-only and column-split chunks.                                                                                                                                                                         | Row-only chunks succeed; column-split chunks fail with "Only dask arrays with chunking along the first axis are supported".                                                                                                                                                                                              |
| `numba_scale_threads.py`        | `sc.pp.scale` on dense Dask data crashing under Dask threads. Runs each configuration three times in child processes.                                                                                                                               | Threaded runs crash with "Numba workqueue threading layer is terminating", also with `NUMBA_NUM_THREADS=1`; synchronous runs complete.                                                                                                                                                                                   |

## scanpy on Dask arrays

These build an in-memory AnnData with a Dask `X` (2000 × 500), without Harpy.

| Script                                    | Shows                                                                                                                                                                        | Expected output                                                                                                                          |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `scanpy_functions.py KIND [ROWS,COLUMNS]` | Each scanpy function on Dask `X`, compared with the same data in memory. `KIND` is `csr_matrix`, `csr_array` or `dense`.                                                     | See "Upstream limitations to document"; for example, `seurat_v3` fails on `csr_matrix`, and `wilcoxon` and `get.aggregate` fail on both. |
| `scanpy_pipeline.py KIND [ROWS,COLUMNS]`  | QC → normalize → log1p → HVG → PCA → neighbors → UMAP → Leiden.                                                                                                              | Completes for `csr_matrix`, and for `dense` with `covariance_eigh`; with `csr_array` only `log1p` succeeds.                              |
| `scanpy_probe_whole_matrix_tasks.py`      | Block shapes received by scanpy's nonzero counter and `check_nonnegative_integers`.                                                                                          | Both receive the full `(2000, 500)` matrix as one block; with column-split chunks the nonzero counter fails.                             |
| `scanpy_probe_seurat_v3.py`               | `seurat_v3` on dense Dask versus in-memory data with outliers.                                                                                                               | About 92% of `highly_variable` flags agree.                                                                                              |
| `scanpy_probe_scale_then_pca.py`          | Sparse `scale(zero_center=True)` produces `np.matrix` blocks that break PCA, and the conversion that fixes it. Fixed in scanpy 1.11.2; this probe shows the 1.11.1 behavior. | PCA fails after scaling and succeeds after `map_blocks(np.asarray, ...)`.                                                                |

`flavor="seurat_v3"` needs scikit-misc, which was not installed. The scripts use
a quadratic stand-in for its loess fit (`_common.install_fake_skmisc`), so only
the Dask behavior of `seurat_v3` is meaningful, not its gene selection.
