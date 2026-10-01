# Downstream analysis with scanpy and rapids-singlecell

Date: 2026-09-30

## Summary

This document evaluates whether Harpy's new table I/O (`hp.tb.read_table`,
`hp.tb.read_table_components`, `hp.tb.write_table`,
`hp.tb.write_table_components`, `hp.tb.delete_table_components`) makes it easier
to run scanpy and rapids-singlecell on large tables stored in SpatialData Zarr
stores, using lazy (Dask) matrices instead of loading tables into memory.

The answer is mostly yes:

- `sd.read_zarr` loads table matrices into memory. For a SpatialData store,
  `hp.tb.read_table(..., mode="lazy")` is therefore the practical way to obtain a
  lazy table without assembling one from AnnData's low-level readers.
- For CSR-stored matrices, the lazy layout that Harpy produces is exactly what
  scanpy and rapids-singlecell require: Dask arrays chunked along rows only, with
  every chunk spanning all features, and `scipy.sparse.csr_matrix` blocks.
- A standard scanpy pipeline ran unchanged on a lazily read Harpy table, and its
  results were persisted with `hp.tb.write_table_components` without rewriting
  the stored counts matrix.
- The identity checks in `write_table_components` correctly refuse to write
  results after cells or genes have been filtered. `hp.tb.write_table` handles
  that case, including when the lazy input still reads from the table being
  replaced.

Three Harpy-side changes would remove most of the remaining friction, in this
order:

1. Lazy chunk layouts do not suit the downstream tools. Two stored layouts are
   split along the feature axis, which scanpy's PCA and QC metrics and
   rapids-singlecell's input check reject:
   - dense matrices, because lazy reads keep their on-disk chunks and Harpy's
     writer produces dense chunks that split the feature axis;
   - CSC matrices, because lazy reads chunk them by genes, with all cells in
     every block. Harpy's Visium and Visium HD readers store `X` as CSC.

   In addition, sparse blocks default to a fixed 1000 rows, far fewer than most
   spatial tables need.

2. Writing several lazy results in one call recomputes their shared upstream
   graph once per result. In the test below, the source matrix was read about
   seven times for one write.
3. There is no helper that persists "what scanpy changed". Callers must know
   where each scanpy function stores its outputs and list those components.

scanpy and rapids-singlecell also have their own limits on lazy input. Those
cannot be fixed in Harpy, but they should be documented next to the recommended
usage pattern.

## Scope and method

The evaluation covered three things:

1. Harpy's table I/O source and the storage contracts in
   `docs/development/storage.md`.
2. Experiments that read synthetic SpatialData stores with Harpy, ran scanpy on
   the lazy tables, wrote the results back with Harpy and reopened them.
3. A source review of scanpy 1.11.1 and of rapids-singlecell at commit `ed56fe5`
   (`main`, 2026-09-30; latest release v0.17.0), including its tutorials
   repository at commit `d24597e`.

rapids-singlecell requires CUDA and could not be run on the test machine. All
statements about it come from its source code, documentation and tutorials.

Evidence labels used below:

- **Verified**: run locally and checked against the equivalent in-memory result
  where applicable.
- **Source**: read from the source or documentation of the stated version, not
  executed.
- **Inferred**: a conclusion from the source that has not been confirmed by
  running it.

Environment for verified results: macOS (Apple silicon), Python 3.13,
scanpy 1.11.1, anndata 0.12.10, dask 2026.7.1, zarr 3.2.1, spatialdata 0.8.0.
dask-ml, TBB and OpenMP were not available.

## What Harpy's table I/O contributes

**Reading.** `hp.tb.read_table(store, table_name=..., mode="lazy")` opens one
table without opening other SpatialData elements.

- Sparse matrices become Dask arrays with `csr_matrix` or `csc_matrix` blocks,
  matching the stored format. The compressed axis is split into
  `sparse_chunk_size` rows (CSR) or columns (CSC), default 1000, and the other
  axis is kept whole.
- Dense matrices become Dask arrays that keep the on-disk Zarr chunks.
- `obs`, `var` and `uns` are always loaded into memory.

`hp.tb.read_table_components` reads selected components, for example only
`("obs",)`, without constructing matrix graphs.

**Writing.** `hp.tb.write_table_components` replaces selected components of an
existing table:

- `obs`, `var`, individual `layers`/`obsm`/`varm`/`obsp`/`varp` entries, and
  whole or nested `uns` entries;
- in memory, lazy or storage-backed values;
- with the observation and feature identities checked against the stored table;
- staged, then published with rollback on handled failures.

Unrequested components, including `X`, are not read or rewritten.
`hp.tb.write_table` replaces a complete table. It is the path for results whose
axes changed, such as filtered cells or genes.

Compared with AnnData's own `read_elem_lazy` and `write_elem`, Harpy adds:

- locating and validating a table inside a SpatialData store;
- a sparse chunking policy that suits downstream Dask consumers;
- identity checks that stop stale results from being attached to the wrong
  observations or features;
- staged publication, rollback and consolidated-metadata handling.

**SpatialData's reader is eager for tables (verified).** After
`sd.read_zarr(path)`, `sdata.tables[name].X` was a `scipy.sparse.csr_matrix`,
whereas the labels in the same store were lazy Dask arrays. Harpy's functions
that start from `sdata.tables` therefore receive fully loaded tables.

## Verified: scanpy on a lazily read Harpy table

Test data: a synthetic SpatialData store with one labels element and an annotated
table of 4000 cells × 600 genes (Poisson counts, 5% non-zero), written with
`SpatialData.write` and read back with `hp.tb.read_table(..., mode="lazy")`.
Lazy `X` had chunks `(1000, 600)` and `csr_matrix` blocks. "Computes" counts
Dask graph evaluations during each step.

| Step                                                 | Result | Computes | Notes                                                                                                                                  |
| ---------------------------------------------------- | ------ | -------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `pp.calculate_qc_metrics(percent_top=None)`          | OK     | 4        | `obs`/`var` columns are loaded into memory; `X` is untouched. The default `percent_top` needs at least 500 genes, independent of Dask. |
| `pp.filter_cells`, `pp.filter_genes`                 | OK     | 2 each   | `X` is subset lazily.                                                                                                                  |
| `pp.normalize_total`, `pp.log1p`                     | OK     | 0        | `X` stays lazy with `csr_matrix` blocks.                                                                                               |
| `pp.highly_variable_genes` (`seurat`)                | OK     | 1        |                                                                                                                                        |
| `pp.scale` (sparse, `zero_center=True`)              | OK     | 2        | See the upstream limits below for the effect on later PCA.                                                                             |
| `pp.pca` (sparse)                                    | OK     | 1        | scanpy uses its own `covariance_eigh` solver; `obsm["X_pca"]` is a lazy Dask array.                                                    |
| `pp.neighbors` on reopened lazy `X_pca`, `tl.leiden` | OK     | 2 and 0  | Graphs are in-memory SciPy matrices.                                                                                                   |

**Writing the results back (verified).** One `write_table_components` call wrote
these components:

- `("obs",)` and `("var",)`
- `("layers", "log1p")` (lazy CSR)
- `("obsm", "X_pca")` (lazy dense)
- `("varm", "PCs")`
- `("uns", "log1p")`, `("uns", "hvg")` and `("uns", "pca")`

A second call wrote `("obsp", "connectivities")`, `("obsp", "distances")`,
`("uns", "neighbors")` and `("uns", "leiden")` together with `obs`. After
reopening:

- all components were present, with a lazy CSR layer and a lazy dense `X_pca`;
- the stored `layers["log1p"]`, `obsm["X_pca"]` and `obsp["connectivities"]`
  matched the values computed before writing;
- `X` still held the original counts.

**After filtering (verified).** When `filter_cells`/`filter_genes` had changed
the table's shape, `write_table_components` refused with
`obs_identity must match the expected identities exactly in value and order`.
`write_table(..., overwrite=True)` then published the filtered table, even
though its lazy `X` still read from the table being replaced (5 computes). The
reopened `X` matched the expected values computed before the write.

**Dense tables (verified).** A dense 4000 × 600 table written with
`hp.tb.write_table` from a NumPy `X` was stored in `(1000, 150)` chunks. Lazy
reads kept that layout, 4 × 4 blocks. `normalize_total`, `log1p`,
`highly_variable_genes` and `scale` worked. `pp.pca` did not:

- with the default solver it requires dask-ml;
- with `svd_solver="covariance_eigh"` it raised
  `Only dask arrays with chunking along the first axis are supported. Got
chunksize (1000, 150)`;
- after `X.rechunk((1000, -1))`, `covariance_eigh` worked without dask-ml.

## rapids-singlecell (source review)

rapids-singlecell supports Dask-backed `X` for most preprocessing steps and some
tools. Its documented large-data workflow is close to what Harpy's lazy reads
produce.

**Input requirements (source).** Every Dask-aware function checks its input with
`_check_gpu_X(X, allow_dask=True)`
([source](https://github.com/scverse/rapids_singlecell/blob/ed56fe5/src/rapids_singlecell/preprocessing/_utils.py#L285-L323)).
The check requires:

- `X.numblocks[1] == 1`, i.e. a single chunk along the feature axis;
- blocks that are `cupy.ndarray` or `cupyx.scipy.sparse.csr_matrix`. CSC blocks
  and CPU blocks are rejected.

`rsc.get.anndata_to_GPU` converts `csr_matrix`, `csr_array` or NumPy blocks to
the GPU lazily with `map_blocks`
([source](https://github.com/scverse/rapids_singlecell/blob/ed56fe5/src/rapids_singlecell/get/_anndata.py#L95-L102)).
`rsc.get.anndata_to_CPU` reverses this. Each block must hold fewer than 2³¹
non-zero values, because CuPy sparse matrices use 32-bit indices.

**Function support (source; see the
[out-of-core documentation](https://rapids-singlecell.scverse.org/en/latest/out_of_core.html)).**

| Supports Dask `X`                                                              | Notes                                                                                                                              |
| ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| `pp.calculate_qc_metrics`, `pp.filter_cells`, `pp.filter_genes`                | Compute immediately.                                                                                                               |
| `pp.normalize_total`                                                           | Stays lazy only with an explicit `target_sum`.                                                                                     |
| `pp.log1p`                                                                     | Lazy.                                                                                                                              |
| `pp.highly_variable_genes`                                                     | All flavors except `pearson_residuals`. Computes immediately.                                                                      |
| `pp.scale`                                                                     | `zero_center=True` makes sparse blocks dense; the tutorials use `zero_center=False`.                                               |
| `pp.regress_out`                                                               | Makes sparse blocks dense.                                                                                                         |
| `pp.pca`                                                                       | `covariance_eigh` only; builds an n_vars × n_vars matrix, so select highly variable genes first. `obsm["X_pca"]` is returned lazy. |
| `tl.score_genes`, `tl.rank_genes_groups`, `get.aggregate`, decoupler functions | `rank_genes_groups` does not support exact `wilcoxon`.                                                                             |

`pp.neighbors`, `tl.umap`, `pp.harmony_integrate`, `pp.scrublet` and the other
graph and embedding steps require in-memory GPU data. `tl.leiden` and
`tl.louvain` work on the in-memory graph.

**Loading pattern in the tutorials (source).** The multi-GPU tutorial
([07_multi_gpu.ipynb](https://github.com/scverse/rapids-singlecell-tutorials/blob/d24597e/07_multi_gpu.ipynb))
builds its AnnData from
`anndata.experimental.read_elem_lazy(f["X"], (50_000, n_vars))` and eagerly read
`obs`/`var`, then calls `rsc.get.anndata_to_GPU`. For a SpatialData store,
`hp.tb.read_table(store, table_name=..., mode="lazy", sparse_chunk_size=50_000)`
replaces that construction. The tutorials use 20,000 to 50,000 rows per chunk.
Neither tutorial writes results back to storage.

**Cluster (source).** Multi-GPU runs use `dask_cuda.LocalCUDACluster` with a
`distributed.Client`, one worker per GPU. The tests run most Dask functions with
`scheduler="synchronous"` and no client. A client is required only for the
`cuml.dask` and `cugraph.dask` paths: dense PCA with `full`/`jacobi`, `logreg`,
and `use_dask=True` clustering.

**Versions (source).** rapids-singlecell requires `anndata>=0.12.14`,
`scanpy>=1.10.0`, Python ≥ 3.12 and RAPIDS ≥ 25.12. Harpy's `anndata>=0.12.10`
floor allows 0.12.14, but Harpy's table I/O has only been tested with 0.12.10.

**Writing results back (inferred).** Call `rsc.get.anndata_to_CPU` and make sure
every value passed to `write_table_components` is CPU-backed. AnnData can write
in-memory CuPy arrays, but whether Harpy's writer handles Dask arrays with CuPy
blocks is untested.

**In-place block updates (inferred).** The `normalize_total` and sparse `scale`
kernels overwrite each block's buffers. That is harmless when blocks are re-read
from Zarr on every compute. It is not harmless when blocks are persisted and
shared, for example after `adata.layers["counts"] = adata.X`: the stored counts
could then be altered.

## Harpy-side gaps and proposed changes

### Gap 1: lazy chunk layouts that do not suit downstream tools

scanpy's lazy code paths and rapids-singlecell expect row-major layouts: Dask
arrays chunked by rows only, with dense or CSR blocks, of a size that keeps tasks
efficient. Three things get in the way:

- dense matrices whose stored chunks split the feature axis;
- CSC matrices, which lazy reads chunk by genes;
- a fixed default of 1000 rows per sparse block, far fewer than most spatial
  tables need.

#### Dense matrices

**Requirement.** Every lazy matrix that scanpy or rapids-singlecell processes must
be chunked by rows only: each Dask block spans all columns, so
`X.chunks == ((rows, rows, ...), (n_vars,))` and `X.numblocks[1] == 1`. scanpy's
PCA and QC metrics and rapids-singlecell's `_check_gpu_X` check this. The
requirement applies at two levels with different strictness:

1. **The lazy Dask array returned by `read_table` (required).** This is what the
   downstream tools receive. Harpy can satisfy it whatever the on-disk layout, by
   reading whole stored chunks into row bands; see "Aligned reading, not a
   rechunk" below.
2. **The on-disk Zarr chunks (recommended, and the primary fix).** If dense
   matrices are stored with row-only chunks, lazy reads use the stored chunks
   unchanged. Compatibility does not require this, but it is the cleanest layout,
   so the proposed change starts here.

**Where dense matrices occur in Harpy.** Sparse transcriptomics tables (CSR) are
not affected. Dense matrices come from:

- image intensity tables built by `hp.tb.aggregate_image`, whose `X` is created
  from a dense array (`src/harpy/table/_allocation_intensity.py`): the
  proteomics and imaging case;
- dense `obsm` entries, such as `X_pca` and feature matrices added with
  `hp.tb.add_feature_matrix`;
- layers that become dense, for example after `scale(zero_center=True)`.

**Evidence (verified).** `hp.tb.write_table` stored a dense 4000 × 600 `X` in
`(1000, 150)` chunks. Harpy passes no chunk arguments, so these are the default
chunks chosen when AnnData writes the NumPy array. `_decode_anndata_element`
reads dense arrays with `read_elem_lazy(element)` and keeps those chunks
(`src/harpy/_storage/_anndata.py`, end of `_decode_anndata_element`).

**Evidence on writes.**

- AnnData ignores Dask chunks when it writes dense arrays (verified). A 40,000 ×
  600 NumPy array and the same shape as a Dask array chunked `(10000, 600)` were
  both stored in `(5000, 75)` chunks. Only an explicit
  `dataset_kwargs={"chunks": (10000, 600)}` gave row-only storage. Every dense
  result Harpy writes today is therefore split by columns on disk, including
  `X_pca` and scaled layers written back with `write_table_components`.
- AnnData passes the same `dataset_kwargs` to every array of an element, for
  example to each `obs` column (source, `write_dataframe` in anndata
  `_io/specs/methods.py`). Whole-table writes therefore cannot use one chunk
  tuple for their dense matrices.

**Impact.**

- scanpy's PCA rejects column-split chunks (verified).
- scanpy's `calculate_qc_metrics` fails with an `IndexError` (verified), and its
  per-row "top genes" metrics can return the wrong shape (source, scanpy
  `_utils/__init__.py` and `preprocessing/_qc.py`).
- rapids-singlecell's `_check_gpu_X` rejects any array with more than one
  feature chunk (source).

**Proposed change.** Fix the layout where it is created, at write time, and make
lazy reads align with whatever is already stored.

1. **Writes (primary fix):** store dense matrices with row-only chunks, with the
   number of rows taken from the memory target (see "Choosing the number of rows
   per chunk" below). Lazy reads of these stores use the stored chunks unchanged,
   so each Dask block is exactly one stored chunk.
2. **Lazy reads (for stores already split by columns):**
   `read_table(..., mode="lazy")` and `read_table_components(..., mode="lazy")`
   return dense matrices chunked as `(rows, n_columns)` by default, i.e. whole
   rows, without callers having to rechunk. This covers tables that Harpy has
   written so far and stores written by other tools.
   - Build the chunks into the read: pass them to
     `read_elem_lazy(element, chunks=(rows, n_columns))`.
     `_decode_anndata_element` already does this for sparse matrices
     (`src/harpy/_storage/_anndata.py`); dense arrays currently fall through to
     `read_elem_lazy(element)`, which keeps the stored chunks.
   - Verified with anndata 0.12.10: on storage chunked `(1000, 150)`,
     `read_elem_lazy(element, chunks=(1000, 600))` returned `(1000, 600)` blocks
     with the same values.

**Aligned reading, not a rechunk.** Dask's guidance is to avoid `rechunk`
operations that move data between many chunks, and to align Dask chunks with
stored chunks: one stored chunk or a whole multiple of them, never a fraction.
Change 2 follows both:

- With `rows` a multiple of the stored row chunk size, each Dask block is a whole
  number of complete stored chunks. For `(1000, 150)` storage, one block is a
  1000-row band made of four stored chunks. The bytes read are the same as with
  the stored chunks, and nothing moves between rows.
- Built into the read, it is not a `rechunk` in the graph. `read_elem_lazy`
  creates dense arrays with `da.from_zarr(elem, chunks=chunks)` (anndata
  `_io/specs/lazy_methods.py`), so each task reads its stored chunks directly.
  Verified on the 4000 × 600 store:

| How                                            | Graph layers       | Tasks |
| ---------------------------------------------- | ------------------ | ----- |
| `read_elem_lazy(element, chunks=(1000, 600))`  | read only          | 5     |
| `read_elem_lazy(element).rechunk((1000, 600))` | read and `rechunk` | 21    |

The remaining cost is that each task holds a full row band rather than one
stored chunk. That is why the number of rows follows a memory target.

**Scope of the default.**

- Dense matrices with `mode="lazy"`: row-only chunks by default.
- CSR matrices: already row-only; unchanged.
- CSC matrices: not converted by default, because the conversion is expensive;
  see "CSC matrices" below.
- `mode="backed"` and `mode="eager"`: unaffected, because they return Zarr arrays
  or in-memory matrices rather than Dask chunks.

**API decisions.**

1. **Rows per block.** `sparse_chunk_size` applies only to sparse matrices, and a
   sparse block's size depends on its non-zero values rather than its columns, so
   one shared row count fits poorly. Dense matrices need their own setting, with a
   default derived from a memory target (see below) rather than a fixed number.
2. **Opting out.** Workflows that read one column at a time need a way to keep the
   stored chunks (see "Trade-off" below).
3. **No general `chunks=` parameter.** `read_table` returns many matrices with
   different shapes, formats and axes: `X`, layers, `obsm` entries with different
   column counts, sparse and dense. One chunk tuple cannot fit them all. Both
   settings below therefore specify only the number of rows; the other axis is
   always whole. Callers who need a different layout can rechunk the result.

Proposed settings:

- `sparse_chunk_size` (exists): rows per CSR block.
- `dense_chunks` (new), one of:
  - `"auto"` (default): keep the stored chunks when they are already row-only;
    otherwise read whole stored chunks into row bands, using the multiple of the
    stored row chunk size closest to the memory target;
  - an integer: that number of rows per block, rounded to a multiple of the
    stored row chunk size;
  - `"storage"`: keep the stored chunks as they are.

**Choosing the number of rows per chunk.** Each block holds
rows × columns × bytes per value, so the row count should follow from a memory
target rather than a fixed number. The target is Dask's `array.chunk-size`
setting, 128 MiB by default; see "Memory target" under "Sparse block size"
below. With the default:

| Matrix                | Bytes per row | 1000 rows | Rows for about 128 MiB |
| --------------------- | ------------- | --------- | ---------------------- |
| 40 channels, float32  | 160 B         | 160 KB    | about 840,000          |
| 20,000 genes, float32 | 80 KB         | 80 MB     | about 1,700            |

A fixed 1000 rows is far too small for intensity tables with tens of channels,
and roughly right for wide gene matrices. Writes use the result directly as the
stored row chunk size. Lazy reads round it to a multiple of the stored row chunk
size, so that they never split a stored chunk.

Decide whether this applies to every dense matrix or only to observation-aligned
ones (`X`, `layers`, `obsm`). Row-only chunking is also valid for `varm`, `varp`
and `raw`.

**Trade-off.** With row-only chunks, reading a single column, such as one
channel for plotting, touches every stored chunk. Whole-matrix analyses such as
scanpy's need row-only chunks, so they should be the default. If fast
per-channel access matters for a workflow, that argues for an additional
column-oriented copy for that workflow, not for a different default.

**Tests.**

- A dense table written and lazily read by Harpy has `numblocks[1] == 1`, and
  its stored chunks are row-only.
- A store written with column-split chunks by another tool reads with whole
  rows and the same values.
- The number of rows per lazy block follows the memory target and is a multiple
  of the on-disk row chunk size.
- With `dense_chunks="auto"`, row-only stored chunks are kept unchanged, and
  column-split stored chunks are read into row bands without a `rechunk` layer in
  the graph.
- An explicit row count is respected, and the storage opt-out keeps the stored
  chunks.
- `backed` and `eager` reads return the same types as before.
- `sc.pp.pca(adata, svd_solver="covariance_eigh")` works on the result.

#### CSC matrices

**Which formats occur.** Spatial transcriptomics tables are normally CSR, with
cells or bins as rows:

- scanpy's `read_10x_h5`, used for Xenium and Visium, returns `csr_matrix`;
- the spatialdata-io CosMx and Stereo-seq readers build CSR;
- Harpy's MERSCOPE reader and `hp.tb.aggregate_points` produce CSR.

The exception is Harpy's own Visium and Visium HD readers. They call
`adata.X = adata.X.tocsc()` (`src/harpy/io/_visium.py` and
`src/harpy/io/_visium_hd.py`). No comment explains the conversion, and no test
relies on it. A likely reason is faster access to individual genes, for example
for plotting, because CSC stores each gene's values contiguously.

**Evidence (verified).** A lazy read splits a CSC matrix along its compressed
axis, the genes, into `sparse_chunk_size` columns (default 1000), and keeps all
cells in every block. Tested with 4000 cells, before and after a lazy conversion
to row-chunked CSR:

| Genes | Lazy layout as read                                   | As read                                                                                                          | After conversion              |
| ----- | ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ----------------------------- |
| 600   | one `(4000, 600)` block: the whole matrix in one task | QC, `normalize_total`, `log1p` and HVG ran; PCA refused                                                          | all steps, including PCA, ran |
| 2500  | three `(4000, 1000)` blocks                           | `calculate_qc_metrics` failed with `Length of values (12000) does not match length of index (4000)`; PCA refused | all steps, including PCA, ran |

PCA refused with `Only sparse dask arrays with CSR-meta format are supported. Got
csc as meta`. rapids-singlecell's input check requires CSR blocks and one chunk
along the genes, so it would reject both layouts (source). With 2500 genes,
`normalize_total`, `log1p` and HVG ran without errors, but their results were not
checked.

**Impact.**

- Visium and Visium HD tables written by Harpy cannot use the lazy scanpy or
  rapids-singlecell paths as they are.
- Every block holds all cells, so memory per task grows with the number of cells
  and bins. For Visium HD, with millions of bins, this undoes the purpose of
  chunking.

**Lazy conversion is possible but expensive.**
`X.rechunk((rows, -1)).map_blocks(sparse.csr_matrix, meta=...)` produces
row-chunked CSR, and the pipeline then works (verified). However:

- every row block of the result depends on every gene block of the input, so
  computing any part of the result reads the whole matrix;
- each input task holds all cells for `sparse_chunk_size` genes;
- a lazy pipeline repeats this for every compute, unless the converted matrix is
  stored or persisted.

**Proposed change.**

1. Store Visium and Visium HD tables as CSR: remove the `.tocsc()` conversion
   from both readers. First check whether any workflow depends on fast per-gene
   access; if so, convert only for that access instead of for storage.
2. Convert existing CSC tables once, rather than on every read. Provide a helper
   or a documented recipe: read `X` lazily, convert it as above, and replace it
   through `write_table_components`, passing `{("X",): converted}` with
   `obs_identity`, `var_names` and `overwrite=True`. The same applies to CSC
   layers.
3. Do not convert silently in `read_table`, because of the cost. If reads should
   support it, add an explicit opt-in argument whose docstring states the cost.

**Tests.**

- Tables written by the Visium and Visium HD readers have CSR `X`.
- A converted table passes rapids-singlecell's layout check (`numblocks[1] == 1`
  and CSR blocks), runs `sc.pp.pca`, and has the same values as before
  conversion.

#### Sparse block size

**Current behavior.** `sparse_chunk_size` (default 1000) sets the rows per CSR
block, or the columns per CSC block. However, the memory a sparse block needs
depends on its non-zero values, not on its rows: about 8 bytes per non-zero value
in memory (a float32 value and an int32 column index), plus the row pointers.

**Evidence (verified).** A 200,000 × 2000 CSR matrix with 8 million non-zero
values (40 per row), written by AnnData:

- `data` and `indices` are stored as 1-D arrays in chunks of 250,000 non-zero
  values, and `indptr` in chunks of 50,001 row pointers. Stored chunks therefore
  follow non-zero values, not rows.
- The total number of non-zero values is the length of `data`, which is
  available from metadata alone.
- Reading the matrix lazily, through a store that counts fetches of the 32
  stored `data` chunks:

| Rows per block | Blocks | Non-zero values per block | Stored chunk fetches | Fetches per stored chunk |
| -------------- | ------ | ------------------------- | -------------------- | ------------------------ |
| 1000           | 200    | about 40,000              | 231                  | 7.2                      |
| 50,000         | 4      | about 2,000,000           | 35                   | 1.1                      |

Each fetch reads and decompresses a whole stored chunk, so small blocks repeat
that work.

**Impact.**

- Many small tasks. An 11-million-bin Visium HD table gets 11,000 blocks, each of
  about 40 KB if bins hold about 5 non-zero values (an assumption, not measured).
  The rapids-singlecell tutorials use 20,000 to 50,000 rows per block.
- Repeated decompression whenever a block holds fewer non-zero values than a
  stored chunk, as measured above.

**Proposed change.** Replace `sparse_chunk_size` with `sparse_chunks`, either
`"auto"` (the default) or an integer.

- `"auto"`: rows per block = memory target ÷ (average non-zero values per row ×
  bytes per non-zero value). The average comes from metadata
  (`len(data) / n_rows`), and the bytes per non-zero value from the stored dtypes
  of `data` and `indices`. The memory target is Dask's `array.chunk-size`
  setting, the same as for dense matrices (see "Memory target" below). With the
  default of 128 MiB and 8 bytes per non-zero value:

  | Average non-zero values per row | Rows for about 128 MiB | Rough example        |
  | ------------------------------- | ---------------------- | -------------------- |
  | 5                               | about 3.4 million      | Visium HD 2 µm bins  |
  | 40                              | about 420,000          | small targeted panel |
  | 500                             | about 34,000           | large panel          |
  | 3000                            | about 5,600            | single-cell RNA      |

  The examples are rough illustrations, not measurements.

- An integer keeps today's meaning: rows per CSR block, or columns per CSC block.
- **No `"storage"` mode.** Sparse matrices have no 2-D chunk grid: the stored
  `data` and `indices` chunks are counted in non-zero values, independent of
  row boundaries, so there is no stored layout to keep. The alignment rule
  instead is to make blocks much larger than one stored chunk, so that each
  stored chunk is fetched by at most two tasks. The `"auto"` sizes do this: in
  the measurement above, blocks of about 2 million non-zero values fetched each
  250,000-value chunk 1.1 times on average.
- **No cap for densification.** `"auto"` sizes blocks for the data as stored and
  does not anticipate what happens downstream. Steps that make blocks dense split
  the rows first; see "Densifying steps" below.
- **Upper bound.** rapids-singlecell limits GPU blocks to 2³¹ − 1 non-zero
  values. A 128 MiB target, about 16.8 million non-zero values, is far below
  that.
- **Renaming is free.** `read_table` and `read_table_components` are not yet
  released: they are not on `main` or in the latest tag, v0.4.4.
- **Same default internally.** Harpy uses a fixed 1000 elsewhere too:
  `_DEFAULT_SPARSE_CHUNK_SIZE` when storage-backed sparse matrices are wrapped
  lazily for writing (`_prepare_anndata_value` in
  `src/harpy/_storage/_anndata.py`), and the `chunk_size` default of
  `write_table_components_by_region`. Both should use the same `"auto"` logic.

**What others do.** No library in the ecosystem builds a densification limit into
its reader:

| Who                                                                                                | Sparse block size                                                                                                       | Densification                                                                                                                |
| -------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| AnnData `read_elem_lazy`                                                                           | fixed 1000 rows for CSR (`_DEFAULT_STRIDE`); dense keeps the stored chunks                                              | not handled                                                                                                                  |
| [scanpy's Dask tutorial](https://scanpy.readthedocs.io/en/stable/tutorials/experimental/dask.html) | 100,000 rows, for 1.46 million cells × 27,714 genes                                                                     | never densifies; advises that users "will likely have to work with smaller chunks … via a rechunking operation" when they do |
| [rapids-singlecell](https://github.com/scverse/rapids_singlecell/blob/ed56fe5/docs/out_of_core.md) | 20,000-row example; chunks "large enough to amortize scheduling but small enough to fit per-worker VRAM"                | tutorials use `scale(zero_center=False)` to avoid it; on out-of-memory errors, "reduce chunk size"                           |
| [Dask](https://docs.dask.org/en/stable/array-chunks.html)                                          | `"auto"` targets `array.chunk-size` from shape × dtype; recommends chunks of 10 MB to 1 GB and tasks longer than 100 ms | does not know about sparsity, so `"auto"` sizes sparse arrays as if they were dense                                          |

The common pattern is to read in large blocks, and to let the step that densifies
split the rows first.

**Memory target: Dask's `array.chunk-size`.** Dask has a global setting,
`array.chunk-size` (default 128 MiB), which it uses whenever chunk sizes are left
to `"auto"`, for example in `X.rechunk({0: "auto"})`. Harpy's
`sparse_chunks="auto"` and `dense_chunks="auto"` take their memory target from
the same setting, instead of a value hard-coded in Harpy. Verified for a
2000-gene float32 matrix:

| How the setting is changed                             | `array.chunk-size` | Rows chosen by Dask's `"auto"` |
| ------------------------------------------------------ | ------------------ | ------------------------------ |
| default                                                | 128 MiB            | 16,777                         |
| `with dask.config.set({"array.chunk-size": "32MiB"}):` | 32 MiB             | 4,194                          |
| environment variable `DASK_ARRAY__CHUNK_SIZE=64MiB`    | 64 MiB             | 8,388                          |

- **One setting for everything.** Changing the value once, for example on a small
  machine or GPU, changes Harpy's lazy reads (sparse and dense) and the split
  before densifying together. The read blocks and the densified blocks cannot
  drift apart because two libraries use different targets.
- **No new Harpy parameter.** A `target_bytes=` argument is not needed: Dask's
  context manager, environment variable and config files already cover this. An
  integer `sparse_chunks=` or `dense_chunks=` still sets an exact number of rows.
- **Read when the lazy array is built.** `read_table` reads the setting when it
  builds the lazy array, not when it computes. Change the setting before the
  call, or wrap the call in `dask.config.set`.
- **One byte budget for sparse and dense.** `"auto"` converts the same target
  into rows differently: from non-zero values for sparse blocks, and from
  columns × dtype for dense blocks.

**Densifying steps.** Blocks sized by non-zero values only stay the right size
while the data is sparse. Steps such as `scale(zero_center=True)`, the default
of `sc.pp.scale`, make blocks dense. The reader should not cap blocks for this.
Instead, the densifying step splits the rows first:

```python
adata.X = adata.X.rechunk({0: "auto"})  # before sc.pp.scale(adata), for example
```

Evidence (verified, 100,000 × 2000 CSR matrix with 4 million non-zero values,
763 MiB when dense):

- Dask's `"auto"` sizes the sparse-backed array as if it were dense: 16,777 rows,
  exactly 128 MiB ÷ (2000 genes × 4 bytes). The split therefore gives dense-safe
  blocks without any Harpy code.
- The split is cheap: splitting 50,000-row blocks into 10,000-row blocks gives
  blocks that each depend on exactly one original block, with no shuffle.
- It is needed. `scale(zero_center=True)` followed by a reduction used about 4×
  the dense block size in temporary memory:

| Rows per block      | Peak traced memory |
| ------------------- | ------------------ |
| 100,000 (one block) | 3,083 MiB          |
| 10,000              | 336 MiB            |

Two consequences:

- **The new default moves risk onto densifying pipelines.** With today's fixed
  1000 rows, `sc.pp.scale()` densifies each block to about 8 MB. With blocks sized
  by non-zero values, a 420,000-row block over 2000 highly variable genes becomes
  3.4 GB dense, plus about 4× that in temporary memory per thread. Harpy's own
  `Preprocess.preprocess` calls `scale(zero_center=True)`. Dropping the cap is
  right only if the split happens somewhere Harpy controls: in Harpy's own
  wrappers (gap 5) and in the user guide (Phase 2).
- **More blocks after densifying are correct, not a problem.** Once data is dense,
  its correct block size is in dense bytes. 1 million cells × 2000 genes in
  float32 is 7.5 GiB, about 60 blocks of 128 MiB, which is Dask's recommended
  size. The problem with today's default is the opposite: thousands of tiny
  sparse blocks, such as 11,000 blocks of about 40 KB for Visium HD.

For rapids-singlecell, GPU memory is the binding limit: pass an integer, or lower
`array.chunk-size`.

**Tests.**

- `"auto"` derives the rows from metadata, without reading `data` or `indices`.
- The rows follow the memory target for different densities, and change with
  `array.chunk-size` when it is set before the read.
- An integer keeps today's meaning for CSR and CSC.
- Lazy values equal the stored matrix for both settings.

### Gap 2: writing several lazy results recomputes their shared graph

**Evidence (verified).** After `normalize_total`, `log1p` and PCA on a lazily
read CSR `X` with 4 row blocks, one `write_table_components` call wrote
`("layers", "log1p")` and `("obsm", "X_pca")`. The call ran 5 Dask computes and
read source `X` blocks 28 times, where 4 is the minimum. How the reads divide
between the sparse and dense writers was not analysed. Nothing is cached between
separate computes, so each component re-runs the read → normalize → log1p chain.

**Impact.** On large tables, rereading and recomputing the source dominates the
write time. The cost grows with the number of lazy components per call and with
the length of the upstream graph.

**Options.**

1. Evaluate all lazy components of one call in a single pass: build all store
   operations without computing, then run one `dask.compute`. This may require
   Harpy to own the chunked writing of dense and sparse matrices instead of
   delegating each element to AnnData's writer.
2. Document a checkpoint pattern: write an intermediate layer, reopen the table,
   and continue from the stored layer. Alternatively, `persist()` shared
   intermediates when they fit in memory.

Option 2 is documentation only and should come first. Before choosing option 1,
measure the effect on a realistic table.

### Gap 3: no helper to persist scanpy's outputs

**Evidence.** Persisting a scanpy run currently requires the caller to:

- know where each scanpy function stores its outputs, e.g. `obsm["X_pca"]`,
  `varm["PCs"]` and `uns["pca"]` for PCA, and `obsp["connectivities"]`,
  `obsp["distances"]` and `uns["neighbors"]` for neighbors;
- list those paths explicitly;
- supply `obs`/`var` dataframes or `obs_identity`/`var_names`.

**Proposal.** A helper that compares a lazily read table with its stored
version and writes only new or changed components. A possible signature:

```python
hp.tb.write_table_updates(store, table_name="counts", adata=adata, overwrite=True)
```

Design notes:

- Lazy matrices that are still the arrays created by `read_table` can be
  recognised as unchanged by their Dask array name. Other matrices are treated
  as new or changed.
- `obs` and `var` are small enough to compare directly with the stored
  dataframes. `uns` entries can be compared by key and value.
- If `obs_names` or `var_names` differ from storage, raise and point to
  `write_table`, rather than silently rewriting the whole table.
- Components missing from `adata` should not be deleted implicitly. Deletion
  stays explicit through `delete_table_components`.

### Gap 4: stale lazy tables after an overwrite

**Evidence (verified).** After `write_table(..., overwrite=True)` replaced a
table, computing the lazy `X` of the previously read AnnData failed with a Zarr
error: `cannot reshape array of size 2002 into shape (4001,)`. The docstring of
`read_table` warns that such tables "may read replacement data with stale
annotations or fail". If the replacement had had compatible shapes, the stale
table could have read the wrong data without an error.

**Options.**

- Have the write functions return the reopened lazy table, so the natural
  pattern is `adata = hp.tb.write_table(...)`.
- Record a generation token in the table's attributes on every publication, and
  have lazy reads check it before reading blocks, raising a clear error when the
  table has been replaced.

### Gap 5: Harpy's own scanpy wrappers load tables into memory

Harpy's table processing functions do not benefit from the lazy I/O yet:

- `Preprocess.preprocess` (`src/harpy/table/_preprocess.py`) and the
  `leiden` and `score_genes` wrappers start from `sdata.tables`, which
  `sd.read_zarr` has fully loaded.
- `ProcessTable._get_adata` (`src/harpy/table/_table.py`) copies the table.
- The preprocessing code uses operations that only work on in-memory matrices,
  such as `issparse`, `.toarray()`, `np.where` and `np.nanquantile`.
- Results are written through `add_table`.

Store-path variants (`store`, `table_name`) that read with
`read_table(mode="lazy")` and write with `write_table_components` or
`write_table` would let these functions scale. They depend on gap 1 and benefit
from gaps 2 and 3.

These variants must split rows before densifying steps. `Preprocess.preprocess`
calls `sc.pp.scale(..., zero_center=True)`, which makes blocks dense; with blocks
sized by non-zero values, the variant should first call
`adata.X = adata.X.rechunk({0: "auto"})` (see "Densifying steps" in gap 1).

## Upstream limitations to document

These are not Harpy changes, but users of the lazy path will hit them.

**scanpy 1.11.1 (verified unless marked source).**

- `highly_variable_genes(flavor="seurat_v3")` does not work on Dask input. On
  dense Dask arrays its clipping step is silently skipped, which changed 8% of
  the gene flags on test data with outliers. On sparse Dask arrays it raises
  `TypeError` (`np.putmask`, `preprocessing/_highly_variable_genes.py`). Use
  `seurat` or `cell_ranger`.
- Sparse `scale(zero_center=True)` turns blocks into dense `np.matrix` in scanpy
  1.11.1, after which PCA fails. This is fixed in scanpy 1.11.2
  ([PR #3597](https://github.com/scverse/scanpy/pull/3597), source; 1.11.2 was
  not tested here). On 1.11.1 and earlier, use `zero_center=False`, or convert the
  blocks with `X.map_blocks(np.asarray, meta=np.array([], dtype=X.dtype))`; PCA
  then works. On any version, zero-centering makes the blocks dense, so split the
  rows first (see "Densifying steps" in gap 1).
- Dense PCA needs dask-ml unless `svd_solver="covariance_eigh"`. Sparse PCA
  always uses `covariance_eigh` and builds an n_vars × n_vars matrix, so select
  highly variable genes first.
- `regress_out`, `rank_genes_groups(method="wilcoxon")` and `get.aggregate` do
  not support Dask input. `rank_genes_groups(method="logreg")` fails on sparse
  Dask input.
- Sparse blocks must be `csr_matrix`. With `csr_array` blocks, every tested step
  except `log1p` failed, including `calculate_qc_metrics`, `normalize_total`,
  `highly_variable_genes` and `pca`. Harpy should keep producing `csr_matrix`
  blocks.
- Two helpers merge the whole matrix into one task (source,
  `_utils/__init__.py`):
  - the nonzero counter behind `n_genes_by_counts`/`n_cells_by_counts`;
  - `check_nonnegative_integers`, which is used by `seurat_v3` and by
    `rank_genes_groups`.

  On large tables these steps load the full matrix in one task.

- Every assignment of a lazy result to `obs` or `var` is a separate compute that
  re-runs the whole graph from storage. Call `persist()` or `compute()` on
  shared intermediates, such as `obsm["X_pca"]`, before running several
  downstream steps.

**Numba and Dask threads (verified on the test machine).**
`sc.pp.scale` on a dense Dask matrix, followed by a compute under Dask's default
threaded scheduler, crashed the Python process with
`Numba workqueue threading layer is terminating: Concurrent access has been
detected` (3 out of 3 runs). `dask.config.set(scheduler="synchronous")` avoided
it, and `NUMBA_NUM_THREADS=1` did not. The cause is Numba's fallback threading
layer, which is used when neither TBB nor OpenMP is available. Installing `tbb`
would likely fix it (inferred). Using `dask.distributed` with one thread per
worker probably also avoids it (inferred).

**rapids-singlecell (source).** See the section above. The main points:

- convert to the GPU with `anndata_to_GPU`;
- one chunk along the feature axis;
- no Dask support for `neighbors` and later steps, so compute `X_pca` first;
- `anndata>=0.12.14`.

## Recommended usage with the current API

scanpy (verified pattern):

```python
import harpy as hp
import scanpy as sc

store, table_name = "sdata.zarr", "counts"
adata = hp.tb.read_table(store, table_name=table_name, mode="lazy")

sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)  # "seurat", not "seurat_v3"
sc.pp.pca(adata, n_comps=50)  # uses var["highly_variable"] when present
adata.obsm["X_pca"] = adata.obsm["X_pca"].compute()
sc.pp.neighbors(adata)
sc.tl.leiden(adata, flavor="igraph", n_iterations=2)

hp.tb.write_table_components(
    store,
    table_name=table_name,
    components={
        ("obs",): adata.obs,
        ("var",): adata.var,
        ("layers", "log1p"): adata.X,
        ("obsm", "X_pca"): adata.obsm["X_pca"],
        ("varm", "PCs"): adata.varm["PCs"],
        ("obsp", "connectivities"): adata.obsp["connectivities"],
        ("obsp", "distances"): adata.obsp["distances"],
        ("uns", "log1p"): adata.uns["log1p"],
        ("uns", "hvg"): adata.uns["hvg"],
        ("uns", "pca"): adata.uns["pca"],
        ("uns", "neighbors"): adata.uns["neighbors"],
        ("uns", "leiden"): adata.uns["leiden"],
    },
    overwrite=True,
)
# Reopen instead of reusing lazy data from before the write.
adata = hp.tb.read_table(store, table_name=table_name, mode="lazy")
```

The supplied `obs` and `var` dataframes provide the observation and feature
identities that the matrix components are checked against. If cells or genes
were filtered, use `hp.tb.write_table(..., overwrite=True)` instead.

Until gap 1 is fixed:

- for dense tables, call `adata.X = adata.X.rechunk((chunk_rows, -1))` after
  reading and use `svd_solver="covariance_eigh"`;
- for CSC tables, such as those written by Harpy's Visium readers, convert `X` to
  row-chunked CSR (see gap 1), preferably once, storing the result;
- on machines without TBB, run dense scaling with
  `dask.config.set(scheduler="synchronous")`.

Independently of gap 1, split the rows before any step that makes the matrix
dense, such as `sc.pp.scale(adata)` with its default `zero_center=True`:
`adata.X = adata.X.rechunk({0: "auto"})`. To use smaller blocks everywhere, lower
Dask's `array.chunk-size` setting.

rapids-singlecell (untested, from its documentation):

```python
import rapids_singlecell as rsc

adata = hp.tb.read_table(store, table_name=table_name, mode="lazy", sparse_chunk_size=50_000)
rsc.get.anndata_to_GPU(adata)
rsc.pp.normalize_total(adata, target_sum=1e4)
rsc.pp.log1p(adata)
rsc.pp.highly_variable_genes(adata, n_top_genes=5000, flavor="cell_ranger")
adata = adata[:, adata.var["highly_variable"]].copy()  # changes the feature axis
rsc.pp.scale(adata, zero_center=False)
rsc.pp.pca(adata, n_comps=100)
adata.obsm["X_pca"] = adata.obsm["X_pca"].compute()
rsc.pp.neighbors(adata)
rsc.tl.leiden(adata)
rsc.get.anndata_to_CPU(adata)
# Persist CPU-backed results: write_table for the subset table, or
# write_table_components on the unsubset table for obs/obsm/obsp/uns results.
```

## Implementation sequence

### Phase 1: row-major layouts for dense and sparse matrices

Implement gap 1 in three slices, in this order. Gap 1 calls row-only writes the
primary fix, because they give the cleanest stored layout. The slices still start
with the read side: it helps existing stores immediately, without rewriting any
data, and it provides the sizing logic that the write side reuses.

| Slice                | Content                                                                        | Main code                                                          | Depends on                                   |
| -------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------ | -------------------------------------------- |
| **1a: read side**    | `sparse_chunks` and `dense_chunks` on all read functions; shared sizing helper | `_storage/_anndata.py`, `table/io/_read.py`, `io/_read_zarr.py`    | nothing                                      |
| **1b: dense writes** | row-only stored chunks for dense 2-D matrices                                  | `_storage/_anndata.py` (`_write_anndata_element`)                  | 1a's sizing helper                           |
| **1c: CSC → CSR**    | Visium readers write CSR; one-time conversion of existing CSC tables           | `io/_visium.py`, `io/_visium_hd.py`, a conversion helper or recipe | the open CSC question; 1a for the lazy reads |

After 1a, existing stores work with scanpy and rapids-singlecell, except CSC
tables. After 1b, new dense tables need no merge on read. After 1c, every table
Harpy writes is stored row-major, as CSR or dense with row-only chunks.

#### Slice 1a: read side

- Replace `sparse_chunk_size` with `sparse_chunks="auto" | int`, and add
  `dense_chunks="auto" | int | "storage"`, on all three public read functions:
  `read_table`, `read_table_components` and `hp.io.read_zarr`
  (`src/harpy/io/_read_zarr.py`), which also exposes `sparse_chunk_size` today.
- Add the shared sizing helper: memory target from Dask's `array.chunk-size` →
  rows per block, from the average non-zero values per row for sparse matrices
  and from columns × dtype for dense ones, rounded to a multiple of the stored
  row chunk size.
- Use the same default for the internal uses of the fixed 1000:
  `_prepare_anndata_value`, which lazily wraps storage-backed sparse inputs
  during writes, and the `chunk_size` default of
  `write_table_components_by_region` and `add_table_components_by_region`
  (`_DEFAULT_REGIONAL_CHUNK_SIZE`). This changes a public default of those two
  write functions.
- Tests: the read-side tests listed under "Dense matrices" and "Sparse block
  size" in gap 1.
- `sparse_chunk_size` appears about 23 times in 6 test files, 15 of them in
  `test_read.py`. Renaming is free for users, because the API is unreleased, but
  those tests need updating.

#### Slice 1b: dense writes

- In `_write_anndata_element`, store dense 2-D matrices (`X`, layers, `obsm`
  entries; `varm`, `varp` and `raw` per the open question on scope) with row-only
  chunks, sized by the 1a helper.
- Implementation: AnnData's `write_dispatched`, with a callback that sets `chunks`
  only for dense 2-D matrices and leaves `obs`/`var` columns and sparse buffers to
  AnnData's defaults. `write_dispatched` is available in anndata 0.12.10; this
  approach has not been tested yet.
- Coverage: all of Harpy's own table writers go through `_write_anndata_element`:
  `write_table`, `write_table_components`, `add_table`, `add_table_components`,
  the aggregation writer behind `aggregate_points`, and canonical centers.
  `aggregate_image` writes through `add_table`. Aggregation therefore needs no
  separate work; add a test that an `aggregate_image` table is stored with
  row-only chunks.
- Not covered: tables saved through SpatialData's own writer, such as
  `sdata.write(output)` in the Visium readers. Those tables are sparse and are
  handled by 1c.
- Side effect: this also fixes the dense results Harpy writes back today, such as
  `X_pca` and scaled layers, which are currently split by columns on disk.
- Tests: the write-side tests listed under "Dense matrices" in gap 1.

#### Slice 1c: CSC → CSR

- Remove the `.tocsc()` conversion from the Visium and Visium HD readers
  (`src/harpy/io/_visium.py`, `src/harpy/io/_visium_hd.py`). They save through
  `sdata.write(output)`, so this in-memory change is all they need.
- Provide the one-time conversion path for existing CSC tables described under
  "CSC matrices" in gap 1.
- Resolve the open question first: why do the readers use CSC, and does any
  workflow depend on it?
- Tests: those listed under "CSC matrices" in gap 1.

### Phase 2: user documentation

Add a user guide page, "Using Harpy tables with scanpy and rapids-singlecell",
containing:

- the recommended usage pattern above;
- when to use `write_table_components` and when to use `write_table`;
- the reopen rule;
- splitting rows before densifying steps, and Dask's `array.chunk-size` setting
  as the single memory target;
- the upstream limitations, including the scheduler/Numba note.

Link it from the table I/O section of `docs/api.md`.

### Phase 3: write cost

Measure the number of source reads and the write time for multi-component writes
on a realistic table (for example 1M cells). Document the checkpoint pattern.
Then decide whether single-pass evaluation (gap 2, option 1) is worth owning in
Harpy.

### Phase 4: write-back helper

Implement the comparison-based helper from gap 3. Test it against the pipeline
above, including filtered tables, which must raise and point to `write_table`.

### Phase 5: stale-table protection

Choose between returning reopened tables and generation tokens (gap 4).

### Phase 6: store-path variants of Harpy's table functions

Add lazy, store-path variants of `preprocess_transcriptomics`,
`preprocess_proteomics`, `leiden` and related wrappers (gap 5). Remove or
replace their in-memory-only operations.

### Phase 7: rapids-singlecell validation on a GPU machine

Run the rapids-singlecell pattern on a CUDA machine with `anndata>=0.12.14`,
including writing results back with Harpy. Every rapids-singlecell statement in
this document is currently based on source review.

## Open questions

- **Scope of rechunking.** Should Gap 1 cover all dense matrices, or only
  observation-aligned ones?
- **CSC in the Visium readers.** Why do the Visium and Visium HD readers store
  CSC, and does any workflow depend on it? Should existing CSC tables be
  converted automatically, for example by a migration step, or only on request?
- **Version floors.** Only scanpy 1.11.1 was tested. Whether the lazy paths work
  with the declared floor `scanpy>=1.9.1` is untested. scanpy 1.11.2 fixes the
  `scale` → PCA failure on sparse Dask input, which argues for `scanpy>=1.11.2`
  on the lazy path. Harpy's I/O should also be tested with `anndata>=0.12.14`,
  which rapids-singlecell requires.
- **Scheduler guidance.** Should Harpy recommend a Dask scheduler configuration
  for downstream analysis, or only document the Numba issue?

## Reproducing the experiments

The scripts behind every verified result are in [`scripts/`](scripts/). See
[`scripts/README.md`](scripts/README.md) for which script supports which section,
how to run them, and the expected output.
