# Moment aggregation: design

Harpy's own functions switch from `RasterAggregator`
(`src/harpy/utils/_aggregate.py`) to `MomentAggregator`
(`src/harpy/experimental/_moments.py`) to compute per-instance statistics from
labels and images. This document fixes the decisions, the design of
`MomentAggregator` as it stands, and the plan for the switch.

## Purpose

`MomentAggregator` computes count, sum, mean, var, skew, kurtosis, min, max
and center of mass for every label in one pass over the image.
`RasterAggregator` makes a separate pass per group of statistics, and each of
its chunks returns a dense row for every label in the mask.

Measured on a 4-channel Xenium slide (407k cells, 1024 x 1024 chunks):

| Statistics                            | `RasterAggregator`                               | `MomentAggregator` |
| ------------------------------------- | ------------------------------------------------ | ------------------ |
| mean                                  | 60.9 s, 8.0 GB                                   | 8.4 s, 0.9 GB      |
| sum, mean, count, var, skew, kurtosis | 215.1 s, 8.3 GB                                  | 8.9 s, 1.2 GB      |
| min, max, center of mass              | 120.9 s, 7.7 GB                                  | 8.6 s, 1.0 GB      |
| all nine                              | about 333 s (estimate: the two runs above added) | 9.8 s, 1.3 GB      |

- Reading the image and the labels alone takes 6.1–6.8 s.
- The `RasterAggregator` times include about 3 s for `da.unique` on the mask,
  as `hp.tb.aggregate_image` does to build the index.
- Memory is peak resident memory of the process.

## Decisions

### Public API: keep both classes

- `hp.utils.RasterAggregator` stays public and unchanged, as documented in
  `docs/api.md`. User code that calls its per-statistic methods
  (`aggregate_mean`, `aggregate_area`, `center_of_mass`, ...) or its GPU path
  keeps working.
- `MomentAggregator` stays in `harpy.experimental`.
- Harpy's own functions use `MomentAggregator` internally.

So stable functions depend on an experimental class, and their tests pin its
behaviour. Moving `MomentAggregator` out of `experimental`, and deprecating
`RasterAggregator`, is a later decision.

### GPU: no `run_on_gpu` for aggregation

`MomentAggregator` runs on CPU only, with numba kernels. Whether the GPU path
of `RasterAggregator` is faster than the new CPU path has not been measured.

| Caller                     | `run_on_gpu` after the switch                                                                                                                                                                          |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `hp.tb.aggregate_image`    | Controlled only the aggregator. Kept, so user code does not break, but ignored: a warning is logged when it is `True`, and the docstring says so.                                                      |
| `hp.tb.add_feature_matrix` | Still controls the `area` morphology feature (`_get_mask_area`). Intensity features ignore it, and the docstring says so.                                                                              |
| `Featurizer`               | Still controls instance extraction and instance statistics. Only the center of mass in `extract_instances` moves to CPU: a CuPy mask is converted to NumPy blocks first.                               |
| Canonical centers          | Already passed `run_on_gpu=False`. No change.                                                                                                                                                          |
| `spatial_pixel_neighbors`  | Created `RasterAggregator` with its default `run_on_gpu=True` for the center of mass of the grid: on GPU when CuPy is installed, otherwise on CPU with a warning. Now always CPU, without the warning. |

### numba becomes a direct dependency

numba is now only a transitive dependency, with a minimum version set in
`[tool.uv] override-dependencies` of `pyproject.toml`. `MomentAggregator`
imports it directly, so it moves to `[project] dependencies`, with the same
minimum version (`numba>=0.64.0`). CI runs `uv sync --locked`, so `uv.lock`
has to be updated in the same change.

## `MomentAggregator` as it stands

- **One pass.** Each chunk returns, per label present in it, the count, the
  mean and the centered sums `M2`, `M3`, `M4` of every channel, plus the
  minimum, the maximum and the sum of the pixel coordinates when requested.
  Partial results are combined exactly, the moments with the binomial
  expansion of the centered sums. Every term is centered, so this is as
  numerically stable as the two-pass algorithm.
- **Sparse partial results.** Labels are grouped with a lookup table over the
  label range of a chunk, not a sort. A chunk returns only the labels present
  in it.
- **Task size.** A task reads the mask of one spatial chunk and at most
  `max_task_bytes` (default `"256MiB"`) of image data: the channel chunks of a
  spatial chunk are split into batches of consecutive chunks, and each batch
  groups the labels again. A channel chunk larger than `max_task_bytes` forms
  a batch on its own. The batches of a spatial chunk are joined right away.
- **No rechunking.** The image is never rechunked, so its chunks are best
  aligned with the chunks on disk. Larger blocks are read as multiples of the
  stored chunks with `da.from_zarr(..., chunks=...)`. Misaligned chunks keep
  stored chunks in memory until all their pieces are used: 1000 x 1000 blocks
  over 1024 x 1024 stored chunks peaked at 3.5–3.8 GB, against 0.85 GB for
  aligned blocks.
- **Compiled merge.** With `engine="numba"` (default), the per-pixel kernels
  and the merge of partial results are compiled loops that release the GIL.
  The merge adds every partial straight into preallocated output arrays.
  `engine="numpy"` is the reference implementation, with numpy expressions.
- **Output.** The same DataFrames as `RasterAggregator.aggregate_stats`, one
  per statistic, with a column per channel and the label column. Moments are
  float32; `min` and `max` keep the image dtype; `center_of_mass` has columns
  `0, 1, 2` with the mean `z, y, x` pixel coordinate in float64, as
  `RasterAggregator.center_of_mass`.

### Task size with many channels

Measured with 40 distinct channels, all nine statistics, on the full slide.
The channels are copies `image + k` of the 4 Xenium channels, so only those 4
are read from disk: the times measure computation, not reading 40 channels.

| Spatial chunks | `max_task_bytes`         | Time    | Peak memory |
| -------------- | ------------------------ | ------- | ----------- |
| 1024 x 1024    | all channels in one task | 45–49 s | 3.9 GB      |
| 2048 x 2048    | all channels in one task | 35–37 s | 5.3–5.5 GB  |
| 2048 x 2048    | 256 MiB                  | 34 s    | 4.3 GB      |
| 2048 x 2048    | 128 MiB                  | 33–34 s | 3.4 GB      |
| 4096 x 4096    | all channels in one task | 47 s    | 16–17 GB    |
| 4096 x 4096    | 512 MiB                  | 31–33 s | 9.6–9.8 GB  |
| 4096 x 4096    | 256 MiB                  | 32–33 s | 7.6 GB      |
| 4096 x 4096    | 128 MiB                  | 37–38 s | 7.0–7.2 GB  |

256 MiB is the default: it caps the memory of large spatial chunks without
the extra label grouping that smaller budgets cost. Results are identical for
every budget.

The compiled merge, tested in isolation on the final merge of 40 channels and
407k cells, takes 0.12 s and 0.61 GB, against 1.36 s and 1.59 GB with numpy.
End to end with 40 channels and 2048 x 2048 chunks, it lowered the time from
about 37 s to about 32 s and the peak memory by about 0.7 GB, with identical
results. With 4 channels it changes nothing measurable.

### numba cache

The kernels use `cache=True`: in a new process, the first call then takes
0.17–0.20 s instead of 0.80 s. numba's cache index is not safe when several
processes compile different signatures of one function at the same time: one
process can load machine code compiled for another dtype, which fails with a
`TypeError` or, when the item sizes match, returns wrong values. This happens
under pytest-xdist, not in normal use. `src/harpy/_tests/conftest.py` gives
every xdist worker its own `NUMBA_CACHE_DIR`.

## Implementation plan

1. **Optional image in `MomentAggregator`.**
   `image_dask_array: da.Array | None = None`. Without an image only `"count"`
   and `"center_of_mass"` are available. Requesting another statistic raises a
   `ValueError` that names the statistics and says to pass
   `image_dask_array`. The checks on the image's shape and chunks run only when
   an image is given.
2. **`hp.tb.aggregate_image`** (`src/harpy/table/_allocation_intensity.py`).
   - One `aggregate_stats` call for `mode`, `obs_stats`, `min`, `max`, and
     `center_of_mass` when `calculate_center_of_mass` is set. This replaces the
     separate passes for min, max and the center of mass.
   - `index=None`, so the labels come from the result, background included as
     before. This drops the `da.unique` pass over the mask.
   - The labels are still rechunked to the image's spatial chunks:
     `MomentAggregator` requires matching chunks.
   - `run_on_gpu`: warning and docstring, see [GPU](#gpu-no-run_on_gpu-for-aggregation).
     The docstring names `MomentAggregator` instead of `RasterAggregator`, and
     the outdated note on `scipy.ndimage.center_of_mass` goes.
   - Output columns, their order and their dtypes do not change.
3. **`hp.tb.add_feature_matrix`** (`_compute_intensity_feature_frame` in
   `src/harpy/table/_add_feature_matrix.py`). One `aggregate_stats` call for
   all intensity features, `min` and `max` included, with
   `index=instance_ids`. Docstring of `run_on_gpu` as above.
4. **Center-of-mass callers**, mask only:
   - canonical centers, `_calculate_centers_with_raster_aggregator` in
     `src/harpy/table/canonical_centers/_calculation.py`: rename the function
     and remove `RasterAggregator` from its error messages;
   - `_get_values_grid_most_frequent` in
     `src/harpy/table/pixel_clustering/_neighbors.py`;
   - `Featurizer.extract_instances` in `src/harpy/utils/_featurize.py`.
5. **Dependency.** numba in `[project] dependencies` and an updated `uv.lock`,
   see [numba](#numba-becomes-a-direct-dependency).
6. **Tests.** Run the modules that cover the switched code, and update stored
   reference values where they differ as described under
   [Expected differences](#expected-differences):
   - `src/harpy/_tests/test_experimental/test_moments.py`
   - `src/harpy/_tests/test_table/test_allocation_intensity.py`
   - `src/harpy/_tests/test_table/test_add_feature_matrix.py`
   - `src/harpy/_tests/test_table/test_add_feature_matrix_storage.py`
   - `src/harpy/_tests/test_table/test_canonical_centers.py`
   - `src/harpy/_tests/test_table/test_pixel_clustering/test_neighbors.py`
   - `src/harpy/_tests/test_utils/test_featurize.py`
   - callers of `aggregate_image`: `test_allocation.py`,
     `test_cluster_intensity.py`, `test_plot/test_cluster_intensity_heatmap.py`,
     `test_io/test_cosmx/test_fovs.py`

   `src/harpy/_tests/test_utils/test_aggregate.py` keeps testing
   `RasterAggregator`.

7. **Distributed.** A one-off check with a Dask `LocalCluster` with process
   workers: every worker compiles or loads the kernels, and the partial
   results must pickle. Not a permanent test, because it is slow.

## Expected differences

Compared with `RasterAggregator` on the Xenium slide:

- count, min, max and center of mass are identical;
- mean and var agree to about 2e-7 relative;
- skew and kurtosis differ by at most 4e-6 and 9e-5 absolute.

`RasterAggregator` casts the sums of every chunk to float32 before adding
them; `MomentAggregator` accumulates in float64 and casts the moments to
float32 at the end, which is more accurate. Behaviour that stays the same:

- zero variance gives NaN for skew and kurtosis, which is replaced by 0 with a
  warning;
- a label of `index` that is absent from the mask gets the largest value of
  the image dtype for `min` and the smallest for `max`.

## Out of scope

- Moving `MomentAggregator` out of `experimental`, and deprecating
  `RasterAggregator`.
- A GPU engine for `MomentAggregator`.
- The other methods of `RasterAggregator`: the quantile and radii methods are
  already commented out, and nothing in Harpy calls
  `aggregate_custom_channel`.
- Lower memory after the reduction in `aggregate_stats`: the `np.where`
  copies and the float32 conversion of the full result.
- A smaller dtype (int32 instead of intp) for the label grouping that every
  channel batch repeats.

## Open question

- Who runs `uv lock` after numba is added to the dependencies: the agent
  doing the change, or the maintainer.
