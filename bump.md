# Dependency bump: failing tests after `uv lock`

After `uv lock` and `./create_env.sh`, about 90 unit tests fail that passed with the
previous lock. Five upstream changes explain all of them:

| Package    | Old     | New    | Step    |
| ---------- | ------- | ------ | ------- |
| anndata    | 0.12.10 | 0.13.4 | 1, 2, 3 |
| matplotlib | 3.10.1  | 3.11.2 | 4       |
| numba      | 0.65.1  | 0.68.0 | 5       |

Other notable bumps that did not cause failures: scanpy 1.11.1 → 1.12.4, zarr 3.2.1 → 3.4.1,
dask 2026.7.1 → 2026.8.0, numpy 2.4.6 → 2.5.4, scipy 1.15.2 → 1.18.1, xarray 2026.2.0 → 2026.9.0.

## Decision: anndata lower bound

Harpy supports only `anndata>=0.13` (`pyproject.toml` updated). The rollback fix (step 1b) would
otherwise need one code path for 0.12 (`_X`) and one for 0.13 (`_layers`), and CI runs only `locked`
(0.13.4) and `stable` (newest), so a 0.12 path would never be tested.

`uv.lock` still records the old specifier, so `uv sync --locked` fails until `uv lock` is rerun. This
can be done together with the numba pin of step 5.

Follow-ups now that 0.12 is unsupported:

- the empty-sparse-matrix workaround in
  [_decode_anndata_element](src/harpy/_storage/_anndata.py#L281) cites an AnnData 0.12.10 bug; check
  whether 0.13 still needs it;
- `roadmap/integration_downstream/` mentions the 0.12.10 floor.

## Steps

- [ ] 1. anndata 0.13: `X` is stored as `layers[None]` (1a done; 1b and 1c open)
- [ ] 2. anndata 0.13: backed sparse datasets reject item assignment with `TypeError`
- [ ] 3. anndata 0.13: zarr's sharding warning is silenced
- [ ] 4. matplotlib 3.11: `Axes.relim()` includes collections
- [ ] 5. numba 0.68: in-place add on unsigned integer arrays fails to type

---

## 1. anndata 0.13: `X` is stored as `layers[None]`

Most of the failures.

### Cause

anndata 0.13 removed `AnnData._X`. `X` is stored in the `_layers` dict under the key `None`:

- iterating `adata.layers` yields `None` as well: `list(adata.layers) == ['counts', None]`;
- the `X` setter writes into `_layers` in place (`self.layers[None] = value`);
- assigning `adata.layers = {...}` replaces `_layers` with a new dict. Without a `None` key it keeps
  the current `X` and emits a `FutureWarning`; `{None: None, ...}` drops `X`;
- `AnnData(X=x, layers={None: y, ...})` raises unless `x is y`;
- `Raw` still has `_X`.

Harpy's table IO assumes `X` and `layers` are separate slots, which fails in three ways.

### 1a. `None` key in `layers` iterations

| Error                                                                | Site                                                                                                                                                                                                                          |
| -------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `If you provide layers[None] and X, they must be identical.`         | [_prepare_anndata_value](src/harpy/_storage/_anndata.py#L784) passes `layers` including `None` and a separately wrapped `X` to [AnnData](src/harpy/_storage/_anndata.py#L793). `map_blocks` makes them two different objects. |
| same                                                                 | [_add_table](src/harpy/table/io/_add_table.py#L188) passes `dict(adata.layers)` and `adata.X`; on views these are two different view objects.                                                                                 |
| `argument should be a str or an os.PathLike object … not 'NoneType'` | [_table_components](src/harpy/table/io/_updates.py#L520) yields `("layers", None)`, and [_exists_on_disk](src/harpy/table/io/_updates.py#L550) calls `joinpath(None)`.                                                        |

Sites without a failing test:

- [_table_matrices](src/harpy/table/_table.py#L68) yields `X` twice (`X` and `layers[None]`);
- [plot_adata](src/harpy/plot/_qc_cells.py#L35) plots `X` twice;
- [`sorted(adata.layers)`](src/harpy/qc/table/_summarize_table.py#L529) raises `TypeError` mixing `None`
  with strings, in the error message for an unknown layer.

Fix: skip the `None` key wherever harpy iterates `layers`; `X` is handled separately at each site.

Done: [_matrix_mapping](src/harpy/_storage/_anndata.py#L77) returns a matrix mapping without `X`
and is used by `_prepare_anndata_value`, `_add_table`, `_table_components` and `_table_matrices`;
`plot_adata` and `_collect_class_matrices` skip `None` inline. The step 1 modules then had 15
failures left: 10 from the `_X` rollback (1b), 3 from 1c and 2 from step 2.

### 1b. Rollback of attached tables uses `_X`

Error: `AttributeError: 'AnnData' object has no attribute '_X'`.

Sites:

- [_update_table_components snapshot](src/harpy/table/io/_components.py#L270) and
  [restore](src/harpy/table/io/_components.py#L304);
- [regional snapshot](src/harpy/table/io/_components_by_region.py#L219) and
  [restore](src/harpy/table/io/_components_by_region.py#L247), with no failing test;
- [_install_memory_updates](src/harpy/table/io/_components.py#L642) sets `table.X`, which in 0.13
  mutates the retained `_layers` dict in place.

Renaming `_X` to `_layers` is not enough: restoring the old `_layers` reference would not restore the
old `X`, because the `X` setter mutated that same dict.

Fix: retain and restore `_layers` for the `X` slot, and install an `X` update as a new `layers` mapping
(`{**layers, None: X}`, or `{None: None, ...}` to remove `X`) so `_layers` is replaced, not mutated.
When the same update also changes `layers`, merge `X` into that mapping. New `layers` mappings
should carry the `None` key to avoid the `FutureWarning`. `raw._X` at
[_components.py:607](src/harpy/table/io/_components.py#L607) stays as it is.

### 1c. Test expectation in `test_read.py`

`test_components_cover_matrix_mappings_and_raw` builds the expected `layers` from
[`dict(adata.layers)`](src/harpy/_tests/test_table/test_io/test_read.py#L354), which now includes
`None`. The stored `layers` group correctly holds only `counts`. Fix: drop the `None` key from the
expected value.

### Failing tests

- `test_table/test_io/test_add_table.py`: all failures
- `test_table/test_io/test_add_table_updates.py`: all failures
- `test_table/test_io/test_components.py`: all failures
- `test_table/test_io/test_write.py`: all failures
- `test_table/test_io/test_write_table_updates.py`: all failures, including
  `test_a_failing_write_writes_nothing` (its regex mismatch is the `NoneType` path error)
- `test_table/test_io/test_read.py::test_components_cover_matrix_mappings_and_raw`
- `test_table/test_featurize/test_featurize_sdata.py::test_featurize_sdata_blobs[True-True]`
- `test_table/test_cell_clustering/test_cell_clustering.py::test_cell_clustering`
- `test_table/test_pixel_clustering/test_cluster_intensity_SOM.py::test_cluster_intensity`
- `test_plot/test_plot_flowsom.py::test_plot_pixel_clusters`
- `test_table/test_pp/test_qc_metrics.py::test_the_columns_persist_with_add_table_updates`

### Verify

```bash
.venv/bin/pytest -q -n auto src/harpy/_tests/test_table/test_io src/harpy/_tests/test_table/test_featurize/test_featurize_sdata.py src/harpy/_tests/test_table/test_cell_clustering/test_cell_clustering.py src/harpy/_tests/test_table/test_pixel_clustering/test_cluster_intensity_SOM.py src/harpy/_tests/test_plot/test_plot_flowsom.py src/harpy/_tests/test_table/test_pp/test_qc_metrics.py
```

---

## 2. anndata 0.13: backed sparse datasets reject item assignment with `TypeError`

### Cause

`_CSRDataset` no longer defines `__setitem__`, so `matrix[0, 1] = 99` raises
`TypeError: '_CSRDataset' object does not support item assignment` instead of a `ValueError`
matching `read.only`. Writes are still rejected, just before reaching storage. Dense backed handles
are unaffected.

### Failing tests

- [test_read.py:533](src/harpy/_tests/test_table/test_io/test_read.py#L533):
  `test_backed_handles_reject_direct_writes[csr-read_table]` and `[csr-read_table_components]`
- [test_allocation.py:576](src/harpy/_tests/test_table/test_allocation.py#L576):
  `test_class_aware_aggregation_writes_one_count_matrix_per_class`

### Fix

Test only: expect `TypeError` for sparse backed handles and keep `ValueError("read.only")` for dense
ones. The assertions that the stored values are unchanged stay.

### Verify

```bash
.venv/bin/pytest -q -n auto src/harpy/_tests/test_table/test_io/test_read.py src/harpy/_tests/test_table/test_allocation.py
```

---

## 3. anndata 0.13: zarr's sharding warning is silenced

### Cause

With `ad.settings.auto_shard_zarr_v3 = True` and no explicit `shards`, anndata now filters zarr's
`Automatic shard shape inference is experimental` warning
(`anndata/_io/specs/methods.py`), so `pytest.warns` sees no warning.

### Failing test

[test_anndata.py:465](src/harpy/_tests/test_storage/test_anndata.py#L465):
`test_automatic_sharding_keeps_row_only_chunks_as_inner_chunks`

### Fix

Test only: remove the `pytest.warns` block. The chunk and shard assertions after it check the
behaviour that matters.

### Verify

```bash
.venv/bin/pytest -q src/harpy/_tests/test_storage/test_anndata.py
```

---

## 4. matplotlib 3.11: `Axes.relim()` includes collections

### Cause

For `stat="percent"`, harpy draws a count histogram and then scales its artists by `100 / N` in
[_scale_histogram_artists](src/harpy/plot/_renderers.py#L155). Its comment says `Axes.relim()` ignores
collections, which held up to matplotlib 3.10. In 3.11 `relim()` reads each collection's
`get_datalim()`.

Seaborn's filled step histogram is a `FillBetweenPolyCollection`, which caches its bounds when it is
created. Harpy scales the vertices in place, so the cached bounds stay at the raw counts, and
[`ax.relim()`](src/harpy/plot/_renderers.py#L172) puts them in the data limits: the y-axis goes to
210 instead of 52.5. `set_data()` is the only public way to refresh the cache, and it needs the
original curve data.

### Failing test

[test_spatial_bin_histograms.py:143](src/harpy/_tests/test_qc/test_spatial_bin_histograms.py#L143):
`test_filled_percentage_histogram_autoscales_to_percentage_not_raw_counts`

### Fix

Leave the scaled collections out of `relim()`, then add their scaled vertices as the code already
does:

```python
scaled = ax.collections[collection_start:]
visible = [collection.get_visible() for collection in scaled]
for collection in scaled:
    collection.set_visible(False)
try:
    ax.relim(visible_only=True)
finally:
    for collection, was_visible in zip(scaled, visible, strict=True):
        collection.set_visible(was_visible)
```

Side effect: other invisible artists no longer count toward the limits.

Rejected alternatives:

- seaborn `weights=100 / N`: numpy refuses automatic bin estimation (`bins="auto"`, harpy's
  default) for weighted data;
- scaling through the collection transform: the vertices would stay in counts, unlike the patches
  and lines.

A monkeypatched prototype passes all 62 tests in `test_spatial_bin_histograms.py` and
`test_instance_plots.py`.

### Verify

```bash
.venv/bin/pytest -q -n auto src/harpy/_tests/test_qc/test_spatial_bin_histograms.py src/harpy/_tests/test_qc/test_table/test_instance_plots.py
```

---

## 5. numba 0.68: in-place add on unsigned integer arrays fails to type

### Cause

Reproduces with numba alone:

```python
import numba
import numpy as np


@numba.njit
def f(a):
    acc = np.zeros(a.shape[0], dtype=a.dtype)
    acc += a.sum(axis=1)
    return acc


f(np.ones((3, 4), dtype=np.uint32))
# AttributeError: 'NoneType' object has no attribute 'args'
# (numba/core/typing/npydecl.py, NumpyRulesInplaceArrayOperator.generic)
```

It fails for `uint8`, `uint16` and `uint32` with numba 0.68.0; `int*`, `float*` and `uint64` work.
scanpy's `top_segment_proportions_sparse_csr`, used by `sc.pp.calculate_qc_metrics` on CSR input, does
exactly this. Harpy's allocation stores
[uint32 counts](src/harpy/table/_allocation.py#L898), so scanpy's QC metrics fail on any allocation
output, including when users call scanpy themselves. No upstream report was found. Older numba
releases between 0.65.1 and 0.68.0 were not tested.

### Failing tests

`test_table/test_preprocess.py::test_preprocess_transcriptomics`, all 4 parametrizations, through
[`sc.pp.calculate_qc_metrics`](src/harpy/table/_preprocess.py#L321).

### Fix

- Pin numba below 0.68 in the `[tool.uv]` `override-dependencies` of `pyproject.toml`, next to
  `numba>=0.64.0`. This applies to both `locked` and `stable` CI jobs, but not to pip installs.
  It needs `uv lock` and an environment rebuild.
- Report the snippet above to numba.

### Verify

After rebuilding the environment:

```bash
.venv/bin/pytest -q src/harpy/_tests/test_table/test_preprocess.py
```
