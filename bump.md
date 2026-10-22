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

## Decisions

### anndata lower bound

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

### `X` in memory and on disk

What anndata 0.13 does, checked on 0.13.4 (see also the
[release post](https://scverse.org/blog/2026-anndata-013/)):

- The public API is unchanged: `adata.X` is read, set and deleted as before.
- In memory, `X` is the entry `None` of `layers`. `AnnData._X` no longer exists: the getter returns
  `layers.get(None)`, and the setter writes `layers[None]` into the existing `_layers` dict.
  Iterating `layers` yields `None` (`['counts', None]`); the repr shows `layers: 'counts', None (.X)`.
- Each `view.X` is a new view. Assigning to a view's `X` no longer changes the parent's `X`.
- `AnnData(layers={None: x})` sets `X`. Passing `X=` as well requires the same object.
- `adata.layers = {...}` without a `None` key keeps `X` and emits a `FutureWarning`; a future release
  may drop `X` instead. `{None: None, ...}` drops `X`.
- On disk nothing changed: `X` is stored at the top-level `X`, the `layers` group holds only the named
  layers, and the encoding is still `anndata` 0.1.0. The writer reports the matrix as `X`.
- `Raw` is unchanged and keeps its own `_X`.

Harpy's contract:

- Component paths are storage paths and stay as they are: `("X",)` names `X` and `("layers", key)` a
  named layer. They are public API (`components=[("X",)]`, `x_to=("X",)`), their segments are
  strings, and they match the layout on disk.
- In memory, harpy follows anndata: `X` is the entry `None` of the `layers` slot. When harpy reads,
  installs or restores components of an attached table, `X` is not a slot of its own.
- One place translates between the two, next to `_MATRIX_MAPPINGS` in
  [_anndata.py](src/harpy/_storage/_anndata.py#L38):
  - `_matrix_components(adata)` yields `(component path, value)` for every entry of every matrix
    mapping in memory, with `layers[None]` as `("X",)`. It iterates `_MATRIX_MAPPINGS` itself.
  - `_memory_path(path)` returns the location of a component path in memory: `("layers", None)` for
    `("X",)`, any other path unchanged.
- Code that copies one AnnData into a new one passes `layers` whole, `None` included, and no `X=`
  argument; anndata sets `X` from `layers[None]`. Checked with Dask and sparse matrices, from a view,
  and without `X`.
- Removing `X` assigns a `layers` mapping with `None: None`. A mapping without the `None` key would
  keep `X` and warn.
- Lists of layer names shown to users skip `None` inline: for users, `X` is not a named layer.

Rejected:

- Filtering `None` out of `layers` at each site (the first attempt at 1a, `_matrix_mapping`). It hides
  the difference between memory and disk instead of stating it, and its `slot` argument depends
  implicitly on `_MATRIX_MAPPINGS`.
- Special-casing `X` in the rollback (the first plan for 1b). With the translation, an `X` update is a
  `layers` update, which the rollback already handles.

### Choosing `X` or a layer: `layer`

Functions that read the matrix aligned with `var` take `layer: str | None`, in scanpy's convention:
`None` reads `X`, a name reads that layer. In anndata 0.13 both are keys of `layers`, and `adata.X`
is `adata.layers.get(None)`. These functions read `adata.layers.get(layer)` or `adata.layers[layer]`,
without a branch on `None` to choose the matrix.

`summarize_table` keeps its behaviour:

- `layer=None` is the default, and reads `X` when the table has one. Without `X`, only the `.obsm`
  matrices are read.
- A named layer is an explicit request: it must exist, and raises otherwise.
- With nothing to read, no `X` and no `obsm_keys`, it raises.

`adata.layers.get(layer)` treats a missing key and a `None` value alike. In 0.13 a named layer can
hold `None` (`adata.layers["counts"] = None` keeps the key); only the key `None` is removed when set
to `None`.

Rejected:

- A flag `read_x: bool = True`, to separate which matrix is read from whether one is read: a second
  parameter for one choice confuses readers, and one of its four combinations with `layer` is invalid.
- A strict `layer=None` that raises when `X` is missing: it drops the `.obsm`-only summaries of
  tables without `X`, which are supported and tested.
- A lenient named layer, read as `.obsm` only when it is missing: with `obsm_keys` given, a typo or a
  stale name would silently drop the requested matrix, and `calculate_qc_metrics` would write no
  `var` columns without an error.

Implemented in the 1a addendum, as it changes no behaviour that users rely on.

## Steps

- [ ] 1. anndata 0.13: `X` is stored as `layers[None]` (1a, its addendum and 1b done; 1c open)
- [ ] 2. anndata 0.13: backed sparse datasets reject item assignment with `TypeError`
- [ ] 3. anndata 0.13: zarr's sharding warning is silenced
- [ ] 4. matplotlib 3.11: `Axes.relim()` includes collections
- [ ] 5. numba 0.68: in-place add on unsigned integer arrays fails to type

---

## 1. anndata 0.13: `X` is stored as `layers[None]`

Most of the failures. Harpy's table IO assumed that `X` and `layers` are separate in memory, which
fails in three ways (1a–1c). The fixes follow the decision
[`X` in memory and on disk](#x-in-memory-and-on-disk). The 1a addendum fixes no failure: it reads
`X` and named layers through `layers`, following the decision
[Choosing `X` or a layer](#choosing-x-or-a-layer-layer).

### 1a. `None` key in `layers` iterations

| Error                                                                | Site                                                                                                                                                                                     |
| -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `If you provide layers[None] and X, they must be identical.`         | [_prepare_anndata_value](src/harpy/_storage/_anndata.py#L780) passes `layers` including `None` and a separately wrapped `X` to `AnnData`. `map_blocks` makes them two different objects. |
| same                                                                 | [_add_table](src/harpy/table/io/_add_table.py#L130) passes `dict(adata.layers)` and `adata.X`; on views these are two different view objects.                                            |
| `argument should be a str or an os.PathLike object … not 'NoneType'` | [_table_components](src/harpy/table/io/_updates.py#L509) yields `("layers", None)`, and [_exists_on_disk](src/harpy/table/io/_updates.py#L542) calls `joinpath(None)`.                   |

Sites without a failing test:

- [_table_matrices](src/harpy/table/_table.py#L65) yields `X` twice (`X` and `layers[None]`);
- [plot_adata](src/harpy/plot/_qc_cells.py#L29) plots `X` twice;
- [_collect_class_matrices](src/harpy/qc/table/_summarize_table.py#L518) calls `sorted(adata.layers)`
  in the error message for an unknown layer. On a table with `X` and at least one named layer, it
  raises `TypeError` mixing `None` with strings; `X` alone sorts fine, which is why the existing
  `layer="absent"` case of `test_invalid_tables_and_arguments_raise` never failed.

Fix:

- `_prepare_anndata_value` and `_add_table` copy one AnnData into a new one: pass `layers` whole and
  drop the `X=` argument.
- `_table_components` and `_table_matrices` name matrices by component path: build them on
  `_matrix_components`.
- `plot_adata` and `_collect_class_matrices` list layer names for users: skip `None` inline.

Done. [_matrix_components](src/harpy/_storage/_anndata.py#L77) replaces the first attempt,
`_matrix_mapping`. `_table_matrices` now yields `X` where `layers` holds it, after the named layers,
so when both `X` and a layer are lazy, `_require_in_memory_table` may name the layer first. The step 1
modules have 15 failures left: 10 from the `_X` rollback (1b), 3 from 1c and 2 from step 2.

### 1a addendum: read `X` and layers through `layers`

No behaviour that users rely on changes. One error message does: `layer="counts"` with
`adata.layers["counts"] = None` (the key exists, its value is `None`) used to pass the key check and
fail in the CSR check, with "adata.layers['counts'] is a NoneType, not a CSR matrix … Convert it
first". It now counts as absent and raises "Layer 'counts' is not a layer of the table". Both are a
`ValueError`.

- [_collect_class_matrices](src/harpy/qc/table/_summarize_table.py#L518): one lookup,
  `adata.layers.get(layer)`, for `X` and named layers; a missing named layer raises. The type check
  of `layer` stays first, so an unhashable value still gets a clear error. The error for nothing to
  read ([line 540](src/harpy/qc/table/_summarize_table.py#L540)) stays.
- [_FeatureClassMatrix](src/harpy/qc/table/_summarize_table.py#L227): `attribute` becomes
  `Literal["layers", "obsm"]`; `label` and `location` already map the key `None` to `x` and `X`.
- [_get_matrix](src/harpy/table/pp/_normalize.py#L226) and
  [_set_matrix](src/harpy/table/pp/_normalize.py#L237): `adata.layers[layer]`. Only the error message
  of `_get_matrix` still tells a missing `X` from a missing layer. The `X` setter's broadcasting of a
  scalar and its reshaping, which normalization never uses, no longer apply.
- [_grouped_obs_mean](src/harpy/table/cell_clustering/_clustering.py#L191): `adata.layers[layer]`.
  On a table without `X` it raises `KeyError` at once, instead of failing later on `None`.

Verify:

```bash
.venv/bin/pytest -q -n auto src/harpy/_tests/test_qc/test_table src/harpy/_tests/test_table/test_pp src/harpy/_tests/test_table/test_cell_clustering
```

Done. `_set_matrix` is gone: its two callers assign `adata.layers[layer]`, which on a view behaves as
setting `X` does (the same warning, and the view becomes an actual table). The verify command passes
(150 tests); an unhashable `layer` still gets the `ValueError` naming the available layers.

The error message lists only named layers whose value is not `None`, matching the lookup: a named
layer set to `None` is absent, and is no longer listed as available.
[test_an_unknown_layer_raises_and_lists_the_named_layers](src/harpy/_tests/test_qc/test_table/test_summarize_table.py#L178)
covers a table with `X` and a named layer, with and without a named layer set to `None`. It fails
without the 1a fix (`TypeError`) and without this filter (the `None` layer is listed). The other
changes of 1a and its addendum are covered by the existing tests; no tests were added for them.

### 1b. Rollback of attached tables uses `_X`

Error: `AttributeError: 'AnnData' object has no attribute '_X'`.

How the rollback works: `add_table_components`, `remove_table_components` and `add_table_updates` end
in `_update_table_components`, which changes the attached table so that a failure can be undone. It
retains the private object of every slot that changes
([snapshot](src/harpy/table/io/_components.py#L269)), builds new slot objects from shallow copies
([_prepare_memory_updates](src/harpy/table/io/_components.py#L576)), installs them through the
setters ([_install_memory_updates](src/harpy/table/io/_components.py#L635)), and on failure puts the
retained objects back ([restore](src/harpy/table/io/_components.py#L304)).

Why this fails in 0.13:

- The snapshot reads `table._X`, which no longer exists.
- Retaining `_layers` instead is not enough: installing `table.X = value`
  ([line 642](src/harpy/table/io/_components.py#L642)) writes `layers[None]` into that same
  `_layers` dict, so the restore would not bring back the old `X`.
- Once `X` goes through `layers`, an absent `X` would make
  [_updated_mapping](src/harpy/table/io/_components.py#L561) pop the key `None`, and anndata reads a
  `layers` mapping without that key as "keep `X`": the store would lose `X` while the attached table
  keeps it, with only a `FutureWarning`.

Sites: only `_components.py`.
[add_table_components_by_region](src/harpy/table/io/_components_by_region.py#L198) accepts
`("obsm", key)` paths only, so its [snapshot](src/harpy/table/io/_components_by_region.py#L219)
never sees `X` and needs no change.

Fix: address the attached table through `_memory_path`, next to `_matrix_components` in
`_anndata.py`. It returns `("layers", None)` for `("X",)` and every other path unchanged. Component
paths stay as they are for storage and the public API.

- [_memory_component](src/harpy/table/io/_components.py#L338) looks up `_memory_path(path)`: `X` is
  the key `None` of `layers`, absent when that key is missing. The `X` case of
  [line 367](src/harpy/table/io/_components.py#L367) goes, the one of `raw` stays, and the
  docstring, which mentions `table.X is None`, follows.
- [_prepare_memory_updates](src/harpy/table/io/_components.py#L576) translates the path of every
  change with `_memory_path`, and turns an absent `X` into the value `None`. An `X` update then takes
  the existing mapping branch ([line 624](src/harpy/table/io/_components.py#L624)):
  `_updated_mapping(table.layers, (None,), value)` returns a new dict with every layer and the new
  `X`, merged with changes to named layers of the same call. `_install_memory_updates` assigns it as
  `table.layers = ...`, which replaces `_layers` instead of writing into it, and no longer sets
  `table.X`.
- `X` is absent, and must become `None: None`, in two cases:
  - a deletion, by [remove_table_components](src/harpy/table/io/_components.py#L158) with `("X",)`
    or `X` in `delete` of [add_table_components](src/harpy/table/io/_components.py#L44);
  - a reopened `X` that the store does not hold: [add_table_updates](src/harpy/table/io/_updates.py#L299)
    passes `reopen_also=(("X",),)` after `x_to=("layers", ...)`, and the reopen gives
    [absent](src/harpy/table/io/_components.py#L297) when the store has no `X`.

  A replacement with `None`, `components={("X",): None}`, already gives the value `None`. Only the
  key `None` of `layers` needs this: `_updated_mapping` stays generic and still pops absent entries
  of `uns`, `obsm` and named layers. The explicit `None: None` also stays correct if anndata later
  drops `X` when the key is missing.

- The [snapshot](src/harpy/table/io/_components.py#L269) takes its slots from `_memory_path`, so an
  `X` update retains `_layers` without a special case.
- `raw._X` at [_components.py:607](src/harpy/table/io/_components.py#L607) stays: `Raw` still has
  `_X`.

Checked:

- `_update_table_components` uses `X` as a slot only at the places above.
- Updating only named layers already keeps `X`: the new dict is built from `table.layers`, which holds
  the key `None`, so no `FutureWarning`.
- `table.layers = {..., None: None}` drops `X`.

Tests: the 10 remaining 1b failures should pass. The risky cases are covered by existing tests, so
no new tests are expected:

- removing `X` asserts `table.X is None`:
  [test_components.py:132](src/harpy/_tests/test_table/test_io/test_components.py#L132) (deletion),
  [:172](src/harpy/_tests/test_table/test_io/test_components.py#L172) (replacement with `None`);
- the absent reopen:
  [test_x_to_a_layer_of_a_table_stored_without_x_leaves_the_attached_x_none](src/harpy/_tests/test_table/test_io/test_add_table_updates.py#L161);
- the rollback:
  [test_a_failure_restores_the_attached_table_and_the_store](src/harpy/_tests/test_table/test_io/test_add_table_updates.py#L174),
  for a failure during installation and during finalization.

Verify:

```bash
.venv/bin/pytest -q -n auto src/harpy/_tests/test_table/test_io src/harpy/_tests/test_table/test_pp/test_qc_metrics.py
```

Done. `_memory_path` is in [_anndata.py](src/harpy/_storage/_anndata.py#L89), next to
`_matrix_components`; `_components.py` uses it in the snapshot, `_memory_component` and
`_prepare_memory_updates`, as planned.

One test needed a change, which the plan missed:
`test_a_failure_restores_the_attached_table_and_the_store` itself retained `table._X` to check that
the rollback restores the exact objects. It now retains `table._layers` and `table.X`, and asserts
that the same dict and the same `X` object come back. The `X` identity also checks that the retained
dict was not written into.

The verify command leaves 5 failures, 3 from 1c and 2 from step 2. The component update tests
(`test_components.py`, `test_add_table_updates.py`, `test_components_by_region.py`,
`test_add_table.py`, 246 tests) also pass with anndata's `FutureWarning` for a `layers` mapping
without the key `None` turned into an error, so no update drops `X` by omission.

Made explicit afterwards, as the removal of `X` relies on how AnnData's `layers` setter reads the
key `None`:

- [_memory_change](src/harpy/table/io/_components.py#L580) names the step: it returns where a
  change goes in memory and what is assigned, and its docstring states the rule (an absent `X`
  becomes `None` under the key `None`; a mapping without the key keeps `X`).
- `test_anndata_reads_x_from_the_key_none_of_an_assigned_layers_mapping` in
  `test_storage/test_anndata.py` pins that behaviour of AnnData: the key `None` sets or drops `X`
  without a warning, and a mapping without it keeps `X` with a `FutureWarning`.
- `docs/development/storage.md` states that every `layers` mapping harpy assigns names `X`
  explicitly, and the two tests in `test_components.py` that remove `X` say why.
- With `_memory_change` replaced by one that leaves the key out, 3 tests fail: the deletion of `X`
  (`test_unbacked_updates_do_not_compute_copy_or_access_storage`, both cases) and the absent reopen
  (`test_x_to_a_layer_of_a_table_stored_without_x_leaves_the_attached_x_none`).
- Not added: a `filterwarnings` entry in `pyproject.toml` turning that `FutureWarning` into an
  error. It would put AnnData's message text in the global test configuration, and stop guarding
  silently if AnnData rewords it.

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
