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

`uv.lock` still records the old specifier, so `uv sync --locked` fails until `uv lock` is rerun.

Follow-ups now that 0.12 is unsupported:

- the empty-sparse-matrix workaround in `_decode_anndata_element` cites an AnnData 0.12.10 bug,
  which 0.13 no longer has: removed in this PR, see [1d](#1d-remove-the-anndata-01210-workaround-for-empty-sparse-matrices);
- `roadmap/integration_downstream/` mentions the 0.12.10 floor. No issue is filed for it.

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

- [x] 1. anndata 0.13: `X` is stored as `layers[None]`
- [x] 2. anndata 0.13: backed sparse datasets reject item assignment with `TypeError`
- [x] 3. anndata 0.13: automatic sharding no longer groups the chunks of dense matrices (#270)
- [x] 4. matplotlib 3.11: `Axes.relim()` includes collections
- [x] 5. numba ≥ 0.67: sums of small unsigned integers are typed as `int64`

---

## 1. anndata 0.13: `X` is stored as `layers[None]`

Most of the failures. Harpy's table IO assumed that `X` and `layers` are separate in memory, which
fails in three ways (1a–1c). The fixes follow the decision
[`X` in memory and on disk](#x-in-memory-and-on-disk). The 1a addendum fixes no failure: it reads
`X` and named layers through `layers`, following the decision
[Choosing `X` or a layer](#choosing-x-or-a-layer-layer). 1d fixes no failure either: it removes a
workaround that the `anndata>=0.13` floor makes unnecessary.

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

Done. The expected value leaves out the key `None`
([test_read.py:355](src/harpy/_tests/test_table/test_io/test_read.py#L355)), with a comment that
the stored `layers` group holds only the named layers. The filter is written out rather than built
on `_matrix_components`, so that the test does not compute its expectation with the translation it
checks. The step 1 verify modules, with `test_storage/test_anndata.py` and `test_qc/test_table`,
leave 3 failures: 2 from step 2 and 1 from step 3.

### 1d. Remove the anndata 0.12.10 workaround for empty sparse matrices

[_decode_anndata_element](src/harpy/_storage/_anndata.py#L295) reads a sparse matrix whose
compressed axis is empty (a CSR matrix without rows, a CSC matrix without columns) through a delayed
`read_elem`, because "AnnData 0.12.10 constructs invalid Dask chunks for an empty compressed axis".

AnnData 0.13 no longer does. Checked with `read_elem_lazy` on an empty CSR `(0, 3)` and an empty CSC
`(2, 0)`, without a block length and with lengths 1 and 1000: the Dask chunks are valid, `((0,), (3,))`
and `((2,), (0,))`, and computing gives the empty matrix. With the `anndata>=0.13` floor, the branch
only guards against a bug that no supported version has.

Without the branch, the empty matrix takes the ordinary path:

- [_sparse_block_length](src/harpy/_storage/_anndata.py#L404) sizes its blocks, and
  [_auto_sparse_block_length](src/harpy/_storage/_anndata.py#L474) already returns 1 for an empty
  axis, so nothing divides by zero, and `read_elem_lazy` accepts that length.
- The Dask array has the chunks the workaround's `from_delayed` gave, so the writer that follows sees
  the same array.

Sites:

- the branch in `_decode_anndata_element`;
- the comment "The caller handles an empty compressed axis before sizing blocks" in
  `_sparse_block_length` ([line 438](src/harpy/_storage/_anndata.py#L438)), which then says that
  `_auto_sparse_block_length` returns one block for an empty axis.

Covered by
[test_empty_matrix_axes_roundtrip](src/harpy/_tests/test_table/test_io/test_write.py#L858): CSR and
CSC matrices with an empty axis, read lazily and written back.

Verify:

```bash
.venv/bin/pytest -q -n auto src/harpy/_tests/test_table/test_io/test_write.py src/harpy/_tests/test_table/test_io/test_read.py src/harpy/_tests/test_storage/test_anndata.py
```

Done. The branch is gone, and with it the `dask.delayed` import and the local `shape`, which only it
used. The comment in `_sparse_block_length` now says that an empty compressed axis reaches it too, and that
`_auto_sparse_block_length` returns 1 for it instead of dividing by the axis length. The verify command leaves only step 3's sharding test failing (338
passed), `test_empty_matrix_axes_roundtrip` included.

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

Done. `test_backed_handles_reject_direct_writes` chooses the expected error from `matrix_kind`, and
the allocation test from the handle's type: `TypeError` for the CSR datasets (`X` and
`obsm["negative"]`), `ValueError` for the canonical centers, a read-only Zarr array. The allocation
test's comment no longer explains that sparse writes target existing entries, as a CSR dataset now
rejects any assignment. Harpy's documentation promises read-only handles without naming an exception
(`read_zarr`'s docstring, `docs/development/storage.md`), so it needs no change. The verify command
passes (223 tests).

---

## 3. anndata 0.13: automatic sharding no longer groups the chunks of dense matrices

Tracked in [saeyslab/harpy#270](https://github.com/saeyslab/harpy/issues/270), which also holds the
full fix. This step only applies the temporary measure.

### Cause

Two changes, the second found after fixing the first:

- With `ad.settings.auto_shard_zarr_v3 = True` and no explicit `shards`, anndata now filters zarr's
  `Automatic shard shape inference is experimental` warning (`anndata/_io/specs/methods.py`), so
  `pytest.warns` saw no warning.
- With the `pytest.warns` block removed, the shard assertion fails: the shards are `(3, 4)`, the
  chunks themselves, instead of `(6, 4)`. AnnData 0.13 sets Zarr's `array.target_shard_size_bytes`
  to 1 GB for writes with automatic sharding, and Zarr (≥ 3.1.4) then infers shards by size: one
  factor shared by all axes, never growing a shard past the array along any axis. Harpy's dense
  chunks span all columns, so the factor stays 1, and every dense shard holds one chunk, whatever
  the matrix size. The 1-D arrays of sparse matrices are still grouped once longer than two chunks
  (5,000,000 values: 9 chunks per shard).

So automatic sharding no longer reduces the number of files for dense matrices. Harpy's Dask write
blocks also cover whole stored chunks, not whole shards, so shards that group chunks were already
partly rewritten by each write block; the issue lists that too.

Automatic sharding is also AnnData's default now: `ad.settings.auto_shard_zarr_v3` is `True` in 0.13,
`False` in 0.12.10. Harpy passes no `shards`, so without changes every write to a Zarr v3 store is
sharded; Zarr v2 stores never are. What Harpy writes to a Zarr v3 store by default:

| Array                             | Shards                              |
| --------------------------------- | ----------------------------------- |
| dense matrix                      | one chunk per shard                 |
| sparse matrix written from memory | grouped once longer than two chunks |
| sparse matrix written lazily      | decided by the first Dask block     |
| `obs` and `var` columns           | AnnData's automatic sharding        |

Zarr fixes the shard shape when it creates an array, from the array's length at that moment, and
appending does not change it. A sparse matrix written from memory is created at its full length:
2,000,000 values give shards of 3 chunks (4 × 524,288 would exceed the length). A lazy one is created
from its first Dask block, and AnnData appends the other blocks, so that block decides the shards:

| Dask blocks (10,000,000 values) | First block                 | Shards of `data`       | Write time, unsharded / sharded |
| ------------------------------- | --------------------------- | ---------------------- | ------------------------------- |
| 100,000 rows                    | ~500,000 values, < 2 chunks | `(524288,)`: 1 chunk   | 1.5 s / 1.3 s                   |
| 1,000,000 rows                  | ~5,000,000 values           | `(4718592,)`: 9 chunks | 0.2 s / 0.2 s                   |

Harpy's lazy reads default to blocks of about Dask's `array.chunk-size` (128 MiB), which usually hold
millions of values, so grouped shards are the common case. Then an appended block that covers part of
a shard makes Zarr read and rewrite it. The cost is bounded: shards are never larger than the first
block, so with equally sized blocks each shard is touched by at most two blocks, and the writes above
showed no measurable difference.

Two comments on #270 correct the issue:
[the first](https://github.com/saeyslab/harpy/issues/270#issuecomment-6341711886) says that sharding
is AnnData's default; [the second](https://github.com/saeyslab/harpy/issues/270#issuecomment-6341907886)
corrects the first, which said that lazy sparse matrices always get one chunk per shard and that
appends never rewrite shards. Both hold only for a first block of fewer than two chunks.

### Failing test

[test_anndata.py:458](src/harpy/_tests/test_storage/test_anndata.py#L458):
`test_automatic_sharding_keeps_row_only_chunks_as_inner_chunks`

### Decision

A temporary measure, so that anndata can be bumped and Harpy released, with the Harpy-owned shard
layout left to #270:

- Harpy writes dense matrices without shards: `_write_element_with_layout` passes `shards=None` for
  them. Sparse matrices keep following AnnData's automatic sharding.
- Under anndata 0.13 this loses nothing, as dense shards held one chunk each. Unlike keeping
  AnnData's automatic shards, it does not depend on Zarr's inference: if that grouped chunks again,
  the write blocks, aligned with chunks, would partly rewrite shards unnoticed.
- As sharding is AnnData's default, this changes Harpy's default output for dense matrices only:
  from one chunk per shard to no shards. Everything else follows AnnData's new default.

Checked with a patched writer, sharding on:

| Zarr | Dense, in memory and Dask | Sparse `data`      | Values     |
| ---- | ------------------------- | ------------------ | ---------- |
| v3   | no shards, harpy's chunks | shards `(524288,)` | round-trip |
| v2   | no shards                 | no shards          | round-trip |

Neither emits a Zarr warning.

### Sites

- [_write_element_with_layout](src/harpy/_storage/_anndata.py#L742): pass `shards=None` for dense
  matrices; the comment at [line 749](src/harpy/_storage/_anndata.py#L749), which calls AnnData's
  automatic sharding opt-in, says instead that it is AnnData's default since 0.13, that dense
  matrices are unsharded for now, and links #270.
- Descriptions of sharding, to say that it applies to sparse matrices only for now, with #270:
  - the `_STORED_CHUNK_BYTES` comment ([_anndata.py:68](src/harpy/_storage/_anndata.py#L68)), which
    calls sharding the better answer for very large tables;
  - the docstring of `_write_anndata_element` ([_anndata.py:785](src/harpy/_storage/_anndata.py#L785));
  - the public docstring of `write_table` ([_write.py:92](src/harpy/table/io/_write.py#L92));
  - [storage.md:188](docs/development/storage.md#L188).
- The test becomes "automatic sharding applies to sparse matrices only": dense unsharded with harpy's
  row-only chunks, sparse sharded, with a link to #270. Its `pytest.warns` block is already removed,
  and its docstring, which calls automatic sharding opt-in, follows the default.

`ad.settings.zarr_write_format`, used in this test, is deprecated in anndata 0.13 and removed in
0.14. AnnData shards according to the format of the group it writes to (`f.metadata.zarr_format` in
`anndata/_io/specs/methods.py`), not that setting, so the test drops it; it was Harpy's only use.

### Verify

```bash
.venv/bin/pytest -q -n auto src/harpy/_tests/test_storage/test_anndata.py src/harpy/_tests/test_table/test_io
```

Done.

- `_write_element_with_layout` passes `shards=None` for dense matrices, with a comment on why and
  #270; the comment on the other elements says that AnnData's automatic sharding is its default
  since 0.13. The function's docstring states the `shards=None` as well.
- The descriptions say that dense matrices are stored without shards and everything else follows
  `auto_shard_zarr_v3`, on by default since 0.13: the `_STORED_CHUNK_BYTES` comment and the
  `_write_anndata_element` docstring, with #270; the public `write_table` docstring, without it;
  `docs/development/storage.md`, with #270.
- `test_automatic_sharding_applies_to_sparse_matrices_only` replaces the old test, for a dense
  matrix in memory and from Dask: the dense matrix keeps Harpy's row-only chunks without shards, and
  a sparse matrix keeps Harpy's chunk length with automatic shards around it. It asserts only that
  the sparse shards exist, not their size, which is Zarr's inference.

The verify command passes (1006 tests), and so do `test_allocation.py`,
`test_allocation_intensity.py` and `test_canonical_centers.py` (167 tests), whose writers store dense
matrices through the same callback.

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

Done, as planned, and extended to an older bug in the same function.

The bug: a second percentage histogram on the same axes left the y-limit at 210 instead of about 52,
for two peaks of 50%. The first histogram's filled area keeps its cached bounds in counts, and both
`relim()` (from 3.11) and the loop over earlier collections (`get_datalim()`, under 3.10 too) read
that cache. Harpy never draws two histograms on one axes itself, but `_scale_histogram_artists`
promises to preserve earlier axes contents.

The fix bounds every polygon collection without offsets by its own vertices, not only the ones scaled
in this call:

- `_bounded_by_vertices` tells such collections, filled step and poly histograms included, from those
  placed by offsets, such as scatter points or `hexbin` cells, which keep `get_datalim()`;
- the polygon collections are hidden from `ax.relim(visible_only=True)` and added through their
  vertices; the others are added through `get_datalim()`, which matters before matplotlib 3.11.

`test_a_percentage_histogram_keeps_the_limits_of_earlier_axes_contents` covers an earlier percentage
histogram, which failed before the fix, and earlier scatter points. All plot and QC tests pass
(`test_qc` and `test_table/test_pl`, 329 tests).

---

## 5. numba ≥ 0.67: sums of small unsigned integers are typed as `int64`

### Cause

Three layers; the second and third only show because of the first.

1. numba 0.67.0 reimplemented `array.sum()` and `np.sum()` (`@overload` with the intrinsics
   `_numpy_sum` and `_numpy_sum_axis`). Its result dtype comes from
   [get_ret_dtype_if_any](.venv/lib/python3.13/site-packages/numba/np/arraymath.py#L420): every
   integer narrower than `intp` becomes `intp`, a signed `int64`. The comment there says "signed
   integers", but the condition does not check signedness, so `uint8`, `uint16` and `uint32` become
   `int64`, where NumPy gives `uint64`. Up to 0.66, `_expand_integer` in
   `numba/core/typing/arraydecl.py` widened unsigned integers to `uintp`, as NumPy does.

   | dtype                      | `.sum(axis=1)` in numba 0.68 | in NumPy |
   | -------------------------- | ---------------------------- | -------- |
   | `uint8`, `uint32`          | `int64`                      | `uint64` |
   | `uint64`, `int32`, `int64` | as NumPy                     |          |

2. scanpy's `top_segment_proportions_sparse_csr`, used by `sc.pp.calculate_qc_metrics` on CSR input,
   adds that sum in place to an array of the data's dtype:

   ```python
   acc = np.zeros(..., dtype=data.dtype)          # uint32 for Harpy's counts
   acc += partitioned[:, prev:ns[j]].sum(axis=1)  # int64 from numba 0.67
   ```

   Writing an `int64` result into a `uint32` array is a cast between kinds, which NumPy refuses in
   place as well (`np.can_cast(int64, uint32, "same_kind")` is `False`). With numba ≤ 0.66 the sum
   was `uint64`, which casts in place to `uint32`.

3. numba then crashes instead of raising a typing error: `NumpyRulesInplaceArrayOperator.generic`
   gets no signature and fails on `assert len(sig.args) == 3`, hence
   `AttributeError: 'NoneType' object has no attribute 'args'`. This is an old numba bug
   ([numba#2784](https://github.com/numba/numba/issues/2784), open since 2018, the same crash for
   unsigned in-place operations). On 0.68.0 every unsigned `+=` signed pair crashes this way, and so
   does every signed `+=` `uint64` pair.

Affected: numba 0.67.0 and 0.68.0, the latest release. 0.66.0 has no `get_ret_dtype_if_any` yet. No
numba issue reports the unsigned sum dtype; the open pull requests on `sum` performance
([numba#10929](https://github.com/numba/numba/pull/10929),
[numba#10924](https://github.com/numba/numba/pull/10924)) do not change it.

Impact: Harpy has no numba code of its own. scanpy's QC metrics fail on any CSR matrix of `uint8`,
`uint16` or `uint32` values, which includes Harpy's allocation output
([uint32 counts](src/harpy/table/_allocation.py#L898)), also when users call scanpy on Harpy's
tables.

### Failing tests

`test_table/test_preprocess.py::test_preprocess_transcriptomics`, all 4 parametrizations, through
[`sc.pp.calculate_qc_metrics`](src/harpy/table/_preprocess.py#L321).

### Decision: `int32` counts instead of a numba pin

Harpy stores its count matrices as `int32` instead of `uint32`, and does not pin numba.

Why not pin `numba<0.67`: numba 0.66.0, the last release before the bug, requires `numpy<2.5`, so the
pin would also hold numpy on 2.4.x. The first dependency that needs numpy ≥ 2.5 or a newer numba
would conflict with it, and the pin could only go once numba fixes the bug upstream and someone
remembers to lift it.

Why `int32`:

- Range: up to 2,147,483,647 points per instance and feature, far above realistic counts.
- Memory: 4 bytes, as `uint32`.
- It works with numba 0.67 and 0.68: in scanpy's kernel, `int32 += int64` is an allowed in-place cast,
  and `calculate_qc_metrics` ran on an `int32` CSR matrix under numba 0.68. Of the integer types, only
  `uint8`, `uint16` and `uint32` fail.
- No compatibility cost: neither `uint32` producer is in the latest release, v0.5.0 (it has no
  aggregation writer, and its allocation does not cast to `uint32`), so no released store has
  `uint32` counts from Harpy.
- Harpy's own QC accepts signed integers and accumulates in `uint64`
  (`test_counts_stored_as_whole_floats_or_signed_integers_give_the_result_of_uint32`).

Rejected: `float32`, the usual scverse default, exact for integers only up to 16,777,216; `int64` and
`uint64`, twice the memory.

The dtype is one constant, `_COUNT_DTYPE = np.dtype(np.int32)`, in
[_aggregation_contracts.py](src/harpy/table/_aggregation_contracts.py#L1), with a comment that gives
the reason above. That module holds the shared contracts of feature-class aggregation, imports only
`_feature_panels` and `utils._names`, and `_allocation.py` already imports from it. Not in
`table/_table.py`: its helpers serve table processing in general, none of its users produces counts,
and it imports `table.io._add_table`, `shape._shape` and `_storage._anndata`, which neither producer
imports today. Once the numba issue is filed, the comment links it.

### Fix

- The constant in `_aggregation_contracts.py`.
- Producers, both through the constant:
  - [_aggregation_writer.py](src/harpy/table/_aggregation_writer.py#L313): the `uint32` dtype and
    meta of the count matrices, the cast at [line 357](src/harpy/table/_aggregation_writer.py#L357),
    and the range check at [line 353](src/harpy/table/_aggregation_writer.py#L353), with
    `np.iinfo(_COUNT_DTYPE).max` and a message naming the constant's dtype;
  - [_allocation.py:898](src/harpy/table/_allocation.py#L898): the cast.
- Documentation: "uint32 CSR" in the allocation docstring
  ([lines 194 and 202](src/harpy/table/_allocation.py#L194)).
- Tests in `test_allocation.py`: the range check
  ([lines 243 and 252](src/harpy/_tests/test_table/test_allocation.py#L243)) and the output dtype
  ([lines 552 and 588](src/harpy/_tests/test_table/test_allocation.py#L552)) take the expected dtype
  and limit from the constant, so that a later change happens in one place. One small test pins
  `_COUNT_DTYPE == np.dtype(np.int32)`, so that such a change is a deliberate, visible test change.
  The QC tests that build `uint32` input stay: Harpy's QC still accepts `uint32` counts from
  elsewhere. The `uint32` arrays in `test_canonical_centers.py` are instance IDs and labels, not
  counts, and stay too.
- `pyproject.toml`: back to `numba>=0.64.0` and its earlier comment, dropping the `<0.67` pin added
  first.
- Still worth reporting to numba for other users with unsigned data, with a reproducer that points at
  `get_ret_dtype_if_any`; the fix there is to return `types.uintp` for unsigned integers. Harpy no
  longer depends on it. Filed as [numba#10939](https://github.com/numba/numba/issues/10939).

  ```python
  import numba
  import numpy as np


  @numba.njit
  def f(a):
      return a.sum(axis=1)


  f(np.ones((2, 2), dtype=np.uint32)).dtype  # int64; NumPy gives uint64
  ```

Done.

- `_COUNT_DTYPE = np.dtype(np.int32)` is in `_aggregation_contracts.py`, with the reason; the
  aggregation writer and the allocation use it, and the allocation docstring says `int32 CSR`.
- `pyproject.toml` is back to `numba>=0.64.0`, as committed.
- `test_allocation.py` takes the expected dtype and the range limit from the constant, checks the
  dtype of `X` as well, and `test_counts_are_stored_as_int32` pins the constant.

One case the switch does not cover: `test_preprocess_transcriptomics` runs scanpy's QC metrics on the
example store of `resolve_example()`, downloaded from Harpy's dataset registry, whose
`table_transcriptomics` holds `uint32` CSR counts (its other tables keep `uint32` in their
`raw_counts` layer). Stored data with `uint32` counts still fails scanpy's QC metrics under numba 0.67
and 0.68, whatever Harpy writes now. The test casts that table's `X` to `_COUNT_DTYPE` before
preprocessing, with a comment that it waits for the numba fix,
[numba#10939](https://github.com/numba/numba/issues/10939).

The verify command passes with numba 0.68.0 and numpy 2.5.4 (268 tests).

Follow-ups:

- Done: the numba issue is filed, [numba#10939](https://github.com/numba/numba/issues/10939), and
  linked at `_COUNT_DTYPE` and in the comment of `test_preprocess_transcriptomics`. Once numba fixes
  it, the cast in that test can go.
- Regenerate the example store with `int32` counts (a new zip in the dataset registry and its hash),
  so that the test needs no cast and users of the example data are not affected. Possibly a Harpy
  issue.

### Verify

The current environment has numba 0.68.0, so no rebuild is needed:

```bash
.venv/bin/pytest -q -n auto src/harpy/_tests/test_table/test_preprocess.py src/harpy/_tests/test_table/test_allocation.py src/harpy/_tests/test_table/test_canonical_centers.py src/harpy/_tests/test_qc/test_table
```
