# `hp.tb.concat`: design

Source: issue #277 and its comment on feature matrices. This document fixes
the contract and the implementation plan; where it departs from the issue is
listed under [Departures from the issue](#departures-from-the-issue).

Depends on the canonical-cache refactor ([canonical_cache.md](canonical_cache.md)),
which landed in #283 and closed #279 and #282. Without #279, a table subset by
per-cell QC would fail canonical validation and could not be concatenated.

## Purpose

Stack Harpy tables along the rows into one table, keeping Harpy's `uns`
records valid. The use case is per-sample QC, one table per sample, followed
by one combined table for downstream analysis. `anndata.concat` drops `uns` by
default; `spatialdata.concatenate` rebuilds only `uns["spatialdata_attrs"]`
and passes the rest to `anndata.concat`.

`concat` involves no storage: it takes AnnData tables and returns one, in
memory or lazy. `hp.tb.io.add_table` or `hp.tb.io.write_table` stores it;
`hp.tb.validate_table` then checks it against the SpatialData object.

## API

```python
def concat(
    tables: Sequence[AnnData] | Mapping[str, AnnData],
    *,
    label: str | None = None,
    uns_merge: Literal["same", "unique", "first", "only"] | None = None,
    feature_matrices_key: str = _FEATURE_MATRICES_KEY,
    drop_mismatched_feature_matrices: bool = False,
) -> AnnData
```

- `tables`: at least one table. Errors name a table by its key in a mapping,
  otherwise by its position: `tables['s1']`, `tables[0]`.
- `label`: `obs` column naming each row's table: the keys of a mapping, or
  the positions as strings (`"0"`, `"1"`, ...) for a sequence, as
  `anndata.concat` does. It must not be an existing `obs` column of any table:
  `anndata.concat` would overwrite it silently.
- `uns_merge`: the `anndata.concat` strategy for the `uns` entries Harpy does
  not own, passed unchanged; see [`uns`](#uns). `None`, the default as in
  AnnData, drops them, so keeping them is an explicit choice. `"unique"` keeps
  records keyed per table, such as `uns["qc"][sample]`.
- `feature_matrices_key`: key in `uns` of the feature-matrix registry, as in
  `validate_table`, `summarize_table` and `calculate_qc_metrics`.
- `drop_mismatched_feature_matrices`: see the
  [feature-matrix rule](#feature-matrix-rule).

No `join` parameter: the rules below require the same features, `obs`
columns, layers and `obsm` keys in every table, so an inner and an outer join
give the same table. `concat` passes `join="inner"` to `anndata.concat`.

Raises `TypeError` for a table that is not an AnnData or that holds backed
matrices, and `ValueError` for any rule below. Every check runs before the
table is built, and the inputs are not modified.

Every error says what differs, where, and how to fix it: it names the
component or column and the tables, by key or position, gives the values that
differ, and states the fix. For example: "obs column 'n_counts' is int64 in
tables['s1'] and float64 in tables['s2']; cast it to one dtype in every table
before concatenating, for example with
tables['s1'].obs['n_counts'].astype('float64')."

Module `src/harpy/table/_concat.py`, exported in `src/harpy/table/__init__.pyi`
next to `validate_table`. Not `hp.tb.io`: no storage is involved. Not
`hp.tb.pp`: it changes structure, not values.

## Usage

```python
tables = {sample: sdata.tables[f"table_{sample}"] for sample in samples}
adata = hp.tb.concat(tables, label="sample", uns_merge="unique")
sdata = hp.tb.io.add_table(
    sdata,
    adata,
    output_table_name="table",
    region=adata.uns["spatialdata_attrs"]["region"],
)
hp.tb.validate_table(sdata, "table")
```

## Contract

### Rows and annotation

- Every table has a SpatialData annotation (`_spatialdata_annotation` in
  `src/harpy/table/_utils.py`), with the same `region_key` and
  `instance_key`.
- Each table passes the checks of `validate_table` on its annotation that
  need no `sdata`: region and instance columns present, without missing
  values; unique `(region, instance)` pairs; declared regions equal to the
  observed ones.
- No region is in two tables, as `spatialdata.concatenate` requires: each
  table has its own canonical and point-aggregation entry for a region, and
  two entries for one region are not merged.
- The instance column has the same dtype in every table, as every `obs`
  column has (see [`obs`](#obs)): pandas would cast `uint64` identifiers with
  `int64` ones to `float64`.
- `obs_names` are unique across tables; `anndata.concat` only warns, and keeps
  the duplicates. The
  index of `aggregate_points`, `<instance>_<labels element>_<token>`, holds the
  labels element, so tables of different regions do not collide.
- `uns["spatialdata_attrs"]` is rebuilt, with the regions in table order and,
  within a table, in its declared order.

### `obs`

- The same columns in every table, otherwise raise, naming the columns and
  the tables without them. `anndata.concat` would drop them silently with
  `join="inner"`, or fill them with missing values with `join="outer"`, which
  changes their dtype. Their order may differ: `anndata.concat` matches them
  by name, and the result follows the first table's order.
- Each `obs` column has the same dtype in every table. Categorical columns
  compare as categorical: their categories may differ, and their union is
  kept, as `anndata.concat` does. Otherwise raise, naming the column, the
  tables and their dtypes, and say to cast the column to one dtype before
  concatenating. Pandas would cast silently: `int64` with `float64` gives
  `float64`, `int64` with `uint64` gives `float64`, `bool` with `int64` gives
  `int64`, and a categorical with a string column gives `object`. Harmless
  widenings, such as `int32` with `int64`, raise too: one `astype` fixes them,
  and the rule stays one sentence.
- The `.obs` columns of `hp.tb.pp.calculate_qc_metrics` stack: they are per
  row. Columns computed per table, such as a `leiden` clustering run on each
  table, stack too, but their values do not compare across tables: the
  docstring says to recompute them after concatenating, as for `X_pca`.

### `var`

- Identical `var_names`, in the same order, otherwise raise: an outer join
  would invent zero counts for features a table did not measure. In ordinary
  mode, `aggregate_points` keeps the features each table's points contain, so
  tables of one panel can differ; the error says so, and points to class-aware
  aggregation or to subsetting the tables to their shared features.
- `var` columns are kept only if identical in every table (`merge="same"`);
  the others are dropped, with a log message naming them and saying why. The
  `.var` statistics of `calculate_qc_metrics` describe one table, so they
  drop.

### `X` and layers

- The same layers in every table. AnnData 0.13 lists `X` in `adata.layers`
  under the key `None`; the comparison leaves it out.
- Backed tables are rejected: if `X`, a layer or an `obsm` matrix of any table
  is a Zarr array or a sparse-dataset handle, as `hp.tb.io.read_table` with
  `mode="backed"` and `hp.io.read_zarr` with `table_mode="backed"` give, raise
  `TypeError`, naming the table and the matrix, and say to read the table
  lazily, the default mode. `anndata.concat` fails on these handles.
- Each component, `X`, a layer or an `obsm` matrix, has the same format in
  every table: dense, CSR or CSC, or a DataFrame in `obsm`. Different
  components of one table may differ: a table with a CSR `X` and a CSC
  `obsm["embedding"]` concatenates with another such table. Otherwise raise,
  naming the component, the tables and their formats. Lazily,
  `anndata.concat` gives COO for mixed formats, which Harpy's matrix check
  (`_matrix_format`) rejects. In memory, it converts mixed formats, and also a
  CSC component, to CSR; lazily, a CSC component stays CSC.
- Each component, `X`, a layer or an `obsm` matrix, is lazy in every table or
  in memory in every table. Otherwise raise, naming the component and the
  tables, and say to compute it (`adata.X = adata.X.compute()`) or to store the
  table and read it lazily. Different components of one table may differ: a
  lazy `X` with an `obsm["X_pca"]` computed on purpose concatenates with
  another such table.
- `concat` converts no matrix. `anndata.concat` handles each component
  separately: lazy in every table, it stays lazy, without computing; in
  memory in every table, it stays in memory. Mixed, `anndata.concat` would
  compute the sparse Dask arrays and wrap the dense in-memory matrices as
  Dask arrays, which the rule above avoids.

### `obsm`

- The same keys in every table, each matrix with the same number of columns.
- Registered matrices follow the
  [feature-matrix rule](#feature-matrix-rule), and `spatial_canonical` the
  [canonical-center rule](#canonical-center-rule).
- Other matrices are stacked. Matrices computed per table, such as `X_pca`
  or `X_umap`, do not compare across tables: the docstring says to delete them
  before concatenating, and to recompute them afterwards. Embeddings of one
  model, such as the `obsm["embedding"]` of `featurize`, do compare.

### `obsp`, `varp`, `varm` and `raw`

Dropped. `concat` logs one warning that names the dropped entries and the
tables that hold them, and says why: a table's graphs (`obsp`), feature
loadings (`varp`, `varm`) and snapshot before normalization (`raw`) describe
that table alone, and mean nothing for the combined table. The warning also
says to concatenate count tables, and to compute these afterwards.

### `uns`

Harpy owns these entries. They are taken out before `anndata.concat`, so that
`uns_merge` never applies to them, and merged by the rules below:

| Entry                                           | Rule                                                                                                       |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `spatialdata_attrs`                             | rebuilt, see [rows and annotation](#rows-and-annotation)                                                   |
| `uns[feature_matrices_key]`                     | [feature-matrix rule](#feature-matrix-rule)                                                                |
| `point_aggregation`                             | [point-aggregation rule](#point-aggregation-rule)                                                          |
| `spatial_coordinates["spatial_canonical"]`      | [canonical-center rule](#canonical-center-rule); other entries of `spatial_coordinates` follow `uns_merge` |
| `<column>_colors` of a categorical `obs` column | [color rule](#color-rule)                                                                                  |

Every other entry goes to `anndata.concat` with `uns_merge`, including a
`*_colors` entry without a categorical `obs` column. As checked:

| `uns_merge`      | Keeps                                                                         |
| ---------------- | ----------------------------------------------------------------------------- |
| `None` (default) | nothing                                                                       |
| `"same"`         | entries equal in every table                                                  |
| `"unique"`       | entries on which the tables that have them agree, such as `uns["qc"][sample]` |
| `"first"`        | each entry from the first table that has it, also when the tables disagree    |
| `"only"`         | entries that appear in one table only                                         |

- Nested dictionaries are merged key by key, so a strategy applies to each
  sub-entry. A list and an equal NumPy array compare equal, so tables in
  memory and tables read from storage, which hold arrays where the others hold
  lists, merge as expected.
- `concat` logs the top-level `uns` keys of the inputs that are not in the
  result, and says to pass `uns_merge` to keep them. Sub-entries dropped inside
  a kept entry are not listed.
- `uns_merge` keeps records as they are, also those of per-table analyses:
  with `"same"`, `"unique"` or `"first"`, scanpy's `uns["neighbors"]` survives
  and still names `obsp` graphs that `concat` drops, and `"first"` keeps the
  first table's `uns["pca"]["variance"]`. The docstring says to recompute these
  analyses after concatenating. `concat` does not drop such records: that would
  need scanpy's record layout in Harpy.

### Feature-matrix rule

- Each table's registry passes `_validate_feature_matrices`
  (`src/harpy/table/_validation.py`): metadata and shapes, no values.
- A registered matrix is registered in every table, with equal records. The
  whole record is compared, not only `feature_columns`: writers add their own
  fields, such as the source channels of `add_feature_matrix`. Records read
  from storage hold NumPy arrays where records in memory hold lists, so arrays
  are compared as lists.
- Except the per-region fields of a record of `add_feature_matrix`
  (`source_kind == "harpy_add_feature_matrix"`): its parallel lists
  `source_label`, `source_image` and `coordinate_system`, one entry per labels
  and image pair, differ between samples. They are merged per region, keyed by
  `source_label`: each table contributes the entries of the regions it
  declares, so entries of regions it no longer declares drop out, as in the
  canonical-center rule, and the merged lists follow the result's region
  order. Every other field of the record is compared, among them
  `feature_columns`, `source_channels`, `features` and `dtype`. This is how
  `add_feature_matrix` merges a record when it adds pairs to an existing
  matrix (`_merge_feature_matrix_metadata`). The merge lives in
  `src/harpy/table/_add_feature_matrix.py`, which owns the record layout, and
  `concat` calls it.
- Records of other writers, `aggregate_points` and code of the user, have no
  per-region fields: `aggregate_points` keeps its sources in
  `point_aggregation`. They are compared whole.
- Otherwise raise, naming the matrices and the tables where they differ or
  are missing. With `drop_mismatched_feature_matrices=True`, these matrices and
  their records are dropped instead, with a log message naming them. Padding
  missing columns with zeros is no option: it would distort
  `n_points_per_panel_feature` and the detected-feature counts. For example, a
  per-sample Grubbs test can keep 9 negative probes in one table and 10 in
  another.
- A registry left empty is removed.

### Canonical-center rule

- Every table has both canonical components, the matrix and its record, or
  none has either; otherwise raise and point to `hp.tb.add_canonical_centers`.
- Each table passes `_validate_canonical_centers_without_sources`, the checks
  of `validate_canonical_centers` that need no `sdata`, from the canonical-cache
  refactor: the matrix's shape, dtype and finite values; the strict parse of
  the record, with an entry for every declared region; z = 0 for 2D labels.
  It assumes a valid annotation, which the annotation's row checks above
  establish, and returns the entries of the declared regions. Only the
  comparison of each entry with its labels element needs `sdata`, and
  `validate_table` makes it after storing.
- The merged record holds each table's entries for its declared regions,
  built with the private schema helpers that `aggregate_points` uses, in
  schema version 2. Entries of regions a table no longer declares drop out,
  so a table split per region and concatenated back works; a union of whole
  records would hold each region twice. A version-1 input fails the check
  above, whose message says to rebuild its centers with
  `hp.tb.add_canonical_centers(..., overwrite=True)`.
- A table subset after its centers were computed, by per-cell QC for example,
  concatenates: the centers of the rows kept do not change, and no record ties
  a region to its rows.

### Point-aggregation rule

- Every table has the record, or none has; otherwise raise.
- `schema_version`, `source_kind`, `feature_key` and `feature_class_key`
  (present or absent) are equal in every table, so ordinary and class-aware
  tables do not mix.
- `regions` holds each table's entries for its declared regions. A subset
  table keeps its whole record, so entries of regions it no longer declares
  drop out, as in the canonical-center rule.
- The record is provenance, which no validation reads; nothing else is
  checked.

### Color rule

Scanpy stores the colors of a categorical `obs` column in
`uns[f"{column}_colors"]` and maps them by position to the column's
`.cat.categories`, unused categories included, not to the values that occur
(`np.unique` would drop unused categories and sort the rest).
`anndata.concat` merges these entries with `uns_merge`, independently of the
union of the categories, whose order it chooses: natural sort when the
categories differ, in the cases checked. `concat` merges them by category name
instead:

- A table's palette is used only if its length equals the number of the
  column's `.cat.categories` in that table, unused categories included, the
  rule AnnData applies when it subsets a table. Otherwise it is ignored, with a
  warning.
- Each category takes its color from the tables that have one for it. When
  tables give one category different colors, the first table in table order
  wins, with a warning naming the category, the tables and the colors. Colors
  are compared as stored strings, so `"red"` and `"#ff0000"` conflict.
- The merged palette follows the result's `.cat.categories`, read after
  `anndata.concat`, so it is correct whatever order AnnData chose. Their union
  keeps unused categories, which get a color like the others.
- If a category gets no color, because no table that lists it in its
  `.cat.categories` has a usable palette, the merged palette is dropped, with
  a warning naming the categories without a color: scanpy assigns its default
  colors on the first plot. No color is invented.
- One info message per column names the column and the tables whose palettes
  were merged.

Columns computed per table, such as a `leiden` clustering, get a merged
palette too, but their categories do not compare across tables: recompute the
column and its palette after concatenating.

## Implementation plan

The canonical-cache refactor ([canonical_cache.md](canonical_cache.md)) landed
in #283 and provides `_validate_canonical_centers_without_sources`.

### Step 1: annotation checks without `sdata`, no change in behaviour

- Split `_validate_table_annotation` (`src/harpy/table/_validation.py`) into
  the row checks, on the AnnData only, and the check that the regions exist in
  `sdata`. `validate_table` calls both. The module docstring asks to reuse
  shared checks rather than add new validation blocks.
- Tests: the existing validation tests, unchanged.

### Step 2: `concat`

In `src/harpy/table/_concat.py`, in this order:

1. Normalize `tables` to names and tables; raise for none. Read each table's
   annotation with `_spatialdata_annotation(table, function="concat")`.
2. Per table: no backed matrix in `X`, layers or `obsm`; the annotation's row
   checks (`_validate_table_annotation_without_sdata`);
   `_validate_feature_matrices`; and
   `_validate_canonical_centers_without_sources` if canonical components are
   present.
3. Across tables: annotation keys, regions, `obs_names`, `obs` columns and
   their dtypes, `label`, `var_names`, layers, `obsm` keys and widths, the
   format of each component (`_matrix_format`) and whether it is lazy.
4. Merge the records: feature matrices, with the matrices to drop and the
   per-region fields of `add_feature_matrix` records merged by region;
   canonical centers; point aggregation.
5. Prepare each table without copying its matrices, as `add_table` does: no
   Harpy-owned `uns` entries, no dropped matrices, no `obsp`, `varp`, `varm`
   or `raw`.
6. `anndata.concat(prepared, join="inner", merge="same", uns_merge=uns_merge,
label=label)`, with a mapping passed as a mapping.
7. Set `spatialdata_attrs` and the merged records; merge the palettes of the
   categorical `obs` columns by the [color rule](#color-rule), in the result's
   category order; check with `TableModel.validate`, and log what was dropped
   or merged, and why, including the top-level `uns` keys that `uns_merge`
   dropped.

### Step 3: documentation

- `docs/api.md`: `tb.concat` under Table, "Processing and analysis", next to
  `tb.validate_table`.
- `docs/development/metadata.md`, "Keeping records consistent":
  `hp.tb.concat` merges the table records, which `anndata.concat` and
  `spatialdata.concatenate` drop.
- The `hp.tb.concat` docstring: each input's declared regions must equal its
  observed ones, as SpatialData requires. Subsetting rows does not update the
  annotation: AnnData drops the unused categories of `obs[region_key]`, but
  `uns["spatialdata_attrs"]["region"]` keeps every region, and `concat` and
  `validate_table` raise "declares regions without observations". This holds
  on both sides of `concat`.

  Before concatenating, a table subset to fewer regions, by per-cell QC for
  example, needs its annotation updated:

  ```python
  subset = adata[adata.obs["region"] == "cells_a"].copy()
  subset.uns["spatialdata_attrs"]["region"] = ["cells_a"]
  ```

  After concatenating, a subset of the result to one or a few regions needs
  the same. `add_table` rebuilds the annotation from its `region` argument, so
  pass the subset's regions, not the old list:

  ```python
  subset = adata[adata.obs["region"].isin(["cells_a", "cells_b"])].copy()
  sdata = hp.tb.io.add_table(
      sdata,
      subset,
      output_table_name="subset",
      region=list(subset.obs["region"].cat.categories),
  )
  ```

  The canonical, point-aggregation and `add_feature_matrix` records of the
  subset keep the entries of the removed regions: validation ignores them, and
  a later `concat` drops them. `docs/canonical_centers.md` shows the same step
  for canonical centers.

- The `hp.tb.concat` docstring: the palettes of categorical `obs` columns are
  merged by category name, not by `uns_merge`; a palette with a category
  without a color is dropped, and scanpy then assigns its default colors.
- The `hp.tb.concat` docstring: `uns_merge` keeps the other `uns` records as
  they are, also those of per-table analyses, such as PCA, neighbors, UMAP or
  a clustering: recompute these after concatenating.

### Step 4: follow-up, lazy-matrix helpers in `_storage`, no change in behaviour

`concat` does not depend on this step: the helpers fit better in `_storage`.

- Move `_lazy_matrix` from `src/harpy/table/io/_write_by_region.py` to
  `src/harpy/_storage/_anndata.py`, next to the readers' block helpers
  (`_sparse_block_length`, `_auto_sparse_block_length`, `_dense_lazy_chunks`):
  it turns a matrix into a Dask array with the blocks of Harpy's lazy reads.
  Move with it `_block_length`, which it calls, and `_matrix_format`, which
  gives the format it takes. The regional writer, `add_feature_matrix`, the
  component writer and `concat` import them from `_storage`.
- Tests: the existing tests, unchanged.

## Tests

In `src/harpy/_tests/test_table/test_concat.py`. Fixture:
`backed_class_aware_sdata`, with two labels and points pairs; `aggregate_points`
with `expression_class="Endogenous"` once per pair, into `table_a` and
`table_b`.

1. Round trip, with `table_b` subset to part of its rows first, as per-cell
   QC does: `concat`, `add_table`, then `validate_table` passes. Rows and
   regions are united, the feature-matrix registry is kept once, the regions
   of the canonical and point-aggregation records are united, and `label`
   names each row's table. An `obsp` graph is dropped, with the warning. A
   morphology feature matrix of `add_feature_matrix`, which needs no image,
   computed on each table, is kept: its merged record lists both tables'
   `source_label`, `source_image` and `coordinate_system` in the result's
   region order.
2. Lazy stays lazy: the stored tables, read lazily, give Dask `X` and `obsm`
   matrices without computing. An `obsm` matrix computed in every table stays
   in memory.
3. A mismatched registered matrix raises; with
   `drop_mismatched_feature_matrices=True`, it is dropped with its record.
4. `uns_merge="unique"` keeps records keyed per table, and Harpy's records
   are not merged by it. With the default `uns_merge=None`, the dropped
   top-level `uns` keys are logged.
5. Colors: palettes `{B: red, T: blue}` and `{M: green, T: blue}` of a
   categorical `obs` column give `{B: red, M: green, T: blue}`, in the result's
   category order, whatever `uns_merge` is. A category with two colors keeps
   the first table's, with a warning. A table that holds a category but has
   no palette drops the merged palette, with a warning.
6. One parametrized test for the other rules that raise: a backed table, a
   shared region, different `var_names`, different `obs` columns, layers or
   `obsm` keys, an `obs` column with different dtypes (the instance column as
   `int64` and `uint64`; a `bool` and an `int64` column), while categorical
   columns with different categories pass, a component that is CSR in one table
   and CSC in another, a component that is lazy in one table and in memory in
   another, a `label` that is an `obs` column, duplicate `obs_names`, and
   canonical centers in one table only.
7. In `src/harpy/_tests/test_table/test_add_feature_matrix.py`, next to the
   module that owns the record layout: a unit test of the per-region merge of
   `add_feature_matrix` records. A record that still lists a region its table
   no longer declares contributes only the declared regions, the merged lists
   follow the given region order, and records whose `source_channels` differ
   raise.

## Departures from the issue

1. The result is not checked with `validate_table`, which takes
   `(sdata, table_name)` and checks the regions and labels in `sdata`. `concat`
   runs the checks that need no `sdata` on each table, so the result is valid
   by construction; `validate_table` runs after `add_table`.
2. `feature_matrices_key` is added, as in `validate_table`; otherwise a
   registry under another key would be dropped as an entry Harpy does not own,
   leaving its matrices unregistered.
3. `drop_mismatched_feature_matrices`, from the issue's comment, compares
   whole records.
4. `obs` columns must be the same in every table; the issue only stacks them.
5. `varm` and `raw` are dropped, like `obsp` and `varp`.
6. Added checks: the dtype of every `obs` column, a `label` that collides,
   duplicate `obs_names`.
7. A component that is lazy in one table and in memory in another raises;
   `concat` converts no matrix.
8. Unregistered `obsm` matrices follow the issue's rule (same keys, stacked),
   with a docstring note, rather than being dropped: embeddings of one model
   compare across tables.
9. The palettes of categorical `obs` columns are merged by category name, by
   Harpy's [color rule](#color-rule), rather than by `uns_merge`.

## Out of scope

- A storage-level convenience in `hp.tb.io`: `concat` with `add_table`
  covers it.
- A `join` parameter, an outer join on `var`, and `index_unique` for colliding
  `obs_names`.
- Backed tables: rejected, rather than wrapped as Dask arrays. Read them
  lazily.
- A component that is lazy in one table and in memory in another: rejected,
  rather than converted. Wrapping its in-memory matrices with `_lazy_matrix`
  could be added later without breaking anything, as it only accepts what is
  rejected now.
- Filling the colors of categories without one, for example from scanpy's
  default palettes: the merged palette is dropped instead.
