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
    table_key: str | None = None,
    uns_merge: Literal["same", "unique", "first", "only"] | None = None,
    feature_matrices_key: str = _FEATURE_MATRICES_KEY,
    drop_mismatched_feature_matrices: bool = False,
) -> AnnData
```

- `tables`: at least one table. Errors name a table by its key in a mapping,
  otherwise by its position: `tables['s1']`, `tables[0]`.
- `table_key`: `obs` column naming each row's table: the keys of a mapping,
  or the positions as strings (`"0"`, `"1"`, ...) for a sequence, as
  `anndata.concat` does with its `label`. It must not be an existing `obs`
  column of any table: `anndata.concat` would overwrite it silently. Not
  named `label`, as `anndata.concat` names it: in Harpy, a label is an
  instance of a labels element.
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

Raises `TypeError` for a table that is not an AnnData, and `ValueError` for
any rule below. Every check runs before the
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
adata = hp.tb.concat(tables, table_key="sample", uns_merge="unique")
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
- Backed tables are accepted, as lazy tables. `aggregate_points` attaches its
  table backed: `X`, its class matrices and `obsm["spatial_canonical"]` are
  sparse-dataset handles or Zarr arrays (`_aggregation_writer.py`). So are the
  matrices of `hp.tb.io.read_table` with `mode="backed"` and `hp.io.read_zarr`
  with `table_mode="backed"`. `anndata.concat` fails on these handles, so
  `concat` turns each into the Dask array that a lazy read gives, with Harpy's
  lazy decoder (`_decode_anndata_element(..., mode="lazy")` in
  `src/harpy/_storage/_anndata.py`, with its default block sizes, as
  `read_table` reads lazily, and as `_lazy_matrix` does for backed values).
  This reads no matrix values and leaves the inputs unchanged. A Zarr array
  without AnnData encoding, which the decoder cannot read, raises, and the
  error says to wrap it with `dask.array.from_zarr`, as `_matrix_format` does.
- Each component, `X`, a layer or an `obsm` matrix, has the same format in
  every table: dense, CSR or CSC, or a DataFrame in `obsm`. Different
  components of one table may differ: a table with a CSR `X` and a CSC
  `obsm["embedding"]` concatenates with another such table. Otherwise raise,
  naming the component, the tables and their formats. Lazily,
  `anndata.concat` gives COO for mixed formats, which Harpy's matrix check
  (`_matrix_format`) rejects. In memory, it converts mixed formats, and also a
  CSC component, to CSR; lazily, a CSC component stays CSC.
- Each component, `X`, a layer or an `obsm` matrix, is lazy in every table or
  in memory in every table; a backed matrix counts as lazy, so a backed table
  and a lazy one concatenate into a lazy result. Otherwise raise, naming the component and the
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

| Entry                                           | Rule                                                                                                                  |
| ----------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `spatialdata_attrs`                             | rebuilt, see [rows and annotation](#rows-and-annotation)                                                              |
| `uns[feature_matrices_key]`                     | [feature-matrix rule](#feature-matrix-rule)                                                                           |
| `point_aggregation`                             | [point-aggregation rule](#point-aggregation-rule)                                                                     |
| `spatial_coordinates`                           | [canonical-center rule](#canonical-center-rule): only `spatial_canonical` is kept, as Harpy writes nothing else there |
| `<column>_colors` of a categorical `obs` column | [color rule](#color-rule)                                                                                             |

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
- Except `regions`, the per-region entries (Step 3): a record of
  `add_feature_matrix` (`source_kind == "harpy_add_feature_matrix"`) has
  `regions: {labels name: {"image_element": ..., "coordinate_system": ...}}`,
  which differs between samples. Each table contributes the entries of the
  regions it declares, so entries of regions it no longer declares drop out,
  as in the canonical-center rule. Every other field of the record is
  compared: `feature_columns`, `source_channels` and `features`. A record of
  schema version 1, as Harpy 0.5 wrote it, fails the check above.
- Records of other writers, `aggregate_points` and code of the user, have no
  `regions`: `aggregate_points` keeps its sources in `point_aggregation`.
  They are compared whole.
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
- The merged record holds each table's stored entries for its declared
  regions, which the check above parsed, in schema version 2. Entries of
  regions a table no longer declares drop out,
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

1. Normalize `tables` to names and tables; raise for none, and raise a
   `TypeError` for a table that is not an AnnData. A table without a
   SpatialData annotation raises in the per-table checks of item 2.
2. Per table: the annotation's row checks
   (`_validate_table_annotation_without_sdata`); `_validate_feature_matrices`;
   and `_validate_canonical_centers_without_sources` if canonical components
   are present.
3. Across tables: annotation keys, regions, `obs_names`, `obs` columns and
   their dtypes, `table_key`, `var_names`, layers, `obsm` keys and widths, the
   format of each component (`_matrix_format`, which reads the format of a
   backed matrix from its metadata) and whether it is lazy, a backed matrix
   counting as lazy.
4. Merge the records: feature matrices, with the matrices to drop and the
   per-region fields of `add_feature_matrix` records merged by region;
   canonical centers; point aggregation.
5. Prepare each table without copying its matrices, as `add_table` does: no
   Harpy-owned `uns` entries, no dropped matrices, no `obsp`, `varp`, `varm`
   or `raw`, and each backed matrix turned into a lazy one with
   `_decode_anndata_element(..., mode="lazy")`.
6. `anndata.concat(prepared, join="inner", merge="same", uns_merge=uns_merge,
label=table_key)`, with a mapping passed as a mapping.
7. Set `spatialdata_attrs` and the merged records; merge the palettes of the
   categorical `obs` columns by the [color rule](#color-rule), in the result's
   category order; check with `TableModel.validate`, and log what was dropped
   or merged, and why, including the top-level `uns` keys that `uns_merge`
   dropped.

### Step 3: follow-up, one layout for the per-region metadata

Three records keep metadata per region in `uns`: canonical centers, point
aggregation and the feature matrices of `add_feature_matrix`. The region names
repeat `obs[region_key]`, but the values do not: they name the labels, points
or image and the coordinate system each region came from, which neither the
matrix nor SpatialData's annotation records. AnnData neither subsets nor
concatenates `uns`, so `concat` merges these records itself. This step gives
them one layout, merged by one function. It is a breaking change of the
stored layout of `add_feature_matrix` records, and of the feature-matrix
schema version.

Current layouts:

- Canonical centers, `uns["spatial_coordinates"]["spatial_canonical"]`:
  `schema_version`, `axes`, and `regions: {labels name: {...}}`, with an entry
  for every declared region.
- Point aggregation, `uns["point_aggregation"]`: `schema_version`,
  `source_kind`, `feature_key`, `feature_class_key` (class-aware only), and
  `regions: {labels name: {"points_element": ..., "coordinate_system": ...}}`.
- `add_feature_matrix`, `uns["feature_matrices"][key]`: whole-matrix fields
  (`schema_version`, `source_kind`, `feature_columns`, `backend`, `dtype`,
  `source_channels`, `features`) and three parallel lists, `source_label`,
  `source_image` and `coordinate_system`, one entry per region the matrix was
  computed for.
- Other feature-matrix records, of `aggregate_points` and of the user's code:
  no per-region fields.

Each layout has its own merge: `_merged_canonical_record` and
`_merged_point_aggregation`, with `_POINT_AGGREGATION_HEADER`, in
`_concat.py`, and `_merge_feature_matrix_metadata`,
`_concat_feature_matrix_metadata`, `_record_sources`, `_with_sources` and
`_without_sources` in `_add_feature_matrix.py`. The parallel lists also lose
`None`: AnnData stores a list of `None` and strings as strings, so the record
of a morphology-only matrix, which has no image, reads back from Zarr with
`source_image == ["None"]`. A `None` dict value survives. The storage test of
`source_image` checks an in-memory table only.

**The rule.** The metadata of a matrix has two parts:

- whole-matrix fields, every field except `regions`, equal in every table
  that `concat` joins;
- `regions`, the per-region entries:
  1. keyed by labels element name, the SpatialData region;
  2. holding the regions with data: fewer than the table declares when a
     matrix was computed for some regions only, with NaN in the others; and,
     after a subset, regions the table no longer declares. Readers use the
     declared regions only;
  3. without a meaningful order: a Zarr round trip returns the keys sorted,
     not in the order written. Region order comes from `obs`;
  4. holding per-region values only: fields of the whole matrix, such as
     `source_channels` and `features`, stay at the top level;
  5. with values that survive a Zarr round trip: strings, integers, `None`,
     and lists or dicts of these;
  6. naming elements by their type, `points_element` and `image_element`,
     and a SpatialData coordinate system `coordinate_system`.

Changes:

- `add_feature_matrix` stores
  `regions: {labels name: {"image_element": image name or None, "coordinate_system": name}}`
  in place of the three lists. It drops `backend` and `dtype`, which nothing
  reads: `backend: "numpy"` is wrong for a lazy matrix, which
  `add_feature_matrix` itself attaches when `sdata` is backed, and `dtype`
  goes stale when the matrix is cast; `validate_table` checks neither, on
  purpose. Its other whole-matrix fields do not change.
- Two functions, in `src/harpy/table/_metadata.py`, next to the names of
  the metadata, merge one record across tables. `_comparable_metadata`
  moves there from `_utils.py`.
  `_metadata_difference` compares the fields other than `regions`, arrays
  as lists, and returns the first table whose fields differ, with those
  fields; `_merged_metadata` takes each table's entries for the regions it
  declares. A record without `regions` is compared and kept whole.
  `_differing_fields` moves there from `_concat.py`. Neither raises: each
  caller words its error, so the messages keep saying how to fix it:
  aggregate every table the same way for point aggregation,
  `drop_mismatched_feature_matrices` for feature matrices. They are two
  because `concat` finds the mismatched feature matrices before it merges
  the others. Canonical centers need no comparison: the parse of each table
  fixes their fields other than `regions`. `concat` uses them
  for canonical centers, point aggregation and every feature-matrix record,
  in place of the helpers above, and `_log_left_out_regions` reads each
  record's `regions`. Which tables must hold a record stays with the caller:
  every table or none for canonical centers and point aggregation, and
  `drop_mismatched_feature_matrices` for feature matrices. When
  `add_feature_matrix` updates some regions, it replaces their entries and
  keeps the others.
- Canonical centers do not change; they stay at schema version 2. They
  already follow the rule. Their entries repeat fixed values
  (`source_element_type`, `source_scale`, `coordinate_frame`), but these need
  no merge. The layout is a public contract, in `docs/canonical_centers.md`,
  which Spatiato reads, and version 2 chose recently that each entry states
  its frame ([canonical_cache.md](canonical_cache.md)). If a version 3 is
  needed for another reason, the fixed values can move to the top level then.
- Point aggregation does not change. Its `regions` is the only record of the
  points element and coordinate system each region was aggregated from.
- `_FEATURE_MATRIX_SCHEMA_VERSION` goes to 2, for every writer of a
  feature-matrix record. A record of version 1, as Harpy 0.5 wrote it, fails
  `validate_table`, `concat` and `hp.qc.summarize_table`, with a message to
  compute the matrix again with the function its `source_kind` names, and a
  regional update of `add_feature_matrix` rejects it as incompatible, with a
  message to remove the matrix with `hp.tb.io.remove_table_components` and
  compute it again. This also fails the class matrices of `aggregate_points`
  from Harpy 0.5, whose layout does not change: one version for the whole
  contract keeps it simple to reason about, and needs no code that
  recognizes the older layout. The older layout is not converted on read.
- Docs: `docs/development/metadata.md` states the rule, for new writers to
  follow, says that the class matrices of `aggregate_points` have no
  `regions` because their sources are in `point_aggregation`, once for them
  and `X`, and describes `regions` in place of the lists; so does the
  [feature-matrix rule](#feature-matrix-rule). In the `concat` docstring,
  `drop_mismatched_feature_matrices` compares the metadata whole except
  `regions`, in place of "except the sources of each region in the metadata
  of `add_feature_matrix`".
- Tests: test 7 becomes a test of the merge function; the `concat` and
  `add_feature_matrix` tests read `regions` and do not depend on its order; a
  morphology-only record read back from Zarr has `image_element` `None`; a
  record of schema version 1 raises in `validate_table`, `concat` and a
  regional update; tests that write records by hand write version 2.

### Step 4: documentation

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

### Step 5: follow-up, lazy-matrix helpers in `_storage`, no change in behaviour

`concat` does not depend on this step: the helpers fit better in `_storage`.

- Move `_lazy_matrix` from `src/harpy/table/io/_write_by_region.py` to
  `src/harpy/_storage/_anndata.py`, next to the readers' block helpers
  (`_sparse_block_length`, `_auto_sparse_block_length`, `_dense_lazy_chunks`):
  it turns a matrix into a Dask array with the blocks of Harpy's lazy reads.
  Move with it `_block_length`, which it calls, and `_matrix_format`, which
  gives the format it takes. The regional writer, `add_feature_matrix`, the
  component writer and `concat` import them from `_storage`.
- Tests: the existing tests, unchanged.

### Step 6: follow-up, metadata names in `_metadata.py`, no change in behaviour

The names of the table metadata in `adata.uns` are spread out:
`src/harpy/table/_metadata.py` holds the point-aggregation key, the schema
versions and the source kind of `aggregate_points`;
`src/harpy/utils/_keys.py` holds `_FEATURE_MATRICES_KEY`;
`_add_feature_matrix.py` holds its own `_SOURCE_KIND`; and the field names are
string literals in about ten modules, most in `_add_feature_matrix.py`,
`_aggregation_writer.py`, `_validation.py`, `_concat.py` and, in the merge
functions of Step 3, `_metadata.py` itself.

- `_metadata.py` defines every key, field name, source kind and schema
  version of the table metadata, in sections laid out as the stored
  metadata, and its docstring links to `docs/development/metadata.md`:
  - fields that several kinds share: `schema_version`, `source_kind`,
    `regions` and, in a `regions` entry, `coordinate_system`;
  - `uns["feature_matrices"]`: the key, the schema version and
    `feature_columns`; the source kind of `add_feature_matrix` and its
    `source_channels`, `features` and, in a `regions` entry,
    `image_element`;
  - `uns["point_aggregation"]`: the key, the schema version, the source kind
    of `aggregate_points`, `feature_key`, `feature_class_key` and, in a
    `regions` entry, `points_element`.
- `_FEATURE_MATRICES_KEY` moves there from `utils/_keys.py`, and
  `_SOURCE_KIND` from `_add_feature_matrix.py`, as
  `_ADD_FEATURE_MATRIX_SOURCE_KIND`. Code reads and writes the metadata
  through these names, not through string literals: a typo fails at import,
  and the uses of a field are found by its name. A stored name still changes
  only with a new schema version.
- Canonical centers keep their names in `harpy.table.canonical_centers`, the
  public import boundary, which exports `CANONICAL_OBSM_KEY` and
  `SPATIAL_COORDINATES_KEY` and has its own parser; `_metadata.py` points
  there.
- The names keep a leading underscore, as the existing constants.
- Out of scope: the feature panels in `sdata.attrs["harpy"]`, whose
  `feature_key` and `feature_class_key` belong to another record.
- Tests: the existing tests, unchanged. They keep string literals, which pin
  the stored format: a test that used a constant would still pass if the
  constant, and with it the stored format, changed by mistake.

## Tests

In `src/harpy/_tests/test_table/test_concat.py`. Fixture:
`backed_class_aware_sdata`, with two labels and points pairs; `aggregate_points`
with `expression_class="Endogenous"` once per pair, into `table_a` and
`table_b`, which it attaches backed. `table_a` has three instances, `table_b`
one. `add_feature_matrix` then adds a morphology matrix to each table, which
it attaches lazily.

1. Round trip, with `table_a` subset to part of its rows first, as per-cell
   QC does: `concat`, `add_table`, then `validate_table` passes. Both tables
   are loaded into memory first: subsetting loads backed matrices but keeps
   lazy ones lazy, so a subset of `table_a` would have a lazy morphology
   matrix and an in-memory `X`. Rows and
   regions are united, the feature-matrix registry is kept once, the regions
   of the canonical and point-aggregation records are united, and the
   `table_key` column names each row's table. An `obsp` graph is dropped, with the warning. A
   morphology feature matrix of `add_feature_matrix`, which needs no image,
   computed on each table, is kept: its merged record holds both tables'
   `regions` entries, with `image_element` `None`, read back from the store.
2. Lazy stays lazy: the backed tables of `aggregate_points` give Dask `X` and
   `obsm` matrices, and so do the same tables read lazily, and a backed table
   with a lazy one. The inputs keep their backed matrices. An `obsm` matrix
   computed in every table stays in memory.
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
6. One parametrized test for the other rules that raise: a Zarr array without
   AnnData encoding, a shared region, different `var_names`, different `obs` columns, layers or
   `obsm` keys, an `obsm` matrix with different widths, an `obs` column with different dtypes (the instance column as
   `int64` and `uint64`; a `bool` and an `int64` column), while categorical
   columns with different categories pass, a component that is CSR in one table
   and CSC in another, a component that is lazy in one table and in memory in
   another, a `table_key` that is an `obs` column, duplicate `obs_names`, and
   canonical centers in one table only.
7. A unit test of `_merged_metadata` and `_metadata_difference`
   (`src/harpy/table/_metadata.py`), in `test_concat.py`: a record that still
   holds a region its table no longer declares contributes only the declared
   regions, a record read from storage, with arrays, compares equal to one in
   memory, a differing field is reported with the table's position, and a
   record without `regions` is kept whole. Test 6 includes an
   `add_feature_matrix` record of schema version 1.

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
6. Added checks: the dtype of every `obs` column, a `table_key` that collides,
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
- Attaching the table of `aggregate_points` lazily, as `add_table` does,
  rather than backed: it would change that function, and `concat` accepts
  backed tables.
- A component that is lazy in one table and in memory in another: rejected,
  rather than converted. Wrapping its in-memory matrices with `_lazy_matrix`
  could be added later without breaking anything, as it only accepts what is
  rejected now.
- Filling the colors of categories without one, for example from scanpy's
  default palettes: the merged palette is dropped instead.
