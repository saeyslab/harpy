# `hp.tb.concat`: design

Source: issue #277 and its comment on feature matrices. This document fixes
the contract and the implementation plan; where it departs from the issue is
listed under [Departures from the issue](#departures-from-the-issue).

Depends on the canonical-cache refactor ([canonical_cache.md](canonical_cache.md)),
its own PR, which lands first and closes #279 and #282. Without #279, a table
subset by per-cell QC fails canonical validation and could not be
concatenated.

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
  not own; `None` drops them. `"unique"` keeps records keyed per table, such as
  `uns["qc"][sample]`.
- `feature_matrices_key`: key in `uns` of the feature-matrix registry, as in
  `validate_table`, `summarize_table` and `calculate_qc_metrics`.
- `drop_mismatched_feature_matrices`: see the
  [feature-matrix rule](#feature-matrix-rule).

Raises `TypeError` for a table that is not an AnnData, and `ValueError` for any
rule below. Every check runs before the table is built, and the inputs are not
modified.

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
- The instance column has the same dtype in every table, so that pandas does
  not cast the identifiers: `uint64` with `int64` gives `float64`.
- `obs_names` are unique across tables; `anndata.concat` does not check. The
  index of `aggregate_points`, `<instance>_<labels element>_<token>`, holds the
  labels element, so tables of different regions do not collide.
- `uns["spatialdata_attrs"]` is rebuilt, with the regions in table order and,
  within a table, in its declared order.

### `obs`

- The same columns in every table, otherwise raise, naming the columns and
  the tables without them. `anndata.concat` would drop them silently with
  `join="inner"`, or fill them with missing values with `join="outer"`, which
  changes their dtype.
- Categorical columns get the union of their categories, as `anndata.concat`
  does.
- The `.obs` columns of `hp.tb.pp.calculate_qc_metrics` stack: they are per
  row.

### `var`

- Identical `var_names`, in the same order, otherwise raise: an outer join
  would invent zero counts for features a table did not measure. In ordinary
  mode, `aggregate_points` keeps the features each table's points contain, so
  tables of one panel can differ; the error says so, and points to class-aware
  aggregation or to subsetting the tables to their shared features.
- `var` columns are kept only if identical in every table (`merge="same"`);
  the others are dropped, with a log message naming them. The `.var`
  statistics of `calculate_qc_metrics` describe one table, so they drop.

### `X` and layers

- The same layers in every table.
- If any matrix of any table is lazy or backed, every matrix of the result is
  a Dask array: backed handles are wrapped as `write_table` wraps them
  (`_prepare_anndata_value` in `src/harpy/_storage/_anndata.py`), in-memory
  matrices with `da.from_array`. `anndata.concat` computes Dask arrays that are
  mixed with in-memory ones. If every matrix is in memory, the result is in
  memory.

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

Dropped, with one log message naming what was dropped: a table's graphs,
loadings and snapshot before normalization mean nothing for the combined
table. Concatenate count tables, and normalize afterwards.

### `uns`

Harpy owns these entries. They are taken out before `anndata.concat`, so that
`uns_merge` never applies to them, and merged by the rules below:

| Entry                                      | Rule                                                                                                       |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------------------- |
| `spatialdata_attrs`                        | rebuilt, see [rows and annotation](#rows-and-annotation)                                                   |
| `uns[feature_matrices_key]`                | [feature-matrix rule](#feature-matrix-rule)                                                                |
| `point_aggregation`                        | [point-aggregation rule](#point-aggregation-rule)                                                          |
| `spatial_coordinates["spatial_canonical"]` | [canonical-center rule](#canonical-center-rule); other entries of `spatial_coordinates` follow `uns_merge` |

Every other entry goes to `anndata.concat` with `uns_merge`.

### Feature-matrix rule

- Each table's registry passes `_validate_feature_matrices`
  (`src/harpy/table/_validation.py`): metadata and shapes, no values.
- A registered matrix is registered in every table, with equal records. The
  whole record is compared, not only `feature_columns`: writers add their own
  fields, such as the source channels of `add_feature_matrix`. Records read
  from storage hold NumPy arrays where records in memory hold lists, so arrays
  are compared as lists.
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
- Each table passes `_validate_canonical_without_sources`, the checks of
  `validate_canonical_payload` that need no `sdata`, from the canonical-cache
  refactor: the matrix's shape, dtype and finite values; the strict parse of
  the record, with an entry for every declared region; z = 0 for 2D labels.
  It assumes a valid annotation, which the annotation's row checks above
  establish, and returns the entries of the declared regions. Only the source
  signatures of the labels need `sdata`, and `validate_table` checks them
  after storing.
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

## Implementation plan

The canonical-cache refactor ([canonical_cache.md](canonical_cache.md)) lands
first and provides `_validate_canonical_without_sources`.

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
2. Per table: the annotation's row checks, `_validate_feature_matrices`, and
   `_validate_canonical_without_sources` if canonical components are present.
3. Across tables: annotation keys, regions, instance dtype, `obs_names`,
   `obs` columns, `label`, `var_names`, layers, `obsm` keys and widths.
4. Merge the records: feature matrices, with the matrices to drop;
   canonical centers; point aggregation.
5. Prepare each table without copying its matrices, as `add_table` does: no
   Harpy-owned `uns` entries, no dropped matrices, no `obsp`, `varp`, `varm`
   or `raw`; Dask matrices if any table has a lazy or backed one.
6. `anndata.concat(prepared, join="inner", merge="same", uns_merge=uns_merge,
label=label)`, with a mapping passed as a mapping.
7. Set `spatialdata_attrs` and the merged records, check with
   `TableModel.validate`, and log what was dropped.

### Step 3: documentation

- `docs/api.md`: `tb.concat` under Table, "Processing and analysis", next to
  `tb.validate_table`.
- `docs/development/metadata.md`, "Keeping records consistent":
  `hp.tb.concat` merges the table records, which `anndata.concat` and
  `spatialdata.concatenate` drop.

## Tests

In `src/harpy/_tests/test_table/test_concat.py`. Fixture:
`backed_class_aware_sdata`, with two labels and points pairs; `aggregate_points`
with `expression_class="Endogenous"` once per pair, into `table_a` and
`table_b`.

1. Round trip, with `table_b` subset to part of its rows first, as per-cell
   QC does: `concat`, `add_table`, then `validate_table` passes. Rows and
   regions are united, the feature-matrix registry is kept once, the regions
   of the canonical and point-aggregation records are united, and `label`
   names each row's table.
2. Lazy stays lazy: the backed tables give Dask `X` and `obsm` matrices, also
   when mixed with a table in memory.
3. A mismatched registered matrix raises; with
   `drop_mismatched_feature_matrices=True`, it is dropped with its record.
4. `uns_merge="unique"` keeps records keyed per table, and Harpy's records
   are not merged by it.
5. One parametrized test for the other rules that raise: a shared region,
   different `var_names`, different `obs` columns, layers or `obsm` keys,
   different instance dtypes, a `label` that is an `obs` column, duplicate
   `obs_names`, and canonical centers in one table only.

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
6. Added checks: the instance dtype, a `label` that collides, duplicate
   `obs_names`.
7. In-memory matrices are wrapped as Dask arrays when a table is lazy.
8. Unregistered `obsm` matrices follow the issue's rule (same keys, stacked),
   with a docstring note, rather than being dropped: embeddings of one model
   compare across tables.

## Out of scope

- A storage-level convenience in `hp.tb.io`: `concat` with `add_table`
  covers it.
- An outer join on `var`, and `index_unique` for colliding `obs_names`.
