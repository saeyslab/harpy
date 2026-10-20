# Table QC metrics in the table: `hp.tb.pp.calculate_qc_metrics`

## Status

**Planned, in two steps.** It builds on the amendment of Slice 11j of
[allocation.md](allocation.md), "every row of the selected regions", which is
implemented.

`hp.qc.summarize_table` describes a table's instances and features from its
count matrices, but writes nothing to the table: its results stay in a
`TableSummary`. scanpy's `calculate_qc_metrics` computes its metrics and adds
them to `.obs` and `.var` in one call, where they can be plotted, filtered on,
or written to the store. This adds that call for Harpy's table summary:

```python
adata = sdata.tables["table"]
# In place, as scanpy.pp; the summary is returned for the plots.
summary = hp.tb.pp.calculate_qc_metrics(adata, obsm_keys=["negative", "system_control"])
hp.qc.instance_scatter(summary, x=("x", "points"), y=("negative", "pct_points"))
hp.tb.io.add_table_updates(sdata, table_name="table")  # persist, for a store-backed sdata
```

**One call, not a summary added afterwards.** A separate
`hp.tb.pp.add_table_summary(adata, summary)` was considered: it would have to
check that the table is still the summary's, its rows by `.obs` name with
their identity and `var_names`, and that the summary covers every region, as
`.obs` needs a value for every row. One call computes from the table and
writes to the same table, so neither check is needed. `hp.qc.summarize_table`
stays the read-only computation, for a summary without writing, also of
selected regions.

## Step 1: the table summaries take the table

```python
hp.qc.summarize_table(
    adata, *, layer=None, obsm_keys=(), feature_names=None,
    feature_matrices_key="feature_matrices", region=None,
) -> TableSummary
hp.qc.summarize_table_by_feature(adata, *, summary, features) -> FeatureTableSummary
```

in place of `summarize_table(sdata, table_name, …)` and
`summarize_table_by_feature(sdata, *, summary, features)`.

- **Why:** both read only the table: its matrices, `var`, `.obsm`, the
  registry in `.uns` and the SpatialData annotation in
  `uns["spatialdata_attrs"]`, against which `region` is checked. `sdata`
  served only to look the table up. The points side keeps `sdata`, as
  `summarize_points` needs the points element's transformations and the panel
  in `sdata.attrs`. The table APIs then all take the table itself, as
  `hp.tb.pl` (Slice 11l) and `hp.tb.pp` do, and as scanpy does, and
  `calculate_qc_metrics` calls `summarize_table` directly.
- **The SpatialData annotation stays required,** for the regions and the
  identity columns: a table without `uns["spatialdata_attrs"]` raises, as
  now. The annotated elements need not be present, as now.
- **`TableSummaryMetadata.table_name` goes:** an `AnnData` does not know its
  name in `sdata`. The record keeps `region`, `layer`, `obsm_keys` and
  `feature_matrices_key`.
- **`summarize_table_by_feature`** takes the table instead of finding it
  through `summary.metadata.table_name`. Its checks that the table is still
  the summary's stay: the classes, the names, and the rows by `.obs` name with
  their region and instance id. They catch another table as well as a changed
  one.
- **Messages** name the matrix, such as `adata.X` or `adata.obsm['negative']`,
  without the table's name.
- **No backward compatibility:** the signatures change without a
  deprecation, as the functions are young.
- **Updates:** the docstrings and examples of both functions, of
  `TableSummary`, `FeatureTableSummary`, `TableSummaryMetadata` and of the
  instance plots, `(sdata, "table")` becoming `(sdata.tables["table"])`; and
  the text of Slices 11j and 11k in [allocation.md](allocation.md): the
  signatures, the metadata and "Only the table".

**Tests,** in `src/harpy/_tests/test_qc/test_table/`: the fixtures pass the
table instead of a `SpatialData` object; a table outside any `SpatialData`
object is summarized; the metadata has no `table_name`; and the tests of a
table changed since the summary also cover `summarize_table_by_feature` on
another table.

## Step 2: `hp.tb.pp.calculate_qc_metrics`

```python
hp.tb.pp.calculate_qc_metrics(
    adata, *, layer=None, obsm_keys=(), feature_names=None,
    feature_matrices_key="feature_matrices", overwrite=False,
) -> TableSummary
```

- **`adata`:** the table, in memory, lazy or backed: only `.obs` and `.var`
  change.
- **`layer`, `obsm_keys`, `feature_names`, `feature_matrices_key`:** as in
  `summarize_table`, which this function calls.
- **No `region`:** every region the table annotates is summarized, as `.obs`
  needs a value for every row.
- **`overwrite`:** `False` raises if a column to add exists already, naming
  the columns; `True` replaces them.
- **Returns the `TableSummary`** it computed, for the plots and the per-class
  and per-region statistics, which are not added to the table. This differs
  from the normalisations of `hp.tb.pp` and from scanpy's in-place calls,
  which return `None`; the docstring says so.
- **The name follows scanpy,** for discoverability; the columns do not, as
  Slice 11j decided for `TableSummary`. The docstring maps them to scanpy's.

### `.obs`: the measures per instance

The measure columns of `summary.per_instance`, under their names, for every
row of the table, as the summary covers every region:

- `n_<class>_points` (uint64) and `n_<class>_detected_features`, per class;
- `pct_<class>_points`, with two classes or more.

A row without points in any class read has 0 points and 0 detected features,
and a missing share (NaN), as 0 / 0 has no value. The region and instance
columns of `per_instance` are the table's own and are not written.

**The rate per panel feature,** computed here, not stored in the summary:

- `n_<class>_points_per_panel_feature` (float64), per class:
  `n_<class>_points / summary.panel_feature_counts[<class>]`, the class's
  points divided by its number of features in the matrices read. For a CosMx
  table: `n_x_points_per_panel_feature`, `n_negative_points_per_panel_feature`
  and `n_system_control_points_per_panel_feature`.
- **Why here:** with the rates in `.obs`, users need not know what to divide by
  what; the denominator is the class's number of features in the matrices read,
  not the features detected, and comes from the summary. The values equal
  those that `instance_histogram` and `instance_scatter` plot with
  `normalization="per_panel_feature"`, so `.obs` and the plots agree.
- **Not in `TableSummary`:** its frames keep counts only, as Slice 11j of
  [allocation.md](allocation.md) decided, symmetric with the points side,
  whose plots also normalize at display time. `.obs` is the convenience layer,
  where the per-cell value is filtered on and plotted, as `pct_<class>_points`
  is.
- **The name** follows the naming rule of Slice 11l: `per_panel_feature` names
  the normalization, a class's points divided by its panel feature count.

**The docstring advises comparing rates on aggregates rather than per cell.** A
foreground/background ratio per cell, such as
`n_negative_points_per_panel_feature / n_x_points_per_panel_feature`, is
noisy, as negative points are sparse, and undefined for a cell without points
in `X`. Taken per region, it is stable:

```python
obs = sdata.tables["table"].obs
per_region = obs.groupby("fov_labels", observed=True)[
    ["n_x_points_per_panel_feature", "n_negative_points_per_panel_feature"]
].mean()
per_region["background_to_signal"] = (
    per_region["n_negative_points_per_panel_feature"] / per_region["n_x_points_per_panel_feature"]
)
```

No ratio column is added.

Compared with `scanpy.pp.calculate_qc_metrics`, these are the counterparts of
`total_counts`, `n_genes_by_counts` and `pct_counts_<qc_var>`, under other
names, so that both can be in `.obs`.

### `.var`: the statistics per feature of `X` or the layer

When `X` or a layer was read, the matrix aligned with `.var`, the rows of
`summary.per_feature` of its class, under their names:

- `n_points` (uint64), `within_class_fraction`, `n_instances_without_feature`,
  `pct_instances_without_feature`, `mean_points_per_instance` and
  `std_points_per_instance`.

The population column `n_instances`, the same for every feature, and
`feature_class`, the same class for every row, are not written. These are the
counterparts of scanpy's `total_counts`, `n_cells_by_counts`,
`pct_dropout_by_counts` and `mean_counts`, under other names, with the same
values, as both count over every row. A table without `X`, summarized from
`.obsm` matrices alone, gets nothing in `.var`.

### Not added

- the features of the `.obsm` matrices, which have no `.var` axis;
- the per-class and per-region statistics, `per_class`, `per_region_class`
  and `per_region_feature`, which are not aligned with rows or features;
- scanpy's `log1p_` columns and `pct_counts_in_top_<n>_genes`.

The first two stay in the returned summary.

## Checks, before anything is written

- **The checks of `summarize_table`,** which runs first.
- **Existing columns:** with `overwrite=False`, a column to add that `.obs` or
  `.var` already holds raises, naming the columns.

The columns are computed before any is written, so a failed check leaves the
table unchanged.

## Documentation

- `docs/api.md` lists `tb.pp.calculate_qc_metrics` under Preprocessing, and its
  introduction widens from "normalisations that scanpy does not provide" to
  the in-place additions of `hp.tb.pp`, with `add_table_updates` to persist.
- The docstring of `hp.qc.summarize_table` points to `calculate_qc_metrics` for
  adding its results to the table.

## Tests

In `src/harpy/_tests/test_table/test_pp/`, on the dummy tables of the tests of
`summarize_table`:

- `.obs`: the values equal `per_instance` of the returned summary, row by row;
  a row without points has 0 points and 0 detected features and a missing
  share; the dtypes; no share column with one class;
- the rates: each `n_<class>_points_per_panel_feature` equals
  `n_<class>_points / summary.panel_feature_counts[<class>]`, 0 for a row
  without points, and the values that `instance_scatter` plots with
  `normalization="per_panel_feature"`;
- `.var`: the values equal the rows of `per_feature` of `X` or the layer, in
  `var_names` order; nothing is added for a table summarized from `.obsm`
  alone;
- the returned summary equals `summarize_table` of the same table and
  arguments;
- existing columns without and with `overwrite`; a failed check leaves `.obs`
  and `.var` unchanged;
- a store-backed table: after `calculate_qc_metrics` and
  `hp.tb.io.add_table_updates`, the reopened table holds the columns.

## Verification

`summarize_table` and `summarize_table_by_feature` take the table, without
`sdata`, and their results are unchanged but for the metadata without
`table_name`. After `calculate_qc_metrics`, every row of the table has each
class's points, detected features and share, equal to `per_instance` of the
returned summary, and each class's points per panel feature, its points
divided by `summary.panel_feature_counts`; the features of `X` or the layer
have the statistics of `per_feature`; nothing else in the table changes; and
`hp.tb.io.add_table_updates` persists the new columns.
