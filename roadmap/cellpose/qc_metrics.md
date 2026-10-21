# Table QC metrics in the table: `hp.tb.pp.calculate_qc_metrics`

## Status

**Implemented, in three steps.** It builds on the amendment of Slice 11j of
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
# Persist, for a store-backed sdata: obs and var exist in the store, so replacing them needs overwrite=True.
hp.tb.io.add_table_updates(sdata, table_name="table", overwrite=True)
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

**Status: implemented.**

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
- **The table is checked to be an `AnnData`,** in both functions, before
  anything else: another type, such as a `SpatialData` object passed by the
  former habit, raises a `TypeError` that asks for the table, such as
  `sdata.tables[name]`, instead of an `AttributeError` from inside.
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
  without the table's name, so that the private helpers no longer take
  `table_name`, which they passed on for the messages only.
- **No backward compatibility:** the signatures change without a
  deprecation, as the functions are young.
- **Updates:** the docstrings and examples of both functions, of
  `TableSummary`, `FeatureTableSummary`, `TableSummaryMetadata` and of the
  instance plots, `(sdata, "table")` becoming `(sdata.tables["table"])`; and
  the text of Slices 11j and 11k in [allocation.md](allocation.md): the
  signatures, the metadata and "Only the table".

**Tests,** in `src/harpy/_tests/test_qc/test_table/`: the fixtures pass the
table instead of a `SpatialData` object, so that every test summarizes a table
outside any `SpatialData` object; the lazy and backed tests still read the
store with `hp.io.read_zarr` and pass its table; the metadata has no
`table_name`; a `SpatialData` object passed to `summarize_table` raises the
`TypeError`; and the tests of a table changed since the summary also cover
`summarize_table_by_feature` on another table.

## Step 2: `hp.tb.pp.calculate_qc_metrics`

**Status: implemented,** in `src/harpy/table/pp/_qc_metrics.py`. The column
names below are those after step 3, which moved every class label to the end
of the name; its table lists the names of step 2.

```python
hp.tb.pp.calculate_qc_metrics(
    adata, *, layer=None, obsm_keys=(), feature_names=None,
    feature_matrices_key="feature_matrices",
) -> TableSummary
```

- **`adata`:** the table, in memory, lazy or backed: only `.obs` and `.var`
  change.
- **`layer`, `obsm_keys`, `feature_names`, `feature_matrices_key`:** as in
  `summarize_table`, which this function calls.
- **No `region`:** every region the table annotates is summarized, as `.obs`
  needs a value for every row.
- **No `overwrite`:** columns that `.obs` or `.var` already hold under the
  names to add are replaced, without a warning, and the docstring says so.
  Columns are only written, never removed: those of other matrices or
  classes, from earlier calls, stay.
  In Harpy, `overwrite` concerns replacing an element of `sdata` or its store,
  as in `hp.tb.io.add_table` and `hp.tb.aggregate_points`; the functions of
  `hp.tb.pp` change the table in place without it, and so does scanpy's
  `calculate_qc_metrics`. A second call recomputes the same columns, so
  replacing them loses nothing; `normalize_by_size` warns before it replaces
  its record only because normalizing twice corrupts the matrix. Persisting
  is another matter: `obs` and `var` exist in the store, so
  `hp.tb.io.add_table_updates` replaces them only with `overwrite=True`, the
  use of `overwrite` that Harpy keeps; the docstring's example shows it.
- **Returns the `TableSummary`** it computed, for the plots and the per-class
  and per-region statistics, which are not added to the table. This differs
  from the normalisations of `hp.tb.pp` and from scanpy's in-place calls,
  which return `None`; the docstring says so.

**The name** keeps scanpy's `calculate_qc_metrics`, in line with the other
functions of `hp.tb.pp`, `normalize_by_size` and `normalize_by_quantile`,
which also take scanpy's verbs, and so that users who know scanpy find it.
Its columns and parameters are not scanpy's, as Slice 11j decided for
`TableSummary`, so the docstring opens with how it differs from
`scanpy.pp.calculate_qc_metrics`: one class per matrix read, in place of
boolean `qc_vars` within `X`; other column names, which it maps to scanpy's;
no `percent_top`, `log1p` or `inplace`; and the summary returned. Two other
names were considered:

- `hp.tb.pp.add_qc_metrics`: in Harpy, `add_` names functions that add to a
  table in `sdata` and its store, such as `hp.tb.add_feature_matrix`, which
  this function does not;
- `hp.tb.pp.calculate_qc_metrics_by_class`, after the pattern of
  `normalize_by_size`: precise, but longer, and the scanpy name with a clear
  docstring was preferred.

### `.obs`: the measures per instance

The measure columns of `summary.per_instance`, under their names, for every
row of the table, as the summary covers every region:

- `n_points_<class>` (uint64) and `n_detected_features_<class>`, per class;
- `pct_points_<class>`, with two classes or more, relative to the classes read
  in the call that wrote it.

A row without points in any class read has 0 points and 0 detected features,
and a missing share (NaN), as 0 / 0 has no value. The region and instance
columns of `per_instance` are the table's own and are not written.

**The rate per panel feature,** computed here, not stored in the summary:

- `n_points_per_panel_feature_<class>` (float64), per class:
  `n_points_<class> / summary.panel_feature_counts[<class>]`, the class's
  points divided by its number of features in the matrices read. For a CosMx
  table: `n_points_per_panel_feature_x`, `n_points_per_panel_feature_negative`
  and `n_points_per_panel_feature_system_control`. The denominator is always
  positive: `summarize_table` raises for `X` or a layer without columns, and
  an `.obsm` matrix has at least one through its names, so every class read
  has a feature and no rate is undefined.
- **Why here:** with the rates in `.obs`, users need not know what to divide by
  what; the denominator is the class's number of features in the matrices read,
  not the features detected, and comes from the summary. The values equal
  those that `instance_histogram` and `instance_scatter` plot with
  `normalization="per_panel_feature"`, so `.obs` and the plots agree.
- **Not in `TableSummary`:** its frames keep counts only, as Slice 11j of
  [allocation.md](allocation.md) decided, symmetric with the points side,
  whose plots also normalize at display time. `.obs` is the convenience layer,
  where the per-cell value is filtered on and plotted, as `pct_points_<class>`
  is.
- **The name** follows the naming rule of Slice 11l: `per_panel_feature` names
  the normalization, a class's points divided by its panel feature count, and
  the class label ends the name.

**The docstring advises comparing rates on aggregates rather than per cell.** A
foreground/background ratio per cell, such as
`n_points_per_panel_feature_negative / n_points_per_panel_feature_x`, is
noisy, as negative points are sparse, and undefined for a cell without points
in `X`. Taken per region, it is stable:

```python
obs = sdata.tables["table"].obs
per_region = obs.groupby("fov_labels", observed=True)[
    ["n_points_per_panel_feature_x", "n_points_per_panel_feature_negative"]
].mean()
per_region["background_to_signal"] = (
    per_region["n_points_per_panel_feature_negative"] / per_region["n_points_per_panel_feature_x"]
)
```

No ratio column is added.

Compared with `scanpy.pp.calculate_qc_metrics`, these are the counterparts of
`total_counts`, `n_genes_by_counts` and `pct_counts_<qc_var>`, under other
names, so that both can be in `.obs`.

### `.var`: the statistics per feature of `X` or the layer

When `X` or a layer was read, the matrix aligned with `.var`, the rows of
`summary.per_feature` of its class, under their names followed by the class
label:

- `n_points_<class>` (uint64), `within_class_fraction_<class>`,
  `n_instances_without_feature_<class>`,
  `pct_instances_without_feature_<class>`, `mean_points_per_instance_<class>`
  and `std_points_per_instance_<class>`.

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
- **Every row is covered:** the columns are written by position, so
  `summary.per_instance.index` must equal `adata.obs_names`, in order. A row
  whose `obs[region_key]` is not a region of the table's annotation is left
  out of the summary, which `TableModel.parse` rules out but a table changed
  afterwards can have; it raises a `ValueError` saying so.

Existing columns of the same names are not a check: they are replaced (see
"No `overwrite`"). The columns are computed before any is written, so a failed
check leaves the table unchanged.

## Documentation

- `docs/api.md` lists `tb.pp.calculate_qc_metrics` under Preprocessing, and its
  introduction widens from "normalisations that scanpy does not provide" to
  the in-place additions of `hp.tb.pp`, with `add_table_updates` to persist,
  with `overwrite=True` for the stored `obs` and `var`.
- The docstring of `hp.qc.summarize_table` points to `calculate_qc_metrics` for
  adding its results to the table.

## Tests

In `src/harpy/_tests/test_table/test_pp/`, on the dummy tables of the tests of
`summarize_table`, limited to what `calculate_qc_metrics` adds: the measures
themselves, such as the missing share of a row without points or the columns
with one class, are those of `summarize_table` and tested there.

- `.obs`: the values equal `per_instance` of the returned summary, with its
  dtypes;
- the rates: each `n_points_per_panel_feature_<class>` equals
  `n_points_<class> / summary.panel_feature_counts[<class>]`, as float64;
- `.var`: the values equal the rows of `per_feature` of `X` or the layer, in
  `var_names` order, labelled with its class; nothing is added for a table
  summarized from `.obsm` alone;
- the returned summary equals `summarize_table` of the same table and
  arguments;
- a second call replaces the columns, with the values of the table as it is
  then, and calls with other matrices keep both sets of columns, each
  labelled;
- a row whose region is not in the table's annotation raises, and leaves
  `.obs` and `.var` unchanged;
- a store-backed table: after `calculate_qc_metrics` and
  `hp.tb.io.add_table_updates(..., overwrite=True)`, the reopened table holds
  the columns.

## Verification

`summarize_table` and `summarize_table_by_feature` take the table, without
`sdata`, and their results are unchanged but for the metadata without
`table_name`. After `calculate_qc_metrics`, every row of the table has each
class's points, detected features and share, equal to `per_instance` of the
returned summary, and each class's points per panel feature, its points
divided by `summary.panel_feature_counts`; the features of `X` or the layer
have the statistics of `per_feature`; nothing else in the table changes; and
`hp.tb.io.add_table_updates` with `overwrite=True` persists the new columns.

## Step 3: class labels as suffixes

**Status: implemented.**

Every class-specific column is named by its base name followed by `_<label>`,
the class label: in `TableSummary.per_instance`, in the `.obs` columns of
`calculate_qc_metrics`, and in its `.var` columns, which so far had no label.
The column of a class is then `f"{column}_{label}"`, wherever it is.

| Where                  | Step 2                          | Step 3, for the class `x`         |
| ---------------------- | ------------------------------- | --------------------------------- |
| `per_instance`, `.obs` | `n_x_points`                    | `n_points_x`                      |
|                        | `n_x_detected_features`         | `n_detected_features_x`           |
|                        | `pct_x_points`                  | `pct_points_x`                    |
| `.obs`, the rate       | `n_x_points_per_panel_feature`  | `n_points_per_panel_feature_x`    |
| `.var`                 | `n_points`                      | `n_points_x`                      |
|                        | `within_class_fraction`         | `within_class_fraction_x`         |
|                        | `n_instances_without_feature`   | `n_instances_without_feature_x`   |
|                        | `pct_instances_without_feature` | `pct_instances_without_feature_x` |
|                        | `mean_points_per_instance`      | `mean_points_per_instance_x`      |
|                        | `std_points_per_instance`       | `std_points_per_instance_x`       |

With `layer="counts"`, the class is `counts`: `n_points_counts`,
`within_class_fraction_counts`, and so on.

- **One rule, without exceptions.** A label within the name, as in
  `n_x_points`, fits the columns that count something, but not
  `within_class_fraction` or the `…_without_feature` pair, which would need a
  second rule.
- **Every `.var` column names its matrix.** In step 2 they did not: after a
  call with `X` and one with `layer="counts"`, `.var` described `counts` while
  `.obs` still held the columns of `x`. With the label, both sets stay side by
  side, each named, as in `.obs`. A record of the call in
  `uns["qc_metrics"]`, removing the columns of the previous call before
  writing, was considered and rejected, to keep the contract lean; scanpy's
  `calculate_qc_metrics` writes no record either.
- **The base names are those of the long frames:** `n_points` and
  `pct_points`, the columns of `SpatialBinSummary.per_bin`, so that a wide
  table column is a long one with a label, and the measures of the instance
  plots, `"points"`, `"detected_features"` and `"pct_points"`, map onto them.
- **scanpy's convention** for its groups of features, such as `total_counts_mt`
  and `pct_counts_mt`.
- **Unchanged:** the columns of the long frames, `per_class`, `per_feature` and
  their per-region frames, which name the class in their `feature_class`
  column, and carry no suffix.
- **The naming rule of Slice 11l:** `per_<unit>` ends the name of a
  statistic, before a class label, as in `mean_points_per_instance_x`;
  `per_panel_feature` likewise, as in `n_points_per_panel_feature_x`.
- **The docstring of `calculate_qc_metrics`** states that columns are only
  written, never removed, so that those of the matrices or classes of earlier
  calls stay; and that `pct_points_<class>` is relative to the classes read in
  the call that wrote it.
- **No backward compatibility:** the columns are renamed without aliases.

**Implementation:** in the schema, one function, `_class_column(column, *,
label)`, which returns `f"{column}_{label}"` as `_statistic_name` names the
statistics, replaces the four per-class templates of `per_instance` and the
rate, applied to the base names `n_points`, `n_detected_features`,
`pct_points` and `n_points_per_panel_feature`; its users follow:
`summarize_table`, the measures of the instance
plots in `_instance_measures.py`, and `calculate_qc_metrics`, which also labels
its `.var` columns. The docstrings follow, the scanpy mappings included, and
the tests of `summarize_table`, the instance plots and `calculate_qc_metrics`.
The roadmap follows: the columns of `per_instance` in Slice 11j, the naming
rule of Slice 11l, and steps 1 and 2 above.

**Verification:** every class-specific column of `per_instance`, `.obs` and
`.var` ends with `_<label>`, and `f"{column}_{label}"` fetches it; the values
are those of step 2; the instance plots select the same values by class and
measure; and two calls with different matrices leave both sets of columns, each
named by its class.
