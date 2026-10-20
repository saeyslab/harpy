# Adding a table summary to its table

## Status

**Planned, in one step.** It builds on the amendment of Slice 11j of
[allocation.md](allocation.md), "every row of the selected regions", and is
implemented after it.

`hp.qc.summarize_table` describes a table's instances and features from its
count matrices, but writes nothing to the table: its results stay in a
`TableSummary`. scanpy's `calculate_qc_metrics` adds its metrics to `.obs` and
`.var` instead, where they can be plotted, filtered on, or written to the
store. This step adds that convenience for a table summary:

```python
summary = hp.qc.summarize_table(sdata, "table", obsm_keys=["negative", "system_control"])
hp.tb.pp.add_table_summary(sdata.tables["table"], summary)  # in place, as scanpy.pp
hp.tb.io.add_table_updates(sdata, table_name="table")  # persist, for a store-backed sdata
```

`summarize_table` stays read-only. A separate function adds the results, as
`hp.tb.pp`'s normalisations change a table in place, like `scanpy.pp`, and
`hp.tb.io.add_table_updates` writes the changed components back.

## The function

```python
hp.tb.pp.add_table_summary(adata, summary, *, overwrite=False) -> None
```

- **`adata`:** the summary's table, `sdata.tables[summary.metadata.table_name]`,
  in memory, lazy or backed: only `.obs` and `.var` change.
- **`summary`:** a `TableSummary` of every region the table annotates.
- **`overwrite`:** `False` raises if a column to add exists already, naming
  the columns; `True` replaces them.

### `.obs`: the measures per instance

The measure columns of `summary.per_instance`, under their names, for every
row of the table:

- `n_<class>_points` (uint64) and `n_<class>_detected_features`, per class;
- `pct_<class>_points`, with two classes or more.

As the summary covers every region, `per_instance` has one row per row of the
table. A row without points in any class read has 0 points and 0 detected
features, and a missing share (NaN), as 0 / 0 has no value. The region and
instance columns of `per_instance` are the table's own and are not written.

**The rate per panel feature,** computed here and not taken from the summary:

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

When the summary read `X` or a layer, the matrix aligned with `.var`, the rows
of `summary.per_feature` of that class, under their names:

- `n_points` (uint64), `within_class_fraction`, `n_instances_without_feature`,
  `pct_instances_without_feature`, `mean_points_per_instance` and
  `std_points_per_instance`.

The population column `n_instances`, the same for every feature, and
`feature_class`, the same class for every row, are not written. These are the
counterparts of scanpy's `total_counts`, `n_cells_by_counts`,
`pct_dropout_by_counts` and `mean_counts`, under other names. A summary
without `X` or a layer, read from `.obsm` matrices alone, adds nothing to
`.var`.

### Not added

- the features of the `.obsm` matrices, which have no `.var` axis;
- the per-class and per-region statistics, `per_class`, `per_region_class`
  and `per_region_feature`, which are not aligned with rows or features.

They stay in the summary.

## Checks, before anything is written

- **The summary's type:** a `TableSummary`, or a `TypeError`.
- **Every region:** `summary.metadata.region` must be every region of the
  table's SpatialData annotation; a summary of fewer regions raises, as the
  rows of the other regions would have no values.
- **The table is still the summary's,** as `summarize_table_by_feature`
  checks it: the instances of `per_instance` are found by `.obs` name and must still
  have their region and instance id, and `X` or the layer must still have the
  summary's names as `var_names`. Otherwise the call raises "recompute
  summarize_table()". The helpers of `summarize_table_by_feature` that do this
  move to a module that both functions import.
- **Existing columns:** with `overwrite=False`, a column to add that `.obs` or
  `.var` already holds raises, naming the columns.

A failed check leaves the table unchanged.

## Documentation

- `docs/api.md` lists `tb.pp.add_table_summary` under Preprocessing, and its
  introduction widens from "normalisations that scanpy does not provide" to
  the in-place additions of `hp.tb.pp`, with `add_table_updates` to persist.
- The docstring of `hp.qc.summarize_table` points to `add_table_summary` for
  adding its results to the table.

## Tests

In `src/harpy/_tests/test_table/test_pp/`, on the dummy tables of the tests of
`summarize_table`:

- `.obs`: the values equal `per_instance`, row by row; a row without points
  has 0 points and 0 detected features and a missing share; the dtypes; no
  share column with one class;
- the rates: each `n_<class>_points_per_panel_feature` equals
  `n_<class>_points / summary.panel_feature_counts[<class>]`, 0 for a row
  without points, and the values that `instance_scatter` plots with
  `normalization="per_panel_feature"`;
- `.var`: the values equal the rows of `per_feature` of `X` or the layer, in
  `var_names` order; nothing is added for a summary read from `.obsm` alone;
- the checks: a summary of one region of two, a table changed since the
  summary, existing columns without and with `overwrite`, and another type
  than `TableSummary`; a failed check leaves `.obs` and `.var` unchanged;
- a store-backed table: after `add_table_summary` and
  `hp.tb.io.add_table_updates`, the reopened table holds the columns.

## Verification

After `add_table_summary`, every row of the table has each class's points,
detected features and share, equal to `per_instance`, and each class's points
per panel feature, its points divided by `summary.panel_feature_counts`;
`TableSummary` itself is unchanged; the features of `X` or
the layer have the statistics of `per_feature`; nothing else in the table
changes; and `hp.tb.io.add_table_updates` persists the new columns.
