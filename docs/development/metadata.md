# Metadata ownership and layout

Harpy stores shared dataset metadata in `sdata.attrs["harpy"]` and table-owned
metadata in `adata.uns`, where `adata = sdata.tables[table_name]`. These are
Harpy conventions, distinct from SpatialData's own annotations and transforms.
The diagrams show the main relationships, not exhaustive schemas; records are
present only when the corresponding operation creates them.

## Dataset and element metadata

```text
sdata.attrs["harpy"]
├── metadata_version
├── provenance                         reader identity and Harpy version
├── images
│   └── <image name>                    source/selection metadata, channels
├── labels
│   └── <labels name>                   source/selection metadata,
│                                      instance-ID encoding or categories
├── points
│   ├── <points name A>
│   │   ├── ...                        optional source/selection metadata
│   │   └── feature_panel: <panel key>
│   └── <points name B>
│       └── feature_panel: <panel key>  same panel may be shared
└── feature_panels
    └── <panel key>                     feature_panel_<content hash>
        ├── feature_key                points column with feature identifiers
        ├── feature_class_key          points column with class assignments
        ├── classes                    sorted class names
        └── features_by_class          class → complete sorted feature list
```

Each points record's `feature_panel` value names a record in the shared
`feature_panels` registry. Multiple points elements can reference the same panel.

Image, labels and points records use exact element names. Source/selection
metadata may describe sample identity, selected data, geometry or acquisition
information. The available fields depend on the writer; a panel association
does not require reader-specific metadata.

The panel is authoritative for feature-to-class assignments and may include
features with no detected points. The class column in the points dataframe is
the per-point annotation of that same relation, validated against the panel.

Stored feature panels require lexicographically sorted `classes` and sorted
feature lists within each `features_by_class` entry. New panel construction
normalizes input into this order; reading stored metadata rejects unsorted
lists rather than silently repairing them. This requirement concerns list
order, not dictionary-key order.

Feature panels are stored once so multiple points elements can share them.
Metadata describing individual elements would more naturally live with those
elements. Storing it in `sdata.attrs["harpy"]` is a storage compromise:
SpatialData preserves root attributes when writing and reopening a Zarr store,
but does not guarantee the same for custom attributes stored directly on
individual elements. Table-specific metadata uses AnnData's `.uns`, which
supports persistence, as shown below.

## Table metadata and the data it describes

```text
adata.uns
├── spatialdata_attrs                   SpatialData-owned region/instance binding
├── feature_matrices
│   └── <matrix key>
│       ├── schema_version, source_kind
│       ├── feature_columns             ordered names for matrix columns
│       ├── regions                     add_feature_matrix: labels name → image
│       │                               (or None) + coordinate system
│       └── ...                         writer-specific metadata of the matrix
├── point_aggregation                   sources of a table from aggregate_points; names no class
│   ├── schema_version, source_kind
│   ├── feature_key                     points column of the features
│   ├── feature_class_key               class-aware only: points column of the classes
│   └── regions                         labels name → points name + coordinate system
└── spatial_coordinates
    └── spatial_canonical
        ├── schema_version, axes
        └── regions                     per labels element: coordinate frame,
                                        algorithm version and source

feature_matrices[<matrix key>]       → adata.obsm[<matrix key>]
point_aggregation                   → provenance of adata.X and, class-aware, of
                                     the non-expression class matrices adata.obsm[snake_case(class)]
spatial_coordinates["spatial_canonical"]
                                   → adata.obsm["spatial_canonical"]
```

For example, a table of
{func}`hp.tb.aggregate_points <harpy.tb.aggregate_points>` in class-aware
mode with two regions, and one matrix of
{func}`hp.tb.add_feature_matrix <harpy.tb.add_feature_matrix>`:

```python
adata.uns = {
    # SpatialData's annotation, not Harpy's: which regions the table annotates.
    "spatialdata_attrs": {"region": ["cells_a", "cells_b"], "region_key": "fov_labels", "instance_key": "cell_ID"},
    # hp.tb.aggregate_points: where X and the class matrices came from.
    "point_aggregation": {
        "schema_version": 1,
        "source_kind": "harpy_aggregate_points",
        "feature_key": "gene",
        "feature_class_key": "code_class",  # class-aware mode only
        "regions": {
            "cells_a": {"points_element": "transcripts_a", "coordinate_system": "sample_a"},
            "cells_b": {"points_element": "transcripts_b", "coordinate_system": "sample_b"},
        },
    },
    # One entry per described obsm matrix, under the matrix's obsm key.
    "feature_matrices": {
        # A class matrix of aggregate_points: no regions, its sources are in point_aggregation.
        "negative": {
            "schema_version": 2,
            "source_kind": "harpy_aggregate_points",
            "feature_columns": ["Negative01", "Negative02"],
        },
        # hp.tb.add_feature_matrix
        "cell_features": {
            "schema_version": 2,
            "source_kind": "harpy_add_feature_matrix",
            "feature_columns": ["mean__DAPI", "mean__CD3", "area"],
            "source_channels": ["DAPI", "CD3"],  # None without intensity features
            "features": ["mean", "area"],
            "regions": {  # image_element is None for morphology only
                "cells_a": {"image_element": "image_a", "coordinate_system": "sample_a"},
                "cells_b": {"image_element": "image_b", "coordinate_system": "sample_b"},
            },
        },
    },
    # hp.tb.add_canonical_centers, or hp.tb.aggregate_points.
    "spatial_coordinates": {
        "spatial_canonical": {
            "schema_version": 2,
            "axes": ["z", "y", "x"],
            "regions": {"cells_a": {...}, "cells_b": {...}},
        },
    },
}
```

[Canonical-center interoperability](../canonical_centers.md) describes the
entries of `spatial_canonical`.

Each `feature_matrices` record describes the `.obsm` matrix with the same
key; the shared key is their only link. `feature_matrices` is the default
`.uns` key of this registry:
{func}`hp.tb.add_feature_matrix <harpy.tb.add_feature_matrix>` can write its
record under another key, and
{func}`hp.tb.validate_table <harpy.tb.validate_table>` checks only the registry
named by its `feature_matrices_key`. No feature matrix may use the `.obsm` keys
`spatial_canonical` or `spatial`, which tools read as coordinates:
{func}`hp.tb.add_feature_matrix <harpy.tb.add_feature_matrix>` and
{func}`hp.tb.aggregate_points <harpy.tb.aggregate_points>` refuse them, and
{func}`hp.tb.validate_table <harpy.tb.validate_table>` rejects a record under
either key.

`feature_columns` identifies the columns of an `.obsm` feature matrix;
`feature_key` names a source points column. They have different roles.
Expression-matrix columns are identified by `adata.var_names`.

{func}`hp.tb.aggregate_points <harpy.tb.aggregate_points>` writes
`point_aggregation` in both modes, as
provenance that no validation reads: it pairs each region with the points
element and coordinate system aggregated into it, which SpatialData's table
annotation does not record. In a class-aware table, a column's class is the
matrix that holds it, so no record maps classes to columns or matrices.
`adata.X` holds the expression class, the one panel class without a matrix
in `.obsm`; each other class appears only as the `.obsm` key of its matrix,
`snake_case(class)`, for example `"negative"` for the class `Negative`. The
function's docstring describes the complete table.

Canonical centers use each source labels element's **intrinsic scale-0 pixel
coordinates**, stored in `(z, y, x)` order, with `z = 0` for 2D. Different regions
need not share a coordinate frame. Source dimensions distinguish 2D from 3D;
these centers are not coordinates in a shared global system. See
[Canonical-center interoperability](../canonical_centers.md).

## The table annotation

`uns["spatialdata_attrs"]["region"]` declares the regions a table annotates.
It must equal the regions that have rows in `obs[region_key]`, in both
directions; unused categories of the column do not count.

- SpatialData checks this with `TableModel.validate`: in `TableModel.parse`,
  in `SpatialData(tables=...)`, when `sdata.tables = {...}` is set, and when a
  table is written. Not when rows are subset or with
  `sdata.tables[name] = adata`, so a table can hold a mismatch in memory until
  it is validated or written.
- Harpy checks it in {func}`hp.tb.validate_table <harpy.tb.validate_table>`
  and {func}`hp.tb.concat <harpy.tb.concat>`, with one message for each
  direction. `validate_table`
  also requires each declared region to be an element of `sdata`; SpatialData
  only warns.
- {func}`hp.tb.io.add_table <harpy.tb.io.add_table>` replaces the declared
  regions with its `region` argument, which must therefore equal the regions
  that have rows.

A mismatch usually comes from:

- subsetting rows, which keeps every declared region: set the declared
  regions to those left, or pass them as `region` to `hp.tb.io.add_table`;
- combining tables with `anndata.concat`, or editing `obs[region_key]`, which
  leaves rows of regions that are not declared: use `hp.tb.concat`, or update
  the declared regions.

## Metadata per region

Each of Harpy's records in `uns` describes a matrix, as the diagram above
shows: a `feature_matrices` record its `.obsm` matrix, `point_aggregation`
`adata.X` and the class matrices, and `spatial_coordinates["spatial_canonical"]`
the canonical centers. A record keeps what differs per region under `regions`,
a mapping keyed by the name of the labels element, one of the regions the
table annotates (`uns["spatialdata_attrs"]["region"]`, see above); every other
field describes the whole matrix. The canonical-center and `point_aggregation`
records, and the `feature_matrices` records of
{func}`hp.tb.add_feature_matrix <harpy.tb.add_feature_matrix>`, follow this
rule, and a new writer should too:

- `regions` holds the regions with data: fewer than the table declares when
  a matrix is computed for some regions only, and, after a subset, regions
  the table no longer declares. Readers use the declared regions only.
- The order of `regions` has no meaning: a Zarr store returns its keys
  sorted. Region order comes from `obs`.
- Values are strings, integers, `None`, and lists or mappings of these. A
  `None` in a list does not survive storage: AnnData writes `[None, "a"]` as
  `["None", "a"]`. A `None` mapping value does.
- Entries name elements by their type, `points_element` or `image_element`,
  and a SpatialData coordinate system `coordinate_system`.

A record without `regions` has no per-region source, or keeps it elsewhere:
the class matrices of
{func}`hp.tb.aggregate_points <harpy.tb.aggregate_points>` come from the same
points as `adata.X`, so their sources are in `point_aggregation`, once for
all of them.

{func}`hp.tb.concat <harpy.tb.concat>` relies on this rule: the fields other
than `regions` must be
equal in every table, and each table contributes the entries of the regions
it declares.

## Keeping records consistent

- Copying or renaming elements requires preserving/updating their root metadata
  records and referenced panels; generic SpatialData operations do not manage
  these Harpy relationships automatically. Deletion can leave orphaned records;
  a shared panel may still be referenced by other elements.
- Matrix rows must stay aligned with `.obs`; feature-column descriptions must
  stay aligned with matrix columns. Coordinate metadata must describe the
  corresponding matrix and source labels.
- Schema versions describe metadata contracts. Reader versions describe software
  provenance; they are not schema versions.
- Concatenating tables with `anndata.concat` or `spatialdata.concatenate`
  drops these records, or keeps one table's; use
  {func}`hp.tb.concat <harpy.tb.concat>`, which merges them by the rule in
  [Metadata per region](#metadata-per-region).

For staging, overwrites and recovery responsibilities, see
[Storage writes and overwrite guarantees](storage.md).
