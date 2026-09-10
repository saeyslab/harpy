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
│       └── ...                         writer-specific source metadata
├── feature_class_aggregation
│   ├── schema_version, source_kind
│   ├── feature_key, feature_class_key, expression_class, classes
│   ├── auxiliary_class_feature_counts  panel feature counts, not point counts
│   ├── count_columns                   class → .obs point-count column
│   ├── auxiliary_points_fraction_column
│   ├── auxiliary_feature_matrix_key    "auxiliary_feature_counts"
│   └── regions                         labels name → points name + coordinate system
└── spatial_coordinates
    └── spatial_canonical
        ├── schema_version, obsm_key, axes, dtype
        ├── region_key, instance_key
        └── regions                     per-label source, coordinate frame,
                                        calculation and row coverage

feature_matrices[<matrix key>]       → adata.obsm[<matrix key>]
feature_class_aggregation           → adata.X / .var, .obs summaries,
                                     .obsm["auxiliary_feature_counts"]
spatial_coordinates["spatial_canonical"]
                                   → adata.obsm["spatial_canonical"]
```

`feature_columns` identifies the columns of an `.obsm` feature matrix;
`feature_key` names a source points column. They have different roles.
Expression-matrix columns are identified by `adata.var_names`.

Canonical centers use each source labels element's **intrinsic scale-0 pixel
coordinates**, stored in `(z, y, x)` order, with `z = 0` for 2D. Different regions
need not share a coordinate frame. Source dimensions distinguish 2D from 3D;
these centers are not coordinates in a shared global system. See
[Canonical-center interoperability](../canonical_centers.md).

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

For staging, overwrites and recovery responsibilities, see
[Storage writes and overwrite guarantees](storage.md).
