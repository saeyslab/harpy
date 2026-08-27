# Class-aware transcript allocation

## Status

Five implementation slices are planned; Slice 1 is implemented:

1. patch the CosMx reader and establish the generic Harpy feature-panel
   metadata contract — implemented;
2. extend the CosMx reader with an explicit, sample-scoped multi-sample API;
3. add class-aware aggregation to `hp.tb.allocate`;
4. add QC functions that summarize the original, unallocated control points;
5. optimize the generic point-to-label assignment and reduction path.

Slice 2 extends the reader foundation from Slice 1 without changing the
single-sample API. Slice 3 consumes the feature-panel metadata produced by
Slice 1 and the sample-aware element contracts established by Slice 2, but
also supports an explicit denominator mapping for generic points. Slice 4
depends on the reader metadata from Slices 1 and 2, not on allocation,
instance labels, or an AnnData table. Slice 5 preserves the public behavior
established by Slice 3 while replacing the private allocation execution path.

## Goal

Establish a general control-aware transcript workflow: readers preserve
authoritative panel information for one or more sample-scoped runs, allocation
creates an AnnData expression matrix containing only the selected biological
class while retaining compact per-instance control summaries in `.obs`, and
separate QC functions summarize all original control points. The raw points
elements remain unchanged and continue to contain biological and control
transcripts.

For the investigated CosMx run, `code_class` has three values:

- `Endogenous`: biological targets that belong in the primary expression
  matrix;
- `Negative`: physical negative-control probes intended not to hybridize to the
  sample, used to estimate nonspecific binding and background; and
- `SystemControl`: reserved false/blank codewords without a corresponding
  targeting probe, used to estimate spot-calling and barcode-decoding errors.

The feature is not CosMx-specific. Other imaging-based transcriptomics
platforms also distinguish biological targets, negative probes, and unused
codewords. The allocation API and implementation should therefore use generic
feature-class terminology and must not hard-code CosMx class names or panel
sizes.

## Slice 1: reader and feature-panel metadata

**Status: implemented.**

This slice changes the CosMx reader and the metadata it writes. It does not
change `hp.tb.allocate` or introduce QC computations.

### Authoritative panel metadata

The number and meaning of control targets belong to the assay panel. They are
not general CosMx constants and must not be guessed from class names or inferred
from detected transcript rows.

The investigated CosMx export contains a small run-level plex file with the
columns `DisplayName`, `CodeClass`, and `ProbeID`. It contains 1,165 unique
targets:

| `CodeClass` | Panel targets |
| --- | ---: |
| `Endogenous` | 958 |
| `Negative` | 10 |
| `SystemControl` | 197 |

The CosMx reader currently does not need the plex to create transcript points,
because each detected transcript already carries its target and code class.
Class-normalized allocation introduces a distinct reason to consume it: unlike
the detected transcript rows, the plex also represents targets with zero
detections. A denominator derived from observed targets would be biased whenever
one of those targets has no calls.

When transcripts are ingested and a plex file is present, the CosMx reader
should therefore discover exactly one plex file, read it once, and associate a
compact feature-panel record with every transcript points element created from
that run. A missing plex must not prevent raw transcript ingestion; it only
limits the normalized allocation metrics available later. The record must
contain at least:

- the feature and class column names;
- the ordered feature-class categories; and
- the authoritative target names grouped by class.

Introduce a vendor-neutral Harpy metadata convention in the root
`SpatialData.attrs`. This is a new Harpy contract, not an existing SpatialData
standard. All metadata written and consumed by Harpy belongs under the single
top-level `harpy` namespace. Do not retain a parallel top-level `cosmx`
metadata tree: the originating reader is identified by
`harpy.provenance.reader`, while downstream operations resolve modality
metadata by SpatialData element type and exact element name.

The namespace has a versioned, element-keyed structure. The CosMx reader must
migrate its existing provenance, image, label, and transcript records into this
structure when feature-panel support is implemented:

```python
harpy_metadata = sdata.attrs.setdefault("harpy", {})
harpy_metadata.update(
    {
        "metadata_version": 1,
        "provenance": {
            "reader": "cosmx",
            "reader_version": "...",
        },
        "images": {
            "morphology_image_mosaic_1": {
                # Existing per-image source metadata lives here.
            },
        },
        "labels": {
            "instance_labels_mosaic_1": {
                # Existing per-label source metadata lives here.
            },
        },
        "points": {
            "transcripts_mosaic_1": {
                # Existing per-points source metadata lives here.
                "feature_panel": "transcripts_panel",
            },
            "transcripts_mosaic_2": {"feature_panel": "transcripts_panel"},
        },
        "feature_panels": {
            "transcripts_panel": {
                "feature_column": "gene",
                "class_column": "code_class",
                "categories": ["Endogenous", "Negative", "SystemControl"],
                "targets_by_class": {
                    "Endogenous": ["Abca2", "Abi1", ...],
                    "Negative": ["Negative01", ..., "Negative10"],
                    "SystemControl": ["SystemControl1", ..., "SystemControl197"],
                },
            },
        },
    }
)
```

The `images`, `labels`, and `points` mappings are keyed by the exact element
names in the corresponding SpatialData collections. This makes metadata
lookup independent of the reader that created an element. The metadata version
applies to the complete Harpy root contract. Root provenance is deliberately
minimal: it identifies the reader and its version, but contains neither source
paths nor FOV or mosaic settings. Actual FOV membership belongs to each
element. Slice 2 adds the sample identity and sample-specific mosaic settings
to those element records. The CosMx whole-store overwrite safety check must
specifically require `harpy.provenance.reader == "cosmx"`; the mere presence of
Harpy metadata is not sufficient evidence that the store is replaceable by
that reader.

The panel identifier is local to the SpatialData object and must be non-empty
and unique. The reader should derive a stable identifier from its transcript
output base name and avoid a collision with an incompatible existing panel.
Multiple points elements may reference the same panel.

The `categories` order is the categorical dtype order shared by all associated
points elements. `targets_by_class` keys must exactly equal those categories;
each target list must contain unique, non-empty strings; and no target may occur
under more than one class. Derive target counts from the list lengths instead of
storing a second potentially inconsistent count mapping. The CosMx reader sorts
both class names and the targets within each class lexicographically so output
does not depend on row order in the plex; this ordering is deterministic rather
than a claim of biological precedence.

Slice 3 only needs those derived counts: it uses the length of each class target
list as the denominator for normalized control metrics. Slice 4 additionally
needs the actual target names. A categorical transcript column contains only
categories represented by the ingested points and cannot, by itself, preserve
the target-to-class relationship for a panel target with no detected rows.
Keeping the authoritative names in `targets_by_class` therefore supports both
uses without storing a separate count mapping.

Only the allocation- and QC-relevant `DisplayName`/`CodeClass` relationship is
persisted. Do not store unused plex fields such as `ProbeID`, and do not
duplicate the panel under every mosaic record. Although the example originates
from CosMx, no key in the Harpy contract is vendor-specific. Other readers can
associate their points with the same structure using their own classes and
targets.

Validate that plex display names are unique, class values are non-null, and
class target counts are positive. Prefix matching is not an acceptable
fallback: this panel contains a target named `NegativeAdd` whose authoritative
`CodeClass` is `Endogenous`.

The CosMx reader stores `code_class` categorically with the same category set
for every mosaic points element from the run. Its categories come from the plex
`CodeClass` values. Parquet preserves the categorical values, but a reopened
Dask dataframe may report them as unknown until supplied with the authoritative
category list. That list is persisted in the shared feature-panel metadata so
Slice 3 can restore a known categorical dtype lazily without scanning the
points. This changes the previous Arrow-string representation only for newly
ingested data; ordinary allocation remains compatible with existing string
columns when class-aware mode is not requested.

### Verification

Focused reader tests should establish that:

- one valid plex is read once and stored as one shared feature-panel record;
- every transcript points element references that shared panel without
  duplicating it;
- the persisted `code_class` column has known categorical categories matching
  the panel;
- target names, classes, and zero-detection panel targets survive a SpatialData
  Zarr round trip;
- duplicate plex files, duplicate or empty target names, null classes,
  conflicting target-to-class mappings, and invalid panel categories are
  rejected before transcript materialization;
- a missing plex still permits raw transcript ingestion but creates no
  feature-panel reference;
- provenance and all image, label, and points metadata are migrated to the
  versioned, element-keyed `harpy` namespace with no parallel top-level
  `cosmx` metadata; and
- whole-store overwrite is permitted only when
  `harpy.provenance.reader == "cosmx"`.

## Slice 2: multi-sample CosMx reader

**Status: specified; not implemented.**

Extend the reader foundation from Slice 1 so several independent CosMx runs
can be written into one SpatialData store. A sample is an explicit unit of
configuration and identity. Do not represent samples through parallel lists of
paths, FOV selections, channels, mosaic modes, tolerances, and orientations;
those lists are difficult to validate and can silently associate a setting
with the wrong run.

### Public contract

Introduce one immutable sample configuration:

```python
@dataclass(frozen=True)
class CosmxSample:
    path: str | Path
    fovs: Sequence[int] | None = None
    channels: Sequence[str] | None = None
    mosaic_mode: Literal["spatial_groups", "single"] = "spatial_groups"
    adjacency_tolerance_px: int | None = None
    flip_x: bool = True
    flip_y: bool = False
```

Add a multi-sample entry point whose mapping keys are the sample identifiers:

```python
sdata = cosmx_samples(
    samples={
        "sample_a": CosmxSample(
            path=sample_a_root,
            fovs=range(1, 58),
            channels=["DAPI", "PanCK"],
            adjacency_tolerance_px=85,
        ),
        "sample_b": CosmxSample(
            path=sample_b_root,
            mosaic_mode="single",
            flip_x=False,
        ),
    },
    output=output_zarr,
    raster_scale_factors=[2, 2, 2, 2, 2],
    overwrite=True,
)
```

Keep `harpy.io.cosmx()` as the backward-compatible single-sample API. It may
delegate to shared preparation and writing primitives, but existing calls and
their element names must remain unchanged. Do not make callers wrap one run in
an arbitrary sample mapping merely to preserve current behavior.

The sample mapping must be non-empty and its keys must be unique, non-empty,
SpatialData-safe identifiers. Preserve mapping iteration order for predictable
execution, while ensuring that output metadata and panel identifiers are
deterministic for the same logical inputs. Reject a sample identifier that
would make any planned element or coordinate-system name collide.

The sample configuration owns values that may differ between runs:

- source path;
- selected FOVs and morphology channels;
- mosaic mode and adjacency tolerance; and
- X/Y orientation.

As in the single-sample API, `adjacency_tolerance_px` applies only to
`mosaic_mode="spatial_groups"`. It is ignored when `mosaic_mode="single"`,
because that mode deliberately constructs one mosaic without adjacency-based
grouping.

Arguments that define the complete output remain on `cosmx_samples`: output
path, modality inclusion, output base names, image and label chunks, raster
scale factors, transcript block size, and overwrite behavior. Do not accept a
list of these output-wide values.

### Sample-scoped elements and coordinate systems

Prefix every element and coordinate system created by the multi-sample API
with its sample identifier. For example:

```text
sample_a_morphology_image_mosaic_1
sample_a_instance_labels_mosaic_1
sample_a_compartment_labels_mosaic_1
sample_a_transcripts_mosaic_1

sample_a_global_1
sample_a_global_1_micron
```

Mosaic numbering is local to a sample. The coordinate systems of different
samples are independent even when their local pixel coordinates, FOV numbers,
or mosaic numbers happen to match. The reader must not place unrelated samples
in a shared active `global` coordinate system or imply that they are aligned.
Cross-sample registration, when available, is a later explicit transformation
step.

Overlapping local instance IDs between samples are valid. SpatialData table
rows are identified by the pair `(region, instance_id)`, where `region` is the
sample-prefixed labels element. Do not reserve sample-wide integer ranges or
change the existing per-FOV `uint32` instance-ID encoding merely to make IDs
globally unique across labels elements.

### Metadata placement

Keep root provenance common to the whole reader invocation and minimal:

```python
sdata.attrs["harpy"]["provenance"] = {
    "reader": "cosmx",
    "reader_version": "...",
}
```

Do not store source paths, a run registry, a union of selected FOVs, or one
scalar mosaic setting at the root. Each created image, labels, and points
record must instead include its sample identity, actual FOV membership, and
sample-specific mosaic construction settings:

```python
{
    "sample_id": "sample_a",
    "fovs": [1, 2, 3, ...],
    "mosaic": {
        "mode": "spatial_groups",
        "adjacency_tolerance_px": 85,
    },
    # Existing modality-specific orientation, origin, scale, and channel data.
}
```

The FOV list describes the exact source tiles represented by that element; it
is not a duplicate invocation-level selection record. A points element keeps
its `feature_panel` reference alongside this sample-scoped metadata. The
`sample_id` field is required for elements created by `cosmx_samples()`. The
backward-compatible `cosmx()` path has no caller-supplied sample identity and
therefore keeps unprefixed element names and omits this field, but it still
stores FOV membership and mosaic construction settings at element level.

### Feature panels across samples

Canonicalize every discovered panel using the Slice 1 contract before writing
any points. Samples with identical canonical panel contents should reference
one shared feature-panel record. Samples with different panels must reference
separate records. Derive stable panel identifiers from canonical content plus
a readable base so panel sharing does not depend on sample input order, and
reject a key collision with incompatible existing content.

Sharing a panel record is a storage optimization, not an assertion that the
samples are spatially aligned. Conversely, two different registry keys do not
necessarily make panels incompatible for allocation: Slice 3 compares the
canonical feature column, class column, categories, and target-to-class
contents. One output table may combine only compatible selected panels.

### Validation and atomic publication

Prepare all samples before writing: discover and validate every manifest,
construct every preview, canonicalize panels, and plan all element names,
coordinate systems, and metadata references. Fail on a configuration or name
collision before decoding raster or transcript payloads.

Refactor the single-sample implementation around reusable internal operations
such as `_prepare_cosmx_sample` and `_write_cosmx_sample`; the exact private
names may differ. The multi-sample reader must not repeatedly call the public
`cosmx()` function and then attempt to merge its stores. It must write samples
sequentially into one staging store to bound peak memory, reopen and validate
the completed SpatialData object, and publish the store once. A failure in any
sample removes reader-generated staging data and leaves an existing output
store intact.

Do not rely on generic SpatialData concatenation to define this contract.
Although concatenation can suffix duplicate element names, the reader must
control sample-prefixed names, coordinate systems, Harpy metadata references,
and feature-panel deduplication explicitly.

### Verification

Focused reader tests should establish that:

- two samples with overlapping FOV and mosaic numbers create distinct,
  sample-prefixed elements and coordinate systems;
- per-sample FOV, channel, mosaic, tolerance, and orientation settings reach
  only that sample's elements;
- per-element metadata records the correct `sample_id`, represented FOVs, and
  mosaic construction settings, while root provenance remains reader-only;
- identical panels are stored once and referenced by both samples, whereas
  incompatible panels remain separate;
- overlapping instance IDs in different labels elements are preserved;
- invalid sample identifiers and all planned name or coordinate-system
  collisions fail before payload materialization;
- failure while writing a later sample leaves an existing destination intact
  and removes staging data; and
- the existing single-sample `cosmx()` call retains its API, element names,
  coordinate systems, and data results, while its mosaic settings use the same
  element-local metadata contract.

## Slice 3: class-aware `hp.tb.allocate`

**Status: specified; not implemented.**

This slice consumes the generic feature-panel contract established by Slice 1
and supports the sample-scoped elements created by Slice 2. It must remain
usable with non-reader points through an explicit denominator mapping, and it
must preserve ordinary allocation behavior when class-aware arguments are
omitted.

### Public contract

Remove the stateful `append` parameter and let one allocation call describe the
complete output table. The element-selection arguments accept one region or a
collection of paired regions:

```python
labels_name: str | list[str]
points_name: str | list[str] = "transcripts"
to_coordinate_system: str | list[str] = "global"
output_table_name: str = "table_transcriptomics"
```

Extend `hp.tb.allocate` with optional class-aware arguments:

```python
name_feature_class_column: str | None = None
expression_class: str | None = None
control_class_denominators: Mapping[str, int] | None = None
```

These parameter names, types, and defaults are final for this slice. The
`append` parameter is removed; `overwrite` controls only whether the completed
table may replace an existing table element.

```python
sdata = hp.tb.allocate(
    sdata,
    labels_name=[
        "sample_a_cellpose_labels_mosaic_1",
        "sample_b_cellpose_labels_mosaic_1",
    ],
    points_name=[
        "sample_a_transcripts_mosaic_1",
        "sample_b_transcripts_mosaic_1",
    ],
    output_table_name="table_transcriptomics",
    to_coordinate_system=["sample_a_global_1", "sample_b_global_1"],
    name_gene_column="gene",
    name_feature_class_column="code_class",
    expression_class="Endogenous",
    control_class_denominators=None,
    overwrite=True,
)
```

`labels_name` defines the number and order of allocation pairs. Require at
least one labels element and reject duplicate labels names. A scalar
`points_name` or `to_coordinate_system` is broadcast to every labels element;
a list must have the same length as `labels_name`. This permits both one shared
points element in a common coordinate system and separate points elements for
independent mosaics. Validate and normalize these pairs before starting any
spatial lookup.

This follows the established multi-element convention in
`hp.tb.add_feature_matrix`, while keeping the biological association explicit:

```text
labels[0]  + points[0]  + coordinate_system[0]
labels[1]  + points[1]  + coordinate_system[1]
    ...
```

Their semantics are:

- `name_feature_class_column` identifies the per-point column that classifies
  each target;
- `expression_class` selects the only class whose targets are retained in
  `adata.X`; and
- `control_class_denominators` optionally supplies the number of panel targets
  for every non-expression class as a fallback for points without feature-panel
  metadata, or as a consistency assertion when such metadata is available.

An ordinary reader-backed call should not contain panel-specific constants. An
explicit mapping remains available for generic points elements that were not
created by a reader supplying feature-panel metadata:

```python
control_class_denominators={
    "Negative": 10,
    "SystemControl": 197,
}
```

Control denominators are assay facts rather than tunable normalization
parameters. Allocation must resolve the two possible sources according to this
contract:

| Feature-panel metadata | Explicit mapping | Result |
| --- | --- | --- |
| available | `None` | derive denominators from panel target-list lengths |
| unavailable | provided | validate and use the explicit mapping |
| available | identical values | accept the assertion and use the panel-derived values |
| available | conflicting values | raise `ValueError` |
| unavailable | `None` | raise `ValueError` before spatial lookup |

When both sources are present, require exact equality after validation: the
class keys and positive integer values must match. Partial mappings, missing or
additional classes, and per-class overrides are not supported. Explicit values
must never silently replace an attached panel's values. A malformed or
column-incompatible attached panel is itself an error and cannot be bypassed by
supplying explicit denominators.

When `name_feature_class_column=None`, allocation must branch to the existing
implementation immediately, before feature-panel lookup or any class-specific
projection, validation, or aggregation. No additional `.obs` columns are
created and every `name_gene_column` value remains in `adata.X`. Other
class-aware parameters must also be `None` in this mode so that a partially
specified request is not silently ignored.

The ordinary path still supports multiple allocation pairs. It uses the
deterministic union of observed targets as the shared feature axis and fills a
target absent from one pair with zero counts for that pair.

### Shared feature axis and panel compatibility

Do not construct independently schematized AnnData objects and combine them
with the default inner join. An inner join would silently discard a target from
the complete table whenever that target has no assigned transcripts in one
region.

For class-aware allocation, resolve one shared feature axis before constructing
the final AnnData:

- when all selected points elements reference compatible feature-panel
  metadata, use the panel's ordered `expression_class` targets, including
  targets with zero detections;
- without feature-panel metadata, use a deterministic, lexicographically
  sorted union of the observed expression targets across all selected points
  elements; and
- construct every per-pair sparse count matrix against this shared axis, so a
  missing target is represented by a zero rather than by dropping the feature.

In class-aware mode, feature-panel metadata must be available for all selected
points elements or for none of them. When present, all referenced panels must
agree on the feature column, class column, ordered categories, target-to-class
mapping, and targets by class. Panels selected from different samples need not
share the same registry key, but their canonical contents must be compatible.
Reject mixed metadata availability and incompatible panels before spatial
allocation. This prevents a zero from ambiguously meaning either "assayed but
not detected" or "not assayed by this panel."

There is one `control_class_denominators` mapping for the complete allocation
call. When compatible panel metadata is present, derive one shared mapping and
compare an explicitly supplied mapping with it as a consistency assertion.
Without panel metadata, require one explicit mapping and apply it to every
pair. Per-region denominator mappings are not supported because they would make
the same normalized `.obs` column have different meanings in different rows.

### Categorical class contract

When `name_feature_class_column` is provided, its points column must have a
categorical dtype. If its Dask categories are unknown after a Parquet round
trip, compatible feature-panel metadata must supply the authoritative ordered
categories and allocation must apply that categorical dtype lazily before
validation. Without such metadata, the input must already have known Dask
categories. The category set is the complete feature-class universe for the
points element, including a class with zero detected or zero assigned points.
Require that:

- every category is a non-empty string;
- `expression_class` is one of the categories;
- the column contains no null values;
- every target maps to exactly one feature class; and
- after metadata resolution or application of an explicit fallback, the keys
  of `control_class_denominators` are exactly
  `set(categories) - {expression_class}`.

Missing and additional denominator keys are both errors, and every denominator
must be a positive integer. Do not silently discard an unconfigured class. An
unused categorical class is valid and produces zero per-instance counts, but it
still requires its authoritative denominator.

Normalize each category deterministically to snake case and construct output
names from that normalized category rather than accepting platform-specific
column names:

```text
Endogenous       -> n_endogenous
Negative         -> n_negative, negative_per_target
SystemControl    -> n_system_control, system_control_per_target
Gene Expression  -> n_gene_expression
```

Reject an empty normalized name, two categories that normalize to the same
name, and collisions with existing `.obs` columns or the fixed
`control_fraction` output. Category order determines output-column order but
does not affect the calculations.

When compatible panel metadata is available, allocation resolves it from the
selected `points_name`, verifies that the stored feature and class columns match
the requested columns, and derives denominators from the target-list lengths.
Callers normally do not pass panel-specific denominators. If they do, the
mapping is treated only as an assertion and must equal the derived mapping. If
the metadata is absent, the caller must provide the complete mapping explicitly
or class-aware allocation fails before the spatial lookup. It must not silently
estimate missing denominators from observed points.

### Output metrics

In the example above, the output columns are:

```text
n_endogenous
n_negative
n_system_control
negative_per_target
system_control_per_target
control_fraction
```

The class count columns contain assigned transcript counts. The normalized
control columns contain the corresponding count divided by the authoritative
panel target count for that control class. For this three-class configuration:

```text
negative_per_target = n_negative / 10
system_control_per_target = n_system_control / 197

control_fraction =
    (n_negative + n_system_control)
    / (n_endogenous + n_negative + n_system_control)
```

### Table-local metadata contract

Record the resolved configuration and generated-column bindings under one
dedicated table-local key:

```python
adata.uns["feature_class_allocation"] = {
    "schema_version": 1,
    "source_kind": "harpy_allocate",
    "feature_column": "gene",
    "class_column": "code_class",
    "expression_class": "Endogenous",
    "categories": ["Endogenous", "Negative", "SystemControl"],
    "control_class_denominators": {
        "Negative": 10,
        "SystemControl": 197,
    },
    "count_columns": {
        "Endogenous": "n_endogenous",
        "Negative": "n_negative",
        "SystemControl": "n_system_control",
    },
    "normalized_columns": {
        "Negative": "negative_per_target",
        "SystemControl": "system_control_per_target",
    },
    "control_fraction_column": "control_fraction",
    "regions": {
        "sample_a_cellpose_labels_mosaic_1": {
            "points_element": "sample_a_transcripts_mosaic_1",
            "coordinate_system": "sample_a_global_1",
        },
        "sample_b_cellpose_labels_mosaic_1": {
            "points_element": "sample_b_transcripts_mosaic_1",
            "coordinate_system": "sample_b_global_1",
        },
    },
}
```

This follows the table-local convention used by Harpy feature matrices and
napari-harpy canonical centers: a dedicated semantic `.uns` key owns a
versioned schema that identifies its generated payload and sources. Do not wrap
this record in `adata.uns["harpy"]`; the root
`sdata.attrs["harpy"]["metadata_version"]` governs a different SpatialData-level
metadata contract and may not accompany an AnnData table used independently.

There is only one class-aware expression matrix and one coherent set of class
summaries per table, so `feature_class_allocation` is a direct record rather
than a registry keyed by an arbitrary artifact name. Its generated-column
mappings bind the metadata to the actual `.obs` payload instead of requiring
downstream code to reconstruct names. The complete feature-panel target lists
remain in SpatialData root metadata and are not duplicated into the table;
only the resolved non-expression denominators needed to interpret normalized
values are retained.

Configuration shared by the complete table lives at the top level. The
`regions` mapping is keyed by the exact labels element name and records the
paired points element and coordinate system used for each region in the same
allocation call. Class-aware allocation creates the complete record once;
ordinary allocation does not create this record.

### Single-pass implementation

Do not create separate endogenous and control points elements, and do not run
the point-to-label spatial lookup once per class. That lookup is the expensive
part of allocation.

Generalize the private allocation primitive so it can retain both the gene
column and the feature-class column while assigning each point to the label
value underneath its coordinates. Perform one spatial lookup per normalized
labels/points/coordinate-system pair, never one lookup per feature class. Each
resulting lazy assigned-points dataframe should feed its reductions in one Dask
computation:

1. validate that every observed target maps to exactly one feature class;
2. group assigned points by instance and target to build one temporary sparse
   matrix containing endogenous and control targets;
3. calculate per-instance class counts by summing the temporary matrix columns
   belonging to each class;
4. calculate the normalized control metrics and `control_fraction`;
5. retain only the `expression_class` columns in the final `adata.X`; and
6. attach the count and normalized metrics to the corresponding `.obs` rows.

After all pairs have been reduced, align their sparse matrices to the previously
resolved shared feature axis, stack them row-wise, concatenate `.obs` and
spatial coordinates in pair order, and construct one AnnData object. Add that
table to `SpatialData` exactly once. "One allocation call" therefore does not
require one monolithic Dask graph across every mosaic; pair-level work may stay
independent and out of core until the compact sparse results are assembled.

Conceptually:

```python
n_endogenous = X_all[:, endogenous_columns].sum(axis=1)
n_negative = X_all[:, negative_columns].sum(axis=1)
n_system_control = X_all[:, system_control_columns].sum(axis=1)

adata = AnnData(X=X_all[:, endogenous_columns], ...)
```

The temporary control columns add few sparse entries compared with the
endogenous matrix and are discarded before the table is written. No complete
transcript dataframe or dense instance-by-target matrix may be materialized in
memory.

Coordinates in `adata.obsm[spatial_key]` should continue to mean the average
position of expression transcripts and must therefore be calculated from the
selected `expression_class`, not from control points. Preserve the current
allocation row contract: an instance without an assigned endogenous transcript
does not receive an expression-table row. Reindex control summaries to the
endogenous row set and fill missing control counts with zero.

Unexpected or null feature classes, targets associated with multiple classes,
non-positive resolved panel target counts, a missing expression class, and
collisions with existing output columns must produce clear errors. A panel
control class with no assigned points is valid and must produce zero counts and
rates. Validate the complete multi-region request and shared
`feature_class_allocation` configuration before writing the output table.

### Boundary with Slice 4

The `.obs` summaries describe only control points that land inside an instance
mask. They are suitable for cell-level histograms, violin plots, and a hexbin
comparison of `negative_per_target` against `system_control_per_target`.

They are not sufficient for a spatial background map. Allocation deliberately
removes points on label value zero, while unassigned controls outside masks are
still informative about sticky tissue, optical crowding, and regional assay
background. A later QC operation should therefore bin the original control
points directly in space and visualize separate normalized `Negative` and
`SystemControl` density maps. This spatial operation must not be folded into
`hp.tb.allocate`.

### Verification

Focused tests should establish that:

- the final `.var` and `adata.X` contain only the selected expression class;
- `n_endogenous` equals the row sum of the final expression matrix;
- assigned control counts are correct and zero-filled for instances without a
  control call;
- normalized rates use the authoritative panel target counts;
- `control_fraction` is correct and finite;
- non-categorical and unknown-categorical feature-class columns are rejected;
- missing and additional denominator keys are rejected against the complete
  category set;
- category-derived output names are deterministic and collisions are rejected;
- metadata-resolved denominators produce the same result as an equivalent
  explicit denominator mapping;
- explicit denominators that conflict with attached panel metadata are
  rejected;
- conflicting target-to-class mappings and invalid class configuration fail
  before writing a table element;
- the versioned `feature_class_allocation` record and its generated-column
  bindings survive a SpatialData Zarr round trip;
- scalar points and coordinate-system inputs broadcast across labels, while
  incompatible list lengths and duplicate labels are rejected;
- multiple allocation pairs create one table and one complete `regions`
  mapping in a single call;
- pairs from different samples retain their sample-prefixed region, points,
  and coordinate-system bindings, even when their local instance IDs overlap;
- the final expression matrix uses the panel-defined feature axis when panel
  metadata is present and the sorted union of observed targets otherwise;
- expression targets missing from one allocation pair are zero-filled rather
  than removed by an inner join;
- mixed feature-panel availability and incompatible panels are rejected before
  spatial lookup;
- one shared denominator mapping applies to every pair and conflicts from any
  referenced panel are rejected;
- an existing output table is replaced only when `overwrite=True`; and
- omitting class-aware arguments reproduces the existing allocation result.

Benchmark the class-aware path on a representative backed crop before the full
run. Compare wall time, peak worker memory, task count, and output-table size
with ordinary allocation. The additional summaries should be small relative to
the point-to-label lookup and target-count reduction.

## Slice 4: original-point control QC

**Status: specified; not implemented.**

Add separate lightweight QC functions over the original transcript points.
This slice is implemented after Slice 3 to keep delivery sequential, but its
runtime contract depends only on the points and feature-panel metadata from
Slice 1 and the sample-aware point metadata from Slice 2. Users may run it
before or after allocation because it does not depend on an instance-label
raster or an AnnData table.

This operation complements the instance-level `.obs` metrics. It must use the
original points element so that controls on label value zero and controls
outside segmented instances remain visible. It must not create another copy of
the points or route individual controls through `hp.tb.allocate`.

### Per-target summary

Produce one small summary row per control target, sample, and points
element/mosaic, containing at least:

- target name;
- authoritative feature class;
- detected point count;
- fraction of the corresponding control-class calls; and
- density per analyzed area when a physical coordinate system is available.

Use the feature-panel metadata to include control targets with zero detections.
Concretely, aggregate the observed control points by sample, points element,
class, and target; reindex that result against the authoritative target names
for every control class; and fill absent counts with zero. This must represent
both a target that is absent from one mosaic and a target that has no detections
anywhere in a sample. Do not identify controls from target-name prefixes. A
ranked bar or dot plot of these summaries should make a single unusually noisy
negative probe or false code immediately visible. Keep `Negative` and
`SystemControl` in separate facets because they measure different technical
processes and have different numbers of panel targets.

### Spatially binned summary

Bin the original control points in the coordinate system of their mosaic and
produce separate spatial density layers for negative probes and system-control
codewords. Normalize each layer by its authoritative number of panel targets;
when operating in a physical coordinate system, also normalize by bin area.

The bin size must be configurable in coordinate-system units. Choose a default
that yields a QC overview rather than single-transcript resolution; for this
dataset, approximately 100-250 micrometres is an appropriate range to evaluate.
Bins with no controls must remain explicit zeros, and sample/mosaic groups must
remain in their independent coordinate systems.

The primary visualization should be a matched pair of spatial heatmaps with a
shared tissue outline or morphology context:

- elevated negative-probe density highlights nonspecific hybridization or
  sticky tissue regions; and
- elevated system-control density highlights spot-calling, optical-crowding,
  or barcode-decoding errors.

Do not render every control transcript as the default visualization. An
optional point overlay can remain a diagnostic for a selected crop, but the
production overview should operate on aggregated bins.

### Computation and outputs

Project only the feature, class, and coordinate columns required from the
backed points element. Construct the per-target and spatial-bin reductions from
the same lazy Dask input and compute them together so the Parquet partitions do
not need an independent full scan for each output. Do not materialize the full
points dataframe in memory.

Keep computation separate from plotting. The computation layer should expose a
small per-target dataframe and coordinate-aware binned arrays that plotting can
consume without re-reading the transcript points. The exact public names and
SpatialData storage representation remain an implementation decision; they
must not place control targets in the endogenous expression matrix or attach
spatial bins to the instance-annotating AnnData table.

If authoritative panel metadata is unavailable, emit raw per-target counts and
raw spatial class counts, clearly mark them as unnormalized, and omit
per-target normalized rates. Never estimate panel denominators from detected
targets.

### Verification

Focused tests should establish that:

- unassigned and outside-mask control points contribute to the summaries;
- panel controls with zero detections appear in the per-target result;
- per-target counts sum to their corresponding raw class totals;
- spatial-bin counts conserve the input control-point totals within the chosen
  extent;
- normalization uses authoritative panel counts and physical bin area;
- sample and mosaic coordinate systems remain independent; and
- the implementation stays lazy until the compact summaries are computed.

## Slice 5: scalable point-to-label assignment and reduction

**Status: specified; not implemented.**

Refactor the private execution path used by `hp.tb.allocate` without changing
the public or biological contracts established by Slice 3. This optimization
must remain generic to raster labels and points elements; it must not depend on
CosMx FOV identifiers or reader-specific partition metadata.

### Current scaling limitation

The current `_aggregate` helper is primarily a point-to-label assignment
operation rather than an aggregation. It enumerates every labels-array chunk
and builds a complete points-dataframe bounding-box query for each chunk. With
`C` labels chunks and `P` effective points partitions, the graph therefore
contains approximately `C * P` spatial-filter tasks. The Parquet read nodes may
be shared within one computation, but every decoded points partition still
feeds many predicates, increasing graph fan-out, CPU work, scheduler pressure,
and the lifetime of intermediate partitions.

On a representative backed mosaic, the natural 78 labels chunks produced 1,170
spatial-query tasks and approximately 2,000 assignment-graph tasks. Virtually
rechunking the same raster to 1,024-pixel blocks produced 1,150 labels chunks,
17,250 spatial-query tasks, and approximately 28,000 graph tasks before the
downstream reductions. These figures are diagnostic baselines rather than
stable unit-test expectations.

The current `dd.from_delayed` call also omits `meta`. Dask consequently computes
the first delayed partition to infer its schema, so constructing the supposedly
lazy assignment can perform source I/O. The optimized implementation must not
read points or labels merely to build the graph.

### Separate assignment from reduction

Replace the overloaded private helper with two explicit stages:

```python
assigned_points = _assign_points_to_labels(
    labels=...,
    points=...,
    value_keys=...,
    to_coordinate_system=...,
)

aggregates = _aggregate_assigned_points(assigned_points, ...)
```

`_assign_points_to_labels` assigns the raster value underneath each point and
filters label value zero. It accepts all value columns needed by the caller,
rather than a single `value_key`, so ordinary and class-aware allocation use the
same spatial lookup. `_aggregate_assigned_points` owns the count and coordinate
reductions. The exact private names may change during implementation, but this
separation of responsibilities is required.

### Chunk-aware assignment

Use the existing scale-zero labels chunks by default. Given their cumulative
boundaries, the assignment stage should:

1. project only the required point columns and normalize point coordinates
   once;
2. filter points against the complete labels extent once, using half-open
   bounds;
3. calculate each point's labels-chunk indices with the cumulative chunk
   boundaries, including irregular final chunks;
4. encode those indices as one temporary `block_id`;
5. partition the points once by `block_id` using an appropriate Dask range or
   peer-to-peer shuffle;
6. pair each points bucket with exactly its corresponding delayed labels
   chunk;
7. perform one vectorized label lookup for the bucket; and
8. discard background assignments and the temporary block columns as early as
   possible.

Conceptually:

```text
points
  │
  ├── coordinate normalization and one extent filter
  │
  ├── calculate block_id once
  │
  └── partition once by block_id
                    │
                    ├── block 0 + labels chunk 0 ── lookup
                    ├── block 1 + labels chunk 1 ── lookup
                    └── ...
```

This replaces repeated full-dataframe predicates with one linear block
classification and one redistribution of the points. Supply explicit Dask
`meta` throughout. Do not materialize the complete points dataframe, the
complete labels raster, or a dense instance-by-target matrix. Preserve the
existing coordinate-system contract initially: points are identity-transformed
in the selected coordinate system and labels may differ by a pixel-aligned
translation. Validate any coordinate-to-pixel rounding and translation
assumptions explicitly rather than relying on integer truncation.

The existing `chunks` option may still request a virtual labels rechunk, but
the natural stored chunks remain the default. Benchmark virtual rechunking
carefully because smaller blocks reduce the size of each labels task while
increasing the number of point buckets, shuffle partitions, and scheduler
tasks.

### Combined reductions

For each normalized labels/points/coordinate-system pair, derive feature counts
and coordinate statistics from the same assigned-points dataframe. Prefer one
grouped intermediate keyed by instance and target, carrying at least transcript
count and the coordinate sums/count needed for instance means. In class-aware
mode, coordinate statistics use only the configured expression class, as
specified by Slice 3.

Derive instance coordinates from the compact grouped result instead of running
a second independent groupby over every assigned point. Retain sparse
instance-by-target construction and compute all compact Dask reductions
together so the assignment graph is executed once. Pair-level work remains
independent; the shared feature-axis alignment and final row-wise stacking from
Slice 3 are unchanged.

### Performance contract and verification

Focused correctness tests should establish that:

- optimized assignment matches a simple in-memory reference for 2D labels;
- half-open chunk edges assign every in-bounds point exactly once;
- background and out-of-bounds points are excluded;
- irregular final chunks and empty spatial buckets are handled correctly;
- supported translated coordinate systems produce the expected raster lookup;
- multiple retained value columns survive assignment with their dtypes and
  categorical metadata intact;
- ordinary and class-aware allocation produce the same counts, rows,
  coordinates, `.obs` metrics, and table metadata as before the refactor; and
- graph construction performs no point or labels source reads.

Do not make unit tests depend on Dask layer names or an exact task count. Use a
separate benchmark and source-read instrumentation to compare the old and new
implementations across increasing point-partition and labels-chunk counts.
Record at least wall time, peak worker memory, graph construction time, task
count, bytes read, shuffle bytes, and spill volume. Include both a small case,
where shuffle overhead can dominate, and a representative full backed mosaic.

The optimized path is acceptable when it preserves exact allocation results,
remains lazy during graph construction, avoids the `C * P` predicate fan-out,
and materially improves the representative large-mosaic workload without a
major regression on the small case.
