# Class-aware transcript allocation

## Status

Three implementation slices are planned; Slice 1 is implemented:

1. patch the CosMx reader and establish the generic Harpy feature-panel
   metadata contract — implemented;
2. add class-aware aggregation to `hp.tb.allocate`; and
3. add QC functions that summarize the original, unallocated control points.

Slice 2 consumes the metadata produced by Slice 1, but also supports an explicit
denominator mapping for generic points. Slice 3 is implemented last but depends
only on Slice 1, not on allocation, instance labels, or an AnnData table.

## Goal

Establish a general control-aware transcript workflow: readers preserve
authoritative panel information, allocation creates an AnnData expression
matrix containing only the selected biological class while retaining compact
per-instance control summaries in `.obs`, and separate QC functions summarize
all original control points. The raw points element remains unchanged and
continues to contain biological and control transcripts.

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
            # Existing CosMx reader, source, and run-selection fields live here.
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
applies to the complete Harpy root contract. The CosMx whole-store overwrite
safety check must specifically require `harpy.provenance.reader == "cosmx"`;
the mere presence of Harpy metadata is not sufficient evidence that the store
is replaceable by that reader.

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

Slice 2 only needs those derived counts: it uses the length of each class target
list as the denominator for normalized control metrics. Slice 3 additionally
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
Slice 2 can restore a known categorical dtype lazily without scanning the
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

## Slice 2: class-aware `hp.tb.allocate`

**Status: specified; not implemented.**

This slice consumes the generic feature-panel contract established by Slice 1.
It must remain usable with non-reader points through an explicit denominator
mapping, and it must preserve ordinary allocation behavior when class-aware
arguments are omitted.

### Public contract

Extend `hp.tb.allocate` with optional class-aware arguments following this
contract:

```python
sdata = hp.tb.allocate(
    sdata,
    labels_name="cellpose_labels_mosaic_1",
    points_name="transcripts_mosaic_1",
    output_table_name="table_transcriptomics",
    to_coordinate_system="global_1",
    name_gene_column="gene",
    name_feature_class_column="code_class",
    expression_class="Endogenous",
    control_class_denominators=None,
    overwrite=True,
)
```

The exact parameter names remain subject to implementation review, but the
semantics are fixed:

- `name_feature_class_column` identifies the per-point column that classifies
  each target;
- `expression_class` selects the only class whose targets are retained in
  `adata.X`; and
- `control_class_denominators` optionally supplies the authoritative number of
  panel targets for every non-expression class. When it is `None`, allocation
  resolves the same mapping from metadata associated with `points_name`.

An ordinary reader-backed call should not contain panel-specific constants. An
explicit mapping remains available for generic points elements that were not
created by a reader supplying feature-panel metadata:

```python
control_class_denominators={
    "Negative": 10,
    "SystemControl": 197,
}
```

When `name_feature_class_column=None`, allocation must branch to the existing
implementation before performing any class-specific projection, validation, or
aggregation. No additional `.obs` columns are created and every
`name_gene_column` value remains in `adata.X`. Other class-aware parameters must
also be `None` in this mode so that a partially specified request is not
silently ignored.

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
- after metadata resolution or application of an explicit override, the keys
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
Callers do not pass panel-specific denominators. If the metadata is absent, the
caller must provide the complete mapping explicitly or class-aware allocation
fails before the spatial lookup. It must not silently estimate missing
denominators from observed points.

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

The resolved panel class counts and normalization semantics should be recorded
compactly in the output table's `.uns` so that the normalized `.obs` values
remain interpretable after round-tripping the SpatialData Zarr store.

### Single-pass implementation

Do not create separate endogenous and control points elements, and do not run
the point-to-label spatial lookup once per class. That lookup is the expensive
part of allocation.

Generalize the private allocation primitive so it can retain both the gene
column and the feature-class column while assigning each point to the label
value underneath its coordinates. The resulting lazy assigned-points dataframe
should feed all reductions in one Dask computation:

1. validate that every observed target maps to exactly one feature class;
2. group assigned points by instance and target to build one temporary sparse
   matrix containing endogenous and control targets;
3. calculate per-instance class counts by summing the temporary matrix columns
   belonging to each class;
4. calculate the normalized control metrics and `control_fraction`;
5. retain only the `expression_class` columns in the final `adata.X`; and
6. attach the count and normalized metrics to the corresponding `.obs` rows
   before adding the table to `SpatialData`.

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
rates. Appending another labels region must preserve the same `.obs` schema and
compatible panel metadata.

### Boundary with Slice 3

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
- conflicting target-to-class mappings and invalid class configuration fail
  before writing a table element;
- append mode preserves the metrics and configuration across multiple mosaic
  regions; and
- omitting class-aware arguments reproduces the existing allocation result.

Benchmark the class-aware path on a representative backed crop before the full
run. Compare wall time, peak worker memory, task count, and output-table size
with ordinary allocation. The additional summaries should be small relative to
the point-to-label lookup and target-count reduction.

## Slice 3: original-point control QC

**Status: specified; not implemented.**

Add separate lightweight QC functions over the original transcript points.
This slice is implemented after Slice 2 to keep delivery sequential, but its
runtime contract depends only on the points and feature-panel metadata from
Slice 1. Users may run it before or after allocation because it does not depend
on an instance-label raster or an AnnData table.

This operation complements the instance-level `.obs` metrics. It must use the
original points element so that controls on label value zero and controls
outside segmented instances remain visible. It must not create another copy of
the points or route individual controls through `hp.tb.allocate`.

### Per-target summary

Produce one small summary row per control target and mosaic, containing at
least:

- target name;
- authoritative feature class;
- detected point count;
- fraction of the corresponding control-class calls; and
- density per analyzed area when a physical coordinate system is available.

Use the feature-panel metadata to include control targets with zero detections.
Concretely, aggregate the observed control points by mosaic, class, and target;
reindex that result against the authoritative target names for every control
class; and fill absent counts with zero. This must represent both a target that
is absent from one mosaic and a target that has no detections anywhere in the
run. Do not identify controls from target-name prefixes. A ranked bar or dot
plot of these summaries should make a single unusually noisy negative probe or
false code immediately visible. Keep `Negative` and `SystemControl` in separate
facets because they measure different technical processes and have different
numbers of panel targets.

### Spatially binned summary

Bin the original control points in the coordinate system of their mosaic and
produce separate spatial density layers for negative probes and system-control
codewords. Normalize each layer by its authoritative number of panel targets;
when operating in a physical coordinate system, also normalize by bin area.

The bin size must be configurable in coordinate-system units. Choose a default
that yields a QC overview rather than single-transcript resolution; for this
dataset, approximately 100-250 micrometres is an appropriate range to evaluate.
Bins with no controls must remain explicit zeros, and mosaic groups must remain
in their independent coordinate systems.

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
- mosaic coordinate systems remain independent; and
- the implementation stays lazy until the compact summaries are computed.
