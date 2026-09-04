# Class-aware transcript allocation

## Status

Thirteen implementation slices are planned; Slices 1 through 9 are implemented:

1. patch the CosMx reader and establish the generic Harpy feature-panel
   metadata contract — implemented;
2. make the canonical `harpy.io.cosmx()` creation API sample-aware —
   implemented;
3. validate existing sample-aware CosMx SpatialData stores — implemented;
4. add new CosMx samples incrementally to a validated sample-aware store —
   implemented;
5. add class-aware aggregation to `hp.tb.aggregate_points` — implemented;
6. preserve per-feature non-expression aggregates and use label centers of
   mass for the complete assigned-instance row universe — implemented;
7. optimize the generic point-to-label assignment, reduction and backed table
   construction path — implemented;
8. promote napari-harpy's canonical-center implementation into Harpy and
   integrate it with `hp.tb.aggregate_points` — implemented;
9. add canonical centers to compatible existing SpatialData tables —
   implemented;
10. support point-to-label assignment through general invertible SpatialData
    transformations into a shared coordinate system;
11. add QC functions that summarize the original, unallocated control points;
12. support general lazy reopening of persisted AnnData tables through
    SpatialData; and
13. optionally optimize Slice 7b's latency after phase-level benchmarks identify
    material checkpoint or writer overhead.

Slice 2 replaces the current single-run reader surface with one coherent,
sample-aware creation contract. Slice 3 validates that an existing store still
satisfies that contract. Slice 4 incrementally extends a validated store without
rebuilding its existing samples. Slice 5 consumes the feature-panel metadata
produced by Slice 1 and the sample-aware element contracts established by
Slices 2–4. Class-aware allocation requires that every selected points element
reference authoritative feature-panel metadata; generic points without that
metadata remain supported by the ordinary, non-class-aware path. Slice 6
preserves the public behavior of `expression_class` while revising the
class-aware table payload so per-feature non-expression counts and auxiliary-only
instances are retained. Slice 7a optimizes the private point-to-label assignment
while preserving that payload through the then-existing in-memory path. Slice
7b replaces that path with one backed-only implementation for partitioned
reduction and out-of-core table construction.
Slice 8 establishes a Harpy-owned canonical-center calculation and metadata
contract that napari-harpy can consume without a reverse dependency. Newly
aggregated 2D and 3D tables receive canonical centers through the same contract;
the 2D case is immediately usable by canonical-center spatial queries, while
the annotation-query boundary explicitly rejects 3D sources. Slice 9 provides
the same contract for compatible tables created outside `aggregate_points`
without making canonical-center attachment a required post-processing step for
new aggregation tables. Slice 10 generalizes the spatial-assignment contract by
mapping point coordinates through the selected shared coordinate system into
the intrinsic labels frame, without resampling the labels raster. Slice 11's
original-point summaries depend on the reader metadata from Slices 1–4 rather
than aggregation or labels; its optional per-instance plotting view derives
temporary rates from the class-aware table. Slice 12 is an independent
integration follow-up that makes later SpatialData Zarr reads retain lazy
AnnData matrices. It is not required for Slice 7b's out-of-core writing or
same-process result. Slice 13 is an
optional, benchmark-driven follow-up: it may reduce repeated checkpoint reads
and small-partition overhead, but must preserve Slice 7b's bounded-memory and
publication contracts.

## Goal

Establish a general control-aware transcript workflow: readers preserve
authoritative panel information for one or more sample-scoped runs, allocation
creates an AnnData expression matrix containing only the selected biological
class, retains sparse per-feature non-expression counts in an auxiliary
observation-aligned matrix, keeps compact per-instance class summaries in
`.obs`, and uses segmentation-mask centers of mass as instance coordinates.
Separate QC functions summarize all original control points. The raw points
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

## Compatibility policy

Slices 1–4 define the canonical CosMx reader contract. The superseded reader
API, unprefixed element names, metadata layout, and stores created from them are
not backward-compatibility constraints. Breaking changes are acceptable when
they produce a smaller and more coherent sample-aware API.

Do not add deprecated aliases, dual metadata layouts, automatic store
migrations, or special legacy branches solely to preserve superseded
implementations. Stores that do not satisfy the canonical contract may be
rejected with a clear message and rebuilt using the current reader. The same
sample-aware API handles one or many samples; do not establish a second
single-sample entry point or unprefixed format.

This does not relax product quality. The resulting contract must remain
explicit, deterministic, validated, documented, tested, and safely versioned.
Any future backward-compatibility and deprecation policy must preserve those
same requirements.

## Slice 1: reader and feature-panel metadata

**Status: implemented.**

This slice changes the CosMx reader and the metadata it writes. It does not
change `hp.tb.aggregate_points` or introduce QC computations.

### Authoritative panel metadata

The number and meaning of control targets belong to the assay panel. They are
not general CosMx constants and must not be guessed from class names or inferred
from detected transcript rows.

The investigated CosMx export contains a small run-level plex file with the
columns `DisplayName`, `CodeClass`, and `ProbeID`. It contains 1,165 unique
targets:

| `CodeClass`     | Panel targets |
| --------------- | ------------: |
| `Endogenous`    |           958 |
| `Negative`      |            10 |
| `SystemControl` |           197 |

The CosMx reader currently does not need the plex to create transcript points,
because each detected transcript already carries its target and code class.
Class-aware allocation and later control QC introduce a distinct reason to
consume it: unlike the detected transcript rows, the plex also represents
targets with zero detections. A denominator derived from observed targets would
be biased whenever one of those targets has no calls.

When transcripts are ingested and a plex file is present, the CosMx reader
should therefore discover exactly one plex file, read it once, and associate a
compact feature-panel record with every transcript points element created from
that run. A missing plex must not prevent raw transcript ingestion, but it
precludes class-aware allocation and panel-normalized QC. The record must
contain at least:

- the points `feature_key` and `feature_class_key` column bindings;
- the ordered feature classes; and
- the authoritative feature names grouped by class.

Introduce a vendor-neutral Harpy metadata convention in the root
`SpatialData.attrs`. This is a new Harpy contract, not an existing SpatialData
standard. All metadata written and consumed by Harpy belongs under the single
top-level `harpy` namespace. Do not retain a parallel top-level `cosmx`
metadata tree: the originating reader is identified by
`harpy.provenance.reader`, while downstream operations resolve modality
metadata by SpatialData element type and exact element name.

The namespace has a versioned, element-keyed structure. The CosMx reader must
write its provenance, image, label, and transcript records in this structure
when feature-panel support is implemented:

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
                "feature_key": "gene",
                "feature_class_key": "code_class",
                "classes": ["Endogenous", "Negative", "SystemControl"],
                "features_by_class": {
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

The `classes` order is the categorical dtype order shared by all associated
points elements. `features_by_class` keys must exactly equal those classes;
each feature list must contain unique, non-empty strings; and no feature may
occur under more than one class. Derive feature counts from the list lengths
instead of storing a second potentially inconsistent count mapping. The CosMx
reader sorts both class names and the features within each class
lexicographically so output does not depend on row order in the plex; this
ordering is deterministic rather than a claim of biological precedence.

Slice 5 uses the complete relation to resolve its shared expression axis and
feature classes, and retains each non-expression feature-list length as a
table-local auxiliary-class feature-count snapshot for later QC. Slice 11 additionally uses the
actual control-feature names. A categorical transcript column
contains only categories represented by the ingested points and cannot, by
itself, preserve the feature-to-class relationship for a panel feature with no
detected rows. Keeping the authoritative names in `features_by_class`
therefore supports both uses without storing a separate count mapping.

Only the allocation- and QC-relevant `DisplayName`/`CodeClass` relationship is
persisted. Do not store unused plex fields such as `ProbeID`, and do not
duplicate the panel under every mosaic record. Although the example originates
from CosMx, no key in the Harpy contract is vendor-specific. Other readers can
associate their points with the same structure using their own classes and
features.

Validate that plex display names are unique, class values are non-null, and
class feature counts are positive. Prefix matching is not an acceptable
fallback: this panel contains a feature named `NegativeAdd` whose authoritative
`CodeClass` is `Endogenous`.

For every transcript points element that references a shared feature-panel
record, validate the points payload against that specific record: every
detected feature in the points element must occur in the panel, and the
detected feature's class value must equal the class assigned to that feature by
the panel. This is a one-way inclusion requirement; authoritative panel
features with zero detected transcripts are valid and remain represented only in the
shared panel metadata. When no panel is available, omit the reference and skip
this cross-validation.

The CosMx reader stores `code_class` categorically with the same category set
for every mosaic points element from the run. Its categorical values come from
the plex `CodeClass` values. Parquet preserves those values, but a reopened
Dask dataframe may report them as unknown until supplied with the authoritative
class list. That list is persisted in the shared feature-panel metadata so
Slice 5 can restore a known categorical dtype lazily without scanning the
points. The categorical representation is the canonical contract for data
created by this reader; no compatibility path is required for stores that use
the superseded Arrow-string representation.

### Verification

Focused reader tests should establish that:

- one valid plex is read once and stored as one shared feature-panel record;
- every transcript points element references that shared panel without
  duplicating it;
- every feature and feature-class pair represented in a transcript points
  element agrees with its referenced shared panel record, while panel features
  with zero detections remain valid;
- the persisted `code_class` column has known categorical categories matching
  the panel;
- feature names, classes, and zero-detection panel features survive a SpatialData
  Zarr round trip;
- duplicate plex files, duplicate or empty feature names, null classes,
  conflicting feature-to-class mappings, and invalid panel classes are
  rejected before transcript materialization;
- a missing plex still permits raw transcript ingestion but creates no
  feature-panel reference;
- provenance and all image, label, and points metadata are written to the
  versioned, element-keyed `harpy` namespace with no parallel top-level
  `cosmx` metadata; and
- whole-store overwrite is permitted only when
  `harpy.provenance.reader == "cosmx"`.

## Slice 2: sample-aware `harpy.io.cosmx()` creation API

**Status: implemented.**

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
    coordinate_system: str = "global"
    flip_x: bool = True
    flip_y: bool = False
```

Although the public constructor accepts general sequences, normalize `fovs`
and `channels` to tuples during `CosmxSample` construction. The frozen sample
configuration must therefore not retain references to caller-owned mutable
lists. Preserve `None` as `None`, preserve sequence order, and perform the
normalization before the configuration participates in validation, planning,
or metadata generation.

Change the canonical creation API so that `cosmx()` accepts a `samples` mapping
whose keys are the sample identifiers:

```python
sdata = cosmx(
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

Make `harpy.io.cosmx()` the canonical creation API and replace its current
single-run signature with the sample mapping shown above. `samples` is required
and always uses the same mapping contract. Reading one sample means passing a
one-entry mapping; do not add a separate `cosmx_samples()` alias or
single-sample code path. Every invocation uses the same sample-aware
preparation, naming, metadata, and writing implementation.

The sample mapping must be non-empty and its keys must be unique identifiers
that match exactly:

```text
^[A-Za-z][A-Za-z0-9_]*$
```

Validate identifiers as supplied and never strip, case-fold, replace
characters, or otherwise normalize them automatically. Preserve mapping
iteration order for predictable execution, while ensuring that output metadata
and panel identifiers are deterministic for the same logical inputs. Reject a
sample identifier that would make any planned element or coordinate-system name
collide.

This is a deliberate CosMx reader contract, not a restatement of SpatialData's
broader element-name validation. SpatialData also permits names containing
hyphens and dots and does not require an initial ASCII letter. Sample
identifiers use the stricter grammar because they are structured prefixes from
which the reader generates several element and coordinate-system names.

The sample configuration owns values that may differ between runs:

- source path;
- selected FOVs and morphology channels;
- mosaic mode and adjacency tolerance;
- coordinate-system base name; and
- X/Y orientation.

`adjacency_tolerance_px` applies only to
`mosaic_mode="spatial_groups"`. When `mosaic_mode="single"`, normalize the
value to `None` before constructing the preview and persist `None` in the
element-level mosaic metadata. Do not reject an explicitly supplied tolerance:
single-mosaic mode deliberately constructs one mosaic without
adjacency-based grouping, so that value has no effect.

Arguments that define the complete output remain on `cosmx`: output
path, modality inclusion, output base names, image and label chunks, raster
scale factors, points block size, and overwrite behavior. Do not accept a
list of these output-wide values. `coordinate_system` is not output-wide; it
belongs to each `CosmxSample`. Require at least one enabled modality.

For each sample, derive one common FOV set from only the enabled modalities:

```text
included FOVs
    = requested FOVs
    ∩ positioned FOVs
    ∩ FOVs available for every enabled modality
```

A missing disabled product must not exclude an otherwise usable FOV. For
example, `images=True` with all other modality flags disabled does not require
instance labels, compartment labels, or transcript files. Conversely, when
`images` and `points` are enabled, an included FOV must provide both morphology
images and transcript files. Every enabled modality is then constructed from
the same included FOVs and mosaic geometries, so corresponding image, labels,
and points elements remain spatially aligned.

Known FOV positions, pixel size, tile dimensions, and morphology TIFF shape
remain mandatory regardless of which payload modalities are enabled because
mosaic construction requires that geometry. Validate morphology channel order
and dtype only when morphology images are enabled. Validate label dtype and
instance-ID encoding only when their corresponding label outputs are enabled.
A per-sample `channels` selection has no effect when `images=False`.

### Sample-scoped elements and coordinate systems

Prefix every element and coordinate system created by `cosmx()` with its sample
identifier, including when `samples` contains only one entry. For example:

```text
sample_a_morphology_image_mosaic_1
sample_a_instance_labels_mosaic_1
sample_a_compartment_labels_mosaic_1
sample_a_transcripts_mosaic_1

sample_a_global_1
sample_a_global_1_micron
```

Treat `CosmxSample.coordinate_system` as a base name rather than a complete
SpatialData coordinate-system name. For mosaic `n`, construct:

```text
<sample_id>_<coordinate_system>_<n>
<sample_id>_<coordinate_system>_<n>_micron
```

The default base name `"global"` therefore works without additional user
configuration for any number of samples: samples `sample_a` and `sample_b`
receive `sample_a_global_1` and `sample_b_global_1`, respectively. Do not
require the raw base names to differ between samples. Require every
`coordinate_system` base name to match the same exact
`^[A-Za-z][A-Za-z0-9_]*$` grammar as a sample identifier. Validate the supplied
value without trimming or otherwise normalizing it. Reject any collision among
the fully generated coordinate-system names during planning.

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

`reader_version` is the Harpy version that most recently committed a CosMx
element and its metadata to the store. `metadata_version` remains the
compatibility gate for the Harpy metadata schema; `reader_version` records the
writer implementation and must not be used as a schema-version substitute. A
staged create-or-replace operation records the current Harpy version before
publishing the completed store.

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
    "acquisition_timestamp": "20260312_140910_S2",  # optional source value
    # Existing modality-specific orientation, origin, scale, and channel data.
}
```

The FOV list describes the exact source tiles represented by that element; it
is not a duplicate invocation-level selection record. A points element keeps
its `feature_panel` reference alongside this sample-scoped metadata. The
`sample_id` field and sample-prefixed names are required for every element
created through `cosmx`, including a call whose mapping contains one sample.
When every morphology TIFF for a sample provides the same non-empty
`OrigTimeStamp`, preserve that source value verbatim as the optional
`acquisition_timestamp` on every element record for the sample. If the value is
absent or inconsistent, omit it and log a warning for inconsistent provided
values; acquisition metadata must not make an otherwise readable run fail.

### Feature panels across samples

Canonicalize every discovered panel using the Slice 1 contract before writing
any points. Samples with identical canonical panel contents should reference
one shared feature-panel record. Samples with different panels must reference
separate records.

Derive the shared record key from the canonical panel contents rather than from
a sample identifier or points-element base name. Serialize the existing
allocation-facing panel record as canonical UTF-8 JSON: mapping keys are sorted,
compact separators are used, category order is retained, and each class's
already canonical target list remains sorted. Hash those bytes with SHA-256 and
use a store-local key of the form:

```text
feature_panel_<first 16 lowercase SHA-256 hex characters>
```

For example, both `sample_a_transcripts_mosaic_1` and
`sample_b_transcripts_mosaic_1` reference the same key when their canonical
panels are identical, regardless of sample input order. A different panel
produces a different key. Do not include `sample_id`, `output_points_name`, or
the generated points-element name in the hash or key: those values describe a
consumer of the panel, not the panel itself.

When the derived key already exists, compare its complete canonical panel
contents. Reuse the record when they match and raise a hash-collision error when
they differ. Each points metadata record stores only the resulting
`feature_panel` reference; do not duplicate the panel contents per sample or
per mosaic.

Both the element-keyed points metadata and the shared panel records live in the
root `SpatialData.attrs` Harpy namespace. They are not stored as Parquet
metadata or directly on an individual points object. The agreed reference
structure is:

```text
sdata.attrs["harpy"]
├── points
│   ├── sample_a_transcripts_mosaic_1
│   │   └── feature_panel ───────────────┐
│   ├── sample_a_transcripts_mosaic_2    │
│   │   └── feature_panel ───────────────┤
│   └── sample_b_transcripts_mosaic_1    │
│       └── feature_panel ───────────────┤
│                                        ▼
└── feature_panels
    └── feature_panel_8a31b240c75e
        ├── feature_key: gene
        ├── feature_class_key: code_class
        ├── classes: [...]
        └── features_by_class: {...}
```

In mapping form, the reference is located at
`sdata.attrs["harpy"]["points"][<points element name>]["feature_panel"]`; it is
not a single `sdata.attrs["points"]["feature_panel"]` value. The referenced
record lives at
`sdata.attrs["harpy"]["feature_panels"][<feature panel key>]`. When no
authoritative panel is available, omit both the points-level reference and the
shared panel record.

Update the public `harpy.io.cosmx()` docstring as part of Slice 2 so its metadata
section includes this reference diagram and paths, alongside the complete
sample-aware element metadata contract. Do not document the content-derived
key in the current single-run reader before the implementation changes, because
that would describe metadata it does not yet write.

Sharing a panel record is a storage optimization, not an assertion that the
samples are spatially aligned. Conversely, two different registry keys do not
necessarily make panels incompatible for allocation: Slice 5 compares the
canonical feature key, feature-class key, classes, and feature-to-class
contents. One output table may combine only compatible selected panels.

### Validation and atomic publication

Prepare all samples before writing: discover and validate every manifest,
construct every preview, canonicalize panels, and plan all element names,
coordinate systems, and metadata references. Fail on a configuration or name
collision before decoding raster or transcript payloads.

Refactor the single-sample implementation around reusable internal operations
such as `_prepare_cosmx_sample` and `_write_cosmx_sample`; the exact private
names may differ. The public `cosmx()` orchestrator calls those private
operations once per sample; it must not recursively invoke itself or create and
merge one store per sample. It writes samples sequentially into one staging
store to bound peak memory, reopens and validates the completed SpatialData
object, and publishes the store once. A failure in any sample removes
reader-generated staging data and leaves an existing output store intact.

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
- FOV eligibility intersects positions and only the enabled modality
  availabilities, and all enabled modalities use the resulting common mosaics;
- a missing disabled product does not exclude an FOV, while a missing enabled
  product does;
- single-mosaic samples normalize any supplied adjacency tolerance to `None`
  without rejecting the request;
- per-element metadata records the correct `sample_id`, represented FOVs, and
  mosaic construction settings, while root provenance remains reader-only;
- root `reader_version` records the Harpy version that created and published
  the staged store;
- identical panels are stored once and referenced by both samples, whereas
  incompatible panels remain separate;
- overlapping instance IDs in different labels elements are preserved;
- invalid sample identifiers and all planned name or coordinate-system
  collisions fail before payload materialization;
- failure while writing a later sample leaves an existing destination intact
  and removes staging data; and
- a one-entry `cosmx` call follows the same sample-prefixed naming, coordinate
  system, metadata, and writing contracts as a multi-entry call.

## Slice 3: CosMx SpatialData store validation

**Status: implemented.**

Add a dedicated, non-mutating validator for an existing sample-aware CosMx
SpatialData Zarr store. Incremental addition must establish that the destination
still satisfies the reader's metadata contract before it plans or writes new
samples. Keep this validation reusable outside incremental ingestion so users
can explicitly audit a store after downstream processing.

### Public contract

Expose:

```python
validate_cosmx_store(
    output: str | Path,
    *,
    check_point_contents: bool = True,
) -> None
```

`output` must identify an existing, readable, backed SpatialData Zarr store.
The function returns normally for a valid store and raises a clear `ValueError`
for an invalid contract. It must never create, remove, rewrite, or repair an
element or root attribute. It is a validator, not a migration or recovery API.

Implement one internal validator over an already opened `SpatialData` object so
the public function and Slice 4's incremental writer use exactly the same
rules. The public function opens the path and delegates to that primitive;
`add_cosmx_samples()` reuses the object it has already opened. Place both in
`harpy/io/_cosmx/_validation.py` and export only
`validate_cosmx_store()` through `harpy.io`; source-run discovery validation
remains in its existing modules.

### Structural validation

With `check_point_contents=False`, validate metadata and dataframe schemas
without scanning existing transcript rows:

- the destination is backed and its Zarr store can be read;
- `harpy` is a mapping with the supported `metadata_version`;
- `harpy.provenance` is a mapping whose `reader` is exactly `"cosmx"` and whose
  `reader_version` is a non-empty string;
- `harpy.images`, `harpy.labels`, `harpy.points`, and
  `harpy.feature_panels`, when present, are mappings;
- at least one image, labels, or points element is registered as CosMx-owned;
  provenance without any registered reader-owned element is not a valid CosMx
  store;
- every image, labels, and points registry key names an existing SpatialData
  element of the corresponding type;
- every registered element record contains a valid `sample_id` satisfying the
  Slice 2 identifier contract, plus the required element-level FOV, mosaic,
  source-origin, orientation, and pixel-size fields with their documented
  types;
- image, instance-label, compartment-label, and points-specific metadata fields
  satisfy their documented structural contracts when present;
- every feature-panel record has valid `feature_key`, `feature_class_key`,
  ordered `classes`, and `features_by_class` values; class keys match exactly,
  features are non-empty and unique within and across classes, and the
  registry key equals the identifier recomputed from the canonical panel
  contents;
- every points-level `feature_panel` reference resolves to an existing shared
  panel record; and
- each referenced points element contains the panel-declared feature and class
  columns with categorical dtypes. The structural check may inspect Parquet or
  Dask schema metadata but must not compute transcript partitions.

Treat the documented fields as required and validate their exact value types
and internal constraints, but permit additional keys in element and panel
records. Compatible metadata extensions must not make a store invalid while
the same `metadata_version` remains supported.

Determine a registered labels element's family from its metadata, never from
its configurable element name. Require exactly one family discriminator:

```text
instance labels      -> instance_id_encoding
compartment labels   -> categories
```

Reject a labels record with both or neither discriminator. Validate
`instance_id_encoding` against the reader's fixed background, positive integer
base, and formula contract. Validate `categories` against the fixed CosMx
compartment mapping written by the reader.

A points record without a `feature_panel` reference remains valid and is not
required to contain categorical feature or class columns; this is the supported
missing-plex reader result. For a referenced panel, accept either known or
unknown Dask categorical metadata for the declared columns. Structural
validation establishes the categorical dtype without calling
`.cat.as_known()` or computing a partition. The default content check performs
the projected payload scan.

Root provenance deliberately contains no sample registry. Derive existing
CosMx sample identifiers exclusively from the union of the exact `sample_id`
values stored in the registered image, labels, and points element records. Do
not infer sample identity by parsing element names.

Allow unrelated SpatialData elements that have no Harpy CosMx element record;
the store may have been extended by downstream analysis. They do not establish
CosMx sample identities. The validator checks the reader-owned metadata
contract, while Slice 4 separately includes every existing element and
coordinate system in collision preflight.

### Feature-panel content validation

With the default `check_point_contents=True`, additionally validate every
points element that references a feature panel against its actual transcript
payload. Project only the panel-declared feature and class columns and validate
each Dask partition independently against the authoritative feature-to-class
mapping. Each partition returns at most one small diagnostic, so this check
requires no global shuffle and does not collect transcript rows in the client.
Require:

- every observed feature occurs in the referenced panel;
- every observed feature has exactly the feature class assigned by that panel;
  and
- null features or classes are rejected. A feature associated with multiple
  observed classes is necessarily rejected because at least one observed class
  disagrees with its single authoritative panel assignment.

This remains a one-way inclusion check: authoritative panel features with zero
detections are valid. The content check must not load spatial columns,
materialize a complete points dataframe, or modify categorical metadata. It
necessarily scans the two projected payload columns and can therefore be
substantially more expensive than structural validation.

Slice 4 explicitly invokes validation with `check_point_contents=True` before
source discovery and mutation. Incremental addition therefore re-reads the two
panel-declared columns of existing, panel-associated transcript elements. This
cost is intentional: a store whose existing transcript contents disagree with
their referenced feature panels must not be extended. Callers performing a
standalone lightweight audit can opt out explicitly with
`validate_cosmx_store(..., check_point_contents=False)`.

### Verification

Focused tests should establish that:

- a valid one- or multi-sample store passes without changing elements or root
  attributes;
- unreadable, unbacked, non-CosMx, unsupported-version, and earlier unprefixed
  stores are rejected;
- malformed registries, missing or wrong-type registered elements, invalid
  sample IDs, and malformed required element metadata are rejected;
- unrelated unregistered downstream elements are accepted;
- invalid, content-hash-mismatched, and unresolved feature panels are rejected;
- `check_point_contents=False` checks referenced points schemas without
  computing their partitions;
- default validation accepts detected panel subsets and zero-detection panel
  features;
- default validation rejects unknown features, feature-to-class disagreement
  (including multiple observed classes), and null values without a global
  shuffle; and
- neither successful nor failed validation writes to the destination.

## Slice 4: incremental CosMx sample addition

**Status: implemented.**

Add an explicit incremental API for appending new, independently named CosMx
samples to an existing sample-aware SpatialData Zarr store. This is an additive
operation, distinct from the staged create-or-replace behavior of `cosmx()`.
It must not rewrite or rename samples already in the destination. The only
existing payload data it reads are the two panel-declared columns in
panel-associated transcript elements, as required by the preflight content
validation.

### Public contract

Introduce a separate entry point rather than overloading `overwrite` or adding
an `append` mode:

```python
sdata = add_cosmx_samples(
    output=existing_zarr,
    samples={
        "sample_c": CosmxSample(
            path=sample_c_root,
            channels=["DAPI", "PanCK"],
            adjacency_tolerance_px=85,
        ),
    },
    raster_scale_factors=[2, 2, 2, 2, 2],
)
```

Reuse the `CosmxSample` configuration and the output-wide modality, naming,
chunking, multiscale, and transcript-partition options from `cosmx()`. `output`
must identify an existing, readable, backed
SpatialData Zarr store created through the sample-aware CosMx API. Require a
supported Harpy metadata version and
`harpy.provenance.reader == "cosmx"`.

Reject stores that do not satisfy the final sample-aware metadata and naming
contract, including stores produced by the earlier unprefixed implementation.
They must be rebuilt with `cosmx()` before incremental addition. Do not
guess a sample identifier, silently migrate elements, or add compatibility
logic for stores that use the superseded contract.

The operation is add-only:

- every requested sample identifier must be new;
- all planned element and coordinate-system names must be absent;
- existing samples, elements, transformations, metadata, and feature-panel
  records must not be replaced; and
- the API has no `overwrite` or sample-replacement behavior.

Replacing or removing an existing sample is a separate lifecycle operation and
is outside this slice.

### Preflight and panel resolution

Before discovering requested sources or writing any payload, reopen the
destination and run Slice 3 validation with `check_point_contents=True` on that
backed object. This scans only the panel-declared feature and class columns of
referenced points elements and validates them partition-wise without a global
shuffle. Then prepare every requested sample exactly as in Slice 2: discover
and validate manifests, construct previews, canonicalize panels, and plan
sample-prefixed element names, coordinate systems, and metadata references.

Validate all requested samples against each other and against the destination
before decoding raster or transcript payloads. This includes sample-ID,
element-name, coordinate-system, and modality-type collisions. Reject malformed
existing state rather than building on ambiguity. In particular, reject a
requested sample identifier when it occurs in the validated element metadata,
even if different output base names would avoid an exact element-name
collision. The caller must ensure that no other process writes to the
destination during the operation.

Resolve feature panels against the existing registry using the same canonical
content contract as Slice 2. Reuse an identical existing panel record; create a
new stable record for a different panel. Never modify an existing panel in
place. Do not persist a newly required panel record before its first associated
points element has been written successfully.

### Direct writes and failure contract

Write new elements directly to the backed destination with `overwrite=False`.
Do not copy the complete existing Zarr into staging and do not call
`_publish_staging_store()`. Retain the sibling staging-and-publish workflow for
new-store creation and complete replacement in `cosmx()`; incremental
addition is the only path with the weaker, element-level failure contract.

Process samples and their planned elements sequentially. Each element is its
own commit boundary:

1. construct the element and its metadata in memory;
2. write the element with `overwrite=False`;
3. after that write succeeds, persist the corresponding element metadata and
   update `harpy.provenance.reader_version` to the current Harpy version in the
   same root-attributes write; and
4. continue to the next element.

Never persist an element metadata record before its element exists on disk.
For transcript points, persist the points metadata, its `feature_panel`
reference, and any newly required shared panel record together after the points
element succeeds.

The underlying Harpy element writers already perform best-effort removal of the
current element when its write fails. If persisting metadata fails after an
element write, make a best-effort attempt to delete only that newly written
element and restore the preceding root attributes, then re-raise. Do not roll
back other elements from the current sample: elements completed earlier remain
on disk together with their already persisted metadata.

Do not introduce pending-sample records or claim sample-level transactional,
crash-atomic, or automatic recovery behavior. Process termination, loss of
storage, or failed cleanup can still leave incomplete state. A later call must
reject collisions with existing sample IDs, element names, or coordinate
systems; it must not infer whether a previous ingestion was interrupted,
overwrite existing elements, or resume automatically. The user decides whether
to retain or explicitly remove completed elements, or rebuild the store.

Root provenance remains the minimal reader/version record and is not rewritten
as a sample registry. Because successfully written elements are retained when a
later element fails, `reader_version` is updated at each successful element
commit rather than only when the complete incremental call returns.

### Shared implementation

Both creation and incremental addition must use the same internal sample
preparation, naming, panel-resolution, and modality-writing primitives. Avoid
maintaining a second reader implementation inside `add_cosmx_samples()`.
Parameter validation and generated element contents must be identical between:

```text
cosmx(samples={"sample_c": config}, output=new_output)
add_cosmx_samples(output=existing_output, samples={"sample_c": config})
```

The only intentional difference is publication: the first writes a new staged
store, while the second adds collision-free elements directly to an existing
sample-aware store.

### Verification

Focused tests should establish that:

- a new sample is added without changing the payload data, elements,
  transformations, or element metadata of existing samples;
- multiple successive calls create deterministic sample-prefixed names and
  independent coordinate systems;
- duplicate sample IDs and element or coordinate-system collisions fail during
  preflight, before source payload reads or destination writes;
- identical existing panels are reused and incompatible panels receive
  separate records;
- metadata for an element is never persisted before that element is written
  successfully;
- an ordinary element-write failure leaves no metadata for the failed element,
  while elements completed earlier retain both their data and metadata;
- a metadata-write failure triggers best-effort cleanup only of the newly
  written element and leaves preceding elements unchanged;
- every successful incremental element commit updates root `reader_version` in
  the same attributes write, including when a later element fails;
- a new feature-panel record and its first points reference are persisted
  together only after that points element succeeds;
- failure in a later requested sample does not remove samples committed earlier
  in the same call;
- a retry that collides with existing sample-scoped state raises without
  inferring interruption, overwriting, or automatically resuming;
- a non-CosMx, unsupported-metadata, unbacked, or earlier unprefixed
  destination is rejected without a migration attempt; and
- `cosmx()` retains staged whole-store creation/replacement, whereas
  `add_cosmx_samples()` performs no whole-store staging or publication swap.

## Slice 5: class-aware `hp.tb.aggregate_points`

**Status: implemented.**

This slice records the implemented class-aware baseline. Slice 6 deliberately
supersedes two lossy output choices from this baseline: per-feature
non-expression counts are retained instead of discarded, and the row universe
includes every instance receiving an assigned point instead of only instances
with an expression-class point. Until Slice 6 is implemented, the behavior
described in this Slice 5 section remains the current code behavior.

This slice consumes the generic feature-panel contract established by Slice 1
and supports the sample-scoped elements created or added by Slices 2 and 4.
Class-aware allocation requires that metadata; ordinary allocation remains
usable with points that do not reference a feature panel and preserves its
existing behavior when `expression_class` is omitted.

### Scope and outcome

Generalize `hp.tb.aggregate_points` so one call can allocate one or more explicitly
paired labels/points/coordinate-system regions and construct one complete
AnnData table. In class-aware mode, every point is spatially assigned once, but
the resulting payload is split by biological role: only the selected expression
class is retained as the sparse feature matrix, while every non-expression
class is reduced to per-instance auxiliary summaries.

For example, CosMx points carry both a feature key and a feature-class key:

```text
gene             code_class
ACTB             Endogenous
Negative01       Negative
SystemControl1   SystemControl
```

The resulting table has the following high-level contract:

```text
adata.X       instances x Endogenous features only
adata.var     the shared Endogenous feature axis
adata.obs     total assigned points per feature class and auxiliary_points_fraction
adata.obsm    mean coordinates calculated from assigned Endogenous points
adata.uns     the versioned feature_class_aggregation configuration and sources
```

Multiple regions are combined without an inner feature join:

```text
labels[0] + points[0] + coordinate_system[0] ┐
labels[1] + points[1] + coordinate_system[1] ┼──> one AnnData table
...                                             ┘
```

Resolve one shared expression-feature axis before constructing any per-region
matrix. A feature assayed by the compatible shared panel but absent from one
region is represented by zeros for that region rather than removed. Auxiliary
class feature counts always come from referenced feature-panel metadata. Class-aware
allocation has no caller-supplied denominator fallback or override.

The point-to-label lookup is performed once per labels/points pair, carrying
both the feature and feature-class columns through the assignment. It is never
repeated once per class. Auxiliary points outside every instance are deliberately
excluded from these cell-level summaries; Slice 11 operates directly on the
original points to provide spatial background QC.

When `expression_class` is omitted, the same multi-region API follows the
ordinary allocation path: all observed features remain in `adata.X`, no
class-derived `.obs` columns are added, and no
`adata.uns["feature_class_aggregation"]` record is written.

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

Extend `hp.tb.aggregate_points` with one optional class-aware argument:

```python
expression_class: str | None = None
```

This parameter name, type, and default are final for this slice. The `append`
parameter is removed; `overwrite` controls only whether the completed table may
replace an existing table element.

Keep semantic instance IDs and the AnnData observation-index name separate:

```python
instance_key: str = "cell_ID"
table_index_name: str | None = None
```

`instance_key` names the label-ID column throughout point assignment,
label-center construction and `adata.obs`. `table_index_name` names only
`adata.obs.index`; when it is `None`, resolve it to
`f"{instance_key}_index"`. Require an explicit table index name to be a
non-empty string and reject collisions with `instance_key`, `region_key`, the
auxiliary-points fraction column or generated per-class summary columns. The
temporary merged-count checkpoint retains a fixed internal
`_CHECKPOINT_INSTANCE_COLUMN = "instance_id"` so its canonical Parquet schema
does not depend on user-facing AnnData names.

Remove the deprecated `update_shapes_elements` parameter at the same boundary.
Allocation constructs the requested table and does not also mutate shapes
elements as an unrelated side effect; callers that need shape filtering should
request that operation separately.

```python
sdata = hp.tb.aggregate_points(
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
    feature_key="gene",
    expression_class="Endogenous",
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

`expression_class` selects the only feature class retained in `adata.X` and is
the switch for class-aware allocation. When `expression_class=None`, allocation
must branch to the ordinary implementation immediately, before feature-panel
lookup or any class-specific projection, validation, or aggregation. No
additional `.obs` columns are created and every `feature_key` value remains
in `adata.X`.

When `expression_class` is provided, every selected points element must have a
root Harpy points record that references an authoritative shared feature panel.
The panel supplies `feature_class_key`, the complete ordered classes,
feature-to-class assignments, and the feature lists whose lengths define the
auxiliary class feature counts. Missing, malformed, or incompatible panel metadata is an
error before spatial lookup. Allocation accepts no denominator fallback,
override, or per-sample mapping.

The public `allocate()` docstring must explain both modes and show the exact
metadata path used in class-aware mode. In particular, it must state that
`feature_class_key` is not a user argument: it is resolved through
`sdata.attrs["harpy"]["points"][points_name]["feature_panel"]` and then read
from the referenced record in `sdata.attrs["harpy"]["feature_panels"]`. The
docstring must also state that auxiliary class feature counts are derived from
the lengths of the panel's non-expression `features_by_class` lists and that class-aware
allocation fails when this metadata is unavailable.

The ordinary path still supports multiple allocation pairs. It uses the
deterministic union of observed targets as the shared feature axis and fills a
target absent from one pair with zero counts for that pair.

### Metadata resolution and source of truth

Resolve class-aware metadata from each selected points element rather than from
a sample identifier or element-name convention:

```text
points_name
    │
    ▼
sdata.attrs["harpy"]["points"][points_name]
    │
    └── feature_panel: "feature_panel_<content hash>"
                            │
                            ▼
sdata.attrs["harpy"]["feature_panels"][feature_panel]
    ├── feature_key
    ├── feature_class_key
    ├── classes
    └── features_by_class
```

For every normalized labels/points/coordinate-system pair, resolve the points
record and its referenced panel before spatial lookup. Require the panel's
`feature_key` to equal the public `feature_key` argument; read
`feature_class_key` directly from the panel rather than accepting it as an
aggregation parameter. Derive the feature count for every auxiliary class from
the length of its authoritative `features_by_class` list.

The metadata roles are deliberately distinct:

1. `sdata.attrs["harpy"]["feature_panels"]` is the authoritative assay
   definition.
2. `sdata.attrs["harpy"]["points"][points_name]["feature_panel"]` binds one
   points element to that definition.
3. `adata.uns["feature_class_aggregation"]` is a derived, table-local snapshot of
   the resolved allocation contract. It records how the table was constructed
   and remains interpretable with the AnnData object, but it is not a second
   authoritative panel definition.
4. The generated `.obs` columns are derived payload and never a metadata source.

Resolve and validate these inputs once into one immutable internal allocation
contract before assigning points. Use that same object to define the shared
feature axis, aggregation behavior, generated `.obs` columns, and table-local
`.uns` record. Do not independently reinterpret root metadata at separate
stages of the operation.

There is no per-sample or per-region auxiliary-class feature-count mapping in one output table.
Likewise, `code_class` categories may differ between samples stored in the same
SpatialData object, but samples with different class universes or incompatible
panels cannot participate in the same class-aware allocation call. They require
separate output tables. Otherwise, later QC would interpret a given class-count
column against different panel feature counts in different rows.

The complete policy is:

```text
all selected points reference compatible panels
    -> derive one shared class contract and auxiliary-class feature-count mapping

missing panel metadata or incompatible panels
    -> reject before spatial lookup
```

### Shared feature axis and panel compatibility

Do not construct independently schematized AnnData objects and combine them
with the default inner join. An inner join would silently discard a target from
the complete table whenever that target has no assigned transcripts in one
region.

For class-aware allocation, resolve one shared feature axis before constructing
the final AnnData:

- use the compatible panel's ordered `expression_class` features, including
  features with zero detections; and
- construct every per-pair sparse count matrix against this shared axis, so a
  missing target is represented by a zero rather than by dropping the feature.

In class-aware mode, feature-panel metadata must be available for every selected
points element. All referenced panels must agree on the feature key,
feature-class key, ordered classes, feature-to-class mapping, and features by
class. Panels selected from different samples need not share the same registry
key, but their canonical contents must be compatible. Reject missing or
incompatible metadata before spatial allocation. This prevents a zero from
ambiguously meaning either "assayed but not detected" or "not assayed by this
panel."

Derive one shared auxiliary-class feature-count mapping from the compatible
panel contract for the complete allocation call. Per-sample and per-region mappings
are not supported because later QC must interpret a given class-count column
against one consistent panel feature count across all rows.

### Categorical class contract

The points column identified by the panel's `feature_class_key` must have a
categorical dtype. If its Dask categories are unknown after a Parquet round
trip, allocation must apply the panel's authoritative ordered classes lazily
before validation. The class set is the complete feature-class universe for the
points element, including a class with zero detected or zero assigned points.
Require that:

- every category is a non-empty string;
- `expression_class` is one of the categories;
- the column contains no null values;
- every target maps to exactly one feature class; and
- every non-expression class has a non-empty authoritative
  `features_by_class` list, yielding a positive feature count.

Do not silently discard a panel class. An unused categorical class is valid and
produces zero per-instance counts, but its authoritative panel feature list
still defines the positive feature count retained for later QC calculations.

Normalize each category deterministically to snake case and construct output
names from that normalized category rather than accepting platform-specific
column names:

```text
Endogenous       -> n_endogenous_points
Negative         -> n_negative_points
SystemControl    -> n_system_control_points
Gene Expression  -> n_gene_expression_points
```

The generated count names follow one generic pattern:

```text
n_<normalized class>_points
```

This counts assigned rows from the points element. Do not include the source
`feature_class_key`, such as CosMx `code_class`, in the generated name: it
identifies the grouping column rather than the measured unit and may differ
between readers.

Reject an empty normalized name, two categories that normalize to the same
name, and collisions with existing `.obs` columns or the fixed
`auxiliary_points_fraction` output. Category order determines output-column order but
does not affect the calculations.

Point aggregation resolves panel metadata from every selected `points_name`,
verifies that the stored feature key matches the public `feature_key` argument,
uses the panel's
`feature_class_key` to select the class column, and derives auxiliary-class
feature counts from the `features_by_class` list lengths. It must not silently estimate them
from observed points or accept caller-supplied assay constants.

### Output metrics

In the example above, the output columns are:

```text
n_endogenous_points
n_negative_points
n_system_control_points
auxiliary_points_fraction
```

The class count columns contain assigned point counts. For this three-class
configuration:

```text
auxiliary_points_fraction =
    (n_negative_points + n_system_control_points)
    / (n_endogenous_points + n_negative_points + n_system_control_points)
```

Do not persist `negative_points_per_feature` or
`system_control_points_per_feature` in `.obs`. They are deterministic rescalings
of the raw class counts by the panel feature counts and add no independent table
information. Slice 11 QC plotting derives them on demand from the raw count
columns and the table-local auxiliary-class feature-count snapshot.

### Table-local metadata contract

Record the resolved configuration and generated-column bindings under one
dedicated table-local key. The `auxiliary_class_feature_counts` field below is a
derived snapshot of the panel values used for this table; it is not accepted as
an allocation argument or treated as an authoritative panel definition:

```python
adata.uns["feature_class_aggregation"] = {
    "schema_version": 1,
    "source_kind": "harpy_aggregate_points",
    "feature_key": "gene",
    "feature_class_key": "code_class",
    "expression_class": "Endogenous",
    "classes": ["Endogenous", "Negative", "SystemControl"],
    "auxiliary_class_feature_counts": {
        "Negative": 10,
        "SystemControl": 197,
    },
    "count_columns": {
        "Endogenous": "n_endogenous_points",
        "Negative": "n_negative_points",
        "SystemControl": "n_system_control_points",
    },
    "auxiliary_points_fraction_column": "auxiliary_points_fraction",
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
summaries per table, so `feature_class_aggregation` is a direct record rather
than a registry keyed by an arbitrary artifact name. Its generated-column
mappings bind the metadata to the actual `.obs` payload instead of requiring
downstream code to reconstruct names. The complete per-class feature lists
remain in SpatialData root metadata and are not duplicated into the table;
only the resolved auxiliary-class feature counts needed for later on-demand QC
calculations are retained.

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
   matrix containing expression and auxiliary features;
3. calculate per-instance class counts by summing the temporary matrix columns
   belonging to each class;
4. calculate `auxiliary_points_fraction`;
5. retain only the `expression_class` columns in the final `adata.X`; and
6. attach the total assigned points per feature class and `auxiliary_points_fraction` to
   the corresponding `.obs` rows.

After all pairs have been reduced, align their sparse matrices to the previously
resolved shared feature axis, stack them row-wise, concatenate `.obs` and
spatial coordinates in pair order, and construct one AnnData object. Add that
table to `SpatialData` exactly once. "One allocation call" therefore does not
require one monolithic Dask graph across every mosaic; pair-level work may stay
independent and out of core until the compact sparse results are assembled.

Conceptually:

```python
n_endogenous_points = X_all[:, endogenous_columns].sum(axis=1)
n_negative_points = X_all[:, negative_columns].sum(axis=1)
n_system_control_points = X_all[:, system_control_columns].sum(axis=1)

adata = AnnData(X=X_all[:, endogenous_columns], ...)
```

The temporary auxiliary columns add few sparse entries compared with the
endogenous matrix and are discarded before the table is written. No complete
transcript dataframe or dense instance-by-target matrix may be materialized in
memory.

Coordinates in `adata.obsm[spatial_key]` should continue to mean the average
position of expression transcripts and must therefore be calculated from the
selected `expression_class`, not from auxiliary points. Preserve the current
allocation row contract: an instance without an assigned endogenous transcript
does not receive an expression-table row. Reindex auxiliary summaries to the
expression row set and fill missing auxiliary counts with zero.

Unexpected or null feature classes, targets associated with multiple classes,
non-positive resolved panel feature counts, a missing expression class, and
collisions with existing output columns must produce clear errors. A panel
auxiliary class with no assigned points is valid and must produce a zero count
column. No auxiliary class produces a persisted per-feature rate. Validate the
complete multi-region request and shared
`feature_class_aggregation` configuration before writing the output table.

### Boundary with Slice 11

The `.obs` summaries describe only auxiliary points that land inside an instance
mask. For CosMx these auxiliary classes are controls, making the summaries
suitable for cell-level histograms and violin plots of the raw class counts and
`auxiliary_points_fraction`. Slice 11 may additionally derive the
following per-instance plotting metrics on demand:

```text
negative_points_per_feature =
    n_negative_points / auxiliary_class_feature_counts["Negative"]

system_control_points_per_feature =
    n_system_control_points / auxiliary_class_feature_counts["SystemControl"]
```

These derived values are useful for comparing control classes with different
panel sizes, for example in a hexbin plot. They are temporary QC values and
must not be written back to `.obs`.

They are not sufficient for a spatial background map. Allocation deliberately
removes points on label value zero, while unassigned controls outside masks are
still informative about sticky tissue, optical crowding, and regional assay
background. A later QC operation should therefore bin the original control
points directly in space and visualize separate normalized `Negative` and
`SystemControl` density maps. This spatial operation must not be folded into
`hp.tb.aggregate_points`.

### Verification

Focused tests should establish that:

- the final `.var` and `adata.X` contain only the selected expression class;
- `n_endogenous_points` equals the row sum of the final expression matrix;
- assigned auxiliary-class counts are correct and zero-filled for instances
  without an auxiliary point;
- no per-feature rate columns are persisted in `.obs`;
- the table-local auxiliary-class feature-count snapshot equals the authoritative panel feature
  counts and is sufficient for later QC derivation;
- `auxiliary_points_fraction` is correct and finite;
- non-categorical feature-class columns are rejected, while unknown Dask
  categories are restored lazily from the panel's ordered `classes` before
  validation;
- the class column is selected from each panel's `feature_class_key` without a
  caller-supplied column parameter;
- missing feature-panel references, missing panel-declared columns, and empty
  non-expression feature lists are rejected before spatial lookup;
- category-derived output names are deterministic and collisions are rejected;
- conflicting feature-to-class mappings and invalid class configuration fail
  before writing a table element;
- the versioned `feature_class_aggregation` record and its generated-column
  bindings survive a SpatialData Zarr round trip;
- scalar points and coordinate-system inputs broadcast across labels, while
  incompatible list lengths and duplicate labels are rejected;
- multiple allocation pairs create one table and one complete `regions`
  mapping in a single call;
- pairs from different samples retain their sample-prefixed region, points,
  and coordinate-system bindings, even when their local instance IDs overlap;
- the final class-aware expression matrix uses the panel-defined feature axis;
- expression targets missing from one allocation pair are zero-filled rather
  than removed by an inner join;
- missing and incompatible feature panels are rejected before spatial lookup;
- one panel-derived auxiliary-class feature-count mapping applies to every pair;
- an existing output table is replaced only when `overwrite=True`; and
- omitting `expression_class` reproduces the existing allocation result.

Benchmark the class-aware path on a representative backed crop before the full
run. Compare wall time, peak worker memory, task count, and output-table size
with ordinary allocation. The additional summaries should be small relative to
the point-to-label lookup and target-count reduction.

## Slice 6: lossless feature-class aggregates and label-derived centers

**Status: implemented.**

Revise the class-aware table payload established by Slice 5 so aggregation does
not discard the per-feature counts of non-expression classes. Preserve
`expression_class` as the selector for the primary expression matrix, retain
all other panel features in one sparse auxiliary matrix, and keep the existing
per-class `.obs` summaries as convenient QC columns. This remains a generic
feature-class contract and must not assume CosMx class names.

This slice also replaces point-derived table coordinates with centers of mass
from the segmentation labels and expands the class-aware row universe to every
instance receiving at least one assigned point. Users, not the aggregation
operation, decide whether to filter auxiliary-only or expression-empty instances
downstream.

### AnnData payload

For class-aware aggregation, the resulting table has this contract:

```text
adata.X
    sparse counts for features in expression_class

adata.var
    the complete panel-defined expression feature axis

adata.obsm["auxiliary_feature_counts"]
    sparse counts for every feature outside expression_class

adata.obsm[spatial_key]
    center of mass of each instance in the segmentation labels raster

adata.obs
    instance/region identity, total assigned points per feature class and
    auxiliary_points_fraction

adata.uns["feature_matrices"]["auxiliary_feature_counts"]
    the auxiliary matrix's independent feature-axis schema

adata.uns["feature_class_aggregation"]
    class-aware construction, summary-column and source bindings
```

For example:

```text
adata.X

instance    EPCAM    VIM
42              7      1
51              0      3
99              0      0

adata.obsm["auxiliary_feature_counts"]

instance    Negative1    Negative2    SystemControl1
42                  2            0                 1
51                  0            1                 0
99                  1            0                 2
```

Instance 99 is retained even though its expression row is all zero because it
received assigned non-expression points. Its `.obs` summaries and auxiliary
feature counts remain available for QC and downstream user-defined filtering.

Do not store the auxiliary counts in `.layers`. AnnData layers must have the
same `(n_obs, n_vars)` shape and variable axis as `X`, whereas the expression
and auxiliary matrices deliberately have different feature axes. Expanding
`.var` to the union of expression and auxiliary features would introduce
artificial all-zero control columns into `X` and expose them as ordinary
expression variables to downstream tooling. `.obsm` is observation-aligned but
permits an independent second axis, making it the appropriate location.

Use one fixed `auxiliary_feature_counts` matrix rather than one dynamically
named matrix per class. Its columns are ordered deterministically by:

1. the authoritative panel's class order with `expression_class` removed; and
2. the authoritative `features_by_class` order within each retained class.

Include panel-declared auxiliary features with zero assigned detections as
explicit all-zero columns. Store the matrix as CSR with `uint32` counts. The
ordered `feature_columns` in table-local metadata make the independent column
axis self-describing without placing those features in `adata.var`. Class
membership is described by the existing `feature_class_aggregation` contract,
not by a second class schema in the generic feature-matrix registry.

### Auxiliary feature-matrix metadata

Follow the Harpy convention already used by `hp.tb.add_feature_matrix`: the
numeric matrix lives in `.obsm`, while its column schema lives under the
table-local `feature_matrices` registry:

```python
adata.uns["feature_matrices"]["auxiliary_feature_counts"] = {
    "schema_version": 1,
    "source_kind": "harpy_aggregate_points",
    "feature_columns": [
        "Negative1",
        "Negative2",
        "SystemControl1",
    ],
}
```

`feature_columns` is the mandatory generic Harpy name for the ordered second
axis of a matrix registered under `adata.uns["feature_matrices"]`. Its length
must equal the auxiliary matrix's second dimension, and its values must be
non-empty and unique. This is deliberately distinct from `feature_key`, which
names the source points column containing feature identifiers.

Keep this registry entry matrix-specific and generic. Do not duplicate
`feature_key`, `feature_class_key`, `classes`, `features_by_class`, or a parallel
`feature_classes` list here. Those fields describe the source panel and the
class-aware aggregation rather than the physical `.obsm` matrix. The existing
`feature_class_aggregation` record already owns `feature_key`,
`feature_class_key`, `expression_class`, the ordered `classes`, and the
panel-derived `auxiliary_class_feature_counts`.

Add one binding to the existing aggregation record:

```python
adata.uns["feature_class_aggregation"]["auxiliary_feature_matrix_key"] = (
    "auxiliary_feature_counts"
)
```

This pointer prevents downstream consumers from guessing an `.obsm` key. The
feature-matrix record owns the auxiliary column schema; the aggregation record
owns the class semantics and must not duplicate the ordered feature names.
During construction, the ordered columns and derived class blocks must exactly
match the non-expression portion of the authoritative feature panel. After
construction, ordinary downstream filtering and reordering are allowed:
resolve each current `feature_columns` value against the authoritative panel
rather than inferring its class from a fixed contiguous slice. Validation must
check the pointer, matrix shape, column metadata and panel membership, but must
not constrain the current storage backend or dtype. `feature_columns` is a
derived record of the current matrix axis, not a second authoritative assay
panel.

This contract adopts the existing Harpy `feature_matrices` convention so that
matrix columns can always be discovered consistently. Integrating
`harpy_aggregate_points` matrices into napari-harpy's object classifier is out
of scope; napari-harpy does not need to recognize this `source_kind` for this
slice.

When `expression_class=None`, preserve ordinary aggregation behavior: every
observed feature remains in `adata.X`, no auxiliary feature matrix or
class-aware metadata is created, and feature-panel metadata is not required.
This slice introduces no new public parameter.

Class-aware mode requires at least one panel class other than
`expression_class`. If a panel contains only the selected expression class,
raise a clear error directing the caller to use `expression_class=None`. This
keeps the auxiliary matrix contract meaningful and its registered
`feature_columns` axis non-empty.

### Maximum-preservation row universe

In class-aware mode, construct rows from the union of instance IDs receiving at
least one assigned point from any panel class. Do not require an assigned
expression-class point. The row rules are:

- expression-only instances have an all-zero auxiliary row;
- non-expression-only instances have an all-zero `adata.X` row;
- instances with both classes retain both matrices;
- label value zero and points outside every instance remain excluded; and
- segmented instances receiving no assigned point from any class are not added
  by this operation.

The last rule keeps the operation defined as aggregation of points rather than
enumeration of an entire labels raster. A separate table-construction operation
may include completely empty segmented instances if that becomes a required
workflow. `hp.tb.aggregate_points` must not silently filter zero-expression
rows after constructing the class-aware table; downstream filtering belongs to
the caller.

For every output row, retain the Slice 5 summaries. They must be arithmetically
consistent with the persisted matrices:

```text
n_<expression class>_points == row_sum(adata.X)

n_<auxiliary class>_points ==
    row_sum(auxiliary_feature_counts[:, columns belonging to that class])

auxiliary_points_fraction ==
    sum(non-expression class counts) / sum(all class counts)
```

Every retained row has at least one assigned point, so the
`auxiliary_points_fraction` denominator is positive. Use the persisted `.obs` summaries
for convenient plotting, but treat disagreement with the matrices as a corrupt
table rather than choosing one payload as an implicit correction source.

### Segmentation-mask centers of mass

Coordinates in `adata.obsm[spatial_key]` must no longer be the mean location of
assigned expression points. Calculate the geometric center of mass of each
retained instance directly from the corresponding labels raster, using the
same `RasterAggregator.center_of_mass` approach and coordinate handling as
`hp.tb.aggregate_image`.

For an integer instance-label raster, every pixel belonging to one instance has
the same nonzero label value, so its center of mass is the geometric centroid
of that mask. This definition is independent of transcript abundance and is
therefore available and biologically stable for expression-only,
non-expression-only and mixed instances alike.

For each normalized labels/points/coordinate-system pair:

1. derive the retained instance IDs from all assigned point classes;
2. calculate labels centers of mass only for those IDs where the raster helper
   supports indexed calculation;
3. account for the labels element's pixel-aligned translation into the selected
   coordinate system in the same way as `aggregate_image`;
4. store coordinates in SpatialData order `(x, y)` or `(x, y, z)`; and
5. align them to the exact table row order by `(region_key, instance_key)`,
   never by incidental dataframe or raster traversal order.

Apply label-derived centers to both ordinary and class-aware
`hp.tb.aggregate_points` output so the meaning of `adata.obsm[spatial_key]`
does not change with `expression_class`. Reuse the labels center-of-mass helper
rather than maintaining a second numerical implementation. Slices 7a and 7b
must preserve this standalone center contract rather than coupling the assignment
handoff to an optional fused label-moment implementation. Any later fusion is a
separate measured optimization and must preserve the canonical-center contract
introduced by Slice 8.

### Single-assignment implementation

Do not perform another point-to-label lookup for the auxiliary matrix. The
existing class-aware reduction already groups assigned points by instance and
feature before filtering to the expression axis. Reuse that one grouped result
to construct:

1. the expression CSR matrix against the panel-defined expression axis;
2. the auxiliary CSR matrix against the panel-defined non-expression axis; and
3. the `.obs` class summaries by summing the corresponding matrix columns.

Resolve both feature axes before per-region matrices are constructed. All
selected points elements must continue to reference compatible panels, so the
same expression and auxiliary axes apply to every region. Stack region matrices
row-wise in normalized pair order, with all-zero sparse rows where a retained
instance has no feature from one axis.

The current implementation computes all per-feature counts before discarding
the non-expression columns. This slice therefore changes output assembly and
row selection, not spatial assignment. The source points remain unchanged and
continue to support Slice 11 QC for unassigned and outside-mask controls.

### Verification

Focused tests should establish that:

- `adata.X` and `.var` contain the complete panel-defined expression axis and
  no auxiliary features;
- `auxiliary_feature_counts` contains every panel-defined non-expression
  feature in deterministic class/feature order, including zero-detection
  columns;
- the auxiliary matrix is `uint32` CSR and its row count equals `adata.n_obs`;
- its `feature_columns` metadata exactly describes every matrix column and
  survives AnnData and SpatialData Zarr round trips;
- the ordered non-expression classes and their panel-feature-count-derived contiguous
  column blocks cover `feature_columns` exactly and match the authoritative
  feature panel;
- expression-only, non-expression-only and mixed instances are all retained,
  while instances receiving no assigned point and label value zero are absent;
- non-expression-only instances have all-zero expression rows and are never
  implicitly removed;
- every `.obs` class count equals the appropriate persisted matrix row sum and
  `auxiliary_points_fraction` remains correct and finite;
- one point-to-label assignment feeds both sparse matrices and class summaries;
- label-derived centers equal a simple in-memory center-of-mass reference for
  irregular masks and do not depend on point positions or classes;
- centers are correctly translated, axis-ordered and aligned by region and
  instance ID for multiple aggregation pairs;
- ordinary and class-aware modes use the same labels-center coordinate
  definition;
- `expression_class=None` creates neither the auxiliary matrix nor its metadata
  and otherwise retains ordinary feature aggregation; and
- malformed auxiliary metadata, a dangling aggregation pointer, a
  feature-column shape mismatch and features assigned to the wrong panel class
  are rejected clearly, while downstream matrix representation, dtype and
  recalculated summary values remain valid.

Benchmark the additional sparse matrix and labels-center calculation on a
representative backed crop. Record output nonzeros and bytes separately for
`X` and `auxiliary_feature_counts`, and confirm that the label-derived center
calculation does not materialize the complete raster unnecessarily. Slices 7a
and 7b own the assignment-graph and out-of-core table-construction
optimizations, respectively.

## Slice 7a: chunk-aware point-to-label assignment

**Status: implemented.**

Refactor the private spatial-assignment path used by `hp.tb.aggregate_points`
without changing the public or biological contracts established by Slices 5
and 6. This optimization must remain generic to raster labels and points
elements; it must not depend on CosMx FOV identifiers or reader-specific
partition metadata.

This slice changes only point-to-label assignment and the private boundary
between assignment and downstream reduction. It deliberately retains the
current reduction and in-memory AnnData construction path. Consequently, Slice
7a removes the labels-chunk by points-partition graph fan-out, but does not yet
remove the driver-memory cost of materializing the complete reduced
instance-feature counts. Slice 7b owns that separate problem.

### Current scaling limitation

The current `_assign_points_to_labels` helper enumerates every labels-array chunk
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

Replace the overloaded private helper with an explicit assignment boundary and
a separate downstream reduction stage:

```python
assigned_points = _assign_points_to_labels(
    labels=...,
    points=...,
    value_keys=...,
    to_coordinate_system=...,
)

aggregates = _aggregate_assigned_points(assigned_points, ...)

centers = _label_centers_of_mass(
    labels=...,
    instance_ids=aggregates.instance_ids,
    to_coordinate_system=...,
)
```

`_assign_points_to_labels` assigns the raster value underneath each point and
filters label value zero. It accepts all value columns needed by the caller,
rather than a single `value_key`, so ordinary and class-aware aggregation use
the same spatial lookup. Its output is a lazy assigned-points dataframe with
one row per retained point, the assigned instance ID and the requested value
columns. This dataframe is the stable private handoff from Slice 7a to Slice
7b.

During Slice 7a, the existing count and row-selection reductions continue to
consume this handoff and the current in-memory AnnData assembly remains in use.
Slice 7b replaces those downstream internals with partitioned reductions and
incremental component writes. Instance coordinates continue to come from the
labels center-of-mass stage required by Slice 6, not from assigned-point
coordinate means. The exact private names may change during implementation,
but this separation of responsibilities is required.

The existing `_assign_points_to_labels` helper is also used by `bin_counts()`.
Keep one shared assignment implementation and preserve both callers' behavior:
`aggregate_points()` may drop source coordinates after assignment, whereas
`bin_counts()` retains them for its existing barcode-coordinate reductions.
Slice 7a must not silently optimize one caller while changing the other.

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

For a two-dimensional labels raster, number blocks in row-major chunk order.
For example:

```text
                 x chunk
                0       1
            ┌───────┬───────┐
y chunk 0   │ id 0  │ id 1  │
            ├───────┼───────┤
y chunk 1   │ id 2  │ id 3  │
            └───────┴───────┘
```

After applying the labels translation and the agreed coordinate-to-pixel
rounding, use the cumulative chunk boundaries to derive each point's chunk
indices. Flatten them deterministically:

```text
# 2D
block_id = y_chunk * number_of_x_chunks + x_chunk

# 3D
block_id = (z_chunk * number_of_y_chunks + y_chunk)
           * number_of_x_chunks + x_chunk
```

The same row-major ordering must index `arr.to_delayed()` so every `block_id`
selects exactly one corresponding labels chunk. A point on a chunk boundary
belongs to the chunk whose half-open interval starts at that boundary. A point
inside the overall labels extent therefore receives exactly one valid
`block_id`; a point outside it is removed before redistribution.

Preserve the current coordinate-to-pixel rounding semantics exactly:
coordinate values are rounded with pandas/NumPy `round()` semantics before
conversion to an integer pixel index. A labels transformation may contribute
only a pixel-aligned translation. Each translation component must be within an
absolute tolerance of `1e-6` pixels of its nearest integer; normalize an
accepted component to that integer and reject fractional translations rather
than silently truncating them.

Dimensionality must be unambiguous. Two-dimensional labels require point
coordinates `x` and `y`; three-dimensional labels require `x`, `y`, and `z`.
Reject a missing required coordinate or an unexpected spatial coordinate that
would otherwise be silently ignored. The block calculation and vectorized
lookup must use the same ordered raster axes as the labels array.

Implement the redistribution with the temporary integer `block_id` as the
Dask index and explicit divisions `0, 1, ..., number_of_blocks`. Explicit
divisions prevent Dask from sampling or eagerly calculating quantiles merely
to plan the shuffle and produce exactly one partition for every labels block,
including an empty one. Do not hard-code a shuffle backend: allow Dask to use
its configured implementation, such as disk for a local scheduler or
peer-to-peer for a distributed client.

The NumPy-style docstring of the implemented `_assign_points_to_labels` helper
must explain this classification and lookup algorithm rather than merely state
that points are assigned to labels. Include the two-dimensional chunk-grid
scheme above, the half-open boundary rule, the labels-translation convention,
the optional 3D extension, the returned lazy dataframe schema, background
filtering and the effect of retaining or dropping coordinate columns. It must
also state that graph construction performs no source reads and that the
temporary `block_id` is not part of the returned dataframe.

The returned dataframe's index, row order and partition order are deliberately
not part of the private contract because redistribution by `block_id` changes
them. Callers may rely only on exactly-once retained rows, the documented
columns and their dtypes. Any input column that callers need as an identifier
must therefore be carried explicitly as a value column rather than inferred
from the dataframe index.

This replaces repeated full-dataframe predicates with one linear block
classification and one redistribution of the points. Supply explicit Dask
`meta` throughout. Do not materialize the complete points dataframe, the
complete labels raster, or a dense instance-by-target matrix. Preserve the
existing coordinate-system contract in this slice: points are
identity-transformed in the selected coordinate system and labels may differ by
a pixel-aligned translation. Validate any coordinate-to-pixel rounding and
translation assumptions explicitly rather than relying on integer truncation.

The existing `chunks` option may still request a virtual labels rechunk, but
the natural stored chunks remain the default. Benchmark virtual rechunking
carefully because smaller blocks reduce the size of each labels task while
increasing the number of point buckets, shuffle partitions, and scheduler
tasks.

### 7a verification and performance contract

Focused correctness tests must establish that:

- optimized assignment matches a simple in-memory reference for 2D labels;
- half-open chunk edges assign every in-bounds point exactly once;
- background and out-of-bounds points are excluded;
- irregular final chunks and empty spatial buckets are handled correctly;
- current half-to-even coordinate rounding is preserved and fractional labels
  translations outside the accepted numerical tolerance are rejected;
- 2D and 3D inputs accept exactly their required spatial coordinates and reject
  dimensional mismatches;
- supported translated coordinate systems produce the expected raster lookup;
- multiple retained value columns survive assignment with their dtypes and
  categorical metadata intact;
- assignment does not promise to preserve the input dataframe index or row and
  partition ordering;
- explicit integer divisions avoid an eager quantile calculation, while the
  configured Dask shuffle backend remains selectable;
- graph construction performs no point or labels source reads;
- `bin_counts()` preserves its barcode assignment, retained-coordinate and
  exactly-once behavior through the shared helper; and
- the unchanged downstream reduction and in-memory assembly produce the exact
  Slice 6 result for ordinary and class-aware aggregation.

Do not make unit tests depend on Dask layer names or an exact task count. Use a
separate benchmark and source-read instrumentation to compare the old and new
assignment implementations across increasing point-partition and labels-chunk
counts. Record wall time, graph-construction time, task count, bytes read,
shuffle bytes, spill volume and worker memory. Include both a small case, where
shuffle overhead can dominate, and a representative full backed mosaic. Vary
the labels chunking independently and document the point at which smaller
virtual chunks become counterproductive.

Slice 7a is complete when assignment remains lazy during graph construction,
avoids the `C * P` predicate fan-out, preserves every Slice 6 table value through
the unchanged downstream path and materially improves the representative
large-mosaic assignment without a major regression on the small case. It does
not claim bounded driver memory for the reduced counts or final table.

The implemented router preserves the natural labels chunks by default,
classifies every point once, redistributes by explicit integer block divisions,
and joins each resulting points partition to one row-major delayed labels
chunk. On the representative backed mosaic used for the diagnostic baseline,
the assignment graph contains 517 tasks for 78 labels chunks instead of
approximately 2,000, and 6,949 tasks for 1,150 chunks instead of approximately
28,000. A focused local threaded benchmark with a 4,096 by 4,096 raster, 64
labels chunks, 16 points partitions and 250,000 points produced identical
238,158 retained rows while reducing graph construction from 0.292 to 0.031
seconds, the graph from 1,856 to 434 tasks and assignment computation from
1.147 to 0.402 seconds. These timings are diagnostic rather than stable test
thresholds; correctness tests do not depend on task-layer names or exact task
counts.

## Slice 7b: out-of-core reduction and AnnData table construction

**Status: implemented.**

The implementation replaces the former driver-materializing pair reductions
with `_aggregation_checkpoint.py` and `_aggregation_writer.py`. Phase A reduces
assigned points locally, shuffles compact rows by aggregation pair and instance,
merges duplicate feature counts, and writes the validated Parquet checkpoint in
one shared computation. Phase B converts each non-empty checkpoint partition to
full-width delayed CSR blocks and writes `X` and the optional auxiliary matrix
through AnnData's component writer. Observation metadata and one center per
retained labels instance remain bounded driver-resident payloads.

Publication uses a hidden workspace below `tables/`, adopts the completed
AnnData group with SpatialData's table attributes, and only then renames it to
the requested table name. The returned same-process table uses backed
`sparse_dataset` handles. Focused tests cover ordinary and class-aware values,
partition-major row alignment, a single Phase A source execution, merged-count
manifests, empty-pair and overflow rejection, Zarr v2/v3 reopening, failure
cleanup, preservation of an existing table, and bypassing
`SpatialData.write_element()`.

Consume the lazy assigned-points dataframe established by Slice 7a and replace
the driver-resident instance-feature reduction and sparse-matrix assembly with
a two-phase, out-of-core count path. Preserve the public and biological
contracts established by Slices 5 and 6. This slice does not reimplement the
spatial assignment optimized by Slice 7a and does not attempt to make general
SpatialData table reopening lazy.

`hp.tb.aggregate_points` requires a `SpatialData` object backed by a writable
Zarr store. Validate this precondition before normalizing aggregation pairs or
constructing any Dask graph. An unbacked object must raise a clear error that
also gives the user the remedy, for example:

```text
hp.tb.aggregate_points requires a SpatialData object backed by a writable Zarr
store. Write it first with:

    sdata.write("sdata.zarr")
```

`SpatialData.write()` updates `sdata.path` by default. Reopening the written
store with `sdata = spatialdata.read_zarr("sdata.zarr")` remains recommended
when the original object contains lazy elements backed by other locations,
because the reopened object is self-contained. The error does not need to
require this additional step. Reader-created backed objects can be passed
directly.

Do not preserve the previous unbacked reduction and AnnData assembly path. This
slice establishes one production implementation and replaces the private
driver-materializing `_reduce_aggregation_pair()` and
`_assemble_aggregation_table()` flow rather than maintaining two execution
paths.

The large payload is the set of observed `(instance, feature)` counts. Keep that
payload partitioned and on disk. The complete `.obs`, `.var`, `.uns` and
label-derived center matrix have only one row per retained instance or one row
per feature; they may be assembled on the driver and written once after their
alignment has been validated.

### Phase A: assign once and stage merged count blocks

For each normalized aggregation pair, use the Slice 7a handoff once in that
pair's coordinate system and reduce every assigned-points partition locally by
`(instance, feature)`. Duplicate points belonging to the same pair within a
partition become one compact count row. In class-aware mode, validate each
unique source points element against the shared feature panel in a separate
computation before spatial assignment. This intentional early validation lets
an invalid source fail before the assignment graph or temporary workspace is
constructed. Derive both the expression and auxiliary payloads from the same
subsequent feature counts.

Across Slices 7a and 7b, the intended dataflow contains two global
redistributions with different keys:

1. Slice 7a shuffles original points by spatial labels-block ID so each point
   can be looked up in the appropriate labels-array chunk.
2. Phase A of Slice 7b shuffles the locally reduced count rows by aggregation
   pair and instance ID so every feature count for one instance is colocated and
   its partial counts can be merged.

The partition-local groupby between these redistributions requires no worker
communication. Phase B performs no further global shuffle.

Keep these partition-local reductions inside the Dask graph; do not write one
Harpy-managed artifact before the shuffle. The local reductions are an
intermediate optimization, not a persistent staging format. Writing them would
create an additional collection of temporary files that Phase A would
immediately need to read and redistribute.

Perform one Dask hash shuffle on `(aggregation_pair, instance_id)`. Leave
`npartitions=None`, so Dask preserves the compact-count dataframe's existing
partition count and chooses its configured shuffle implementation. This key is
intentional: shuffling by the full `(aggregation_pair, instance_id, feature)`
key could send different features belonging to one instance to different
partitions, which would prevent Phase B from constructing that instance's
complete sparse row locally.

Dask's default is a partition **count**, not an adaptive byte-size target, and
it does not guarantee equal row or byte sizes. Accept that behavior in this
slice. Normal Dask spilling can control broader intermediate pressure, although
it cannot subdivide one pathologically large pandas partition. Do not add an
instance-ID span, perform an eager
`repartition(partition_size=...)`, or compact checkpoint fragments. Benchmark
the resulting distribution first; explicit sizing is a future optimization
only if representative data demonstrate problematic skew or file counts.

After the shuffle, merge duplicate `(aggregation_pair, instance_id, feature)`
rows within each resulting partition. All partials for one instance are now
colocated. Only then establish the durable computation boundary by writing the
merged long-form counts to one logical, Harpy-owned temporary Parquet dataset.
The staged data are not AnnData objects and do not contain the original points.

For example, suppose Slice 7a has already routed the original points by spatial
labels block and attached the looked-up instance IDs:

```text
assigned partition 0                  assigned partition 1
pair instance feature                 pair instance feature
0    42       EPCAM                   0    42       EPCAM
0    42       EPCAM                   0    42       EPCAM
0    42       VIM                     0    51       VIM
0    51       VIM
             │                                     │
             ▼                                     ▼
partition-local partial counts         partition-local partial counts
pair instance feature count            pair instance feature count
0    42       EPCAM       2            0    42       EPCAM       2
0    42       VIM         1            0    51       VIM         1
0    51       VIM         1
             │                                     │
             └──────────────────┬──────────────────┘
                                ▼
              shuffle compact rows by (pair, instance)
                                │
                                ▼
                 merge by (pair, instance, feature)
                                │
                                ▼
checkpoint rows
pair instance feature count
0    42       EPCAM       4
0    42       VIM         1
0    51       VIM         2
```

If aggregation pair 1 also contains numeric instance ID 42, its shuffle key is
`(1, 42)` rather than `(0, 42)`, so it cannot be merged with the rows above. At
the checkpoint boundary, `(pair, instance, feature)` is globally unique and all
rows for one `(pair, instance)` occur in exactly one physical partition.

The checkpoint is one logical dataset written with Dask's ordinary Parquet
writer:

```text
<Harpy-owned temporary directory>/merged_counts/
└── <Dask-managed Parquet part files>
```

The physical layout normally follows the output Dask partitions. Empty output
partitions may be omitted or represented by an empty part according to the
pinned Dask/Parquet implementation; Phase B ignores either representation.

The merged checkpoint has this internal schema:

| Column             | Dtype                 | Contract                                                                |
| ------------------ | --------------------- | ----------------------------------------------------------------------- |
| `aggregation_pair` | `int64`               | Zero-based ordinal of the normalized points-to-labels aggregation pair. |
| `instance_id`      | `uint64`              | Positive label value; background zero is absent.                        |
| `feature`          | non-null UTF-8 string | Normalized value from the requested points `feature_key`.               |
| `count`            | `uint64`              | Merged assigned-point count for this instance and feature.              |

At the checkpoint boundary, `(aggregation_pair, instance_id, feature)` must be
globally unique. Keep local and merged counts as `uint64`; do not cast the
partition-local groupby directly to the final matrix dtype. Before constructing
a CSR block, reject any merged count greater than `uint32` maximum and only then
cast its values to the persisted `uint32` matrix dtype. Reject null features,
non-integral or non-positive retained instance IDs and malformed pair keys
before publishing any table.

Each non-empty checkpoint partition returns only a small manifest containing
its path, output-partition ordinal and sorted composite
`(aggregation_pair, instance_id)` output-row keys. The partition contents are
validated while this manifest is constructed; schema and row-count fields are
not retained in the manifest. Empty output partitions do not contribute an
output row block. Do not return every fragment's complete observed-feature set
to the driver. In ordinary mode, calculate fragment-local feature sets and
combine them through a Dask tree reduction, returning one global sorted feature
axis. Class-aware mode uses the panel-defined axes and does not calculate an
observed-feature union.

Submit local reduction, compact-count shuffling, duplicate merging,
checkpoint writing, manifest construction and the ordinary feature-set tree
reduction as one Dask computation so the shared point-to-label assignment graph
executes exactly once. Dask may use its own scheduler-managed shuffle or spill
files during this computation; those transient files are not part of Harpy's
staging contract. In particular, do not run an eager byte-sizing computation or
call `.compute()` once to discover axes and then call it again to construct the
checkpoint.

In class-aware mode, the expression and auxiliary feature axes are already
defined by the panel and include features with zero detections. In ordinary
mode, use the single sorted result of the distributed feature-set tree
reduction. Never collect either every fragment's repeated feature set or the
complete instance-feature count series merely to discover that axis.

### Phase B: construct AnnData blocks from the merged checkpoint

Read only the staged merged-count dataset; Phase B must have no dependency on
the original points or the Slice 7a assignment graph. Treat each non-empty
checkpoint partition as one output row block and finalize it on a worker
against the shared feature axes. Build its CSR expression
matrix and, in class-aware mode, its CSR auxiliary matrix and class summary
columns. Expose these conversions as delayed, independently reusable block
readers backed by the merged checkpoint rather than writing a second collection
of temporary CSR artifacts. Reading a compact merged-count fragment again for
separate `X` and auxiliary component writes is acceptable; rerunning assignment
or the compact-count shuffle is not.

Within each checkpoint partition, sort the unique composite
`(aggregation_pair, instance_id)` identities and retain them as that
partition's `output_row_keys`. Concatenate these keys in checkpoint-partition
order to establish the final row order. Their cumulative lengths determine the
corresponding CSR row blocks; no explicit `row_start` or half-open row interval
is stored. This partition-major row order is the manifest's source of truth for
all output components; it is not a promise of globally ascending instance IDs
and need not remain identical if a future Dask partition plan changes.
Consumers identify observations through their composite identity, not their
physical row position.

The complete row universe is the union of instances receiving any assigned
feature class. Observation identity and cross-component alignment use the
composite `(labels_name, instance_id)` key because different labels elements
may contain the same numeric instance ID. The final row order from the
checkpoint manifests must be shared by `X`, `.obs`, the optional auxiliary
feature matrix and the center matrix.

Every normalized aggregation pair must contribute at least one retained,
non-background instance. If a requested labels/points pair produces none,
raise a clear `ValueError` naming both elements before any table is published.
Do not silently omit the empty pair: doing so would make the requested regions,
the aggregation metadata and SpatialData's table annotation disagree. This
also defines the all-empty call without a separate special case.

### Label-derived centers

Centers are deliberately not part of the partitioned count path. Once Phase A
has established the retained instance IDs, loop over the normalized labels
elements, calculate the requested centers of mass once per labels element with
the existing standalone labels-center implementation and concatenate the
resulting dataframes in pair order. Align the concatenated centers to the final
row manifest through `(labels_name, instance_id)` and reject missing, duplicate
or non-finite centers.

The complete dense center matrix may reside on the driver: its size is
proportional to retained instances times spatial dimensions, rather than to the
much larger number of observed instance-feature pairs. Write
`.obsm[spatial_key]` once with `anndata.io.write_elem`; do not recalculate
centers per count block or rescan a labels raster for every output block.

### Component-wise AnnData writing

The optimized assignment graph alone is insufficient for large datasets. The
current implementation computes every observed `(instance, feature)` count into
one driver-resident pandas `Series`, converts the complete result to sparse
matrices, and only then writes the completed `AnnData`. The scalable backed path
must instead consume the staged merged-count blocks and write the final
components separately with the public
[`anndata.io.write_elem`](https://anndata.readthedocs.io/en/stable/generated/anndata.io.write_elem.html)
API:

```text
assigned points
      │
      ▼
local reductions (Dask graph only; no per-partition artifacts)
      │
      ▼
compact-row shuffle + duplicate merge
      │
      ▼
Phase A checkpoint: merged long-form count blocks
      ├──► axes + row manifest ──► obs + centers ─────────┐
      │                                                   │
      └──► Phase B: delayed CSR conversion                │
                    ├──► sparse X block write ────────────┤
                    └──► sparse auxiliary block write ───┤
                                                          ▼
                                               final SpatialData table
```

The component writer must follow this sequence:

1. Validate each merged-count partition against the expected long-form schema
   while constructing its manifest. Require every composite instance identity
   to be owned by exactly one checkpoint partition and use each manifest's
   sorted `output_row_keys` as the same within-block ordering for every output
   component. CSR conversion receives the authoritative feature axis directly;
   it does not persist or compare a separate feature-axis hash.
2. Initialize a staging AnnData group through `anndata.io.write_elem`, using a
   small AnnData skeleton containing the shared `.var` axis and required empty
   mappings. Do not hand-author AnnData root or component encoding attributes
   when the AnnData writer can create them.
3. Expose the ordered expression CSR blocks as one sparse Dask array and write
   it to `X` with `anndata.io.write_elem`. In the pinned AnnData version, the
   registered Dask-sparse writer writes the first CSR chunk and appends later
   chunks through AnnData's backed
   [`sparse_dataset`](https://anndata.readthedocs.io/en/stable/generated/anndata.io.sparse_dataset.html)
   implementation. Version-gated tests must confirm CSR format, 64-bit
   `indices`/`indptr` safety, output shape and values. The independently reusable
   staged merged-count sources ensure these per-chunk computations cannot rerun
   point assignment or the compact-count shuffle.

   Every delayed CSR chunk must have shape
   `(block_n_obs, len(shared_feature_axis))`: chunks partition rows only and
   each chunk spans the complete feature axis. The Dask array therefore has one
   column chunk and ordered row chunks matching the row manifests. Apply the
   same full-width rule to the auxiliary matrix against
   `shared_auxiliary_feature_axis`; this keeps AnnData's sequential CSR append
   path well-defined and prevents independently constructed column blocks from
   drifting out of alignment.

4. Concatenate the bounded checkpoint-partition `.obs` frames in manifest order
   and write the resulting one-row-per-instance pandas dataframe once. It must
   not contain the much larger per-instance-feature reductions.
5. In class-aware mode, expose the ordered auxiliary CSR blocks as another
   sparse Dask array and write `.obsm["auxiliary_feature_counts"]` through
   `anndata.io.write_elem`. Ordinary mode omits this component.
6. Align the complete concatenated labels-center dataframe to the row manifest
   and write `.obsm[spatial_key]` once.
7. Write the final `.uns` mapping once. It includes `spatialdata_attrs` and, in
   class-aware mode, the aggregation and auxiliary feature-matrix contracts
   established by Slices 5 and 6. The shared `.var` axis was written by the
   skeleton and must not be independently reconstructed or reordered after `X`
   has been written.
8. Validate the completed component shapes before publication:

   ```text
   X.shape[0] == len(obs) == obsm[spatial_key].shape[0]
   X.shape[1] == len(var) == len(shared_feature_axis)

   # class-aware mode
   obsm["auxiliary_feature_counts"].shape ==
       (len(obs), len(shared_auxiliary_feature_axis))
   ```

   Also verify the exact observation-identity sequence against the row
   manifest; compatible shapes alone do not prove alignment.

9. Publish the completed group as a SpatialData table only after all component
   writes and validation have succeeded. Use the same lower-level
   AnnData-writing approach that `hp.tb.add_feature_matrix` uses for backed
   table updates: write AnnData components with `anndata.io.write_elem`, add the
   required SpatialData table-group metadata through one isolated helper, and
   consolidate Zarr metadata only after the complete table is valid. Do not pass
   the completed table through `SpatialData.write_element()`, because that would
   re-enter the ordinary in-memory table-writing path. Remove only Harpy-owned
   staging artifacts on failure and do not leave a partial table visible under
   `output_table_name`.

Use this out-of-core writing path for every supported call. An unbacked object
has no destination in which to construct the table and is rejected; there is no
fallback in-memory implementation. This slice does not add an output-path
parameter or a public checkpoint-partition sizing parameter to
`hp.tb.aggregate_points`.

### Temporary store and publication contract

The implementation targets the local filesystem-backed Zarr stores accepted by
the current `SpatialData.path` API. Before constructing the Slice 7a graph,
validate `output_table_name` and its overwrite policy against both the in-memory
object and `<sdata.path>/tables`. Create one unique, hidden, Harpy-owned working
directory beneath the tables directory:

```text
<sdata.path>/tables/
├── <existing visible tables>
└── .harpy-aggregate-<uuid>/
    ├── merged_counts/     # Phase A checkpoint
    └── table/             # Phase B AnnData group
```

The hidden directory is not a SpatialData table and must never be registered in
`sdata.tables`. Record every path created by this call and restrict failure
cleanup to those paths. The checkpoint and staged AnnData group share one
working directory so their ownership and cleanup boundary is unambiguous.

After AnnData has written every component and Harpy has validated the completed
staged group, an isolated publication helper must add exactly the SpatialData
table-group attributes used by `SpatialData.write_table`:

```yaml
spatialdata-encoding-type: ngff:regions_table
version: "0.2"
region: [...]
region_key: region
instance_key: instance_id
```

The actual `region`, `region_key` and `instance_key` values come from the
validated table annotation rather than from these illustrative values. Those
group attributes are the adoption boundary; Harpy must not duplicate AnnData's
component encodings.

Publication uses local directory renames, not `zarr.Group.move()`, which is not
implemented by the pinned Zarr version:

- for a new output, rename the staged `table/` directory to
  `tables/<output_table_name>`;
- for `overwrite=True`, rename the existing table to a unique hidden backup,
  rename the completed staged table to the final name, restore the backup if
  the second rename fails, and remove the backup only after successful
  publication; and
- for `overwrite=False`, reject the collision before assignment or reduction
  begins.

Consolidate Zarr metadata only after the final directory is in place. Then open
the final published group and create the `sparse_dataset` handles used by the
returned same-process AnnData shell. Never retain or attach a sparse handle
opened against `table/` before its rename, because its stored path becomes
stale. Once the final table has been attached successfully, remove the
checkpoint and the now-empty working directory. On any earlier failure, leave
the previous visible table intact or restore it and remove only paths owned by
the current call.

### AnnData and SpatialData integration boundary

AnnData remains responsible for every AnnData-encoded component. In particular,
do not write sparse `X` directly with raw Zarr operations. Its documented
[on-disk encoding](https://anndata.readthedocs.io/en/stable/fileformat-prose.html)
is a group containing `data`, `indices` and `indptr` arrays plus shape and
encoding metadata, rather than a regular two-dimensional Zarr array. Correct
row appends must update the CSR offsets, shape, index widths and all three
arrays consistently. AnnData's Dask-sparse `write_elem` path already owns this
logic. Raw resizing would duplicate format-sensitive code in Harpy.

The deliberately driver-resident components are `.obs`, `.var`, `.uns` and the
dense labels-center matrix. Their memory scales with retained instances,
features or metadata, not with observed instance-feature pairs. Measure `.obs`
and center memory separately. Do not add a private AnnData dataframe appender or
partition the center calculation unless representative benchmarks demonstrate
a separate problem.

The pinned SpatialData version has no public operation that adopts a prewritten
AnnData group as a table. Implement one narrow Harpy publication adapter that
sets only the required SpatialData table-group metadata after AnnData has
written and validated the components. This helper is an integration boundary,
not an alternative table serializer.

After publication, `aggregate_points` must make the new table available on the
returned `SpatialData` without materializing the newly written sparse matrices.
Do not use Harpy's existing post-write `spatialdata.read_zarr()` reload for this
path because it would read the completed table back into memory and negate the
out-of-core construction benefit.

Instead, open the final published table group and construct one same-process
backed AnnData shell. Reuse the `.obs`, `.var`, `.uns` and dense center arrays
that were already assembled for the component writes. Bind `adata.X` and, in
class-aware mode, `adata.obsm["auxiliary_feature_counts"]` directly to their
published on-disk CSR groups with the public
[`anndata.io.sparse_dataset`](https://anndata.readthedocs.io/en/stable/generated/anndata.io.sparse_dataset.html)
API:

```python
adata = AnnData(
    X=sparse_dataset(table_group["X"]),
    obs=obs,
    var=var,
    uns=uns,
    obsm={
        "auxiliary_feature_counts": sparse_dataset(
            table_group["obsm"]["auxiliary_feature_counts"]
        ),
        spatial_key: centers,
    },
)
```

Ordinary mode omits the auxiliary entry. Validate this shell with
`TableModel.validate()` and attach it to
`sdata.tables[output_table_name]` only after the final table has been published.
Open the final group after publication so no attached sparse handle refers to a
temporary staging path. This is a narrow same-process binding required by the
existing `aggregate_points` return contract; it neither calls an experimental
AnnData lazy reader nor establishes a general SpatialData lazy-reading
contract.

Making a later `spatialdata.read_zarr()` call reopen tables lazily is explicitly
deferred to a separate follow-up slice and must not require changes to
SpatialData internals in Slice 7b.

### Implementation structure

Do not add the complete out-of-core implementation to the already large
`_allocation.py` module. Keep responsibilities separated as follows:

- `_allocation.py` retains the public API, normalized aggregation-pair and
  feature-panel contracts, Slice 7a handoff, and high-level orchestration;
- a private `_aggregation_checkpoint.py` module owns Phase A schemas,
  pair-and-instance shuffling, compact-count merging, Parquet manifests and the
  ordinary feature-axis tree reduction; and
- a private `_aggregation_writer.py` module owns Phase B CSR block readers,
  component-wise AnnData writes, SpatialData table-group adoption and local
  publication/rollback.

Put shared immutable payload models in the lowest-level module that owns their
contract, or in a narrowly scoped private contracts module if both phases need
them. Keep dependency direction from orchestration to checkpoint/writer code;
the public table validator may consume their persisted schemas, but these
modules must not import the validator and create a cycle.

### 7b verification and performance contract

Focused correctness tests should establish that:

- Phase A executes the Slice 7a assignment graph exactly once, keeps local
  partial reductions graph-internal and writes only merged long-form count
  blocks, not copies of the original assigned points; no eager byte-size
  repartition prepass is allowed to trigger a second execution;
- the pair-and-instance shuffle uses Dask's default preserved output-partition
  count, lets the ordinary Dask Parquet writer determine the corresponding part
  files, ignores empty results and does not perform custom sizing or
  post-checkpoint compaction;
- checkpoint rows have the exact declared schema, contain no null features,
  use positive `uint64` instance IDs and `uint64` counts, and are globally
  unique by `(aggregation_pair, instance_id, feature)`; all rows for one
  `(aggregation_pair, instance_id)` occur in one checkpoint partition;
- an unbacked `SpatialData` is rejected before pair normalization or Dask graph
  construction with an error that demonstrates `sdata.write("sdata.zarr")`;
- class-aware checkpoint partitions share the complete panel-derived
  expression and auxiliary axes, including features with zero detections on
  either axis;
- ordinary checkpoint partitions share the exact sorted result of a
  distributed feature-set tree union; complete per-fragment feature lists are
  not all copied to the driver;
- partial counts for one instance that originate in different input partitions
  are merged into exactly one output row, and a merged value above `uint32`
  maximum is rejected before CSR conversion rather than wrapped by a cast;
- any requested aggregation pair with no retained non-background instance is
  rejected by an error naming its labels and points elements, with no partial
  table publication;
- output rows follow the checkpoint manifest's partition-major order, every
  output component uses that exact order, and composite
  `(labels_name, instance_id)` identities prevent collisions across regions;
- no complete instance-feature count series or sparse expression/auxiliary
  matrix is materialized on the driver;
- every delayed CSR block spans its complete shared feature axis and partitions
  rows only, with `uint32` values and 64-bit-safe index arrays;
- one center calculation per labels element supplies a complete finite center
  dataframe, which is aligned through composite identities and written once;
- injected local-reduction, merged-checkpoint, block-conversion, `X`, `.obs`,
  `.obsm`, metadata and directory-rename failures leave no visible partial
  table, preserve or restore an overwritten table and clean up only paths
  recorded as owned by the current call;
- the out-of-core path uses AnnData's lower-level component writers and
  never calls `SpatialData.write_element()` for the completed table; its
  isolated publication helper writes the same table-group attributes as
  `SpatialData.write_table()`;
- the returned same-process AnnData uses backed `sparse_dataset` handles for
  `X` and the optional auxiliary matrix, reuses the already-constructed small
  components, opens those handles from the final path only and does not call
  `spatialdata.read_zarr()`;
- supported Zarr v2 and v3 stores, where both are supported by the pinned
  AnnData/SpatialData stack, receive equivalent AnnData and SpatialData table
  metadata and survive reopening;
- `X`, `.obs`, the auxiliary feature matrix and spatial centers use the exact
  row order declared by the manifests, rather than merely compatible first-axis
  sizes;
- the final `.uns`, `spatialdata_attrs`, table-group metadata, categorical
  columns, `.var` axis, auxiliary schema and `.obsm` payloads survive AnnData
  and SpatialData Zarr round trips; and
- the out-of-core implementation produces the same values as focused
  in-memory reference calculations in ordinary and class-aware tests; the
  reference is test code, not a supported production branch.

Benchmark Slice 7b independently from the Slice 7a assignment benchmark. Record
wall time, peak worker memory, spill volume, peak driver memory during Phase A,
Phase B and component publication, maximum checkpoint-partition rows/nonzeros,
expression and auxiliary output bytes, center calculation time and bytes,
temporary-store size, checkpoint file count and size distribution, and `.obs`
assembly cost.
Include both a small case, where staging overhead can dominate, and a
representative full backed mosaic. Report partition skew and checkpoint file
sizes, but do not make equal partition sizes a correctness requirement. Confirm
that Dask's preserved partition count and normal spilling are adequate for the
representative workload. Treat custom repartitioning or compaction as a future
optimization only if those measurements reveal a concrete problem. Confirm
that peak driver memory no longer scales with the total number of observed
instance-feature pairs and that materializing the one-row-per-instance `.obs`
and center payloads is acceptable.

Slice 7b is complete when it executes the Slice 7a handoff exactly once, merges
compact reductions into the specified validated long-form checkpoint,
preserves the complete Slice 6 payload and label-derived centers,
never materializes the complete instance-feature reduction or final sparse
matrices on the driver, rejects empty requested pairs and count overflow,
publishes through the defined local staging/rollback contract without exposing
a partial table, and materially lowers peak memory on the representative
large-mosaic workload without a major regression on the small case. General
lazy reopening of persisted SpatialData tables is not part of this completion
criterion.

## Slice 8: Harpy-owned canonical centers and `aggregate_points` integration

**Status: implemented.**

The Harpy-owned implementation was ported against napari-harpy main commit
`698a51dd54d80cf5707aa625f5c353cd24ff2dd7`. Migrating napari-harpy's imports
and annotation-query boundary to consume this public Harpy package is a
coordinated downstream change in that repository; it does not block completion
of the Harpy package and `aggregate_points` integration described here.

Promote the viewer-independent canonical-center calculation, typed contracts,
storage schema and validation from napari-harpy into Harpy, then integrate that
shared implementation into `hp.tb.aggregate_points`. At implementation time,
fetch and inspect the current main branch of
[`vibspatial/napari-harpy`](https://github.com/vibspatial/napari-harpy) rather
than copying from an older installed release or assuming that the local
checkout still represents the authoritative implementation. Port the focused
canonical-center tests together with the contract.

Treat that current main-branch implementation as the source implementation,
not merely as design inspiration. Copy its viewer-independent models, schema
handling, validation, calculation logic, error semantics and focused tests into
Harpy with minimal changes. Limit adaptations to package ownership and imports,
the agreed 3D extension, the single-canonical-payload contract and direct
`aggregate_points` integration. Do not independently redesign equivalent
contracts during the port. Widget/controller behavior and napari-specific
mutation or persistence remain outside the copied core.

The numerical definition remains the center of mass of uniformly weighted,
non-background pixels belonging to an instance. Napari-harpy already computes
this through `harpy.utils.RasterAggregator.center_of_mass`; the functionality
being promoted is principally its source and table binding, cache validation,
versioned metadata, deterministic instance-set identity and spatial-query
interoperability.

### Ownership and dependency direction

Harpy must own the reusable, non-UI implementation. Core Harpy code must never
import `napari_harpy`: napari-harpy already depends on `harpy-analysis`, so the
reverse import would create a dependency cycle and make a core table operation
depend on a viewer application.

Separate the main-branch implementation along this boundary:

- move or adapt canonical source signatures, region bindings, instance-set
  digests, metadata models, storage serialization/parsing, matrix validation
  and center calculation into the public
  `harpy.table.canonical_centers` package;
- keep widgets, workers, query-controller state and napari-specific component
  persistence in napari-harpy;
- make napari-harpy consume the Harpy-owned canonical contracts and calculation
  after the Harpy implementation is available; and
- avoid two independently evolving definitions of canonical coordinates or
  their metadata schema.

The port should retain the established storage keys and schema semantics rather
than introduce a second Harpy-specific canonical-center representation. Extend
the owned schema deliberately for three-dimensional labels as described below,
and update napari-harpy to consume that shared Harpy contract.

### Module and public API boundary

Use a dedicated public package rather than adding this substantial contract to
`_allocation.py` or placing it in one private `_canonical_centers.py` module:

```text
src/harpy/table/canonical_centers/
├── __init__.py
├── _models.py
├── _schema.py
└── _calculation.py
```

The responsibilities are:

- `_models.py` owns the immutable source-signature, region-binding, metadata,
  cache-report and result models together with deterministic instance-set
  identity;
- `_schema.py` owns the fixed storage keys, schema version, source-signature
  construction, serialization, parsing and matrix/metadata validation;
- `_calculation.py` normalizes 2D and 3D labels, calls
  `harpy.utils.RasterAggregator.center_of_mass` and validates instance ordering
  and finite coordinates; and
- `__init__.py` exposes the stable symbols that napari-harpy and Harpy's table
  aggregation code are allowed to consume.

The center-of-mass numerical engine remains
`harpy.utils.RasterAggregator.center_of_mass`; the new package owns the
canonical semantics around that calculation: fixed axes, intrinsic source
frame, labels/table binding, persisted metadata and validation. Because
napari-harpy is an external consumer of this contract, it must import from
`harpy.table.canonical_centers`, not from a private Harpy module.

Keep aggregation orchestration and component writing outside this package.
`_allocation.py` should replace its private `_label_centers()` path with the
shared canonical calculation and assemble one payload per aggregation pair.
`_aggregation_writer.py` should align those payloads to the checkpoint row
manifest and write `spatial_canonical` plus its metadata. Focused Harpy tests
for the reusable contract should live in
`src/harpy/_tests/test_table/test_canonical_centers.py`; aggregation integration
tests remain with the existing table-aggregation tests.

### Canonical table payload

For supported two- and three-dimensional labels, `hp.tb.aggregate_points`
should produce exactly one center-coordinate payload:

```text
adata.obsm["spatial_canonical"]
    dense float64 intrinsic-label coordinates in fixed (z, y, x) order

adata.uns["spatial_coordinates"]["spatial_canonical"]
    versioned canonical matrix, calculation, source and coverage metadata
```

The canonical matrix always has shape `(adata.n_obs, 3)`. For a 2D source its
`z` column is exactly zero; for a 3D source it contains the measured center of
mass along the source `z` axis. All values are expressed in the intrinsic pixel
coordinate frame of the corresponding `scale0` labels element. Do not apply
the scale, translation or other SpatialData transformation into the selected
aggregation coordinate system. Integer array indices represent pixel centers,
so a one-pixel 2D instance at row 10 and column 20 has canonical center
`(0.0, 10.0, 20.0)`.

For a table spanning multiple labels regions, rows may therefore refer to
different intrinsic labels frames. The per-region metadata binds every row to
its source labels element; the matrix must not be interpreted as one shared
global coordinate system.

The metadata record should preserve the main-branch schema, including at
least:

- schema version, fixed `obsm_key`, axes and dtype;
- the table's `region_key` and `instance_key`;
- one record for every labels region represented by the table;
- the source labels element, element type, scale and intrinsic coordinate
  frame;
- the center-of-mass method, pixel weighting, background value, pixel-center
  convention and algorithm version;
- complete table-row coverage for the region, its row count and deterministic
  instance-set digest; and
- the source dimensions, shape and normalized integer dtype.

“All rows for region” means all rows that `aggregate_points` retained for that
labels region. It does not mean all nonzero IDs present in the labels raster;
instances receiving no assigned point remain outside the aggregation table as
specified by Slice 6.

### Single canonical-coordinate contract

Replace the existing configurable `adata.obsm[spatial_key]` center payload with
the fixed `adata.obsm["spatial_canonical"]` contract. Do not retain a second
translated `spatial` matrix. Remove the `spatial_key` parameter from
`hp.tb.aggregate_points`; the canonical key is an interoperability contract and
is not user-configurable.

`to_coordinate_system` remains necessary for assigning points to labels and is
recorded with the aggregation-region metadata. It does not determine the
stored center coordinates. Canonical centers always remain in the source
labels element's intrinsic `scale0` frame.

This is the frame expected by the spatial-query path. Annotation geometry is
transformed from the selected query coordinate system into the relevant labels
element's intrinsic frame before containment is evaluated against the
canonical `x` and `y` columns. No global or pair-coordinate-system copy of the
centers is needed.

Apply the single canonical payload to both ordinary and class-aware
aggregation. Concatenate each region's canonical block in exactly the same
normalized pair and instance row order as `X`, `.obs` and the optional
auxiliary feature matrix.

### Construction and publication

Calculate and validate the canonical matrix and metadata as part of table
construction, replacing the existing transformed-center payload. The canonical
payload must be present before the table is published rather than being added
through a later napari-harpy mutation.

Integrate with Slice 7b's required backed component writer. The complete dense
canonical matrix may reside on the driver, must use the established row
manifest and is written once through `anndata.io.write_elem` before
publication. Do not reintroduce an unbacked or in-memory `add_table` branch
merely to add canonical centers.

The implementation sequence should be:

1. resolve the exact retained instance IDs and row positions for every labels
   region;
2. calculate one intrinsic center block per region through the shared Harpy
   implementation;
3. concatenate all blocks in final table-row order and align them through the
   composite `(labels_name, instance_id)` identity;
4. construct the complete per-region canonical metadata registry;
5. validate the matrix, binding, source signatures, coverage and serialized
   schema together; and
6. write the canonical coordinate payload once through Slice 7b's backed
   component writer before publication.

Extend the public `hp.tb.validate_table()` validator to check the canonical
contract whenever either `adata.obsm["spatial_canonical"]` or its
`adata.uns["spatial_coordinates"]["spatial_canonical"]` metadata record is
present. Reuse the Harpy-owned schema validation used during construction and
require matrix/metadata symmetry, schema version 1, the fixed matrix key and
`(z, y, x)` axes, dense `float64` shape, valid table region/instance linkage,
resolvable per-region source labels, matching source dimensions, shape and
dtype, complete row coverage, and the expected instance-set digest.

This is a structural and metadata-consistency check. `validate_table()` must
not recompute centers of mass or scan labels pixel values. Tables containing
neither canonical component remain valid generic SpatialData tables; tables
with only one component, or with an inconsistent canonical payload, are
rejected.

Napari-harpy's ensure/read path remains useful for older or externally created
tables without canonical coordinates. For a new Harpy aggregation table,
napari-harpy cache inspection should report the selected region as valid and
reuse the stored centers without executing labels-array tasks.

### Dimensional scope

Use schema version 1 for the Harpy-owned canonical-center contract and allow
exactly these source layouts:

```text
2D source dims: (y, x)     -> canonical axes: (z, y, x), with z = 0
3D source dims: (z, y, x)  -> canonical axes: (z, y, x)
```

Both dimensions use the same calculation, alignment, metadata and AnnData
integration path. Normalize a 2D labels array by prepending a singleton `z`
axis before `RasterAggregator.center_of_mass`; pass a 3D `(z, y, x)` source
through directly. Reject other dimension orders rather than transposing them
implicitly.

The napari-harpy implementation being promoted currently restricts its
schema-v1 parser to 2D. Broaden that validation as part of moving ownership to
Harpy; do not introduce schema version 2, a compatibility parser or a migration
path. The 2D serialized representation does not change, while 3D becomes an
additional valid source dimensionality under the same Harpy-owned schema.

Update napari-harpy to consume and validate the shared schema-v1
representation. Its cache inspection may recognize both 2D and 3D canonical
payloads as valid, but the annotation spatial-query entry point must explicitly
require a 2D `("y", "x")` source and raise a clear
unsupported-dimensionality error for 3D. The limitation belongs to the
two-dimensional polygon-query operation, not to canonical-center construction
or storage.

### Verification

Focused tests should establish that:

- Harpy imports no `napari_harpy` module and the package dependency direction
  remains napari-harpy to Harpy;
- Harpy's port matches the fetched main-branch calculation and serialized
  semantics for representative 2D labels;
- `spatial_canonical` is dense `float64`, has shape `(n_obs, 3)`, uses exact
  `(z, y, x)` ordering and has a zero `z` column for 2D labels;
- 3D labels produce the same matrix shape and axis order with their measured
  `z` centers rather than a synthetic zero column;
- canonical values remain intrinsic when their labels elements have nonzero
  SpatialData transformations;
- exactly one labels center-of-mass reduction supplies the single canonical
  matrix and no translated center matrix is written;
- irregular labels, multiple chunks and requested instance-ID ordering match a
  simple in-memory reference;
- multiple regions remain aligned with table rows even when their local
  instance IDs overlap;
- every canonical metadata region binds the exact table rows and source labels
  signature through the expected instance-set digest;
- malformed matrices, unsupported schema versions, incomplete metadata,
  missing instance centers and matrix/metadata asymmetry are rejected without
  publishing a table;
- the matrix and registry survive AnnData and SpatialData Zarr round trips;
- napari-harpy recognizes a Harpy-produced 2D table cache as valid and performs
  a spatial canonical-center query without recalculating labels centers;
- ordinary and class-aware aggregation receive the same canonical-center
  contract;
- Harpy accepts and validates a schema-v1 3D canonical payload; and
- napari-harpy recognizes that 3D payload structurally but rejects it at the
  annotation-query boundary with a clear dimensionality error.

## Slice 9: canonical centers for existing SpatialData tables

**Status: implemented.**

Add a high-level operation that attaches the canonical-center contract from
Slice 8 to a compatible table created outside `hp.tb.aggregate_points`. This is
a complementary entry point, not a replacement for constructing canonical
centers inside `aggregate_points`: newly aggregated tables must continue to be
published with their canonical payload already present and valid.

The public API should be:

```python
hp.tb.add_canonical_centers(
    sdata,
    table_name="table",
    labels_name=None,
    overwrite=False,
)
```

with the corresponding parameter contract:

```python
def add_canonical_centers(
    sdata: SpatialData,
    *,
    table_name: str,
    labels_name: str | Sequence[str] | None = None,
    overwrite: bool = False,
) -> SpatialData: ...
```

The operation mutates the selected table in the supplied `SpatialData` object
and returns that same object, following the convention of other Harpy table
operations.

`labels_name=None` derives all source labels names from the SpatialData table
annotation. An explicit string or sequence is an assertion about those source
regions and must resolve to the complete set registered by the table; it does
not define a partial output or change table-row order. Reject duplicate,
missing, additional or non-label elements. The first implementation supports
2D and 3D labels sources only. Shapes or other instance representations remain
outside its scope. Every labels region declared by the table annotation must
own at least one observation; reject an annotation containing a declared region
with no table rows rather than creating empty canonical-region metadata.

### Table-bound calculation

Canonical centers are table-aligned label-instance centers rather than an
unqualified list of every centroid in a labels raster. The operation must
therefore require a valid SpatialData table annotation and derive:

- `region_key` and `instance_key` from `TableModel` metadata;
- the registered labels region for each observation;
- the exact positive instance IDs retained by the table; and
- the final row positions to which each region's centers belong.

Use the public `harpy.table.canonical_centers` models, schema and calculation
from Slice 8. Calculate only the instance IDs represented by table rows, retain
the exact table order, and persist intrinsic `scale0` coordinates in fixed
`(z, y, x)` order. A 2D source receives `z=0`; a 3D source keeps its measured
`z` center. No points element is needed, and no point-to-label assignment is
performed.

The existing pure
`hp.tb.canonical_centers.calculate_canonical_centers()` function remains the
non-mutating calculation primitive for an established region binding. Do not
introduce a second numerical center-of-mass implementation or make callers
construct low-level cache reports for the high-level add operation.

### Metadata and collision contract

Write exactly the same two coordinated components as `aggregate_points`:

```text
adata.obsm["spatial_canonical"]
adata.uns["spatial_coordinates"]["spatial_canonical"]
```

Treat `adata.uns["spatial_coordinates"]` as a registry. Preserve any unrelated
records already stored beneath it; the relevant collision is the nested
`"spatial_canonical"` record, not the registry key itself. Likewise, preserve
all unrelated `.obsm`, `.uns`, `.obs`, `.var`, `.layers` and `X` content.

When either canonical component already exists, `overwrite=False` must raise a
clear collision or incomplete-payload error without changing the table. With
`overwrite=True`, recalculate and replace both canonical components together,
including when the previous payload is stale or asymmetric. Never retain a new
matrix with old metadata, or new metadata with an old matrix.

Preflight validation must distinguish the table being updated from the
canonical payload being repaired. Refactor the shared table-validation internals
so this operation can validate the SpatialData annotation, registered feature
matrices, feature-class aggregation metadata and all other recognized Harpy
contracts without first requiring the existing canonical pair to be valid. Do
not add a permissive option to the public `hp.tb.validate_table()` API; that
validator must remain strict.

Apply validation and collision handling in this order:

1. validate the arguments, local writable backing store and in-memory/on-disk
   table identity;
2. validate the table annotation and every recognized non-canonical table
   contract;
3. resolve the complete annotated labels-region set and its row bindings;
4. inspect whether either existing canonical component is present;
5. with `overwrite=False`, reject any complete or incomplete canonical
   collision without calculating centers;
6. with `overwrite=True`, allow the old canonical pair to be absent, valid,
   stale, malformed or asymmetric, but do not ignore an unrelated table
   inconsistency;
7. calculate and assemble a complete replacement payload for every annotated
   region; and
8. validate that newly constructed payload with the shared Slice 8 validator
   before publication.

The persisted result must cover every annotated table row and region. A missing
label ID, unsupported labels layout, non-integer labels dtype, invalid table
identity, non-finite center or source-signature mismatch must fail without
publishing a partial update.

### Backed publication

Initially require a SpatialData object backed by a writable local Zarr store,
matching the bounded-memory publication boundary of `aggregate_points`. Update
the canonical matrix and registry through a staged component replacement with
rollback across both coordinated writes. Do not rewrite or materialize the
table's expression and auxiliary matrices merely to add the small dense center
matrix and metadata record. On success, refresh the in-memory table handles and
consolidated SpatialData metadata; on failure, restore the previous canonical
components when they existed.

Stage and publish only these two component paths:

```text
tables/<table_name>/obsm/spatial_canonical
tables/<table_name>/uns/spatial_coordinates/spatial_canonical
```

The second path is one entry in a registry: create its parent registry when
absent and preserve every unrelated child when it already exists. Keep rollback
copies until both component replacements, the in-memory refresh and
consolidated-metadata write have succeeded. A failure at any of those steps must
restore both previous disk components and both previous in-memory values,
including the asymmetric case in which only one component existed.

Do not use the aggregation-specific `_read_backed_table()` helper or a general
`anndata.read_zarr()` call to reconstruct the existing table. The former only
covers components written by Harpy's aggregation path, while the latter could
materialize unrelated data. Instead, attach the newly published storage-backed
canonical array and decoded canonical metadata record to the existing in-memory
AnnData table. This preserves its object payload—including `X`, `.layers`,
`.raw`, `.obsp`, `.varm`, `.varp` and unrelated `.obsm`/`.uns` entries—without
reading or rewriting those components.

### Verification

Focused tests should establish that:

- a valid externally created single-region table receives the same canonical
  matrix and metadata as an equivalent `aggregate_points` table;
- a multi-region table remains aligned when local instance IDs overlap;
- `labels_name=None` derives all annotated labels regions and an explicit
  mismatching selection is rejected before calculation;
- only table-retained instance IDs are calculated and every table row receives
  exactly one center;
- 2D and 3D labels use the shared schema-v1 dimensional contract;
- unrelated `spatial_coordinates` registry entries and all other AnnData
  components survive the operation unchanged;
- collision, stale and asymmetric payload cases obey `overwrite` exactly;
- `overwrite=True` repairs an invalid canonical payload but still rejects an
  unrelated invalid table contract;
- a declared labels region without table observations is rejected before
  calculation;
- a failed calculation, validation or component write leaves the prior table
  and canonical payload intact;
- the function returns the same updated `SpatialData` object;
- the result passes `hp.tb.validate_table()` and survives an AnnData and
  SpatialData Zarr round trip; and
- `aggregate_points` continues to create canonical centers directly without
  requiring this follow-up API call.

## Slice 10: general point-to-label transformations

**Status: implemented.**

Generalize the spatial-assignment boundary used by `hp.tb.aggregate_points`
and `hp.tb.bin_counts`. Points and labels must still share the selected
`to_coordinate_system`, but neither element should be required to use a
particular transformation into it. Replace the current requirement that points
use an identity transformation and labels use only a pixel-aligned translation
with one relative, invertible transformation from the intrinsic points frame to
the intrinsic labels frame.

This slice changes coordinate normalization before the Slice 7a chunk-routing
algorithm. It does not change feature aggregation, checkpoint construction,
AnnData publication, canonical-center storage, or the biological meaning of an
assigned point.

### Relative transformation contract

For one normalized aggregation pair, let:

- `T_points` map intrinsic point coordinates into `to_coordinate_system`; and
- `T_labels` map intrinsic labels coordinates into the same coordinate system.

Map every point into the labels raster's intrinsic pixel frame as follows:

```text
points intrinsic -- T_points --> selected shared coordinate system
                                             ^
                                             | T_labels
                                             |
labels intrinsic <-- T_labels^-1 ------------+

points intrinsic -- T_labels^-1 o T_points --> labels intrinsic
```

Equivalently, for a point coordinate vector `p`:

```text
p_labels = T_labels^-1(T_points(p))
```

With homogeneous column-vector matrices, the relative matrix is therefore:

```text
M_points_to_labels = inverse(M_labels) @ M_points
```

For example, suppose one labels pixel spans `0.5` shared-coordinate units and
the labels origin is `(100, 50)`. A point at shared coordinate `(102, 51)` maps
to intrinsic labels coordinate `(4, 2)` and is looked up as
`labels[y=2, x=4]`.

Resolve this relation directly through SpatialData's public transformation API.
Fetch each element's transformation to the explicitly selected shared system:

```python
T_points = get_transformation(points, to_coordinate_system=to_coordinate_system)
T_labels = get_transformation(labels, to_coordinate_system=to_coordinate_system)
```

Direct lookup is the shared implementation contract because
`hp.tb.bin_counts` constructs a temporary points element that is not registered
in the supplied `SpatialData` object. A graph operation such as
`get_transformation_between_coordinate_systems()` may be used only when both
elements are registered and the path is explicitly constrained through
`to_coordinate_system`; it must not be the sole resolver used by the common
assignment helper.

Convert `T_points` and `T_labels` separately to homogeneous matrices using the
same canonical axes. Calculate the relative matrix with a linear solve rather
than forming an explicit numerical inverse:

```python
M_points_to_labels = np.linalg.solve(M_labels, M_points)
```

This implements `inverse(M_labels) @ M_points` while providing a direct failure
for a singular labels transformation. Validate the point transformation's
invertibility separately so both element registrations satisfy the stated
coordinate-system contract.

Resolve and validate the relative transformation once per aggregation pair,
before creating the aggregation workspace or constructing the assignment graph.
Represent the result as one small immutable private contract containing the
point axes, labels axes, and homogeneous point-to-label matrix in canonical
`(x, y)` or `(x, y, z)` coordinate order. Pass that resolved contract to the
private assignment implementation. Do not repeatedly inspect transformation
metadata inside Dask partitions.

### Partition-wise assignment

Do not resample or transform the labels raster. Instead, extend the
partition-local classification phase from Slice 7a:

```text
points partition
      |
      v
apply M_points_to_labels vectorially
      |
      v
continuous intrinsic labels coordinates
      |
      v
round once to integer labels-pixel indices
      |
      v
extent filter and labels-block classification
      |
      v
existing block shuffle and vectorized labels lookup
```

The matrix application must remain lazy and partition-wise. It may allocate
coordinate arrays proportional to one points partition, but must not materialize
the complete points element. Do not implement this by calling a whole-element
SpatialData points transformation when that operation materializes Dask
partitions or non-coordinate columns. Reuse SpatialData's transformation
definitions and composition semantics while retaining Harpy's bounded,
partition-local execution.

Apply the relative transformation before rounding. Rounding coordinates in the
shared coordinate system before inverse scaling or rotation would select the
wrong labels pixel. Preserve the existing half-to-even `round()` convention for
the final intrinsic labels coordinates, then reorder `(x, y[, z])` values into
the raster's `(y, x)` or `(z, y, x)` indexing order. The existing half-open
extent and chunk-boundary rules, row-major block IDs, one shuffle by block ID,
background filtering, and one vectorized lookup per labels block remain
unchanged.

Store labels-local floating and integer coordinates in collision-safe temporary
columns and remove them after lookup. Do not overwrite retained source
point-coordinate columns with labels-local values. When `drop_coordinates=True`,
as in `aggregate_points`, return only the requested value columns and assigned
instance ID. When coordinates are retained by another caller, document that
they remain the source point coordinates; intrinsic lookup coordinates are an
internal implementation detail.

Resampling labels into the shared coordinate system is explicitly out of scope.
It would allocate a new raster, require nearest-neighbour interpolation to
preserve IDs, potentially change segmentation boundaries, and decouple the
assignment graph from the stored labels chunks. Inverse point mapping preserves
the original integer labels and naturally reuses the existing chunk-aware
lookup.

### Supported transformations and validation

Support same-dimensional SpatialData transformations that can be represented
as a finite, invertible affine matrix, including:

- identity and arbitrary finite translation;
- scale;
- rotation, reflection, and shear;
- invertible axis permutation; and
- sequences composed from those transformations.

This removes the pixel-aligned-translation restriction. A fractional labels
translation is valid because the complete relative transform is applied before
the final labels-pixel rounding.

Reject clearly during pair preflight, before creating the aggregation workspace,
constructing the assignment graph, or reading either source:

- either element missing `to_coordinate_system`;
- a singular or otherwise non-invertible points or labels transformation;
- a transformation that cannot be represented as an affine matrix;
- a matrix with non-finite entries;
- incompatible point and labels dimensionality;
- dimension-adding or dimension-dropping mappings; and
- unsupported labels dimensions outside `(y, x)` and `(z, y, x)`.

Source-coordinate validity is necessarily a partition-time check. Reject a
point partition containing non-finite source coordinates or producing
non-finite labels-local coordinates when that partition executes. This failure
may occur while the checkpoint graph is executing, so other partitions may
already have written temporary files in the call-owned workspace. The
checkpoint must not be accepted or consumed, the final table must not be
published, and the complete owned workspace must be cleaned normally. Do not
add a separate eager scan solely to validate coordinate values, because that
would read every points partition twice.

“General transformation” in this contract means any compatible, invertible
SpatialData affine transformation. It does not promise support for arbitrary
nonlinear deformation fields. A future nonlinear implementation would require
an explicit inverse-coordinate API with the same partition-wise and
pixel-sampling semantics.

Keep `to_coordinate_system` as the public selection mechanism; no public matrix
or transformation parameter is needed. The SpatialData element registrations
remain authoritative, so do not duplicate the composed relative matrix in table
metadata. Aggregation-region metadata continues to record the selected shared
coordinate system.

Canonical centers are unaffected. They remain in each labels element's
intrinsic `scale0` pixel frame under the Slice 8 contract, regardless of the
transformations used to establish point-to-label correspondence. Generalized
assignment must not introduce a second transformed center matrix.

### Verification and performance contract

Focused tests must establish that:

- identity points plus translated labels produce the exact existing assignment
  result;
- translated or scaled points and independently translated or scaled labels are
  composed in the correct direction;
- rotation, reflection, shear, and invertible axis permutation match a simple
  in-memory reference;
- 2D and 3D transformations respect `(x, y[, z])` matrix order and
  `(z, y, x)` raster indexing order;
- points crossing labels-chunk boundaries after transformation are routed and
  assigned exactly once;
- background and out-of-bounds points after inverse mapping are removed;
- rounding happens only after mapping into the intrinsic labels frame;
- source point-coordinate columns are not replaced by transient labels-local
  coordinates;
- missing coordinate systems, dimensional mismatch, non-finite matrices, and
  singular transformations fail during pair preflight before workspace creation
  or graph execution;
- non-finite source or transformed point coordinates fail in their executing
  partition before checkpoint publication, without an additional validation
  scan;
- graph construction performs no source reads and the complete points or labels
  elements are never materialized;
- ordinary and class-aware `aggregate_points` produce unchanged feature and
  metadata contracts apart from accepting the broader registrations;
- `bin_counts` preserves its assignment and retained-coordinate behavior; and
- canonical centers remain intrinsic and unchanged for non-identity
  registrations.

Include an invariance regression test that prepends the same invertible affine
transformation `A` to both element registrations. Assignments must not change,
because:

```text
inverse(A o T_labels) o (A o T_points)
    = inverse(T_labels) o T_points
```

Benchmark the generalized partition transform against the existing
identity/translation case on representative 2D and 3D point partitions. Record
assignment wall time, graph-construction time, task count, shuffle volume, and
peak worker memory. The additional work should remain linear in the number of
points and bounded by one partition. Retain a specialized identity/translation
fast path only if measurements show that the general vectorized affine path
causes a material regression; both paths must share the same rounding, bounds,
and lookup contracts.

## Slice 11: original-point control QC

**Status: specified; not implemented.**

Add separate lightweight QC functions over the original transcript points.
This slice is scheduled after Slice 10, but its runtime contract depends only on
the points and feature-panel metadata from Slice 1 and the sample-aware point
metadata from Slices 2–4. The original-point summaries may run before or after
aggregation and do not depend on an instance-label raster or an AnnData table.
The optional per-instance plotting view described below requires a class-aware
table produced from the Slice 5 contract but derives its rates without
modifying that table.

This operation complements the instance-level `.obs` metrics. It must use the
original points element so that controls on label value zero and controls
outside segmented instances remain visible. It must not create another copy of
the points or route individual controls through `hp.tb.aggregate_points`.

### Derived per-instance plotting metrics

When a class-aware aggregation table is available, QC plotting may derive
`negative_points_per_feature` and `system_control_points_per_feature` from the
persisted raw class counts. Resolve the relevant `.obs` columns through
`adata.uns["feature_class_aggregation"]["count_columns"]` and divide them by the
corresponding positive values in `auxiliary_class_feature_counts`:

```text
negative_points_per_feature =
    n_negative_points / auxiliary_class_feature_counts["Negative"]

system_control_points_per_feature =
    n_system_control_points / auxiliary_class_feature_counts["SystemControl"]
```

These rates normalize for the different numbers of panel features in the two
control classes. They should be temporary series or plotting-dataframe columns;
do not persist them back into `adata.obs`. The plotting layer must validate
that the referenced count columns exist and that the stored auxiliary class
feature counts are positive. This optional cell-level view complements, but does not replace, the
original-point summaries below.

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
processes and have different numbers of panel features.

### Spatially binned summary

Bin the original control points in the coordinate system of their mosaic and
produce separate spatial density layers for negative probes and system-control
codewords. Normalize each layer by its authoritative number of panel features;
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
- per-instance plotting rates use the table's recorded count-column bindings
  and authoritative denominator snapshot without modifying `.obs`;
- sample and mosaic coordinate systems remain independent; and
- the implementation stays lazy until the compact summaries are computed.

## Slice 12: lazy SpatialData table reopening

**Status: follow-up; not implemented.**

Make persisted AnnData tables reopen lazily when a user later calls
`spatialdata.read_zarr()`. This is separate from Slice 7b's out-of-core writer:
Slice 7b must construct and publish a table without materializing its sparse
matrices and attach that table to the same-process result with backed AnnData
`sparse_dataset` handles, but it does not change SpatialData's general Zarr
reader.

Investigate the current SpatialData table I/O boundary and prefer an upstream
public integration over patching private reader internals in Harpy. A reopened
table should retain lazy or backed `X`, sparse auxiliary feature matrices and
dense coordinate matrices while preserving pandas-compatible `.obs` and
`.var`, `TableModel` validation, table annotations and the existing
`SpatialData` mapping interface. The implementation must work for ordinary and
class-aware aggregation tables without depending on CosMx metadata.

Focused tests should establish that a table written by Slice 7b can be reopened
without loading its complete matrices, passes the SpatialData table contract,
supports normal row and feature access, and produces the same materialized
values as `anndata.read_zarr`. Benchmark store-open time and driver memory
independently from Slice 7b's construction benchmark.

## Slice 13: optional Slice 7b latency optimization

**Status: optional follow-up; not implemented.**

Optimize wall-clock performance of the Slice 7b out-of-core path only where
measurements show material overhead. Slice 7b deliberately exchanges some
latency for bounded driver memory: it writes a durable merged-count Parquet
checkpoint and later consumes that checkpoint to construct the AnnData
components. On datasets whose reduced instance-feature counts would already
fit comfortably in memory, checkpoint serialization, repeated decoding and
Dask task scheduling can therefore make the scalable path slower than the
former driver-materializing implementation. This is an expected trade-off, not
by itself evidence that the count results are incorrect.

Do not introduce a second in-memory implementation as the default response.
First profile the existing generic path on representative small, medium and
large datasets. Record wall-clock time, peak driver and worker memory, task
count, checkpoint partition count and sizes, and bytes read and written for at
least these phases:

1. chunk-aware point-to-label assignment and local count reduction;
2. compact-count shuffle, duplicate merge and Parquet checkpoint publication;
3. label-derived center calculation;
4. class-summary construction;
5. expression-matrix construction and write;
6. auxiliary-matrix construction and write; and
7. staged publication and consolidated-metadata update.

Compare ordinary and class-aware aggregation separately. Also record the input
points partition count and labels chunk grid, because Slice 7a creates one
routed points partition per labels block and Slice 7b currently carries that
partitioning into the compact-count shuffle. A moderate number of large blocks
may be efficient, while many tiny non-empty checkpoint parts can make Parquet,
Dask scheduling and AnnData component-write overhead dominate.

### Candidate optimizations

Apply the following only in response to the measurements, in this order:

1. **Reuse Phase A class summaries.** While each merged-count partition is
   already available for its checkpoint manifest, derive its compact
   one-row-per-instance class totals and retain those bounded summaries in the
   manifest or a companion artifact. Phase B can then construct the `.obs`
   class columns without rereading every Parquet part.
2. **Decode a checkpoint partition once for both sparse matrices.** In
   class-aware mode, expression `X` and `auxiliary_feature_counts` currently
   trigger independent component writes. Arrange a shared Phase B computation
   that maps one checkpoint decode to both CSR row blocks and writes both
   components without keeping the complete matrices on the driver. Merely
   sharing a delayed parent is insufficient if two sequential `write_elem`
   calls recompute it; the write graphs must execute together or use another
   public AnnData-compatible mechanism with equivalent safety.
3. **Coalesce demonstrably tiny checkpoint partitions.** Use observed
   checkpoint sizes rather than a fixed assumed Dask partition size. Coalescing
   must retain complete `(aggregation_pair, instance_id)` ownership within one
   output partition, the established partition-major row manifest and bounded
   worker memory. Do not expose a user-facing partition-tuning parameter unless
   benchmarks demonstrate that an automatic policy is insufficient.
4. **Measure publication overhead independently.** Consolidated-metadata
   updates and staged renames should not be optimized speculatively. If they
   are material, improve them without weakening rollback or exposing the hidden
   workspace as a SpatialData element.

The preferred result is at most one checkpoint decode per non-empty partition
during class-aware Phase B, producing its expression block, auxiliary block and
compact class summary from the same input. It must retain the durable Phase A
checkpoint; removing it would give up the recomputation and memory-pressure
boundary established by Slice 7b.

### Required invariants

Any optimization must preserve:

- one execution of the source assignment graph;
- globally merged and unique `(aggregation_pair, instance_id, feature)` count
  rows with checked `uint32` output conversion;
- the exact expression and auxiliary feature axes and partition-major row
  alignment;
- the complete assigned-instance row universe and label-derived centers;
- feature-panel validation and all class-aware `.obs`, `.obsm` and `.uns`
  contracts;
- backed sparse matrices in the same-process result;
- Zarr v2 and v3 output, staged publication, overwrite rollback and workspace
  cleanup; and
- bounded driver memory without materializing the complete instance-feature
  count relation or either complete sparse matrix.

Focused performance tests should prove numerical and metadata equivalence with
Slice 7b, verify that every checkpoint part is decoded no more than once during
the optimized class-aware Phase B, and report timing and peak-memory results
rather than relying on task-count reductions alone. A change should be retained
only when it gives a reproducible improvement on at least one representative
workload without a material regression in the large out-of-core case.
