# Class-aware transcript allocation

## Status

Seven implementation slices are planned; Slices 1 through 5 are implemented:

1. patch the CosMx reader and establish the generic Harpy feature-panel
   metadata contract — implemented;
2. make the canonical `harpy.io.cosmx()` creation API sample-aware —
   implemented;
3. validate existing sample-aware CosMx SpatialData stores — implemented;
4. add new CosMx samples incrementally to a validated sample-aware store —
   implemented;
5. add class-aware aggregation to `hp.tb.allocate` — implemented;
6. add QC functions that summarize the original, unallocated control points;
7. optimize the generic point-to-label assignment and reduction path.

Slice 2 replaces the current single-run reader surface with one coherent,
sample-aware creation contract. Slice 3 validates that an existing store still
satisfies that contract. Slice 4 incrementally extends a validated store without
rebuilding its existing samples. Slice 5 consumes the feature-panel metadata
produced by Slice 1 and the sample-aware element contracts established by
Slices 2–4. Class-aware allocation requires that every selected points element
reference authoritative feature-panel metadata; generic points without that
metadata remain supported by the ordinary, non-class-aware path. Slice 6's
original-point summaries depend on the reader metadata from Slices 1–4 rather
than allocation or labels; its optional per-instance plotting view derives
temporary rates from the Slice 5 table. Slice 7 preserves the public behavior
established by Slice 5 while replacing the private allocation execution path.

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
change `hp.tb.allocate` or introduce QC computations.

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
table-local denominator snapshot for later QC. Slice 6 additionally uses the
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

## Slice 5: class-aware `hp.tb.allocate`

**Status: implemented.**

This slice consumes the generic feature-panel contract established by Slice 1
and supports the sample-scoped elements created or added by Slices 2 and 4.
Class-aware allocation requires that metadata; ordinary allocation remains
usable with points that do not reference a feature panel and preserves its
existing behavior when `expression_class` is omitted.

### Scope and outcome

Generalize `hp.tb.allocate` so one call can allocate one or more explicitly
paired labels/points/coordinate-system regions and construct one complete
AnnData table. In class-aware mode, every point is spatially assigned once, but
the resulting payload is split by biological role: only the selected expression
class is retained as the sparse feature matrix, while every non-expression
class is reduced to per-instance control summaries.

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
adata.obs     class point counts and control_fraction
adata.obsm    mean coordinates calculated from assigned Endogenous points
adata.uns     the versioned feature_class_allocation configuration and sources
```

Multiple regions are combined without an inner feature join:

```text
labels[0] + points[0] + coordinate_system[0] ┐
labels[1] + points[1] + coordinate_system[1] ┼──> one AnnData table
...                                             ┘
```

Resolve one shared expression-feature axis before constructing any per-region
matrix. A feature assayed by the compatible shared panel but absent from one
region is represented by zeros for that region rather than removed. Control
denominators always come from referenced feature-panel metadata. Class-aware
allocation has no caller-supplied denominator fallback or override.

The point-to-label lookup is performed once per labels/points pair, carrying
both the feature and feature-class columns through the assignment. It is never
repeated once per class. Control points outside every instance are deliberately
excluded from these cell-level summaries; Slice 6 operates directly on the
original points to provide spatial background QC.

When `expression_class` is omitted, the same multi-region API follows the
ordinary allocation path: all observed features remain in `adata.X`, no
class-derived `.obs` columns are added, and no
`adata.uns["feature_class_allocation"]` record is written.

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

Extend `hp.tb.allocate` with one optional class-aware argument:

```python
expression_class: str | None = None
```

This parameter name, type, and default are final for this slice. The `append`
parameter is removed; `overwrite` controls only whether the completed table may
replace an existing table element.

Remove the deprecated `update_shapes_elements` parameter at the same boundary.
Allocation constructs the requested table and does not also mutate shapes
elements as an unrelated side effect; callers that need shape filtering should
request that operation separately.

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
additional `.obs` columns are created and every `name_gene_column` value remains
in `adata.X`.

When `expression_class` is provided, every selected points element must have a
root Harpy points record that references an authoritative shared feature panel.
The panel supplies `feature_class_key`, the complete ordered classes,
feature-to-class assignments, and the feature lists whose lengths define the
control denominators. Missing, malformed, or incompatible panel metadata is an
error before spatial lookup. Allocation accepts no denominator fallback,
override, or per-sample mapping.

The public `allocate()` docstring must explain both modes and show the exact
metadata path used in class-aware mode. In particular, it must state that
`feature_class_key` is not a user argument: it is resolved through
`sdata.attrs["harpy"]["points"][points_name]["feature_panel"]` and then read
from the referenced record in `sdata.attrs["harpy"]["feature_panels"]`. The
docstring must also state that control denominators are derived from the lengths
of the panel's non-expression `features_by_class` lists and that class-aware
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
`feature_key` to equal `name_gene_column`; read `feature_class_key` directly
from the panel rather than accepting it as an allocation parameter. Derive the
denominator for every non-expression class from the length of its authoritative
`features_by_class` list.

The metadata roles are deliberately distinct:

1. `sdata.attrs["harpy"]["feature_panels"]` is the authoritative assay
   definition.
2. `sdata.attrs["harpy"]["points"][points_name]["feature_panel"]` binds one
   points element to that definition.
3. `adata.uns["feature_class_allocation"]` is a derived, table-local snapshot of
   the resolved allocation contract. It records how the table was constructed
   and remains interpretable with the AnnData object, but it is not a second
   authoritative panel definition.
4. The generated `.obs` columns are derived payload and never a metadata source.

Resolve and validate these inputs once into one immutable internal allocation
contract before assigning points. Use that same object to define the shared
feature axis, aggregation behavior, generated `.obs` columns, and table-local
`.uns` record. Do not independently reinterpret root metadata at separate
stages of the operation.

There is no per-sample or per-region denominator mapping in one output table.
Likewise, `code_class` categories may differ between samples stored in the same
SpatialData object, but samples with different class universes or incompatible
panels cannot participate in the same class-aware allocation call. They require
separate output tables. Otherwise, later QC would interpret a given class-count
column against different panel denominators in different rows.

The complete policy is:

```text
all selected points reference compatible panels
    -> derive one shared class contract and denominator mapping

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

Derive one shared denominator mapping from the compatible panel contract for
the complete allocation call. Per-sample and per-region denominator mappings
are not supported because later QC must interpret a given class-count column
against one consistent panel denominator across all rows.

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
  `features_by_class` list, yielding a positive denominator.

Do not silently discard a panel class. An unused categorical class is valid and
produces zero per-instance counts, but its authoritative panel feature list
still defines the positive denominator retained for later QC calculations.

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
`control_fraction` output. Category order determines output-column order but
does not affect the calculations.

Allocation resolves panel metadata from every selected `points_name`, verifies
that the stored feature key matches `name_gene_column`, uses the panel's
`feature_class_key` to select the class column, and derives denominators from
the `features_by_class` list lengths. It must not silently estimate denominators
from observed points or accept caller-supplied assay constants.

### Output metrics

In the example above, the output columns are:

```text
n_endogenous_points
n_negative_points
n_system_control_points
control_fraction
```

The class count columns contain assigned point counts. For this three-class
configuration:

```text
control_fraction =
    (n_negative_points + n_system_control_points)
    / (n_endogenous_points + n_negative_points + n_system_control_points)
```

Do not persist `negative_points_per_feature` or
`system_control_points_per_feature` in `.obs`. They are deterministic rescalings
of the raw class counts by the panel denominators and add no independent table
information. Slice 6 QC plotting derives them on demand from the raw count
columns and the table-local denominator snapshot.

### Table-local metadata contract

Record the resolved configuration and generated-column bindings under one
dedicated table-local key. The `control_class_denominators` field below is a
derived snapshot of the panel values used for this table; it is not accepted as
an allocation argument or treated as an authoritative panel definition:

```python
adata.uns["feature_class_allocation"] = {
    "schema_version": 1,
    "source_kind": "harpy_allocate",
    "feature_key": "gene",
    "feature_class_key": "code_class",
    "expression_class": "Endogenous",
    "classes": ["Endogenous", "Negative", "SystemControl"],
    "control_class_denominators": {
        "Negative": 10,
        "SystemControl": 197,
    },
    "count_columns": {
        "Endogenous": "n_endogenous_points",
        "Negative": "n_negative_points",
        "SystemControl": "n_system_control_points",
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
downstream code to reconstruct names. The complete per-class feature lists
remain in SpatialData root metadata and are not duplicated into the table;
only the resolved non-expression denominators needed for later on-demand QC
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
   matrix containing endogenous and control targets;
3. calculate per-instance class counts by summing the temporary matrix columns
   belonging to each class;
4. calculate `control_fraction`;
5. retain only the `expression_class` columns in the final `adata.X`; and
6. attach the class point counts and `control_fraction` to the corresponding
   `.obs` rows.

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
non-positive resolved panel feature counts, a missing expression class, and
collisions with existing output columns must produce clear errors. A panel
control class with no assigned points is valid and must produce a zero count
column. No control class produces a persisted per-feature rate. Validate the
complete multi-region request and shared
`feature_class_allocation` configuration before writing the output table.

### Boundary with Slice 6

The `.obs` summaries describe only control points that land inside an instance
mask. They are suitable for cell-level histograms and violin plots of the raw
class counts and `control_fraction`. Slice 6 may additionally derive the
following per-instance plotting metrics on demand:

```text
negative_points_per_feature =
    n_negative_points / control_class_denominators["Negative"]

system_control_points_per_feature =
    n_system_control_points / control_class_denominators["SystemControl"]
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
`hp.tb.allocate`.

### Verification

Focused tests should establish that:

- the final `.var` and `adata.X` contain only the selected expression class;
- `n_endogenous_points` equals the row sum of the final expression matrix;
- assigned control counts are correct and zero-filled for instances without a
  control call;
- no per-feature rate columns are persisted in `.obs`;
- the table-local denominator snapshot equals the authoritative panel feature
  counts and is sufficient for later QC derivation;
- `control_fraction` is correct and finite;
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
- the versioned `feature_class_allocation` record and its generated-column
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
- one panel-derived denominator mapping applies to every pair;
- an existing output table is replaced only when `overwrite=True`; and
- omitting `expression_class` reproduces the existing allocation result.

Benchmark the class-aware path on a representative backed crop before the full
run. Compare wall time, peak worker memory, task count, and output-table size
with ordinary allocation. The additional summaries should be small relative to
the point-to-label lookup and target-count reduction.

## Slice 6: original-point control QC

**Status: specified; not implemented.**

Add separate lightweight QC functions over the original transcript points.
This slice is implemented after Slice 5 to keep delivery sequential, but its
runtime contract depends only on the points and feature-panel metadata from
Slice 1 and the sample-aware point metadata from Slices 2–4. The original-point
summaries may run before or after allocation and do not depend on an
instance-label raster or an AnnData table. The optional per-instance plotting
view described below requires a Slice 5 table but derives its rates without
modifying that table.

This operation complements the instance-level `.obs` metrics. It must use the
original points element so that controls on label value zero and controls
outside segmented instances remain visible. It must not create another copy of
the points or route individual controls through `hp.tb.allocate`.

### Derived per-instance plotting metrics

When a Slice 5 allocation table is available, QC plotting may derive
`negative_points_per_feature` and `system_control_points_per_feature` from the
persisted raw class counts. Resolve the relevant `.obs` columns through
`adata.uns["feature_class_allocation"]["count_columns"]` and divide them by the
corresponding positive values in `control_class_denominators`:

```text
negative_points_per_feature =
    n_negative_points / control_class_denominators["Negative"]

system_control_points_per_feature =
    n_system_control_points / control_class_denominators["SystemControl"]
```

These rates normalize for the different numbers of panel features in the two
control classes. They should be temporary series or plotting-dataframe columns;
do not persist them back into `adata.obs`. The plotting layer must validate
that the referenced count columns exist and that the stored denominators are
positive. This optional cell-level view complements, but does not replace, the
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

## Slice 7: scalable point-to-label assignment and reduction

**Status: specified; not implemented.**

Refactor the private execution path used by `hp.tb.allocate` without changing
the public or biological contracts established by Slice 5. This optimization
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
existing coordinate-system contract in this slice: points are
identity-transformed in the selected coordinate system and labels may differ by
a pixel-aligned translation. Validate any coordinate-to-pixel rounding and
translation assumptions explicitly rather than relying on integer truncation.

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
specified by Slice 5.

Derive instance coordinates from the compact grouped result instead of running
a second independent groupby over every assigned point. Retain sparse
instance-by-target construction and compute all compact Dask reductions
together so the assignment graph is executed once. Pair-level work remains
independent; the shared feature-axis alignment and final row-wise stacking from
Slice 5 are unchanged.

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
