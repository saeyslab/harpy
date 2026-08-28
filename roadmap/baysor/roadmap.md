# Scalable Baysor integration roadmap

Date: 2026-08-28

## Executive summary

Harpy should integrate the current native C++ Baysor as a points-first CLI
workflow with two execution modes:

1. an untiled mode that establishes correctness and provides a reference
   result; and
2. an optional tiled mode for datasets or deployment environments that cannot
   run Baysor comfortably as one process.

Both modes should share the same input preparation, subprocess runner, output
contract, provenance model, and SpatialData importer. Tiling should be an
execution strategy within this integration rather than a separate raster
segmentation implementation.

The benchmark in [benchmark.md](benchmark.md) showed that an approximately
47-million-transcript synthetic CosMx mosaic completed a one-iteration resource
run with 19.81 GB peak resident memory. This does not establish segmentation
quality or the runtime of a converged run, but it demonstrates that the UCB-sized
sample does not require tiling solely to fit on the available 32 GB machine.
Consequently, the first production-quality experiment should be untiled on the
actual sample. The untiled result is also needed as the reference against which
the tiled implementation will be validated.

The tiled design must reconcile molecule-to-cell assignments using stable
transcript identities in overlapping tiles. It should not rasterize every tile
and rely primarily on pixel-IoU stitching. Baysor's molecule assignments are the
authoritative result; cell polygons, count matrices, cell statistics, and
optional raster labels should be derived globally after reconciliation.

## Goals

The integration should:

- run a pinned C++ Baysor executable without installing Baysor into Harpy's
  Python environment;
- accept a SpatialData points element and an optional labels element as a
  transcript-native prior;
- support large backed SpatialData objects without materializing the complete
  transcript table in pandas;
- preserve one stable identity for every retained transcript;
- produce globally consistent molecule assignments, cell identifiers, shapes,
  counts, and cell statistics;
- expose all scale-sensitive and resource-sensitive Baysor parameters;
- make runs reproducible, inspectable, resumable, and safe to retry; and
- demonstrate that tiled results are not materially dependent on tile seams or
  grid placement.

The initial implementation is 2D. Arbitrary affine transformations, 3D Baysor,
and automatic distributed deployment are outside the first milestone.

## Why this should be a new points-first integration

Harpy's existing `baysor_callable` targets the older Julia CLI contract. It
writes per-chunk CSV and TIFF inputs, sets `JULIA_NUM_THREADS`, reads GeoJSON,
and immediately rasterizes the polygons. The generic `segment_points` path then
reconciles independently generated raster labels across Dask chunks.

That design is not a good fit for current Baysor C++:

- Baysor already performs its expensive work in native C++ and exposes a CLI;
- it accepts and emits Parquet;
- its main result is a molecule-to-cell assignment with confidence and noise
  information;
- its Parquet bundle includes cell statistics and GeoParquet boundaries; and
- converting each local result to a raster before stitching loses the strongest
  cross-tile evidence: the assignments of the same transcripts in both tiles.

A dedicated high-level points operation, conceptually
`harpy.pt.segment_baysor`, should therefore own the complete workflow. The exact
public name can be settled during API review. The existing Julia-era callable
can be deprecated independently after the new integration is established.

## Output contract

A completed run should add coordinated elements to SpatialData.

### Molecule assignments

The authoritative result is a points element containing the original retained
transcript columns plus at least:

- a stable transcript identifier;
- the final global cell identifier;
- `is_noise`;
- Baysor's molecule confidence, when available;
- assignment confidence, when available;
- the sampled global prior label; and
- optional globally meaningful molecule-cluster information.

There must be exactly one final row per retained input transcript. Original
transcript attributes and feature-panel metadata should be preserved.

### Cell shapes

A shapes element should contain one final polygon per retained global cell. Its
index must use the same instance identifiers as the table. Tile-local polygons
are intermediate data and must not be exposed as the final segmentation.

### Cell table

An AnnData table should contain:

- the global cell-by-gene count matrix;
- cell centroids;
- transcript counts;
- noise and assignment-confidence summaries where meaningful;
- dominant prior nucleus and nucleus-ownership QC fields;
- area and shape statistics; and
- SpatialData region and instance annotations linking it to the shapes element.

Counts must be rebuilt from the final molecule assignments. Tile-level feature
matrices cannot be concatenated because halo transcripts are duplicated and
tile-local cell identifiers are not globally meaningful.

### Optional labels raster

Raster labels may be generated from the final global shapes when explicitly
requested. Shapes and molecule assignments should remain the primary result;
the workflow must not require a full-mosaic raster merely to represent Baysor's
output.

### Provenance

Harpy metadata should record:

- the Baysor executable path, revision or executable checksum;
- all resolved Baysor and Harpy-side parameters;
- source points, prior labels, and coordinate system;
- global filtering decisions and feature-panel identity;
- tile core and halo bounds;
- per-tile input and output checksums and status;
- per-tile wall time and peak memory;
- reconciliation thresholds and ambiguity counts; and
- final coverage and seam-QC metrics.

## Common untiled foundation

The first implementation should be a thin untiled wrapper around pinned Baysor
C++. It should prepare and validate Parquet, invoke the executable using a list
of subprocess arguments rather than a shell command, capture logs and resource
measurements, validate the output schemas, and import the result.

At minimum, the public API must expose:

- `scale` and `scale_std`;
- `n_cells_init`;
- `cluster_method` and related cluster parameters;
- `prior_segmentation_confidence`;
- `min_molecules_per_cell`;
- `iters` and `tol`;
- executable path;
- OpenMP thread count;
- work directory and intermediate-retention policy; and
- overwrite and resume behavior.

The integration must always set an explicit initial cell count for large
nuclei-prior runs. Automatic initialization produced approximately 1.43 million
components in the benchmark and led to pathologically large polygon/output
work. For the UCB sample, the first range remains 40,000 to 60,000 initial
components.

The initial actual-sample experiment should use:

- the Cellpose 4 nuclei sampled into a transcript column such as `prior_cell`;
- `cluster_method=none`;
- sample-specific `scale` estimated from the nuclei or cell labels;
- `n_cells_init` values of 40,000, 50,000, and 60,000;
- prior confidence 0.5, with 0.2 as a sensitivity check;
- a 100-iteration run before committing to 500 iterations; and
- `tol=0.005` unless convergence diagnostics justify another value.

## Tiled architecture

### 1. Global preflight and filtering

Input preparation must happen globally before tiling:

- require backed SpatialData for large runs;
- validate finite coordinates and required columns;
- establish or create a stable signed 64-bit transcript identifier;
- validate uniqueness of that identifier;
- resolve the target coordinate system;
- initially support only identity/translation relationships between points and
  prior labels;
- apply QV, feature-class, excluded-gene, and rare-gene filtering once;
- establish one global feature panel and gene encoding; and
- estimate one global `scale` that is reused by every tile.

Per-tile gene filtering is not allowed because it would give different gene
models to different tiles. Per-tile scale estimation is likewise not allowed.

### 2. Sample the prior at transcript positions

The nuclei label must be sampled at each transcript coordinate to create a
transcript-native prior column. Transcripts outside a nucleus must remain in the
table with prior label `0`; they must not be discarded.

The existing blockwise point-to-label aggregation code can provide useful
building blocks, but the Baysor preparation path needs a variant that retains
background rows and preserves all required transcript attributes.

If global label identifiers are very large or sparse, each tile may compact its
prior labels and keep a sidecar mapping from tile-local prior labels to global
labels. This avoids allocations based on a large maximum label while preserving
global nucleus identity for reconciliation.

### 3. Plan tile cores and halos

Tile cores form a disjoint half-open partition of the data extent. Every
transcript therefore has exactly one core owner. Each core is expanded by a halo
before Baysor is run so cells and molecule neighborhoods near a core edge have
context from the adjacent tile.

For the UCB dimensions, a useful first configuration is:

- core size: 10,000 by 10,000 coordinate units;
- halo: `4 * scale`, or 212 units when `scale=53`;
- six columns by three rows, for 18 tiles;
- approximately 4.42 million transcripts in a full expanded tile at the average
  dataset density; and
- approximately 1.081 times the original transcript volume after halo
  duplication.

The halo multiplier is an experimental parameter rather than a default. Values
of `2 * scale`, `4 * scale`, and `6 * scale` should be compared.

Tile planning should use a configurable maximum transcript budget. A coarse
spatial count pass should estimate expanded-tile counts before writing inputs.
Tiles exceeding the budget should be recursively split. The planner must handle
irregular edge tiles and T-junctions without changing core ownership semantics.

### 4. Stage tile inputs in one pass

Harpy should not run Baysor's crop flags repeatedly against the full source
Parquet file, because each run would still need to scan the source. Instead, a
single distributed pass should route transcripts to all expanded tiles that
contain them.

The staging pass should:

- retain global coordinates;
- include only the columns Baysor and reconciliation need;
- duplicate halo transcripts while retaining their stable identifier;
- attach the sampled prior and any approved global filter/cluster columns;
- write one Parquet input per expanded tile; and
- write a manifest containing counts, bounds, checksums, local prior mappings,
  resolved parameters, and run status.

The manifest makes the workflow resumable. A completed tile may be reused only
when its input checksum, executable identity, and resolved parameters all match.

### 5. Run Baysor with resource-aware concurrency

Every tile is an independent Baysor subprocess. The scheduler must limit both
memory and CPU oversubscription. OpenMP thread counts should be set explicitly
for each process, and tile concurrency should be derived from a memory budget.

On the 32 GB benchmark machine, the initial comparison should be:

- one Baysor process using eight OpenMP threads; versus
- two Baysor processes using four OpenMP threads each.

Each tile must use:

- the same fixed global scale and scale standard deviation;
- globally consistent filtering;
- `cluster_method=none` for the first implementation;
- `--skip-ncv-color`;
- the same iteration and convergence parameters; and
- an explicit tile-level initial component count.

A practical tile-level initial count is approximately two to three times the
number of active prior nuclei in the expanded tile, with a global cell-density
fallback for tissue without prior nuclei. Automatic initialization must not be
used.

Tile stdout, stderr, resolved parameters, wall time, peak memory, and return code
must be retained. A failed tile should not invalidate successful tiles, but the
workflow must not reconcile a partial tile set.

### 6. Reconcile local cells through shared transcripts

The same stable transcripts occur in both runs on either side of an internal
seam. They provide direct evidence for whether two tile-local cell identifiers
represent the same biological cell.

For each pair of spatially adjacent tiles:

1. Join the two molecule outputs by stable transcript identifier within their
   shared halo.
2. For every pair of non-noise local cell identifiers, count transcripts assigned
   to both cells.
3. Compute evidence including shared-transcript count, overlap coefficient,
   Jaccard similarity, mutual-best status, centroid distance, assignment
   confidence, and agreement on the dominant global prior nucleus.
4. Generate candidate match edges only when minimum evidence criteria are met.
5. Process candidate edges from strongest to weakest with constrained
   union-find.

The union operation must maintain the invariant that a global component contains
at most one local cell from any individual tile. This prevents a chain of
pairwise matches from merging two cells that the same Baysor run considered
distinct.

Many-to-one matches, conflicting nucleus evidence, low-overlap matches, and
components that would violate the one-cell-per-tile invariant are ambiguous.
They should be recorded for QC and resolved conservatively rather than merged
silently.

Thresholds should be learned from tiled-versus-untiled comparisons. They must
not be chosen solely from the synthetic resource benchmark.

### 7. Select one final assignment per transcript

After local cell identifiers have been mapped to global components, a transcript
may still have several tile predictions. The preferred prediction should come
from the tile in which that transcript is farthest from the outer boundary of
the expanded tile, because this prediction had the most spatial context.
Assignment confidence can be used as a secondary criterion. The unique core
owner provides the deterministic fallback.

Noise is a valid competing assignment and should not be overwritten merely
because another tile assigned the transcript to a low-confidence cell.

The result of this stage is a single partitioned molecule table with exactly one
row per retained transcript and one globally unique cell identifier or noise.

### 8. Build global cell products

All final products must be rebuilt from the reconciled molecule table:

- cell-by-gene counts;
- centroids and transcript counts;
- confidence and nucleus-ownership summaries;
- cell-level QC fields; and
- cell boundaries.

Tile count matrices must be ignored. Tile polygons must not simply be clipped
and unioned because doing so can create visible seams and can disagree with the
final transcript assignments.

The preferred boundary path is to estimate each final cell polygon from its
global assigned molecule cloud. This is naturally parallel per cell and keeps
memory bounded. If exact reuse of Baysor's C++ boundary estimator is required,
a standalone upstream boundary operation would be preferable to duplicating the
algorithm indefinitely in Harpy.

## Baysor upstream requirements and optimizations

### Stable transcript identity in Parquet output

This is the most important correctness requirement. At revision
`d7077a7ded6f4b941915badc894f767532d39fd2`, Baysor reads numeric
`transcript_id`, and legacy CSV can write it, but `molecules.parquet` does not
include it. Tiled reconciliation should not depend permanently on undocumented
row-order preservation.

The preferred solution is an upstream Baysor change that round-trips
`transcript_id` in Parquet output. Until that exists, a pinned adapter may attach
input IDs to output rows only after strict validation of row count and exact
gene/x/y equality. Such a fallback is temporary and must fail closed when the
validation does not hold.

### Selective outputs

Tile-level count matrices, cell statistics, and polygons are not authoritative,
yet Parquet output currently generates them. Upstream switches to skip these
products would reduce tile runtime, memory, and disk use. A molecules-only mode
is desirable.

### Boundary-only operation

A C++ subcommand that consumes final molecule assignments and writes cell
boundaries would allow Harpy to use Baysor's own boundary semantics after
reconciliation without rerunning segmentation.

### Optional precomputed confidence

Noise estimation was a major phase in the exact-size benchmark. An option to
accept and retain a precomputed molecule-confidence column could support a
future globally calibrated confidence model and avoid repeated fitting in
overlapping tiles. This is an optimization, not a requirement for the first
tiled prototype.

## Implementation sequence

### Phase 0: Establish the actual-data reference

Run the actual UCB mosaic untiled with the Cellpose nuclei prior and explicit
initial cell counts. Compare `cluster_method=none` with Louvain on representative
crops before deciding whether clustering materially improves segmentation.

Deliverables:

- selected baseline parameters;
- an untiled 100-iteration full-mosaic result;
- resource measurements;
- visual overlays and biological QC; and
- a frozen reference molecule-assignment dataset for tiled comparisons.

Exit criterion: a scientifically plausible untiled result and a parameter set
worth reproducing in tiled mode.

### Phase 1: Implement the modern untiled integration

Build the shared foundation before any tile orchestration.

Deliverables:

- parameter validation and executable preflight;
- global prior sampling and Parquet preparation;
- safe C++ CLI subprocess execution;
- output-schema validation;
- points, shapes, table, and optional-label import;
- provenance and resource recording; and
- focused tests using a fake executable plus a small optional Baysor integration
  test.

Exit criterion: Harpy reproduces a direct pinned-Baysor run without using the
Julia-era raster callback.

### Phase 2: Implement tile planning and staging

Deliverables:

- stable transcript-ID creation and validation;
- half-open core and halo planner;
- density/count preflight and adaptive tile splitting;
- single-pass routing to tile Parquet files;
- local-to-global prior-label mappings;
- checksummed run manifest; and
- resume and stale-output detection.

Exit criterion: every input transcript has exactly one core owner and the
expected halo memberships, with no uncovered or unexpectedly duplicated rows.

### Phase 3: Implement resource-aware tile execution

Deliverables:

- bounded subprocess scheduling;
- explicit OpenMP configuration;
- explicit per-tile `n_cells_init` calculation;
- retained logs and resource traces;
- retry/resume behavior; and
- validation that every expected tile completed with compatible schemas and
  parameters.

Exit criterion: a complete set of reproducible tile-local molecule assignments
can be generated without exceeding the configured memory budget.

### Phase 4: Implement reconciliation

Deliverables:

- overlap joins by transcript ID;
- candidate cell-match metrics;
- constrained union-find;
- ambiguity reporting;
- final per-transcript assignment selection; and
- deterministic global cell relabeling.

Exit criterion: one final row exists per retained transcript, reconciliation is
order-independent, and all graph invariants pass.

### Phase 5: Build and import global products

Deliverables:

- global sparse cell-by-gene aggregation;
- cell statistics and nucleus-ownership QC;
- global boundary generation;
- SpatialData points, shapes, and table elements;
- optional rasterization; and
- complete provenance metadata.

Exit criterion: assignments, shapes, table instances, and optional labels are
mutually consistent and survive a SpatialData write/read round trip.

### Phase 6: Validate tiled quality and choose defaults

Deliverables:

- tiled-versus-untiled comparison on the actual UCB mosaic;
- halo-size comparison;
- tile-size and concurrency benchmarks;
- a half-tile grid-shift experiment;
- seam-specific QC plots; and
- documented initial defaults and failure thresholds.

Exit criterion: tiled results meet the agreed quality gates and do not show
material seam or grid-placement dependence.

### Phase 7: Distributed hardening

Only after the local tiled workflow is correct:

- add scheduler resource annotations;
- support shared worker-local staging and remote-backed inputs where practical;
- improve adaptive splitting for heterogeneous density;
- define intermediate cleanup and retention policies; and
- test worker loss, partial retries, and manifest recovery.

Exit criterion: distributed execution changes throughput and capacity without
changing the scientific result or reconciliation semantics.

## Test strategy

### Focused unit tests

- tile cores cover the domain exactly once;
- half-open bounds behave correctly on seams and dataset edges;
- halo membership is correct at faces, corners, and T-junctions;
- stable IDs remain unique after routing;
- tile initial-cell estimates are explicit and bounded;
- candidate scores are invariant to input ordering;
- constrained union-find rejects two cells from the same tile in one component;
- ambiguous split, merge, and noise cases remain flagged;
- final transcript selection is deterministic; and
- manifests reject parameter, input, or executable mismatches.

### Small integration tests

Use a tiny synthetic dataset with known cells crossing tile seams. Run both
untiled and tiled Baysor and verify coverage, matching, counts, shapes, and
round-trip SpatialData integrity. The test should include duplicate coordinates,
background transcripts, cells without a prior nucleus, and nuclei close to a
tile boundary.

### Actual-data validation

Compare the tiled result to the untiled UCB reference using:

- retained and assigned transcript counts;
- noise fraction;
- molecule-assignment agreement after cell matching;
- per-cell transcript-set Jaccard similarity;
- total and substantial cell counts;
- molecules per cell and cell-by-gene count correlation;
- one-nucleus-per-cell ownership;
- split nuclei and multi-nucleus cells;
- cell area and shape distributions;
- known-marker coherence;
- disagreement as a function of distance to the nearest seam; and
- visual overlays at seams and in representative tissue regions.

Repeat the tiled run after shifting the grid by half a core in x and y. A
scientifically reliable tiled implementation should be nearly invariant to this
change.

## Initial acceptance gates

The exact biological thresholds should be finalized after the untiled
actual-data run, but initial engineering gates are:

- 100% of retained transcript IDs have exactly one final output row;
- no unknown, duplicated, or uncovered transcript IDs;
- no global cell contains two local cells from the same tile;
- all shapes and table instances refer to existing global cell IDs;
- matched-cell count profiles correlate with the untiled reference at greater
  than 0.99;
- nucleus-ownership metrics do not materially degrade relative to untiled;
- seam-proximal disagreement is not materially worse than interior
  disagreement; and
- shifting the tile grid does not materially change the result.

Quality gates should fail the run or mark it experimental; they should not be
reduced to warnings hidden in logs.

## Principal risks

### Loss of global statistical context

Noise estimation and optional molecule clustering operate independently per
tile. Halos reduce boundary effects but do not make these models global. The
first tiled implementation therefore uses `cluster_method=none` and validates
tile-local noise behavior explicitly. Global or consistently transferred
clustering can be investigated later if crop experiments show a meaningful
benefit.

### Incorrect cell merging

Aggressive transitive reconciliation can merge neighboring cells. Stable shared
transcript evidence, the one-cell-per-tile component invariant, prior-nucleus
evidence, and explicit ambiguity reporting are required safeguards.

### Insufficient halo

Cells or relevant molecule neighborhoods larger than the halo can remain
truncated. Halo-size experiments, boundary-touch flags, and grid-shift
validation are required before selecting a default.

### Excessive intermediate data

Halo duplication is modest for the proposed UCB grid, but tile inputs and full
tile output bundles can still consume substantial disk. Checksummed resume,
selective upstream output, a documented retention policy, and capacity
preflight are required.

### Undocumented Baysor output assumptions

Row-order preservation is visible in the pinned implementation but is not an
adequate long-term identity contract. Parquet transcript-ID round-tripping is a
production requirement.

## Decision

Proceed in the following order:

1. validate current C++ Baysor untiled on the actual UCB sample;
2. implement a modern points-first untiled Harpy integration;
3. add optional core-plus-halo tiling around that shared foundation;
4. reconcile cells using shared transcript identities and constrained graph
   matching;
5. rebuild all global derived products from the reconciled molecule table; and
6. promote tiled mode only after it matches the untiled reference and passes
   seam and grid-shift validation.

Tiling is therefore a planned scalability capability, not the default for the
current 47-million-transcript sample. The architecture must nevertheless make
it possible to scale beyond a single machine without changing the authoritative
data model or scientific interpretation of the result.
