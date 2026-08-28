# Scalable Baysor integration roadmap

Date: 2026-08-28

## Executive summary

Harpy should integrate the current native C++ Baysor through a separate
`baysor_python` package as a points-first workflow with two execution modes:

1. an untiled mode that establishes correctness and provides a reference
   result; and
2. an optional tiled mode for datasets or deployment environments that cannot
   run Baysor comfortably as one process.

Both modes should share the same input preparation, `baysor_python` segmentation
runner, output contract, provenance model, and SpatialData importer. Tiling should
be an execution strategy within this integration rather than a separate raster
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

The separate `baysor_python` repository will own the pinned Baysor source and
native build. It will expose segmentation as a Python API backed initially by a
managed Baysor subprocess, and expose boundary estimation through a direct native
binding. Harpy will own SpatialData integration, tiling, scheduling,
reconciliation, and global-product construction, but no Baysor-derived C++ code.

## Goals

The integration should:

- run a pinned C++ Baysor implementation through a versioned `baysor_python`
  dependency without building or vendoring Baysor inside Harpy;
- accept a SpatialData points element and an optional labels element as a
  transcript-native prior;
- support large backed SpatialData objects without materializing the complete
  transcript table in pandas;
- preserve one stable identity for every retained transcript;
- produce globally consistent molecule assignments, cell identifiers, shapes,
  counts, and cell statistics;
- reproduce Baysor's boundary semantics through the `baysor_python` native API,
  with explicit upstream provenance and parity tests;
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

## `baysor_python` integration boundary

`baysor_python` will be a separately versioned repository and Python
distribution. It will own:

- the pinned Baysor source and native dependency build;
- discovery or distribution of the matching Baysor executable;
- `segment(...)`, initially implemented as a safe managed subprocess call;
- `boundaries(...)`, implemented as a direct array-oriented native binding;
- translation of native failures into documented Python exceptions and result
  objects; and
- Baysor-version reporting, build provenance, wheels, and CLI-versus-Python
  parity tests.

Harpy will own:

- SpatialData input preparation and output import;
- transcript-stable identity and global prior sampling;
- tile planning, staging, concurrency, retry, and resume;
- cross-tile assignment reconciliation;
- global counts, QC, Shapely/GeoDataFrame conversion, and optional rasterization;
  and
- end-to-end scientific and seam validation.

Calling segmentation through Python does not require running the algorithm in the
Python process. The initial `baysor_python.segment(...)` implementation should
invoke the pinned executable without a shell and return a structured result. This
preserves process isolation, complete memory reclamation, independent OpenMP
configuration, cancellation, retry, and retained logs for each tile.

A direct in-process segmentation API should be added only after Baysor exposes a
stable C++ entry point returning a segmentation result from in-memory molecule
data and options. The Baysor CLI and Python binding must both call that same entry
point; `baysor_python` must not duplicate the CLI's orchestration logic.

During development, the pinned Baysor source may be included as a Git submodule.
Published source distributions must bundle the exact source snapshot and must not
require Git, submodule initialization, or network access during installation.

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

- the `baysor_python` version, embedded Baysor revision, executable path, and
  executable checksum;
- all resolved Baysor and Harpy-side parameters;
- source points, prior labels, and coordinate system;
- global filtering decisions and feature-panel identity;
- tile core and halo bounds;
- per-tile input and output checksums and status;
- per-tile wall time and peak memory;
- reconciliation thresholds and ambiguity counts; and
- final coverage and seam-QC metrics.

## Common untiled foundation

The first implementation should be a thin untiled Harpy wrapper around
`baysor_python.segment(...)`. Harpy should prepare and validate Parquet and pass a
structured request to `baysor_python`; the latter should invoke its pinned
executable using a list of subprocess arguments rather than a shell command and
return paths, logs, resource measurements, resolved native versions, and status
in a structured result. Harpy then validates the output schemas and imports the
result.

At minimum, the public API must expose:

- `scale` and `scale_std`;
- `n_cells_init`;
- `cluster_method` and related cluster parameters;
- `prior_segmentation_confidence`;
- `min_molecules_per_cell`;
- `iters` and `tol`;
- `baysor_python` backend and optional executable override;
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

### How tiling artefacts are avoided

The tiled workflow does **not** treat boundary polygons as the objects that must
be stitched. It treats molecule assignments as the authoritative intermediate
result. Cell identities are reconciled through the stable transcripts shared by
overlapping tiles, and one new global polygon is generated only after that
reconciliation is complete.

Consider one biological cell crossing the boundary between two tile cores. The
halo makes the cell and its surrounding molecule context visible in both Baysor
runs. Baysor may call it `cell_42` in tile A and `cell_17` in tile B, but many of
the same stable transcript IDs in the shared halo will be assigned to both local
cells. Sufficient agreement on those transcripts, supported by assignment
confidence, spatial proximity, and the prior nucleus, establishes that the two
local identifiers represent one global cell:

```text
tile_A/cell_42 ---\
                   +--- global_cell_9001
tile_B/cell_17 ---/
```

After all compatible local identifiers have been mapped to global cells, Harpy
selects one final assignment for every transcript. It then gathers the complete
molecule cloud of `global_cell_9001` from both sides of the seam and estimates a
single new boundary from that global cloud.

The intended flow is therefore:

```text
overlapping tile context
          |
          v
match local cells using shared transcript identities
          |
          v
select one global cell assignment per transcript
          |
          v
estimate one new global polygon per cell
```

Tile-local polygons may be retained for diagnostics, but they are not clipped,
glued together, or unioned to form the final boundary. Direct polygon stitching
can preserve straight cuts at tile edges, introduce gaps or overlaps, and create
a shape that disagrees with the final molecule assignments. Re-estimating the
boundary after transcript reconciliation removes the tile seam from the geometry
construction altogether.

The halo is the first defence against artefacts because it prevents a core seam
from also being a Baysor context boundary. Shared-transcript matching is the
second defence because it resolves duplicate local identities. Global boundary
re-estimation is the third defence because it prevents tile edges from appearing
in the final cell geometry.

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
when its input checksum, `baysor_python` version, embedded Baysor/executable
identity, and resolved parameters all match.

### 5. Run Baysor through `baysor_python` with resource-aware concurrency

Harpy calls `baysor_python.segment(...)` for every tile; the initial backend runs
each tile as an independent Baysor subprocess. The Harpy scheduler must limit
both memory and CPU oversubscription. OpenMP thread counts should be set
explicitly for each process, and tile concurrency should be derived from a memory
budget.

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

For example, if tile A splits a region into `cell_A1` and `cell_A2` while tile B
calls the same region `cell_B1`, the reconciler must not merge all three cells:
doing so would merge two cells that one Baysor run explicitly kept separate. It
should accept at most the strongest compatible match and flag the remaining
conflict. An ambiguous or halo-touching group can be rerun as a rescue region
centred on the seam with a larger halo, or on a grid shifted so that the seam is
inside a tile core. If the conflict remains, retaining a conservative separation
with an explicit QC flag is preferable to an unsupported merge.

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

The selected boundary path is to estimate each final cell polygon once from the
globally reconciled molecule assignments. Thus, a cell crossing a seam receives
one boundary calculation over the molecules on both sides, rather than two
polygon fragments followed by a geometric merge.

"Global molecule cloud" does not mean that the estimator may see only the
molecules assigned to the target cell. Baysor's boundary algorithm also uses
nearby molecules assigned to other cells or to noise to reject admixture around
the Delaunay boundary. An exact whole-dataset call therefore receives the complete
reconciled coordinates and cell labels. A bounded-memory batched call may receive
only a subset of target cells, but it must also receive every contextual molecule
intersecting those cells' required bounding regions. The global boundary-distance
parameter must be computed once or passed unchanged to every batch.

### `baysor_python` boundary-estimator decision

Neither Harpy nor `baysor_python` will reimplement the estimator in Python, and
Harpy will not merge tile-local polygons. The separate `baysor_python` repository
will build the pinned Baysor source and expose its boundary estimator through a
thin `pybind11` or `nanobind` extension. Harpy will depend on this public Python
API rather than owning Baysor-derived C++ code.

The native extension boundary must remain array-oriented and independent of
SpatialData, GeoPandas, Shapely, Arrow, and Parquet. Its conceptual contract is:

- input: contiguous molecule coordinates, final global cell labels, optional
  target cell identifiers, and an optional precomputed global boundary-distance
  parameter;
- output: packed polygon vertices, polygon offsets, and the corresponding global
  cell identifiers; and
- execution: release the Python GIL, support sparse global cell identifiers, and
  handle empty, one-molecule, two-molecule, and collinear cells explicitly.

A Harpy wrapper will prepare the arrays, call `baysor_python.boundaries(...)`,
convert the packed result to Shapely polygons and a GeoDataFrame, and add the
result through Harpy's normal shapes API.

`baysor_python` should be built with CMake and `scikit-build-core`, and prebuilt
wheels should initially cover macOS ARM64 and Linux x86-64 for Harpy's supported
Python versions. Its releases must include Baysor's MIT notice, the exact
upstream commit and original source paths, an inventory of binding-related Baysor
changes, and an update procedure. The Python package must report both its own
version and the embedded Baysor revision.

Parity tests must compare the native API geometry with the pinned Baysor CLI
output after normalizing irrelevant polygon orientation and starting-vertex
differences. Repeated-call tests must also check memory release, exception
translation, and OpenMP behavior.

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

### Stable native library entry points

The initial `baysor_python` segmentation API is intentionally backed by the CLI.
A future in-process API requires Baysor to expose one stable C++ segmentation
entry point accepting in-memory molecule data and options and returning a
structured result. The CLI and Python binding must share this entry point so
their parameter resolution and scientific behavior cannot drift.

The boundary API may initially bind the existing boundary functions, but
bounded-memory use requires explicit target-cell selection while retaining all
required contextual molecules. This capability should preferably be implemented
in Baysor's public C++ library rather than as a divergent algorithm in the
binding layer.

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

### Phase 1: Establish `baysor_python` segmentation and the untiled integration

Build the shared package and Harpy foundation before any tile orchestration.

Deliverables:

- a separately versioned `baysor_python` repository and distribution;
- pinned Baysor source, initially usable as a development submodule and bundled
  into release source distributions;
- `baysor_python.segment(...)` with managed subprocess execution, structured
  results, documented exceptions, version reporting, and executable preflight;
- parameter validation shared without duplicating Baysor's scientific logic;
- global prior sampling and Parquet preparation;
- output-schema validation;
- points, shapes, table, and optional-label import;
- provenance and resource recording; and
- focused tests using a fake executable, CLI-versus-Python parity tests, and a
  small Harpy integration test.

Exit criterion: `baysor_python.segment(...)` reproduces a direct pinned-Baysor CLI
run, and Harpy reproduces that result without using the Julia-era raster callback.

### Phase 2: Establish the `baysor_python` native boundary API

Resolve the highest-risk native packaging and semantic-parity questions before
building the tiled workflow around this component.

Deliverables:

- an array-oriented binding to the pinned Baysor C++ boundary implementation;
- `baysor_python.boundaries(...)` and a thin Harpy conversion wrapper;
- explicit full-context and batched-target input contracts;
- packed polygon output and conversion to Harpy shapes;
- focused edge-case, repeated-call, and CLI-parity tests;
- macOS ARM64 and Linux x86-64 build and wheel smoke tests; and
- dependency, licence, provenance, source-bundling, and upstream-update
  documentation.

Exit criterion: for identical molecule coordinates, assignments, and parameters,
the Python API produces geometry equivalent to pinned Baysor and its wheel can be
installed on both initially supported platforms.

### Phase 3: Implement tile planning and staging

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

### Phase 4: Implement resource-aware tile execution

Deliverables:

- bounded scheduling of `baysor_python.segment(...)` subprocess jobs;
- explicit OpenMP configuration;
- explicit per-tile `n_cells_init` calculation;
- retained logs and resource traces;
- retry/resume behavior; and
- validation that every expected tile completed with compatible schemas and
  parameters.

Exit criterion: a complete set of reproducible tile-local molecule assignments
can be generated without exceeding the configured memory budget.

### Phase 5: Implement reconciliation

Deliverables:

- overlap joins by transcript ID;
- candidate cell-match metrics;
- constrained union-find;
- ambiguity reporting;
- final per-transcript assignment selection; and
- deterministic global cell relabeling.

Exit criterion: one final row exists per retained transcript, reconciliation is
order-independent, and all graph invariants pass.

### Phase 6: Build and import global products

Deliverables:

- global sparse cell-by-gene aggregation;
- cell statistics and nucleus-ownership QC;
- global boundary generation through `baysor_python.boundaries(...)`, with
  complete contextual molecules for each exact or batched call;
- SpatialData points, shapes, and table elements;
- optional rasterization; and
- complete provenance metadata.

Exit criterion: assignments, shapes, table instances, and optional labels are
mutually consistent and survive a SpatialData write/read round trip.

### Phase 7: Validate tiled quality and choose defaults

Deliverables:

- tiled-versus-untiled comparison on the actual UCB mosaic;
- halo-size comparison;
- tile-size and concurrency benchmarks;
- a half-tile grid-shift experiment;
- seam-specific QC plots; and
- documented initial defaults and failure thresholds.

Exit criterion: tiled results meet the agreed quality gates and do not show
material seam or grid-placement dependence.

### Phase 8: Distributed hardening

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

At the `baysor_python` layer, verify that `segment(...)` matches the direct pinned
CLI for identical inputs, parameters, and seed, and that `boundaries(...)` matches
the CLI-generated geometry for identical assignments. Native tests must cover
repeated calls, exception translation, sparse cell identifiers, noise, degenerate
cells, and supported wheel installation.

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

### `baysor_python` and Baysor version drift

A binding release can silently diverge from its executable if source revisions,
parameter defaults, or orchestration are not coupled. Every release must pin and
report one Baysor revision, test Python-versus-CLI parity, and reject incompatible
executable overrides. In-process segmentation must not be introduced until the
CLI and binding share the same C++ library entry point.

## Decision

Proceed in the following order:

1. validate current C++ Baysor untiled on the actual UCB sample;
2. create `baysor_python`, pin and package Baysor, and expose segmentation through
   a managed-subprocess Python API;
3. integrate that segmentation API into Harpy's modern points-first untiled
   workflow;
4. expose Baysor boundary estimation as an array-oriented native
   `baysor_python` API and verify CLI parity;
5. add optional core-plus-halo tiling around the shared foundation;
6. reconcile cells using shared transcript identities and constrained graph
   matching;
7. rebuild all global derived products from the reconciled molecule table, using
   `baysor_python` for final boundary estimation; and
8. promote tiled mode only after it matches the untiled reference and passes
   seam and grid-shift validation.

Tiling is therefore a planned scalability capability, not the default for the
current 47-million-transcript sample. The architecture must nevertheless make
it possible to scale beyond a single machine without changing the authoritative
data model or scientific interpretation of the result.
