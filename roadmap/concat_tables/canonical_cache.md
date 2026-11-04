# Canonical cache: refactor

Status: decided on 2026-11-01. Its own PR, before `hp.tb.concat`
([concat_tables.md](concat_tables.md)). Closes #279 and #282.

## Why

- **#279:** a table subset after its canonical centers were computed fails
  `hp.tb.validate_table`, because the record binds each region to its row
  count and instance digest. Per-cell QC followed by `hp.tb.concat` hits this.
- **#282:** `hp.tb.add_canonical_centers` requires SpatialData backed by a
  Zarr store. Spatiato's Spatial Query also runs on SpatialData without one.
- **Spatiato stops computing canonical centers** (P8 of its harpy-bump
  roadmap). It keeps no cache logic of its own and uses only:
  - `validate_canonical_centers(...)`, for the status and before every run;
  - `hp.tb.add_canonical_centers(sdata, table_name=..., overwrite=True)`, when
    the centers are absent or invalid;
  - the constants (`CANONICAL_OBSM_KEY`, ...).

  It finds the rows of the selected labels element from the table's
  annotation, as its Object Classification does.

  Spatiato drops its regional cache: computing one labels element at a time,
  with NaN in the rows of the others. Nothing creates partial caches any more,
  so Harpy drops the per-region cache machinery, which no Harpy code uses
  either.

- `hp.tb.concat` needs the canonical checks that need no `sdata`.

## Decisions

1. **All or nothing.** A table has canonical centers for every labels element
   it annotates, with every center finite, or has none. No regional updates:
   `add_canonical_centers(..., region=...)` was considered for Spatiato and
   dropped.
2. **Its own PR, before `hp.tb.concat`.** Spatiato's sub-step 3c.2 waits on
   #282, and Spatiato develops against Harpy releases.
3. **Removed outright, without deprecation:** the cache API and
   `build_instance_set_digest`. Canonical centers came with 0.5.0, and their
   only known consumer, Spatiato, moves at the same time. This departs from
   #279, point 4, which deprecates `build_instance_set_digest`. #282's
   digest-domain rename falls away with the digest.
4. **No backward compatibility: only version 2 is read and written.** A
   version-1 record raises, and the message says to rebuild with
   `hp.tb.add_canonical_centers(sdata, table_name=..., overwrite=True)`, which
   checks only that the components are present, not that they are valid. Every
   table `aggregate_points` wrote in 0.5.x, and every record Spatiato wrote, is
   rebuilt once; `concat` raises on version-1 inputs. The version still goes
   to 2, so that a version-1 record fails on its version, with that hint, and
   older Harpy rejects version-2 records clearly.
5. **`add_canonical_centers` installs through
   `hp.tb.io.add_table_components`.** That covers unbacked SpatialData (#282)
   and replaces the canonical-specific staging and publication. Two changes
   follow:
   - for backed SpatialData, the matrix is reattached lazily, as a Dask array,
     as `add_feature_matrix` does, not as a read-only `zarr.Array`;
   - the new components are not validated before they are installed, neither
     in memory nor as a staged, serialized copy: everything the validator
     checks holds by construction, and the tests run `validate_table` after
     `add_canonical_centers`.
6. **The annotation is read from the table, not passed.** `labels_name` is
   removed from `add_canonical_centers`: it could only restate every region
   the table annotates. `validate_canonical_centers(sdata, *, table_name)`
   takes the table by name, as Harpy's functions that need other SpatialData
   elements do, and reads the region and instance keys and the declared
   regions from the table's SpatialData annotation, so a caller cannot pass an
   annotation that disagrees with the table. Spatiato develops against this PR
   and calls the new API.
7. **Breaking changes are accepted, also between steps.** A step does not keep
   alive what a later step removes: it lists what it leaves broken, and the
   first test run is after step 5.

## Contract after the refactor

- **Matrix:** `obsm["spatial_canonical"]`, dense `float64` of shape
  `(n_obs, 3)`, in `(z, y, x)` order, every value finite; `z = 0` in the rows
  of 2D labels.
- **Record:** `uns["spatial_coordinates"]["spatial_canonical"]`, schema
  version 2:

  ```python
  {
      "schema_version": 2,
      "axes": ["z", "y", "x"],
      "regions": {
          "<labels element>": {
              "source_element_type": "labels",
              "source_scale": "scale0",
              "coordinate_frame": {
                  "type": "element_intrinsic",
                  "pixel_coordinate_convention": "integer_indices_are_pixel_centers",
              },
              "algorithm_version": 1,
              "source": {"dims": ["y", "x"], "shape": [4000, 4000], "dtype": "uint32"},
          },
      },
  }
  ```

  The record keeps what gives the numbers their meaning and nothing else in
  the table states: the axes and the frame, the algorithm version and the
  labels source. Every field but `algorithm_version` and `source` is a
  constant that the parser requires, so a change of meaning needs a new
  schema version. Compared with version 1, it drops:
  - `coverage` (#279);
  - what repeats the annotation (`region_key`, `instance_key`), the registry
    key (`obsm_key`), the matrix (`dtype`), the region key (`source_element`,
    `coordinate_frame.element`, `source.element_path`) or the top-level `axes`
    (`coordinate_frame.axes`);
  - the `calculation` block. Its `method`, `weighting` and
    `background_value` never vary: algorithm version 1, which
    `docs/canonical_centers.md` defines, stands for them. Unlike the
    parameters scanpy and squidpy record in `uns[...]["params"]`, they are no
    choices. Its `implementation` was a code path: moving the class would have
    invalidated every record. `algorithm_version` moves up to the region
    entry, and `pixel_coordinate_convention` to `coordinate_frame`;
  - `generated_by`, which Harpy never wrote.

  The frame stays explicit, as spatialdata keeps frames explicit on its
  elements. `pixel_coordinate_convention` belongs to it: Harpy places pixel
  centers at integer indices, while spatialdata places them at half-integers
  (`get_centroids` gives `(0.5, 0.5)` for a label on pixel `(0, 0)`, Harpy
  `(0, 0)`), so mapping canonical centers through a labels element's
  transformation needs `+ 0.5` first.

- **Validation** (`validate_canonical_centers(sdata, *, table_name)`):
  - both components, or neither;
  - with components, a SpatialData annotation, `uns["spatialdata_attrs"]`,
    which gives the region and instance keys and the declared regions;
  - the matrix: dense, shape, dtype, finite values;
  - the record parses strictly, in schema version 2, with an entry for every
    declared region. Entries of regions the table no longer declares, as after
    subsetting it to some of its regions, are ignored and left out of the
    returned metadata;
  - per declared region: the source signature equals the labels element's
    (the only check that needs `sdata`), the algorithm version is supported,
    and `z = 0` for 2D labels.

  The source signature detects a labels element replaced by one of another
  shape, dimensions or dtype, such as another sample's labels under the same
  name. It does not detect a re-segmentation of the same image, which would
  need reading pixels.

  The annotation itself is assumed valid, such as every row's region being a
  declared one: `validate_table` checks it, through `TableModel.validate` and
  its own annotation checks, and every writer and `concat` check it before
  the canonical components.

  No check ties a region to its rows, so subsetting rows or regions,
  reordering and concatenating rows keep the payload valid. The trade-off, from #279: a
  centers matrix that does not belong to its rows, for example after
  reassigning a reordered `adata.obs`, goes undetected; the alignment of
  `obs` and `obsm` is AnnData's responsibility, as the docstring of
  `validate_table` will state.

### Public package `harpy.table.canonical_centers`

Kept:

- `CANONICAL_OBSM_KEY`, `SPATIAL_COORDINATES_KEY`, `CANONICAL_AXES`,
  `CANONICAL_SCHEMA_VERSION`, `CANONICAL_ALGORITHM_VERSION`;
- `validate_canonical_centers`, renamed from `validate_canonical_payload`
  (see step 3);
- `CanonicalMetadata` and `CanonicalRegionMetadata`, which
  `validate_canonical_centers` returns.

Made private, renamed with a leading underscore and no longer exported, still
used by Harpy's writers: `build_canonical_metadata`,
`canonical_metadata_to_storage`, `parse_canonical_metadata`, and, with a new
name, signature or fields (see step 2), `build_canonical_source_signature`,
`calculate_canonical_centers`, `build_canonical_region_binding` and
`CanonicalRegionBinding`.

Deleted: `inspect_canonical_cache`, `read_canonical_centers_from_cache`,
`build_canonical_cache_update_payload`, `build_instance_set_digest`,
`CanonicalSourceSignature` (merged into `CanonicalRegionMetadata`, see step 2),
`CanonicalCacheState`, `CanonicalCacheReport`, `CanonicalCacheMismatch`,
`CanonicalMismatchCode`, `CanonicalCacheUpdateAction`,
`CanonicalCacheUpdatePayload`, `CanonicalCacheUpdateResult` and
`CanonicalCentersResult`.

## Implementation plan

Files in `src/harpy/table/canonical_centers/` unless stated otherwise.

### Step 1: contract (#279)

- `_schema.py`:
  - `CANONICAL_SCHEMA_VERSION = 2`; `canonical_metadata_to_storage` writes no
    `coverage`.
  - The record takes the version-2 layout of the contract above.
    `canonical_metadata_to_storage`, `parse_canonical_metadata`,
    `_parse_region_metadata`, `_TOP_LEVEL_KEYS` and `_REGION_KEYS` follow it;
    `_serialize_generated_by` goes. Validation loses the check that the
    record's `region_key` and `instance_key` equal the annotation's: once the
    binding is gone, nothing else reads them, and renaming the region column
    would invalidate correct centers. `build_canonical_metadata` loses those
    two parameters.
  - `parse_canonical_metadata` reads version 2 only. Another version raises,
    and the message says to rebuild with `hp.tb.add_canonical_centers(sdata,
table_name=..., overwrite=True)`.
  - `_validate_schema_v1_source`, and the messages and docstrings that name
    version 1, become version-neutral.
  - `_validate_canonical_matrix` rejects non-finite values.
  - `validate_canonical_payload(sdata, *, table_name)` loses `table`,
    `region_key`, `instance_key` and `regions`, the per-region bindings, the
    comparison of the stored row count and digest with the current rows, and
    the overlap, coverage and per-region finite checks. It looks the table up
    and calls the private `_validate_canonical_payload(sdata, table)`, which
    validates a table that need not be in `sdata`. That one calls a helper
    without `sdata`, `_validate_canonical_without_sources(table)`, which
    `concat` reuses, then compares each region's source signature with the
    labels. When canonical components are present, the helper reads the
    annotation with spatialdata's `get_table_keys`, which raises without one,
    and normalizes the declared regions with `_names`
    (`src/harpy/utils/_names.py`); this replaces the same check in
    `validate_table`. It does not check the annotation's
    rows, which the annotation checks cover. It requires an entry for every
    declared region, instead of the record's regions equal to the declared
    ones, and returns only those entries, so a table subset to some of its
    regions stays valid and the source signatures are compared only for
    labels the table still declares. Its `z = 0` check takes the dimensions
    from the record.
  - Callers: `validate_table` calls `validate_canonical_payload`. The writer
    of `aggregate_points` (`src/harpy/table/_aggregation_writer.py`) and
    `add_canonical_centers` validate a staged table, not in `sdata`, with
    `_validate_canonical_payload`; the staged table of `aggregate_points`
    carries its annotation from `TableModel.parse`.
- `_models.py`:
  - `CanonicalRegionMetadata` loses `n_obs`, `instance_set_digest` and the
    three `generated_by_*` fields; `CanonicalMetadata` loses `region_key` and
    `instance_key`.
  - `CanonicalRegionBinding` loses `instance_set_digest`. The checks that its
    instance IDs are not empty and unique run inside the digest today, and
    move to `__post_init__`.
  - Delete `build_instance_set_digest`, `_build_instance_set_digest`,
    `_validate_digest`, `_DIGEST_DOMAIN`, `_DIGEST_ENCODING_VERSION`, and the
    helpers only the digest uses.

Left broken until later steps:

- `inspect_canonical_cache` still compares the removed fields: deleted in
  step 2.
- `add_canonical_centers` validates a staged table built without the
  annotation: step 4.
- `__init__.py` still exports `build_instance_set_digest`, so
  `harpy.table.canonical_centers` does not import: step 2. `harpy` itself
  imports again after step 4.
- `_canonical_table_payload` and `_assemble_canonical_table_payload` still
  pass the removed fields to `CanonicalRegionMetadata` and
  `build_canonical_metadata`: step 3.
- The tests: step 5.

### Step 2: remove the cache API

- Delete the names listed under "Deleted" above, and the private report and
  mismatch helpers of `_schema.py` (`_report`, `_all_regions_mismatch`,
  `_region_mismatch`, `_bounded_detail`) and of `_models.py`
  (`_ALL_REGIONS_MISMATCH_CODES`, and `_normalize_canonical_centers`, which
  only `CanonicalCacheUpdatePayload` and `CanonicalCentersResult` use: the
  calculation checks that the centers are finite, `z = 0` holds by
  construction for 2D labels, and the validation checks the rest).
- `build_canonical_metadata`, `canonical_metadata_to_storage` and
  `parse_canonical_metadata` get a leading underscore.
- `CanonicalSourceSignature` merges into
  `CanonicalRegionMetadata(dims, shape, dtype, algorithm_version)`. A separate
  type served the cache API, which carried the labels source without an
  algorithm version. Its checks and those of `_validate_source` move to
  `CanonicalRegionMetadata.__post_init__`; its unused `ndim` goes.
  - `labels_name` and `source_scale` go, not move: the first is the key in
    `CanonicalMetadata.regions`, whose check that the two match goes too; the
    second is a constant, which the parser checks and the writer writes, as
    `source_element_type`.
  - The record is unchanged: `canonical_metadata_to_storage` and
    `_parse_region_metadata` map the flat fields to the nested `source` block.
  - `build_canonical_source_signature` becomes `_build_region_metadata(sdata,
labels_name)`, the entry a new calculation records, with
    `CANONICAL_ALGORITHM_VERSION`. The writers record it, and
    `_validate_canonical_payload` compares it with the stored entry: the
    algorithm-version check before it reduces that to comparing the `source`.
  - The parser checks the record's structure and constants and converts the
    stored values; `CanonicalRegionMetadata` checks the values. So
    `_parse_region_metadata` and `_parse_canonical_metadata` drop the checks
    the models repeat: dims, shape, dtype, the positive integers and the
    region names. `_SUPPORTED_SOURCE_DIMS` becomes a tuple, so that dims that
    cannot be hashed raise `ValueError`.
- `_UnsupportedSchemaError`, `_TopLevelContractError` and
  `_RegionMetadataError` go: they only let `inspect_canonical_cache` map
  failures to mismatch codes. The parser raises `ValueError`; the `_require_*`
  helpers lose `error_type`, `_require_integer` loses `positive`, and
  `_raise_key_error` folds into `_require_exact_keys`.
- `calculate_canonical_centers(sdata, report)` becomes
  `_calculate_canonical_centers(sdata, binding, region_metadata) ->
NDArray[np.float64]` in `_calculation.py`. It keeps the checks that instance
  IDs fit the labels dtype and that every center is finite. The check that the
  labels source is unchanged goes: it caught a labels element replaced between
  an inspection and the calculation, and without reports, both callers build
  the entry right before calculating.
- `build_canonical_region_binding(table, *, labels_name)` becomes
  `_build_canonical_region_binding`, and reads the region and instance keys
  and the declared regions with `get_table_keys` and `_names`, as the
  validation does; `table_name`, `region_key`, `instance_key` and `regions`
  go.
- `CanonicalRegionBinding` becomes `_CanonicalRegionBinding`, keeps
  `labels_name`, `row_positions` and `instance_ids`, and loses `table_name`,
  `region_key` and `instance_key`. Read from the annotation, the keys would
  only repeat it. Once the cache models are deleted, `table_name` only names
  the table in messages, and it and `region_key` are read only by the binding
  checks of `_canonical_table_payload` and
  `_assemble_canonical_table_payload`, which step 3 replaces. `instance_key`
  only names the instance column of `RasterAggregator`'s output:
  `_calculate_canonical_centers` passes a fixed name instead.
- Both become private because only the calculation needs a binding. A
  reader, such as Spatiato, finds a region's rows from the annotation; nothing
  about that is specific to canonical centers. The binding's checks that the
  instance IDs are positive integers of at most 64 bits serve the calculation;
  `validate_table` checks what a reader relies on: unique `(region, instance)`
  pairs and declared regions equal to the observed ones.
- `__init__.py` exports only the names under "Kept".

Left broken until later steps (`harpy` does not import until step 4):

- `_canonical_center_payloads` (`src/harpy/table/_allocation.py`) still
  imports `CanonicalRegionBinding` from the package, builds a
  `CanonicalCacheReport`, bindings with the removed fields, and calls
  `calculate_canonical_centers`; both assembly functions take
  `CanonicalCacheUpdatePayload`: step 3.
- `add_canonical_centers` still calls `inspect_canonical_cache` and
  `calculate_canonical_centers`: step 4.
- `docs/canonical_centers.md` lists the deleted and the now private names, so
  the docs build fails: step 7.
- The tests: step 5.

### Step 3: one assembly path for both writers

- `_models.py`: `_CanonicalRegionCenters(binding, metadata, centers)`, a
  frozen dataclass: the calculated centers of one labels region, with the
  rows they belong to and the entry the record stores. It replaces
  `CanonicalCacheUpdatePayload` between the calculation and the assembly.
- `_calculation.py`:
  - `_calculate_canonical_centers(sdata, binding) -> _CanonicalRegionCenters`
    builds the entry itself with `_build_region_metadata`, right before it
    reads the pixels, so the record describes the source the centers came
    from. Both writers would otherwise make the same two calls per region.
  - `_assemble_canonical_components(n_obs, regions)` replaces
    `_canonical_table_payload` (`src/harpy/table/_aggregation_writer.py`, for
    `aggregate_points`) and `_assemble_canonical_table_payload`
    (`src/harpy/table/_canonical_centers.py`), which both place per-region
    centers in an `(n_obs, 3)` matrix and build the record. It checks that
    each labels region appears once, that rows neither overlap nor exceed
    `n_obs`, and that every row has a finite center; it returns the matrix and
    the stored record.
- `aggregate_points` drops its check that each binding matches the
  checkpoint's output rows: `_calculate_aggregation_canonical_centers` builds
  the bindings from the same `output_row_keys`. The helper's coverage and
  finite checks remain.
- `_canonical_center_payloads` (`src/harpy/table/_allocation.py`) becomes
  `_calculate_aggregation_canonical_centers`: "payloads" named the deleted
  `CanonicalCacheUpdatePayload`. It builds `_CanonicalRegionBinding`s,
  imported from `_models.py`, without `table_name`, `region_key` and
  `instance_key`, calls `_calculate_canonical_centers` directly instead of
  building a `CanonicalCacheReport`, and returns one `_CanonicalRegionCenters`
  per aggregation pair. Its `region_key` and `instance_key` parameters go, and
  its docstring no longer mentions coverage metadata.
- `add_canonical_centers` calls the helper instead of
  `_assemble_canonical_table_payload`.
- `validate_canonical_payload` and `_validate_canonical_payload` become
  `validate_canonical_centers` and `_validate_canonical_centers`: "payload"
  was the cache design's word, and the name now pairs with
  `add_canonical_centers`.
- The staleness error says "Canonical centers of labels element … are stale",
  with the cause and the rebuild hint, instead of "Canonical source signature
  … is stale": `CanonicalSourceSignature` is gone.

Left broken until step 4: `add_canonical_centers` still calls
`inspect_canonical_cache` and `calculate_canonical_centers`, so `harpy` does
not import.

### Step 4: `add_canonical_centers` through `add_table_components` (#282)

In `src/harpy/table/_canonical_centers.py`:

1. Check the arguments, then the table with
   `_validate_table_without_canonical_centers(sdata, table_name)`.
2. Check that every annotated region is a labels element with rows (today's
   `_normalize_labels_names`, without `labels_name`).
3. Per region: `_build_canonical_region_binding` and
   `_calculate_canonical_centers`; then `_assemble_canonical_components`.
4. `hp.tb.io.add_table_components(sdata, table_name=..., components={("obsm",
"spatial_canonical"): centers, ("uns", "spatial_coordinates",
"spatial_canonical"): record}, obs_identity=table.obs[[region_key,
instance_key]], overwrite=overwrite)`:
   - backed: staging, publication, the check of `obs` against the store, the
     collision check in the store, rollback, consolidation, and a lazy
     reattachment;
   - unbacked: installation in memory, with the previous values restored on
     failure;
   - the other entries under `uns["spatial_coordinates"]` stay;
   - a backed SpatialData whose table exists only in memory keeps raising.

Deleted: `_CanonicalCentersDestination`,
`_validate_canonical_centers_destination`, `_normalize_labels_names`,
`_validate_existing_canonical_components`, `_stage_canonical_components` and
`_install_canonical_components`.

As implemented:

- Point 2 is one check, that every region is a labels element:
  `_validate_table_without_canonical_centers` already requires each declared
  region to exist in `sdata` and to have rows.
- `_validate_table_without_canonical` and
  `_validate_canonical_without_sources` become
  `_validate_table_without_canonical_centers` and
  `_validate_canonical_centers_without_sources`: "canonical" alone did not
  say what is left out.
- The new components are not validated before installation (decision 5). A
  first version validated them on a candidate table, but every check held by
  construction: the matrix shape, dtype and finite values come from
  `_assemble_canonical_components`, the region entries from
  `_build_region_metadata` moments earlier, and `z = 0` for 2D labels from the
  calculation. It could only catch a bug in Harpy's own code, which the tests
  catch.
- `overwrite` only concerns the store, as in `add_table_components` and
  `add_feature_matrix`: unbacked, existing canonical centers are replaced
  without it. A first version also raised for centers in the attached table,
  unlike Harpy's other writers.
- Errors from `add_table_components` pass through: `FileExistsError` when the
  store has canonical centers and `overwrite` is `False`; `FileNotFoundError`
  when the table of a backed SpatialData exists only in memory.
- The `add_canonical_centers` docstring is rewritten here, not in step 7: the
  old one documented `labels_name` and the cache's own publication.
- `harpy` imports again.

### Step 5: tests

In `src/harpy/_tests/test_table/test_canonical_centers.py`:

- Delete the tests of the digest encoding and of the cache states; the
  calculation tests call `_calculate_canonical_centers`; the tests that build
  `CanonicalRegionMetadata` pass `dims`, `shape`, `dtype` and
  `algorithm_version`, without a `CanonicalSourceSignature`; the tests call
  `validate_canonical_centers` with `sdata` and `table_name`, and the test
  names follow the new function name; the tests import
  `_CanonicalRegionBinding` and `_build_canonical_region_binding` from
  `_models.py` and `_schema.py`, and pass the latter the table and
  `labels_name`.
- `test_validate_canonical_payload_rejects_malformed_components`: the
  "schema" case sets version 1 instead of 2, and matches the rebuild hint.
- `test_validate_canonical_payload_rejects_a_stale_source_signature` matches
  "are stale" instead of "source signature".
- `add_canonical_centers` has no `labels_name` (step 4): delete
  `test_add_canonical_centers_rejects_an_incomplete_or_duplicate_labels_assertion`;
  the multi-region test drops `labels_name`.
- Two tests patch the deleted staging helpers of `_canonical_centers.py`:
  delete `test_canonical_shared_publication_restores_components_and_attachment`,
  since `add_table_components`' own tests cover its rollback, and
  `test_canonical_centers_reject_invalid_staged_payload_before_publication`,
  since the components are no longer validated before installation (step 4).
  The tests of `add_canonical_centers` run `validate_table` afterwards,
  backed and unbacked.
- Add:
  - unbacked 2D and 3D labels give the same centers as backed ones, and
    `validate_table` passes;
  - a multi-region table, unbacked;
  - backed: `overwrite=False` with either component in the store raises
    `FileExistsError`; `overwrite=True` replaces both;
  - unbacked: existing canonical centers are replaced without `overwrite`;
  - a failure while installing on unbacked SpatialData restores the previous
    `obsm` and `uns`;
  - unbacked: no store is opened or written;
  - another entry under `uns["spatial_coordinates"]` stays;
  - a row subset of a table with canonical centers passes `validate_table`,
    and so does a subset to one region, stored with that region only;
  - a NaN in the matrix fails validation.

### Step 6: reserved `.obsm` keys for feature matrices

`add_feature_matrix` accepts `feature_key="spatial_canonical"`. On a table
without canonical centers it writes a feature matrix there, and the table then
fails `validate_table` ("Canonical centers require both the matrix and its
spatial-coordinate metadata"). On a table with them it raises only after
computing the features, with a message about the matrix shape.
`aggregate_points` already refuses class matrices under `spatial_canonical` or
`spatial`, which tools read as coordinates.

- `src/harpy/table/_validation.py`: one constant,
  `_RESERVED_OBSM_KEYS = frozenset({CANONICAL_OBSM_KEY, _SPATIAL})`, next to
  the feature-matrix contract.
- `_validate_feature_matrices` rejects a record under a reserved key, so
  `validate_table` rejects it whatever wrote the table.
- `add_feature_matrix` rejects a reserved `feature_key` with its other argument
  checks, before any features are computed. It does not run the validator.
- `aggregate_points` uses the constant instead of its inline set and keeps its
  check before counting; its staged validation would also catch the key, but
  only after counting.
- Breaking, accepted: `feature_key="spatial"` raises too, and a table with a
  feature matrix registered under either key fails `validate_table`.
- Tests: in `test_add_feature_matrix.py`, both keys raise and nothing is
  written; in `test_validation.py`, a record under a reserved key fails
  `validate_table`.

### Step 7: documentation

- `docs/canonical_centers.md`: list only the public names; describe the
  version-2 record field by field, the frame and its pixel convention
  included; define algorithm version 1: the center of mass of each label's
  pixels, with uniform weights, background 0 excluded; state what the source
  signature detects and that it misses a re-segmentation of the same image;
  no "table-row coverage".
- Docstrings: `aggregate_points` (no coverage metadata); `validate_table` (no
  instance digests, and the alignment of `obs` and `obsm`). The
  `add_canonical_centers` docstring was rewritten in step 4.
- `docs/development/metadata.md`: no "row coverage" in the canonical record;
  a feature matrix may not use the reserved keys `spatial_canonical` and
  `spatial` (step 6).
- `docs/development/storage.md`: `validate_canonical_centers()` instead of
  `validate_canonical_payload()`.

## For Spatiato

- The calls it plans: `validate_canonical_centers(sdata, table_name=...)`,
  renamed from `validate_canonical_payload`, which reads the annotation from
  the table, and `add_canonical_centers(sdata, table_name=..., overwrite=True)`,
  unchanged.
- `build_canonical_region_binding` and `CanonicalRegionBinding` become
  private. Spatiato finds the rows of the selected labels element from the
  annotation, with spatialdata's `get_table_keys(table)`, as its Object
  Classification does, and their instance IDs in the instance column. Where it
  reads the region or instance key from a binding today, such as to keep
  annotations from overwriting those columns, it reads them with
  `get_table_keys(table)` too.
- `CanonicalSourceSignature` goes: `CanonicalRegionMetadata` holds `dims`,
  `shape` and `dtype`.
- The instance digest goes (#279), so the digest domains of Harpy and
  Spatiato no longer need to match.
- After `add_canonical_centers`, `obsm["spatial_canonical"]` is, for backed
  SpatialData, a lazy Dask array, as after `add_feature_matrix`, no longer a
  read-only `zarr.Array` (decision 5); for unbacked SpatialData, a NumPy
  array in memory. Spatiato's `read_matrix_rows(...)` reads the selected
  labels element's rows from either.
- Every record Spatiato wrote is version 1, fails validation, and is rebuilt
  once with `add_canonical_centers(..., overwrite=True)`.

## Out of scope

- Regional updates of canonical centers (decision 1).
- Switching to spatialdata's half-integer pixel centers (decided on
  2026-11-02): it would be algorithm version 2. The record states the
  convention, so the current numbers stay unambiguous.
- A per-row copy of the instance IDs next to the centers, to detect a
  misaligned `obsm`: more metadata that can go out of sync (#279,
  "Trade-off").
