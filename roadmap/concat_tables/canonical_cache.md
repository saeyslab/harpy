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
  - `validate_canonical_payload(...)`, for the status and before every run;
  - `hp.tb.add_canonical_centers(sdata, table_name=..., overwrite=True)`, when
    the centers are absent or invalid;
  - `build_canonical_region_binding(...)`, for the row positions of the
    selected labels element;
  - the constants (`CANONICAL_OBSM_KEY`, ...).

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
4. **Version-1 records are read, with `coverage` ignored; only version 2 is
   written.** Every table `aggregate_points` wrote in 0.5.x holds a version-1
   record; rejecting it would fail `validate_table` and `concat` on each of
   them until its centers are recomputed.
5. **`add_canonical_centers` installs through
   `hp.tb.io.add_table_components`.** That covers unbacked SpatialData (#282)
   and replaces the canonical-specific staging and publication. Two changes
   follow:
   - for backed SpatialData, the matrix is reattached lazily, as a Dask array,
     as `add_feature_matrix` does, not as a read-only `zarr.Array`;
   - the components are validated in memory before they are installed, not
     as a staged, serialized copy.
6. **`labels_name` is removed from `add_canonical_centers`.** It could only
   restate every region the table annotates.

## Contract after the refactor

- **Matrix:** `obsm["spatial_canonical"]`, dense `float64` of shape
  `(n_obs, 3)`, in `(z, y, x)` order, every value finite; `z = 0` in the rows
  of 2D labels.
- **Record:** `uns["spatial_coordinates"]["spatial_canonical"]`, schema
  version 2. Top-level keys are unchanged. Each region's entry keeps
  `source_element`, `source_element_type`, `source_scale`,
  `coordinate_frame`, `calculation`, `source` and the optional
  `generated_by`; `coverage` goes.
- **Validation** (`validate_canonical_payload`):
  - both components, or neither;
  - the matrix: dense, shape, dtype, finite values;
  - the record parses strictly, with linkage keys equal to the table's
    annotation and regions equal to the declared ones;
  - every row's region is a declared one;
  - per region: the source signature equals the labels element's (the only
    check that needs `sdata`), the algorithm version is supported, and `z = 0`
    for 2D labels.

  No check ties a region to its rows, so subsetting, reordering and
  concatenating rows keep the payload valid. The trade-off, from #279: a
  centers matrix that does not belong to its rows, for example after
  reassigning a reordered `adata.obs`, goes undetected; the alignment of
  `obs` and `obsm` is AnnData's responsibility, as the docstring of
  `validate_table` will state.

### Public package `harpy.table.canonical_centers`

Kept:

- `CANONICAL_OBSM_KEY`, `SPATIAL_COORDINATES_KEY`, `CANONICAL_AXES`,
  `CANONICAL_SCHEMA_VERSION`, `CANONICAL_ALGORITHM_VERSION`;
- `validate_canonical_payload`;
- `build_canonical_region_binding` and `CanonicalRegionBinding`;
- `CanonicalMetadata`, `CanonicalRegionMetadata` and
  `CanonicalSourceSignature`, which `validate_canonical_payload` returns.

Made private, still used by Harpy's writers: `build_canonical_source_signature`,
`build_canonical_metadata`, `canonical_metadata_to_storage`,
`parse_canonical_metadata`, and `calculate_canonical_centers` (with a new
signature, see step 2).

Deleted: `inspect_canonical_cache`, `read_canonical_centers_from_cache`,
`build_canonical_cache_update_payload`, `build_instance_set_digest`,
`CanonicalCacheState`, `CanonicalCacheReport`, `CanonicalCacheMismatch`,
`CanonicalMismatchCode`, `CanonicalCacheUpdateAction`,
`CanonicalCacheUpdatePayload`, `CanonicalCacheUpdateResult` and
`CanonicalCentersResult`.

## Implementation plan

Files in `src/harpy/table/canonical_centers/` unless stated otherwise.

### Step 1: contract (#279)

- `_schema.py`:
  - `CANONICAL_SCHEMA_VERSION = 2`; `canonical_metadata_to_storage` writes no
    `coverage`; `build_canonical_metadata` builds version 2 only.
  - `parse_canonical_metadata` accepts versions 1 and 2. Version 1 requires
    its own key set, `coverage` included, and ignores that entry's content.
  - `_validate_canonical_matrix` rejects non-finite values; the finite checks
    per region in `validate_canonical_payload` and `inspect_canonical_cache`
    go with them.
  - `validate_canonical_payload`: drop the comparison of the stored row count
    and digest with the current rows, and the per-region bindings with their
    overlap and coverage checks: one region column cannot assign a row to two
    regions, so checking that every row's region is declared is enough.
    Structure it as a private helper without `sdata`,
    `_validate_canonical_without_sources(table, *, region_key, instance_key,
regions)`, plus the source signatures per region, so that `concat` reuses
    the helper. The `z = 0` check takes the dimensions from the record, which
    the source-signature check then compares with the labels.
- `_models.py`:
  - `CanonicalRegionMetadata` loses `n_obs` and `instance_set_digest`.
  - `CanonicalRegionBinding` loses `instance_set_digest`. The check that
    instance IDs are unique within a region runs inside the digest today, and
    moves to `__post_init__`.
  - Delete `build_instance_set_digest`, `_build_instance_set_digest`,
    `_validate_digest`, `_DIGEST_DOMAIN` and `_DIGEST_ENCODING_VERSION`.

### Step 2: remove the cache API

- Delete the names listed under "Deleted" above, and the private report and
  mismatch helpers of `_schema.py` (`_report`, `_all_regions_mismatch`,
  `_region_mismatch`, `_bounded_detail`) and of `_models.py`
  (`_ALL_REGIONS_MISMATCH_CODES`).
- `calculate_canonical_centers(sdata, report)` becomes
  `_calculate_canonical_centers(sdata, binding, source_signature) ->
NDArray[np.float64]` in `_calculation.py`. It keeps the checks that the
  labels source is unchanged, that instance IDs fit the labels dtype, and that
  every center is finite.
- `__init__.py` exports only the names under "Kept".

### Step 3: one assembly path for both writers

- `_canonical_table_payload` (`src/harpy/table/_aggregation_writer.py`, for
  `aggregate_points`) and `_assemble_canonical_table_payload`
  (`src/harpy/table/_canonical_centers.py`) both place per-region centers in
  an `(n_obs, 3)` matrix and build the record. One private helper replaces
  them: bindings, source signatures and centers in, the matrix and the stored
  record out. `aggregate_points` keeps its check that each binding matches the
  checkpoint's output rows.
- `_canonical_center_payloads` (`src/harpy/table/_allocation.py`) calls
  `_calculate_canonical_centers` directly instead of building a
  `CanonicalCacheReport`.

### Step 4: `add_canonical_centers` through `add_table_components` (#282)

In `src/harpy/table/_canonical_centers.py`:

1. Check the arguments, then the table with
   `_validate_table_without_canonical(sdata, table_name)`.
2. Check that every annotated region is a labels element with rows (today's
   `_normalize_labels_names`, without `labels_name`).
3. If either canonical component is in the attached table and `overwrite` is
   `False`, raise. This covers unbacked SpatialData, where
   `add_table_components` replaces in-memory components without `overwrite`.
4. Per region: `build_canonical_region_binding`, the source signature, and
   `_calculate_canonical_centers`; then the helper of step 3.
5. Validate the candidate components against the table's `obs` with
   `validate_canonical_payload`.
6. `hp.tb.io.add_table_components(sdata, table_name=..., components={("obsm",
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
`_validate_canonical_centers_destination`,
`_validate_existing_canonical_components`, `_stage_canonical_components` and
`_install_canonical_components`.

### Step 5: tests

In `src/harpy/_tests/test_table/test_canonical_centers.py`:

- Delete the tests of the digest encoding and of the cache states; the
  calculation tests call `_calculate_canonical_centers`; the tests that build
  `CanonicalRegionMetadata` drop the digest.
- Add:
  - unbacked 2D and 3D labels give the same centers as backed ones, and
    `validate_table` passes;
  - a multi-region table, unbacked;
  - `overwrite=False` with an existing component raises, backed and unbacked;
    `overwrite=True` replaces both;
  - a failure while installing on unbacked SpatialData restores the previous
    `obsm` and `uns`;
  - unbacked: no store is opened or written;
  - another entry under `uns["spatial_coordinates"]` stays;
  - a row subset of a table with canonical centers passes `validate_table`;
  - a NaN in the matrix fails validation;
  - a version-1 record passes validation.

### Step 6: documentation

- `docs/canonical_centers.md`: list only the public names; schema version 2;
  no "table-row coverage".
- Docstrings: `add_canonical_centers` (no store required, no `labels_name`,
  lazy reattachment, `add_table_components`' guarantees instead of its own
  publication); `aggregate_points` and `_canonical_center_payloads` (no
  coverage metadata); `validate_table` (no instance digests, and the alignment
  of `obs` and `obsm`).
- `docs/development/metadata.md`: no "row coverage" in the canonical record.

## For Spatiato

- The calls it plans, `validate_canonical_payload(...)`,
  `add_canonical_centers(sdata, table_name=..., overwrite=True)` and
  `build_canonical_region_binding(...)`, keep their signatures.
  `CanonicalRegionBinding` loses only `instance_set_digest`.
- Records Spatiato wrote are version 1 with the same layout, so complete ones
  validate after this change. Partial ones, with labels elements missing from
  the record or NaN rows, fail validation and are recomputed with
  `add_canonical_centers(..., overwrite=True)`.

## Out of scope

- Regional updates of canonical centers (decision 1).
- A per-row copy of the instance IDs next to the centers, to detect a
  misaligned `obsm`: more metadata that can go out of sync (#279,
  "Trade-off").
