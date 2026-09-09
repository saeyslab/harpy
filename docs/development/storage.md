# Storage writes and overwrite guarantees

Harpy separates **serializing new data** from **replacing existing data**. This
overview describes the shared local-filesystem storage helpers and their use by
table aggregation, canonical-center updates and ordinary element overwrites.
It is an internal developer contract, not a public storage API or a guarantee
about every Harpy reader and writer.

> Writing a table component-by-component does not imply updating the existing
> table component-by-component. The paths supplied to the publisher determine
> the replacement scope.

## Terminology

- **SpatialData element:** one named image, labels, points, shapes or table
  element, such as `tables/my_table`.
- **AnnData component:** a part of a table, such as `X`, `obs`,
  `obsm/spatial_canonical` or `uns/spatial_coordinates/spatial_canonical`.
  AnnData's `write_elem` uses "element" for these encoded values; that does not
  necessarily mean a whole SpatialData element.
- **Workspace:** a temporary, writer-owned directory containing newly prepared
  data and, where needed, intermediate computation results.
- **Staged path:** a file or directory containing fully written new data inside
  the workspace, ready to move to its permanent destination.
- **Backup:** previous destination data retained during replacement so it can
  be restored on failure. Backups are separate from the workspace and are made
  by renaming existing paths, not by reserializing their contents.
- **Publication:** moving staged paths to permanent destinations while retaining
  backups until the caller finishes installing the replacement.
- **Installation:** reopening the published data, attaching it to the in-memory
  object, validating it as appropriate and refreshing consolidated metadata.

## Responsibilities

The private package lives at `src/harpy/_storage/`:

| Module            | Owns                                                                                                            | Does not own                                                                                 |
| ----------------- | --------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `_anndata.py`     | AnnData component encoding, storage-backed reading and SpatialData's disk-level table attributes                | Choosing replacement scope, publication or rollback                                          |
| `_spatialdata.py` | Whole-element replacement using SpatialData's serializers and readers; restoring the affected in-memory element | Specialized aggregation or canonical-center computation; restoring arbitrary caller metadata |
| `_publication.py` | Path validation, filesystem moves, backups, disk-path rollback and owned-path cleanup                           | Serialization, reopening, scientific validation or in-memory state                           |

Operation-specific orchestration stays in `src/harpy/table/_aggregation_writer.py`
and `src/harpy/table/_canonical_centers.py`. These modules choose the payload and
destination paths, use `_anndata` for serialization and reading, and call the
shared publisher through small operation-specific wrappers.

These are separate write paths sharing publication, not a pipeline through all
three storage modules:

```text
aggregate_points
    _aggregation_writer: build complete table using _anndata
        -- one whole-table path -------------------+
                                                   |
add_canonical_centers                               |
    _canonical_centers: stage two components        |
                       using _anndata              |
        -- two component paths --------------------+--> _publication
                                                   |    _publish_staged_paths
ordinary backed element overwrite                  |        |
    _spatialdata: stage element using SpatialData   |        v
        -- one whole-element path -----------------+    caller reopens,
                                                        attaches and
                                                        consolidates
```

## The three write paths

### Aggregation: replace a complete table

`aggregate_points()` prepares the counts and canonical-center payload, then
hands table construction to `_write_aggregation_table()`:

1. The aggregation workspace holds the merged-count Parquet checkpoint and a
   staged AnnData group at `workspace/table`.
2. `_write_anndata_element()` writes `.obs`, `.var`, `.uns` and canonical centers
   as a table without `.X`, then writes `.X` and the optional
   `.obsm["auxiliary_feature_counts"]` separately. The sparse outputs are Dask
   arrays of CSR row blocks; AnnData computes and appends those blocks rather
   than materializing the complete matrix for one write.
3. The writer validates the staged payload and calls
   `_write_spatialdata_table_attrs()`. `TableModel.parse()` records the semantic
   table relationship in `.uns`; this helper supplies the additional disk-level
   attributes needed by SpatialData's reader.
4. `_install_aggregation_table()` enters
   `_publish_staged_aggregation_table()`, which passes **one** `_StagedPath` to
   `_publish_staged_paths()`:
   `workspace/table` -> `sdata.zarr/tables/<output_table_name>`.
5. While the backup is retained, `_read_backed_table()` reopens the published
   table. The installer validates and attaches it to `sdata.tables`, then calls
   `sdata.write_consolidated_metadata()`.

The existing table remains untouched while its replacement is being built.
With `overwrite=True`, the **whole output table** is replaced, not merged with
the previous table. Previous custom columns or matrices are not implicitly
preserved. Without an existing output, the same publication path installs a new
table without an old table to back up.

Aggregation includes canonical centers in the table it constructs; it does not
call the public `add_canonical_centers()` afterward.

### Canonical centers: replace two components

`add_canonical_centers()` operates on an existing table. After computing and
aligning centers to its observation rows:

1. `_stage_canonical_components()` uses `_write_anndata_element()` to write only:

   ```text
   workspace/
   |-- obsm/spatial_canonical
   `-- uns/spatial_coordinates/spatial_canonical
   ```

2. `_validate_staged_canonical_components()` reopens and validates those values.
3. `_install_canonical_components()` enters
   `_publish_staged_canonical_components()`. The wrapper passes **two**
   `_StagedPath` entries to the shared publisher, mapping each staged component
   to the matching path inside the existing table.
4. The installer uses `_read_anndata_element()` at the published paths, updates
   the in-memory `.obsm` and `.uns` entries, validates the canonical payload and
   refreshes consolidated metadata.

The rest of the table, including `.X`, `.obs`, layers and unrelated metadata,
is not rewritten. Both the center matrix and its metadata participate in the
same rollback context. If the `uns/spatial_coordinates` parent mapping is
missing, the wrapper creates it and removes it on failure; this parent cleanup
is separate from the publisher's two component paths.

### Ordinary overwrites: replace one SpatialData element

`_incremental_io_on_disk()` delegates to `_replace_element_on_disk()` for an
existing backed image, labels, points, shapes or table element:

1. `_replace_element_on_disk()` builds a temporary SpatialData container holding
   the replacement element and calls `SpatialData.write()` once.
2. It passes **one** `_StagedPath` to `_publish_staged_paths()`, mapping the staged
   element directory to `sdata.zarr/<element_type>/<element_name>`.
3. It refreshes consolidated metadata before reopening through SpatialData's
   reader, attaches the replacement and yields to its caller. After the caller's
   context body succeeds, it refreshes consolidated metadata again before the
   publisher releases the backup.

The replacement's lazy computation may read from the old element during
staging. The adapter rejects replacement if other attached elements still have
lazy dependencies on that destination.

This adapter is not used by the specialized aggregation and canonical-center
writers. Also, a first-time ordinary element write through
`_write_element_with_cleanup()` writes the new element directly and attempts
cleanup on failure; it does not use this replacement workflow.

## Writing AnnData components

`_write_anndata_element()` encodes a value at the supplied group and component
path using AnnData's `write_elem`. It does not create backups, provide rollback
or automatically read the result. The caller chooses the staging location and
coordinates writing, validation and publication as separate operations.

## Publication and failure handling

All three publication paths eventually call `_publish_staged_paths()` with
already-written data. For each entry, `entry.staged` is the new data and
`entry.destination` is its permanent path:

```text
existing entry.destination --rename--> backup (only when present)
entry.staged               --rename--> entry.destination
                                           |
                                   remove workspace
                                           |
                                   yield to caller
                                   reopen final paths,
                                   attach, validate,
                                   update metadata
                                           |
                                 +---------+---------+
                                 |                   |
                              success              failure
                                 |                   |
                          remove backups     remove published paths,
                                             restore old destinations,
                                             re-raise
```

All existing destinations are backed up before any staged paths are moved.
The shared context manager yields no data itself. The table-specific wrappers
open and yield the relevant published Zarr group; the SpatialData adapter
reopens and yields the updated SpatialData object.

The workspace is removed **before yielding**, after its payloads have moved.
Rollback needs the separate backups, not the workspace. This also removes
aggregation's intermediate count checkpoint once it is no longer needed.

Responsibility depends on when a failure happens:

- **During preparation or staged validation:** the writer cleans its workspace;
  the old payload has not been replaced. Setup may have created empty container
  groups, so this is not a promise that no store metadata has been touched.
- **During publication or the caller's context body:** the publisher catches
  `BaseException`, attempts to remove published replacements and restores old
  destinations. A newly created destination has no backup and is removed.
- **After disk rollback:** the operation-specific installer or SpatialData
  adapter restores the affected in-memory objects and attempts to refresh
  consolidated metadata. The exception is propagated to the caller.

For example, if canonical metadata publication fails after the new center
matrix has moved, both component paths are covered by the publisher: it removes
the new matrix and restores the previous matrix and metadata, when present.
If a staged aggregation `.X` write fails, publication has not begun, so the
previous output table remains in place.

### Metadata outside the published paths

The publisher restores **only the supplied paths**. Metadata inside a replaced
whole table moves with it; canonical metadata is protected because its path is
explicitly included. Arbitrary changes to `sdata.attrs`, other elements or
in-memory objects are not automatically restored by the publisher.

A caller modifying such metadata must snapshot its previous state, perform the
updates while the backup is retained and restore both its in-memory and
persisted state on failure. Catch failures around the **entire** `with`
statement: reopening or final consolidation can fail outside the caller's
context body. `_replace_element_on_disk()` provides that context;
`_incremental_io_on_disk()` simply enters and exits it with no additional work.

Consolidated metadata is the store's metadata index, not another copy of the
matrix data. It must be refreshed after paths change and after rollback so
readers do not use stale descriptions of the store. Refresh after rollback is
best-effort; a failure is logged rather than hidden by a claim of full recovery.

## Reading after publication

The reading helpers either return one component or reconstruct a table. They
do not write or publish data:

- `_read_anndata_element()` locates one component by its path, such as
  `obsm/spatial_canonical`, and delegates decoding to `_read_backed_element()`.
- `_read_backed_element()` decides how to represent that stored value: dense
  arrays remain Zarr arrays and encoded sparse matrices become AnnData
  sparse-dataset handles. Dataframes and mappings are decoded into memory with
  `read_elem`.
- `_read_backed_table()` reconstructs the components used by the out-of-core
  aggregation writer into an AnnData object. It uses `_read_backed_element()`
  for `X` and entries in `obsm`, and `read_elem` for `obs`, `var` and `uns`.
  It is not a general reader for arbitrary AnnData stores containing other slots.

The object installed in `sdata` must use handles opened at the **permanent
destination**, because those handles retain their backing location. Handles
opened in staging may be used for validation but must not survive installation.
Storage-backed does not mean Dask-backed here. Canonical validation may
temporarily materialize the relatively small `(n_obs, 3)` center matrix.

## Limits of the guarantee

This is rollback for caught failures, not a crash-atomic transaction:

- Python must execute the exception handler. `SIGKILL`, interpreter/OS crashes
  and power loss can leave partially published paths, missing destinations or
  leftover workspaces/backups. There is no persisted recovery journal, automatic
  repair on reopening or power-loss durability guarantee.
- Filesystem errors during rollback may prevent full restoration. The publisher
  raises a `RuntimeError` identifying retained backup data for manual recovery.
  Housekeeping failures can also leave temporary paths behind and are logged.
- There is no reader/writer locking. Concurrent readers can observe intermediate
  states, and concurrent writers can interfere with recovery.
- Publication requires local, same-filesystem paths. Managed workspaces and
  staged/destination paths cannot themselves be symlinks; ancestor-directory
  symlinks are allowed.

## Checklist for a new writer

1. Decide whether the logical update replaces a whole element or selected
   components, including any metadata that must change with them.
2. Validate the request and serialize the replacement into owned staging using
   the appropriate format-specific writer. Finish lazy writes before publishing.
3. Supply the exact non-overlapping replacement paths to the shared publisher.
4. Reopen at permanent paths and finish attachment, validation and metadata work
   while the backups are still retained.
5. Restore caller-owned memory and metadata on failure, and handle staging/setup
   cleanup. Do not implement a second filesystem backup-and-rename mechanism.

Storage tests live in `src/harpy/_tests/test_storage/`: `test_publication.py`
covers filesystem publication and rollback, `test_spatialdata.py` covers the
whole-element adapter, and `test_anndata.py` covers AnnData/SpatialData encoding
and reading contracts. Operation-specific tests cover table aggregation and
canonical-center installation in `src/harpy/_tests/test_table/`.
