# Storage writes and overwrite guarantees

Harpy separates **serializing new data** from **replacing existing data**. This
document defines replacement scope, staging, publication and recovery
responsibilities for writers using the shared local-filesystem storage helpers.
It is an internal developer contract, not a public storage API or a guarantee
about every Harpy reader and writer. It does not cover replacing an entire
SpatialData store.

For where metadata lives and which data it describes, see
[Metadata ownership and layout](metadata.md).

> Writing a table component-by-component does not imply updating the existing
> table component-by-component. The paths supplied to the publisher determine
> the replacement scope.

## Terminology

- **SpatialData element:** one named image, labels, points, shapes or table
  element, such as `tables/my_table`.
- **AnnData component:** a part of a table, such as `X`, `obs`,
  `obsm/my_matrix` or `uns/my_matrix_metadata`.
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

## Replacement scope

A table is a SpatialData element; an AnnData component is something inside that
table. The writer must declare which paths form one logical update:

| Scope                      | Example destinations inside `sdata.zarr`                                      | What is replaced                                                                     |
| -------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Whole SpatialData element  | `tables/my_table` or `points/my_points`                                       | The complete element, including all data and metadata inside its directory           |
| One AnnData component      | `tables/my_table/X`                                                           | Only that component; sibling components remain untouched                             |
| Several related components | `tables/my_table/obsm/my_matrix` and `tables/my_table/uns/my_matrix_metadata` | Both components within one rollback context; the rest of the table remains untouched |

These paths are illustrative, not required names. Replacement does not merge
old and new contents. To preserve anything inside a replaced directory, the
writer must include it in the staged replacement. For example, replacing an
entire table does not implicitly preserve its previous custom `.obs` columns
or `.obsm` entries.

When several components must change together, their paths must be included in
the same publication context. Success means all are installed; a caught failure
triggers rollback across the supplied paths, subject to the limitations below.
It does not make multiple filesystem moves crash-atomic.

An overwrite option authorizes replacement; it does not itself provide rollback.
The calling API owns collision checks and overwrite policy. The publisher acts
on the exact paths it receives, which may include both existing destinations
and new ones. Direct writes that bypass staging and publication do not acquire
these protections automatically.

## Responsibilities

- **Writer:** choose the replacement scope, validate the request and payload,
  finish serialization into an owned workspace, and clean up failed staging.
- **Publisher:** validate path ownership, preserve existing destinations, move
  staged data and attempt disk-path rollback on failure.
- **Caller installing the update:** attach the published data, perform any
  required validation and metadata updates, and restore affected in-memory
  state and metadata outside the published paths on failure.

One function or adapter may perform several of these roles. The private package
at `src/harpy/_storage/` separates the reusable responsibilities:

- `_anndata.py` owns AnnData encoding and reading contracts, including
  disk-level table format metadata. Serialization alone is not a transaction.
- `_spatialdata.py` owns whole-element replacement through SpatialData and
  restoration of the affected in-memory element. It does not restore arbitrary
  caller-owned metadata.
- `_publication.py` owns filesystem publication, backups and disk-path rollback.
  It does not serialize, reopen or understand the scientific payload.

## Shared storage APIs and how they interact

There are two ways to coordinate a replacement: the caller can combine AnnData
I/O helpers with the publisher, or use the whole-element SpatialData adapter.
Arrows below mean **calls/uses**, not execution order:

```text
A. AnnData component or table update

┌──────────────────────────────┐
│ Caller coordinates update    │
└─────────────┬────────────────┘
              │
              ├── write/read ──► _anndata helpers
              │
              └── publish ─────► _publish_staged_paths


B. Whole SpatialData element replacement

┌──────────────────────────────┐
│ Caller                       │
└─────────────┬────────────────┘
              │ delegates update
              ▼
┌──────────────────────────────┐
│ _replace_element_on_disk     │
└─────────────┬────────────────┘
              │
              ├── write/read ──► SpatialData I/O
              │
              └── publish ─────► _publish_staged_paths
```

The publisher is shown in both panels for readability: both refer to the
**same `_publish_staged_paths()` implementation**, not separate publishers.

With the AnnData helpers, the caller coordinates staging, publication and
reading. With `_replace_element_on_disk()`, the adapter coordinates that
workflow. Neither route requires `_anndata.py` and `_spatialdata.py` to call
each other; both routes use the same publisher.

### Writing AnnData components

`_write_anndata_element(group, path, value, ...)` writes a value at a logical
AnnData path, such as `("obsm", "my_matrix")`, and returns `None`. It centralizes
path handling and AnnData encoding through `write_elem`. The optional
`create_parents=True` creates missing parents as AnnData-encoded mappings.

This helper does not create a workspace, publish paths, retain backups or
automatically read the result. The caller supplies the target group, normally
in staging when preparing a replacement.

#### SpatialData-specific table metadata

AnnData's writer stores the table's components and their AnnData encodings.
This includes `adata.uns["spatialdata_attrs"]`, where `TableModel.parse()`
records which spatial elements the table annotates and which `.obs` columns
identify regions and instances.

SpatialData's on-disk table format also uses attributes on the **table's Zarr
group itself**, in addition to AnnData's encoding attributes. For example, the
group at `sdata.zarr/tables/my_table` receives:

```json
{
  "spatialdata-encoding-type": "ngff:regions_table",
  "version": "0.2",
  "region": ["my_labels"],
  "region_key": "region",
  "instance_key": "cell_ID"
}
```

Here, `region` lists the annotated spatial elements; `region_key` and
`instance_key` name the `.obs` columns identifying the region and instance.
`version` is the SpatialData table-format version, not the Zarr or Harpy version.
These attributes belong to the **table group**, not `sdata.attrs` at the store
root, and are separate from the entry in `.uns`.

SpatialData's own table writer writes the AnnData contents and then adds these
group attributes. AnnData's writer alone does not copy the relationship from
`.uns` into the group attributes. When writing a table through direct AnnData
I/O, Harpy supplies that additional step with `_write_spatialdata_table_attrs()`.
This helper centralizes the extra attributes required by SpatialData; it does
not rewrite the table's components or publish the table.

### Reading AnnData components and tables

The reading helpers provide two entry points with one shared decoding policy:

- `_read_anndata_element(group, path)` locates **one component** and returns
  its decoded value. It delegates decoding to `_read_backed_element()`, so
  callers reading a component by path do not repeat path traversal or encoding
  checks. It does not construct a table or attach the result to an object.
- `_read_backed_element(element)` accepts a Zarr array or group and implements
  the **shared decoding policy**. Dense arrays remain Zarr arrays; encoded
  CSR/CSC matrices become AnnData sparse-dataset handles. Dataframes and
  mappings are decoded into memory through `read_elem`. This keeps matrix
  reads storage-backed without treating every encoded value as lazy.
- `_read_backed_table(group)` returns an **AnnData object** from a table group,
  using `_read_backed_element()` for `X` and entries in `obsm`, and `read_elem`
  for `obs`, `var` and `uns`. It exists to reconstruct a table without loading
  its complete matrices into memory. It covers these five slots, not arbitrary
  AnnData stores containing other slots such as `layers` or `raw`.

The component reader and table reader both use `_read_backed_element()`; the
table reader does not call `_read_anndata_element()`. None of these readers
writes, publishes or restores data. The caller chooses when to read and must
use permanent backing paths for installed objects, as described below.

### Replacing a whole SpatialData element

`_replace_element_on_disk(sdata, element_name, element, ...)` takes an existing
backed element's name and its replacement value. It is a context manager that
yields the **same SpatialData object with the replacement attached**. It
centralizes the whole-element workflow: write to staging through SpatialData,
publish one element path, reopen through SpatialData and update the in-memory
collection while the backup remains available.

On failure, the publisher attempts disk-path rollback and the adapter restores
the previous in-memory element and attempts to refresh consolidated metadata.
Caller-owned changes outside that element, such as root attributes, still need
explicit recovery by the caller. This adapter handles existing elements, not
first-time creation. The old element must exist both in memory and on disk.

When replacement is the entire update, enter this context with an empty body:

```python
with _replace_element_on_disk(
    sdata, element_name=element_name, element=replacement, element_type=element_type
):
    pass
```

The `pass` means **no additional caller-side work**, not "do nothing". Entering
and exiting the context still performs replacement, attachment and finalization.
Data and metadata inside the replacement element are written together; the
empty body means there are no extra updates to coordinate. There is no need to
reassign `sdata`, because the existing object is updated.

Run related metadata updates inside the body if they must succeed together with
the replacement, and let failures propagate out of the context to trigger
rollback. Being inside the body does **not** automatically protect root
attributes: the caller must snapshot and restore that metadata in memory and
on disk, catching failures around the entire `with` statement. See
[Metadata outside the published paths](#metadata-outside-the-published-paths)
for the recovery responsibilities.

Successful context exit completes publication and backup cleanup. A later
failure cannot trigger rollback of that completed replacement.

### Publishing prepared paths

`_publish_staged_paths(root=..., workspace=..., paths=..., operation=...)` accepts
fully written `_StagedPath` entries and provides the shared filesystem
publication and rollback context. It yields control, not a Zarr group, AnnData
table or SpatialData object. It exists so format-specific writers do not need
separate backup-and-rename implementations.

Both callers managing AnnData updates and the whole-element SpatialData adapter
use this helper. Its staging requirements, protected scope and recovery limits
are the contracts in the following sections.

## Staging requirements

The writer must finish preparing and validating the replacement before
publication starts. Staged data must be fully written to disk, not just described
by an unevaluated computation. The old payload remains available during staging.

Each `_StagedPath` pairs a prepared `staged` path with its permanent
`destination`. Staged paths must be inside the writer-owned workspace;
destinations must be inside the store and outside the workspace. Paths must be
unique and non-overlapping within each set. For example, do not publish both a
whole table and its `X` child as separate entries in the same update.

Destination parent directories must already exist. Creating missing parents,
and any cleanup they require on failure, is the writer's responsibility; the
publisher does not track them. Such setup may modify container metadata even
though the old payload has not been replaced.

## Publication and failure handling

`_publish_staged_paths()` receives already-written data. For each entry,
`entry.staged` is the new data and `entry.destination` is its permanent path:

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
The context manager yields control while those backups are retained, allowing
the caller to complete installation. The publisher does not reopen any data.

The workspace is removed **before yielding**, after its payloads have moved.
Rollback needs the separate backups, not the workspace. Remaining temporary
artifacts inside the workspace are removed along with it.

Responsibility depends on when a failure happens:

- **During preparation or staged validation:** the writer cleans its workspace;
  the old payload has not been replaced. Setup may have created empty container
  groups, so this is not a promise that no store metadata has been touched.
- **During publication or the caller's context body:** the publisher catches
  `BaseException`, attempts to remove published replacements and restores old
  destinations. A newly created destination has no backup and is removed.
- **After disk rollback:** the caller or adapter restores the affected
  in-memory objects and attempts to refresh consolidated metadata. The exception
  is propagated to the caller.

For example, consider a matrix and its metadata published together. If moving
the metadata fails after the new matrix has moved, the publisher attempts to
remove the new matrix and restore both previous destinations, when present.
If writing that matrix fails while it is still in staging, publication has not
begun and the previous destinations remain in place.

### Metadata outside the published paths

The publisher restores **only the supplied paths**. Metadata inside a replaced
whole table moves with it. Metadata stored elsewhere must either be included
as another published path or explicitly restored by the caller. Arbitrary
changes to `sdata.attrs`, other elements or in-memory objects are not
automatically restored by the publisher.

A caller modifying such metadata must snapshot its previous state, perform the
updates while the backup is retained and restore both its in-memory and
persisted state on failure. Catch failures around the **entire** `with`
statement: reopening or final consolidation in an adapter can fail outside
the caller's context body. Writing associated metadata after the publication
context has successfully exited is too late to use its backups for rollback.

Consolidated metadata is the store's metadata index, not another copy of the
matrix data. It must be refreshed after paths change and after rollback so
readers do not use stale descriptions of the store. Refresh after rollback is
best-effort; a failure is logged rather than hidden by a claim of full recovery.

## Storage-backed references after publication

The object installed in `sdata` must use handles opened at the **permanent
destination**, because those handles retain their backing location. Handles
opened in staging may be used for validation but must not survive installation.

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
