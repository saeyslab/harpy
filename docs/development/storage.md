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

#### Public scoped table writers

`hp.tb.write_table(store, table_name=..., adata=..., overwrite=...)` writes a
complete table. `hp.tb.write_table_components(store, table_name=...,
components=..., overwrite=...)` replaces only named logical components of an
existing table. Both operate on an existing local SpatialData root, preserve
its Zarr format, and return `None` after publication and metadata finalization.
An existing destination requires `overwrite=True`.

Whole-table replacement is not a merge: old components absent from the new
AnnData are not retained. Component writes preserve omitted siblings; replacing
an `.uns` mapping replaces its whole value. `None` is an encoded value where
supported, not a deletion command. There are no dataframe-column or regional
row updates in these APIs.

Component writes preserve existing axis identities, their order, and spatial linkage.
Matrix updates require the corresponding `obs_identity`, `var_names`, or
`raw_var_names`, unless a supplied `obs`, `var`, or `raw.var` dataframe already
provides them. For annotated tables, observation identity means ordered
`(region, instance_id)` pairs; the identity dataframe's index is ignored.
Unannotated tables use observation names. Identities must be unique; different
ordering is rejected rather than automatically corrected. Actual dataframe
replacements also preserve their stored index. Axis or linkage changes require
a complete-table write. Scientific metadata consistency remains the caller's
responsibility.

Absent raw data can be created by supplying `("raw", "X")` and either
`raw_var_names` or a `("raw", "var")` dataframe. Names alone produce a feature
dataframe with that index and no annotation columns. Raw shares the table's
observations but defines its own feature axis; optional `("raw", "varm", key)`
entries must align with it. A matrix is required to initialize raw. Replacing a
stored null entry requires `overwrite=True`, while a genuinely absent raw path
does not. Existing raw components can still be updated independently without
changing their feature axis.

Creation stages the complete raw container and publishes it as one path,
together with any other requested components in the same rollback operation.
It does not rewrite the main table or unrelated matrices. Rollback restores
the prior absence or null entry as well as coupled updates. Raw serialization
uses AnnData's raw encoding, not a generic mapping.

The internal `_write_table_operation()` coordinates validation, serialization
through `_write_anndata_element()`, and publication through
`_publish_staged_paths()`. Lazy data is fully serialized into staging while old
destinations remain readable, including when a replacement depends on those
destinations. Matrix validation inspects structure rather than numerical values.
Annotation-only updates do not read or rewrite unrelated matrices; consolidation
may inspect metadata across the store.

Lazy and storage-backed matrices are serialized incrementally, without preliminary
whole-matrix materialization or densification. Memory use includes requested
annotations, active chunks, and computations needed to produce them; no fixed
memory limit is guaranteed.

The operation retains backups through its caller's with-body and final
consolidation. Public writers use an empty body because they do not attach data.
Adapters using this internal operation can install reopened data before commit
and must restore their own affected in-memory state on failure. Parent groups
created by the operation are removed on failure.

Consolidated metadata at the SpatialData store root contains collected metadata
from descendant groups and arrays, including table components. Restoring a table
or component directory does not restore this root-level index. The operation
therefore restores saved root-metadata file contents (or their prior absence)
separately, without depending on another successful consolidation attempt. The
shared publisher's recovery limitations still apply.

Inputs are not modified, and path-based writes do not synchronize an existing
`sdata` or external references. Reopen affected data after writing; dirty/stale
tracking remains caller-owned as described below.

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
For unannotated tables the same format attributes are written, with `region`,
`region_key` and `instance_key` set to `None` rather than fabricated linkage.

### Reading AnnData components and tables

`hp.tb.read_table(store, table_name=...)` reads one complete, detached AnnData.
`hp.tb.read_table_components(store, table_name=..., components=...)` returns
only the requested values, keyed by logical tuple paths. Both accept a local
SpatialData Zarr root (format 2 or 3), open it read-only and leave other elements
unopened. They do not use SpatialData's whole-store reader or attach results to
a live SpatialData object.

```python
adata = hp.tb.read_table("sdata.zarr", table_name="counts")
backed = hp.tb.read_table("sdata.zarr", table_name="counts", mode="backed")
values = hp.tb.read_table_components(
    "sdata.zarr", table_name="counts", components=[("obs",), ("obsm", "embedding")]
)
```

| Requested value                        | `mode="lazy"` (default)                     | `mode="backed"`                      | `mode="eager"`                     |
| -------------------------------------- | ------------------------------------------- | ------------------------------------ | ---------------------------------- |
| Dense or CSR/CSC matrix                | Dask array; sparse blocks stay sparse       | Zarr array or CSR/CSC dataset handle | NumPy array or SciPy sparse matrix |
| `obs`, `var`, dataframe-valued entries | In-memory pandas dataframe                  | Same                                 | Same                               |
| `uns`, including nested arrays         | In-memory metadata                          | Same                                 | Same                               |
| Matrix mapping, e.g. `("obsm",)`       | Dictionary using these rules for each entry | Same policy, backed matrices         | Same policy, eager matrices        |

`mode` replaces the former `lazy` boolean. Backed handles are read-only, not
write-through views. Indexing them reads the selected values into memory;
Dask selections remain deferred until computed.

Harpy returns an ordinary AnnData container whose `isbacked` property is `False`,
even when its matrices depend on disk. AnnData's flag describes its own
file-managed backing mode; Harpy's `mode` selects the representation of individual
matrices. Components can have different representations after editing—for example,
`.X` may be materialized while layers remain lazy. Consequently, no single
table-level flag describes whether all data is in memory. Storage write permissions
are separate from these representations.

Both readers expose `sparse_chunk_size=1000`: the positive number of rows per
CSR chunk or columns per CSC chunk, keeping the other axis whole. Harpy passes
this choice explicitly to AnnData. Dense arrays use their on-disk chunk layout
without a chunk override. The option only affects lazy sparse reads; it changes
neither disk storage nor backed/eager reads and is not a memory-byte limit.
Lazy and backed matrix reads accept dense `array`/`string-array` encoding version
`0.2.0` and CSR/CSC encoding version `0.1.0`. Harpy rejects other matrix encodings
or versions with `ValueError` before decoding. CSR/CSC version checks also apply
to eager reads.

Complete reads preserve all slots, including layers, pairwise matrices and
`raw` with its independent feature axis. Component paths cannot select
dataframe columns, array slices or encoding internals; nested `uns` paths
traverse mappings only. Missing components raise `KeyError`, or are omitted
with `missing="omit"`; a present encoded `None` remains in the result.
Annotation-only reads neither fetch unrelated matrix chunks nor construct
their graphs. Unsupported encodings fail rather than falling back to a
whole-table read. These readers do not validate scientific metadata.

Local root validation is shared through `_storage._spatialdata`;
`table._io` locates the selected table, and `_storage._anndata` decodes it:

```text
read_table             -> _read_anndata_table (assemble all slots)
                                     |
read_table_components  -> _read_anndata_element (locate one logical path)
                                     |
                          _decode_anndata_element
                          (AnnData encoding registry and matrix read policy)
```

The decoder isolates AnnData's public experimental `read_elem_lazy` API.
Internal `_read_backed_table` and `_read_backed_element` callers share this
infrastructure and retain the same matrix representations as `mode="backed"`.

Returned annotations and metadata are independently owned. Editing them or
assigning another expression matrix does not write to disk. Lazy matrices and
backed handles depend on their source paths: they are **not immutable snapshots**. Reopen
after overwriting backing data; downstream operations may materialize matrices.
None of these readers writes, publishes or restores data. Use permanent backing
paths for installed objects, as described below.

### Reading a SpatialData store with selected tables

`hp.io.read_zarr(store, table_name=None, table_mode="lazy", sparse_chunk_size=1000)`
returns a SpatialData object containing all non-table elements and the selected
tables. `table_name=None` reads all tables, a name or sequence selects exact names,
and `table_name=[]` skips tables. Duplicate names are rejected; missing names raise
`FileNotFoundError`. Only local store paths are supported.

```text
hp.io.read_zarr
    ├── spatialdata.read_zarr (explicitly exclude tables)
    │       └── non-table elements, transformations, root attributes and path
    └── hp.tb.read_table (each selected table)
            └── attach to the returned SpatialData object
```

`table_mode` and `sparse_chunk_size` use the table-reader contracts above;
they do not alter non-table reading. Unselected tables are not decoded, and
selected tables are never first loaded eagerly through SpatialData. Ordinary
`spatialdata.read_zarr()` behavior is unchanged.

The returned `sdata.is_backed()` is true because it has a store path; its
tables still have `adata.isbacked=False`, regardless of their matrix mode.
Reading performs no writes. In-memory edits are not automatically persisted,
backed table handles remain read-only, and lazy/backed matrices must be
reopened after their source data is overwritten.

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

### Caller-owned dirty and stale state

Harpy owns I/O, publication and rollback within the guarantees documented here.
The calling application or workflow owns state tracking for tables, images,
labels and other storage-backed elements:

- **Dirty:** local changes have not been persisted. Callers track these changes
  and mark them clean only after successful writing; newer local changes must
  remain dirty.
- **Stale:** backing data has changed since reading. Callers must use reopened
  data and replace or invalidate dependent references and computations.

Some Harpy operations both write data and update the supplied `sdata` object
with the reopened replacement. That attached replacement is ready to use without
reopening it again. Other variables, handles or Dask graphs referencing the
previous element's data are not refreshed; their lifetime remains the caller's
responsibility. Harpy does not automatically track local edits or invalidate
those dependent references and computations.

Caller-managed tracking covers changes known to the caller. It does not
automatically detect external writes or provide concurrent-access protection.

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
