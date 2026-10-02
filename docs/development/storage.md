# Storage writes and overwrite guarantees

Harpy separates **serializing new data** from **replacing existing data**. This
document defines replacement/deletion scope, staging, publication and recovery
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
- **Deletion target:** an explicitly requested component path with no replacement
  payload. Its previous contents remain recoverable in a backup until success.
- **Backup:** previous destination data retained during replacement so it can
  be restored on failure. Backups are separate from the workspace and are made
  by renaming existing paths, not by reserializing their contents.
- **Publication:** moving staged paths to permanent destinations while retaining
  backups until the caller finishes installing the update. Explicit deletion
  targets are moved to backups too, leaving their permanent paths absent.
- **Installation:** reopening the published data, attaching it to the in-memory
  object, validating it as appropriate and refreshing consolidated metadata.
- **Block:** one piece of a lazy Dask array, computed by one task. Dask itself
  calls these chunks, as in `X.chunks`.
- **Stored chunk:** a separately compressed piece of a Zarr array on disk, which
  Zarr decompresses whole whenever any part of it is read.

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
An existing replacement destination requires `overwrite=True`.

Whole-table replacement is not a merge: old components absent from the new
AnnData are not retained. Component writes preserve unrequested siblings; replacing
an `.uns` mapping replaces its whole value. `None` is an encoded value where
supported, not a deletion command. These two APIs do not provide dataframe-column
or regional row updates; use the regional writer below for selected regions.

Component writes preserve existing axis identities, their order, and spatial linkage.
Matrix updates require the corresponding `obs_identity`, `var_names`, or
`raw_var_names`, unless a supplied `obs`, `var`, or `raw.var` dataframe already
provides them. When both a dataframe and explicit identities are supplied, both
are validated and must agree in value and order. For annotated tables, observation
identity means ordered `(region, instance_id)` pairs; the identity dataframe's index is ignored.
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

`hp.tb.delete_table_components(store, table_name=..., components=[...])` removes
explicitly named optional components. A mixed update uses
`hp.tb.write_table_components(..., components={...}, delete=[...])` instead;
its replacement mapping must still be nonempty. Both entry points share
`_write_table_operation()` and one publication/rollback context. Separate calls
commit independently.

| Deletion target                                                         | Contract                                                               |
| ----------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| Individual `layers`, `obsm`, `varm`, `obsp`, `varp`, `raw.varm` entries | Remove only the named entry; preserve its parent mapping.              |
| Individual or nested `uns` records                                      | Supported except `uns["spatialdata_attrs"]` and its descendants.       |
| `X`                                                                     | Preserve `obs`, `var` and table shape; reopen with `X=None`.           |
| Entire `raw`                                                            | Remove the container as a unit; reopen with `raw=None`.                |
| `obs`, `var`, `raw.var`, `raw.X`, whole mapping roots                   | Rejected. Required axes and containers cannot be removed individually. |

Deletion-only requests need no identities or overwrite flag. Explicit deletion
authorizes removal; `overwrite` in a mixed request controls replacements only.
Replacement values retain their existing identity/shape checks. All paths must
be unique and non-overlapping across both sets, including absent targets.
Removing dataframe columns still requires a whole-frame replacement, not a
column deletion. Callers explicitly identify related scientific records; Harpy
does not infer cascading deletions or removals from omitted values.

A valid absent target is logged at INFO and skipped. If every deletion target
is absent and there are no replacements, the validated request returns without
staging, backups or metadata writes. Encoded `None` is a present value. Invalid
paths, malformed parents, missing stores/tables and I/O errors still raise.
Deletion inspects metadata only, never the target's matrix or dataframe payload.

All replacement serialization finishes before **any** deletion target moves.
For example, a lazy replacement `.X = layers["counts"] * 2` can read the old
layer while staging even when that layer is listed in `delete`. Publication
then backs up both old paths, installs the new `X` and leaves the layer absent.
Backups are discarded only after installation and consolidation succeed;
handled failures restore both paths and the saved store-root metadata.

These path-based APIs do not update live objects. An installing adapter owns
removal or refresh of affected in-memory entries and restores them on failure.
The shared crash-recovery and concurrent-access limitations apply equally to
deletions and replacements.

`hp.tb.write_table_components_by_region(store, table_name=..., components=...,
obs_identity=..., fill_values=..., chunk_size=1000, overwrite=...)` updates individual `.obsm`
matrices for complete regions of an existing annotated table. The nonempty
`obs_identity` dataframe must contain exactly the stored region and instance
columns, with a categorical region column and non-null, unique region/instance
pairs. Actual region values select the regions; unused categories do not.
Declared regions must match those actually present in stored observations
before selection is validated.

Supply every observation of each selected region exactly once, in stored table
order, with each submitted matrix in that same order. If regions interleave,
do not concatenate region-by-region blocks without aligning them first.
The identity dataframe's index is ignored. Neither axis identities nor linkage
can change; there is no automatic reordering or arbitrary subset-of-cells update.

Submitted matrices must be numeric, two-dimensional dense, CSR or CSC matrices
with known shapes. Dask inputs must also have known chunk sizes. DataFrame-valued
matrices and `None` matrix payloads are rejected.

- Existing matrices preserve unselected measurements, including missing values.
  Updates require matching formats (dense/dense, CSR/CSR or CSC/CSC), unchanged
  column counts, and safe casts into the stored dtype. The submitted regional
  matrix and the stored matrix may have different storage backing and chunk
  layouts. No automatic format conversion occurs.
- New matrices retain the submitted format and dtype. Unselected rows require
  an explicit scalar in `fill_values`, keyed by the submitted component path.
  Those unselected rows require a zero fill for sparse matrices; dense fills
  must be dtype-compatible. No fill is needed when every row is supplied.
  Fills for existing entries are ignored; they never replace existing measurements.
- Omitted components remain unchanged. Optional accompanying `.uns` replacements
  update their entire values and share rollback with the matrices; callers
  prepare coherent scientific metadata. Protected SpatialData annotation cannot change.

Regional writes construct complete replacements in chunks and rewrite each
affected `.obsm` entry in full. They do not read or rewrite unrelated matrices.
`chunk_size` controls rows per computational chunk for dense/CSR matrices or
columns for CSC matrices, keeping the other axis whole. It applies to in-memory
inputs, backed sparse reads, and new matrices requiring unselected rows.
Input preparation preserves existing Dask and dense Zarr chunks; merging may
lazily repartition a working view without changing supplied arrays. When only
some regions are selected, existing dense targets retain their chunk layout
during merging.
When all observations are supplied, the replacement instead keeps the prepared
input's computational chunks, even when updating an existing entry.
`chunk_size` does not specify on-disk chunk sizes.

Memory may include identity columns, requested metadata, active chunks and
caller-supplied computations. Inputs are not modified. The function returns
`None` without refreshing live objects; reopen affected data after writing.

The internal `_write_table_operation()` coordinates validation, serialization
through `_write_anndata_element()`, and publication through
`_publish_staged_paths()`. Lazy data is fully serialized into staging while old
destinations remain readable, including when a replacement depends on those
destinations. Matrix validation inspects structure rather than numerical values.
Annotation-only updates do not read or rewrite unrelated matrices; consolidation
may inspect metadata across the store.

For a complete-table write, this internal operation also validates any optional
`obs_identity`, `var_names`, or `raw_var_names` against the submitted AnnData, not
the old destination axes. Conflicting identities are rejected before staging.
Public `write_table()` and `add_table()` obtain identities from the AnnData itself;
they do not require separate identity arguments.

Lazy and storage-backed matrices are serialized incrementally, without preliminary
whole-matrix materialization or densification. Memory use includes requested
annotations, active chunks, and computations needed to produce them; no fixed
memory limit is guaranteed.

To use AnnData's chunked writers, Harpy internally wraps Zarr arrays and Zarr-backed
CSR/CSC dataset handles in Dask arrays. Constructing these graphs does not
materialize the matrices; they are evaluated during serialization. The caller's
matrices retain their original representations.

The operation retains backups through its caller's with-body and final
consolidation. Public writers use an empty body because they do not attach data.
Adapters using this internal operation can install reopened data before commit
and must restore their own affected in-memory state on failure. Parent groups
created by the operation are removed on failure.

The publication/finalization part is shared through `_publish_table_paths()`.
Writers with domain-specific staging, such as aggregation and canonical-center
updates, use it directly without serializing their staged data a second time.
They retain responsibility for scientific construction, domain validation and
in-memory restoration. Aggregation keeps its partitioned checkpoint and sparse
row-block writes; canonical updates publish only the coordinated matrix and
metadata entries. Both reopen read-only handles from permanent paths.

Canonical centers deliberately retain their own staging step. Generic component
validation checks shapes, identities and SpatialData linkage; canonical validation
additionally checks coordinate values and consistency with canonical metadata and
source labels. `add_canonical_centers()` runs `validate_canonical_payload()` on the
serialized, reopened components before publication, so validating the original
in-memory inputs alone is not sufficient.

`_write_table_operation()` currently offers no caller-defined validation between
staging and publication: its caller's `with` body runs only after publication.
Supporting canonical validation there would require a pre-publication callback.
For now, explicit canonical staging keeps this sequence visible without adding a
callback contract solely for this case. Publication, disk rollback and metadata
finalization remain shared through `_publish_table_paths()`. Reconsider a callback
if another component writer needs the same staged-validation extension point.

Table-writing staging workspaces live beside the SpatialData Zarr store, not
inside its `tables` group. This applies both to ordinary table writes and writers
that prepare their own staged data. Missing destination parents are created only
within shared publication setup, which tracks them for cleanup.

Component-update adapters refresh only the affected in-memory entries, preserving
unrelated local state. Before attaching an observation-aligned matrix to an
existing AnnData, the installing caller must verify that the complete in-memory
observation identities match storage in value and order. This applies even after
a regional update: the regional writer validates the submitted subset, whereas
attachment assigns every row of the resulting matrix by position. Checking only
selected observations cannot protect unchanged measurements from being attached
to the wrong in-memory observations.

For example, `hp.tb.add_feature_matrix()` follows this contract when attaching
the complete updated feature matrix after a regional write.

Consolidated metadata at the SpatialData store root contains collected metadata
from descendant groups and arrays, including table components. Restoring a table
or component directory does not restore this root-level index. The operation
therefore restores saved root-metadata file contents (or their prior absence)
separately, without depending on another successful consolidation attempt. The
shared publisher's recovery limitations still apply.

Inputs are not modified, and path-based writes do not synchronize an existing
`sdata` or external references. Reopen affected data after writing; dirty/stale
tracking remains caller-owned as described below.

#### Adding a table to SpatialData

`hp.tb.add_table(sdata, adata, output_table_name=..., region=..., overwrite=...)`
combines table preparation with attachment to the supplied `sdata`:

- **Unbacked `sdata`:** attach the prepared table without writing or computing its
  matrices. Existing entries are replaced regardless of `overwrite`, preserving
  this API's unbacked behavior.
- **Backed `sdata`:** use `_write_table_operation()` for a complete-table write,
  reopen only the affected published table using `_read_anndata_table()` in lazy
  mode, and attach it before metadata finalization and backup disposal.
  Replacing an attached or stored table requires `overwrite=True`, including
  tables omitted when reading a store selectively.

The caller's AnnData is not modified: parsing uses copied annotations while
retaining matrix representations without a preliminary whole-matrix copy or
computation. Unbacked results may still share matrix data with the input.
Any StringDType compatibility conversion applies only to the prepared or reopened
target, not other tables.

The shared writer owns disk and root-metadata recovery; the adapter restores the
previous attached entry on failure, or removes the new entry if none existed.
Other attached elements and detached references are not refreshed. After success,
use `sdata.tables[output_table_name]` for the current table.

For example, write a complete result and refresh its attached representation:

```python
sdata = hp.tb.add_table(
    sdata, processed, output_table_name="processed", region=["cells"],
    region_key="region", instance_key="instance_id", overwrite=True,
)
processed = sdata.tables["processed"]
```

For an annotation-only update, the SpatialData-aware component adapter avoids
rewriting matrices and refreshes only the requested live component:

```python
hp.tb.add_table_components(
    sdata, table_name="processed", components={("obs",): updated_obs}, overwrite=True,
)
```

Here `updated_obs` preserves the stored index and, for an annotated table, its
region/instance identities and order. Complete-table writes are needed to change
axes or linkage.

#### Updating components of an attached table

There are two explicit ownership levels:

| APIs                                                | Effect                                                                                   | Return               |
| --------------------------------------------------- | ---------------------------------------------------------------------------------------- | -------------------- |
| `write_table_components`, `delete_table_components` | Update the supplied store only; do not modify live objects.                              | `None`               |
| `add_table_components`, `remove_table_components`   | Update selected components of an attached table, and its store when `sdata.path` is set. | The supplied `sdata` |

The SpatialData-aware adapters retain the existing AnnData object. They require
the table to be attached and, when backed, already present in storage. They do
not load or create a missing table implicitly. Destination AnnData views are
rejected: callers must explicitly prepare and attach a non-view table. HDF5-backed
AnnData destinations are also rejected because their setters can write through
to a separate file, outside Harpy's Zarr publication operation. Harpy's read-only
Zarr handles and lazy arrays do not make AnnData HDF5-backed.

The same component paths, explicit identity arguments, protected annotation,
raw-creation and deletion rules apply at both ownership levels. In particular,
passing `sdata` does not establish the order of replacement matrix rows or
columns; callers still supply the corresponding identities or axis dataframes.
Only relevant axes are checked. Backed updates additionally check that these
complete in-memory axes match storage in identity and order before positional
attachment. Raw uses its independent feature axis.

- **Unbacked SpatialData:** updates stay in memory, without serialization or
  numerical computation. Complete-component replacements retain matrix
  representations and may share data with inputs. As with `add_table`,
  `overwrite` is ignored; validation still applies.
- **Backed SpatialData:** `overwrite=True` is required if a replacement target
  exists in either memory or storage. Reopen only requested replacements from
  their permanent paths, using lazy matrices and eager annotations. Unrelated
  local edits and matrix references are not refreshed or persisted.

`add_table_components(..., components={...}, delete=[...])` combines replacements
and deletions. `remove_table_components(..., components=[...])` is deletion-only;
explicit deletion paths authorize removal without an overwrite flag. Presence
is resolved independently in memory and storage: a memory-only entry is removed
without staging or consolidation, and a disk-only entry is still removed from
storage. Targets absent in both locations are logged and skipped. Malformed
parents remain errors, not missing targets.

Both adapters prepare new mapping containers without mutating unrelated entries.
For backed updates, `_write_table_operation()` finishes replacement serialization
before publication, including when a lazy replacement reads a deletion target.
The adapter installs requested entries inside the operation's context, while
backups remain available; consolidation and commit follow successful installation.
On handled failure, the shared writer restores storage/root metadata and the
adapter restores the original affected in-memory references. Unbacked failures
restore those references without any storage work.

Regional adapters, such as `hp.tb.add_table_components_by_region()`, follow the
same observation-alignment, selective-installation and recovery guarantees.
Unselected observations retain measurements from the stored target when backed,
or from the attached target when unbacked. For backed updates, storage determines
target existence, format, shape and dtype: a target present only in memory is new
on disk and follows the new-entry fill rules. Preserving unrelated local components
does not preserve unsaved values within a requested matrix.

For unbacked SpatialData, `hp.tb.add_table_components_by_region()` attaches the
complete updated matrix lazily, without executing the numerical merge or writing
to disk. This guarantee concerns the regional update itself, not calculations
callers perform to prepare the submitted measurements.

External references are not refreshed. Callers continue to own dirty/stale
tracking, application events and coherent scientific metadata; the existing
concurrency and crash-recovery limitations are unchanged.

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

Backed handles are read-only, not write-through views. Indexing them reads the
selected values into memory; Dask selections remain deferred until computed.

Harpy returns an ordinary AnnData container whose `isbacked` property is `False`,
even when its matrices depend on disk. AnnData's flag describes its own
file-managed backing mode; Harpy's `mode` selects the representation of individual
matrices. Components can have different representations after editing—for example,
`.X` may be materialized while layers remain lazy. Consequently, no single
table-level flag describes whether all data is in memory. Storage write permissions
are separate from these representations.

Lazy reads choose a block layout through `sparse_chunks` and `dense_chunks`,
both `"auto"` by default. Neither changes disk storage or backed/eager reads.

- Sparse blocks keep the uncompressed axis whole: rows per CSR block, columns
  per CSC block. `"auto"` sizes them to about Dask's `array.chunk-size`, reading
  only metadata. Sparse matrices have no stored chunk grid along rows to align
  with; `_sparse_block_length` explains why that stays cheap.
- Dense blocks are whole rows, the layout {doc}`scanpy <scanpy:index>` and
  {doc}`rapids-singlecell <rapids_singlecell:index>` require, built from whole
  stored chunks so that no stored chunk is split. `"storage"` keeps the stored
  chunks instead. `_dense_lazy_chunks` documents the sizing rule.

`array.chunk-size` is read when the lazy arrays are built. Tables reopened by
`add_table`, `add_table_components` and the regional adapter use the same
defaults. The regional writer reads existing matrices with
`dense_chunks="storage"`, so existing dense targets keep their stored layout
during merging.

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
`table.io._read` locates the selected table, and `_storage._anndata` decodes it:

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

`hp.io.read_zarr(store, table_name=None, table_mode="lazy", sparse_chunks="auto", dense_chunks="auto")`
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

`table_mode`, `sparse_chunks` and `dense_chunks` use the table-reader contracts above;
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
  in-memory objects and metadata outside the published paths. Table writers use
  `_publish_table_paths()` to restore saved store-root metadata, without another
  consolidation attempt. Other element adapters may attempt to refresh
  consolidated metadata. The exception is propagated to the caller.

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
matrix data. It must be updated after publication and restored after rollback so
readers do not use stale descriptions of the store. Table publication restores
its saved contents directly. Non-table adapters may instead attempt a best-effort
refresh after rollback; a failed refresh is logged rather than hidden by a claim
of full recovery.

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
