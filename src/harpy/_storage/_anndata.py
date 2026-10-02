"""Serialization and storage-backed reading for AnnData stored in Zarr.

These helpers own AnnData encodings and SpatialData's on-disk table format
metadata. They do not publish replacements or perform rollback; table writers
coordinate these operations through ``harpy._storage._publication``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from math import prod
from numbers import Integral
from typing import Literal

import dask
import dask.array as da
import numpy as np
import zarr
from anndata import AnnData, Raw
from anndata.abc import CSCDataset, CSRDataset
from anndata.experimental import read_elem_lazy
from anndata.io import read_elem, sparse_dataset, write_elem
from dask import delayed
from dask.utils import parse_bytes
from scipy import sparse
from spatialdata.models import TableModel

# Harpy owns these literals as the compatibility boundary for tables assembled
# with the low-level AnnData writers, without depending on private SpatialData
# writer APIs.
_SPATIALDATA_TABLE_ENCODING_TYPE = "ngff:regions_table"
_SPATIALDATA_TABLE_FORMAT_VERSION = "0.2"
_ReadMode = Literal["backed", "lazy", "eager"]
_MATRIX_MAPPINGS = ("layers", "obsm", "varm", "obsp", "varp")
# Lazy block sizes: "auto" derives them from Dask's array.chunk-size setting,
# an integer sets them directly, and dense "storage" keeps the stored chunks.
_SparseChunks = Literal["auto"] | int
_DenseChunks = Literal["auto", "storage"] | int


def _validate_sparse_chunks(value: _SparseChunks) -> _SparseChunks:
    return _validate_chunks(value, name="sparse_chunks", modes=("auto",))


def _validate_dense_chunks(value: _DenseChunks) -> _DenseChunks:
    return _validate_chunks(value, name="dense_chunks", modes=("auto", "storage"))


def _validate_chunks(value: object, *, name: str, modes: tuple[str, ...]) -> str | int:
    """Accept one of the named modes or a positive integer, including NumPy integers."""
    allowed = " or ".join([*(repr(mode) for mode in modes), "a positive integer"])
    if isinstance(value, str):
        if value not in modes:
            raise ValueError(f"{name} must be {allowed}.")
        return value
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be {allowed}.")
    if value < 1:
        raise ValueError(f"{name} must be {allowed}.")
    return int(value)


class _MissingAnnDataElement(KeyError):
    """A logical path is absent, as distinct from an error decoding its value."""


def _read_backed_table(group: zarr.Group) -> AnnData:
    """Reopen a table with Zarr arrays and AnnData sparse-dataset handles.

    Internal installation callers retain these handles rather than Dask
    arrays. Use the permanent table path: matrices depend on that location.
    """
    return _read_anndata_table(group, mode="backed")


def _read_anndata_table(
    group: zarr.Group,
    *,
    mode: _ReadMode,
    sparse_chunks: _SparseChunks = "auto",
    dense_chunks: _DenseChunks = "auto",
) -> AnnData:
    """Read a complete AnnData table using the component reader ``_read_anndata_element``.

    ``obs`` and ``var`` are always read eagerly as pandas DataFrames; ``uns``
    is also loaded into memory, including any arrays it contains.

    ``X`` and array-valued entries in ``layers``, ``obsm``, ``varm``, ``obsp``
    and ``varp`` follow ``mode``: ``lazy`` builds Dask arrays without reading
    matrix values, ``backed`` retains Zarr arrays or sparse-dataset handles,
    and ``eager`` loads them into memory. DataFrame-valued entries are always
    eager. Unsupported lazy matrix encodings raise rather than falling back
    to an eager read. ``sparse_chunks`` and ``dense_chunks`` set the lazy block
    layout, as described for ``_decode_anndata_element``.

    When present, ``raw`` follows the same policy: its ``var`` is eager,
    while its ``X`` and ``varm`` follow the requested matrix-reading mode.
    """
    if group.attrs.get("encoding-type") != "anndata" or group.attrs.get("encoding-version") != "0.1.0":
        raise ValueError(f"Unsupported AnnData table encoding at {group.name!r}.")
    chunk_options = {"sparse_chunks": sparse_chunks, "dense_chunks": dense_chunks}
    values = {slot: _read_anndata_element(group, (slot,), mode="eager") for slot in ("obs", "var")}
    values.update(
        {
            slot: _read_anndata_element(group, (slot,), mode=mode, **chunk_options)
            for slot in ("X", "uns", *_MATRIX_MAPPINGS)
            if slot in group
        }
    )
    uns = values.get("uns", {})
    if not isinstance(uns, Mapping):
        raise ValueError("AnnData Zarr component 'uns' must decode to a mapping.")
    uns = dict(uns)
    # AnnData decodes lists as arrays; SpatialData expects a list of regions.
    spatialdata_attrs = uns.get(TableModel.ATTRS_KEY)
    if isinstance(spatialdata_attrs, Mapping):
        spatialdata_attrs = dict(spatialdata_attrs)
        region = spatialdata_attrs.get(TableModel.REGION_KEY)
        if isinstance(region, np.ndarray):
            spatialdata_attrs[TableModel.REGION_KEY] = region.tolist()
        uns[TableModel.ATTRS_KEY] = spatialdata_attrs
    values["uns"] = uns
    if "raw" in group:
        raw = group["raw"]
        if raw.attrs.get("encoding-type") == "null":
            # An existing raw entry can encode None rather than contain raw data.
            # read_elem decodes this explicit absence back to None.
            values["raw"] = read_elem(raw)
        elif raw.attrs.get("encoding-type") == "raw" and raw.attrs.get("encoding-version") == "0.1.0":
            # raw.X can retain genes removed from the main table, so read its
            # own var and varm rather than reuse the main table's feature annotations.
            values["raw"] = {
                slot: _read_anndata_element(group, ("raw", slot), mode=mode, **chunk_options)
                for slot in ("X", "var", "varm")
                if slot in raw
            }
        else:
            raise ValueError(f"Unsupported AnnData raw encoding at {raw.name!r}.")
    return AnnData(**values)


def _read_backed_element(element: zarr.Array | zarr.Group) -> object:
    """Decode one AnnData element while preserving array storage backing.

    Dense arrays remain ``zarr.Array`` objects and encoded CSR/CSC matrices
    become AnnData sparse-dataset handles. Mappings are decoded recursively
    with the same policy; pandas dataframes are loaded into memory. The policy
    depends on the stored encoding rather than its AnnData slot, so it also
    applies to array-valued ``layers``, ``obsm``, ``varm``, ``obsp`` and
    ``varp`` entries.
    """
    return _decode_anndata_element(element, mode="backed")


def _decode_anndata_element(
    element: zarr.Array | zarr.Group,
    *,
    mode: _ReadMode,
    sparse_chunks: _SparseChunks = "auto",
    dense_chunks: _DenseChunks = "auto",
) -> object:
    """Decode an AnnData element eagerly, lazily, or as a storage-backed handle.

    Mappings are decoded entry by entry using the same mode and chunk settings.

    Parameters
    ----------
    element
        Encoded Zarr array or group.
    mode
        ``"lazy"`` builds Dask arrays, ``"backed"`` returns Zarr arrays or
        sparse-dataset handles, and ``"eager"`` loads values into memory.
    sparse_chunks
        Lazy CSR/CSC blocks keep the uncompressed axis whole. ``"auto"`` chooses
        the rows (CSR) or columns (CSC) per block so that a block holds about
        Dask's ``array.chunk-size`` bytes, using only array metadata (see
        ``_sparse_block_length``); an integer gives rows per CSR block or columns
        per CSC block directly.
    dense_chunks
        Lazy dense blocks span all axes after the first. ``"auto"`` sizes rows
        from ``array.chunk-size`` and an integer requests a number of rows; both
        are aligned with the stored row chunks (see ``_dense_lazy_chunks``).
        ``"storage"`` keeps the stored chunks, as do string arrays in any case.

    Both settings are validated in every mode, but only used in lazy mode.

    Raises
    ------
    TypeError
        If ``sparse_chunks`` or ``dense_chunks`` is neither a supported mode
        name nor an integer.
    ValueError
        If ``sparse_chunks`` or ``dense_chunks`` is an unsupported mode name or
        an integer below 1. Also if a CSR/CSC element has an unsupported or
        missing encoding version, regardless of the requested mode, or
        lazy/backed mode does not support the element's encoding or version.
    """
    # Reject unknown settings here rather than silently treating them as "auto".
    sparse_chunks = _validate_sparse_chunks(sparse_chunks)
    dense_chunks = _validate_dense_chunks(dense_chunks)
    encoding = element.attrs.get("encoding-type")
    version = element.attrs.get("encoding-version")
    if encoding in {"csr_matrix", "csc_matrix"}:
        # This is the on-disk sparse format, not the AnnData package version.
        # Unknown formats must not bypass Harpy's sparse chunking policy.
        if version != "0.1.0":
            raise ValueError(f"Unsupported {encoding} encoding version {version!r}; expected '0.1.0'.")
    # Preserve the requested reading mode and chunk settings for matrices
    # inside mappings, including nested mappings.
    if encoding == "dict" and version == "0.1.0":
        return {
            key: _decode_anndata_element(
                element[key], mode=mode, sparse_chunks=sparse_chunks, dense_chunks=dense_chunks
            )
            for key in element.keys()
        }
    if mode == "eager":
        return read_elem(element)
    if encoding in {"dataframe", "null"}:
        return read_elem(element)

    # Harpy defines the supported matrix formats for both lazy and backed reads.
    is_dense = isinstance(element, zarr.Array) and encoding in {"array", "string-array"} and version == "0.2.0"
    is_sparse = isinstance(element, zarr.Group) and encoding in {"csr_matrix", "csc_matrix"}
    if not (is_dense or is_sparse):
        raise ValueError(f"Unsupported AnnData encoding {encoding!r} (version {version!r}) for mode={mode!r}.")
    if mode == "backed":
        if is_dense:
            return element
        # Return a storage-backed handle; values are read on indexing or to_memory().
        return sparse_dataset(element)
    if is_sparse:
        shape = tuple(element.attrs["shape"])
        compressed_axis = 0 if encoding == "csr_matrix" else 1
        if shape[compressed_axis] == 0:
            # AnnData 0.12.10 constructs invalid Dask chunks for an empty
            # compressed axis. Defer its ordinary decoder for this zero-sized
            # matrix instead; even its sparse buffers remain unread here.
            matrix_type = sparse.csr_matrix if compressed_axis == 0 else sparse.csc_matrix
            meta = matrix_type((0, 0), dtype=element["data"].dtype)
            return da.from_delayed(delayed(read_elem)(element), shape=shape, dtype=meta.dtype, meta=meta)
        # Harpy owns the sparse block size, independently of AnnData's defaults.
        # Keep the other axis whole.
        length = _sparse_block_length(element, compressed_axis=compressed_axis, sparse_chunks=sparse_chunks)
        chunks = (length, -1) if compressed_axis == 0 else (-1, length)
        return read_elem_lazy(element, chunks=chunks)
    # String sizes cannot be derived from the dtype, so string arrays keep their
    # stored chunks.
    chunks = None if encoding == "string-array" else _dense_lazy_chunks(element, dense_chunks=dense_chunks)
    # chunks=None makes AnnData keep the stored chunks.
    return read_elem_lazy(element, chunks=chunks)


def _chunk_size_target() -> int:
    """Return Dask's ``array.chunk-size`` setting in bytes, the target of ``"auto"`` blocks."""
    return parse_bytes(dask.config.get("array.chunk-size"))


def _sparse_block_length(element: zarr.Group, *, compressed_axis: int, sparse_chunks: _SparseChunks) -> int:
    """Return rows per lazy CSR block, or columns per lazy CSC block.

    ``"auto"`` divides the memory target by the bytes per row (CSR) or column
    (CSC)::

        average non-zero values × (itemsize of data + itemsize of indices)
        + itemsize of indptr

    The average is the length of ``data`` divided by the length of the
    compressed axis, so only array metadata is read.

    Terms used below:

    - *block*: one piece of the returned Dask array, computed by one task
      (Dask calls these chunks);
    - *stored chunk*: a separately compressed piece of a Zarr array on disk,
      which Zarr decompresses whole.

    Unlike dense arrays, sparse matrices have no stored chunk grid along rows
    to align with: ``data`` and ``indices`` are chunked by non-zero values, and
    rows hold different numbers of them, so block boundaries fall at arbitrary
    positions inside stored chunks. Only the edge chunks of a block are shared
    with its neighbours and decompressed twice. That stays cheap while blocks
    are much larger than stored chunks (as for Harpy's own writes: Harpy sets
    no sparse chunk sizes, so Zarr's default applies, sized from the array or,
    for Dask writes, from the first block written; it grows from about 0.5 MiB
    for 10 MiB to about 4 MiB for 10 GiB, far below ``"auto"`` blocks of about
    ``array.chunk-size``). Stores written with much larger chunks make each
    block decompress more than it uses, which slows reads but does not change
    results.
    """
    # The caller has validated sparse_chunks: "auto" or a positive int.
    if sparse_chunks != "auto":
        return sparse_chunks
    # The caller handles an empty compressed axis before sizing blocks.
    length = int(element.attrs["shape"][compressed_axis])
    data = element["data"]
    nnz = int(data.shape[0])
    if nnz == 0:
        # Only the row pointers remain, so one block is small.
        return length
    bytes_per_line = nnz / length * (data.dtype.itemsize + element["indices"].dtype.itemsize)
    bytes_per_line += element["indptr"].dtype.itemsize
    return min(max(int(_chunk_size_target() // bytes_per_line), 1), length)


def _dense_lazy_chunks(element: zarr.Array, *, dense_chunks: _DenseChunks) -> tuple[int, ...] | None:
    """Return a row-only block layout (Dask chunks) for a dense array, or None to keep its stored chunks.

    Blocks span all axes after the first, because the lazy code paths of
    :mod:`scanpy` (e.g. PCA and QC metrics) and the input check of
    :mod:`rapids_singlecell` require a single block along the columns
    (``X.numblocks[1] == 1``), while stored chunks, such as AnnData's defaults,
    often split the columns.

    Their number of rows is the largest multiple of the stored row chunk size
    that does not exceed the requested rows, and at least one stored chunk, so
    a lazy block never splits a stored chunk. Zarr decompresses stored chunks
    whole, so splitting one would not save memory and would repeat the
    decompression. ``element.chunks`` reports the inner chunks of sharded
    arrays, which can be read individually.

    ``"auto"`` requests the rows that fit the memory target; an integer
    requests that number of rows directly.
    """
    # The caller has validated dense_chunks: "auto", "storage" or a positive int.
    if dense_chunks == "storage" or element.ndim == 0:
        return None
    n_rows = int(element.shape[0])
    stored_rows = int(element.chunks[0])
    if dense_chunks == "auto":
        bytes_per_row = element.dtype.itemsize * prod(element.shape[1:])
        if bytes_per_row == 0:
            # Without columns, the whole array is metadata-sized: read one block.
            return (max(n_rows, 1), *(-1,) * (element.ndim - 1))
        requested_rows = _chunk_size_target() // bytes_per_row
    else:
        requested_rows = dense_chunks
    rows = max(requested_rows // stored_rows, 1) * stored_rows
    return (min(rows, max(n_rows, 1)), *(-1,) * (element.ndim - 1))


def _write_anndata_element(
    group: zarr.Group,
    path: tuple[str, ...],
    value: object,
    *,
    create_parents: bool,
) -> None:
    """Write one logical AnnData path through AnnData's encoding registry.

    Callers must explicitly choose whether to create missing parent groups.
    ``create_parents=False`` requires existing parents; ``create_parents=True`` creates
    missing parents as AnnData-encoded mappings, which is useful for building a
    partial hierarchy in an isolated staging store. Keeping path traversal here
    gives full table writers and partial component writers the same encoding
    boundary; safe replacement is provided separately by
    :func:`harpy._storage._publication._publish_staged_paths`.
    """
    parent, key = _resolve_anndata_parent(group, path, create_parents=create_parents)
    write_elem(parent, key, _prepare_anndata_value(value))


def _prepare_anndata_value(value: object) -> object:
    """Keep storage-backed matrices chunked and serialization independent of input state."""
    if isinstance(value, Raw):
        # Raw shares its parent's observation axis but serializes only X, var
        # and varm. A shape-only parent avoids copying unrelated table data.
        return Raw(
            AnnData(shape=(value.n_obs, 0)),
            X=_prepare_anndata_value(value.X),
            var=value.var.copy(deep=True),
            varm={key: _prepare_anndata_value(item) for key, item in value.varm.items()},
        )
    if isinstance(value, AnnData):
        values = {
            slot: {key: _prepare_anndata_value(item) for key, item in getattr(value, slot).items()}
            for slot in _MATRIX_MAPPINGS
        }
        if value.raw is not None:
            values["raw"] = {
                "X": _prepare_anndata_value(value.raw.X),
                "var": value.raw.var.copy(deep=True),
                "varm": {key: _prepare_anndata_value(item) for key, item in value.raw.varm.items()},
            }
        return AnnData(
            X=_prepare_anndata_value(value.X),
            obs=value.obs.copy(deep=True),
            var=value.var.copy(deep=True),
            uns=_prepare_anndata_value(deepcopy(value.uns)),
            **values,
        )
    if isinstance(value, Mapping):
        return {key: _prepare_anndata_value(item) for key, item in value.items()}
    # Writer contract: serialize lazy/backed matrices incrementally, without
    # preliminary whole-matrix materialization or densification, and leave
    # caller-owned inputs unchanged. Storage handles alone do not guarantee
    # chunked writes, so wrap them in Dask to use AnnData's chunked writers.
    # Building these graphs does not materialize the matrix or modify the input.
    if isinstance(value, (CSRDataset, CSCDataset)):
        value = _decode_anndata_element(value.group, mode="lazy")
    elif isinstance(value, zarr.Array):
        value = da.from_zarr(value)
    if isinstance(value, da.Array) and sparse.issparse(value._meta):
        # AnnData's sparse Dask writer assigns int64 indptr/indices arrays to the
        # first computed block. That block may be shared with caller-owned
        # in-memory or persisted data. Our contract forbids modifying those inputs,
        # so copy each block lazily before serialization, not the whole matrix upfront.
        return value.map_blocks(_copy_sparse_block, meta=value._meta)
    return value


def _copy_sparse_block(block):
    """Isolate a sparse block from writer-side mutations of its index arrays."""
    return block.copy()


def _read_anndata_element(
    group: zarr.Group,
    component_path: tuple[str, ...],
    *,
    mode: _ReadMode = "backed",
    sparse_chunks: _SparseChunks = "auto",
    dense_chunks: _DenseChunks = "auto",
) -> object:
    """Read an AnnData component by its logical keys, not its storage internals.

    Parameters
    ----------
    group
        Zarr group containing the table's AnnData hierarchy or staged
        components, not the SpatialData root.
    component_path
        Nonempty tuple of keys relative to ``group``, such as ``("X",)``,
        ``("obsm", "embedding")`` or ``("uns", "analysis")``. This is a
        logical component address, not a filesystem path.
    mode
        Matrix representation: ``"lazy"`` returns Dask arrays, ``"backed"``
        returns Zarr arrays or sparse-dataset handles, and ``"eager"`` loads
        values into memory. DataFrames and all values below ``uns`` are
        always eager.
    sparse_chunks, dense_chunks
        Lazy block layout of sparse and dense matrices, as described for
        ``_decode_anndata_element``. Ignored in other modes.

    Raises
    ------
    _MissingAnnDataElement
        If a key is absent anywhere in ``component_path``, including the final
        key. This exception is a subclass of KeyError.
    ValueError
        If the path is empty or traversal encounters an unsupported parent
        container.

    Examples
    --------
    For a stored table with sparse ``X`` and a dictionary at ``uns["analysis"]``:

    .. code-block:: python

        # Follow dictionary keys, as in adata.uns["analysis"]["threshold"].
        threshold = _read_anndata_element(group, ("uns", "analysis", "threshold"))

        # Read the sparse matrix as one logical value.
        matrix = _read_anndata_element(group, ("X",))

        # Raises ValueError: "data" is an internal sparse storage buffer.
        _read_anndata_element(group, ("X", "data"))
    """
    if not component_path:
        raise ValueError("An AnnData element path must not be empty.")
    element = group
    for key_index, key in enumerate(component_path):
        parent_path = component_path[:key_index]
        expected_encoding = "raw" if parent_path == ("raw",) else "dict"
        if not isinstance(element, zarr.Group) or (
            key_index > 0
            and (
                element.attrs.get("encoding-type") != expected_encoding
                or element.attrs.get("encoding-version") != "0.1.0"
            )
        ):
            raise ValueError(f"AnnData element parent {parent_path!r} is not a mapping.")
        try:
            element = element[key]
        except KeyError:
            raise _MissingAnnDataElement(component_path) from None
    if component_path[0] in {"obs", "var", "uns"} or component_path == ("raw", "var"):
        mode = "eager"
    if (len(component_path) == 1 and component_path[0] in _MATRIX_MAPPINGS) or component_path == ("raw", "varm"):
        if element.attrs.get("encoding-type") != "dict" or element.attrs.get("encoding-version") != "0.1.0":
            raise ValueError(f"AnnData component {component_path!r} must be an encoded mapping.")
    return _decode_anndata_element(element, mode=mode, sparse_chunks=sparse_chunks, dense_chunks=dense_chunks)


def _resolve_anndata_parent(
    group: zarr.Group,
    path: tuple[str, ...],
    *,
    create_parents: bool,
) -> tuple[zarr.Group, str]:
    """Resolve where to write a logical AnnData component, without writing its value.

    Parameters
    ----------
    group
        Starting Zarr group, typically a table group or staging root.
    path
        Nonempty tuple of logical component names, relative to group.
    create_parents
        Create missing parents as encoded dictionaries. Otherwise, all parents
        must already exist. When traversing into raw, its container must already
        exist; new raw containers are written as complete Raw values.

    Returns
    -------
    parent : zarr.Group
        The group that will contain the component, not the component itself.
    key : str
        The final name in path, to pass with parent to write_elem. This entry
        need not exist yet.

    Examples
    --------
    For path=("raw", "varm", "loadings"), return group["raw"]["varm"] and
    "loadings" after validating the parents. For path=("X",), return group
    itself and "X", without traversing any parents.
    """
    if not path or any(not isinstance(key, str) or not key for key in path):
        raise ValueError(f"AnnData element path must contain non-empty string keys, found {path!r}.")
    parent = group
    for key_index, key in enumerate(path[:-1]):
        if key not in parent and create_parents:
            write_elem(parent, key, {})
        if key not in parent or not isinstance(parent[key], zarr.Group):
            raise ValueError(f"AnnData element parent path {path[:-1]!r} does not exist as a Zarr group.")
        parent = parent[key]
        # Dataframes and sparse matrices are also Zarr groups. Traverse only
        # logical containers (raw or dictionaries), not their encoding internals
        # such as ("X", "data"). The version describes the on-disk encoding.
        expected_encoding = "raw" if path[: key_index + 1] == ("raw",) else "dict"
        if parent.attrs.get("encoding-type") != expected_encoding or parent.attrs.get("encoding-version") != "0.1.0":
            raise ValueError(f"AnnData element parent {path[: key_index + 1]!r} is not an encoded mapping.")
    return parent, path[-1]


def _write_spatialdata_table_attrs(
    group: zarr.Group,
    *,
    regions: Sequence[str] | None,
    region_key: str | None,
    instance_key: str | None,
) -> None:
    """Write SpatialData's disk-level regions-table contract.

    ``TableModel.parse()`` records the semantic table relationship in
    ``adata.uns["spatialdata_attrs"]``. A SpatialData Zarr store also requires
    attributes on the AnnData group itself so that its reader recognizes the
    group as a regions table. Harpy writes the AnnData components directly in
    its out-of-core table path, so this helper adds that second, on-disk
    representation without calling SpatialData's private writer APIs.

    Parameters
    ----------
    group
        AnnData Zarr group that will become a SpatialData table element.
    regions
        Spatial elements annotated by the table, or None for an unannotated table.
    region_key
        Column in ``adata.obs`` that identifies the spatial element, or None
        for an unannotated table.
    instance_key
        Column in ``adata.obs`` that identifies an instance within that
        element, or None for an unannotated table.
    """
    group.attrs["spatialdata-encoding-type"] = _SPATIALDATA_TABLE_ENCODING_TYPE
    group.attrs["region"] = None if regions is None else list(regions)
    group.attrs["region_key"] = region_key
    group.attrs["instance_key"] = instance_key
    group.attrs["version"] = _SPATIALDATA_TABLE_FORMAT_VERSION
