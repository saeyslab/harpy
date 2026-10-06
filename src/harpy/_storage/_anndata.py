"""Serialization and storage-backed reading for AnnData stored in Zarr.

These helpers own AnnData encodings and SpatialData's on-disk table format
metadata. They do not publish replacements or perform rollback; table writers
coordinate these operations through ``harpy._storage._publication``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from functools import partial
from math import prod
from numbers import Integral
from pathlib import Path, PurePosixPath
from typing import Literal

import dask
import dask.array as da
import numpy as np
import zarr
from anndata import AnnData, Raw
from anndata.abc import CSCDataset, CSRDataset
from anndata.experimental import read_elem_lazy, write_dispatched
from anndata.io import read_elem, sparse_dataset, write_elem
from dask import delayed
from dask.utils import parse_bytes
from scipy import sparse
from spatialdata.models import TableModel
from zarr.storage import LocalStore

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
# Two sizes are easy to confuse here:
#
# - Stored chunk: a separately compressed piece of a Zarr array on disk. Zarr
#   always decompresses a stored chunk whole, so its size is the minimum cost
#   of reading any part of it. It is fixed when the array is written.
# - Block: a piece of a lazy Dask array in memory, computed by one task. Its
#   target size is Dask's array.chunk-size setting (128 MiB by default), and it
#   is chosen at read time; lazy reads combine whole stored chunks into blocks.
#
# _STORED_CHUNK_BYTES sets the stored chunks of the matrices Harpy writes: the
# target size of a dense matrix's row-only chunks, and the upper limit of the chunks
# of each array of a sparse matrix. It is deliberately much smaller than
# array.chunk-size. A stored chunk is the smallest unit of both reading and
# writing, while lazy reads combine whole stored chunks into blocks of about
# array.chunk-size (32 chunks of 4 MiB per 128 MiB block), so computations
# still get large blocks. Larger stored chunks would make reading a few rows
# decompress the whole chunk, and would force write blocks of at least that
# size, raising the memory of writes from inputs with small blocks. The
# layout on disk must also not depend on the Dask settings of whoever wrote
# the store.
#
# It is also fixed rather than growing with the array, as Zarr's default does
# (about 0.5 MiB for 10 MiB, 4 MiB for 10 GiB, 16 MiB for 1 TiB). A fixed size
# keeps the cost of a partial read the same whatever the table size. Zarr grows
# its chunks mainly to limit the number of files for very large arrays; for
# tables up to tens of GiB both give similar file counts, and sharding is the
# better answer beyond that.
#
# The docstrings of the public table writers (write_table and the functions
# that refer to it) and docs/development/storage.md state this value as 4 MiB;
# update them if it changes.
_STORED_CHUNK_BYTES = 4 * 1024 * 1024


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
            array = da.from_delayed(delayed(read_elem)(element), shape=shape, dtype=meta.dtype, meta=meta)
            return _register_lazy_read(array, element)
        # Harpy owns the sparse block size, independently of AnnData's defaults.
        # Keep the other axis whole.
        length = _sparse_block_length(element, compressed_axis=compressed_axis, sparse_chunks=sparse_chunks)
        chunks = (length, -1) if compressed_axis == 0 else (-1, length)
        return _register_lazy_read(read_elem_lazy(element, chunks=chunks), element)
    # String sizes cannot be derived from the dtype, so string arrays keep their
    # stored chunks.
    chunks = None if encoding == "string-array" else _dense_lazy_chunks(element, dense_chunks=dense_chunks)
    # chunks=None makes AnnData keep the stored chunks.
    return _register_lazy_read(read_elem_lazy(element, chunks=chunks), element)


# The stored element that each lazily read matrix reads, keyed by the Dask
# array's name: the resolved root of its LocalStore and its path in the store.
# A Dask operation that changes the graph gives a new name, while copies,
# pickling, persist() and a rechunk to the same chunks keep it; so an array
# whose name is registered is a read that has not been changed since. The
# name identifies a computation, not the state of the store: it does not
# promise that the values equal what is stored now.
_LAZY_READS: dict[str, tuple[Path, str]] = {}


def _register_lazy_read(array: da.Array, element: zarr.Array | zarr.Group) -> da.Array:
    """Record the stored element that a lazily read array reads, then return the array.

    Only reads from a ``LocalStore`` are registered: its resolved root and the
    element's path identify the element. Reads from other stores, in memory or
    remote, are not, so they count as changed. Every lazy read is registered,
    also the writers' reads of staged components, whose paths never match a
    user's arrays. Entries are small, and repeated reads reuse the same names.
    """
    if isinstance(element.store, LocalStore):
        _LAZY_READS[array.name] = (Path(element.store.root).resolve(), element.path)
    return array


def _lazy_read_source(value: object) -> tuple[Path, str] | None:
    """Return the stored element a lazily read matrix still reads, or ``None``.

    The element is identified by the resolved store root and its path in the
    store. ``None`` means the value is not a Dask array, or not one that a
    lazy read created and nothing has changed since: a derived array, a read
    from a store other than a ``LocalStore``, or an array built outside Harpy's
    readers under another name.
    """
    if not isinstance(value, da.Array):
        return None
    return _LAZY_READS.get(value.name)


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
    are much larger than stored chunks. Harpy's own writes store at most
    ``_STORED_CHUNK_BYTES`` per chunk of each array
    (``_choose_sparse_stored_chunks``), far below ``"auto"`` blocks of about
    ``array.chunk-size``. Stores written with much larger chunks make each
    block decompress more than it uses, which slows reads but does not change
    results.
    """
    # The caller has validated sparse_chunks: "auto" or a positive int.
    if sparse_chunks != "auto":
        return sparse_chunks
    # The caller handles an empty compressed axis before sizing blocks.
    data = element["data"]
    return _auto_sparse_block_length(
        int(element.attrs["shape"][compressed_axis]),
        int(data.shape[0]),
        entry_bytes=data.dtype.itemsize + element["indices"].dtype.itemsize,
        indptr_itemsize=element["indptr"].dtype.itemsize,
    )


def _auto_sparse_block_length(
    length: int, nnz: int, *, entry_bytes: int, indptr_itemsize: int, max_length: int | None = None
) -> int:
    """Return rows (CSR) or columns (CSC) per block of about ``array.chunk-size`` bytes.

    Shared by lazy reads (``_sparse_block_length``, from stored metadata) and
    the regional writer (``_block_length``, from an in-memory matrix or a
    dataset handle's metadata).

    Parameters
    ----------
    length
        Length of the compressed axis over which the ``nnz`` non-zero values
        are spread; gives the average bytes per row (CSR) or column (CSC).
    nnz
        Number of non-zero values.
    entry_bytes
        Itemsize of ``data`` plus that of ``indices``.
    indptr_itemsize
        Itemsize of ``indptr``, one entry per row (CSR) or column (CSC).
    max_length
        Upper limit of the result, by default ``length``. The regional writer
        passes the full table's rows for a new CSR entry, which it sizes from
        the rows of its input only (see ``_block_length``).
    """
    max_length = length if max_length is None else max_length
    if nnz == 0 or length == 0:
        # Only the row pointers remain, so one block is small.
        return max(max_length, 1)
    bytes_per_line = nnz / length * entry_bytes + indptr_itemsize
    return min(max(int(_chunk_size_target() // bytes_per_line), 1), max_length)


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


def _is_matrix_path(path: tuple[str, ...]) -> bool:
    """Return whether a logical AnnData path holds a matrix.

    These are the paths that lazy reads return as matrices: ``X``, the entries
    of ``layers``, ``obsm``, ``varm``, ``obsp`` and ``varp``, and raw's ``X``
    and ``varm`` entries. The matrix can be dense or sparse; the caller checks
    the encoding. Columns of a dataframe-valued ``obsm`` entry are one level
    deeper and are excluded.
    """
    return (
        path in {("X",), ("raw", "X")}
        or (len(path) == 2 and path[0] in _MATRIX_MAPPINGS)
        or (len(path) == 3 and path[:2] == ("raw", "varm"))
    )


def _choose_dense_stored_chunks(shape: tuple[int, ...], itemsize: int) -> tuple[int, ...]:
    """Choose the stored chunk shape for a dense array about to be written.

    The result is passed to Zarr as ``chunks`` when the array is created. Each
    stored chunk spans all axes after the first and holds as many whole rows as
    fit in ``_STORED_CHUNK_BYTES``: at least one row, and at most the array's
    rows.
    """
    n_rows, other_axes = shape[0], shape[1:]
    bytes_per_row = itemsize * prod(other_axes)
    if bytes_per_row == 0:
        # Rows without values: one stored chunk for the whole array, as lazy
        # reads use one block for it.
        rows_per_chunk = n_rows
    else:
        # At least one row, even when a single row exceeds the constant.
        rows_per_chunk = max(_STORED_CHUNK_BYTES // bytes_per_row, 1)
    # At most the array's rows: Zarr stores every chunk at the full chunk
    # shape and fills the part outside the array, so a small array would
    # otherwise get a padded 4 MiB chunk. Sparse matrices have no such cap
    # (_choose_sparse_stored_chunks): for a Dask matrix, AnnData calls the write
    # callback again with its first block, and a cap computed from that block
    # would decide the whole layout.
    rows_per_chunk = min(rows_per_chunk, n_rows)
    # Zarr requires chunk edges of at least 1, also along empty axes, as in
    # its own default.
    return (max(rows_per_chunk, 1), *(max(size, 1) for size in other_axes))


def _choose_sparse_stored_chunks() -> tuple[int]:
    """Choose the stored chunk shape for the arrays of a sparse matrix about to be written.

    AnnData passes one ``chunks`` setting to the ``data``, ``indices`` and
    ``indptr`` arrays of a CSR or CSC matrix, so they share one length:
    ``_STORED_CHUNK_BYTES`` divided by 8 bytes, the width of int64, the widest
    index dtype, and at least one. No array's chunks then exceed
    ``_STORED_CHUNK_BYTES`` for data types up to 8 bytes. Dividing by a
    smaller itemsize, such as that of float32 data, would double the int64
    ``indptr`` chunks, which every lazy read block decompresses again.

    The length depends on nothing about the matrix and is not capped at its
    size, unlike the dense rule. For a Dask matrix, AnnData calls the write
    callback twice: first with the Dask array, then with its first computed
    block as a SciPy matrix. A cap computed in the second call would let the
    first block decide the chunks of the whole matrix; a fixed length makes
    both calls agree. A small matrix therefore gets one padded chunk per
    array, which compression makes cheap.
    """
    return (max(_STORED_CHUNK_BYTES // 8, 1),)


def _rechunk_to_write_blocks(value: da.Array, *, chosen_chunk_rows: int) -> da.Array:
    """Rechunk a dense Dask array into write blocks, so that each stored chunk is written once.

    Terms used below, as in ``docs/development/storage.md``:

    - *stored chunks*: the chunks of the Zarr array about to be written, of
      ``chosen_chunk_rows`` rows each;
    - *input blocks*: the blocks of ``value``, whose layout comes from the caller;
    - *write blocks*: the blocks returned here, each written to Zarr in one step.

    AnnData writes Dask arrays with ``dask.array.store``, which holds one lock
    for the whole array and writes one write block at a time. Zarr writes
    whole stored chunks: a write block that covers only part of a stored chunk
    makes Zarr read, merge and rewrite that chunk, once for every write block
    that shares it. Each write block returned here spans all axes after the
    first and a whole number of stored chunks, so each stored chunk lies in
    exactly one write block and is written once, without being read back.

    The number of stored chunks per write block is the largest input block's
    bytes divided by a stored chunk's bytes, rounded down and at least one.
    Only the boundaries move: write blocks keep about the memory of the input
    blocks, and Zarr compresses the stored chunks of a write block in
    parallel.

    Parameters
    ----------
    value
        The dense Dask array about to be written; its blocks are the input
        blocks.
    chosen_chunk_rows
        Rows per stored chunk, along the first axis, of the Zarr array about
        to be created. ``_choose_dense_stored_chunks`` computes it from
        ``_STORED_CHUNK_BYTES``: as many whole rows as fit in that many bytes;
        its docstring gives the exact rule. It is not read from disk: the
        array does not exist yet.

    Returns
    -------
    dask.array.Array
        ``value`` rechunked into write blocks, or ``value`` itself when it
        holds no values or its blocks already are write blocks.
    """
    if value.size == 0:
        # No values are written, so there is nothing to align.
        return value
    bytes_per_row = value.dtype.itemsize * prod(value.shape[1:])
    stored_chunk_bytes = chosen_chunk_rows * bytes_per_row
    largest_input_block_bytes = value.dtype.itemsize * prod(max(sizes) for sizes in value.chunks)
    # Stored chunks per write block: as many as fit in the largest input block, at least one.
    stored_chunks_per_write_block = max(largest_input_block_bytes // stored_chunk_bytes, 1)
    # Never more rows than the array has.
    rows_per_write_block = min(stored_chunks_per_write_block * chosen_chunk_rows, value.shape[0])
    # Dask's rechunk returns the array itself when its chunks already match,
    # so an input already in these write blocks is not rechunked.
    return value.rechunk({0: rows_per_write_block, **dict.fromkeys(range(1, value.ndim), -1)})


def _write_element_with_layout(
    write_func,
    parent: zarr.Group,
    key: str,
    value: object,
    *,
    iospec,
    dataset_kwargs: Mapping[str, object],
    root: str,
    logical_path: tuple[str, ...],
) -> None:
    """``write_dispatched`` callback: store matrices in Harpy's stored layout, other elements unchanged.

    AnnData calls it just before writing each element, nested ones included,
    and passes the first four arguments positionally. The encoding alone does
    not identify matrices, because numeric ``obs`` columns and arrays in
    ``uns`` are also encoded as ``array``, and so are lists in ``uns``.

    For a dense matrix, it takes three steps:

    1. ``_choose_dense_stored_chunks`` chooses the stored chunk shape from
       ``_STORED_CHUNK_BYTES``: whole rows along the first axis, all other axes
       whole;
    2. for a Dask value, ``_rechunk_to_write_blocks`` rechunks it into write
       blocks of whole stored chunks, using the chosen rows per stored chunk;
    3. the chosen shape goes to Zarr as ``chunks`` in ``dataset_kwargs``, and
       ``write_func`` writes the value.

    For a sparse matrix, ``_choose_sparse_stored_chunks`` chooses one chunk
    length for its ``data``, ``indices`` and ``indptr`` arrays, which goes to
    Zarr as ``chunks`` in the same way. There is nothing to rechunk: AnnData
    writes a Dask sparse matrix by appending one block at a time. For such a
    matrix, AnnData calls this callback a second time with its first computed
    block, which gets the same chunk length.

    Parameters
    ----------
    write_func
        AnnData's write function for this element, which does the writing.
    parent
        The Zarr group that will contain the element.
    key
        The element's key in ``parent``.
    value
        The in-memory value about to be written, such as a NumPy or Dask
        array, a dataframe or a mapping. It is not on disk yet.
    iospec
        The encoding AnnData will use, such as ``array`` or ``csr_matrix``.
    dataset_kwargs
        Zarr options for creating the arrays; dense and sparse matrices get
        ``chunks``.
    root, logical_path
        The Zarr path where ``_write_anndata_element`` writes its value, and
        that value's logical path. The element's logical path is
        ``logical_path`` followed by the element's position below ``root``.

    Examples
    --------
    With ``_STORED_CHUNK_BYTES`` lowered to 96 bytes for readability, a 12 × 4
    float64 matrix (32-byte rows) gets stored chunks of 96 // 32 = 3 rows with
    all 4 columns, ``chunks=(3, 4)``. Zarr writes whole stored chunks, so the
    write blocks must not split them.

    Input blocks smaller than a stored chunk, 2 rows × 2 columns (32 bytes),
    become write blocks of one stored chunk, ``value.rechunk({0: 3, 1: -1})``::

        row             0  1  2  3  4  5  6  7  8  9 10 11
        stored chunks  [---0---][---1---][---2---][---3---]   all 4 columns
        input blocks   [----][----][----][----][----][----]   each in 2 column halves
        write blocks   [---0---][---1---][---2---][---3---]   all 4 columns

    Without the rechunk, stored chunk 1 (rows 3–5) would be written four
    times, by the input blocks of rows 2–3 and 4–5 in both column halves, each
    time read back and merged first.

    Input blocks larger than a stored chunk, 7 rows × 4 columns (224 bytes),
    hold two whole stored chunks (224 // 96 = 2), so they become write blocks
    of 6 rows, ``value.rechunk({0: 6, 1: -1})``::

        row             0  1  2  3  4  5  6  7  8  9 10 11
        stored chunks  [---0---][---1---][---2---][---3---]   all 4 columns
        input blocks   [---------0---------][------1------]   all 4 columns
        write blocks   [-------0--------][-------1--------]   all 4 columns

    Without the rechunk, the boundary at row 7 would split stored chunk 2
    (rows 6–8) between both input blocks, row 6 in the first and rows 7–8 in
    the second, so it would be written twice: once with row 6, then read
    back, merged with rows 7–8 and written again. The rechunk moves the
    boundary down to row 6; the write blocks keep about the size of the input
    blocks.
    Input blocks of 6 rows already are write blocks and are written as they
    are. ``_rechunk_to_write_blocks`` documents the rule.
    """
    relative = PurePosixPath(f"{parent.name.rstrip('/')}/{key}").relative_to(root).parts
    is_matrix = _is_matrix_path(logical_path + relative)
    is_dense_matrix = is_matrix and iospec.encoding_type == "array"
    is_sparse_matrix = is_matrix and iospec.encoding_type in {"csr_matrix", "csc_matrix"}
    if is_dense_matrix and getattr(value, "ndim", 0) > 0:
        chunks = _choose_dense_stored_chunks(value.shape, value.dtype.itemsize)
        if isinstance(value, da.Array):
            value = _rechunk_to_write_blocks(value, chosen_chunk_rows=chunks[0])
        dataset_kwargs = {**dataset_kwargs, "chunks": chunks}
    elif is_sparse_matrix:
        # No cap at the matrix size, unlike dense arrays: for a Dask matrix,
        # AnnData calls this callback again with its first block as a SciPy
        # matrix, and a cap computed from that block would decide the chunks of
        # the whole matrix. The fixed length makes both calls agree.
        dataset_kwargs = {**dataset_kwargs, "chunks": _choose_sparse_stored_chunks()}
    # No shards are passed, so AnnData's opt-in automatic sharding still
    # applies and keeps these chunks as the inner chunks of its shards.
    write_func(parent, key, value, dataset_kwargs=dataset_kwargs)


def _write_anndata_element(
    group: zarr.Group,
    path: tuple[str, ...],
    value: object,
    *,
    logical_path: tuple[str, ...],
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

    ``logical_path`` is the position of ``value`` in the AnnData table, which
    can differ from its staged ``path``: ``()`` for a whole table, ``("raw",)``
    for raw, or a component path such as ``("obsm", "X_pca")`` staged under a
    temporary name. It is required, so that no caller silently gets a layout
    meant for other arrays.

    Dense arrays at matrix paths below it (``_is_matrix_path``) are stored
    in row-only chunks of about ``_STORED_CHUNK_BYTES`` (``_choose_dense_stored_chunks``),
    and Dask inputs are rechunked so that each stored chunk is written once
    (``_rechunk_to_write_blocks``). Sparse matrices at matrix paths get one
    fixed chunk length for their ``data``, ``indices`` and ``indptr`` arrays,
    at most ``_STORED_CHUNK_BYTES`` per chunk and not capped at the matrix size
    (``_choose_sparse_stored_chunks``). Other elements, such as ``obs``/``var``
    columns, ``uns`` and string arrays, keep AnnData's defaults. Sharding
    follows AnnData's ``auto_shard_zarr_v3`` setting.
    """
    parent, key = _resolve_anndata_parent(group, path, create_parents=create_parents)
    callback = partial(
        _write_element_with_layout, root=f"{parent.name.rstrip('/')}/{key}", logical_path=tuple(logical_path)
    )
    write_dispatched(parent, key, _prepare_anndata_value(value), callback=callback)


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
