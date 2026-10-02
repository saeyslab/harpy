"""Path-based reading of tables in local SpatialData Zarr stores."""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from typing import Literal

import zarr
from anndata import AnnData

from harpy._storage._anndata import (
    _MATRIX_MAPPINGS,
    _DenseChunks,
    _MissingAnnDataElement,
    _read_anndata_element,
    _read_anndata_table,
    _ReadMode,
    _SparseChunks,
    _validate_dense_chunks,
    _validate_sparse_chunks,
)
from harpy._storage._spatialdata import _open_spatialdata_group

type ComponentPath = tuple[str, ...]

_RESERVED_NAMES = {".", "..", ".zarray", ".zattrs", ".zgroup", ".zmetadata", "zarr.json"}


def read_table(
    store: str | PathLike[str],
    *,
    table_name: str,
    mode: _ReadMode = "lazy",
    sparse_chunks: _SparseChunks = "auto",
    dense_chunks: _DenseChunks = "auto",
) -> AnnData:
    """Read one complete table without opening other SpatialData elements.

    Parameters
    ----------
    store
        Local path to an existing SpatialData Zarr root, not a table group.
    table_name
        Name of the stored table.
    mode
        Matrix representation: ``"lazy"`` (default) returns Dask arrays with
        sparse blocks preserved; ``"backed"`` returns Zarr arrays or CSR/CSC
        dataset handles; ``"eager"`` loads NumPy/SciPy matrices into memory.
        Annotations, dataframe-valued entries and ``uns`` are always in memory.
        The returned AnnData has ``isbacked=False``; this does not mean that
        its matrices are loaded into memory.
    sparse_chunks
        Block size of lazy sparse matrices along their compressed axis: rows
        per CSR block or columns per CSC block, keeping the other axis whole.
        ``"auto"`` (default) chooses the rows (CSR) or columns (CSC) per block
        so that a block holds about ``array.chunk-size`` bytes, a
        :doc:`Dask configuration setting <dask:configuration>` (128 MiB by
        default). The bytes per row or column follow from the average number
        of non-zero values and the stored dtypes, reading only metadata. A
        positive integer sets the size directly. Ignored for dense arrays and
        other modes.
    dense_chunks
        Block layout of lazy dense arrays. ``"auto"`` (default) and a positive
        integer give blocks of whole rows, spanning all other axes, the layout
        :doc:`scanpy <scanpy:index>` and
        :doc:`rapids-singlecell <rapids_singlecell:index>` expect for lazy
        matrices. Their number of rows is the largest multiple of the stored
        row chunk size that does not exceed the requested rows, and at least
        one stored chunk. ``"auto"`` requests the rows that fit in Dask's
        ``array.chunk-size`` setting, as for ``sparse_chunks``; an integer
        requests that number of rows. A block therefore never splits a stored
        chunk; when one stored chunk exceeds the target, a block is that chunk.
        ``"storage"`` keeps the stored chunks. String arrays always keep their
        stored chunks. Ignored for sparse matrices and other modes.

    Returns
    -------
    AnnData
        A detached table preserving all slots, including layers, pairwise
        matrices and raw data with its independent feature axis.

    Notes
    -----
    The returned table is separate from any table attached to a SpatialData
    object. Editing its in-memory annotations or replacing its matrices does
    not update that object or write to disk. Storage is opened read-only in
    every mode; writes through backed handles are also disallowed.

    In ``"lazy"`` and ``"backed"`` modes, matrices still depend on the source
    store. If their backing data is overwritten, the returned table does not
    refresh and may read replacement data with stale annotations or fail.
    Call ``read_table`` again and use the new result. To preserve the old
    result, keep its backing data unchanged or materialize it before the
    overwrite. ``"eager"`` results are already independent of the source store.

    No scientific metadata validation is performed.

    ``"auto"`` block sizes read ``array.chunk-size`` when the lazy arrays are
    built, so set it before calling, for example with
    ``dask.config.set({"array.chunk-size": "64MiB"})``.

    See Also
    --------
    harpy.table.read_table_components : Read only selected components.
    harpy.table.write_table : Persist a complete table through staging and publication.

    Examples
    --------
    .. code-block:: python

        adata = hp.tb.read_table("sdata.zarr", table_name="counts", mode="lazy")
        selected = adata[adata.obs["region"] == "labels_a"].copy()
        # Matrices remain lazy; custom metadata may need adjustment after selection.

        # Replacing X changes only this AnnData object, not the stored matrix.
        adata.X = adata.X * 2

        # Changing values through a backed handle instead attempts a disk write.
        backed = hp.tb.read_table("sdata.zarr", table_name="counts", mode="backed")
        backed.X[0, 1] = 99  # Raises ValueError: storage is read-only.
    """
    _validate_read_mode(mode)
    sparse_chunks = _validate_sparse_chunks(sparse_chunks)
    dense_chunks = _validate_dense_chunks(dense_chunks)
    group = _open_table_group(store, table_name=table_name)
    return _read_anndata_table(group, mode=mode, sparse_chunks=sparse_chunks, dense_chunks=dense_chunks)


def read_table_components(
    store: str | PathLike[str],
    *,
    table_name: str,
    components: Sequence[ComponentPath],
    mode: _ReadMode = "lazy",
    sparse_chunks: _SparseChunks = "auto",
    dense_chunks: _DenseChunks = "auto",
    missing: Literal["raise", "omit"] = "raise",
) -> dict[ComponentPath, object]:
    """Read selected logical components without constructing an AnnData.

    Parameters
    ----------
    store
        Local path to an existing SpatialData Zarr root.
    table_name
        Name of the stored table.
    components
        Nonempty sequence of unique, non-overlapping tuple paths. Supports
        ``("X",)``, ``("obs",)``, ``("var",)``, matrix mappings such as
        ``("obsm",)`` or individual entries such as ``("obsm", "embedding")``,
        and ``("uns",)`` or nested metadata paths. Raw components support
        ``("raw", "X")``, ``("raw", "var")`` and ``("raw", "varm")`` or
        its entries. Dataframe columns, array slices and encoding internals
        cannot be selected; nested metadata traversal requires mappings.
    mode
        Matrix representation: ``"lazy"`` (default) returns Dask arrays with
        sparse blocks preserved; ``"backed"`` returns read-only Zarr arrays or
        CSR/CSC dataset handles; ``"eager"`` materializes only the requested
        scope as NumPy/SciPy matrices. Dataframes and all values below ``uns``
        are always decoded into memory.
    sparse_chunks, dense_chunks
        Lazy block layout of sparse and dense matrices, as described for
        :func:`harpy.table.read_table`. Ignored in other modes.
    missing
        Raise KeyError for an absent component, or omit its dictionary entry.
        A present encoded None is retained. Invalid paths, invalid traversal
        and decoding errors still raise, even with ``missing="omit"``.
        Missing stores or tables always raise FileNotFoundError.

    Returns
    -------
    dict
        Decoded values keyed by requested paths, in request order. Annotation
        reads neither construct nor compute unrelated matrix graphs.

    Notes
    -----
    Results follow the same ownership and backing-path rules as
    :func:`harpy.table.read_table`. Neither reader writes to disk.

    See Also
    --------
    harpy.table.read_table : Read a complete table.
    harpy.table.write_table_components : Persist only selected components.

    Examples
    --------
    .. code-block:: python

        values = hp.tb.read_table_components(
            "sdata.zarr", table_name="counts", components=[("obs",), ("uns", "analysis")]
        )
        obs = values[("obs",)]
    """
    _validate_read_mode(mode)
    sparse_chunks = _validate_sparse_chunks(sparse_chunks)
    dense_chunks = _validate_dense_chunks(dense_chunks)
    if not isinstance(missing, str):
        raise TypeError("missing must be 'raise' or 'omit'.")
    if missing not in {"raise", "omit"}:
        raise ValueError("missing must be 'raise' or 'omit'.")
    paths = _validate_component_paths(components)
    group = _open_table_group(store, table_name=table_name)
    result = {}
    for path in paths:
        try:
            result[path] = _read_anndata_element(
                group, path, mode=mode, sparse_chunks=sparse_chunks, dense_chunks=dense_chunks
            )
        except _MissingAnnDataElement:
            if missing != "omit":
                raise
    return result


def _validate_read_mode(mode: _ReadMode) -> None:
    if not isinstance(mode, str):
        raise TypeError("mode must be 'backed', 'lazy' or 'eager'.")
    if mode not in {"backed", "lazy", "eager"}:
        raise ValueError("mode must be 'backed', 'lazy' or 'eager'.")


def _validate_path_segment(name: str) -> None:
    if not isinstance(name, str):
        raise TypeError("Table names and component path segments must be strings.")
    if not name or name in _RESERVED_NAMES or any(character in name for character in ("/", "\\", "\x00")):
        raise ValueError(f"Invalid table name or component path segment: {name!r}.")


def _validate_component_paths(
    components: Sequence[ComponentPath], *, to_write: bool = False
) -> tuple[ComponentPath, ...]:
    """Validate the logical AnnData address space, not physical Zarr paths.

    Parameters
    ----------
    components
        Nonempty sequence of logical component paths, such as ``("obsm", "embedding")``.
        Paths must be valid, unique and non-overlapping.
    to_write
        Apply write-specific restrictions: reject whole matrix mappings
        (``layers``, ``obsm``, ``varm``, ``obsp``, ``varp`` and ``raw.varm``).
        For example, ``("layers",)`` is rejected but ``("layers", "counts")``
        is allowed. Replacing the whole ``("uns",)`` mapping remains allowed.
        False permits mapping roots for reading. This flag only controls path
        validation; it does not read or write data.
    """
    if isinstance(components, (str, bytes)) or not isinstance(components, Sequence):
        raise TypeError("components must be a sequence of tuple paths.")
    if not components:
        raise ValueError("components must not be empty.")
    paths = []
    for path in components:
        if not isinstance(path, tuple):
            raise TypeError("Each component path must be a tuple of strings.")
        if not path:
            raise ValueError("Component paths must not be empty.")
        for segment in path:
            _validate_path_segment(segment)
        slot = path[0]
        valid = (
            (slot in {"X", "obs", "var"} and len(path) == 1)
            or (slot in _MATRIX_MAPPINGS and len(path) <= 2)
            or slot == "uns"
            or (slot == "raw" and (path[1:] in {("X",), ("var",), ("varm",)} or (len(path) == 3 and path[1] == "varm")))
        )
        if not valid:
            raise ValueError(f"Invalid logical AnnData component path: {path!r}.")
        if to_write and ((slot in _MATRIX_MAPPINGS and len(path) == 1) or path == ("raw", "varm")):
            raise ValueError(f"Write individual entries, not the mapping root {path!r}.")
        paths.append(path)
    _check_component_path_overlap(paths)
    return tuple(paths)


def _check_component_path_overlap(paths: Sequence[ComponentPath]) -> None:
    """Reject duplicate and parent/child paths, including across update intents."""
    for index, path in enumerate(paths):
        # Compare both directions: ("uns", "analysis") and
        # ("uns", "analysis", "method") conflict in either request order.
        # Identical paths also conflict, even for a write/delete pair.
        if any(path[: len(other)] == other or other[: len(path)] == path for other in paths[:index]):
            raise ValueError("Component paths must be unique and non-overlapping.")


def _open_table_group(store: str | PathLike[str], *, table_name: str) -> zarr.Group:
    """Locate one table read-only, without SpatialData's whole-store reader."""
    _validate_path_segment(table_name)
    root = _open_spatialdata_group(store)
    for kind in ("images", "labels", "points", "shapes"):
        if f"{kind}/{table_name}" in root:
            raise ValueError(f"Table name {table_name!r} collides with a {kind} element.")
    try:
        group = root[f"tables/{table_name}"]
    except KeyError:
        raise FileNotFoundError(f"Table {table_name!r} does not exist in {str(store)!r}.") from None
    if not isinstance(group, zarr.Group):
        raise ValueError(f"Table {table_name!r} must be an AnnData Zarr group.")
    if group.attrs.get("encoding-type") != "anndata" or group.attrs.get("encoding-version") != "0.1.0":
        raise ValueError(f"Unsupported AnnData table encoding at {group.name!r}.")
    return group
