"""Path-based reading of tables in local SpatialData Zarr stores."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from os import PathLike, fspath
from pathlib import Path
from typing import Literal

import zarr
from anndata import AnnData

from harpy._storage._anndata import (
    _DEFAULT_SPARSE_CHUNK_SIZE,
    _MATRIX_MAPPINGS,
    _MissingAnnDataElement,
    _read_anndata_element,
    _read_anndata_table,
    _ReadMode,
)

type ComponentPath = tuple[str, ...]

_RESERVED_NAMES = {".", "..", ".zarray", ".zattrs", ".zgroup", ".zmetadata", "zarr.json"}


def read_table(
    store: str | PathLike[str],
    *,
    table_name: str,
    mode: _ReadMode = "lazy",
    sparse_chunk_size: int = _DEFAULT_SPARSE_CHUNK_SIZE,
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
        ``"backed"`` does not imply AnnData's ``isbacked`` flag.
    sparse_chunk_size
        Rows per CSR chunk or columns per CSC chunk when ``mode="lazy"``.
        Ignored for dense arrays and other modes.

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

    See Also
    --------
    harpy.table.read_table_components : Read only selected components.

    Examples
    --------
    .. code-block:: python

        adata = hp.tb.read_table("sdata.zarr", table_name="counts")
        selected = adata[adata.obs["region"] == "labels_a"].copy()
        # Matrices remain lazy; custom metadata may need adjustment after selection.

        # Reopen with the storage handles used by aggregate_points().
        backed = hp.tb.read_table("sdata.zarr", table_name="counts", mode="backed")
    """
    _validate_read_mode(mode)
    sparse_chunk_size = _validate_sparse_chunk_size(sparse_chunk_size)
    group = _open_table_group(store, table_name=table_name)
    return _read_anndata_table(group, mode=mode, sparse_chunk_size=sparse_chunk_size)


def read_table_components(
    store: str | PathLike[str],
    *,
    table_name: str,
    components: Sequence[ComponentPath],
    mode: _ReadMode = "lazy",
    sparse_chunk_size: int = _DEFAULT_SPARSE_CHUNK_SIZE,
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
    sparse_chunk_size
        Rows per CSR chunk or columns per CSC chunk when ``mode="lazy"``.
        Ignored for dense arrays and other modes.
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

    Examples
    --------
    .. code-block:: python

        values = hp.tb.read_table_components(
            "sdata.zarr", table_name="counts", components=[("obs",), ("uns", "analysis")]
        )
        obs = values[("obs",)]
    """
    _validate_read_mode(mode)
    sparse_chunk_size = _validate_sparse_chunk_size(sparse_chunk_size)
    if not isinstance(missing, str):
        raise TypeError("missing must be 'raise' or 'omit'.")
    if missing not in {"raise", "omit"}:
        raise ValueError("missing must be 'raise' or 'omit'.")
    paths = _validate_component_paths(components)
    group = _open_table_group(store, table_name=table_name)
    result = {}
    for path in paths:
        try:
            result[path] = _read_anndata_element(group, path, mode=mode, sparse_chunk_size=sparse_chunk_size)
        except _MissingAnnDataElement:
            if missing != "omit":
                raise
    return result


def _validate_read_mode(mode: _ReadMode) -> None:
    if not isinstance(mode, str):
        raise TypeError("mode must be 'backed', 'lazy' or 'eager'.")
    if mode not in {"backed", "lazy", "eager"}:
        raise ValueError("mode must be 'backed', 'lazy' or 'eager'.")


def _validate_sparse_chunk_size(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("sparse_chunk_size must be a positive integer.")
    if value < 1:
        raise ValueError("sparse_chunk_size must be a positive integer.")
    return int(value)


def _validate_path_segment(name: str) -> None:
    if not isinstance(name, str):
        raise TypeError("Table names and component path segments must be strings.")
    if not name or name in _RESERVED_NAMES or any(character in name for character in ("/", "\\", "\x00")):
        raise ValueError(f"Invalid table name or component path segment: {name!r}.")


def _validate_component_paths(components: Sequence[ComponentPath]) -> tuple[ComponentPath, ...]:
    """Validate the logical AnnData address space, not physical Zarr paths."""
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
        if any(path[: len(other)] == other or other[: len(path)] == path for other in paths):
            raise ValueError("Component paths must be unique and non-overlapping.")
        paths.append(path)
    return tuple(paths)


def _open_table_group(store: str | PathLike[str], *, table_name: str) -> zarr.Group:
    """Locate one table read-only, without SpatialData's whole-store reader."""
    _validate_path_segment(table_name)
    if not isinstance(store, (str, PathLike)):
        raise TypeError("store must be a local path to a SpatialData Zarr root.")
    value = fspath(store)
    if not isinstance(value, str):
        raise TypeError("store must be a string path, not bytes.")
    if not value or "://" in value:
        raise ValueError("store must be a local path to a SpatialData Zarr root.")
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"SpatialData store {str(path)!r} does not exist.")
    if not path.is_dir():
        raise ValueError("store must be a SpatialData Zarr root directory.")
    root = zarr.open_group(store=str(path), mode="r", use_consolidated=False)
    if root.attrs.get("encoding-type") in {"anndata", "dict", "raw"}:
        raise ValueError("store must be the SpatialData root, not an AnnData group or component.")
    for kind in ("images", "labels", "points", "shapes"):
        if f"{kind}/{table_name}" in root:
            raise ValueError(f"Table name {table_name!r} collides with a {kind} element.")
    try:
        group = root[f"tables/{table_name}"]
    except KeyError:
        raise FileNotFoundError(f"Table {table_name!r} does not exist in {str(path)!r}.") from None
    if not isinstance(group, zarr.Group):
        raise ValueError(f"Table {table_name!r} must be an AnnData Zarr group.")
    if group.attrs.get("encoding-type") != "anndata" or group.attrs.get("encoding-version") != "0.1.0":
        raise ValueError(f"Unsupported AnnData table encoding at {group.name!r}.")
    return group
