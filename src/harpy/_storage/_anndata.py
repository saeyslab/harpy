"""Serialization and storage-backed reading for AnnData stored in Zarr.

These helpers own AnnData encodings and SpatialData's on-disk table format
metadata. They do not publish replacements or perform rollback; table writers
coordinate these operations through ``harpy._storage._publication``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import dask.array as da
import numpy as np
import zarr
from anndata import AnnData
from anndata.experimental import read_elem_lazy
from anndata.io import read_elem, sparse_dataset, write_elem
from dask import delayed
from scipy import sparse
from spatialdata.models import TableModel

# Harpy owns these literals as the compatibility boundary for tables assembled
# with the low-level AnnData writers, without depending on private SpatialData
# writer APIs.
_SPATIALDATA_TABLE_ENCODING_TYPE = "ngff:regions_table"
_SPATIALDATA_TABLE_FORMAT_VERSION = "0.2"
_ReadMode = Literal["backed", "lazy", "eager"]
_MATRIX_MAPPINGS = ("layers", "obsm", "varm", "obsp", "varp")
_DEFAULT_SPARSE_CHUNK_SIZE = 1000


class _MissingAnnDataElement(KeyError):
    """A logical path is absent, as distinct from an error decoding its value."""


def _read_backed_table(group: zarr.Group) -> AnnData:
    """Reopen a table with Zarr arrays and AnnData sparse-dataset handles.

    Internal installation callers retain these handles rather than Dask
    arrays. Use the permanent table path: matrices depend on that location.
    """
    return _read_anndata_table(group, mode="backed")


def _read_anndata_table(
    group: zarr.Group, *, mode: _ReadMode, sparse_chunk_size: int = _DEFAULT_SPARSE_CHUNK_SIZE
) -> AnnData:
    """Read a complete AnnData table using the component reader ``_read_anndata_element``.

    ``obs`` and ``var`` are always read eagerly as pandas DataFrames; ``uns``
    is also loaded into memory, including any arrays it contains.

    ``X`` and array-valued entries in ``layers``, ``obsm``, ``varm``, ``obsp``
    and ``varp`` follow ``mode``: ``lazy`` builds Dask arrays without reading
    matrix values, ``backed`` retains Zarr arrays or sparse-dataset handles,
    and ``eager`` loads them into memory. DataFrame-valued entries are always
    eager. Unsupported lazy matrix encodings raise rather than falling back
    to an eager read.

    When present, ``raw`` follows the same policy: its ``var`` is eager,
    while its ``X`` and ``varm`` follow the requested matrix-reading mode.
    """
    if group.attrs.get("encoding-type") != "anndata" or group.attrs.get("encoding-version") != "0.1.0":
        raise ValueError(f"Unsupported AnnData table encoding at {group.name!r}.")
    values = {slot: _read_anndata_element(group, (slot,), mode="eager") for slot in ("obs", "var")}
    values.update(
        {
            slot: _read_anndata_element(group, (slot,), mode=mode, sparse_chunk_size=sparse_chunk_size)
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
                slot: _read_anndata_element(group, ("raw", slot), mode=mode, sparse_chunk_size=sparse_chunk_size)
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
    element: zarr.Array | zarr.Group, *, mode: _ReadMode, sparse_chunk_size: int = _DEFAULT_SPARSE_CHUNK_SIZE
) -> object:
    """Decode an AnnData element eagerly, lazily, or as a storage-backed handle.

    Mappings are decoded entry by entry using the same mode and sparse chunk size.

    Raises
    ------
    ValueError
        If a CSR/CSC element has an unsupported or missing encoding version,
        regardless of the requested mode, or lazy/backed mode does not support
        the element's encoding or version.
    """
    encoding = element.attrs.get("encoding-type")
    version = element.attrs.get("encoding-version")
    if encoding in {"csr_matrix", "csc_matrix"}:
        # This is the on-disk sparse format, not the AnnData package version.
        # Unknown formats must not bypass Harpy's sparse chunking policy.
        if version != "0.1.0":
            raise ValueError(f"Unsupported {encoding} encoding version {version!r}; expected '0.1.0'.")
    # Preserve the requested reading mode and sparse chunk size for matrices
    # inside mappings, including nested mappings.
    if encoding == "dict" and version == "0.1.0":
        return {
            key: _decode_anndata_element(element[key], mode=mode, sparse_chunk_size=sparse_chunk_size)
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
        # Harpy owns the sparse default, independently of AnnData's defaults.
        # Keep the other axis whole; dense arrays retain their on-disk chunks.
        chunks = (sparse_chunk_size, -1) if compressed_axis == 0 else (-1, sparse_chunk_size)
        return read_elem_lazy(element, chunks=chunks)
    return read_elem_lazy(element)


def _write_anndata_element(
    group: zarr.Group,
    path: tuple[str, ...],
    value: object,
    *,
    create_parents: bool = False,
) -> None:
    """Write one logical AnnData path through AnnData's encoding registry.

    Parent groups normally must already exist. ``create_parents=True`` creates
    missing parents as AnnData-encoded mappings, which is useful for building a
    partial hierarchy in an isolated staging store. Keeping path traversal here
    gives full table writers and partial component writers the same encoding
    boundary; safe replacement is provided separately by
    :func:`harpy._storage._publication._publish_staged_paths`.
    """
    parent, key = _resolve_anndata_parent(group, path, create_parents=create_parents)
    write_elem(parent, key, value)


def _read_anndata_element(
    group: zarr.Group,
    component_path: tuple[str, ...],
    *,
    mode: _ReadMode = "backed",
    sparse_chunk_size: int = _DEFAULT_SPARSE_CHUNK_SIZE,
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
    sparse_chunk_size
        Rows per CSR chunk or columns per CSC chunk for lazy sparse reads.
        Ignored for dense arrays and other modes.

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
    return _decode_anndata_element(element, mode=mode, sparse_chunk_size=sparse_chunk_size)


def _resolve_anndata_parent(
    group: zarr.Group,
    path: tuple[str, ...],
    *,
    create_parents: bool,
) -> tuple[zarr.Group, str]:
    """Resolve the existing parent group and leaf key of one AnnData path."""
    if not path or any(not isinstance(key, str) or not key for key in path):
        raise ValueError(f"AnnData element path must contain non-empty string keys, found {path!r}.")
    parent = group
    for key in path[:-1]:
        if key not in parent and create_parents:
            write_elem(parent, key, {})
        if key not in parent or not isinstance(parent[key], zarr.Group):
            raise ValueError(f"AnnData element parent path {path[:-1]!r} does not exist as a Zarr group.")
        parent = parent[key]
    return parent, path[-1]


def _write_spatialdata_table_attrs(
    group: zarr.Group,
    *,
    regions: Sequence[str],
    region_key: str,
    instance_key: str,
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
        Labels elements annotated by the table.
    region_key
        Column in ``adata.obs`` that identifies the labels element.
    instance_key
        Column in ``adata.obs`` that identifies an instance within that labels
        element.
    """
    group.attrs["spatialdata-encoding-type"] = _SPATIALDATA_TABLE_ENCODING_TYPE
    group.attrs["region"] = list(regions)
    group.attrs["region_key"] = region_key
    group.attrs["instance_key"] = instance_key
    group.attrs["version"] = _SPATIALDATA_TABLE_FORMAT_VERSION
