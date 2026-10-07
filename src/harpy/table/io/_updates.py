"""Decide which components of a table changed against its store, for writing back only those.

``_component_changed`` decides, for one value of an AnnData table whose path
exists in the store, whether it differs from the stored element.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import zarr
from anndata.abc import CSCDataset, CSRDataset
from scipy import sparse

from harpy._storage._anndata import (
    _decode_anndata_element,
    _element_identity,
    _lazy_read_source,
    _read_anndata_element,
)
from harpy.table.io._read import ComponentPath


def _component_changed(group: zarr.Group, path: ComponentPath, value: object) -> bool:
    """Return whether a value of an AnnData table differs from its stored element.

    The element at ``path`` must exist in the table ``group``; a path that does
    not exist is new, which the caller decides before calling this. The rules:

    - ``uns`` entries, per top-level key, are compared by value
      (``_same_uns_value``);
    - dataframes, ``obs``, ``var``, ``raw.var`` and DataFrame-valued ``obsm`` or
      ``varm`` entries, are compared strictly (``_same_dataframe``);
    - a Dask array is unchanged only if it is a registered lazy read of exactly
      this element (``_lazy_read_source``); any other Dask array, derived or a
      read of another element, is changed, without computing it;
    - a backed handle (a ``zarr.Array`` or a CSR/CSC dataset) is unchanged only
      if it points to exactly this element, wherever its store was opened;
    - an in-memory NumPy or SciPy matrix is compared with the stored values,
      metadata first, then block by block (``_matrix_equals_stored``);
    - any other value counts as changed, and the writer validates it.

    Elements are identified by their resolved path on disk
    (``_element_identity``). Only a ``LocalStore`` gives elements one, so
    registered reads and backed handles of elements in other stores count as
    changed.
    """
    if path[0] == "uns":
        return not _same_uns_value(value, _read_anndata_element(group, path, mode="eager"))
    element = group["/".join(path)]
    stored_dataframe = element.attrs.get("encoding-type") == "dataframe"
    if isinstance(value, pd.DataFrame) or stored_dataframe:
        # A dataframe replacing a matrix, or the reverse, is changed without reading the stored value.
        if not (isinstance(value, pd.DataFrame) and stored_dataframe):
            return True
        return not _same_dataframe(value, _read_anndata_element(group, path, mode="eager"))
    identity = _element_identity(element)
    if isinstance(value, da.Array):
        return identity is None or _lazy_read_source(value) != identity
    if isinstance(value, (zarr.Array, CSRDataset, CSCDataset)):
        return identity is None or _backed_identity(value) != identity
    return not _matrix_equals_stored(value, element)


def _backed_identity(value: zarr.Array | CSRDataset | CSCDataset) -> Path | None:
    """Return the identity of the stored element a backed handle points to."""
    return _element_identity(value.group if isinstance(value, (CSRDataset, CSCDataset)) else value)


def _same_dataframe(value: pd.DataFrame, stored: pd.DataFrame) -> bool:
    """Compare two dataframes strictly.

    The same index, the same columns in the same order, the same dtypes,
    including categorical categories and their order, and the same values, with
    NaN equal to NaN. Anything else differs: a false "changed" only rewrites a
    small dataframe, while a false "unchanged" would lose data.
    """
    try:
        pd.testing.assert_frame_equal(
            value, stored, check_exact=True, check_index_type=True, check_column_type=True, check_like=False
        )
    except (AssertionError, TypeError, ValueError):
        return False
    return True


def _same_uns_value(value: object, stored: object) -> bool:
    """Compare one ``uns`` value with its stored value, by value.

    Mappings are compared key by key, recursively; dataframes with
    ``_same_dataframe``; everything else as arrays with ``np.array_equal``,
    with NaN equal to NaN for floating point values, so a list equals the
    array AnnData stores it as. A value whose comparison raises or is ambiguous
    counts as different. Separate from ``_same_metadata``, which also guards
    ``uns["spatialdata_attrs"]`` and must keep its own semantics.
    """
    if isinstance(value, Mapping) or isinstance(stored, Mapping):
        if not (isinstance(value, Mapping) and isinstance(stored, Mapping)):
            return False
        return value.keys() == stored.keys() and all(_same_uns_value(value[key], stored[key]) for key in value)
    if isinstance(value, pd.DataFrame) or isinstance(stored, pd.DataFrame):
        return isinstance(value, pd.DataFrame) and isinstance(stored, pd.DataFrame) and _same_dataframe(value, stored)
    if value is None or stored is None:
        return value is None and stored is None
    try:
        left, right = np.asarray(value), np.asarray(stored)
        if left.shape != right.shape:
            return False
        return bool(np.array_equal(left, right, equal_nan=_floating(left) and _floating(right)))
    except (TypeError, ValueError):
        return False


def _matrix_equals_stored(value: object, element: zarr.Array | zarr.Group) -> bool:
    """Compare an in-memory matrix with its stored element: metadata first, then block by block.

    1. Metadata, without reading values: a different encoding (dense, CSR,
       CSC), shape or dtype differs, and so does, for a sparse matrix, a
       different number of stored values (the length of ``data``). This is
       conservative: a sparse matrix that differs only in explicit zeros has
       another number of stored values.
    2. Values: the stored element is read lazily with the readers, in their
       ``"auto"`` layout, whose blocks lie along one axis: rows for dense and
       CSR matrices, columns for CSC matrices. Each block is computed and
       compared with the matching slice of ``value`` in turn, stopping at the
       first difference. Memory stays at ``value`` plus about one block.

    Values other than a numeric NumPy array or a CSR/CSC matrix, string arrays
    for example, are not compared and count as different.
    """
    encoding = element.attrs.get("encoding-type")
    if isinstance(value, np.ndarray):
        if encoding != "array" or tuple(element.shape) != value.shape or element.dtype != value.dtype:
            return False
        axis = 0
    elif sparse.issparse(value) and value.format in {"csr", "csc"}:
        if (
            encoding != f"{value.format}_matrix"
            or tuple(element.attrs["shape"]) != value.shape
            or element["data"].dtype != value.dtype
            or element["data"].shape[0] != value.nnz
        ):
            return False
        axis = 0 if value.format == "csr" else 1
    else:
        return False
    stored = _decode_anndata_element(element, mode="lazy")
    start = 0
    for block_index, length in enumerate(stored.chunks[axis]):
        if axis == 0:
            value_block, stored_block = value[start : start + length], stored.blocks[block_index]
        else:
            value_block, stored_block = value[:, start : start + length], stored.blocks[:, block_index]
        if not _same_block(value_block, stored_block.compute()):
            return False
        start += length
    return True


def _same_block(value_block: object, stored_block: object) -> bool:
    """Compare one block of an in-memory matrix with the matching stored block.

    Dense blocks are compared directly. Sparse blocks are compared in canonical
    form (sorted indices, summed duplicates), so that two blocks with the same
    stored values in a different order are equal; a block whose structure
    differs, for example only in explicit zeros, differs. NaN equals NaN.
    """
    if sparse.issparse(value_block):
        if not sparse.issparse(stored_block) or value_block.format != stored_block.format:
            return False
        left, right = value_block.copy(), stored_block.copy()
        left.sum_duplicates()
        right.sum_duplicates()
        return (
            left.shape == right.shape
            and np.array_equal(left.indptr, right.indptr)
            and np.array_equal(left.indices, right.indices)
            and np.array_equal(left.data, right.data, equal_nan=_floating(left.data))
        )
    left, right = np.asarray(value_block), np.asarray(stored_block)
    return left.shape == right.shape and bool(np.array_equal(left, right, equal_nan=_floating(left)))


def _floating(array: np.ndarray) -> bool:
    """Return whether NaN can occur in ``array``, so that ``equal_nan`` applies."""
    return array.dtype.kind in "fc"
