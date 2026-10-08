"""SpatialData reading with explicit table selection and matrix representations."""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from pathlib import Path

import zarr
from spatialdata import SpatialData
from spatialdata import read_zarr as read_spatialdata_zarr

from harpy._storage._anndata import (
    _DenseChunks,
    _ReadMode,
    _SparseChunks,
    _validate_dense_chunks,
    _validate_sparse_chunks,
)
from harpy._storage._spatialdata import _open_spatialdata_group
from harpy.table.io import read_table
from harpy.table.io._read import _validate_path_segment, _validate_read_mode


def read_zarr(
    store: str | PathLike[str],
    *,
    table_name: str | Sequence[str] | None = None,
    table_mode: _ReadMode = "lazy",
    sparse_chunks: _SparseChunks = "auto",
    dense_chunks: _DenseChunks = "auto",
) -> SpatialData:
    """Read a SpatialData store with selected tables in the requested matrix mode.

    Parameters
    ----------
    store
        Local path to an existing SpatialData Zarr root.
    table_name
        Exact table name or sequence of names. None reads all tables; an empty
        sequence skips tables. Names must be unique; missing names raise
        FileNotFoundError. All non-table elements are read through SpatialData.
    table_mode
        Table matrix representation: ``"lazy"`` returns Dask arrays,
        ``"backed"`` returns read-only Zarr arrays or CSR/CSC dataset handles,
        and ``"eager"`` returns in-memory NumPy/SciPy matrices. Annotations,
        dataframe-valued entries and ``uns`` are always in memory.
        Does not control how images, labels, points or shapes are read.
    sparse_chunks, dense_chunks
        Lazy block layout of sparse and dense table matrices, as described for
        :func:`harpy.table.io.read_table`. Ignored for other table modes.

    Returns
    -------
    SpatialData
        Object with the selected tables, non-table elements, transformations,
        root attributes and backing-store path.

    Notes
    -----
    - Reading does not modify the store. Editing returned annotations or
      replacing matrices changes only the returned object, not the stored data.
    - Backed table handles are read-only: values cannot be written directly
      through these handles.
    - Lazy matrices and backed handles depend on their source data. If that
      data is overwritten, call ``read_zarr`` again and use the new result.

    ``sdata.is_backed()`` indicates a backing-store path. The tables still have
    ``adata.isbacked=False``: AnnData's flag describes its own file-managed
    mode, not whether individual matrices are lazy or storage-backed.

    See Also
    --------
    harpy.table.io.read_table : Read one table without opening other elements.
    harpy.table.io.read_table_components : Read selected AnnData components.
    spatialdata.read_zarr : SpatialData's reader with element-type selection.

    Examples
    --------
    .. code-block:: python

        sdata = hp.io.read_zarr("sdata.zarr", table_name=["raw_counts", "processed"])
        backed = hp.io.read_zarr("sdata.zarr", table_name="raw_counts", table_mode="backed")
        spatial_only = hp.io.read_zarr("sdata.zarr", table_name=[])
    """
    _validate_read_mode(table_mode)
    sparse_chunks = _validate_sparse_chunks(sparse_chunks)
    dense_chunks = _validate_dense_chunks(dense_chunks)
    if table_name is not None:
        if isinstance(table_name, str):
            table_name = (table_name,)
        elif isinstance(table_name, bytes) or not isinstance(table_name, Sequence):
            raise TypeError("table_name must be a table name, a sequence of names, or None.")
        else:
            table_name = tuple(table_name)
        for name in table_name:
            _validate_path_segment(name)
        if len(set(table_name)) != len(table_name):
            raise ValueError("table_name must not contain duplicate names.")

    root = _open_spatialdata_group(store)
    # Explicit selections need not inspect any unselected table groups.
    if table_name is None:
        if "tables" not in root:
            table_name = ()
        else:
            table_group = root["tables"]
            if not isinstance(table_group, zarr.Group):
                raise ValueError("The SpatialData tables entry must be a Zarr group.")
            table_name = tuple(sorted(table_group.keys()))
    for name in table_name:
        if f"tables/{name}" not in root:
            raise FileNotFoundError(f"Table {name!r} does not exist in {str(store)!r}.")

    # SpatialData reads non-table elements; Harpy reads the selected tables
    # below using the requested table_mode.
    sdata = read_spatialdata_zarr(Path(store), selection=("images", "labels", "points", "shapes"))
    for name in table_name:
        sdata.tables[name] = read_table(
            store, table_name=name, mode=table_mode, sparse_chunks=sparse_chunks, dense_chunks=dense_chunks
        )
    return sdata
