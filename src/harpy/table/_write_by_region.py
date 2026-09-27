"""Regional observation selection and chunked preparation of `.obsm` replacements."""

from __future__ import annotations

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from numbers import Integral, Number
from os import PathLike

import dask.array as da
import numpy as np
import pandas as pd
import zarr
from anndata.abc import CSCDataset, CSRDataset
from dask import delayed
from scipy import sparse

from harpy._storage._anndata import _decode_anndata_element, _read_anndata_element
from harpy.table._io import ComponentPath, _open_table_group, _validate_component_paths
from harpy.table._write import _write_table_operation
from harpy.table._write_validation import (
    _annotation_columns,
    _check_component_destination,
    _match_identity,
    _observation_pairs,
    _read_observation_identity,
    _read_spatialdata_attrs,
    _validate_observation_annotation,
)

_DEFAULT_REGIONAL_CHUNK_SIZE = 1000


def write_table_components_by_region(
    store: str | PathLike[str],
    *,
    table_name: str,
    components: Mapping[ComponentPath, object],
    obs_identity: pd.DataFrame,
    fill_values: Mapping[ComponentPath, object] | None = None,
    chunk_size: int = _DEFAULT_REGIONAL_CHUNK_SIZE,
    overwrite: bool = False,
) -> None:
    """Update `.obsm` measurements for complete regions, preserving other observations.

    Parameters
    ----------
    store
        Local path to an existing SpatialData Zarr root.
    table_name
        Name of an existing SpatialData-annotated table.
    components
        Nonempty mapping containing individual ``("obsm", key)`` matrices and
        optional accompanying ``uns`` replacements. Each matrix contains only
        the selected observations, in the order of ``obs_identity``. Supports
        numeric 2D dense, CSR and CSC matrices, in memory, lazy or Zarr-backed.
        Existing entries require matching formats (dense/dense, CSR/CSR,
        CSC/CSC), unchanged column counts and safe casts into the stored dtype.
        DataFrame-valued matrices and None matrix payloads are rejected.
        Metadata paths replace their entire value, not a regional subset;
        nested None values encode absence, not deletion. Omitted paths are unchanged.
    obs_identity
        Nonempty two-column dataframe using the stored region and instance keys,
        for example ``adata.obs.loc[selected, [region_key, instance_key]]``.
        Region values must be categorical. Actual values select the regions,
        not unused categories. Supply every observation of each selected region
        exactly once, in stored table order, including when regions interleave.
        The dataframe index is ignored. No automatic row reordering occurs.
    fill_values
        For example, ``fill_values={("obsm", "morphology"): np.nan}``
        initializes unselected rows of a newly created ``obsm["morphology"]``
        matrix with NaN. Each supplied fill must be a scalar compatible with
        the matrix's dtype; sparse matrices support only zero fills.

        Required only when a new matrix has unselected rows. Ignored for
        existing entries, whose unselected measurements remain unchanged.
        None means no fills were supplied. Keys must refer to submitted matrices.
    chunk_size
        Positive integer controlling computational chunk shapes::

            dense: (chunk_size rows, all columns)
            CSR:   (chunk_size rows, all columns)
            CSC:   (all rows, chunk_size columns)

        Used for in-memory inputs, Zarr-backed sparse reads, and creating new
        ``.obsm`` entries for only some regions, filling the remaining
        observations with ``fill_values``. Existing Dask input chunks and dense
        Zarr chunks are preserved; existing dense targets retain their chunk
        layout during merging.

        Controls computation, not on-disk chunk sizes or a fixed memory limit.
    overwrite
        Allow updates to existing requested matrix and metadata entries.

    Notes
    -----
    Each affected matrix is rewritten in full, but constructed in chunks without
    collecting the old or merged matrix in memory. Input matrices are not modified.
    Storage backing and chunk layouts may differ; matrix formats must match.
    New entries retain the supplied format and dtype. Sparse matrices are never
    implicitly densified.

    Ensure supplied matrix columns have the same meaning and order as the
    stored columns. Any accompanying scientific metadata must describe the
    resulting data, including measurements retained for unselected regions.

    Observation identities and SpatialData linkage cannot change. All requested
    matrices and metadata share the staging and rollback operation described in
    :func:`harpy.table.write_table_components`. This path-based function returns
    None and does not refresh live SpatialData objects; reopen affected data.

    See Also
    --------
    harpy.table.write_table_components : Replace complete components.
    harpy.table.read_table_components : Read selected stored components.

    Examples
    --------
    .. code-block:: python

        selected = adata.obs[region_key].eq("cells_sample_a")
        # updated_metadata describes the complete resulting matrix, including
        # measurements retained for other regions.
        hp.tb.write_table_components_by_region(
            "sdata.zarr", table_name="cell_features",
            components={
                ("obsm", "morphology"): regional_features,
                ("uns", "feature_matrices", "morphology"): updated_metadata,
            },
            obs_identity=adata.obs.loc[selected, [region_key, instance_key]],
            overwrite=True,
        )
    """
    with _write_table_components_by_region_operation(
        store,
        table_name=table_name,
        components=components,
        obs_identity=obs_identity,
        fill_values=fill_values,
        chunk_size=chunk_size,
        overwrite=overwrite,
    ):
        pass


@contextmanager
def _write_table_components_by_region_operation(
    store: str | PathLike[str],
    *,
    table_name: str,
    components: Mapping[ComponentPath, object],
    obs_identity: pd.DataFrame,
    fill_values: Mapping[ComponentPath, object] | None = None,
    chunk_size: int = _DEFAULT_REGIONAL_CHUNK_SIZE,
    overwrite: bool = False,
) -> Generator[zarr.Group, None, None]:
    """Prepare regional replacements, then yield within the shared rollback window.

    Future SpatialData adapters can install the reopened table in their with-body;
    the shared writer retains responsibility for publication and finalization.
    """
    if not isinstance(components, Mapping):
        raise TypeError("components must be a mapping from tuple paths to values.")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean.")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, Integral):
        raise TypeError("chunk_size must be a positive integer.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer.")
    chunk_size = int(chunk_size)
    paths = _validate_component_paths(tuple(components), to_write=True)
    matrix_paths = {path for path in paths if path[0] == "obsm"}
    if not matrix_paths or any(path[0] not in {"obsm", "uns"} for path in paths):
        raise ValueError("Regional writes require individual obsm matrices and optional uns replacements only.")
    if fill_values is not None and not isinstance(fill_values, Mapping):
        raise TypeError("fill_values must be a mapping from submitted obsm paths to scalar fills.")
    fills = {} if fill_values is None else dict(fill_values)
    if fills.keys() - matrix_paths:
        raise ValueError("fill_values keys must refer to submitted obsm matrices.")

    group = _open_table_group(store, table_name=table_name)
    spatialdata_attrs = _read_spatialdata_attrs(group)
    if spatialdata_attrs is None:
        raise ValueError("Regional writes require a SpatialData-annotated table.")
    region_key, instance_key = _annotation_columns(spatialdata_attrs)
    stored_identity = _read_observation_identity(group, spatialdata_attrs)
    # 1) Validate the stored annotation against all stored observations.
    # Declared regions must match the regions actually present in the table.
    stored_pairs = _validate_observation_annotation(stored_identity, spatialdata_attrs, label="Stored observation")

    # 2) Validate the submitted identities against the selected stored observations.
    # An A-only update must include every observation of A in stored order,
    # but need not include observations from B.
    if not isinstance(obs_identity, pd.DataFrame):
        raise TypeError("obs_identity must be a two-column region/instance dataframe.")
    if obs_identity.empty or len(obs_identity.columns) != 2 or set(obs_identity.columns) != {region_key, instance_key}:
        raise ValueError("obs_identity must be nonempty and contain exactly the stored region and instance columns.")
    supplied_pairs = _observation_pairs(
        obs_identity, region_key=region_key, instance_key=instance_key, label="obs_identity"
    )
    regions = set(obs_identity[region_key].unique())
    unknown = regions - set(stored_identity[region_key].unique())
    if unknown:
        raise ValueError(f"Unknown regions in obs_identity: {sorted(unknown, key=str)!r}.")
    selected = stored_identity[region_key].isin(regions).to_numpy()
    _match_identity(supplied_pairs, stored_pairs[selected], label="obs_identity")
    selected_rows = np.flatnonzero(selected)

    replacements = dict(components)
    for path in paths:
        _check_component_destination(group, path, overwrite=overwrite)
        if path[0] != "obsm":
            continue
        supplied = components[path]
        matrix_format = _matrix_format(supplied, label=f"Component {path!r}")
        if supplied.shape[0] != len(selected_rows):
            raise ValueError(f"Component {path!r} must contain {len(selected_rows)} selected rows.")
        existing = None
        if "/".join(path) in group:
            element = group["/".join(path)]
            # Dataframe decoding is always eager. Reject its encoding before
            # opening the value, not after loading an unrelated full frame.
            if element.attrs.get("encoding-type") == "dataframe":
                raise TypeError("Regional writes do not support DataFrame-valued obsm entries.")
            existing = _read_anndata_element(group, path, mode="lazy", sparse_chunk_size=chunk_size)
            if _matrix_format(existing, label=f"Stored component {path!r}") != matrix_format:
                raise ValueError(f"Component {path!r} must match the stored matrix format (dense, CSR or CSC).")
            if existing.shape != (len(stored_identity), supplied.shape[1]):
                raise ValueError(f"Component {path!r} must preserve the stored matrix shape and column count.")
            if not np.can_cast(supplied.dtype, existing.dtype, casting="safe"):
                raise ValueError(f"Component {path!r} cannot be safely cast to stored dtype {existing.dtype}.")

        fill = None
        if existing is None:
            if len(selected_rows) != len(stored_identity) and path not in fills:
                raise ValueError(f"New component {path!r} requires a fill for unselected rows.")
            if path in fills:
                fill = _scalar_fill(fills[path], supplied.dtype)
                if matrix_format != "dense" and len(selected_rows) != len(stored_identity) and fill != 0:
                    raise ValueError("New sparse matrices with unselected rows require a zero fill.")
        replacements[path] = _regional_matrix(
            _lazy_matrix(supplied, matrix_format, chunk_size=chunk_size),
            existing=existing,
            selected_rows=selected_rows,
            n_obs=len(stored_identity),
            matrix_format=matrix_format,
            fill=fill,
            chunk_size=chunk_size,
        )

    # The prepared matrices now cover the full table axis. Delegate both rounds
    # of structural validation, serialization and publication to the ordinary
    # component writer, using full identities rather than the regional subset.
    with _write_table_operation(
        store, table_name=table_name, components=replacements, obs_identity=stored_identity, overwrite=overwrite
    ) as published:
        yield published


def _matrix_format(value: object, *, label: str) -> str:
    """Inspect supported numeric matrix metadata without computing or decoding values."""
    meta = value._meta if isinstance(value, da.Array) else value
    if isinstance(meta, (np.ndarray, zarr.Array)):
        matrix_format = "dense"
    elif sparse.issparse(meta) and meta.format in {"csr", "csc"}:
        matrix_format = meta.format
    elif isinstance(meta, (CSRDataset, CSCDataset)) and isinstance(meta.group, zarr.Group):
        matrix_format = meta.format
    else:
        raise TypeError(f"{label} must be a dense, CSR or CSC matrix; None and DataFrames are not supported.")
    shape = value.shape
    if len(shape) != 2 or any(not isinstance(size, Integral) or size < 0 for size in shape):
        raise ValueError(f"{label} requires a known two-dimensional shape.")
    if np.dtype(value.dtype).kind not in "biufc":
        raise TypeError(f"{label} must have a numeric dtype.")
    if isinstance(value, da.Array) and any(
        not isinstance(size, Integral) for chunks in value.chunks for size in chunks
    ):
        raise ValueError(f"{label} requires known chunk sizes.")
    return matrix_format


def _scalar_fill(value: object, dtype: np.dtype) -> object:
    """Validate a scalar fill without widening the new matrix's dtype."""
    if not isinstance(value, Number) or not np.can_cast(np.min_scalar_type(value), dtype, casting="safe"):
        raise ValueError(f"Fill must be a numeric scalar compatible with dtype {dtype}.")
    return np.dtype(dtype).type(value)


def _lazy_matrix(value: object, matrix_format: str, *, chunk_size: int) -> da.Array:
    if isinstance(value, da.Array):
        return value
    if isinstance(value, (CSRDataset, CSCDataset)):
        return _decode_anndata_element(value.group, mode="lazy", sparse_chunk_size=chunk_size)
    if isinstance(value, zarr.Array):
        return da.from_zarr(value)
    chunks = {
        "dense": (chunk_size, -1),
        "csr": (chunk_size, -1),
        "csc": (-1, chunk_size),
    }[matrix_format]
    return da.from_array(value, chunks=chunks, asarray=False)


def _regional_matrix(
    supplied: da.Array,
    *,
    existing: da.Array | None,
    selected_rows: np.ndarray,
    n_obs: int,
    matrix_format: str,
    fill: object,
    chunk_size: int,
) -> da.Array:
    """Build full-axis replacements from independently chunked old and selected rows.

    Each task reads only its destination rectangle and the corresponding slice
    of the submitted rows. The rank within selected_rows, not the source block
    number, determines that slice. Sparse output chunks span the uncompressed
    axis, as required by AnnData's sparse writer.
    """
    dtype = supplied.dtype if existing is None else existing.dtype
    if len(selected_rows) == n_obs:
        return supplied.astype(dtype)
    shape = (n_obs, supplied.shape[1])
    if existing is not None:
        chunks = existing.chunks
    elif matrix_format == "csc":
        chunks = (-1, chunk_size)
    else:
        chunks = (chunk_size, -1)
    chunks = list(da.core.normalize_chunks(chunks, shape=shape, dtype=dtype))
    if matrix_format == "csr":
        chunks[1] = (shape[1],)
    elif matrix_format == "csc":
        chunks[0] = (shape[0],)
    meta = (
        np.empty((0, 0), dtype=dtype)
        if matrix_format == "dense"
        else getattr(sparse, f"{matrix_format}_matrix")((0, 0), dtype=dtype)
    )
    rows = []
    row_start = 0
    for height in chunks[0]:
        row_stop = row_start + height
        start, stop = np.searchsorted(selected_rows, [row_start, row_stop])
        local_rows = selected_rows[start:stop] - row_start
        columns = []
        col_start = 0
        for width in chunks[1]:
            col_stop = col_start + width
            original = None if existing is None else existing[row_start:row_stop, col_start:col_stop]
            updates = supplied[start:stop, col_start:col_stop] if stop > start else None
            block = delayed(_merge_regional_block)(
                original, updates, local_rows, (height, width), dtype, matrix_format, fill
            )
            columns.append(da.from_delayed(block, shape=(height, width), dtype=dtype, meta=meta))
            col_start = col_stop
        rows.append(da.concatenate(columns, axis=1))
        row_start = row_stop
    return da.concatenate(rows, axis=0)


def _merge_regional_block(original, updates, selected_rows, shape, dtype, matrix_format, fill):
    """Replace selected local rows without modifying input blocks or densifying sparse data."""
    if matrix_format == "dense":
        result = np.full(shape, fill, dtype=dtype) if original is None else original.copy()
        if updates is not None:
            result[selected_rows] = updates
        return result
    # Keep only unselected old entries and remap incoming sparse row indices.
    # No assignment into shared sparse buffers (or dense temporary) is needed.
    keep_rows = np.ones(shape[0], dtype=bool)
    keep_rows[selected_rows] = False
    old = sparse.coo_matrix(shape, dtype=dtype) if original is None else original.tocoo()
    keep = keep_rows[old.row]
    new = sparse.coo_matrix((0, shape[1]), dtype=dtype) if updates is None else updates.tocoo()
    result = sparse.coo_matrix(
        (
            np.concatenate((old.data[keep], new.data)).astype(dtype, copy=False),
            (np.concatenate((old.row[keep], selected_rows[new.row])), np.concatenate((old.col[keep], new.col))),
        ),
        shape=shape,
    )
    return result.asformat(matrix_format)
