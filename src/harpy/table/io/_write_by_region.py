"""Regional observation selection and chunked preparation of `.obsm` replacements."""

from __future__ import annotations

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from numbers import Integral, Number
from os import PathLike
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import zarr
from anndata.abc import CSCDataset, CSRDataset
from dask import delayed
from numpy.typing import NDArray
from scipy import sparse

from harpy._storage._anndata import _decode_anndata_element, _read_anndata_element
from harpy.table.io._read import ComponentPath, _open_table_group, _validate_component_paths
from harpy.table.io._write import _write_table_operation
from harpy.table.io._write_validation import (
    _annotation_columns,
    _check_component_write_destination,
    _match_identity,
    _observation_pairs,
    _read_observation_identity,
    _read_spatialdata_attrs,
    _validated_observation_pairs,
)

type _MatrixBlock = np.ndarray | sparse.csr_matrix | sparse.csc_matrix | sparse.csr_array | sparse.csc_array

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
        Mapping containing one or more ``("obsm", key)`` matrices, for example
        ``{("obsm", "morphology"): regional_features}``. Each matrix contains
        only the observations identified by ``obs_identity``, with matching
        row counts and row order.

        Matrices must be numeric and two-dimensional: dense, CSR or CSC,
        supplied in memory, as lazy arrays, or as Zarr-backed handles.
        DataFrame-valued matrices and ``None`` matrix values are not supported.

        When updating an existing matrix, preserve its format and column count.
        Supplied values must be safely castable to its stored dtype.
        No conversion between dense, CSR and CSC formats occurs.

        Optional ``uns`` entries may accompany the matrices. Each supplied
        metadata path replaces its entire value, not a regional subset.
        ``None`` metadata values are stored, not treated as deletion.
        Omitted component paths remain unchanged.
    obs_identity
        Nonempty two-column dataframe identifying the rows of every ``.obsm``
        matrix in ``components``, in the same order. Use the stored region and
        instance keys, for example
        ``adata.obs.loc[selected, [region_key, instance_key]]``.

        May cover a subset of the table, but must include every observation of
        each selected region exactly once, in stored table order, including
        when regions interleave. No automatic row reordering occurs.

        The region column must be categorical. Actual values select the regions,
        not unused categories. The dataframe index is ignored.
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
        ``.obsm`` entries for only some regions (filling the remaining
        observations with ``fill_values``). Input preparation preserves existing
        Dask and dense Zarr chunks; merging may lazily repartition a working view
        without changing supplied arrays. Existing dense targets retain their
        chunk layout during merging.

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
    harpy.table.add_table_components_by_region : Also update the attached SpatialData table.

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

    SpatialData adapters install requested reopened components in their with-body;
    the shared writer retains responsibility for publication and finalization.
    """
    paths = _validate_regional_request(components, fill_values=fill_values, chunk_size=chunk_size, overwrite=overwrite)
    chunk_size = int(chunk_size)
    group = _open_table_group(store, table_name=table_name)
    root = Path(store)
    table_path = root / "tables" / table_name
    spatialdata_attrs = _read_spatialdata_attrs(group)
    if spatialdata_attrs is None:
        raise ValueError("Regional writes require a SpatialData-annotated table.")
    stored_identity = _read_observation_identity(group, spatialdata_attrs)
    table_row_positions = _regional_row_positions(stored_identity, spatialdata_attrs, obs_identity)

    replacements = dict(components)
    for path in paths:
        _check_component_write_destination(group, path, table_path=table_path, root=root, overwrite=overwrite)
        if path[0] != "obsm":
            continue
        existing = None
        if "/".join(path) in group:
            element = group["/".join(path)]
            # Dataframe decoding is always eager. Reject its encoding before
            # opening the value, not after loading an unrelated full frame.
            if element.attrs.get("encoding-type") == "dataframe":
                raise TypeError("Regional writes do not support DataFrame-valued obsm entries.")
            # Keep the stored dense chunk layout during merging, as documented,
            # rather than the readers' row-only default (until slice 1d revisits it).
            existing = _read_anndata_element(group, path, mode="lazy", sparse_chunks=chunk_size, dense_chunks="storage")
            # A stored null is an invalid matrix, not an absent entry.
            _matrix_format(existing, label=f"Stored component {path!r}")
        replacements[path] = _prepare_regional_matrix(
            path,
            components[path],
            existing=existing,
            table_row_positions=table_row_positions,
            n_obs=len(stored_identity),
            fill_values=fill_values,
            chunk_size=chunk_size,
        )

    # The prepared matrices now cover the full table axis. Delegate both rounds
    # of structural validation, serialization and publication to the ordinary
    # component writer, using full identities rather than the regional subset.
    with _write_table_operation(
        store, table_name=table_name, components=replacements, obs_identity=stored_identity, overwrite=overwrite
    ) as published:
        yield published


def _validate_regional_request(
    components: Mapping[ComponentPath, object],
    *,
    fill_values: Mapping[ComponentPath, object] | None,
    chunk_size: int,
    overwrite: bool,
) -> tuple[ComponentPath, ...]:
    """Validate regional scopes and options identically for disk and memory updates."""
    if not isinstance(components, Mapping):
        raise TypeError("components must be a mapping from tuple paths to values.")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean.")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, Integral):
        raise TypeError("chunk_size must be a positive integer.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer.")
    paths = _validate_component_paths(tuple(components), to_write=True)
    matrix_paths = {path for path in paths if path[0] == "obsm"}
    if not matrix_paths or any(path[0] not in {"obsm", "uns"} for path in paths):
        raise ValueError("Regional writes require individual obsm matrices and optional uns replacements only.")
    if fill_values is not None and not isinstance(fill_values, Mapping):
        raise TypeError("fill_values must be a mapping from submitted obsm paths to scalar fills.")
    fills = {} if fill_values is None else dict(fill_values)
    if fills.keys() - matrix_paths:
        raise ValueError("fill_values keys must refer to submitted obsm matrices.")
    return paths


def _regional_row_positions(
    destination_obs: pd.DataFrame, spatialdata_attrs: Mapping, obs_identity: pd.DataFrame
) -> NDArray[np.intp]:
    """Match complete selected regions to zero-based rows of the destination table.

    destination_obs describes the full stored or attached observation axis;
    it must contain the identity columns and may include other annotations.
    obs_identity describes only the submitted matrix rows, in destination order.
    Its dataframe index is ignored; selection and alignment use region/instance pairs.
    """
    if not isinstance(spatialdata_attrs, Mapping):
        raise ValueError("Regional writes require a SpatialData-annotated table.")
    region_key, instance_key = _annotation_columns(spatialdata_attrs)
    # 1) Validate the destination annotation against all its observations.
    # Declared regions must match the regions actually present in the table.
    destination_pairs = _validated_observation_pairs(
        destination_obs, spatialdata_attrs, label="Destination observation"
    )

    # 2) Validate submitted identities against the selected destination observations.
    # An A-only update must include every observation of A in destination order,
    # but need not include observations from B.
    if not isinstance(obs_identity, pd.DataFrame):
        raise TypeError("obs_identity must be a two-column region/instance dataframe.")
    if obs_identity.empty or len(obs_identity.columns) != 2 or set(obs_identity.columns) != {region_key, instance_key}:
        raise ValueError(
            "obs_identity must be nonempty and contain exactly the destination region and instance columns."
        )
    supplied_pairs = _observation_pairs(
        obs_identity, region_key=region_key, instance_key=instance_key, label="obs_identity"
    )
    regions = set(obs_identity[region_key].unique())
    unknown = regions - set(destination_obs[region_key].unique())
    if unknown:
        raise ValueError(f"Unknown regions in obs_identity: {sorted(unknown, key=str)!r}.")
    selected_row_mask = destination_obs[region_key].isin(regions).to_numpy()
    _match_identity(supplied_pairs, destination_pairs[selected_row_mask], label="obs_identity")
    return np.flatnonzero(selected_row_mask)


def _prepare_regional_matrix(
    path: ComponentPath,
    regional_values: object,
    *,
    existing: object | None,
    table_row_positions: NDArray[np.intp],
    n_obs: int,
    fill_values: Mapping[ComponentPath, object] | None,
    chunk_size: int,
) -> da.Array:
    """Validate matrix compatibility and prepare a lazy full-observation replacement.

    Parameters
    ----------
    path
        Submitted obsm path, used to resolve its fill and identify validation errors.
    regional_values
        Measurements for selected observations only, in destination-table order.
    existing
        Destination matrix to retain outside selected rows, or None for a new entry.
        Comes from storage for disk updates and from the attached table otherwise.
    table_row_positions
        Full-table destination of each row in regional_values, already validated.
    n_obs
        Number of observations in the complete destination table.
    fill_values, chunk_size
        Validated options from the public regional-update APIs.
    """
    matrix_format = _matrix_format(regional_values, label=f"Component {path!r}")
    if regional_values.shape[0] != len(table_row_positions):
        raise ValueError(f"Component {path!r} must contain {len(table_row_positions)} selected rows.")
    if existing is not None:
        if _matrix_format(existing, label=f"Existing component {path!r}") != matrix_format:
            raise ValueError(f"Component {path!r} must match the destination matrix format (dense, CSR or CSC).")
        if existing.shape != (n_obs, regional_values.shape[1]):
            raise ValueError(f"Component {path!r} must preserve the destination matrix shape and column count.")
        if not np.can_cast(regional_values.dtype, existing.dtype, casting="safe"):
            raise ValueError(f"Component {path!r} cannot be safely cast to destination dtype {existing.dtype}.")
        existing = _lazy_matrix(existing, matrix_format, chunk_size=chunk_size)
        # Backed updates read existing from storage with merge-compatible sparse chunks.
        # Unbacked updates take existing from table.obsm; if it is already a Dask
        # array, _lazy_matrix() preserves its chunks, which may split both axes.
        # _regional_matrix() requires CSR blocks to span all columns, or CSC blocks
        # to span all rows. Therefore, we prepare a compatible lazy working array
        # when needed, without changing the original matrix's chunks:
        # CSR ((3, 3), (2, 2)) -> ((3, 3), (4,)); CSC keeps columns and joins rows.
        if len(table_row_positions) != n_obs and matrix_format != "dense":
            whole_axis = 1 if matrix_format == "csr" else 0
            if len(existing.chunks[whole_axis]) > 1:
                if existing.shape[1] == 0:
                    # In Dask 2026.7.1, _compute_rechunk() returns a dense empty
                    # array for zero-sized inputs, losing their sparse representation.
                    # An empty feature axis has no measurements to retain: prepare
                    # its full-row CSC block lazily without that format conversion.
                    existing = da.from_delayed(
                        delayed(sparse.csc_matrix)(existing.shape, dtype=existing.dtype),
                        shape=existing.shape,
                        dtype=existing.dtype,
                        meta=existing._meta,
                    )
                else:
                    existing = existing.rechunk({whole_axis: -1}, method="tasks")

    fill = None
    if existing is None:
        fills = {} if fill_values is None else fill_values
        if len(table_row_positions) != n_obs and path not in fills:
            raise ValueError(f"New component {path!r} requires a fill for unselected rows.")
        if path in fills:
            fill = _scalar_fill(fills[path], regional_values.dtype)
            if matrix_format != "dense" and len(table_row_positions) != n_obs and fill != 0:
                raise ValueError("New sparse matrices with unselected rows require a zero fill.")
    return _regional_matrix(
        _lazy_matrix(regional_values, matrix_format, chunk_size=chunk_size),
        existing=existing,
        table_row_positions=table_row_positions,
        n_obs=n_obs,
        matrix_format=matrix_format,
        fill=fill,
        chunk_size=chunk_size,
    )


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


def _scalar_fill(value: object, dtype: np.dtype) -> np.generic:
    """Validate a scalar fill without widening the new matrix's dtype."""
    if not isinstance(value, Number) or not np.can_cast(np.min_scalar_type(value), dtype, casting="safe"):
        raise ValueError(f"Fill must be a numeric scalar compatible with dtype {dtype}.")
    return np.dtype(dtype).type(value)


def _lazy_matrix(value: object, matrix_format: str, *, chunk_size: int) -> da.Array:
    if isinstance(value, da.Array):
        return value
    if isinstance(value, (CSRDataset, CSCDataset)):
        return _decode_anndata_element(value.group, mode="lazy", sparse_chunks=chunk_size)
    if isinstance(value, zarr.Array):
        return da.from_zarr(value)
    chunks = {
        "dense": (chunk_size, -1),
        "csr": (chunk_size, -1),
        "csc": (-1, chunk_size),
    }[matrix_format]
    return da.from_array(value, chunks=chunks, asarray=False)


def _regional_matrix(
    regional_values: da.Array,
    *,
    existing: da.Array | None,
    table_row_positions: NDArray[np.intp],
    n_obs: int,
    matrix_format: str,
    fill: np.generic | None,
    chunk_size: int,
) -> da.Array:
    """Build full-axis replacements from independently chunked old and selected rows.

    regional_values contains the measurements to write for the selected observations.
    table_row_positions[i] gives the zero-based full-table destination of row i
    in regional_values.
    The returned .obsm matrix covers all table observations.

    Computational chunking of the returned matrix::

        All observations supplied
            -> Keep regional_values.chunks, as prepared by _lazy_matrix():
               Dask: existing input chunks
               dense Zarr: on-disk chunks
               in-memory / backed sparse: chunks based on chunk_size
               No additional output chunking is needed.

        Partial selection, existing entry
            -> Use existing.chunks, as prepared by the reader or
               _prepare_regional_matrix():
               dense Zarr / Dask: existing chunks
               in-memory dense:  (chunk_size rows, all columns)
               CSR: all columns; keep attached Dask row chunks, otherwise chunk_size
               CSC: all rows; keep attached Dask column chunks, otherwise chunk_size

        Partial selection, new entry
            -> Choose chunks from chunk_size and the full output shape:
               dense / CSR: (chunk_size rows, all columns)
               CSC:         (all output rows, chunk_size columns)

    For new entries, do not derive output chunk sizes from regional_values.chunks:
    a five-row input must not force five-row chunks across a large table.

    For partial selections, a lazy working view of regional_values groups the
    selected rows by their destination row band. Its delayed blocks are shared
    across merge tasks; the supplied array and output chunk policy are unchanged.
    Sparse output chunks span the uncompressed axis, as required by AnnData's
    sparse writer. Existing matrices must have the same complete chunk layout
    as the finalized output; their delayed blocks are shared across merge tasks.
    """
    dtype = regional_values.dtype if existing is None else existing.dtype
    if len(table_row_positions) == n_obs:
        return regional_values.astype(dtype)
    shape = (n_obs, regional_values.shape[1])
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
    # The reader returns CSR chunks spanning all columns and CSC chunks spanning
    # all rows; _prepare_regional_matrix() also prepares this layout for attached
    # arrays. Their chunks must match the finalized output, including terminal
    # chunks and sparse-axis adjustments. A mismatch here is an internal
    # preparation/merge error, not an unsupported public input layout.
    existing_blocks = None
    if existing is not None:
        if tuple(chunks) != existing.chunks:
            raise RuntimeError("Existing matrix chunks do not match the finalized output layout.")
        existing_blocks = existing.to_delayed()
    meta = (
        np.empty((0, 0), dtype=dtype)
        if matrix_format == "dense"
        else getattr(sparse, f"{matrix_format}_matrix")((0, 0), dtype=dtype)
    )
    # regional_values is the supplied .obsm matrix of new measurements, containing
    # only the observations being updated. table_row_positions maps its rows to
    # their destinations in the complete table (zero-based).
    # Example: a 12-observation table, with four supplied measurement rows:
    # regional_values[0], [1], [2], [3] belong at table rows 1, 2, 3, 6.
    #
    # table_row_positions        = [1, 2, 3, 6]
    # chunks[0]                  = (3, 3, 3, 3)  # three table rows per output chunk
    #
    # The calculations below produce:
    # table_row_chunk_boundaries = [0, 3, 6, 9, 12]
    # regional_row_offsets       = [0, 2, 3, 4, 4]
    # regional_row_counts        = [2, 1, 1, 0]
    # regional_row_chunks        = (2, 1, 1)  # skip empty update chunks
    #
    # Thus regional_values[0:2], [2:3] and [3:4] supply the first three output
    # row chunks. The fourth has no updates, so it needs no regional update chunk.
    #
    # Supplied observations follow stored table order, so updates for each output
    # row chunk form a consecutive slice of regional_values.
    # For each table chunk boundary, count the selected observations strictly
    # before it. These cumulative counts become slice boundaries in regional_values.
    # Consecutive offsets give each chunk's start and stop; equal offsets mean
    # that chunk has no updates. Their differences give the update row counts.
    table_row_chunk_boundaries = np.concatenate(([0], np.cumsum(chunks[0])))
    regional_row_offsets = np.searchsorted(table_row_positions, table_row_chunk_boundaries)
    regional_row_counts = np.diff(regional_row_offsets)
    regional_row_chunks = tuple(int(count) for count in regional_row_counts if count > 0)
    if shape[1]:
        # Rechunk the supplied measurements to match these per-output-chunk update
        # counts. Each resulting block contains exactly the updates for one output
        # block. The updated rows need not be consecutive in that output block;
        # block_row_positions specifies where each update belongs.
        # Prepare these blocks together in one shared graph, avoiding independently
        # constructed slices for every merge task. Task-based rechunking supports
        # both NumPy and SciPy blocks.
        regional_blocks = regional_values.rechunk((regional_row_chunks, chunks[1]), method="tasks").to_delayed()
        regional_block_rows = iter(regional_blocks)
    else:
        # No values need updating on an empty feature axis. Skipping rechunking
        # also avoids Dask replacing empty sparse blocks with dense ones.
        regional_block_rows = iter(())
    merge_block = delayed(_merge_regional_block)
    row_blocks = []
    table_row_start = 0
    for table_block_row, height in enumerate(chunks[0]):
        table_row_stop = table_row_start + height
        regional_row_start, regional_row_stop = regional_row_offsets[table_block_row : table_block_row + 2]
        block_row_positions = table_row_positions[regional_row_start:regional_row_stop] - table_row_start
        # Empty output row bands do not consume a row of regional update blocks.
        updates_for_row = next(regional_block_rows) if regional_row_stop > regional_row_start and shape[1] else None
        column_blocks = []
        for table_block_column, width in enumerate(chunks[1]):
            original = None if existing_blocks is None else existing_blocks[table_block_row, table_block_column]
            updates = None if updates_for_row is None else updates_for_row[table_block_column]
            block = merge_block(original, updates, block_row_positions, (height, width), dtype, matrix_format, fill)
            column_blocks.append(da.from_delayed(block, shape=(height, width), dtype=dtype, meta=meta))
        row_blocks.append(da.concatenate(column_blocks, axis=1))
        table_row_start = table_row_stop
    return da.concatenate(row_blocks, axis=0)


def _merge_regional_block(
    original: _MatrixBlock | None,
    updates: _MatrixBlock | None,
    block_row_positions: NDArray[np.intp],
    shape: tuple[int, int],
    dtype: np.dtype,
    matrix_format: str,
    fill: np.generic | None,
) -> _MatrixBlock:
    """Replace selected local rows without modifying input blocks or densifying sparse data.

    Parameters
    ----------
    original
        Computed NumPy/SciPy block of existing measurements. None when creating
        a new entry.
    updates
        Computed measurements for the selected observations in this block,
        with rows ordered to match block_row_positions. None when the block
        contains no selected observations.
    block_row_positions
        Zero-based row positions within the output block, one per row of updates.
    shape
        Output block shape as (number of rows, number of columns).
    dtype
        Output dtype, already validated as compatible with the updates.
    matrix_format
        Output representation: "dense", "csr" or "csc".
    fill
        Scalar for unselected rows of a new entry. Ignored, and may be None,
        when original is provided. New sparse blocks use implicit zeros.
    """
    if matrix_format == "dense":
        result = np.full(shape, fill, dtype=dtype) if original is None else original.copy()
        if updates is not None:
            result[block_row_positions] = updates
        return result
    # Keep only unselected old entries and remap incoming sparse row indices.
    # No assignment into shared sparse buffers (or dense temporary) is needed.
    keep_rows = np.ones(shape[0], dtype=bool)
    keep_rows[block_row_positions] = False
    old = sparse.coo_matrix(shape, dtype=dtype) if original is None else original.tocoo()
    keep = keep_rows[old.row]
    new = sparse.coo_matrix((0, shape[1]), dtype=dtype) if updates is None else updates.tocoo()
    result = sparse.coo_matrix(
        (
            np.concatenate((old.data[keep], new.data)).astype(dtype, copy=False),
            (np.concatenate((old.row[keep], block_row_positions[new.row])), np.concatenate((old.col[keep], new.col))),
        ),
        shape=shape,
    )
    return result.asformat(matrix_format)
