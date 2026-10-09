"""Normalisations of table matrices that scanpy does not provide, for in-memory and lazy tables."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import dask.array as da
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse

_SIZE_ENTRY = "normalize_by_size"
_QUANTILE_ENTRY = "normalize_by_quantile"
_QUANTILE_COLUMN = "quantile"


def normalize_by_size(
    adata: AnnData,
    *,
    size_key: str = "area",
    scale_factor: float = 100,
    layer: str | None = None,
    key_added: str | None = None,
) -> None:
    """Divide each row of a table's matrix by the size of its instance, in place.

    Computes ``X / size * scale_factor``, with the sizes from ``adata.obs[size_key]``,
    such as the area of each cell. Like :mod:`scanpy.pp`, the function changes ``adata``
    and returns ``None``, so that it fits between scanpy calls. On a table stored in
    SpatialData, write the result back with :func:`harpy.tb.io.add_table_updates`.

    Parameters
    ----------
    adata
        The table, with its matrix in memory (a NumPy array, or a SciPy CSR or CSC matrix)
        or lazy (a Dask array, as read with ``harpy.tb.io.read_table(..., mode="lazy")``).
    size_key
        The column of ``adata.obs`` with the size of each instance.
    scale_factor
        The factor applied after the division. The default of 100 keeps the values in a
        range where later steps, such as the selection of highly variable genes, behave well.
    layer
        The layer to normalise; ``None`` normalises ``adata.X``.
    key_added
        The key in ``adata.uns`` under which the parameters are recorded; ``None`` uses
        ``"normalize_by_size"``. A key per normalisation keeps the records of several
        normalised matrices of one table apart, for example of a layer normalised by
        cell area and another by nucleus area.

    Raises
    ------
    ValueError
        If ``adata.obs`` has no column ``size_key``, or a size is missing, zero or negative.
        Also if ``layer`` is not a layer of ``adata``, or the matrix is missing.
    TypeError
        If the matrix is neither a NumPy array, a SciPy CSR or CSC matrix, nor a Dask array
        of those; for example a storage-backed matrix.

    Notes
    -----
    Integer values become ``float32``; floating-point values keep their dtype. A sparse
    matrix keeps its format: CSR stays CSR and CSC stays CSC.

    On a lazy table, the function only extends the Dask graph: nothing is computed. The
    scaling runs per block whenever the matrix is computed: by ``.compute()``, by a scanpy
    function, or by :func:`harpy.tb.io.add_table_updates` or
    :func:`harpy.tb.io.write_table_updates` when they write it back. It keeps the chunks
    and the ``csr_matrix`` blocks that scanpy requires. The
    matrix is then derived from the stored one, so a write-back of ``X`` with
    :func:`harpy.tb.io.add_table_updates` needs ``x_to``, as after
    :func:`scanpy.pp.normalize_total`.

    The parameters are recorded in ``adata.uns[key_added]``, by default
    ``adata.uns["normalize_by_size"]``, with the ``layer`` that was normalised. If that
    entry exists, the function warns that a matrix may already be normalised, for example
    by a second run of the same call, and replaces the entry.

    Examples
    --------
    .. code-block:: python

        adata = hp.tb.io.read_table("sdata.zarr", table_name="counts", mode="lazy")
        hp.tb.pp.normalize_by_size(adata, size_key="area")
        sc.pp.log1p(adata)
        hp.tb.io.write_table_updates(
            "sdata.zarr", table_name="counts", adata=adata, x_to=("layers", "size_normalised_log1p")
        )
    """
    matrix = _get_matrix(adata, layer)
    if size_key not in adata.obs:
        raise ValueError(f"adata.obs has no column {size_key!r} with the size of each instance.")
    sizes = pd.to_numeric(adata.obs[size_key], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
    invalid = ~np.isfinite(sizes) | (sizes <= 0)
    if invalid.any():
        raise ValueError(
            f"adata.obs[{size_key!r}] has {int(invalid.sum())} missing, zero or negative sizes; "
            "every instance needs a positive size."
        )
    dtype = _float_dtype(matrix.dtype)
    result = _scale_rows(matrix, (scale_factor / sizes).astype(dtype), dtype)
    entry = _SIZE_ENTRY if key_added is None else key_added
    _warn_if_normalised(adata, entry)
    _set_matrix(adata, layer, result)
    adata.uns[entry] = {"size_key": size_key, "scale_factor": scale_factor, "layer": layer}


def normalize_by_quantile(
    adata: AnnData,
    *,
    q: float = 0.999,
    max_value: float | None = 1,
    quantiles: pd.Series | Sequence[float] | np.ndarray | None = None,
    layer: str | None = None,
    key_added: str | None = None,
) -> None:
    """Divide each channel of a table's matrix by a quantile of its non-zero values, then clip, in place.

    Per channel, the ``q`` quantile of the non-zero values maps to 1, and values are
    clipped at ``max_value``, so that each channel lies in ``[0, max_value]``. That is the
    usual percentile normalisation of multiplex imaging intensities: the clipping caps
    the few values above the quantile, such as bright spots or artefacts. Like
    :mod:`scanpy.pp`, the function changes ``adata`` and returns ``None``. On a table
    stored in SpatialData, write the result back with
    :func:`harpy.tb.io.add_table_updates`.

    Parameters
    ----------
    adata
        The table, with its matrix in memory or lazy (a Dask array, as read with
        ``harpy.tb.io.read_table(..., mode="lazy")``).
    q
        The quantile of the non-zero values of each channel that maps to 1, between 0
        and 1. Ignored if ``quantiles`` is given.
    max_value
        The value at which the result is clipped; ``None`` does not clip. Named after
        ``max_value`` of :func:`scanpy.pp.scale`, which also clips.
    quantiles
        Quantiles computed before, to apply instead of computing them; for example
        ``other.var["quantile"]`` of another sample with the same channels. A pandas
        Series is aligned with ``adata.var_names`` by channel name; any other sequence
        holds one value per channel, in the order of ``adata.var``.
    layer
        The layer to normalise; ``None`` normalises ``adata.X``.
    key_added
        The key in ``adata.var`` and in ``adata.uns`` under which the quantiles and the
        parameters are recorded; ``None`` uses ``var["quantile"]`` and
        ``uns["normalize_by_quantile"]``. A key per normalisation keeps the records of
        several normalised matrices of one table apart, such as the quantiles of each.

    Raises
    ------
    ValueError
        If the matrix is a lazy sparse matrix (densify it first, or read the table in
        memory: ``mode="eager"`` in :func:`harpy.tb.io.read_table`, ``table_mode="eager"``
        in :func:`harpy.io.read_zarr`), ``q`` is not between 0 and 1, or ``quantiles`` misses a
        channel or has another length than the number of channels. Also if ``layer`` is
        not a layer of ``adata``, or the matrix is missing.
    TypeError
        If the matrix is neither a NumPy array, a SciPy sparse matrix, nor a dense Dask
        array; for example a storage-backed matrix.

    Notes
    -----
    The quantiles depend on all instances, so they are computed when the function is
    called: on a lazy table, the call reads the data once, which takes time on a large
    table. Dask computes them with the algorithm of :func:`numpy.nanquantile`, on blocks
    that hold all rows of a few channels: they equal the in-memory quantiles, within
    ``float32`` rounding for ``float32`` data. They then enter the Dask graph as
    constants, so the scaling and the clipping stay lazy: a later compute of the result,
    by ``.compute()``, by a scanpy function, or by
    :func:`harpy.tb.io.add_table_updates` or :func:`harpy.tb.io.write_table_updates`
    when they write it back, does not compute the quantiles again.

    With ``quantiles``, the function computes no quantiles and reads no data when it is
    called: on a lazy table, it only extends the Dask graph, as
    :func:`~harpy.tb.pp.normalize_by_size` does.

    In-memory sparse input is converted to a dense matrix first, and the result is
    dense. A channel without non-zero values keeps its values, all zero, and its
    quantile is NaN. Integer values become ``float32``; floating-point values keep
    their dtype.

    The quantile applied to each channel is recorded in ``adata.var[key_added]``, by
    default ``adata.var["quantile"]``, and the parameters in ``adata.uns[key_added]``, by
    default ``adata.uns["normalize_by_quantile"]``, with the ``layer`` that was
    normalised. If that ``uns`` entry exists, the function warns that a matrix may
    already be normalised, for example by a second run of the same call, and replaces
    both records.

    Examples
    --------
    .. code-block:: python

        adata = hp.tb.io.read_table("sdata.zarr", table_name="intensities", mode="lazy")
        hp.tb.pp.normalize_by_quantile(adata, q=0.999)
        hp.tb.io.write_table_updates(
            "sdata.zarr", table_name="intensities", adata=adata, x_to=("layers", "normalised")
        )

        # The same normalisation for another sample with the same channels.
        hp.tb.pp.normalize_by_quantile(other, quantiles=adata.var["quantile"])
    """
    matrix = _get_matrix(adata, layer)
    if isinstance(matrix, da.Array):
        if sparse.issparse(matrix._meta):
            raise ValueError(
                "normalize_by_quantile does not support a lazy sparse matrix. Densify it first, or read the "
                "table in memory: mode='eager' in harpy.tb.io.read_table, table_mode='eager' in "
                "harpy.io.read_zarr."
            )
    elif sparse.issparse(matrix):
        matrix = matrix.toarray()
    elif not isinstance(matrix, np.ndarray):
        raise TypeError(
            f"normalize_by_quantile needs a NumPy array, a SciPy sparse matrix or a Dask array, not "
            f"{type(matrix).__name__}."
        )
    if quantiles is None:
        if not 0 <= q <= 1:
            raise ValueError(f"q must be between 0 and 1, not {q!r}.")
        values = _channel_quantiles(matrix, q)
        recorded_q = q
    else:
        values = _aligned_quantiles(adata, quantiles)
        recorded_q = None
    dtype = _float_dtype(matrix.dtype)
    # A channel without non-zero values has a NaN quantile: divide it by 1, so that it stays zero.
    divisors = np.where(np.isfinite(values) & (values != 0), values, 1).astype(dtype)
    result = matrix.astype(dtype) / divisors
    if max_value is not None:
        result = da.minimum(result, max_value) if isinstance(result, da.Array) else np.minimum(result, max_value)
    entry, column = (_QUANTILE_ENTRY, _QUANTILE_COLUMN) if key_added is None else (key_added, key_added)
    _warn_if_normalised(adata, entry)
    _set_matrix(adata, layer, result)
    adata.var[column] = values
    adata.uns[entry] = {"q": recorded_q, "max_value": max_value, "layer": layer}


def _get_matrix(adata: AnnData, layer: str | None) -> object:
    """Return ``adata.X`` or the requested layer, which must exist."""
    if layer is None:
        if adata.X is None:
            raise ValueError("adata.X is missing; pass the layer to normalise with layer=.")
        return adata.X
    if layer not in adata.layers:
        raise ValueError(f"adata has no layer {layer!r}.")
    return adata.layers[layer]


def _set_matrix(adata: AnnData, layer: str | None, value: object) -> None:
    if layer is None:
        adata.X = value
    else:
        adata.layers[layer] = value


def _float_dtype(dtype: np.dtype) -> np.dtype:
    """Keep a floating-point dtype; integer and boolean values become ``float32``."""
    return dtype if np.issubdtype(dtype, np.floating) else np.dtype(np.float32)


def _scale_rows(matrix: object, factors: np.ndarray, dtype: np.dtype) -> object:
    """Multiply each row by its factor, in memory or lazily, keeping the matrix format and, if lazy, its chunks."""
    if isinstance(matrix, da.Array):
        # One factor per row, in the row chunks of the matrix; it broadcasts over the column blocks.
        row_factors = da.from_array(factors[:, None], chunks=(matrix.chunks[0], 1))
        return da.map_blocks(_scale_block_rows, matrix, row_factors, dtype=dtype, meta=matrix._meta.astype(dtype))
    if isinstance(matrix, np.ndarray) or sparse.issparse(matrix):
        return _scale_block_rows(matrix, factors[:, None])
    raise TypeError(
        f"normalize_by_size needs a NumPy array, a SciPy CSR or CSC matrix or a Dask array, not "
        f"{type(matrix).__name__}."
    )


def _scale_block_rows(block: object, row_factors: np.ndarray) -> object:
    """Multiply each row of one block by its factor, keeping a sparse block's format and class."""
    factors = np.asarray(row_factors)[:, 0]
    dtype = factors.dtype
    if not sparse.issparse(block):
        return np.asarray(block).astype(dtype) * factors[:, None]
    if block.format not in {"csr", "csc"}:
        raise TypeError(f"normalize_by_size needs CSR or CSC sparse matrices, not {block.format.upper()}.")
    result = block.astype(dtype, copy=True)
    if result.format == "csr":
        result.data *= np.repeat(factors, np.diff(result.indptr))
    else:
        result.data *= factors[result.indices]
    return result


def _channel_quantiles(matrix: np.ndarray | da.Array, q: float) -> np.ndarray:
    """Return the ``q`` quantile of the non-zero values of each channel, NaN for a channel without any."""
    with warnings.catch_warnings():
        # A channel without non-zero values is an all-NaN column: its NaN quantile is handled by the caller.
        warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
        if isinstance(matrix, da.Array):
            masked = da.where(matrix == 0, np.nan, matrix)
            return np.asarray(da.nanquantile(masked, q, axis=0).compute(), dtype=np.float64)
        masked = np.where(matrix == 0, np.nan, matrix)
        return np.asarray(np.nanquantile(masked, q, axis=0), dtype=np.float64)


def _aligned_quantiles(adata: AnnData, quantiles: pd.Series | Sequence[float] | np.ndarray) -> np.ndarray:
    """Return supplied quantiles in the order of ``adata.var_names``: by channel name for a Series."""
    if isinstance(quantiles, pd.Series):
        missing = adata.var_names.difference(quantiles.index)
        if len(missing):
            raise ValueError(
                f"quantiles has no value for {len(missing)} channels of adata, such as {list(missing[:5])}."
            )
        return quantiles.reindex(adata.var_names).to_numpy(dtype=np.float64)
    values = np.asarray(quantiles, dtype=np.float64)
    if values.shape != (adata.n_vars,):
        raise ValueError(f"quantiles needs one value per channel, {adata.n_vars}, not shape {values.shape}.")
    return values


def _warn_if_normalised(adata: AnnData, entry: str) -> None:
    if entry in adata.uns:
        warnings.warn(
            f"adata.uns[{entry!r}] exists: a matrix may already be normalised. The entry is replaced; to keep "
            "the records of several normalisations of one table, pass a different key_added.",
            UserWarning,
            # Point at the caller of the public function.
            stacklevel=3,
        )
