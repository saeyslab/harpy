from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData
from loguru import logger as log


def _to_fixed_unicode_array(values: list[str]) -> np.ndarray:
    """Return a fixed-width unicode array to avoid StringDType in `.uns`."""
    max_len = max((len(v) for v in values), default=1)
    return np.asarray(values, dtype=f"U{max_len}")


def _resolve_connectivity_key(adata: AnnData, connectivity_key: str) -> str:
    if connectivity_key in adata.obsp:
        return connectivity_key

    squidpy_key = f"{connectivity_key}_connectivities"
    if squidpy_key in adata.obsp:
        return squidpy_key

    available = sorted(adata.obsp.keys())
    raise KeyError(f"Connectivity key '{connectivity_key}' not found in `adata.obsp`. Available keys: {available}.")


def _compute_nhood_composition(
    adata: AnnData,
    cell_type_column: str,
    connectivity_key: str = "spatial_connectivities",
    composition_key: str = "nhood_composition",  # TODO -> clean up
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-cell neighborhood cell-type fractions from an existing spatial graph.

    The resulting dense matrix is stored in `adata.obsm[composition_key]`, while
    the associated metadata is stored in `adata.uns[composition_key]`.
    """
    if cell_type_column not in adata.obs.columns:
        raise KeyError(
            f"Cell type column '{cell_type_column}' not found in `adata.obs`. "
            f"Available columns: {adata.obs.columns.to_list()}."
        )

    resolved_connectivity_key = _resolve_connectivity_key(adata, connectivity_key)
    connectivities = adata.obsp[resolved_connectivity_key]

    if not hasattr(connectivities, "tocsr"):
        raise TypeError(
            f"Connectivity matrix '{resolved_connectivity_key}' in `adata.obsp` must be sparse and support `.tocsr()`."
        )

    cell_types = adata.obs[cell_type_column].astype("category")
    if cell_types.isna().any():
        raise ValueError(
            f"Cell type column '{cell_type_column}' contains missing values. "
            "Please assign all cells to a category before calculating neighborhood composition."
        )

    if composition_key in adata.obsm or composition_key in adata.uns:
        log.warning(
            f"Neighborhood composition key '{composition_key}' already exists in the AnnData object. "
            "Proceeding to overwrite it."
        )

    connectivities = connectivities.tocsr()
    onehot = pd.get_dummies(cell_types, sparse=True)
    onehot_mat = onehot.sparse.to_coo().tocsr()

    counts = connectivities.dot(onehot_mat)
    neigh_totals = np.asarray(connectivities.sum(axis=1)).ravel()

    with np.errstate(divide="ignore", invalid="ignore"):
        fractions = counts.multiply(1 / neigh_totals[:, None]).toarray()

    fractions = np.asarray(fractions, dtype=np.float32)
    fractions[~np.isfinite(fractions)] = 0.0

    adata.obsm[composition_key] = fractions
    adata.uns[composition_key] = {
        "cell_type_column": cell_type_column,
        "connectivity_key": resolved_connectivity_key,
        "columns": _to_fixed_unicode_array(cell_types.cat.categories.to_list()),
    }

    return fractions, neigh_totals
