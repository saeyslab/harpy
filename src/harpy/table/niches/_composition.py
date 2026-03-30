from __future__ import annotations

import json

import numpy as np
import pandas as pd
from anndata import AnnData
from loguru import logger as log


def _to_fixed_unicode_array(values: list[str]) -> np.ndarray:
    """Return a fixed-width unicode array to avoid StringDType in `.uns`."""
    max_len = max((len(v) for v in values), default=1)
    return np.asarray(values, dtype=f"U{max_len}")


def _serialize_cluster_categories(values: list[str]) -> str | np.ndarray:
    """Serialize cluster categories safely across NumPy versions affected by gh-28609."""
    if np.lib.NumpyVersion(np.__version__) < np.lib.NumpyVersion("2.2.5"):
        # Upstream NumPy issue: https://github.com/numpy/numpy/issues/28609
        return json.dumps(values)
    return _to_fixed_unicode_array(values)


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
    cluster_key: str,
    connectivity_key: str = "spatial_connectivities",
    key_added: str = "nhood_composition",
) -> None:
    """
    Compute per-cell neighborhood cell-type fractions from an existing spatial graph.

    The resulting dense matrix is stored in `adata.obsm[key_added]`, while
    the associated metadata is stored in `adata.uns[key_added]`.
    """
    if cluster_key not in adata.obs.columns:
        raise KeyError(
            f"Cluster key '{cluster_key}' not found in `adata.obs`. "
            f"Available columns: {adata.obs.columns.to_list()}."
        )

    resolved_connectivity_key = _resolve_connectivity_key(adata, connectivity_key)
    connectivities = adata.obsp[resolved_connectivity_key]

    if not hasattr(connectivities, "tocsr"):
        raise TypeError(
            f"Connectivity matrix '{resolved_connectivity_key}' in `adata.obsp` must be sparse and support `.tocsr()`."
        )

    instance_types = adata.obs[cluster_key].astype("category")
    if instance_types.isna().any():
        raise ValueError(
            f"Cluster key '{cluster_key}' contains missing values. "
            "Please assign all cells to a category before calculating neighborhood composition."
        )

    if key_added in adata.obsm or key_added in adata.uns:
        log.warning(
            f"Neighborhood composition key '{key_added}' already exists in the AnnData object. "
            "Proceeding to overwrite it."
        )

    connectivities = connectivities.tocsr()
    onehot = pd.get_dummies(instance_types, sparse=True)
    onehot_mat = onehot.sparse.to_coo().tocsr()

    counts = connectivities.dot(onehot_mat)
    neigh_totals = np.asarray(connectivities.sum(axis=1)).ravel()

    with np.errstate(divide="ignore", invalid="ignore"):
        fractions = counts.multiply(1 / neigh_totals[:, None]).toarray()

    fractions = np.asarray(fractions, dtype=np.float32)
    fractions[~np.isfinite(fractions)] = 0.0

    adata.obsm[key_added] = fractions
    cluster_categories = instance_types.cat.categories.to_list()
    adata.uns[key_added] = {
        "cluster_key": cluster_key,
        "connectivity_key": resolved_connectivity_key,
        # Older NumPy releases have upstream StringDType copy issues after zarr round-trips.
        "cluster_categories": _serialize_cluster_categories(cluster_categories),
    }

    return None
