from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData
from loguru import logger as log
from sklearn.cluster import KMeans
from spatialdata import SpatialData

from harpy.table._table import ProcessTable, add_table_layer
from harpy.table.niches._composition import _compute_nhood_composition
from harpy.utils._keys import _ANNOTATION_KEY, _INSTANCE_KEY, _REGION_KEY


def nhood_kmeans(
    sdata: SpatialData,
    table_layer: str,
    output_layer: str,
    cell_type_column: str = _ANNOTATION_KEY,
    labels_layer: str | Iterable[str] | None = None,
    connectivity_key: str = "spatial_connectivities",
    composition_key: str = "nhood_composition",
    output_column: str = "nhood_kmeans",
    n_clusters: int = 5,
    random_state: int = 100,
    nan_label: int | str | None = -1,
    overwrite: bool = False,
    **kwargs: Any,
) -> SpatialData:
    """
    Cluster cells based on neighborhood cell-type composition using KMeans.

    This function expects a precomputed spatial connectivity matrix in
    `sdata.tables[table_layer].obsp` and does not calculate neighbors itself.
    Neighborhood cell-type fractions are computed from that graph, stored in
    `adata.obsm[composition_key]`, and then used as the feature matrix for
    :class:`~sklearn.cluster.KMeans`.

    Parameters
    ----------
    sdata
        The input SpatialData object.
    table_layer
        The table layer in `sdata` on which to perform niche clustering.
    output_layer
        The output table layer in `sdata` to which the updated table layer will
        be written.
    cell_type_column
        Column in `adata.obs` containing the cell type annotations used to build
        the neighborhood composition.
    labels_layer
        Optional labels layer or layers used to subset the table before
        clustering. If provided, only observations linked to these labels layers
        are considered.
    connectivity_key
        Key pointing to the connectivity matrix in `adata.obsp`. If the exact
        key is not found, `{connectivity_key}_connectivities` is tried as a
        convenience for graphs created with `squidpy.gr.spatial_neighbors(...,
        key_added=...)`.
    composition_key
        Key used to store the computed neighborhood composition matrix in
        `adata.obsm` and its metadata in `adata.uns`.
    output_column
        Column in `adata.obs` where the niche labels are written. # TODO -> better name
    n_clusters
        Number of KMeans clusters to compute.
    random_state
        Random state used for reproducible clustering.
    nan_label
        Label assigned to isolated cells with zero graph degree.
    overwrite
        If `True`, overwrite `output_layer` if it already exists in `sdata`.
    **kwargs
        Additional keyword arguments passed to :class:`~sklearn.cluster.KMeans`.

    Returns
    -------
    The updated SpatialData object.
    """
    process_table_instance = ProcessTable(sdata, labels_layer=labels_layer, table_layer=table_layer)
    adata = process_table_instance._get_adata()

    if output_column in adata.obs.columns:
        log.warning(f"The column '{output_column}' already exists in the AnnData object. Proceeding to overwrite it.")

    fractions, neigh_totals = _compute_nhood_composition(
        adata,
        cell_type_column=cell_type_column,
        connectivity_key=connectivity_key,
        composition_key=composition_key,
    )

    mask_valid = neigh_totals > 0
    n_valid = int(mask_valid.sum())

    if n_valid == 0:
        raise ValueError(
            "No cells have neighbors in the provided connectivity graph. "
            "Please provide a graph with at least one non-isolated observation."
        )

    if n_valid < n_clusters:
        raise ValueError(
            f"Cannot fit KMeans with n_clusters={n_clusters} on only {n_valid} cells with at least one neighbor."
        )

    kmeans_kwargs = {
        "n_clusters": n_clusters,
        "random_state": random_state,
        "n_init": "auto",
    }
    kmeans_kwargs.update(kwargs)

    kmeans = KMeans(**kmeans_kwargs).fit(fractions[mask_valid])

    labels_full = np.full(adata.shape[0], nan_label, dtype=object)
    labels_full[mask_valid] = kmeans.labels_
    adata.obs[output_column] = pd.Categorical(labels_full)

    # TODO -> clean up
    adata.uns[output_column] = {
        "cell_type_column": cell_type_column,
        "connectivity_key": adata.uns[composition_key]["connectivity_key"],
        "composition_key": composition_key,
        "n_clusters": n_clusters,
        "random_state": random_state,
    }

    sdata = add_table_layer(
        sdata,
        adata=adata,
        output_layer=output_layer,
        region=_get_output_regions(adata, process_table_instance),  # TODO -> clean up, fetch it from the table
        instance_key=process_table_instance.instance_key or _INSTANCE_KEY,
        region_key=process_table_instance.region_key or _REGION_KEY,
        overwrite=overwrite,
    )

    return sdata


def _get_output_regions(adata: AnnData, process_table_instance: ProcessTable) -> list[str] | None:
    if process_table_instance.labels_layer is not None:
        return process_table_instance.labels_layer

    if process_table_instance.region_key is None or process_table_instance.region_key not in adata.obs.columns:
        return None

    region_obs = adata.obs[process_table_instance.region_key]
    if hasattr(region_obs, "cat"):
        return region_obs.cat.categories.to_list()

    return pd.Index(region_obs).unique().to_list()
