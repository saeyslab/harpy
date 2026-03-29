from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData
from loguru import logger as log
from sklearn.cluster import KMeans
from spatialdata import SpatialData
from spatialdata.models import TableModel

from harpy.table._table import ProcessTable, add_table_layer
from harpy.table.niches._composition import _compute_nhood_composition
from harpy.utils._keys import _ANNOTATION_KEY


def nhood_kmeans(
    sdata: SpatialData,
    table_layer: str,
    output_layer: str,
    instance_type_key: str = _ANNOTATION_KEY,
    labels_layer: str | Iterable[str] | None = None,
    connectivity_key: str = "spatial_connectivities",
    composition_key: str = "nhood_composition",
    key_added: str = "nhood_kmeans",
    n_clusters: int = 5,
    random_state: int = 100,
    nan_label: int | str | None = -1,
    overwrite: bool = False,
    **kwargs: Any,
) -> SpatialData:
    """
    Cluster cells (instances) based on neighborhood cell-type composition using KMeans.

    This function expects a precomputed spatial connectivity matrix in
    `sdata.tables[table_layer].obsp[connectivity_key]` and does not calculate
    neighbors itself.
    For example, the graph can be computed beforehand with
    :func:`squidpy.gr.spatial_neighbors` and stored in
    `sdata.tables[table_layer].obsp[connectivity_key]`. Neighborhood
    cell-type fractions are then computed from that graph, stored in
    `adata.obsm[composition_key]`, and used as the feature matrix for
    :class:`~sklearn.cluster.KMeans`. The resulting niche assignments are
    written to `adata.obs[key_added]`.

    Parameters
    ----------
    sdata
        The input SpatialData object.
    table_layer
        The table layer in `sdata` on which to perform niche clustering.
    output_layer
        The output table layer in `sdata` to which the updated table layer will
        be written.
    instance_type_key
        Key in `adata.obs` containing the instance type annotations used to build
        the neighborhood composition.
    labels_layer
        Optional labels layer or layers used to subset the table before
        clustering. If provided, only observations linked to these labels layers
        are considered.
    connectivity_key
        Key pointing to the cell-cell connectivity matrix in `adata.obsp`,
        with shape `(n_cells, n_cells)`. If the exact key is not found,
        `{connectivity_key}_connectivities` is tried as a convenience for
        graphs created with :func:`squidpy.gr.spatial_neighbors` using
        `key_added=...`.
    composition_key
        Key used to store the computed neighborhood composition. The dense
        neighborhood-fraction matrix is written to
        `adata.obsm[composition_key]` with shape `(n_cells, n_instance_types)`,
        where each row contains, for one cell, the fraction of neighbors that
        belong to each category in `instance_type_key`. Related metadata is stored
        under `adata.uns[composition_key]`, including the instance type key that
        was used, the resolved connectivity key from `adata.obsp`,
        and the ordered instance type labels corresponding to the columns of
        `adata.obsm[composition_key]`. Using the same `composition_key` in both
        places keeps the feature matrix and its column definitions linked and
        makes it easier to reuse the computed neighborhood features in
        downstream analyses. For example, `adata.obsm[composition_key]` could
        look like `[[0.75, 0.25, 0.00], [0.00, 0.50, 0.50]]`, meaning that the
        first cell has neighbors composed of 75% of the first cell type and 25%
        of the second, while
        `adata.uns[composition_key]["instance_type_categories"]` would store
        the ordered labels for those columns.
    key_added
        Key in `adata.obs` where the resulting niche labels are written.
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

    if key_added in adata.obs.columns:
        log.warning(f"The column '{key_added}' already exists in the AnnData object. Proceeding to overwrite it.")

    _compute_nhood_composition(
        adata,
        instance_type_key=instance_type_key,
        connectivity_key=connectivity_key,
        key_added=composition_key,
    )
    fractions = adata.obsm[composition_key]
    neigh_totals = np.asarray(adata.obsp[adata.uns[composition_key]["connectivity_key"]].sum(axis=1)).ravel()

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
    adata.obs[key_added] = pd.Categorical(labels_full)

    adata.uns[key_added] = {
        "instance_type_key": instance_type_key,
        "connectivity_key": adata.uns[composition_key]["connectivity_key"],
        "composition_key": composition_key,
        "n_clusters": n_clusters,
        "random_state": random_state,
    }

    sdata = add_table_layer(
        sdata,
        adata=adata,
        output_layer=output_layer,
        region=sdata[table_layer].uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY],
        instance_key=sdata[table_layer].uns[TableModel.ATTRS_KEY][TableModel.INSTANCE_KEY],
        region_key=sdata[table_layer].uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY],
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
