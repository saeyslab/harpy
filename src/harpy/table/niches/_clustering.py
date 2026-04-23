from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger as log
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from spatialdata import SpatialData
from spatialdata.models import TableModel

from harpy.table._table import ProcessTable, add_table
from harpy.table.niches._composition import _compute_nhood_composition, _compute_nhood_counts
from harpy.utils._keys import _ANNOTATION_KEY


def nhood_kmeans(
    sdata: SpatialData,
    table_name: str,
    output_table_name: str,
    cluster_key: str = _ANNOTATION_KEY,
    labels_name: str | Iterable[str] | None = None,
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
    `sdata.tables[table_name].obsp[connectivity_key]` and does not calculate
    neighbors itself.
    For example, the graph can be computed beforehand with
    :func:`squidpy.gr.spatial_neighbors` and stored in
    `sdata.tables[table_name].obsp[connectivity_key]`. Neighborhood
    cell-type fractions are then computed from that graph, stored in
    `adata.obsm[composition_key]`, and used as the feature matrix for
    :class:`~sklearn.cluster.KMeans`. The resulting niche assignments are
    written to `adata.obs[key_added]`.

    Parameters
    ----------
    sdata
        The input SpatialData object.
    table_name
        The table element in `sdata` on which to perform niche clustering.
    output_table_name
        The output table element in `sdata` to which the updated table element will
        be written.
    cluster_key
        Key in `adata.obs` containing the cluster annotations used to build
        the neighborhood composition.
    labels_name
        Optional labels element or elements used to subset the table before
        clustering. If provided, only observations linked to these labels elements
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
        `adata.obsm[composition_key]` with shape `(n_cells, n_categories)`,
        where each row contains, for one cell, the fraction of neighbors that
        belong to each category in `cluster_key`. Related metadata is stored
        under `adata.uns[composition_key]`, including the cluster key that
        was used, the resolved connectivity key from `adata.obsp`,
        and the ordered cluster labels corresponding to the columns of
        `adata.obsm[composition_key]`. Using the same `composition_key` in both
        places keeps the feature matrix and its column definitions linked and
        makes it easier to reuse the computed neighborhood features in
        downstream analyses. For example, `adata.obsm[composition_key]` could
        look like `[[0.75, 0.25, 0.00], [0.00, 0.50, 0.50]]`, meaning that the
        first cell has neighbors composed of 75% of the first cell type and 25%
        of the second, while
        `adata.uns[composition_key]["cluster_categories"]` stores
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
        If `True`, overwrite `output_table_name` if it already exists in `sdata`.
    **kwargs
        Additional keyword arguments passed to :class:`~sklearn.cluster.KMeans`.

    Returns
    -------
    The updated SpatialData object.
    """
    process_table_instance = ProcessTable(sdata, labels_name=labels_name, table_name=table_name)
    adata = process_table_instance._get_adata()

    if key_added in adata.obs.columns:
        log.warning(f"The column '{key_added}' already exists in the AnnData object. Proceeding to overwrite it.")

    _compute_nhood_composition(
        adata,
        cluster_key=cluster_key,
        connectivity_key=connectivity_key,
        key_added=composition_key,
    )
    fractions = adata.obsm[composition_key]
    # Calculating neigh_totals from the fractions is sufficient to detect isolated cells: normalized composition rows
    # sum to 1 for non-isolated cells and to 0 for isolated cells.
    neigh_totals = fractions.sum(axis=1)

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
        "cluster_key": cluster_key,
        "connectivity_key": adata.uns[composition_key]["connectivity_key"],
        "composition_key": composition_key,
        "n_clusters": n_clusters,
        "random_state": random_state,
    }

    sdata = add_table(
        sdata,
        adata=adata,
        output_table_name=output_table_name,
        region=sdata[table_name].uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY],
        instance_key=sdata[table_name].uns[TableModel.ATTRS_KEY][TableModel.INSTANCE_KEY],
        region_key=sdata[table_name].uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY],
        overwrite=overwrite,
    )

    return sdata


def nhood_lda(
    sdata: SpatialData,
    table_name: str,
    output_table_name: str,
    cluster_key: str = _ANNOTATION_KEY,
    labels_name: str | Iterable[str] | None = None,
    connectivity_key: str = "spatial_connectivities",
    counts_key: str = "nhood_counts",
    topic_key: str = "nhood_lda_topics",
    key_added: str = "nhood_lda",
    n_topics: int = 5,
    random_state: int = 100,
    nan_label: int | str | None = -1,
    overwrite: bool = False,
    **kwargs: Any,
) -> SpatialData:
    """
    Cluster cells (instances) into niche topics using LDA on neighborhood cell-type counts.

    This function expects a precomputed spatial connectivity matrix in
    `sdata.tables[table_name].obsp[connectivity_key]` and does not calculate
    neighbors itself.
    For example, the graph can be computed beforehand with
    :func:`squidpy.gr.spatial_neighbors` and stored in
    `sdata.tables[table_name].obsp[connectivity_key]`.
    For each cell, the neighborhood graph is used to compute
    counts of neighboring cell types defined by `cluster_key`. These counts are
    treated as a non-negative "document-term" matrix and used to fit
    :class:`~sklearn.decomposition.LatentDirichletAllocation`. The dominant
    topic per cell is written to `adata.obs[key_added]`, while the full topic
    mixture per cell is stored in `adata.obsm[topic_key]`. This matrix has shape
    `(n_cells, n_topics)` and can be interpreted as `P(topic | cell)`, where
    each row gives, for one cell, the topic proportions of its neighborhood.
    Related metadata is stored in `adata.uns[key_added]`, including the
    clustering inputs and parameters together with
    `topic_celltype_distribution`, a matrix of shape
    `(n_topics, n_cell_types)` that can be interpreted as `P(cell_type | topic)`.
    Each row describes the cell-type composition of one latent niche topic, with
    columns aligned to the ordered cell-type categories stored in
    `adata.uns[counts_key]["cluster_categories"]`.

    Parameters
    ----------
    sdata
        The input SpatialData object.
    table_name
        The table element in `sdata` on which to perform niche clustering.
    output_table_name
        The output table element in `sdata` to which the updated table element will
        be written.
    cluster_key
        Key in `adata.obs` containing the cluster annotations used to build
        the neighborhood count matrix.
    labels_name
        Optional labels element or elements used to subset the table before
        clustering. If provided, only observations linked to these labels elements
        are considered.
    connectivity_key
        Key pointing to the cell-cell connectivity matrix in `adata.obsp`,
        with shape `(n_cells, n_cells)`. If the exact key is not found,
        `{connectivity_key}_connectivities` is tried as a convenience for
        graphs created with :func:`squidpy.gr.spatial_neighbors` using
        `key_added=...`.
    counts_key
        Key used to store the computed neighborhood count matrix in
        `adata.obsm[counts_key]` and related metadata in `adata.uns[counts_key]`.
    topic_key
        Key in `adata.obsm` where the per-cell topic probabilities are stored.
        The matrix has shape `(n_cells, n_topics)` and can be interpreted as
        `P(topic | cell)`.
    key_added
        Key in `adata.obs` where the resulting dominant niche topic labels are
        written. Related metadata is stored in `adata.uns[key_added]`, including
        the resolved connectivity key, the neighborhood-count key, the topic key,
        the number of topics, the random state, and
        `topic_celltype_distribution`, which summarizes the inferred
        `P(cell_type | topic)` distributions.
    n_topics
        Number of latent topics to infer.
    random_state
        Random state used for reproducible clustering.
    nan_label
        Label assigned to isolated cells with zero graph degree.
    overwrite
        If `True`, overwrite `output_table_name` if it already exists in `sdata`.
    **kwargs
        Additional keyword arguments passed to
        :class:`~sklearn.decomposition.LatentDirichletAllocation`.

    Returns
    -------
    The updated SpatialData object.
    """
    process_table_instance = ProcessTable(sdata, labels_name=labels_name, table_name=table_name)
    adata = process_table_instance._get_adata()

    if key_added in adata.obs.columns:
        log.warning(f"The column '{key_added}' already exists in the AnnData object. Proceeding to overwrite it.")
    if topic_key in adata.obsm:
        log.warning(f"The key '{topic_key}' already exists in `adata.obsm`. Proceeding to overwrite it.")

    _compute_nhood_counts(
        adata,
        cluster_key=cluster_key,
        connectivity_key=connectivity_key,
        key_added=counts_key,
    )
    counts = adata.obsm[counts_key]
    neigh_totals = counts.sum(axis=1)

    mask_valid = neigh_totals > 0
    n_valid = int(mask_valid.sum())

    if n_valid == 0:
        raise ValueError(
            "No cells have neighbors in the provided connectivity graph. "
            "Please provide a graph with at least one non-isolated observation."
        )

    if n_valid < n_topics:
        raise ValueError(f"Cannot fit LDA with n_topics={n_topics} on only {n_valid} cells with at least one neighbor.")

    lda_kwargs = {
        "n_components": n_topics,
        "random_state": random_state,
        "learning_method": "batch",
    }
    lda_kwargs.update(kwargs)

    lda = LatentDirichletAllocation(**lda_kwargs)
    topic_probabilities_valid = lda.fit_transform(counts[mask_valid])

    labels_full = np.full(adata.shape[0], nan_label, dtype=object)
    labels_full[mask_valid] = topic_probabilities_valid.argmax(axis=1)
    adata.obs[key_added] = pd.Categorical(labels_full)

    topic_probabilities = np.zeros((adata.shape[0], n_topics), dtype=np.float32)
    topic_probabilities[mask_valid] = topic_probabilities_valid.astype(np.float32)
    adata.obsm[topic_key] = topic_probabilities

    topic_celltype_distribution = lda.components_ / lda.components_.sum(axis=1, keepdims=True)
    adata.uns[key_added] = {
        "cluster_key": cluster_key,
        "connectivity_key": adata.uns[counts_key]["connectivity_key"],
        "counts_key": counts_key,
        "topic_key": topic_key,
        "n_topics": n_topics,
        "random_state": random_state,
        "topic_celltype_distribution": np.asarray(topic_celltype_distribution, dtype=np.float32),
    }

    sdata = add_table(
        sdata,
        adata=adata,
        output_table_name=output_table_name,
        region=sdata[table_name].uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY],
        instance_key=sdata[table_name].uns[TableModel.ATTRS_KEY][TableModel.INSTANCE_KEY],
        region_key=sdata[table_name].uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY],
        overwrite=overwrite,
    )

    return sdata
