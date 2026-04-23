from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping
from types import MappingProxyType
from typing import Any

import scanpy as sc
from anndata import AnnData
from loguru import logger as log
from sklearn.cluster import KMeans
from spatialdata import SpatialData

from harpy.table._table import ProcessTable, add_table


def kmeans(
    sdata: SpatialData,
    labels_name: str | list[str] | None,
    table_name: str,
    output_table_name: str,
    calculate_umap: bool = True,
    rank_genes: bool = True,
    n_neighbors: int = 35,  # ignored if calculate_umap=False
    n_pcs: int = 17,  # ignored if calculate_umap=False
    n_clusters: int = 5,
    key_added="kmeans",
    index_names_var: Iterable[str] | None = None,
    index_positions_var: Iterable[int] | None = None,
    random_state: int = 100,
    overwrite: bool = False,
    **kwargs,  # keyword arguments for _kmeans
):
    """
    Applies KMeans clustering on the `table_name` of the SpatialData object with optional UMAP calculation and gene ranking.

    This function executes the KMeans clustering algorithm (via :class:`~sklearn.cluster.KMeans`) on spatial data encapsulated by a SpatialData object.
    It optionally computes a UMAP (Uniform Manifold Approximation and Projection) for dimensionality reduction
    and ranks genes based on their contributions to the clustering. The clustering results, along with optional
    UMAP and gene ranking, are added to the `sdata.tables[output_table_name]` for downstream analysis.

    Parameters
    ----------
    sdata
        The input SpatialData object.
    labels_name
        The labels element(s) of `sdata` used to select the cells via the region key in `sdata.tables[table_name].obs`.
        Note that if `output_table_name` is equal to `table_name` and overwrite is True,
        cells in `sdata.tables[table_name]` linked to other `labels_name` (via the region key), will be removed from `sdata.tables[table_name]`.
        If a list of labels elements is provided, they will therefore be clustered together (e.g. multiple samples).
    table_name
        The table element in `sdata` on which to perform clustering.
    output_table_name
        The output table element in `sdata` to which table element with results of clustering will be written.
    calculate_umap
        If `True`, calculates a UMAP via :func:`~scanpy.tl.umap` for visualization of computed clusters.
    rank_genes
        If `True`, ranks genes based on their contributions to the clusters via :func:`~scanpy.tl.rank_genes_groups`, with default parameters.
        Note that :func:`~scanpy.tl.rank_genes_groups` will be run on the `.raw` attribute of the :class:`~anndata.AnnData` table, if `.raw` is not `None`.
    n_neighbors
        The number of neighbors to consider when calculating neighbors via :func:`~scanpy.pp.neighbors`. Ignored if `calculate_umap` is False.
    n_pcs
        The number of principal components to use when calculating neighbors via :func:`~scanpy.pp.neighbors`. Ignored if `calculate_umap` is False.
    n_clusters
        The number of clusters to form.
    key_added
        The key under which the clustering results are added to the SpatialData object (in `sdata.tables[table_name].obs`).
    index_names_var
        List of index names to subset in `sdata.tables[table_name].var`. `index_positions_var` will be used if `None`.
    index_positions_var
        List of integer positions to subset in `sdata.tables[table_name].var`. Used if `index_names_var` is None.
    random_state
        A random state for reproducibility of the clustering.
    overwrite
        If True, overwrites the `output_table_name` if it already exists in `sdata`.
    **kwargs
        Additional keyword arguments passed to the KMeans algorithm (:class:`~sklearn.cluster.KMeans`).

    Returns
    -------
    The input `sdata` with the clustering results added.

    Notes
    -----
    - The function adds a table element, adding clustering labels, and optionally UMAP coordinates
      and gene rankings, facilitating downstream analyses and visualization.
    - Gene ranking based on cluster contributions is intended for identifying marker genes that characterize each cluster.

    Warnings
    --------
    - The function is intended for use with spatial omics data. Input data should be appropriately preprocessed
      (e.g. via :func:`~harpy.tb.preprocess_transcriptomics` or :func:`~harpy.tb.preprocess_proteomics`) to ensure meaningful clustering results.
    - The `rank_genes` functionality is marked for relocation to enhance modularity and clarity of the codebase.

    See Also
    --------
    harpy.tb.preprocess_transcriptomics : preprocess transcriptomics data.
    harpy.tb.preprocess_proteomics : preprocess proteomics data.
    """
    cluster = Cluster(sdata, labels_name=labels_name, table_name=table_name)
    cluster.cluster(
        output_table_name=output_table_name,
        cluster_callable=_kmeans,
        key_added=key_added,
        index_names_var=index_names_var,
        index_positions_var=index_positions_var,
        calculate_umap=calculate_umap,
        calculate_neighbors=False,
        rank_genes=rank_genes,
        neigbors_kwargs={"n_neighbors": n_neighbors, "n_pcs": n_pcs, "random_state": random_state},
        umap_kwargs={"random_state": random_state},
        n_clusters=n_clusters,
        random_state=random_state,
        overwrite=overwrite,
        **kwargs,
    )

    return sdata


def leiden(
    sdata: SpatialData,
    labels_name: str | list[str] | None,
    table_name: str,
    output_table_name: str,
    calculate_umap: bool = True,
    calculate_neighbors: bool = True,
    rank_genes: bool = True,
    n_neighbors: int = 35,
    n_pcs: int = 17,
    resolution: float = 0.8,
    key_added: str = "leiden",
    index_names_var: Iterable[str] | None = None,
    index_positions_var: Iterable[int] | None = None,
    random_state: int = 100,
    overwrite: bool = False,
    **kwargs,
):
    """
    Applies leiden clustering on the `table_name` of the SpatialData object with optional UMAP calculation and gene ranking.

    This function executes the leiden clustering algorithm (via `sc.tl.leiden`) on spatial data encapsulated by a SpatialData object.
    It optionally computes a UMAP (Uniform Manifold Approximation and Projection) for dimensionality reduction
    and ranks genes based on their contributions to the clustering. The clustering results, along with optional
    UMAP and gene ranking, are added to the `sdata.tables[output_table_name]` for downstream analysis.

    Parameters
    ----------
    sdata
        The input SpatialData object.
    labels_name
        The labels element(s) of `sdata` used to select the cells via the region key in `sdata.tables[table_name].obs`.
        Note that if `output_table_name` is equal to `table_name` and `overwrite` is `True`,
        cells in `sdata.tables[table_name]` linked to other `labels_name` (via the region key), will be removed from `sdata.tables[table_name]`.
        If a list of labels elements is provided, they will therefore be clustered together (e.g. multiple samples).
    table_name:
        The table element in `sdata` on which to perform clustering on.
    output_table_name
        The output table element in `sdata` to which table element with results of clustering will be written.
    calculate_umap
        If `True`, calculates a UMAP via :func:`~scanpy.tl.umap` for visualization of computed clusters.
    calculate_neighbors
        If `True`, calculates neighbors via :func:`~scanpy.pp.neighbors` required for leiden clustering. Set to False if neighbors are already calculated for `sdata.tables[table_name]`.
    rank_genes
        If `True`, ranks genes based on their contributions to the clusters via :func:`~scanpy.tl.rank_genes_groups` with default parameters.
        Note that :func:`~scanpy.tl.rank_genes_groups` will be run on the `.raw` attribute of the :class:`~anndata.AnnData` table, if `.raw` is not `None`.
    n_neighbors
        The number of neighbors to consider when calculating neighbors via :func:`~scanpy.pp.neighbors`. Ignored if `calculate_umap` is `False`.
    n_pcs
        The number of principal components to use when calculating neighbors via :func:`~scanpy.pp.neighbors`. Ignored if `calculate_umap` is `False`.
    resolution
        Cluster resolution passed to :func:`~scanpy.tl.leiden`.
    key_added
        The key under which the clustering results are added to the SpatialData object (in `sdata.tables[table_name].obs`).
    index_names_var
        List of index names to subset in `sdata.tables[table_name].var`. `index_positions_var` will be used if `None`.
    index_positions_var
        List of integer positions to subset in `sdata.tables[table_name].var`. Used if `index_names_var` is `None`.
    random_state
        A random state for reproducibility of the clustering.
    overwrite
        If `True`, overwrites the `output_table_name` if it already exists in `sdata`.
    **kwargs
        Additional keyword arguments passed to the leiden clustering algorithm (`sc.tl.leiden`).

    Returns
    -------
    The input `sdata` with the clustering results added.

    Notes
    -----
    - The function updates the SpatialData object in-place, adding clustering labels, and optionally UMAP coordinates
      and gene rankings, facilitating downstream analyses and visualization.
    - Gene ranking based on cluster contributions is intended for identifying marker genes that characterize each cluster.

    Warnings
    --------
    - The function is intended for use with spatial omics data. Input data should be appropriately preprocessed
      (e.g. via :func:`~harpy.tb.preprocess_transcriptomics` or :func:`~harpy.tb.preprocess_proteomics`) to ensure meaningful clustering results.
    - The `rank_genes` functionality is marked for relocation to enhance modularity and clarity of the codebase.

    See Also
    --------
    harpy.tb.preprocess_transcriptomics : preprocess transcriptomics data.
    harpy.tb.preprocess_proteomics : preprocess proteomics data.
    """
    cluster = Cluster(sdata, labels_name=labels_name, table_name=table_name)
    sdata = cluster.cluster(
        output_table_name=output_table_name,
        cluster_callable=_leiden,
        key_added=key_added,
        index_names_var=index_names_var,
        index_positions_var=index_positions_var,
        calculate_umap=calculate_umap,
        calculate_neighbors=calculate_neighbors,
        rank_genes=rank_genes,
        neigbors_kwargs={"n_neighbors": n_neighbors, "n_pcs": n_pcs, "random_state": random_state},
        umap_kwargs={"random_state": random_state},
        resolution=resolution,
        random_state=random_state,
        overwrite=overwrite,
        **kwargs,  # keyword arguments for _leiden
    )
    return sdata


def _kmeans(
    adata: AnnData,
    key_added: str = "kmeans",
    **kwargs,
) -> AnnData:
    kmeans = KMeans(**kwargs).fit(adata.X)
    adata.obs[key_added] = kmeans.labels_
    adata.obs[key_added] = adata.obs[key_added].astype(int).astype("category")
    return adata


def _leiden(
    adata: AnnData,
    key_added: str = "leiden",
    resolution: float = 0.8,
    **kwargs,  # kwargs passed to leiden
) -> AnnData:
    if "neighbors" not in adata.uns.keys():
        raise RuntimeError(
            "Please first compute neighbors before calculating leiden cluster, by passing 'calculate_neighbors=True' to 'harpy.tb.leiden'"
        )

    sc.tl.leiden(adata, copy=False, resolution=resolution, key_added=key_added, **kwargs)
    adata.obs[key_added] = adata.obs[key_added].astype(int).astype("category")

    return adata


class Cluster(ProcessTable):
    def _perform_clustering(self, adata: AnnData, cluster_callable: Callable, key_added: str, **kwargs):
        """Perform the specified clustering on the AnnData object."""
        cluster_callable(adata, key_added=key_added, **kwargs)

    def cluster(
        self,
        output_table_name: str,
        cluster_callable: Callable = _leiden,  # callable that takes in adata and returns adata with adata.obs[ "key_added" ] column added.
        key_added: str = "leiden",
        index_names_var: Iterable[str] | None = None,
        index_positions_var: Iterable[int] | None = None,
        calculate_umap: bool = True,
        calculate_neighbors: bool = True,
        rank_genes: bool = True,
        neigbors_kwargs: Mapping[str, Any] = MappingProxyType({}),  # keyword arguments passed to sc.pp.neighbors
        umap_kwargs: Mapping[str, Any] = MappingProxyType({}),  # keyword arguments passed to sc.tl.umap
        overwrite: bool = False,
        **kwargs,
    ) -> SpatialData:
        """Run the preprocessing, optional neighborhood graph computation, optional UMAP computation, and clustering on 'sdata.tables[table_name]'."""
        adata = self._get_adata(index_names_var=index_names_var, index_positions_var=index_positions_var)

        if calculate_neighbors:
            if "neighbors" in adata.uns.keys():
                log.warning(
                    "'neighbors' already in 'adata.uns', recalculating neighbors. Consider passing 'calculate_neigbors=False'."
                )
            self._type_check_before_pca(adata)
            sc.pp.neighbors(adata, copy=False, **neigbors_kwargs)
        if calculate_umap:
            if "neighbors" not in adata.uns.keys():
                log.info("'neighbors not in 'adata.uns', computing neighborhood graph before calculating umap.")
                self._type_check_before_pca(adata)
                sc.pp.neighbors(adata, copy=False, **neigbors_kwargs)
            sc.tl.umap(adata, copy=False, **umap_kwargs)

        if key_added in adata.obs.columns:
            log.warning(f"The column '{key_added}' already exists in the Anndata object. Proceeding to overwrite it.")

        self._sanity_check(cluster_callable=cluster_callable)
        self._perform_clustering(adata, cluster_callable=cluster_callable, key_added=key_added, **kwargs)
        assert key_added in adata.obs.columns

        if rank_genes:
            sc.tl.rank_genes_groups(adata, copy=False, layer=None, groupby=key_added, method="wilcoxon")

        self.sdata = add_table(
            self.sdata,
            adata=adata,
            output_table_name=output_table_name,
            region=self.labels_name,
            instance_key=self.instance_key,
            region_key=self.region_key,
            overwrite=overwrite,
        )

        return self.sdata

    def _sanity_check(self, cluster_callable: Callable):
        assert "key_added" in inspect.signature(cluster_callable).parameters, (
            f"Callable '{cluster_callable.__name__}' must include the parameter 'key_added'."
        )
