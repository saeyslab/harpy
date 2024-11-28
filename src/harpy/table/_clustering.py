from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping
from types import MappingProxyType
from typing import Any

import scanpy as sc
from anndata import AnnData
from sklearn.cluster import KMeans
from spatialdata import SpatialData

from harpy.table._table import ProcessTable, add_table_layer
from harpy.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def kmeans(
    sdata: SpatialData,
    labels_layer: str | list[str] | None,
    table_layer: str,
    output_layer: str,
    calculate_umap: bool = True,
    rank_genes: bool = True,  # TODO move this to other function
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
    Applies KMeans clustering on the `table_layer` of the SpatialData object with optional UMAP calculation and gene ranking.

    This function executes the KMeans clustering algorithm (via `sklearn.cluster.KMeans`) on spatial data encapsulated by a SpatialData object.
    It optionally computes a UMAP (Uniform Manifold Approximation and Projection) for dimensionality reduction
    and ranks genes based on their contributions to the clustering. The clustering results, along with optional
    UMAP and gene ranking, are added to the `sdata.tables[output_layer]` for downstream analysis.

    Parameters
    ----------
    sdata
        The input SpatialData object.
    labels_layer
        The labels layer(s) of `sdata` used to select the cells via the _REGION_KEY in `sdata.tables[table_layer].obs`.
        Note that if `output_layer` is equal to `table_layer` and overwrite is True,
        cells in `sdata.tables[table_layer]` linked to other `labels_layer` (via the _REGION_KEY), will be removed from `sdata.tables[table_layer]`.
        If a list of labels layers is provided, they will therefore be clustered together (e.g. multiple samples).
    table_layer
        The table layer in `sdata` on which to perform clustering.
    output_layer
        The output table layer in `sdata` to which table layer with results of clustering will be written.
    calculate_umap
        If `True`, calculates a UMAP via `scanpy.tl.umap` for visualization of computed clusters.
    rank_genes
        If `True`, ranks genes based on their contributions to the clusters via `scanpy.tl.rank_genes_groups`. TODO: To be moved to a separate function.
    n_neighbors
        The number of neighbors to consider when calculating neighbors via `scanpy.pp.neighbors`. Ignored if `calculate_umap` is False.
    n_pcs
        The number of principal components to use when calculating neighbors via `scanpy.pp.neighbors`. Ignored if `calculate_umap` is False.
    n_clusters
        The number of clusters to form.
    key_added
        The key under which the clustering results are added to the SpatialData object (in `sdata.tables[table_layer].obs`).
    index_names_var
        List of index names to subset in `sdata.tables[table_layer].var`. `index_positions_var` will be used if `None`.
    index_positions_var
        List of integer positions to subset in `sdata.tables[table_layer].var`. Used if `index_names_var` is None.
    random_state
        A random state for reproducibility of the clustering.
    overwrite
        If True, overwrites the `output_layer` if it already exists in `sdata`.
    **kwargs
        Additional keyword arguments passed to the KMeans algorithm (`sklearn.cluster.KMeans`).

    Returns
    -------
    The input `sdata` with the clustering results added.

    Notes
    -----
    - The function adds a table layer, adding clustering labels, and optionally UMAP coordinates
      and gene rankings, facilitating downstream analyses and visualization.
    - Gene ranking based on cluster contributions is intended for identifying marker genes that characterize each cluster.

    Warnings
    --------
    - The function is intended for use with spatial omics data. Input data should be appropriately preprocessed
      (e.g. via `harpy.tb.preprocess_transcriptomics` or `harpy.tb.preprocess_proteomics`) to ensure meaningful clustering results.
    - The `rank_genes` functionality is marked for relocation to enhance modularity and clarity of the codebase.

    See Also
    --------
    harpy.tb.preprocess_transcriptomics : preprocess transcriptomics data.
    harpy.tb.preprocess_proteomics : preprocess proteomics data.
    """
    cluster = Cluster(sdata, labels_layer=labels_layer, table_layer=table_layer)
    cluster.cluster(
        output_layer=output_layer,
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
    labels_layer: str | list[str] | None,
    table_layer: str,
    output_layer: str,
    calculate_umap: bool = True,
    calculate_neighbors: bool = True,
    rank_genes: bool = True,  # TODO move this to other function
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
    Applies leiden clustering on the `table_layer` of the SpatialData object with optional UMAP calculation and gene ranking.

    This function executes the leiden clustering algorithm (via `sc.tl.leiden`) on spatial data encapsulated by a SpatialData object.
    It optionally computes a UMAP (Uniform Manifold Approximation and Projection) for dimensionality reduction
    and ranks genes based on their contributions to the clustering. The clustering results, along with optional
    UMAP and gene ranking, are added to the `sdata.tables[output_layer]` for downstream analysis.

    Parameters
    ----------
    sdata
        The input SpatialData object.
    labels_layer
        The labels layer(s) of `sdata` used to select the cells via the `_REGION_KEY` in `sdata.tables[table_layer].obs`.
        Note that if `output_layer` is equal to `table_layer` and `overwrite` is `True`,
        cells in `sdata.tables[table_layer]` linked to other `labels_layer` (via the `_REGION_KEY`), will be removed from `sdata.tables[table_layer]`.
        If a list of labels layers is provided, they will therefore be clustered together (e.g. multiple samples).
    table_layer:
        The table layer in `sdata` on which to perform clustering on.
    output_layer
        The output table layer in `sdata` to which table layer with results of clustering will be written.
    calculate_umap
        If `True`, calculates a UMAP via `scanpy.tl.umap` for visualization of computed clusters.
    calculate_neighbors
        If `True`, calculates neighbors via `scanpy.pp.neighbors` required for leiden clustering. Set to False if neighbors are already calculated for `sdata.tables[table_layer]`.
    rank_genes
        If `True`, ranks genes based on their contributions to the clusters via `scanpy.tl.rank_genes_groups`. TODO: To be moved to a separate function.
    n_neighbors
        The number of neighbors to consider when calculating neighbors via `scanpy.pp.neighbors`. Ignored if `calculate_umap` is `False`.
    n_pcs
        The number of principal components to use when calculating neighbors via `scanpy.pp.neighbors`. Ignored if `calculate_umap` is `False`.
    resolution
        Cluster resolution passed to `scanpy.tl.leiden`.
    key_added
        The key under which the clustering results are added to the SpatialData object (in `sdata.tables[table_layer].obs`).
    index_names_var
        List of index names to subset in `sdata.tables[table_layer].var`. `index_positions_var` will be used if `None`.
    index_positions_var
        List of integer positions to subset in `sdata.tables[table_layer].var`. Used if `index_names_var` is `None`.
    random_state
        A random state for reproducibility of the clustering.
    overwrite
        If `True`, overwrites the `output_layer` if it already exists in `sdata`.
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
      (e.g. via `harpy.tb.preprocess_transcriptomics` or `harpy.tb.preprocess_proteomics`) to ensure meaningful clustering results.
    - The `rank_genes` functionality is marked for relocation to enhance modularity and clarity of the codebase.

    See Also
    --------
    harpy.tb.preprocess_transcriptomics : preprocess transcriptomics data.
    harpy.tb.preprocess_proteomics : preprocess proteomics data.
    """
    cluster = Cluster(sdata, labels_layer=labels_layer, table_layer=table_layer)
    sdata = cluster.cluster(
        output_layer=output_layer,
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
        output_layer: str,
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
        """Run the preprocessing, optional neighborhood graph computation, optional UMAP computation, and clustering on 'sdata.tables[table_layer]'."""
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

        # TODO move this ranking of genes to somewhere else
        if rank_genes:
            sc.tl.rank_genes_groups(adata, copy=False, layer=None, groupby=key_added, method="wilcoxon")

        self.sdata = add_table_layer(
            self.sdata,
            adata=adata,
            output_layer=output_layer,
            region=self.labels_layer,
            overwrite=overwrite,
        )

        return self.sdata

    def _sanity_check(self, cluster_callable: Callable):
        assert (
            "key_added" in inspect.signature(cluster_callable).parameters
        ), f"Callable '{cluster_callable.__name__}' must include the parameter 'key_added'."
