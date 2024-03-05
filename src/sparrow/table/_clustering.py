import inspect
from types import MappingProxyType
from typing import Any, Callable, Mapping

import pandas as pd
import scanpy as sc
import spatialdata
from anndata import AnnData
from sklearn.cluster import KMeans
from spatialdata import SpatialData

from sparrow.table._table import ProcessTable
from sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def kmeans(
    sdata: SpatialData,
    labels_layer: str,
    calculate_umap: bool = True,
    rank_genes: bool = True,  # TODO move this to other function
    n_neighbors: int = 35,  # ignored if calculate_umap=False
    n_pcs: int = 17,  # ignored if calculate_umap=False
    n_clusters: int = 5,
    key_added="kmeans",
    random_state: int = 100,
    **kwargs,  # keyword arguments for _kmeans
):
    """
    Applies KMeans clustering on the SpatialData object with optional UMAP calculation and gene ranking.

    This function executes the KMeans clustering algorithm (via `sklearn.cluster.KMeans`) on spatial data encapsulated by a SpatialData object.
    It optionally computes a UMAP (Uniform Manifold Approximation and Projection) for dimensionality reduction
    and ranks genes based on their contributions to the clustering. The clustering results, along with optional
    UMAP and gene ranking, are added to the `sdata.table` for downstream analysis.

    Parameters
    ----------
    sdata : SpatialData
        The input SpatialData object.
    labels_layer : str
        The labels layer of `sdata` used to select the cells via the _REGION_KEY.
        Note that cells in `sdata.table` linked to other labels_layer (via the _REGION_KEY), will be removed from `sdata.table`.
    calculate_umap : bool, default=True
        If True, calculates a UMAP via `scanpy.tl.umap` for visualization of computed clusters.
    rank_genes : bool, default=True
        If True, ranks genes based on their contributions to the clusters via `scanpy.tl.rank_genes_groups`. TODO: To be moved to a separate function.
    n_neighbors : int, default=35
        The number of neighbors to consider when calculating neighbors via `scanpy.pp.neighbors`. Ignored if `calculate_umap` is False.
    n_pcs : int, default=17
        The number of principal components to use when calculating neighbors via `scanpy.pp.neighbors`. Ignored if `calculate_umap` is False.
    n_clusters : int, default=5
        The number of clusters to form.
    key_added : str, default="kmeans"
        The key under which the clustering results are added to the SpatialData object (in `sdata.table.obs`).
    random_state : int, default=100
        A random state for reproducibility of the clustering.
    **kwargs
        Additional keyword arguments passed to the KMeans algorithm.

    Returns
    -------
    SpatialData
        The input `sdata` with the clustering results added.

    Notes
    -----
    - The function updates the SpatialData object in-place, adding clustering labels, and optionally UMAP coordinates
      and gene rankings, facilitating downstream analyses and visualization.
    - Gene ranking based on cluster contributions is intended for identifying marker genes that characterize each cluster.

    Warnings
    --------
    - The function is intended for use with spatial omics data. Input data should be appropriately preprocessed
      (e.g. via `sp.tb.preprocess_transcriptomics` or `sp.tb.preprocess_proteomics`) to ensure meaningful clustering results.
    - The `rank_genes` functionality is marked for relocation to enhance modularity and clarity of the codebase.
    """
    cluster = Cluster(sdata, labels_layer=labels_layer)
    cluster.cluster(
        _kmeans,
        key_added=key_added,
        calculate_umap=calculate_umap,
        calculate_neighbors=False,
        rank_genes=rank_genes,
        neigbors_kwargs={"n_neighbors": n_neighbors, "n_pcs": n_pcs, "random_state": random_state},
        umap_kwargs={"random_state": random_state},
        n_clusters=n_clusters,
        random_state=random_state,
        **kwargs,
    )

    return sdata


def leiden(
    sdata: SpatialData,
    labels_layer: str,
    calculate_umap: bool = True,
    calculate_neighbors: bool = True,
    rank_genes: bool = True,  # TODO move this to other function
    n_neighbors: int = 35,
    n_pcs: int = 17,
    resolution: float = 0.8,
    key_added: str = "leiden",
    random_state: int = 100,
    **kwargs,
):
    """
    Applies leiden clustering on the SpatialData object with optional UMAP calculation and gene ranking.

    This function executes the leiden clustering algorithm (via `sc.tl.leiden`) on spatial data encapsulated by a SpatialData object.
    It optionally computes a UMAP (Uniform Manifold Approximation and Projection) for dimensionality reduction
    and ranks genes based on their contributions to the clustering. The clustering results, along with optional
    UMAP and gene ranking, are added to the `sdata.table` for downstream analysis.

    Parameters
    ----------
    sdata : SpatialData
        The input SpatialData object.
    labels_layer : str
        The labels layer of `sdata` used to select the cells via the _REGION_KEY.
        Note that cells in `sdata.table` linked to other labels_layer (via the _REGION_KEY), will be removed from `sdata.table`.
    calculate_umap : bool, default=True
        If True, calculates a UMAP via `scanpy.tl.umap` for visualization of computed clusters.
    calculate_neighbors : bool, default=True
        If True, calculates neighbors via `scanpy.pp.neighbors` required for leiden clustering. Set to False if neighbors are already calculated for `sdata.table`.
    rank_genes : bool, default=True
        If True, ranks genes based on their contributions to the clusters via `scanpy.tl.rank_genes_groups`. TODO: To be moved to a separate function.
    n_neighbors : int, default=35
        The number of neighbors to consider when calculating neighbors via `scanpy.pp.neighbors`. Ignored if `calculate_umap` is False.
    n_pcs : int, default=17
        The number of principal components to use when calculating neighbors via `scanpy.pp.neighbors`. Ignored if `calculate_umap` is False.
    resolution : float, default=0.8
        Cluster resolution passed to `scanpy.tl.leiden`.
    key_added : str, default="leiden"
        The key under which the clustering results are added to the SpatialData object (in `sdata.table.obs`).
    random_state : int, default=100
        A random state for reproducibility of the clustering.
    **kwargs
        Additional keyword arguments passed to the leiden clusteting algorithm.

    Returns
    -------
    SpatialData
        The input `sdata` with the clustering results added.

    Notes
    -----
    - The function updates the SpatialData object in-place, adding clustering labels, and optionally UMAP coordinates
      and gene rankings, facilitating downstream analyses and visualization.
    - Gene ranking based on cluster contributions is intended for identifying marker genes that characterize each cluster.

    Warnings
    --------
    - The function is intended for use with spatial omics data. Input data should be appropriately preprocessed
      (e.g. via `sp.tb.preprocess_transcriptomics` or `sp.tb.preprocess_proteomics`) to ensure meaningful clustering results.
    - The `rank_genes` functionality is marked for relocation to enhance modularity and clarity of the codebase.
    """
    cluster = Cluster(sdata, labels_layer=labels_layer)
    cluster.cluster(
        _leiden,
        key_added=key_added,
        calculate_umap=calculate_umap,
        calculate_neighbors=calculate_neighbors,
        rank_genes=rank_genes,
        neigbors_kwargs={"n_neighbors": n_neighbors, "n_pcs": n_pcs, "random_state": random_state},
        umap_kwargs={"random_state": random_state},
        resolution=resolution,
        random_state=random_state,
        **kwargs,  # keyword arguments for _leiden
    )
    return sdata


def _kmeans(
    adata: AnnData,
    key_added: str = "kmeans",
    **kwargs,
) -> AnnData:
    kmeans = KMeans(**kwargs).fit(adata.X)
    adata.obs[key_added] = pd.Categorical(kmeans.labels_)
    return adata


def _leiden(
    adata: AnnData,
    key_added: str = "leiden",
    resolution: float = 0.8,
    **kwargs,  # kwargs passed to leiden
) -> AnnData:
    if "neighbors" not in adata.uns.keys():
        raise RuntimeError(
            "Please first compute neighbors before calculating leiden cluster, by passing 'calculate_neighbors=True' to 'sp.tb.leiden'"
        )

    sc.tl.leiden(adata, resolution=resolution, key_added=key_added, **kwargs)

    return adata


class Cluster(ProcessTable):
    def _perform_clustering(self, adata: AnnData, cluster_callable: Callable, key_added: str, **kwargs):
        """Perform the specified clustering on the AnnData object."""
        cluster_callable(adata, key_added=key_added, **kwargs)

    def cluster(
        self,
        cluster_callable: Callable = _leiden,  # callable that takes in adata and returns adata with adata.obs[ "key_added" ] column added.
        key_added: str = "leiden",
        calculate_umap: bool = True,
        calculate_neighbors: bool = True,
        rank_genes: bool = True,
        neigbors_kwargs: Mapping[str, Any] = MappingProxyType({}),  # keyword arguments passed to sc.pp.neighbors
        umap_kwargs: Mapping[str, Any] = MappingProxyType({}),  # keyword arguments passed to sc.tl.umap
        **kwargs,
    ):
        """Run the preprocessing, optional neighborhood graph computation, optional UMAP computation, and clustering on 'sdata.table'."""
        adata = self._get_adata()

        if calculate_neighbors:
            if "neighbors" in adata.uns.keys():
                log.warning(
                    "'neighbors' already in 'adata.uns', recalculating neighbors. Consider passing 'calculate_neigbors=False'."
                )
            sc.pp.neighbors(adata, **neigbors_kwargs)
        if calculate_umap:
            if "neighbors" not in adata.uns.keys():
                log.info("'neighbors not in 'adata.uns', computing neighborhood graph before calculating umap.")
                sc.pp.neighbors(adata, **neigbors_kwargs)
            sc.tl.umap(adata, **umap_kwargs)

        if key_added in adata.obs.columns:
            log.warning(f"The column '{key_added}' already exists in the Anndata object. Proceeding to overwrite it.")

        assert (
            "key_added" in inspect.signature(cluster_callable).parameters
        ), f"Callable '{cluster_callable.__name__}' must include the parameter 'key_added'."
        self._perform_clustering(adata, cluster_callable=cluster_callable, key_added=key_added, **kwargs)

        # TODO move this ranking of genes to somewhere else
        if rank_genes:
            sc.tl.rank_genes_groups(adata, groupby=key_added, method="wilcoxon")

        # Update the SpatialData object
        if self.sdata.table:
            del self.sdata.table
        self.sdata.table = spatialdata.models.TableModel.parse(adata)

        return self.sdata
