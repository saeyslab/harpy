import inspect
from types import MappingProxyType
from typing import Any, Callable, Mapping

import pandas as pd
import scanpy as sc
import spatialdata
from anndata import AnnData
from sklearn.cluster import KMeans
from spatialdata import SpatialData

from sparrow.utils._keys import _REGION_KEY
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
    """Implementation of kmeans clustering."""
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
    """Implementation of leiden clustering."""
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


class Cluster:
    def __init__(
        self,
        sdata: SpatialData,
        labels_layer: str,
    ):
        """
        Initialize the Cluster object with SpatialData, labels layer.

        Parameters
        ----------
        - spatial_data: SpatialData
            The SpatialData object containing spatial data and annotations.
        - labels_layer: str
            The label layer to use for clustering.
        """
        self.sdata = sdata
        self.labels_layer = labels_layer
        self._validate_labels_layer()

    def _validate_labels_layer(self):
        """Validate if the specified labels layer exists in the SpatialData object."""
        if self.labels_layer not in self.sdata.table.obs[_REGION_KEY].cat.categories:
            raise ValueError("labels layer not in table")

    def _preprocess(self):
        """Preprocess the data by filtering based on the labels layer and setting attributes."""
        adata = self.sdata.table[self.sdata.table.obs[_REGION_KEY] == self.labels_layer].copy()
        adata.uns["spatialdata_attrs"]["region"] = [self.labels_layer]
        return adata

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
        adata = self._preprocess()

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
