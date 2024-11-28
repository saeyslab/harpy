from anndata import AnnData

from sparrow.utils._keys import ClusteringKey
from sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

try:
    import flowsom as fs
except ImportError:
    log.warning("'flowsom' not installed, 'sp.tb.flowsom' and `sp.im.flowsom` will not be available.")


def _flowsom(
    adata: AnnData,
    n_clusters: int = 10,
    **kwargs,
) -> tuple[AnnData, fs.FlowSOM]:
    fsom = fs.FlowSOM(adata, n_clusters=n_clusters, cols_to_use=None, **kwargs)
    if "cols_used" in adata.var:
        # can not back boolean column to zarr store
        adata.var["cols_used"] = adata.var["cols_used"].astype(int)
    # otherwise clusters can not be visualized in napari-spatialdata
    adata.obs[ClusteringKey._METACLUSTERING_KEY.value] = adata.obs[ClusteringKey._METACLUSTERING_KEY.value].astype(
        "category"
    )
    adata.obs[ClusteringKey._CLUSTERING_KEY.value] = adata.obs[ClusteringKey._CLUSTERING_KEY.value].astype("category")

    return adata, fsom
