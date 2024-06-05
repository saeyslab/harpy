from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from anndata import AnnData
from spatialdata import SpatialData

from sparrow.table._table import ProcessTable, _add_table_layer
from sparrow.table.cell_clustering._preprocess import cell_clustering_preprocess
from sparrow.table.cell_clustering._utils import _get_mapping
from sparrow.utils._flowsom import _flowsom
from sparrow.utils._keys import ClusteringKey
from sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

try:
    import flowsom as fs
except ImportError:
    log.warning("'flowsom' not installed, 'sp.tb.flowsom' will not be available.")


def flowsom(
    sdata: SpatialData,
    labels_layer_cells: str | Iterable[str],
    labels_layer_clusters: str | Iterable[str],
    output_layer: str,
    q: float | None = 0.999,
    chunks: str | int | tuple[int, ...] | None = None,
    n_clusters: int = 20,
    index_names_var: Iterable[str] | None = None,
    index_positions_var: Iterable[int] | None = None,
    random_state: int = 100,
    overwrite: bool = False,
    **kwargs,  # keyword arguments for _flowsom
) -> tuple[SpatialData, fs.FlowSOM]:
    """
    Prepares the data obtained from pixel clustering for cell clustering (see docstring of `sp.tb.cell_clustering_preprocess`) and then executes the FlowSOM clustering algorithm on the resulting table layer (`output_layer`) of the SpatialData object.

    This function applies the FlowSOM clustering algorithm (via `fs.FlowSOM`) on spatial data contained in a SpatialData object.
    The algorithm organizes data into self-organizing maps and then clusters these maps, grouping them into `n_clusters`.
    The results of this clustering are added to a table layer in the `sdata` object.

    Typically one would first process `sdata` via `sp.im.pixel_clustering_preprocess` and `sp.im.flowsom` before using this function.

    Parameters
    ----------
    sdata
        The input SpatialData object.
    labels_layer_cells
        The labels layer(s) in `sdata` that contain cell segmentation masks. These masks should be previously generated using `sp.im.segment`.
        If a list of labels layers is provided, they will be clustered together (e.g. multiple samples).
    labels_layer_clusters
        The labels layer(s) in `sdata` that contain metacluster or SOM cluster masks. These should be obtained via `sp.im.flowsom`.
    output_layer
        The output table layer in `sdata` where results of the clustering and metaclustering will be stored.
    q
        Quantile used for normalization. If specified, each pixel SOM/meta cluster column in `output_layer` is normalized by this quantile prior to flowsom clustering. Values are multiplied by 100 after normalization.
    chunks
        Chunk sizes for processing the data. If provided as a tuple, it should detail chunk sizes for each dimension `(z)`, `y`, `x`.
    n_clusters
        The number of metaclusters to form from the self-organizing maps.
    index_names_var
        Specifies the variable names to be used from `sdata.tables[table_layer].var` for clustering. If None, `index_positions_var` will be used if not None.
    index_positions_var
        Specifies the positions of variables to be used from `sdata.tables[table_layer].var` for clustering. Used if `index_names_var` is None.
    random_state
        A random state for reproducibility of the clustering.
    overwrite
        If True, overwrites the existing data in `output_layer` if it already exists.
    **kwargs
        Additional keyword arguments passed to the `fs.FlowSOM` clustering algorithm.

    Returns
    -------
    tuple
        - The updated `sdata` with the clustering results added.
        - An instance of `fs.FlowSOM` containing the trained FlowSOM model.

    See Also
    --------
    sparrow.im.flowsom : flowsom pixel clustering
    sparrow.tb.cell_clustering_preprocess : prepares data for cell clustering.
    """
    # first do preprocessing
    sdata = cell_clustering_preprocess(
        sdata,
        labels_layer_cells=labels_layer_cells,
        labels_layer_clusters=labels_layer_clusters,
        output_layer=output_layer,
        q=q,
        chunks=chunks,
        overwrite=overwrite,
    )

    process_table_instance = ProcessTable(sdata, labels_layer=labels_layer_cells, table_layer=output_layer)
    adata = process_table_instance._get_adata(index_names_var=index_names_var, index_positions_var=index_positions_var)

    adata, fsom = _flowsom(
        adata,
        n_clusters=n_clusters,
        seed=random_state,
        **kwargs,
    )

    _keys = [ClusteringKey._CLUSTERING_KEY.value, ClusteringKey._METACLUSTERING_KEY.value]
    mapping = _get_mapping(adata, keys=_keys)

    # calculate the mean cluster 'intensity' both for the _CLUSTERING_KEY and _METACLUSTERING_KEY
    for _key in _keys:
        df = _grouped_obs_mean(adata, group_key=_key)
        df = df.transpose()
        df.index.name = _key
        df.columns = adata.var_names

        df = pd.merge(
            df,
            adata.obs[_key].value_counts(),
            how="left",
            left_index=True,
            right_index=True,
        )
        if _key == ClusteringKey._CLUSTERING_KEY.value:
            df = pd.merge(df, mapping, how="left", left_index=True, right_index=True)

        log.info(f"Adding mean cluster intensity to '.uns['{_key}']'")
        adata.uns[_key] = df

    sdata = _add_table_layer(
        sdata,
        adata=adata,
        output_layer=output_layer,
        region=process_table_instance.labels_layer,
        overwrite=overwrite,
    )

    return sdata, fsom


def _grouped_obs_mean(adata: AnnData, group_key: str, layer: str = None) -> pd.DataFrame:
    if layer is not None:
        getX = lambda x: x.layers[layer]
    else:
        getX = lambda x: x.X

    grouped = adata.obs.groupby(group_key, observed=False)
    columns = list(grouped.groups.keys())
    out = pd.DataFrame(
        np.zeros((adata.shape[1], len(grouped)), dtype=np.float64), columns=columns, index=adata.var_names
    )

    for group, idx in grouped.indices.items():
        X = getX(adata[idx])
        out[group] = np.ravel(X.mean(axis=0, dtype=np.float64))
    return out
