from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from anndata import AnnData
from loguru import logger as log
from spatialdata import SpatialData

from harpy.table._table import ProcessTable, add_table
from harpy.table.cell_clustering._preprocess import cell_clustering_preprocess
from harpy.table.cell_clustering._utils import _get_mapping
from harpy.utils._keys import _CELL_INDEX, _CELLSIZE_KEY, _INSTANCE_KEY, _RAW_COUNTS_KEY, _REGION_KEY, ClusteringKey

try:
    import flowsom as fs

    from harpy.utils._flowsom import _flowsom

except ImportError:
    log.warning(
        "'flowsom' not installed, to use 'harpy.tb.flowsom', please install this library (https://git@github.com/saeyslab/FlowSOM_Python)."
    )


def flowsom(
    sdata: SpatialData,
    cells_labels_name: str | Iterable[str],
    cluster_labels_name: str | Iterable[str],
    output_table_name: str,
    q: float | None = 0.999,
    chunks: str | int | tuple[int, ...] | None = None,
    n_clusters: int = 20,
    index_names_var: Iterable[str] | None = None,
    index_positions_var: Iterable[int] | None = None,
    random_state: int = 100,
    region_key: str = _REGION_KEY,
    instance_key: str = _INSTANCE_KEY,
    cell_index_name: str = _CELL_INDEX,
    instance_size_key: str = _CELLSIZE_KEY,
    raw_counts_key: str = _RAW_COUNTS_KEY,
    overwrite: bool = False,
    **kwargs,  # keyword arguments for _flowsom
) -> tuple[SpatialData, fs.FlowSOM]:
    """
    Run FlowSOM cell clustering on pixel-cluster-derived cell features.

    Prepare the data obtained from pixel clustering for cell clustering (see
    :func:`~harpy.tb.cell_clustering_preprocess`) and execute FlowSOM on the
    resulting table layer (`output_table_name`) of the SpatialData object.

    This function applies the FlowSOM clustering algorithm (via
    :class:`flowsom.FlowSOM`) on spatial data contained in a SpatialData object.
    The algorithm organizes data into self-organizing maps and then clusters these maps, grouping them into `n_clusters`.
    The results of this clustering are added to a table layer in the `sdata` object.

    Typically, one would first process `sdata` via
    :func:`~harpy.im.pixel_clustering_preprocess` and :func:`~harpy.im.flowsom`
    before using this function.

    Parameters
    ----------
    sdata
        The input SpatialData object.
    cells_labels_name
        The labels layer(s) in `sdata` that contain cell segmentation masks.
        These masks should be previously generated using :func:`~harpy.im.segment`.
        If a list of labels layers is provided, they will be clustered together (e.g. multiple samples).
    cluster_labels_name
        The labels layer(s) in `sdata` that contain metacluster or SOM cluster masks.
        These should be obtained via :func:`~harpy.im.flowsom`.
    output_table_name
        The output table layer in `sdata` where results of the clustering and metaclustering will be stored.
    q
        Quantile used for normalization. If specified, each pixel SOM/meta
        cluster column in `output_table_name` is normalized by this quantile prior
        to FlowSOM clustering. Values are multiplied by 100 after normalization.
    chunks
        Chunk sizes for processing the data. If provided as a tuple, it should detail chunk sizes for each dimension `(z)`, `y`, `x`.
    n_clusters
        The number of metaclusters to form from the self-organizing maps.
    index_names_var
        Specifies the variable names to be used from `sdata.tables[output_table_name].var`
        for clustering. If `None`, `index_positions_var` is used if not `None`.
    index_positions_var
        Specifies the positions of variables to be used from
        `sdata.tables[output_table_name].var`. Used if `index_names_var` is `None`.
    random_state
        A random state for reproducibility of the clustering.
    instance_key
        Instance key. The name of the column in :class:`~anndata.AnnData` table `.obs` that will hold the instance ids.
    region_key
        Region key. The name of the column in :class:`~anndata.AnnData` table `.obs`
        that will hold the name of the element(s) that are annotated by the resulting table.
    cell_index_name
        The name of the index of the resulting :class:`~anndata.AnnData` table.
    instance_size_key
        The key in the :class:`~anndata.AnnData` table `.obs` that will hold the size of the instances (obtained from `cells_labels_name`).
    raw_counts_key
        Name of the :class:`~anndata.AnnData` layer where the non-preprocessed counts will be stored.
    overwrite
        If True, overwrites the existing data in `output_table_name` if it already exists.
    **kwargs
        Additional keyword arguments passed to :class:`flowsom.FlowSOM`.

    Returns
    -------
    tuple:

        - The updated `sdata` with the clustering results added.

        - An instance of :class:`flowsom.FlowSOM` containing the trained FlowSOM model.

    See Also
    --------
    :func:`~harpy.im.flowsom` : FlowSOM pixel clustering.
    :func:`~harpy.tb.cell_clustering_preprocess` : Prepare data for cell clustering.
    """
    # first do preprocessing (this creates an AnnData table)
    sdata = cell_clustering_preprocess(
        sdata,
        cells_labels_name=cells_labels_name,
        cluster_labels_name=cluster_labels_name,
        output_table_name=output_table_name,
        q=q,
        chunks=chunks,
        region_key=region_key,
        instance_key=instance_key,
        cell_index_name=cell_index_name,
        instance_size_key=instance_size_key,
        raw_counts_key=raw_counts_key,
        overwrite=overwrite,
    )

    process_table_instance = ProcessTable(sdata, labels_name=cells_labels_name, table_name=output_table_name)
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

    sdata = add_table(
        sdata,
        adata=adata,
        output_table_name=output_table_name,
        region=process_table_instance.labels_name,
        instance_key=process_table_instance.instance_key,
        region_key=process_table_instance.region_key,
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
