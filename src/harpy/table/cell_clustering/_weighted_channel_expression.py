import numpy as np
import pandas as pd
from pandas.testing import assert_index_equal
from spatialdata import SpatialData
from spatialdata.models import TableModel

from harpy.table._table import ProcessTable, add_table_layer
from harpy.table.cell_clustering._utils import _get_mapping
from harpy.utils._keys import _CELLSIZE_KEY, _INSTANCE_KEY, _RAW_COUNTS_KEY, ClusteringKey
from harpy.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def weighted_channel_expression(
    sdata: SpatialData,
    table_layer_cell_clustering: str,
    table_layer_pixel_cluster_intensity: str,
    output_layer: str,
    clustering_key: ClusteringKey = ClusteringKey._METACLUSTERING_KEY,
    overwrite: bool = False,
    # specify whether the metaclustering or SOM clustering labels layer of pixel clustering results was used as input for harpy.tb.flowsom, i.e. key that was used for pixel clustering
) -> SpatialData:
    """
    Calculation of weighted channel expression in the context of cell clustering.

    Calculates the average channel expression (via `table_layer_pixel_cluster_intensity`) for each cell weighted by pixel SOM/META cluster count (via `table_layer_cell_clustering`).
    Values are normalized by the size of the cell.

    Average marker expression for each cell weighted by pixel cluster count is added to `sdata.tables[output_layer].obs`.
    Mean over the obtained cell clusters (both SOM and meta clusters) of the average marker expression for each cell weighted by pixel cluster count is added to `sdata.tables[output_layer].uns`.

    This function should be run after running `harpy.tb.flowsom` and `harpy.tb.cluster_intensity`.

    Parameters
    ----------
    sdata
        The input SpatialData object containing the necessary data tables.
    table_layer_cell_clustering
        The name of the table layer in `sdata` where FlowSOM cell clustering results are stored (obtained via 'harpy.tb.flowsom').
        This layer should contain the cell cluster labels derived from the FlowSOM clustering algorithm and the non-normalized pixel cluster counts in `.layers[ _RAW_COUNTS_KEY ]`, as obtained after running `harpy.tb.flowsom`.
    table_layer_pixel_cluster_intensity
        The name of the table layer in `sdata` containing pixel cluster intensity values as obtained by running `harpy.tb.cluster_intensity`.
        These intensities are used to calculate the weighted expression of each channel for the cell clusters.
    output_layer
        The name of the output table layer in `sdata` where the results of the weighted channel expression computation will be stored.
    clustering_key
        Specifies the key that was used for pixel clustering, indicating whether metaclustering or SOM clustering labels were used as input for flowsom cell clustering (`harpy.tb.flowsom`).
    overwrite
        If True, overwrites any existing data in the `output_layer` if it already exists.

    Returns
    -------
    The updated `sdata` object with the results of the weighted channel expression added to the specified `output_layer`.

    See Also
    --------
    harpy.tb.flowsom : flowsom cell clustering
    harpy.tb.cluster_intensity : calculates average intensity SOM/meta cluster (pixel clusters).
    """
    # subset over all _labels_layer in 'table_layer_cell_clustering'
    _labels_layer = [*sdata[table_layer_cell_clustering].uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY]]
    process_table_clustering = ProcessTable(
        sdata, labels_layer=_labels_layer, table_layer=table_layer_cell_clustering
    )  # get all of the labels layers, to keep API not too complex, do not allow subsetting labels layers
    adata_cell_clustering = process_table_clustering._get_adata()
    missing_keys = [
        key
        for key in [ClusteringKey._CLUSTERING_KEY.value, ClusteringKey._METACLUSTERING_KEY.value]
        if key not in adata_cell_clustering.obs
    ]
    assert not missing_keys, (
        "Please first run 'harpy.tb.flosom' before running this function. "
        f"Missing keys in '.obs' of table layer '{table_layer_cell_clustering}' are: {', '.join(missing_keys)}."
    )
    index_columns = adata_cell_clustering.to_df().columns.astype(int)
    index_columns.name = None

    cell_counts_matrix = adata_cell_clustering.layers[_RAW_COUNTS_KEY]

    process_table_intensity = ProcessTable(sdata, labels_layer=None, table_layer=table_layer_pixel_cluster_intensity)
    adata_cluster_intensity = process_table_intensity._get_adata()

    if clustering_key.value == ClusteringKey._METACLUSTERING_KEY.value:
        df_intensity = adata_cluster_intensity.uns[clustering_key.value].copy()
        df_intensity.set_index(clustering_key.value, inplace=True)
        df_intensity.index = df_intensity.index.astype(int)
    elif clustering_key.value == ClusteringKey._CLUSTERING_KEY.value:
        df_intensity = adata_cluster_intensity.to_df().copy()
        df_intensity[clustering_key.value] = adata_cluster_intensity.obs[_INSTANCE_KEY]
        df_intensity.set_index(clustering_key.value, inplace=True)
    else:
        raise ValueError(
            f"'clustering_key' should either be {ClusteringKey._METACLUSTERING_KEY} or {ClusteringKey._CLUSTERING_KEY}."
        )

    assert (
        df_intensity.shape[0] == cell_counts_matrix.shape[1]
    ), f"Average intensities for {clustering_key} provided via table '{table_layer_pixel_cluster_intensity}' should contain as many rows as there are columns in table '{table_layer_cell_clustering}'."
    df_intensity.index.name = None
    assert_index_equal(df_intensity.index.sort_values(), index_columns.sort_values())
    df_intensity = df_intensity.reindex(index_columns)

    data = np.matmul(cell_counts_matrix, df_intensity.values) / adata_cell_clustering.obs[_CELLSIZE_KEY].values.reshape(
        -1, 1
    )
    df = pd.DataFrame(data)
    df.columns = adata_cluster_intensity.var_names
    df.index = adata_cell_clustering.obs_names

    _keys = [ClusteringKey._CLUSTERING_KEY.value, ClusteringKey._METACLUSTERING_KEY.value]

    mapping = _get_mapping(adata_cell_clustering, keys=_keys)

    for _key in _keys:
        df_cluster_mean = df.assign(cluster=adata_cell_clustering.obs[_key]).groupby("cluster", observed=False).mean()
        df_cluster_mean.index.name = _key
        if _key == ClusteringKey._CLUSTERING_KEY.value:
            # add the mapping of clusters to metaclusters to the df_cluster_mean dataframe
            df_cluster_mean = pd.merge(df_cluster_mean, mapping, how="left", left_index=True, right_index=True)
        _uns_name = f"{_key}_{adata_cluster_intensity.var_names.name}"
        log.info(
            f"Adding mean over obtained cell clusters '({_key})' of the average marker expression for each cell weighted by pixel cluster count to '.uns[ '{_uns_name}' ]' of table layer '{output_layer}'"
        )
        adata_cell_clustering.uns[_uns_name] = df_cluster_mean

    log.info(
        f"Adding average marker expression for each cell weighted by pixel cluster count to '.obs' of table layer '{output_layer}'"
    )
    adata_cell_clustering.obs = pd.concat([adata_cell_clustering.obs, df], axis=1)

    sdata = add_table_layer(
        sdata,
        adata=adata_cell_clustering,
        output_layer=output_layer,
        region=process_table_clustering.labels_layer,
        overwrite=overwrite,
    )

    return sdata
