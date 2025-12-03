from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from loguru import logger as log
from spatialdata import SpatialData

from harpy.image._image import _get_spatial_element
from harpy.table._table import ProcessTable, add_table_layer
from harpy.utils._aggregate import _get_mask_area
from harpy.utils._keys import _CELLSIZE_KEY, _INSTANCE_KEY, _REGION_KEY
from harpy.utils.utils import _make_list


def cluster_intensity(
    sdata: SpatialData,
    table_layer: str,
    labels_layer: str | Iterable[str],
    output_layer: str,
    cluster_key: str,
    cluster_key_uns: str | None = None,
    layer_mean_intensities: str | None = None,
    instance_size_key: str = _CELLSIZE_KEY,
    instance_key: str = _INSTANCE_KEY,
    region_key: str = _REGION_KEY,
    overwrite: bool = False,
) -> SpatialData:
    """
    Calculates weighted (by instance size) average intensity per cluster.

    Calculates weighted average intensity for each cluster, and stores the result in `sdata.tables[output_layer].uns[cluster_key_uns]`.
    The intensities in `sdata.tables[table_layer].X` or `sdata.tables[table_layer].layers[layer_mean_intensities]`
    should contain the mean (by instance size) intensities for each label in `labels_layer`.

    Parameters
    ----------
    sdata
        SpatialData object.
    table_layer
        The table layer containing the mean intensities per instance in 'sdata.tables[table_layer].X' or
        'sdata.tables[table_layer].layers[layer_mean_intensities]' if `layer_mean_intensities` is not `None`; and the `cluster_key` in `sdata.tables[table_layer].obs`.
        Mean intensities can be calculated using `harpy.tb.allocate_intensity(..., mode="mean",...)`.
        See docstring of `harpy.pl.cluster_intensity_heatmap` for an example.
    labels_layer
        The labels layer(s) of `sdata` used to select the instances via the `region_key` in `sdata.tables[table_layer].obs`.
        Note that if `output_layer` is equal to `table_layer` and `overwrite` is `True`,
        instances in `sdata.tables[table_layer]` linked to other `labels_layer` (via the `region_key`), will be removed from `sdata.tables[table_layer]`.
        If a list of labels layers is provided, intensities per cluster will be calculated over all labels layers, which is usefull in the multi-sample scenario.
    layer_mean_intensities
        Layer of `sdata.tables[table_layer]` holding the mean intensities per instance. If not specified, it is assumed 'sdata.tables[table_layer].X' holds the mean intensity values per instance.
    output_layer
        The output table layer in `sdata` where results are stored.
    cluster_key
        The cluster key in `sdata.tables[table_layer].obs`.
    cluster_key_uns
        The key in `sdata.tables[table_layer].uns` where the weighted mean intensities per cluster will be stored.
        If not provided `cluster_key_uns` is set to `{cluster_key}_weighted_intensity`.
    instance_size_key
        The key in `sdata.tables[table_layer].obs` that holds instance size.
        Instance sizes will be calculated from `labels_layer` if not found in `.obs`.
    instance_key
        Instance key
    region_key
        Region key
    overwrite
        If `True`, overwrites the `output_layer` if it already exists in `sdata`.

    Returns
    -------
    SpatialData object with `output_layer` containing weighted mean intensity per cluster at `.uns[cluster_key_uns]`.

    Examples
    --------
    See docstring of `harpy.pl.cluster_intensity_heatmap` for an example.


    See Also
    --------
    harpy.tb.allocate_intensity : calculates total intensity per instance per channel.
    harpy.tb.preprocess_proteomics: calculates instance size and normalizes intensity by instance size.
    harpy.pl.cluster_intensity_heatmap: plot heatmap of mean intensity per cluster.
    """
    if cluster_key_uns is None:
        cluster_key_uns = f"{cluster_key}_weighted_intensity"
    log.info(
        f"Weighted (by instance size) average intensity per cluster (cluster key: '{cluster_key}') "
        f"will be stored in 'sdata[{output_layer}].uns[{cluster_key_uns}]'."
    )
    labels_layer = _make_list(labels_layer)
    # get the adata
    process_table_instance = ProcessTable(sdata, labels_layer=labels_layer, table_layer=table_layer)
    adata = process_table_instance._get_adata()
    if cluster_key not in adata.obs.columns:
        raise ValueError(f"Cluster key '{cluster_key}' not found in 'sdata[{table_layer}].obs'.")
    if cluster_key_uns in adata.uns.keys():
        log.warning(
            f"Key '{cluster_key_uns}' found in sdata[{table_layer}].uns. Object at that location will be removed and recalculated."
        )
        adata.uns.pop(cluster_key_uns)

    # we do not want to loose the index (_CELL_INDEX)
    old_index = adata.obs.index
    index_name = adata.obs.index.name or "index"
    if instance_size_key not in adata.obs.columns:
        log.warning(
            f"Column with name '{instance_size_key}' not found in 'sdata[{table_layer}].obs', "
            f"calculating instance size for all instances in {labels_layer}."
        )
        for i, _labels_layer in enumerate(process_table_instance.labels_layer):
            log.info(f"Calculating instance size from provided labels layer '{_labels_layer}'")
            se = _get_spatial_element(sdata, layer=_labels_layer)
            _shapesize = _get_mask_area(se.data if se.data.ndim == 3 else se.data[None, ...])
            _shapesize[region_key] = _labels_layer
            if i == 0:
                shapesize = _shapesize
            else:
                shapesize = pd.concat([shapesize, _shapesize], ignore_index=True)
        # note that we checked that adata.obs[ _INSTANCE_KEY ] is unique for given region (see self._get_adata())
        adata.obs = pd.merge(adata.obs.reset_index(), shapesize, on=[instance_key, region_key], how="left")
        adata.obs.index = old_index
        adata.obs = adata.obs.drop(columns=[index_name])
    if layer_mean_intensities is None:
        df = pd.DataFrame(adata.X, columns=adata.var_names)
    else:
        df = pd.DataFrame(adata.layers[layer_mean_intensities], columns=adata.var_names)
    df.index = adata.obs.index
    channels = df.columns
    df[instance_size_key] = adata.obs[instance_size_key]
    df[cluster_key] = adata.obs[cluster_key]

    df = _mean_intensity_per_cluster_key(df, channels=channels, cluster_key=cluster_key, weight_key=instance_size_key)

    adata.uns[cluster_key_uns] = df

    sdata = add_table_layer(
        sdata,
        adata=adata,
        output_layer=output_layer,
        region=process_table_instance.labels_layer,
        overwrite=overwrite,
    )

    return sdata


def _mean_intensity_per_cluster_key(
    df: pd.DataFrame, channels: Iterable[str], cluster_key: str, weight_key: str = _CELLSIZE_KEY
) -> pd.DataFrame:
    def weighted_mean(x, data, weight):
        """Calculate weighted mean for a column."""
        return (x * data[weight]).sum() / data[weight].sum()

    # Calculate weighted average for each marker per cluster_key
    weighted_averages = df.groupby(
        cluster_key,
    ).apply(
        lambda x: pd.Series(
            {col: weighted_mean(x[col], x, weight_key) for col in channels},
        ),
        include_groups=False,
    )

    return weighted_averages.reset_index()
