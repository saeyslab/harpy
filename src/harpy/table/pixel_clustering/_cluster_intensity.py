from __future__ import annotations

import os
import uuid
from collections.abc import Iterable
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
from anndata import AnnData
from spatialdata import SpatialData

from harpy.image._image import _get_spatial_element
from harpy.table._allocation_intensity import allocate_intensity
from harpy.table._preprocess import preprocess_proteomics
from harpy.table._table import add_table_layer
from harpy.utils._keys import _CELL_INDEX, _CELLSIZE_KEY, _INSTANCE_KEY, _RAW_COUNTS_KEY, ClusteringKey
from harpy.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def cluster_intensity(
    sdata: SpatialData,
    mapping: pd.Series,  # pandas series with at the index the clusters and as values the metaclusters # TODO maybe should also allow passing None, and calculate mapping from provided som labels layer and meta cluster labels layer
    img_layer: str | Iterable[str],
    labels_layer: str | Iterable[str],
    output_layer: str,
    to_coordinate_system: str | Iterable[str] = "global",
    channels: int | str | Iterable[int] | Iterable[str] | None = None,
    chunks: str | int | tuple[int, ...] | None = 10000,
    overwrite=False,
) -> SpatialData:
    """
    Calculates average intensity of each channel in `img_layer` per SOM cluster as available in the `labels_layer`, and saves it as a table layer in `sdata` as `output_layer`. Average intensity per metacluster is calculated using the `mapping`.

    This function computes average intensity for each SOM cluster identified in the `labels_layer` and stores the results in a new table layer (`output_layer`).
    Average intensity per metacluster is added to `sdata.tables[output_layer].uns`.
    The intensity calculation can be subset by channels and adjusted for chunk size for efficient processing. SOM clusters can be calculated using `harpy.im.flowsom`.

    Parameters
    ----------
    sdata
        The input SpatialData object.
    mapping
        A pandas Series mapping SOM cluster IDs (index) to metacluster IDs (values).
    img_layer
        The image layer of `sdata` from which the intensity is calculated.
    labels_layer
        The labels layer in `sdata` that contains the SOM cluster IDs. I.e. the `output_layer_clusters` labels layer obtained through `harpy.im.flowsom`.
    output_layer
        The output table layer in `sdata` where results are stored.
    to_coordinate_system
        The coordinate system that holds `img_layer` and `labels_layer`.
        If `img_layer` and `labels_layer` are provided as a list,
        elements in `to_coordinate_system` are the respective coordinate systems that holds the elements in `img_layer` and `labels_layer`.
    channels
        Specifies the channels to be included in the intensity calculation.
    chunks
        Chunk sizes for processing. If provided as a tuple, it should contain chunk sizes for `c`, `(z)`, `y`, `x`.
    overwrite
        If True, overwrites the `output_layer` if it already exists in `sdata`.

    Returns
    -------
    The input `sdata` with the new table layer added.

    Raises
    ------
    AssertionError
        If number of provided `img_layer`, `labels_layer` and `to_coordinate_system` is not equal.
    AssertionError
        If some labels in `labels_layer` are not found in the provided mapping pandas Series.

    See Also
    --------
    harpy.im.flowsom : flowsom pixel clustering.
    """
    img_layer = list(img_layer) if isinstance(img_layer, Iterable) and not isinstance(img_layer, str) else [img_layer]
    labels_layer = (
        list(labels_layer)
        if isinstance(labels_layer, Iterable) and not isinstance(labels_layer, str)
        else [labels_layer]
    )
    to_coordinate_system = (
        list(to_coordinate_system)
        if isinstance(to_coordinate_system, Iterable) and not isinstance(to_coordinate_system, str)
        else [to_coordinate_system]
    )

    assert (
        len(img_layer) == len(labels_layer) == len(to_coordinate_system)
    ), "The number of provided 'img_layer', 'labels_layer' and 'to_coordinate_system' should be equal."

    for i, (_img_layer, _labels_layer, _to_coordinate_system) in enumerate(
        zip(img_layer, labels_layer, to_coordinate_system)
    ):
        se = _get_spatial_element(sdata, layer=_labels_layer)

        labels = da.unique(se.data).compute()

        assert np.all(
            np.in1d(labels[labels != 0], mapping.index.astype(int))
        ), f"Some labels labels layer {_labels_layer} could not be found in the provided pandas Series that maps SOM cluster ID's to metacluster IDs."

        # allocate the intensity to via the clusters labels layer

        if i == 0:
            append = False
        else:
            append = True
        sdata = allocate_intensity(
            sdata,
            img_layer=_img_layer,
            labels_layer=_labels_layer,
            output_layer=output_layer,
            channels=channels,
            to_coordinate_system=_to_coordinate_system,
            chunks=chunks,
            append=append,
            overwrite=overwrite,
        )

    # for size normalization of cluster intensities
    sdata = preprocess_proteomics(
        sdata,
        labels_layer=labels_layer,
        table_layer=output_layer,
        output_layer=output_layer,
        size_norm=True,
        log1p=False,
        scale=False,
        calculate_pca=False,
        overwrite=True,
    )

    # we are interested in the non-normalized counts (to account for multiple fov's)
    array = sdata.tables[output_layer].layers[_RAW_COUNTS_KEY]
    df = pd.DataFrame(array)
    df[_INSTANCE_KEY] = sdata.tables[output_layer].obs[_INSTANCE_KEY].values
    df = df.groupby(_INSTANCE_KEY).sum()
    df.sort_index(inplace=True)
    df_obs = sdata.tables[output_layer].obs.copy()
    df_obs = df_obs.groupby(_INSTANCE_KEY).sum(_CELLSIZE_KEY)
    df_obs.sort_index(inplace=True)
    df = df * (100 / df_obs.values)

    var = pd.DataFrame(index=sdata[output_layer].var_names)
    var.index = var.index.map(str)
    var.index.name = "channels"

    cells = pd.DataFrame(index=df.index)
    _uuid_value = str(uuid.uuid4())[:8]
    cells.index = cells.index.map(lambda x: f"{x}_{output_layer}_{_uuid_value}")
    cells.index.name = _CELL_INDEX
    adata = AnnData(X=df.values, obs=cells, var=var)
    adata.obs[_INSTANCE_KEY] = df.index

    adata.obs[_CELLSIZE_KEY] = df_obs[
        _CELLSIZE_KEY
    ].values  # for multiple fov's this is the sum of the size over all the clusters

    # append metacluster labels to the table using the mapping
    mapping = mapping.reset_index().rename(columns={"index": _INSTANCE_KEY})  # _INSTANCE_KEY is the cluster ID
    mapping[_INSTANCE_KEY] = mapping[_INSTANCE_KEY].astype(int)
    old_index = adata.obs.index
    adata.obs = pd.merge(adata.obs.reset_index(), mapping, on=[_INSTANCE_KEY], how="left")
    adata.obs.index = old_index
    adata.obs = adata.obs.drop(columns=[_CELL_INDEX])

    assert (
        not adata.obs[ClusteringKey._METACLUSTERING_KEY.value].isna().any()
    ), "Not all SOM cluster IDs could be linked to a metacluster."

    # calculate mean intensity per metacluster
    df = adata.to_df().copy()
    df[[_CELLSIZE_KEY, ClusteringKey._METACLUSTERING_KEY.value]] = adata.obs[
        [_CELLSIZE_KEY, ClusteringKey._METACLUSTERING_KEY.value]
    ].copy()

    if channels is None:
        channels = adata.var.index.values
    else:
        channels = list(channels) if isinstance(channels, Iterable) and not isinstance(channels, str) else [channels]

    df = _mean_intensity_per_metacluster(df, channels=channels)

    adata.uns[f"{ClusteringKey._METACLUSTERING_KEY.value}"] = df

    sdata = add_table_layer(
        sdata,
        adata=adata,
        output_layer=output_layer,
        region=None,  # can not be linked to a region, because it contains average over multiple labels layers (ID of the SOM clusters) in multiple fov scenario
        overwrite=True,
    )

    return sdata


def _mean_intensity_per_metacluster(df, channels: Iterable[str]):
    # Assuming df is your dataframe
    def weighted_mean(x, data, weight):
        """Calculate weighted mean for a column."""
        return (x * data[weight]).sum() / data[weight].sum()

    # Calculate weighted average for each marker per pixel_meta_cluster
    weighted_averages = df.groupby(
        ClusteringKey._METACLUSTERING_KEY.value,
    ).apply(
        lambda x: pd.Series(
            {col: weighted_mean(x[col], x, _CELLSIZE_KEY) for col in channels},
        ),
        include_groups=False,
    )

    return weighted_averages.reset_index()


def _export_to_ark_format(adata: AnnData, output: str | Path | None = None) -> pd.DataFrame:
    """Export avg intensity per SOM cluster calculated via `harpy.tb.cluster_intensity` to a csv file that can be visualized by the ark gui."""
    df = adata.to_df().copy()
    df["pixel_meta_cluster"] = adata.obs[ClusteringKey._METACLUSTERING_KEY.value].copy()
    df["pixel_som_cluster"] = adata.obs[_INSTANCE_KEY].copy()
    df["count"] = adata.obs[_CELLSIZE_KEY].copy()

    if output is not None:
        output_file = os.path.join(output, "average_intensities_SOM_clusters.csv")
        log.info(f"writing average intensities per SOM cluster to {output_file}")
        df.to_csv(output_file, index=False)

    return df
