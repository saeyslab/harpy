from __future__ import annotations

import os
import uuid
from collections.abc import Iterable
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
from anndata import AnnData
from loguru import logger as log
from spatialdata import SpatialData

from harpy.image._image import _get_spatial_element
from harpy.table._allocation_intensity import allocate_intensity
from harpy.table._preprocess import preprocess_proteomics
from harpy.table._table import add_table
from harpy.utils._keys import _RAW_COUNTS_KEY, ClusteringKey


def cluster_intensity_SOM(
    sdata: SpatialData,
    mapping: pd.Series,  # pandas series with at the index the clusters and as values the metaclusters # TODO maybe should also allow passing None, and calculate mapping from provided som labels element and meta cluster labels element
    image_name: str | Iterable[str],
    labels_name: str | Iterable[str],
    output_table_name: str,
    to_coordinate_system: str | Iterable[str] = "global",
    channels: int | str | Iterable[int] | Iterable[str] | None = None,
    chunks: str | int | tuple[int, ...] | None = 10000,
    instance_key: str = "SOM_cluster_ID",
    instance_size_key: str = "SOM_cluster_size",
    index_name: str = "SOM_cluster_ID_index",
    overwrite=False,
) -> SpatialData:
    """
    Calculates average intensity of each channel in `image_name` per SOM cluster as available in the `labels_name`, and saves it as a table element in `sdata` as `output_table_name`. Average intensity per metacluster is calculated using the `mapping`.

    This function computes average intensity for each SOM cluster identified in the `labels_name` and stores the results in a new table element (`output_table_name`).
    Average intensity per metacluster is added to `sdata.tables[output_table_name].uns`.
    The intensity calculation can be subset by channels and adjusted for chunk size for efficient processing. SOM clusters can be calculated using `harpy.im.flowsom`.

    Parameters
    ----------
    sdata
        The input SpatialData object.
    mapping
        A pandas Series mapping SOM cluster IDs (index) to metacluster IDs (values).
    image_name
        The image element of `sdata` from which the intensity is calculated.
    labels_name
        The labels element in `sdata` that contains the SOM cluster IDs. I.e. the `output_cluster_labels_name` labels element obtained through `harpy.im.flowsom`.
    output_table_name
        The output table element in `sdata` where results are stored.
    to_coordinate_system
        The coordinate system that holds `image_name` and `labels_name`.
        If `image_name` and `labels_name` are provided as a list,
        elements in `to_coordinate_system` are the respective coordinate systems that holds the elements in `image_name` and `labels_name`.
    channels
        Specifies the channels to be included in the intensity calculation.
    chunks
        Chunk sizes for processing. If provided as a `tuple`, it should contain chunk sizes for `c`, `(z)`, `y`, `x`.
    instance_key
        Instance key. The name of the column in :class:`~anndata.AnnData` table `.obs` that will hold the instance ids (SOM cluster IDs).
    instance_size_key
        The key in the :class:`~anndata.AnnData` table `.obs` that will hold the size of the instances.
    index_name
        The name of the index of the resulting :class:`~anndata.AnnData` table.
    overwrite
        If True, overwrites the `output_table_name` if it already exists in `sdata`.

    Returns
    -------
    The input `sdata` with the new table element added.

    Raises
    ------
    AssertionError
        If number of provided `image_name`, `labels_name` and `to_coordinate_system` is not equal.
    AssertionError
        If some labels in `labels_name` are not found in the provided mapping pandas Series.

    See Also
    --------
    harpy.im.flowsom : flowsom pixel clustering.
    """
    image_name = (
        list(image_name) if isinstance(image_name, Iterable) and not isinstance(image_name, str) else [image_name]
    )
    labels_name = (
        list(labels_name) if isinstance(labels_name, Iterable) and not isinstance(labels_name, str) else [labels_name]
    )
    to_coordinate_system = (
        list(to_coordinate_system)
        if isinstance(to_coordinate_system, Iterable) and not isinstance(to_coordinate_system, str)
        else [to_coordinate_system]
    )

    assert len(image_name) == len(labels_name) == len(to_coordinate_system), (
        "The number of provided 'image_name', 'labels_name' and 'to_coordinate_system' should be equal."
    )

    for i, (_image_name, _labels_name, _to_coordinate_system) in enumerate(
        zip(image_name, labels_name, to_coordinate_system, strict=True)
    ):
        se = _get_spatial_element(sdata, element_name=_labels_name)

        labels = da.unique(se.data).compute()

        assert np.all(np.isin(labels[labels != 0], mapping.index.astype(int))), (
            f"Some labels labels element {_labels_name} could not be found in the provided pandas Series that maps SOM cluster ID's to metacluster IDs."
        )

        # allocate the intensity to via the clusters labels element

        if i == 0:
            append = False
        else:
            append = True
        log.info(
            f"Start allocation of intensities of image element with name '{_image_name}' by labels in labels element with name '{_labels_name}'."
        )
        sdata = allocate_intensity(
            sdata,
            image_name=_image_name,
            labels_name=_labels_name,
            output_table_name=output_table_name,
            channels=channels,
            mode="sum",
            to_coordinate_system=_to_coordinate_system,
            chunks=chunks,
            append=append,
            calculate_center_of_mass=False,
            instance_key=instance_key,
            instance_size_key=instance_size_key,
            cell_index_name=index_name,
            overwrite=overwrite,
        )
        log.info(
            f"End allocation of image element with name '{_image_name}' and labels element with name '{_labels_name}'."
        )

    log.info("Start preprocessing.")
    # for size normalization of cluster intensities
    # note, we could also have done allocate_intensity( mode="sum", obs_stats="counts"), instead of also having to run preprocess_proteomics.
    sdata = preprocess_proteomics(
        sdata,
        labels_name=labels_name,
        table_name=output_table_name,
        output_table_name=output_table_name,
        size_norm=True,
        log1p=False,
        scale=False,
        calculate_pca=False,
        instance_size_key=instance_size_key,
        raw_counts_key=_RAW_COUNTS_KEY,
        overwrite=True,
    )
    log.info("End preprocessing.")

    # we are interested in the non-normalized counts (to account for multiple fov's)
    array = sdata.tables[output_table_name].layers[_RAW_COUNTS_KEY]
    df = pd.DataFrame(array)
    df[instance_key] = sdata.tables[output_table_name].obs[instance_key].values
    df = df.groupby(instance_key).sum()
    df.sort_index(inplace=True)
    df_obs = sdata.tables[output_table_name].obs.copy()
    df_obs = df_obs.groupby(instance_key).sum(instance_size_key)
    df_obs.sort_index(inplace=True)
    df = df * (100 / df_obs.values)

    var = pd.DataFrame(index=sdata[output_table_name].var_names)
    var.index = var.index.map(str)
    var.index.name = "channels"

    cells = pd.DataFrame(index=df.index)
    _uuid_value = str(uuid.uuid4())[:8]
    cells.index = cells.index.map(lambda x: f"{x}_{output_table_name}_{_uuid_value}")
    cells.index.name = index_name
    adata = AnnData(X=df.values, obs=cells, var=var)
    adata.obs[instance_key] = df.index

    adata.obs[instance_size_key] = df_obs[
        instance_size_key
    ].values  # for multiple fov's this is the sum of the size over all the clusters

    # append metacluster labels to the table using the mapping
    mapping = mapping.reset_index().rename(columns={"index": instance_key})  # instance_key is the cluster ID
    mapping[instance_key] = mapping[instance_key].astype(int)
    old_index = adata.obs.index
    adata.obs = pd.merge(adata.obs.reset_index(), mapping, on=[instance_key], how="left")
    adata.obs.index = old_index
    adata.obs = adata.obs.drop(columns=[index_name])

    assert not adata.obs[ClusteringKey._METACLUSTERING_KEY.value].isna().any(), (
        "Not all SOM cluster IDs could be linked to a metacluster."
    )

    # calculate mean intensity per metacluster
    df = adata.to_df().copy()
    df[[instance_size_key, ClusteringKey._METACLUSTERING_KEY.value]] = adata.obs[
        [instance_size_key, ClusteringKey._METACLUSTERING_KEY.value]
    ].copy()

    if channels is None:
        channels = adata.var.index.values
    else:
        channels = list(channels) if isinstance(channels, Iterable) and not isinstance(channels, str) else [channels]

    df = _mean_intensity_per_metacluster(df, channels=channels, instance_size_key=instance_size_key)

    adata.uns[f"{ClusteringKey._METACLUSTERING_KEY.value}"] = df

    sdata = add_table(
        sdata,
        adata=adata,
        output_table_name=output_table_name,
        region=None,  # can not be linked to a region, because it contains average over multiple labels elements (ID of the SOM clusters) in multiple fov scenario
        overwrite=True,
    )

    return sdata


def _mean_intensity_per_metacluster(df, channels: Iterable[str], instance_size_key: str = "SOM_cluster_size"):
    # Assuming df is your dataframe
    def weighted_mean(x, data, weight):
        """Calculate weighted mean for a column."""
        return (x * data[weight]).sum() / data[weight].sum()

    # Calculate weighted average for each marker per pixel_meta_cluster
    weighted_averages = df.groupby(
        ClusteringKey._METACLUSTERING_KEY.value,
    ).apply(
        lambda x: pd.Series(
            {col: weighted_mean(x[col], x, instance_size_key) for col in channels},
        ),
        include_groups=False,
    )

    return weighted_averages.reset_index()


def _export_to_ark_format(
    adata: AnnData,
    output: str | Path | None = None,
    instance_key: str = "SOM_cluster_ID",
    instance_size_key: str = "SOM_cluster_size",
) -> pd.DataFrame:
    """Export avg intensity per SOM cluster calculated via `harpy.tb.cluster_intensity` to a csv file that can be visualized by the ark gui."""
    df = adata.to_df().copy()
    df["pixel_meta_cluster"] = adata.obs[ClusteringKey._METACLUSTERING_KEY.value].copy()
    df["pixel_som_cluster"] = adata.obs[instance_key].copy()
    df["count"] = adata.obs[instance_size_key].copy()

    if output is not None:
        output_file = os.path.join(output, "average_intensities_SOM_clusters.csv")
        log.info(f"writing average intensities per SOM cluster to {output_file}")
        df.to_csv(output_file, index=False)

    return df
