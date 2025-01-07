from __future__ import annotations

import uuid
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import anndata as ad
import dask.array as da
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, ward
from scipy.sparse import csr_matrix
from scipy.stats import zscore
from sklearn.metrics.pairwise import cosine_similarity
from spatialdata import SpatialData, bounding_box_query
from spatialdata.models import TableModel

from harpy.image._image import _get_spatial_element
from harpy.utils._keys import _INSTANCE_KEY, _REGION_KEY, ClusteringKey
from harpy.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

try:
    import spatialdata_plot  # noqa: F401

except ImportError:
    log.warning("'spatialdata-plot' not installed, to use 'harpy.pl.plot_pixel_clusters', please install this library.")


def pixel_clusters(
    sdata: SpatialData,
    labels_layer: str,
    crd: tuple[int, int, int, int] | None = None,
    to_coordinate_system: str = "global",
    output: str | Path | None = None,
    render_labels_kwargs: Mapping[str, Any] = MappingProxyType({}),  # passed to pl.render_labels
    **kwargs,  # passed to pl.show() of spatialdata_plot
):
    """
    Visualize spatial distribution of pixel clusters based on labels in a `SpatialData` object, obtained using `harpy.im.flowsom`.

    Parameters
    ----------
    sdata
        The input SpatialData object.
    labels_layer
        The layer in `sdata` containing labels used to identify clusters.
    crd
        The coordinates for the region of interest in the format `(xmin, xmax, ymin, ymax)`. If `None`, the entire image is considered, by default `None`.
    to_coordinate_system
        Coordinate system to plot.
    output
        The path to save the generated plot. If `None`, the plot will be displayed directly using `plt.show()`.
    render_labels_kwargs
        Additional keyword arguments passed to `sdata.pl.render_labels`, such as `cmap` or `alpha`.
    **kwargs
        Additional keyword arguments passed to `sdata.pl.render_labels.pl.show`, such as `dpi` or `title`.

    Returns
    -------
    None
        The function generates and displays or saves a spatial plot of pixel clusters.

    Raises
    ------
    ValueError
        If the cropped spatial element derived from `crd` is `None`.

    See Also
    --------
    harpy.im.flowsom
    """
    se = _get_spatial_element(sdata, layer=labels_layer)

    labels_layer_crop = None
    if crd is not None:
        se_crop = bounding_box_query(
            se,
            axes=["x", "y"],
            min_coordinate=[crd[0], crd[2]],
            max_coordinate=[crd[1], crd[3]],
            target_coordinate_system=to_coordinate_system,
        )
        if se_crop is not None:
            labels_layer_crop = f"__labels_{uuid.uuid4()}__"
            sdata[labels_layer_crop] = se_crop
            se = se_crop
        else:
            raise ValueError(f"Cropped spatial element using crd '{crd}' is None.")

    unique_values = da.unique(se.data).compute()
    labels = unique_values[unique_values != 0]

    cluster_ids = labels

    intermediate_table_key = f"__value_clusters__{uuid.uuid4()}"

    # create a dummy anndata object, so we can plot cluster ID's spatially using spatialdata plot
    obs = pd.DataFrame({_INSTANCE_KEY: cluster_ids}, index=cluster_ids)
    obs.index = obs.index.astype(str)  # index needs to be str, otherwise anndata complains

    count_matrix = csr_matrix((labels.shape[0], 0))

    adata = ad.AnnData(X=count_matrix, obs=obs)
    adata.obs[_INSTANCE_KEY] = adata.obs[_INSTANCE_KEY].astype(int)

    adata.obs[_REGION_KEY] = labels_layer if labels_layer_crop is None else labels_layer_crop
    adata.obs[_REGION_KEY] = adata.obs[_REGION_KEY].astype("category")

    adata = TableModel.parse(
        adata=adata,
        region=labels_layer if labels_layer_crop is None else labels_layer_crop,
        region_key=_REGION_KEY,
        instance_key=_INSTANCE_KEY,
    )

    sdata[intermediate_table_key] = adata

    sdata[intermediate_table_key].obs[f"{_INSTANCE_KEY}_cat"] = (
        sdata[intermediate_table_key].obs[_INSTANCE_KEY].astype("category")
    )

    ax = sdata.pl.render_labels(
        labels_layer if labels_layer_crop is None else labels_layer_crop,
        table_name=intermediate_table_key,
        color=f"{_INSTANCE_KEY}_cat",
        **render_labels_kwargs,
    ).pl.show(
        **kwargs,
        return_ax=True,
    )
    if output is not None:
        ax.figure.savefig(output)
    else:
        plt.show()
    plt.close(ax.figure)

    del sdata.tables[intermediate_table_key]
    if labels_layer_crop is not None:
        del sdata.labels[labels_layer_crop]


def pixel_clusters_heatmap(
    sdata: SpatialData,
    table_layer: str,  # obtained via hp.tb.cluster_intensity
    metaclusters: bool = True,
    z_score: bool = True,
    clip_value: float | None = 3,
    output: str | Path | None = None,
    figsize: tuple[int, int] = (20, 20),
    fig_kwargs: Mapping[str, Any] = MappingProxyType({}),  # kwargs passed to plt.figure, e.g. dpi
    **kwargs,  # kwargs passed to sns.heatmap
):
    """
    Generate and visualize a heatmap of mean channel intensities for clusters or metaclusters.

    Parameters
    ----------
    sdata
        The input SpatialData object.
    table_layer
        The table layer in `sdata` containing cluster intensity for clusters and metaclusters, obtained via `hp.tb.cluster_intensity`.
    metaclusters
        Whether to display mean channel intensity per metacluster (`True`) or per cluster (`False`).
    z_score
        Whether to z-score the intensity values for normalization. We recommend setting this to `True`.
    clip_value
        The value to clip the z-scored data to, for better visualization. If `None`, no clipping is performed.
        Ignored if `z_score` is `False`.
    output
        The path to save the generated heatmap. If `None`, the heatmap will be displayed directly using `plt.show()`.
    figsize
        Size of the figure in inches (width, height).
    fig_kwargs
        Additional keyword arguments passed to `plt.figure`, such as `dpi`.
    **kwargs
        Additional keyword arguments passed to `sns.heatmap`, such as `annot`, `cmap`, or `cbar_kws`.

    Returns
    -------
    None
        The function generates and displays or saves a heatmap.

    Notes
    -----
    The heatmap shows mean channel intensities for either clusters or metaclusters, optionally normalized using z-scoring.
    Clusters and metaclusters are ordered based on hierarchical clustering of their channel intensity profiles.

    The function uses cosine similarity to compute the distance matrix for hierarchical clustering of channels.

    Example
    -------

    >>> harpy.tb.pixel_clusters_heatmap(
    ...     sdata,
    ...     table_layer="counts_clusters",
    ...     figsize=(30, 20),
    ...     fig_kwargs={"dpi": 100},
    ...     metaclusters=True,
    ...     z_score=True,
    ...     output="heatmap.png"
    ... )

    See Also
    --------
    harpy.tb.cluster_intensity
    """
    # clusters
    df = sdata.tables[table_layer].to_df().copy()
    df.index = sdata.tables[table_layer].obs[_INSTANCE_KEY].values

    # sort clusters by metaclusters
    cluster_info = sdata.tables[table_layer].obs.copy()
    cluster_info_sorted = cluster_info.sort_values([ClusteringKey._METACLUSTERING_KEY.value, _INSTANCE_KEY])
    cluster_info_sorted.index = cluster_info_sorted[_INSTANCE_KEY]
    sorted_clusters = cluster_info_sorted.index.tolist()
    df = df.loc[sorted_clusters, :]

    new_index_labels = []
    for cid in df.index:
        mc_id = cluster_info_sorted.loc[cid, ClusteringKey._METACLUSTERING_KEY.value]
        new_index_labels.append(f"{cid} ({mc_id})")
    df.index = new_index_labels

    # metaclusters
    df_metaclusters = sdata.tables[table_layer].uns[ClusteringKey._METACLUSTERING_KEY.value].copy()
    df_metaclusters.index = df_metaclusters[ClusteringKey._METACLUSTERING_KEY.value]
    df_metaclusters.drop(ClusteringKey._METACLUSTERING_KEY.value, axis=1, inplace=True)

    if z_score:
        df = df.apply(zscore)
        df_metaclusters = df_metaclusters.apply(zscore)
        if clip_value is not None:
            df = df.clip(lower=-clip_value, upper=clip_value)
            df_metaclusters = df_metaclusters.clip(lower=-clip_value, upper=clip_value)

    # create dendogram to cluster channel names together that have similar features
    # ( features are the intensity per metacluster here )
    dist_matrix = cosine_similarity(df_metaclusters.values.T)
    linkage_matrix = ward(dist_matrix)
    channel_names = df_metaclusters.columns
    dendro_info = dendrogram(linkage_matrix, labels=channel_names, no_plot=True)
    channel_order = dendro_info["ivl"]
    # sort both metaclusters and clusters based on dendogram clustering results
    df_metaclusters = df_metaclusters[channel_order]
    df = df[channel_order]

    # Create a heatmap
    plt.figure(figsize=figsize, **fig_kwargs)
    annot = kwargs.pop("annot", False)
    cmap = kwargs.pop("cmap", "coolwarm")
    fmt = kwargs.pop("fmt", ".2f")
    _label = "Mean Intensity (z-score)" if z_score else "Mean Intensity"
    cbar_kws = kwargs.pop("cbar_kws", {"label": _label})
    sns.heatmap(
        df_metaclusters.transpose() if metaclusters else df.transpose(),
        annot=annot,
        cmap=cmap,
        fmt=fmt,
        cbar_kws=cbar_kws,
        **kwargs,
    )
    _title = "metacluster" if metaclusters else "cluster"
    plt.title(f"Mean Channel Intensity per {_title}")
    plt.ylabel("Channel")
    _x_label = "Metacluster ID" if metaclusters else "Cluster ID (Metacluster ID)"
    plt.xlabel(_x_label)

    if output is None:
        plt.show()
    else:
        plt.savefig(output, bbox_inches="tight")
    plt.close()
