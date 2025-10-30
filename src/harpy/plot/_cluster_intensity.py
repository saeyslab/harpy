from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes
from scipy.cluster.hierarchy import dendrogram, ward
from scipy.stats import zscore
from sklearn.metrics.pairwise import cosine_similarity
from spatialdata import SpatialData

from harpy.utils.pylogger import get_pylogger
from harpy.utils.utils import _make_list

log = get_pylogger(__name__)


def cluster_intensity_heatmap(
    sdata: SpatialData,
    table_layer: str,
    cluster_key: str,
    cluster_key_uns: str | None = None,
    channels: Iterable[str] | None = None,
    z_score: bool = True,
    clip_value: float | None = 3,
    ax: Axes = None,
    output: str | Path | None = None,
    figsize: tuple[int, int] = (20, 20),
    fig_kwargs: Mapping[str, Any] = MappingProxyType({}),  # kwargs passed to plt.figure, e.g. dpi
    **kwargs,  # kwargs passed to sns.heatmap
) -> Axes:
    """
    Generate and visualize a heatmap of mean channel intensities per cluster for each channel.

    The heatmap shows mean channel intensities per cluster, optionally normalized using z-scoring.
    Clusters are ordered based on hierarchical clustering of their channel intensity profiles.

    The function uses cosine similarity to compute the distance matrix for hierarchical clustering of channels.

    Parameters
    ----------
    sdata
        SpatialData object.
    table_layer
        The table layer containing the weighted mean intensities per cluster in `sdata[table_layer].uns[cluster_key_uns]`.
    cluster_key
        The cluster key in `sdata.tables[table_layer].obs`.
    cluster_key_uns
        The key in `sdata.tables[table_layer].uns` where the weighted mean intensitiy per cluster is stored.
    channels
        The channels to visualize. If `None` all channels are visualized.
    z_score
        Whether to z-score the intensity values for normalization. We recommend setting this to `True`.
    clip_value
        The value to clip the z-scored data to, for better visualization. If `None`, no clipping is performed.
        Ignored if `z_score` is `False`.
    ax
        Matplotlib axes object to plot on. If `None`, a new figure is created using `fig_kwargs`.
    output
        The path to save the generated heatmap.
    figsize
        Tuple specifying the size of the figure in inches as `(width, height)`.
        The width determines the spacing available for cluster IDs, while the height adjusts space for channels.
        If labels (cluster or channel names) are truncated, increase the respective dimension.
        Increase `width` if cluster names are not fully visible.
        Increase `height` if channel names are not fully visible.
    fig_kwargs
        Additional keyword arguments passed to :func:`matplotlib.pyplot.figure`, such as `dpi`. Ignored if `ax` is `None`.
    **kwargs
        Additional keyword arguments passed to :func:`seaborn.heatmap`, such as `annot`, `cmap`, or `cbar_kws`.

    Returns
    -------
    A :func:`matplotlib.axes.Axes` object.

    Example
    -------
    >>> import numpy as np
    >>> import harpy as hp
    >>>
    >>> # Load example dataset
    >>> sdata = hp.datasets.pixie_example()
    >>>
    >>> img_layer = "raw_image_fov0"
    >>> labels_layer = "label_whole_fov0"
    >>> table_layer = "table_intensities"
    >>> to_coordinate_system = "fov0"
    >>> cluster_key = "cluster_id"
    >>>
    >>> # Calculate total intensity values for each label in label_layer, for each channel in img_layer
    >>> sdata = hp.tb.allocate_intensity(
    ...     sdata,
    ...     img_layer=img_layer,
    ...     labels_layer=labels_layer,
    ...     output_layer=table_layer,
    ...     mode="sum",
    ...     to_coordinate_system=to_coordinate_system,
    ...     overwrite=True,
    ... )
    >>>
    >>> # Normalize intensities by instance size
    >>> sdata = hp.tb.preprocess_proteomics(
    ...     sdata,
    ...     labels_layer=labels_layer,
    ...     table_layer=table_layer,
    ...     output_layer=table_layer,
    ...     size_norm=True,
    ...     log1p=False,
    ...     scale=False,
    ...     calculate_pca=False,
    ...     overwrite=True,
    ... )
    >>>
    >>> # Add a dummy cluster ID
    >>> n_obs = sdata[table_layer].shape[0]
    >>> RNG = np.random.default_rng(seed=42)
    >>> sdata[table_layer].obs[cluster_key] = RNG.choice(range(10), size=n_obs)
    >>> sdata[table_layer].obs[cluster_key] = (
    ...     sdata[table_layer].obs[cluster_key].astype("category")
    ... )
    >>>
    >>> Calculate mean intensity per cluster
    >>> sdata = hp.tb.cluster_intensity(
    ...     sdata,
    ...     table_layer=table_layer,
    ...     labels_layer=labels_layer,
    ...     cluster_key=cluster_key,
    ...     output_layer=table_layer,
    ... )
    >>>
    >>> # Plot heatmap of mean intensity per cluster
    >>> fig_kwargs = {"dpi": 200}
    >>> hp.pl.cluster_intensity_heatmap(
    ...     sdata,
    ...     table_layer=table_layer,
    ...     cluster_key=cluster_key,
    ...     z_score=True,
    ...     figsize=(10, 5),
    ... )

    See Also
    --------
    harpy.tb.cluster_intensity: calculates weighted (by instance size) average intensity per cluster for every channel.
    harpy.tb.allocate_intensity : calculates total intensity per instance per channel.
    harpy.tb.preprocess_proteomics: calculates instance size and normalizes intensity by instance size.
    """
    if channels is not None:
        channels = _make_list(channels)
        if len(channels) <= 1:
            raise ValueError("Please specify at least two channels.")
    if cluster_key_uns is None:
        cluster_key_uns = f"{cluster_key}_weighted_intensity"
    # get weighted (by instance size) mean intensity per cluster (cluster_key)
    if cluster_key_uns not in sdata.tables[table_layer].uns.keys():
        raise ValueError(
            f"Key '{cluster_key_uns}' not found in sdata.tables[{table_layer}].uns. "
            "Run 'harpy.tb.cluster_intensity()' first."
        )
    if cluster_key not in sdata.tables[table_layer].uns[cluster_key_uns].columns:
        raise ValueError(
            f"Cluster key '{cluster_key}' not found in sdata.tables[{table_layer}].uns[{cluster_key_uns}]. "
            "Run harpy.tb.cluster_intensity() first."
        )
    df = sdata.tables[table_layer].uns[cluster_key_uns].copy()
    df.index = df[cluster_key]
    df.drop(cluster_key, axis=1, inplace=True)
    if channels is not None:
        df = df[channels]

    if z_score:
        df = df.apply(zscore)
        if clip_value is not None:
            df = df.clip(lower=-clip_value, upper=clip_value)

    # create dendogram to cluster channel names together that have similar features
    # ( features are the intensity per metacluster here )
    dist_matrix = cosine_similarity(df.values.T)
    linkage_matrix = ward(dist_matrix)
    channel_names = df.columns
    dendro_info = dendrogram(linkage_matrix, labels=channel_names, no_plot=True)
    channel_order = dendro_info["ivl"]
    # sort channels based on dendogram clustering results
    df = df[channel_order]

    # Create axes if not provided
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, **fig_kwargs)

    # Create a heatmap
    annot = kwargs.pop("annot", False)
    cmap = kwargs.pop("cmap", "coolwarm")
    fmt = kwargs.pop("fmt", ".2f")
    _label = "Mean Intensity (z-score)" if z_score else "Mean Intensity"
    cbar_kws = kwargs.pop("cbar_kws", {"label": _label})
    ax = sns.heatmap(
        df.transpose(),
        annot=annot,
        cmap=cmap,
        fmt=fmt,
        cbar_kws=cbar_kws,
        ax=ax,
        **kwargs,
    )
    _title = "cluster"
    ax.set_title(f"Mean Channel Intensity per {_title}")
    _x_label = "Cluster"

    ax.set_ylabel("Channel")
    ax.set_xlabel(_x_label)

    if output is not None:
        ax.figure.savefig(output)

    return ax
