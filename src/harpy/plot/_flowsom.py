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
from scipy.sparse import csr_matrix
from scipy.stats import zscore
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
    # for unit test, check if we have to do:
    # if output is not None:
    #    matplotlib.use("Agg")
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
    if metaclusters:
        key = ClusteringKey._METACLUSTERING_KEY.value
    else:
        key = ClusteringKey._CLUSTERING_KEY.value

    if metaclusters:
        df = sdata.tables[table_layer].uns[key].copy()
        df.index = df[key]
        df.drop(key, axis=1, inplace=True)
    else:
        pass
        # get the clusters

    if z_score:
        df = df.apply(zscore)
        # df = df.apply(lambda x: zscore(x, axis=0), axis=1)
        if clip_value is not None:
            df = df.clip(lower=-clip_value, upper=clip_value)

    # Create a heatmap
    plt.figure(figsize=figsize, **fig_kwargs)
    annot = kwargs.pop("annot", False)
    cmap = kwargs.pop("cmap", "coolwarm")
    fmt = kwargs.pop("fmt", ".2f")
    cbar_kws = kwargs.pop("cbar_kws", {"label": "Mean Intensity (z-score)"})
    sns.heatmap(
        df.transpose(),
        annot=annot,
        cmap=cmap,
        fmt=fmt,
        cbar_kws=cbar_kws,
        **kwargs,
    )
    plt.title("Mean Channel Intensity per metacluster")
    plt.xlabel("Channels")
    plt.ylabel("Cluster IDs")

    if output is None:
        plt.show()
    else:
        plt.savefig(output, bbox_inches="tight")
    plt.close()
