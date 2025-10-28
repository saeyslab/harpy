import importlib

import dask.array as da
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from matplotlib.colors import Normalize
from spatialdata import SpatialData, get_element_instances
from spatialdata.models import TableModel

from harpy.image._image import _get_spatial_element
from harpy.plot._plot_spatialdata import plot_spatialdata
from harpy.utils._keys import _INSTANCE_KEY, _REGION_KEY

matplotlib.use("Agg")  # to make sure plots do not pop up


def _prep_sdata(
    sdata: SpatialData,
    labels_layer: str = "blobs_multiscale_labels",
    table_layer: str = "other_table",
    obs_column: str | None = "category",
    instance_key: str = _INSTANCE_KEY,
    region_key: str = _REGION_KEY,
) -> SpatialData:
    """Add categorical column to table `table_layer` that is annotated by `labels_layer`."""
    RNG = np.random.default_rng(seed=42)

    instances = get_element_instances(sdata[labels_layer])
    n_obs = len(instances)
    adata = AnnData(
        RNG.normal(size=(n_obs, 10)),
        obs=pd.DataFrame(RNG.normal(size=(n_obs, 3)), columns=["a", "b", "c"]),
    )
    adata.obs[instance_key] = instances.values
    adata.obs[obs_column] = RNG.choice(["a", "b", "c"], size=adata.n_obs)
    adata.obs[obs_column][:3] = ["a", "b", "c"]
    adata.obs[region_key] = labels_layer
    table = TableModel.parse(
        adata=adata,
        region_key=region_key,
        instance_key=instance_key,
        region=labels_layer,
    )
    sdata[table_layer] = table
    sdata[table_layer].obs[obs_column] = sdata[table_layer].obs[obs_column].astype("category")

    sdata[table_layer].uns[f"{obs_column}_colors"] = ["#800080", "#008000", "#FFFF00"]  # purple, green ,yellow
    # placeholder, otherwise "category_colors" will be ignored by spatialdata
    sdata[table_layer].uns[obs_column] = "__value__"
    return sdata


@pytest.mark.skipif(not importlib.util.find_spec("spatialdata_plot"), reason="requires the spatialdata-plot library")
def test_plot_spatialdata_image(sdata: SpatialData, tmp_path):
    img_layer = "blobs_multiscale_image"

    # 1) plot one channel
    channel_name = 0  # plot channel with name 0

    se = _get_spatial_element(sdata, layer=img_layer)
    channels = se.c.data

    c_id = np.where(channels == channel_name)[0].item()
    vmax = da.percentile(se.data[c_id].flatten(), q=99).compute()  # clip to 99% percentile
    norm = Normalize(vmax=vmax, clip=True)

    # https://matplotlib.org/stable/gallery/color/named_colors.html -> list of colors that can be passed to palette
    render_images_kwargs = {
        "cmap": "grey",
        "norm": norm,
        "scale": "scale2",
    }

    show_kwargs = {
        "title": str(channel_name),
        "colorbar": False,
        "dpi": 200,
        "figsize": (20, 20),
    }

    fig, ax = plt.subplots()

    plot_spatialdata(
        sdata,
        img_layer=img_layer,
        channel=channel_name,
        render_images_kwargs=render_images_kwargs,
        show_kwargs=show_kwargs,
        ax=ax,
    )

    fig.savefig(
        tmp_path / "image.png", dpi=100
    )  # we need to pass dpi from showkwargs here, otherwise will save with default resolution


@pytest.mark.skipif(not importlib.util.find_spec("spatialdata_plot"), reason="requires the spatialdata-plot library")
def test_plot_spatialdata_image_multichannel(sdata: SpatialData, tmp_path):
    img_layer = "blobs_multiscale_image"
    channel_name = [0, 1, 2]
    dpi = 100
    # https://matplotlib.org/stable/gallery/color/named_colors.html -> list of colors that can be passed to palette
    # do not pass multiple cmaps when working with multiple channels, because spatialdata plot will plot them over each other.
    render_images_kwargs = {
        "palette": ["red", "green", "blue"],
        "scale": "scale2",
    }

    show_kwargs = {
        "title": str(channel_name),
        "colorbar": False,
        "dpi": dpi,
        "figsize": (20, 20),
    }

    fig, ax = plt.subplots()

    plot_spatialdata(
        sdata,
        img_layer=img_layer,
        channel=channel_name,
        render_images_kwargs=render_images_kwargs,
        show_kwargs=show_kwargs,
        ax=ax,
    )

    fig.savefig(tmp_path / "image_multi_channels.png", dpi=dpi)


@pytest.mark.skipif(not importlib.util.find_spec("spatialdata_plot"), reason="requires the spatialdata-plot library")
def test_plot_spatialdata_image_multichannel_multi_axes(sdata: SpatialData, tmp_path):
    img_layer = "blobs_multiscale_image"
    channel_name = [0, 1, 2]
    dpi = 100

    fig, ax = plt.subplots(1, 2)

    render_images_kwargs = {
        "cmap": "grey",
        "scale": "scale2",
    }

    show_kwargs = {
        "title": str(channel_name[0]),
        "colorbar": False,
        "dpi": dpi,
        "figsize": (20, 20),
    }

    plot_spatialdata(
        sdata,
        img_layer=img_layer,
        channel=channel_name[0],
        render_images_kwargs=render_images_kwargs,
        show_kwargs=show_kwargs,
        ax=ax[0],
    )

    render_images_kwargs = {
        "cmap": "viridis",
        "scale": "scale2",
    }

    show_kwargs = {
        "title": str(channel_name[1]),
        "colorbar": False,
        "dpi": dpi,
        "figsize": (20, 20),
    }

    plot_spatialdata(
        sdata,
        img_layer=img_layer,
        channel=channel_name[1],
        render_images_kwargs=render_images_kwargs,
        show_kwargs=show_kwargs,
        ax=ax[1],
    )

    fig.savefig(tmp_path / "image_multi_channels_multi_axes.png", dpi=dpi)


@pytest.mark.skipif(not importlib.util.find_spec("spatialdata_plot"), reason="requires the spatialdata-plot library")
def test_plot_spatialdata_label(sdata: SpatialData, tmp_path):
    img_layer = "blobs_multiscale_image"
    labels_layer = "blobs_multiscale_labels"
    channel_name = 0
    dpi = 100

    fig, ax = plt.subplots()

    render_images_kwargs = {
        "cmap": "grey",
        "scale": "scale2",
    }
    render_labels_kwargs = {
        "scale": "scale2",
        "fill_alpha": 0.2,
        "outline_alpha": 0.3,
    }

    show_kwargs = {
        "title": str(channel_name),
        "colorbar": False,
        "dpi": dpi,
        "figsize": (20, 20),
    }

    plot_spatialdata(
        sdata,
        channel=channel_name,
        img_layer=img_layer,
        labels_layer=labels_layer,
        render_images_kwargs=render_images_kwargs,
        render_labels_kwargs=render_labels_kwargs,
        show_kwargs=show_kwargs,
        ax=ax,
    )

    fig.savefig(tmp_path / "image_label.png", dpi=dpi)

    fig, ax = plt.subplots()

    # now with a crop
    # note that when cropping colormap changes. I.e. colors of similar cells before and after crop are not the same

    plot_spatialdata(
        sdata,
        channel=channel_name,
        img_layer=img_layer,
        labels_layer=labels_layer,
        crd=[0, 200, 0, 200],
        render_images_kwargs=render_images_kwargs,
        render_labels_kwargs=render_labels_kwargs,
        show_kwargs=show_kwargs,
        ax=ax,
    )

    fig.savefig(tmp_path / "image_label_crop.png", dpi=dpi)


@pytest.mark.skipif(not importlib.util.find_spec("spatialdata_plot"), reason="requires the spatialdata-plot library")
def test_plot_spatialdata_label_categorical(sdata: SpatialData, tmp_path):
    img_layer = "blobs_multiscale_image"
    labels_layer = "blobs_multiscale_labels"
    table_layer = "other_table"
    obs_column_name = "my_category"
    channel_name = 0
    dpi = 100

    sdata = _prep_sdata(
        sdata,
        labels_layer=labels_layer,
        table_layer=table_layer,
        obs_column=obs_column_name,
    )

    fig, ax = plt.subplots()

    render_images_kwargs = {
        "cmap": "grey",
        "scale": "scale2",
    }
    render_labels_kwargs = {
        "scale": "scale2",
        "fill_alpha": 0.2,
        "outline_alpha": 0.3,
    }

    show_kwargs = {
        "title": str(channel_name),
        "colorbar": False,
        "dpi": dpi,
        "figsize": (20, 20),
    }

    plot_spatialdata(
        sdata,
        channel=channel_name,
        img_layer=img_layer,
        labels_layer=labels_layer,
        table_layer=table_layer,
        color=obs_column_name,
        render_images_kwargs=render_images_kwargs,
        render_labels_kwargs=render_labels_kwargs,
        show_kwargs=show_kwargs,
        ax=ax,
    )

    fig.savefig(tmp_path / "image_label_categorical.png", dpi=dpi)

    # now with a crop

    fig, ax = plt.subplots()

    plot_spatialdata(
        sdata,
        channel=channel_name,
        img_layer=img_layer,
        labels_layer=labels_layer,
        table_layer=table_layer,
        color=obs_column_name,
        crd=[0, 200, 0, 200],
        render_images_kwargs=render_images_kwargs,
        render_labels_kwargs=render_labels_kwargs,
        show_kwargs=show_kwargs,
        ax=ax,
    )

    fig.savefig(tmp_path / "image_label_categorical_crop.png", dpi=dpi)


@pytest.mark.skipif(not importlib.util.find_spec("spatialdata_plot"), reason="requires the spatialdata-plot library")
def test_plot_spatialdata_label_numerical(sdata: SpatialData, tmp_path):
    img_layer = "blobs_multiscale_image"
    labels_layer = "blobs_multiscale_labels"
    table_layer = "other_table"
    var_column_name = "2"
    channel_name = 0
    dpi = 100

    sdata = _prep_sdata(
        sdata,
        labels_layer=labels_layer,
        table_layer=table_layer,
    )

    fig, ax = plt.subplots()

    render_images_kwargs = {
        "cmap": "grey",
        "scale": "scale2",
    }
    render_labels_kwargs = {
        "scale": "scale2",
        "fill_alpha": 0.9,
        "outline_alpha": 0.3,
    }

    show_kwargs = {
        "title": str(channel_name),
        "colorbar": False,
        "dpi": dpi,
        "figsize": (20, 20),
    }

    plot_spatialdata(
        sdata,
        channel=channel_name,
        img_layer=img_layer,
        labels_layer=labels_layer,
        table_layer=table_layer,
        color=var_column_name,
        render_images_kwargs=render_images_kwargs,
        render_labels_kwargs=render_labels_kwargs,
        show_kwargs=show_kwargs,
        ax=ax,
    )

    fig.savefig(tmp_path / "image_label_numerical.png", dpi=dpi)

    # now with a crop

    fig, ax = plt.subplots()

    plot_spatialdata(
        sdata,
        channel=channel_name,
        img_layer=img_layer,
        labels_layer=labels_layer,
        table_layer=table_layer,
        color=var_column_name,
        crd=[0, 200, 0, 200],
        render_images_kwargs=render_images_kwargs,
        render_labels_kwargs=render_labels_kwargs,
        show_kwargs=show_kwargs,
        ax=ax,
    )

    fig.savefig(tmp_path / "image_label_numerical_crop.png", dpi=dpi)
