import importlib
import re

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
from harpy.plot._plot_sdata import plot_sdata, plot_sdata_genes
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

    sdata[table_layer].uns[f"{obs_column}_colors"] = ["#800080", "#008000", "#FFFF00"]  # purple, green, yellow
    # placeholder, otherwise "category_colors" will be ignored by spatialdata
    sdata[table_layer].uns[obs_column] = "__value__"
    return sdata


@pytest.mark.skipif(not importlib.util.find_spec("spatialdata_plot"), reason="requires the spatialdata-plot library")
def test_plot_sdata_image(sdata: SpatialData, tmp_path):
    img_layer = "blobs_multiscale_image"

    # 1) plot one channel
    channel_name = 0  # plot channel with name 0

    se = _get_spatial_element(sdata, layer=img_layer)
    channels = se.c.data

    c_id = np.where(channels == channel_name)[0].item()
    vmax = da.percentile(se.data[c_id].flatten(), q=99).compute()  # clip to 99% percentile
    norm = Normalize(vmax=vmax, clip=True)

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

    plot_sdata(
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
def test_plot_sdata_image_multichannel(sdata: SpatialData, tmp_path):
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

    plot_sdata(
        sdata,
        img_layer=img_layer,
        channel=channel_name,
        render_images_kwargs=render_images_kwargs,
        show_kwargs=show_kwargs,
        ax=ax,
    )

    fig.savefig(tmp_path / "image_multi_channels.png", dpi=dpi)


@pytest.mark.skipif(not importlib.util.find_spec("spatialdata_plot"), reason="requires the spatialdata-plot library")
def test_plot_sdata_image_multichannel_multi_axes(sdata: SpatialData, tmp_path):
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

    plot_sdata(
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

    plot_sdata(
        sdata,
        img_layer=img_layer,
        channel=channel_name[1],
        render_images_kwargs=render_images_kwargs,
        show_kwargs=show_kwargs,
        ax=ax[1],
    )

    fig.savefig(tmp_path / "image_multi_channels_multi_axes.png", dpi=dpi)


@pytest.mark.skipif(not importlib.util.find_spec("spatialdata_plot"), reason="requires the spatialdata-plot library")
def test_plot_sdata_label(sdata: SpatialData, tmp_path):
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

    plot_sdata(
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
    # note that when cropping colormap changes. I.e. colors of similar cells before and after crop are not the same (if not colored by e.g. .obs)
    plot_sdata(
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
def test_plot_sdata_label_categorical(sdata: SpatialData, tmp_path):
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

    plot_sdata(
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

    plot_sdata(
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
def test_plot_sdata_label_numerical(sdata: SpatialData, tmp_path):
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

    plot_sdata(
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

    plot_sdata(
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


def test_plot_sdata_raises(
    sdata: SpatialData,
):
    img_layer = "blobs_multiscale_image"
    labels_layer = "blobs_labels"
    table_layer = "table"
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Please specify a labels layer (which is annotated by the table layer '{table_layer}') if 'table_layer' is specified."
        ),
    ):
        plot_sdata(
            sdata,
            img_layer=img_layer,
            labels_layer=None,
            table_layer=table_layer,
            channel=0,
        )
    with pytest.raises(
        ValueError,
        match=re.escape("Please specify a 'table_layer' if 'color' is specified."),
    ):
        plot_sdata(
            sdata,
            img_layer=img_layer,
            labels_layer=labels_layer,
            color="channel_0_sum",
            table_layer=None,
            channel=0,
        )
    with pytest.raises(
        ValueError,
        match=re.escape("'coordinate_systems' found as key in 'show_kwargs'"),
    ):
        show_kwargs = {"coordinate_systems": "global"}
        plot_sdata(
            sdata,
            img_layer=img_layer,
            show_kwargs=show_kwargs,
            channel=0,
        )
    with pytest.raises(ValueError, match="After applying the bounding-box query with coordinates"):
        plot_sdata(
            sdata,
            img_layer=img_layer,
            labels_layer=labels_layer,
            channel=0,
            crd=[2000, 3000, 2000, 3000],
        )
    render_labels_kwargs = {"table_name": labels_layer}
    with pytest.raises(ValueError, match="Please specify 'table_name' via the keyword argument 'table_layer'"):
        plot_sdata(
            sdata,
            img_layer=img_layer,
            labels_layer=labels_layer,
            render_labels_kwargs=render_labels_kwargs,
            channel=0,
        )
    with pytest.raises(ValueError, match="Please specify a 'table_layer' if 'color' is specified"):
        plot_sdata(
            sdata,
            img_layer=img_layer,
            labels_layer=labels_layer,
            color="channel_0_sum",
            render_labels_kwargs=render_labels_kwargs,
            channel=0,
        )
    render_labels_kwargs = {"color": "channel_0_sum"}
    with pytest.raises(ValueError, match="Please specify 'color' via the keyword argument 'color'"):
        plot_sdata(
            sdata,
            img_layer=img_layer,
            labels_layer=labels_layer,
            table_layer=table_layer,
            render_labels_kwargs=render_labels_kwargs,
            channel=0,
        )
    labels_layer = "blobs_multiscale_labels"
    table_layer = "table"
    with pytest.raises(
        ValueError,
        match=f"The labels layer '{labels_layer}' does not seem to be annotated by the table layer '{table_layer}'",
    ):
        plot_sdata(
            sdata,
            img_layer=img_layer,
            labels_layer=labels_layer,
            color="channel_0_sum",
            table_layer=table_layer,
            render_labels_kwargs=render_labels_kwargs,
            channel=0,
        )


@pytest.mark.skipif(not importlib.util.find_spec("spatialdata_plot"), reason="requires the spatialdata-plot library")
@pytest.mark.parametrize(
    "genes",
    [
        ["gene_b", "gene_a"],
        ["gene_b"],
        None,
    ],
)
def test_plot_sdata_genes(sdata: SpatialData, tmp_path, genes):
    points_layer = "blobs_points"
    img_layer = "blobs_image"
    dpi = 200

    fig, ax = plt.subplots()

    plot_sdata_genes(
        sdata,
        points_layer=points_layer,
        img_layer=img_layer,
        name_gene_column="genes",
        genes=genes,
        ax=ax,
    )

    name = "_".join(genes) if genes is not None else "not_categorical"

    fig.savefig(tmp_path / f"plot_genes_{name}.png", dpi=dpi)


@pytest.mark.skipif(not importlib.util.find_spec("spatialdata_plot"), reason="requires the spatialdata-plot library")
@pytest.mark.parametrize(
    "frac",
    [
        0.5,
        None,
    ],
)
def test_plot_sdata_genes_frac(sdata: SpatialData, tmp_path, frac):
    points_layer = "blobs_points"
    img_layer = "blobs_image"
    dpi = 200

    fig, ax = plt.subplots()

    plot_sdata_genes(
        sdata,
        points_layer=points_layer,
        img_layer=img_layer,
        name_gene_column="genes",
        genes=None,
        frac=frac,
        ax=ax,
    )

    name = f"frac_{frac}" if frac is not None else "no_frac"

    fig.savefig(tmp_path / f"plot_genes_{name}.png", dpi=dpi)


@pytest.mark.skipif(not importlib.util.find_spec("spatialdata_plot"), reason="requires the spatialdata-plot library")
@pytest.mark.parametrize(
    "crd",
    [
        [0, 100, 0, 90],  # This crop has only gene_b in it. We check if color is the same as without crop.
        None,
    ],
)
def test_plot_sdata_genes_crop(sdata: SpatialData, tmp_path, crd):
    points_layer = "blobs_points"
    img_layer = "blobs_image"
    dpi = 200

    fig, ax = plt.subplots()

    plot_sdata_genes(
        sdata,
        points_layer=points_layer,
        img_layer=img_layer,
        name_gene_column="genes",
        crd=crd,
        genes=["gene_a", "gene_b"],
        ax=ax,
    )

    name = "crop" if crd is not None else "no_crop"

    fig.savefig(tmp_path / f"plot_genes_{name}.png", dpi=dpi)


def test_plot_sdata_genes_name_raises(sdata: SpatialData):
    points_layer = "blobs_points"
    img_layer = "blobs_image"

    name_gene_column = "genes_wrong"

    with pytest.raises(ValueError, match=f"Column '{name_gene_column}' not found"):
        plot_sdata_genes(
            sdata,
            points_layer=points_layer,
            img_layer=img_layer,
            name_gene_column=name_gene_column,
        )


def test_plot_sdata_genes_show_kwargs_raises(sdata: SpatialData):
    points_layer = "blobs_points"
    img_layer = "blobs_image"

    show_kwargs = {"coordinate_systems": "global"}

    with pytest.raises(ValueError, match="'coordinate_systems' found as key in 'show_kwargs'"):
        plot_sdata_genes(
            sdata,
            points_layer=points_layer,
            img_layer=img_layer,
            name_gene_column="genes",
            show_kwargs=show_kwargs,
        )


def test_plot_sdata_genes_palette(sdata: SpatialData, tmp_path):
    points_layer = "blobs_points"
    img_layer = "blobs_image"

    dpi = 200

    fig, ax = plt.subplots()

    plot_sdata_genes(
        sdata,
        points_layer=points_layer,
        img_layer=img_layer,
        name_gene_column="genes",
        genes=["gene_a", "gene_b"],
        palette=["pink", "red"],
        ax=ax,
    )

    fig.savefig(tmp_path / "plot_genes_palette.png", dpi=dpi)


def test_plot_sdata_genes_palette_raises(sdata: SpatialData, tmp_path):
    points_layer = "blobs_points"
    img_layer = "blobs_image"

    with pytest.raises(ValueError, match="The number of genes specified via 'genes' "):
        plot_sdata_genes(
            sdata,
            points_layer=points_layer,
            img_layer=img_layer,
            name_gene_column="genes",
            genes=["gene_a", "gene_b"],
            palette=["pink"],
        )
