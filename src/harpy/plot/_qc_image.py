"""Calculate various image quality metrics"""

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import dask
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import skimage as ski
from matplotlib.axes import Axes
from spatialdata import SpatialData

from harpy.image import normalize
from harpy.image._image import _get_spatial_element
from harpy.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

try:
    import textalloc as ta

except ImportError:
    log.warning(
        "'textalloc' not installed, to use 'harpy.pl.group_snr_ratio' and 'harpy.pl.snr_ratio', please install this library."
    )


def histogram(
    sdata: SpatialData,
    img_layer: str,
    channel: str,
    bins: int,
    range: tuple[float, float] | None = None,
    ax: Axes = None,
    output: str | Path = None,
    fig_kwargs: dict[str, Any] = MappingProxyType({}),  # kwargs passed to plt.figure, e.g. dpi, figsize
    bar_kwargs: Mapping[str, Any] = MappingProxyType({}),  # kwargs passed to ax.bar, e.g. color and alpha
    **kwargs,
) -> Axes:
    """
    Generate and visualize a histogram for a specified image channel within an image of a `SpatialData` object.

    Parameters
    ----------
    sdata
        The input `SpatialData` object containing the image data.
    img_layer
        The name of the image layer within `sdata` to analyze.
    channel
        The specific channel of the image data to use for the histogram.
    bins
        The number of bins for the histogram.
    range
        The range of values for the histogram as `(min, max)`.
        If not provided, range is simply `(dask.array.nanmin(...), dask.array.nanmax(...))` thus excluding NaN.
        Values outside the range are ignored.
    fig_kwargs
        Additional keyword arguments passed to `plt.figure`, such as `dpi` or `figsize`. Ignored if `ax` is `None`.
    bar_kwargs
        Additional keyword arguments passed to `ax.bar`, such as `color` or `alpha`.
    ax
        An existing axes object to plot the histogram. If `None`, a new figure and axes will be created.
    output
        The path to save the generated plot. If `None`, the plot will not be saved.
    **kwargs
        Additional keyword arguments passed to :func:`dask.array.histogram`.

    Returns
    -------
        The axes object containing the histogram plot.

    Raises
    ------
    AssertionError
        If `img_layer` is not found in `sdata.images`.

    Examples
    --------
    >>> ax = histogram(
    ...     sdata,
    ...     img_layer="raw_image_crop_preprocessed",
    ...     channel="Anti Rabbit (PE C1)",
    ...     bins=100,
    ...     range=(0, 1.0),
    ...     fig_kwargs={"figsize": (5, 5)},
    ...     bar_kwargs={"color": "blue", "alpha": 0.7},
    ...     output="histogram.png"
    ... )
    """
    assert img_layer in sdata.images, f"'{img_layer}' not found in 'sdata.images'."
    se = _get_spatial_element(sdata, layer=img_layer)

    array = se.data[se.c.data.tolist().index(channel)]

    if range is None:
        range = (da.nanmin(array), da.nanmax(array))

    hist, bin_edges = da.histogram(array, bins=bins, range=range, **kwargs)

    hist, bin_edges = dask.compute(hist, bin_edges)

    # Create axes if not provided
    if ax is None:
        fig, ax = plt.subplots(**fig_kwargs)

    # Plot
    bar_kwargs = dict(bar_kwargs)
    fig_kwargs = dict(fig_kwargs)
    color = bar_kwargs.pop("color", "blue")
    alpha = bar_kwargs.pop("alpha", 0.7)
    align = bar_kwargs.pop("align", "edge")

    ax.bar(bin_edges[:-1], hist, width=(bin_edges[1] - bin_edges[0]), align=align, alpha=alpha, color=color)
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Frequency")
    ax.set_title(channel)

    if output is not None:
        fig.savefig(output)

    return ax


def calculate_snr(img, nbins=65536):
    """Calculate the signal to noise ratio of an image.

    The threshold is calculated using the Otsu method.
    The signal is the mean intensity of the pixels above the threshold and the noise is the mean of the pixels below the threshold.
    """
    thres = ski.filters.threshold_otsu(img, nbins=nbins)
    mask = img > thres
    signal = img[mask].mean()
    noise = img[~mask].mean()
    snr = signal / noise
    return snr, signal


def calculate_snr_ratio(
    sdata,
    table_name="table",
    image="raw_image",
    block_size=10000,
    channel_names=None,
    cycles=None,
    signal_threshold=None,
):
    log.debug("Calculating SNR ratio")
    data = []
    table = sdata[table_name]
    if channel_names is None:
        channel_names = table.var_names
    if cycles:
        if cycles in table.var.keys():
            cycles = table.var[cycles]
        else:
            cycles = cycles
    else:
        cycles = [None] * len(channel_names)
    for image in sdata.images:
        for cycle, channel_name in zip(cycles, channel_names):
            float_block = sdata[image].sel(c=channel_name).data.rechunk(block_size)
            img = float_block.compute()
            snr, signal = calculate_snr(img)
            if signal_threshold and signal < signal_threshold:
                continue
            data += [(image, cycle, channel_name, snr, signal)]
            del img
    df_img = pd.DataFrame(data, columns=["image", "cycle", "channel", "snr", "signal"])
    return df_img


def snr_ratio(sdata, ax=None, loglog=True, color="black", groupby=None, **kwargs):
    """Plot the signal to noise ratio. On the x-axis is the signal intensity and on the y-axis is the SNR-ratio"""
    log.debug("Plotting SNR ratio")
    if ax is None:
        fig, ax = plt.subplots()
    df_img = calculate_snr_ratio(sdata, cycles="cycle" if color == "cycle" else None, **kwargs)
    if loglog:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)

    # group by "channel" and take the mean of "image" and "cycle"
    if groupby is None:
        groupby = ["channel"]
    df_img = df_img.groupby(["channel"]).mean(numeric_only=True)
    # sort by channel
    df_img = df_img.sort_values("channel")
    # do a scatter plot
    if color == "cycle":
        palette = sns.color_palette("viridis", n_colors=len(df_img["cycle"].unique()))
        cmap = sns.color_palette("viridis", n_colors=len(df_img["cycle"].unique()), as_cmap=True)
        df_img["cycle"] = get_hexes(df_img["cycle"], palette=palette)
    log.debug(df_img.head())
    _plot_snr_ratio(df_img, ax, color, text_list=sdata.table.var_names)
    ax.set_xlabel("Signal intensity")
    ax.set_ylabel("Signal-to-noise ratio")
    # cbar_ax = fig.add_axes([1, 0.1, 0.02, 0.8])
    # cbar.set_label("Cycles")
    # cbar.set_ticklabels(['0', '10','20', '30', '40', '51'])  # Adjust the tick labels as needed

    # # add colorbar for scatter plot
    # if color == "cycle":
    #     cbar = fig.colorbar(palette, ax=ax)
    #     cbar.set_label("Cycles")
    if color == "cycle":
        cbar_ax = fig.add_axes([1, 0.1, 0.02, 0.8])
        mappable = plt.cm.ScalarMappable(cmap=cmap)
        mappable.set_clim(0, len(df_img["cycle"].max()))
        cbar = plt.colorbar(mappable, cax=cbar_ax)
        cbar.set_label("Cycles")
    ax.legend()
    return ax


def _plot_snr_ratio(df, ax, color, text_list):
    for _i, row in df.iterrows():
        # do a scatter plot
        if color == "cycle":
            i_color = row["cycle"]
        else:
            i_color = color
        ax.scatter(row["signal"], row["snr"], color=i_color)
        # use textalloc to add channel names
    x = df["signal"]
    y = df["snr"]
    ta.allocate(ax, x=x, y=y, text_list=text_list, x_scatter=x, y_scatter=y)
    # ax.set_xlabel("Signal intensity")
    # ax.set_ylabel("Signal-to-noise ratio")
    return ax


def group_snr_ratio(sdata, groupby, ax=None, loglog=True, color="black", **kwargs):
    """Plot the signal to noise ratio. On the x-axis is the signal intensity and on the y-axis is the SNR-ratio"""
    log.debug("Plotting SNR ratio")
    df_img = calculate_snr_ratio(sdata, cycles="cycle" if color == "cycle" else None, **kwargs)

    df_img = df_img.groupby(groupby).mean(numeric_only=True)
    # sort by channel
    df_img = df_img.sort_values("channel")

    n_groups = df_img.index.levels[0].shape[0]

    # Set up subplots
    n_by_2 = n_groups // 2 + n_groups % 2
    fig, axs = plt.subplots(n_by_2, 2, figsize=(10, 5 * (n_by_2)))

    if color == "cycle":
        palette = sns.color_palette("viridis", n_colors=len(df_img["cycle"].unique()))
        cmap = sns.color_palette("viridis", n_colors=len(df_img["cycle"].unique()), as_cmap=True)
        df_img["cycle"] = get_hexes(df_img["cycle"], palette=palette)

    # Iterate over unique samples and create separate plots
    for ax, sample in zip(axs.flatten(), df_img.index.levels[0]):
        if loglog:
            ax.set_xscale("log", base=2)
            ax.set_yscale("log", base=2)
        log.debug(sample)
        sample_df = df_img.loc[sample]
        #     ax = axs[i // 2, i % 2]  # Get the correct subplot
        ax.set_title(sample)
        _plot_snr_ratio(sample_df, ax, color, text_list=sdata.table.var_names)
        ax.set_xlabel("Signal intensity")
        ax.set_ylabel("Signal-to-noise ratio")

        #     # Add points with color gradient based on cycle value
        #     for index, row in sample_df.iterrows():
        #         cycle_value = row["cycle"]
        #         color = plt.cm.seismic(cycle_value / sample_df["cycle"].max())  # Normalize cycle value to span from 0 to 51
        #         (
        #             ax.scatter(
        #                 row["signal_log"],
        #                 row["snr_log"],
        #                 label=row["channel"],
        #                 color=color,
        #                 edgecolors="black",
        #                 s=1,
        #                 alpha=0.7,
        #             ),
        #         )
        #         ax.text(
        #             row["signal_log"] + 0.01,
        #             row["snr_log"],
        #             row["channel"],
        #             horizontalalignment="left",
        #             fontsize=9,
        #             color=color,
        #             bbox=dict(facecolor="none", edgecolor="black", boxstyle="round,pad=0.2", alpha=0.5),
        #         )

        # Set x-axis label
        # ax.set_xlabel("Signal intensity (log2)", size=12)
        # Set y-axis label
        # ax.set_ylabel("Signal-to-noise ratio (log2)", size=12)

    # # Add a colorbar with title
    if color == "cycle":
        cbar_ax = fig.add_axes([1, 0.1, 0.02, 0.8])
        mappable = plt.cm.ScalarMappable(cmap=cmap)
        mappable.set_clim(0, len(df_img["cycle"].max()))
        cbar = plt.colorbar(mappable, cax=cbar_ax)
        cbar.set_label("Cycles")
    ax.legend()
    plt.tight_layout()
    return axs


def arc_transform(df):
    divider = 5 * np.quantile(df, 0.2, axis=0)
    divider[divider == 0] = df.max(axis=0)[divider == 0]
    scaled = np.arcsinh(df / divider)
    return scaled


def calculate_mean_norm(sdata, overwrite=False, c_mask=None, key="normalized_", func_transform=np.arcsinh, **kwargs):
    """Calculate the mean of the normalized images and return a DataFrame with the mean for each image channel"""
    data = []
    metadata = []
    for image_name in [x for x in sdata.images if key not in x]:
        norm_image_name = key + image_name
        if overwrite or norm_image_name not in sdata.images:
            normalize(sdata, image_name, output_layer=norm_image_name, overwrite=True, **kwargs)
        # caluculate the mean of the normalized image for each channel
        c_means = sdata[norm_image_name].mean(["x", "y"]).compute().data
        data.append(c_means)
        metadata.append(image_name)
    df = pd.DataFrame(data, columns=sdata.table.var_names)
    if func_transform is not None:
        df = func_transform(df)
    # remove c_mask columns if it is not None
    if c_mask is not None:
        df: pd.DataFrame = df.drop(columns=c_mask)
    df.index = pd.Index(metadata, name="image_name")
    # sort by index
    df = df.sort_index()
    return df


def get_hexes(col, palette="Set1"):
    if isinstance(palette, str):
        palette = sns.color_palette(palette, n_colors=len(col.unique()))
    lut = dict(zip(col.unique().astype(str), palette.as_hex()))
    return col.astype(str).map(lut)


def clustermap(*args, **kwargs):
    return sns.clustermap(*args, **kwargs)


def signal_clustermap(sdata, signal_threshold=None, fill_value=0, **kwargs):
    df = calculate_snr_ratio(sdata, signal_threshold=signal_threshold)
    df = df.groupby(["image", "channel"]).mean(numeric_only=True).reset_index().drop(columns="snr")
    df = df.set_index(["image", "channel"]).unstack()
    df.columns = df.columns.droplevel(0)
    df.fillna(fill_value, inplace=True)
    return clustermap(df, **kwargs)


def snr_clustermap(sdata, signal_threshold=None, fill_value=0, **kwargs):
    df = calculate_snr_ratio(sdata, signal_threshold=signal_threshold)
    df = df.groupby(["image", "channel"]).mean(numeric_only=True).reset_index().drop(columns="signal")
    df = df.set_index(["image", "channel"]).unstack()
    df.columns = df.columns.droplevel(0)
    df.fillna(fill_value, inplace=True)
    return clustermap(df, **kwargs)


def make_cols_colors(df, palettes=None):
    df = df.copy()
    if palettes is None:
        palettes = [f"Set{i + 1}" for i in range(len(df.columns))]
    for c, p in zip(df.columns, palettes):
        df[c] = get_hexes(df[c], palette=p)
    return df
