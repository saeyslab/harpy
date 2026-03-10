"""Calculate various image quality metrics"""

from collections.abc import Mapping, Sequence
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
from loguru import logger as log
from matplotlib.axes import Axes
from spatialdata import SpatialData

from harpy.image import normalize
from harpy.image._image import get_dataarray

try:
    import textalloc as ta

except ImportError:
    log.warning(
        "'textalloc' not installed, to use 'harpy.pl.group_snr_ratio' and 'harpy.pl.snr_ratio', please install this library."
    )


def histogram(
    sdata: SpatialData,
    img_layer: str,
    channel: str | int | Sequence[str | int],
    bins: int,
    range: tuple[float, float] | None = None,
    ax: Axes | np.ndarray | None = None,
    output: str | Path = None,
    fig_kwargs: dict[str, Any] = MappingProxyType({}),  # kwargs passed to plt.figure, e.g. dpi, figsize
    bar_kwargs: Mapping[str, Any] = MappingProxyType({}),  # kwargs passed to ax.bar, e.g. color and alpha
    density: bool = False,
    log_y: bool = False,
    percentile_lines: Sequence[float] | None = None,
    kind: str = "hist",
    exclude_zeros: bool = True,
    exclude_nan: bool = True,
    title: str | None = None,
    ncols: int = 3,
    subplot_width: float = 4,
    subplot_height: float = 3.5,
    sharex: bool = False,
    sharey: bool = False,
    **kwargs,
) -> Axes | np.ndarray:
    """
    Generate and visualize a histogram for a specified image channel within an image of a `SpatialData` object.

    Parameters
    ----------
    sdata
        The input `SpatialData` object containing the image data.
    img_layer
        The name of the image layer within `sdata` to analyze.
    channel
        The specific channel of the image data to use for the histogram. Can be a single channel name or index,
        or a sequence of channel names and/or indices.
    bins
        The number of bins for the histogram.
    range
        The range of values for the histogram as `(min, max)`.
        If not provided, range is simply `(dask.array.nanmin(...), dask.array.nanmax(...))` thus excluding NaN.
        For `kind="hist"`, values outside the range are ignored. For `kind="ecdf"`, the range is used to set the
        x-axis limits only.
    fig_kwargs
        Additional keyword arguments passed to `plt.subplots`, such as `dpi` or `figsize`, when `ax=None`
        (and this function therefore creates the subplot(s)). Ignored if `ax` is provided.
    bar_kwargs
        Additional keyword arguments passed to `ax.bar`, such as `color` or `alpha`.
    ax
        An existing axes object to plot the histogram. If `None`, a new figure and axes will be created.
    output
        The path to save the generated plot. If `None`, the plot will not be saved.
    density
        If `True`, normalize the histogram to a density instead of plotting raw counts.
    log_y
        If `True`, use a logarithmic scale for the y-axis.
    percentile_lines
        Percentile values in the interval `[0, 100]` to visualize as vertical guide lines.
    kind
        Plot kind. Choose between `"hist"` for a histogram and `"ecdf"` for an empirical cumulative distribution plot.
    exclude_zeros
        If `True`, exclude zero-valued pixels before plotting and before computing percentile guide lines.
    exclude_nan
        If `True`, exclude NaN values before plotting and before computing percentile guide lines.
    title
        Custom plot title. Defaults to the channel name. Only applied directly in the single-channel case.
    ncols
        Number of subplot columns to use when plotting multiple channels.
    subplot_width
        Width of each subplot column when plotting multiple channels and no explicit `figsize` is provided in
        `fig_kwargs`.
    subplot_height
        Height of each subplot row when plotting multiple channels and no explicit `figsize` is provided in
        `fig_kwargs`.
    sharex
        Whether to share the x-axis across subplots when plotting multiple channels.
    sharey
        Whether to share the y-axis across subplots when plotting multiple channels.
    **kwargs
        Additional keyword arguments passed to :func:`dask.array.histogram` when `kind="hist"`.

    Returns
    -------
        The axes object containing the histogram plot, or an array of axes when multiple channels are provided.

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
    ...     density=True,
    ...     log_y=False,
    ...     exclude_zeros=True,
    ...     percentile_lines=[0.1, 99.9],
    ...     fig_kwargs={"figsize": (5, 5)},
    ...     bar_kwargs={"color": "blue", "alpha": 0.7},
    ...     output="histogram.png"
    ... )
    """
    assert img_layer in sdata.images, f"'{img_layer}' not found in 'sdata.images'."
    se = get_dataarray(sdata, layer=img_layer)
    channel_names = _resolve_channels(se.c.data.tolist(), channel)

    if len(channel_names) == 1:
        axis = ax if isinstance(ax, Axes) or ax is None else np.asarray(ax).ravel()[0]
        channel_title = title if title is not None else channel_names[0]
        result_ax = _plot_histogram_for_channel(
            se=se,
            channel=channel_names[0],
            bins=bins,
            range=range,
            ax=axis,
            fig_kwargs=fig_kwargs,
            bar_kwargs=bar_kwargs,
            density=density,
            log_y=log_y,
            percentile_lines=percentile_lines,
            kind=kind,
            exclude_zeros=exclude_zeros,
            exclude_nan=exclude_nan,
            title=channel_title,
            **kwargs,
        )
        if output is not None:
            result_ax.figure.savefig(output)
        return result_ax

    if title is not None:
        log.warning("Parameter 'title' is ignored when plotting multiple channels.")

    fig_kwargs = dict(fig_kwargs)
    fig_kwargs.setdefault(
        "figsize",
        (
            subplot_width * min(ncols, len(channel_names)),
            subplot_height * int(np.ceil(len(channel_names) / ncols)),
        ),
    )
    fig, axes = _prepare_histogram_axes(
        n_plots=len(channel_names),
        ncols=ncols,
        ax=ax,
        fig_kwargs=fig_kwargs,
        sharex=sharex,
        sharey=sharey,
    )

    axes_flat = axes.ravel()
    axes_in_use = axes_flat[: len(channel_names)]
    for axis, channel_name in zip(axes_in_use, channel_names, strict=True):
        _plot_histogram_for_channel(
            se=se,
            channel=channel_name,
            bins=bins,
            range=range,
            ax=axis,
            fig_kwargs=fig_kwargs,
            bar_kwargs=bar_kwargs,
            density=density,
            log_y=log_y,
            percentile_lines=percentile_lines,
            kind=kind,
            exclude_zeros=exclude_zeros,
            exclude_nan=exclude_nan,
            title=channel_name,
            **kwargs,
        )

    for axis in axes_flat[len(channel_names) :]:
        fig.delaxes(axis)

    fig.tight_layout()
    if output is not None:
        fig.savefig(output)
    return axes


def _plot_histogram_for_channel(
    se,
    channel: str,
    bins: int,
    range: tuple[float, float] | None,
    ax: Axes | None,
    fig_kwargs: Mapping[str, Any],
    bar_kwargs: Mapping[str, Any],
    density: bool,
    log_y: bool,
    percentile_lines: Sequence[float] | None,
    kind: str,
    exclude_zeros: bool,
    exclude_nan: bool,
    title: str,
    **kwargs,
) -> Axes:
    if kind not in {"hist", "ecdf"}:
        raise ValueError(f"Unknown 'kind': {kind}. Expected one of ['hist', 'ecdf'].")

    array = se.data[se.c.data.tolist().index(channel)].ravel()
    array = _filter_image_values(array, exclude_zeros=exclude_zeros, exclude_nan=exclude_nan)

    if range is None:
        range = tuple(dask.compute(da.nanmin(array), da.nanmax(array)))

    # Create axes if not provided
    fig_kwargs = dict(fig_kwargs)
    if ax is None:
        fig_kwargs.setdefault("figsize", (6, 4))
        _, ax = plt.subplots(**fig_kwargs)
    else:
        _ = ax.figure

    # Plot
    bar_kwargs = dict(bar_kwargs)
    color = bar_kwargs.pop("color", sns.color_palette("deep")[0])
    alpha = bar_kwargs.pop("alpha", bar_kwargs.pop("ahlpa", 0.8))
    align = bar_kwargs.pop("align", "edge")
    linewidth = bar_kwargs.pop("linewidth", 0)

    if kind == "hist":
        hist, bin_edges = da.histogram(array, bins=bins, range=range, density=density, **kwargs)
        hist, bin_edges = dask.compute(hist, bin_edges)
        ax.bar(
            bin_edges[:-1],
            hist,
            width=(bin_edges[1] - bin_edges[0]),
            align=align,
            alpha=alpha,
            color=color,
            linewidth=linewidth,
        )
        ax.set_ylabel("Density" if density else "Frequency")
    else:
        values = np.sort(array.compute())
        y = np.arange(1, len(values) + 1) / len(values)
        ax.step(values, y, where="post", color=color, alpha=alpha, linewidth=max(linewidth, 1.5))
        ax.set_ylabel("Cumulative fraction")
        if range is not None:
            ax.set_xlim(range)

    if percentile_lines is not None:
        percentile_values = da.percentile(array, q=list(percentile_lines)).compute()
        percentile_values = np.atleast_1d(percentile_values)
        for percentile, value in zip(percentile_lines, percentile_values, strict=True):
            ax.axvline(value, color=color, linestyle="--", linewidth=1, alpha=0.6)
            ax.text(
                value,
                0.98,
                f"q{percentile:g}",
                transform=ax.get_xaxis_transform(),
                rotation=90,
                va="top",
                ha="right",
                color=color,
                alpha=0.8,
            )

    ax.set_xlabel("Intensity")
    ax.set_title(title)

    if log_y:
        ax.set_yscale("log")

    sns.despine(ax=ax)

    return ax


def _resolve_channels(channel_names: list[str], channel: str | int | Sequence[str | int]) -> list[str]:
    if isinstance(channel, (str, int, np.integer)):
        return [_resolve_channel_name(channel_names, channel)]
    return [_resolve_channel_name(channel_names, item) for item in channel]


def _resolve_channel_name(channel_names: list[str], channel: str | int) -> str:
    if isinstance(channel, str):
        if channel not in channel_names:
            raise ValueError(f"Channel '{channel}' not found in image layer.")
        return channel
    if isinstance(channel, (int, np.integer)):
        return channel_names[channel]
    raise TypeError(f"Unsupported channel type: {type(channel)!r}.")


def _prepare_histogram_axes(
    n_plots: int,
    ncols: int,
    ax: Axes | np.ndarray | None,
    fig_kwargs: Mapping[str, Any],
    sharex: bool,
    sharey: bool,
) -> tuple[plt.Figure, np.ndarray]:
    ncols = max(1, min(ncols, n_plots))
    nrows = int(np.ceil(n_plots / ncols))

    if ax is None:
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey, **fig_kwargs)
        return fig, np.atleast_1d(axes)

    axes = np.atleast_1d(ax)
    if axes.size < n_plots:
        raise ValueError(f"Provided 'ax' contains {axes.size} axes, but {n_plots} channels were requested.")
    first_axis = axes.ravel()[0]
    return first_axis.figure, axes


def _filter_image_values(array: da.Array, *, exclude_zeros: bool, exclude_nan: bool) -> da.Array:
    if not exclude_nan and not exclude_zeros:
        return array

    mask = da.ones(array.shape, dtype=bool, chunks=array.chunks)
    if exclude_nan:
        mask &= ~da.isnan(array)
    if exclude_zeros:
        mask &= array != 0
    array = da.compress(mask, array)
    if any(np.isnan(c).any() for c in array.chunks):
        array = array.compute_chunk_sizes()
    return array


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
    image_names=None,
    block_size=10000,
    channel_names=None,
    cycles=None,
    signal_threshold=None,
):
    log.debug("Calculating SNR ratio")
    data = []
    table = sdata[table_name]
    if image_names is None:
        image_names = sdata.images
    if channel_names is None:
        channel_names = table.var_names
    if cycles:
        if cycles in table.var.keys():
            cycles = table.var[cycles]
        else:
            cycles = cycles
    else:
        cycles = [None] * len(channel_names)
    for image in image_names:
        for cycle, channel_name in zip(cycles, channel_names, strict=True):
            float_block = sdata[image].sel(c=channel_name).data.rechunk(block_size)
            img = float_block.compute()
            snr, signal = calculate_snr(img)
            if signal_threshold and signal < signal_threshold:
                continue
            data += [(image, cycle, channel_name, snr, signal)]
            del img
    df_img = pd.DataFrame(data, columns=["image", "cycle", "channel", "snr", "signal"])
    return df_img


def snr_ratio(sdata, ax=None, loglog=True, color="black", **kwargs):
    """Plot the signal to noise ratio. On the x-axis is the signal intensity and on the y-axis is the SNR-ratio"""
    log.debug("Plotting SNR ratio")
    if ax is None:
        fig, ax = plt.subplots()
    df_img = calculate_snr_ratio(sdata, cycles="cycle" if color == "cycle" else None, **kwargs)
    if loglog:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)

    # group by "channel" and take the mean of "image" and "cycle"
    df_img = df_img.groupby(["channel"]).mean(numeric_only=True)
    # sort by channel
    df_img = df_img.sort_values("channel")
    # do a scatter plot
    if color == "cycle":
        palette = sns.color_palette("viridis", n_colors=len(df_img["cycle"].unique()))
        cmap = sns.color_palette("viridis", n_colors=len(df_img["cycle"].unique()), as_cmap=True)
        df_img["cycle"] = get_hexes(df_img["cycle"], palette=palette)
    log.debug(df_img.head())
    _plot_snr_ratio(df_img, ax, color, text_list=df_img.index.values)
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
    for ax, sample in zip(axs.flatten(), df_img.index.levels[0], strict=True):
        if loglog:
            ax.set_xscale("log", base=2)
            ax.set_yscale("log", base=2)
        log.debug(sample)
        sample_df = df_img.loc[sample]
        #     ax = axs[i // 2, i % 2]  # Get the correct subplot
        ax.set_title(sample)
        _plot_snr_ratio(sample_df, ax, color, text_list=sample_df.index.values)
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
    lut = dict(zip(col.unique().astype(str), palette.as_hex(), strict=True))
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
    for c, p in zip(df.columns, palettes, strict=True):
        df[c] = get_hexes(df[c], palette=p)
    return df


def marker_supervenn(markers_per_image: dict[str, list[str]]):
    from supervenn import supervenn

    image_names = markers_per_image.keys()
    marker_set_per_image = [set(v) for v in markers_per_image.values()]
    plot = supervenn(marker_set_per_image, set_annotations=list(image_names))

    # change axis labels of plot
    axes = plot.axes
    axes["main"].set_ylabel("Samples")
    axes["main"].set_xlabel("Marker names")

    return plot


def supervenn_of_images(sdata: SpatialData):
    markers_per_image = {image: sdata[image].coords["c"].to_numpy().tolist() for image in sdata.images}
    return marker_supervenn(markers_per_image)
