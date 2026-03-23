from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import dask
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger as log
from matplotlib.axes import Axes
from spatialdata import SpatialData
from xarray import DataTree

from harpy.image._image import get_dataarray


def image_histogram(
    sdata: SpatialData,
    img_layer: str,
    channel: str | int | Sequence[str | int],
    bins: int,
    scale: str | None = None,
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
    Generate and visualize a histogram for a specified image channel within an image of a ``SpatialData`` object.

    Parameters
    ----------
    sdata
        The input ``SpatialData`` object containing the image data.
    img_layer
        The name of the image layer within `sdata` to analyze.
    channel
        The specific channel of the image data to use for the histogram. Can be a single channel name or index,
        or a sequence of channel names and/or indices.
    bins
        The number of bins for the histogram.
    scale
        Pyramid level to use when ``img_layer`` is multiscale. If ``None``,
        the histogram is computed from ``"scale0"``. Using a lower-resolution
        scale provides a faster but approximate histogram.
    range
        The range of values for the histogram as ``(min, max)``.
        If not provided, range is simply ``(dask.array.nanmin(...), dask.array.nanmax(...))``, thus excluding NaN.
        For ``kind="hist"``, values outside the range are ignored. For ``kind="ecdf"``, the range is used to set the
        x-axis limits only.
    fig_kwargs
        Additional keyword arguments passed to ``plt.subplots``, such as ``dpi`` or ``figsize``, when ``ax=None``
        (and this function therefore creates the subplot(s)). Ignored if ``ax`` is provided.
    bar_kwargs
        Additional keyword arguments passed to ``ax.bar``, such as ``color`` or ``alpha``.
    ax
        An existing axes object to plot the histogram. If ``None``, a new figure and axes will be created.
    output
        The path to save the generated plot. If ``None``, the plot will not be saved.
    density
        If ``True``, normalize the histogram to a density instead of plotting raw counts.
    log_y
        If ``True``, use a logarithmic scale for the y-axis.
    percentile_lines
        Percentile values in the interval ``[0, 100]`` to visualize as vertical guide lines.
    kind
        Plot kind. Choose between ``"hist"`` for a histogram and ``"ecdf"`` for an empirical cumulative
        distribution plot.
    exclude_zeros
        If ``True``, exclude zero-valued pixels before plotting and before computing percentile guide lines.
    exclude_nan
        If ``True``, exclude NaN values before plotting and before computing percentile guide lines.
    title
        Custom plot title. Defaults to the channel name. Only applied directly in the single-channel case.
    ncols
        Number of subplot columns to use when plotting multiple channels.
    subplot_width
        Width of each subplot column when plotting multiple channels and no explicit ``figsize`` is provided in
        ``fig_kwargs``. Ignored when ``fig_kwargs`` contains ``figsize``.
    subplot_height
        Height of each subplot row when plotting multiple channels and no explicit ``figsize`` is provided in
        ``fig_kwargs``. Ignored when ``fig_kwargs`` contains ``figsize``.
    sharex
        Whether to share the x-axis across subplots when plotting multiple channels.
    sharey
        Whether to share the y-axis across subplots when plotting multiple channels.
    **kwargs
        Additional keyword arguments passed to :func:`dask.array.histogram` when ``kind="hist"``.

    Raises
    ------
    AssertionError
        If ``img_layer`` is not found in ``sdata.images``.

    Examples
    --------
    .. code-block:: python

        import harpy as hp

        sdata = hp.datasets.pixie_example()

        ax = hp.qc.image_histogram(
            sdata,
            img_layer="raw_image_fov0",
            channel=hp.im.get_dataarray(sdata, layer="raw_image_fov0").c.data,
            percentile_lines=[0.1, 99.9],
            kind="hist",
            ncols=5,
            subplot_height=3,
            subplot_width=3,
            log_y=False,
            exclude_nan=True,
            exclude_zeros=True,
            density=False,
            bins=100,
        )
    """
    assert img_layer in sdata.images, f"'{img_layer}' not found in 'sdata.images'."
    if scale is not None and not isinstance(sdata.images[img_layer], DataTree):
        log.warning(
            f"Parameter 'scale={scale}' was ignored for image layer '{img_layer}' because it is not multiscale; "
            "histogram will be computed at full resolution."
        )
    se = get_dataarray(sdata, layer=img_layer, scale=scale)
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

    fig_kwargs = dict(fig_kwargs)
    if ax is None:
        fig_kwargs.setdefault("figsize", (6, 4))
        _, ax = plt.subplots(**fig_kwargs)

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
                f"p{percentile:g}",
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
