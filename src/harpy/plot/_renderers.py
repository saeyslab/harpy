"""The histogram and scatter renderers shared by the QC plots of ``hp.qc`` and the table plots of ``hp.tb.pl``.

Callers select the values to draw; nothing here reads a SpatialData object, a
table or a summary.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes


def _plot_histogram(
    values: pd.Series,
    *,
    median: float,
    std: float,
    ax: Axes | None,
    bins: int | str,
    range: tuple[float, float] | None,
    quantile_range: tuple[float, float] | None,
    histplot_kwargs: Mapping[str, Any],
    median_line_kwargs: Mapping[str, Any],
    median_text_kwargs: Mapping[str, Any],
    figsize: tuple[float, float],
    title: str | None,
    color: str,
    show_median: bool,
    show_std: bool,
    xlabel: str,
    count_ylabel: str,
    percent_ylabel: str,
    empty_message: str = "No retained spatial bins",
) -> Axes:
    """Render one numerical population with supplied full-population annotations.

    Callers select the population and remove missing values. Display limits
    only filter what is drawn, not the percentage denominator or annotations.
    No source data or summary objects are accessed here. Seaborn draws count
    histograms and the matching KDE in percentage mode; scale both by 100 / N,
    where N is the population size before display filtering.
    """
    options = dict(histplot_kwargs)
    reserved = sorted(set(options) & {"data", "x", "y", "hue", "weights", "ax"})
    if reserved:
        raise ValueError(f"histplot_kwargs cannot override histogram data selection or weighting: {reserved}.")
    if options.get("multiple") == "fill":
        raise ValueError("histplot_kwargs multiple='fill' would change the histogram's population normalization.")
    options.setdefault("kde", True)
    options.setdefault("stat", "count")
    options.setdefault("element", "step")
    options.setdefault("fill", True)
    if options["fill"]:
        options.setdefault("edgecolor", "none")
    # Unfilled histograms need a visible outline; filled ones omit borders.
    if "lw" not in options:
        options.setdefault("linewidth", 0 if options["fill"] else 1.5)
    options.setdefault("alpha", 0.5)
    options.setdefault("color", color)
    # Seaborn can update these nested dictionaries while constructing its KDE.
    for key in ("kde_kws", "line_kws"):
        if key in options:
            options[key] = dict(options[key])
    kde_line_options = options.setdefault("line_kws", {})
    if "lw" not in kde_line_options:
        kde_line_options.setdefault("linewidth", 2)

    plot_range = range
    if plot_range is None and quantile_range is not None:
        qmin, qmax = quantile_range
        if not 0 <= qmin <= qmax <= 1:
            raise ValueError(
                f"Parameter 'quantile_range' must satisfy 0 <= qmin <= qmax <= 1; received {quantile_range!r}."
            )
        if not values.empty:
            plot_range = tuple(float(value) for value in values.quantile([qmin, qmax]))
    if plot_range is not None and (not np.isfinite(plot_range).all() or plot_range[0] > plot_range[1]):
        raise ValueError("Parameter 'range' must contain finite, nondecreasing bounds.")
    displayed = values if plot_range is None else values[values.between(*plot_range)]
    if not values.empty and displayed.empty:
        raise ValueError(f"No values remaining after applying range {plot_range!r} for plotting.")

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    percent = options["stat"] == "percent"
    if values.empty:
        ax.text(0.5, 0.5, empty_message, ha="center", va="center", transform=ax.transAxes)
    else:
        log_scale = options.get("log_scale", False)
        log_x = log_scale[0] if isinstance(log_scale, (tuple, list)) else log_scale
        if (log_x or ax.get_xscale() == "log") and (displayed <= 0).any():
            raise ValueError(
                "A logarithmic histogram cannot display non-positive counts; set an explicit display range."
            )
        options["kde"] = bool(options["kde"]) and len(displayed) > 1 and displayed.nunique() > 1
        if bins is not None:
            options.setdefault("bins", bins)
        # Equal quantile limits are valid for constant populations. Let the
        # bin estimator expand that single value to a nonzero-width interval.
        if plot_range is not None and plot_range[0] < plot_range[1]:
            options.setdefault("binrange", plot_range)
        if percent:
            options["stat"] = "count"
        starts = (len(ax.patches), len(ax.collections), len(ax.lines))
        sns.histplot(x=displayed, ax=ax, **options)
        if percent:
            _scale_histogram_artists(ax, starts=starts, factor=100.0 / len(values))

        if show_median:
            line_options = dict(median_line_kwargs)
            line_options.setdefault("color", "black")
            line_options.setdefault("linestyle", "--")
            line_options.setdefault("linewidth", 1.5)
            ax.axvline(median, **line_options)

            text_options = dict(median_text_kwargs)
            text_options.setdefault("transform", ax.transAxes)
            text_options.setdefault("ha", "left")
            text_options.setdefault("va", "top")
            text_options.setdefault("fontsize", 10)
            text_options.setdefault("family", "monospace")
            text_options.setdefault(
                "bbox", {"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "none", "alpha": 0.8}
            )
            annotation = f"{'Median':<6}: {_format_metric_value(median)}"
            if show_std:
                annotation += f"\n{'SD':<6}: {_format_metric_value(std)}"
            ax.text(0.02, 0.95, annotation, **text_options)
        if len(displayed) < len(values):
            ax.text(
                0.98,
                0.95,
                f"Displayed: {len(displayed):,} / {len(values):,}",
                ha="right",
                va="top",
                transform=ax.transAxes,
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "none", "alpha": 0.8},
            )

    if title is not None:
        ax.set_title(title, weight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(percent_ylabel if percent else count_ylabel)
    _style_qc_axis(ax)
    return ax


def _scale_histogram_artists(ax: Axes, *, starts: tuple[int, int, int], factor: float) -> None:
    """Scale only the current histogram/KDE, preserving earlier axes contents.

    Seaborn's bars, filled step/poly histograms, and lines use different
    Matplotlib artists. Scaling their count heights together keeps the KDE on
    the same percentage scale without changing histogram edges or refitting it.
    """
    patch_start, collection_start, line_start = starts
    for patch in ax.patches[patch_start:]:
        patch.set_y(patch.get_y() * factor)
        patch.set_height(patch.get_height() * factor)
    for collection in ax.collections[collection_start:]:
        for path in collection.get_paths():
            path.vertices[:, 1] *= factor
        collection.stale = True
    for line in ax.lines[line_start:]:
        line.set_ydata(np.asarray(line.get_ydata()) * factor)
    ax.relim()
    # Axes.relim() ignores collections. Existing ones retain their limits,
    # but filled histograms can cache the pre-scaling bounds, so derive the
    # new histogram limits from its scaled vertices instead.
    for collection in ax.collections[:collection_start]:
        bounds = collection.get_datalim(ax.transData)
        if np.isfinite(bounds.get_points()).all():
            ax.update_datalim(bounds.get_points())
    for collection in ax.collections[collection_start:]:
        # Convert the scaled vertices to data coordinates so axis limits reflect
        # percentages, not the original counts. Transform subtraction composes
        # with the inverse of ax.transData; it is not numeric subtraction.
        to_data = collection.get_transform() - ax.transData
        for path in collection.get_paths():
            ax.update_datalim(to_data.transform(path.vertices))
    ax.autoscale_view()


def _plot_scatter(
    x_values: pd.Series,
    y_values: pd.Series,
    *,
    x_label: str,
    y_label: str,
    ax: Axes | None,
    figsize: tuple[float, float],
    title: str | None,
    cmap: str | None,
    histplot_kwargs: Mapping[str, Any],
    show_regplot: bool,
    regplot_kwargs: Mapping[str, Any],
) -> Axes:
    """Render two aligned numerical series as a 2D histogram, with an optional lowess line.

    Shared by :func:`harpy.tb.pl.scatter` and :func:`harpy.qc.instance_scatter`:
    callers select the values and remove missing ones. No source data or
    summary is accessed here, and the option mappings are not modified.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    histplot_kwargs = dict(histplot_kwargs)
    histplot_kwargs.setdefault("bins", 60)
    histplot_kwargs.setdefault("cbar", True)
    histplot_kwargs.setdefault("pmax", 0.98)
    if cmap is not None:
        histplot_kwargs.setdefault("cmap", cmap)
    sns.histplot(x=x_values, y=y_values, ax=ax, **histplot_kwargs)

    if show_regplot:
        regplot_kwargs = dict(regplot_kwargs)
        regplot_kwargs.setdefault("scatter", False)
        regplot_kwargs.setdefault("lowess", True)
        regplot_kwargs.setdefault("line_kws", {"color": "#1F3B4D", "lw": 1.5})
        sns.regplot(x=x_values, y=y_values, ax=ax, **regplot_kwargs)

    if title is not None:
        ax.set_title(title, weight="bold")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    _style_qc_axis(ax)
    return ax


def _style_qc_axis(ax: Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)


def _format_metric_value(value: float) -> str:
    if not np.isfinite(value):
        return "N/A"
    if np.isclose(value, round(value)):
        return f"{value:,.0f}"
    if abs(value) >= 100:
        return f"{value:,.1f}"
    return f"{value:,.2f}"
