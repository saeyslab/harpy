from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from spatialdata import SpatialData

from harpy.table._table import ProcessTable

_DEFAULT_COLUMN_COLORS = {
    "total_counts": "#4C78A8",
    "n_genes_by_counts": "#F58518",
    "pct_counts_in_top_2_genes": "#54A24B",
    "pct_counts_in_top_5_genes": "#E45756",
    "n_cells_by_counts": "#72B7B2",
    "mean_counts": "#B279A2",
    "pct_dropout_by_counts": "#FF9DA6",
}


def qc_metric_histogram(
    sdata: SpatialData,
    table_layer: str,
    labels_layer: str | Iterable[str] | None = None,
    column: str = "total_counts",
    display_column: str | None = None,
    dataframe: Literal["obs", "var", "auto"] = "auto",
    ax: Axes | None = None,
    bins: int | str = "auto",
    range: tuple[float, float] | None = None,
    quantile_range: tuple[float, float] | None = None,
    histplot_kwargs: Mapping[str, Any] = MappingProxyType({}),
    median_line_kwargs: Mapping[str, Any] = MappingProxyType({}),
    median_text_kwargs: Mapping[str, Any] = MappingProxyType({}),
    figsize: tuple[float, float] = (5.5, 4.5),
    title: str | None = None,
    color: str | None = None,
    show_median: bool = True,
    show_std: bool = True,
    ylabel: str | None = None,
) -> Axes:
    """
    Plot a QC metric histogram for an :class:`~anndata.AnnData` table.

    This function is read-only and expects QC metrics to already be present on the selected table,
    typically after running :func:`scanpy.pp.calculate_qc_metrics` during preprocessing.

    Parameters
    ----------
    sdata
        :class:`~spatialdata.SpatialData` object containing the table.
    table_layer
        Table layer in ``sdata.tables``.
    labels_layer
        Label layer or layers used to subset the selected table via :class:`~harpy.table._table.ProcessTable`.
        If ``None``, all observations in ``table_layer`` are used.
    column
        QC metric column to plot. The column is searched in ``.obs`` and/or ``.var`` depending on ``dataframe``.
    display_column
        Display name used for the title and x-axis label. If ``None``, ``column`` is converted into a readable label.
    dataframe
        Which annotation dataframe to search for ``column``. With ``"auto"``, the function first checks
        ``adata.obs`` and ``adata.var`` and raises if the column is ambiguous or absent.
    ax
        Matplotlib axes to draw on. If ``None``, a new figure and axes are created.
    bins
        Histogram bin specification passed to :func:`seaborn.histplot`.
        The default ``"auto"`` uses NumPy's automatic bin estimator.
    range
        Lower and upper bounds of the histogram x-axis. Values outside this range are excluded from the plotted
        histogram, but are still included when calculating the median and standard deviation annotations.
    quantile_range
        Quantile interval used to derive the histogram x-axis automatically when ``range`` is ``None``.
        Values outside this interval are excluded from the plotted histogram, but are still included when
        calculating the median and standard deviation annotations.
    histplot_kwargs
        Keyword arguments passed to :func:`seaborn.histplot`.
    median_line_kwargs
        Keyword arguments passed to :meth:`matplotlib.axes.Axes.axvline` for the median guide line.
    median_text_kwargs
        Keyword arguments passed to :meth:`matplotlib.axes.Axes.text` for the median annotation.
    figsize
        Figure size used when ``ax`` is ``None``.
    title
        Plot title. Defaults to ``display_column``.
    color
        Histogram color. If ``None``, a metric-specific default is used when available.
    show_median
        If ``True``, add a dashed median line and annotate the median.
    show_std
        If ``True``, include the standard deviation in the annotation box.
    ylabel
        Y-axis label. If ``None``, a label is chosen based on the selected dataframe.

    Returns
    -------
    :class:`matplotlib.axes.Axes`
        Axes containing the histogram.
    """
    process_table = ProcessTable(sdata, labels_layer=labels_layer, table_layer=table_layer)
    adata = sdata.tables[table_layer]
    obs_mask = None
    if process_table.labels_layer is not None:
        obs_mask = adata.obs[process_table.region_key].isin(process_table.labels_layer).to_numpy()

    resolved_dataframe = _resolve_dataframe(adata, column=column, dataframe=dataframe)
    if resolved_dataframe == "var" and obs_mask is not None and not obs_mask.all():
        raise ValueError(
            "Plotting '.var' QC metrics for a subset of 'labels_layer' is not supported without recomputing QC metrics. "
            "Please plot a table layer that already contains the desired subset-specific QC metrics, or use dataframe='obs'."
        )

    values = getattr(adata, resolved_dataframe)[column]
    if resolved_dataframe == "obs" and obs_mask is not None:
        values = values.loc[obs_mask]

    if not pd.api.types.is_numeric_dtype(values):
        raise TypeError(
            f"Column '{column}' in 'adata.{resolved_dataframe}' is not numeric and cannot be visualized as a histogram."
        )

    values = values.dropna()
    if values.empty:
        raise ValueError(f"Column '{column}' in 'adata.{resolved_dataframe}' does not contain any non-null values.")
    plot_range = range
    if plot_range is None and quantile_range is not None:
        qmin, qmax = quantile_range
        if not 0 <= qmin <= qmax <= 1:
            raise ValueError(
                f"Parameter 'quantile_range' must satisfy 0 <= qmin <= qmax <= 1; received {quantile_range!r}."
            )
        lower, upper = values.quantile([qmin, qmax]).to_numpy()
        plot_range = (float(lower), float(upper))

    values_for_plot = values
    if plot_range is not None:
        values_for_plot = values[(values >= plot_range[0]) & (values <= plot_range[1])]
        if values_for_plot.empty:
            raise ValueError(
                f"No values remaining in column '{column}' after applying range {plot_range!r} for plotting."
            )

    display_name = display_column if display_column is not None else _format_display_name(column)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    histplot_kwargs = dict(histplot_kwargs)
    histplot_kwargs.setdefault("kde", True)
    histplot_kwargs.setdefault("stat", "count")
    histplot_kwargs.setdefault("edgecolor", "white")
    histplot_kwargs.setdefault("linewidth", 0.8)
    histplot_kwargs.setdefault("alpha", 0.9)
    histplot_kwargs.setdefault("color", color if color is not None else _DEFAULT_COLUMN_COLORS.get(column, "#4C78A8"))
    if bins is not None:
        histplot_kwargs.setdefault("bins", bins)
    if plot_range is not None:
        histplot_kwargs.setdefault("binrange", plot_range)

    sns.histplot(values_for_plot, ax=ax, **histplot_kwargs)

    if show_median:
        median_value = float(values.median())
        line_kwargs = dict(median_line_kwargs)
        line_kwargs.setdefault("color", "black")
        line_kwargs.setdefault("linestyle", "--")
        line_kwargs.setdefault("linewidth", 1.5)
        ax.axvline(median_value, **line_kwargs)

        text_kwargs = dict(median_text_kwargs)
        text_kwargs.setdefault("transform", ax.transAxes)
        text_kwargs.setdefault("ha", "left")
        text_kwargs.setdefault("va", "top")
        text_kwargs.setdefault("fontsize", 10)
        text_kwargs.setdefault("family", "monospace")
        text_kwargs.setdefault(
            "bbox",
            {"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )
        annotation_text = f"{'Median':<6}: {_format_metric_value(median_value)}"
        if show_std:
            std_value = float(values.std())
            annotation_text += f"\n{'SD':<6}: {_format_metric_value(std_value)}"
        ax.text(
            0.02,
            0.95,
            annotation_text,
            **text_kwargs,
        )

    if title is not None:
        ax.set_title(title, weight="bold")
    ax.set_xlabel(display_name)
    ax.set_ylabel(ylabel if ylabel is not None else _default_ylabel(resolved_dataframe))
    _style_qc_axis(ax)
    return ax


def _resolve_dataframe(adata, column: str, dataframe: Literal["obs", "var", "auto"]) -> Literal["obs", "var"]:
    if dataframe in {"obs", "var"}:
        if column not in getattr(adata, dataframe).columns:
            raise ValueError(f"Column '{column}' not found in 'adata.{dataframe}'.")
        return dataframe

    in_obs = column in adata.obs.columns
    in_var = column in adata.var.columns

    if in_obs and in_var:
        raise ValueError(
            f"Column '{column}' is present in both 'adata.obs' and 'adata.var'. Please set 'dataframe' explicitly."
        )
    if in_obs:
        return "obs"
    if in_var:
        return "var"
    raise ValueError(f"Column '{column}' was not found in either 'adata.obs' or 'adata.var'.")


def _style_qc_axis(ax: Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)


def _format_display_name(column: str) -> str:
    return column.replace("_", " ").strip().title()


def _default_ylabel(dataframe: Literal["obs", "var"]) -> str:
    return "Number of cells" if dataframe == "obs" else "Number of genes"


def _format_metric_value(value: float) -> str:
    if np.isclose(value, round(value)):
        return f"{value:,.0f}"
    if abs(value) >= 100:
        return f"{value:,.1f}"
    return f"{value:,.2f}"
