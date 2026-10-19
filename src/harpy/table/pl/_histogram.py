"""Histograms of the numeric ``.obs`` or ``.var`` columns of an AnnData table."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib.axes import Axes

from harpy.plot._renderers import _plot_histogram
from harpy.table._utils import _region_rows

_DEFAULT_COLUMN_COLORS = {
    "log1p_total_counts": "#577590",
    "total_counts": "#4C78A8",
    "n_genes_by_counts": "#F58518",
    "pct_counts_in_top_2_genes": "#54A24B",
    "pct_counts_in_top_5_genes": "#E45756",
    "n_cells_by_counts": "#72B7B2",
    "mean_counts": "#B279A2",
    "pct_dropout_by_counts": "#FF9DA6",
}


def histogram(
    adata: AnnData,
    column: str = "total_counts",
    *,
    region: str | Sequence[str] | None = None,
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
) -> Axes:
    """
    Plot the histogram of a numeric ``.obs`` or ``.var`` column of a table.

    For example a QC metric that :func:`scanpy.pp.calculate_qc_metrics` added.
    The table is not modified.

    Parameters
    ----------
    adata
        The table, for example ``sdata.tables["table"]``.
    column
        The column to plot, searched in ``.obs`` and/or ``.var`` depending on ``dataframe``.
    region
        ``None``, the default, plots every row. Otherwise, one or more regions
        of the table's SpatialData annotation, ``uns["spatialdata_attrs"]["region"]``,
        whose rows of ``.obs`` are plotted, whatever the type of the annotated
        elements. A ``.var`` column describes the whole table, so it cannot be
        plotted for a subset of the regions; a table that the caller subsets,
        such as ``adata[adata.obs["region"] == "a"]``, also keeps the ``.var``
        columns of the whole table.
    display_column
        Display name used for the x-axis label. If ``None``, ``column`` is converted into a readable label.
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
        Plotting options passed to :func:`seaborn.histplot`; defaults to a
        borderless filled-step count histogram with ``alpha=0.5`` and a KDE
        of line width 2. ``stat="percent"`` uses the full selected, non-null
        population, before display filtering, as its denominator.
        Unlike direct Seaborn normalization, clipped bars may sum to less
        than 100%. Data selection, grouping, and weights cannot be overridden.
    median_line_kwargs
        Keyword arguments passed to :meth:`matplotlib.axes.Axes.axvline` for the median guide line.
    median_text_kwargs
        Keyword arguments passed to :meth:`matplotlib.axes.Axes.text` for the median annotation.
    figsize
        Figure size used when ``ax`` is ``None``.
    title
        Optional plot title. None leaves the axes title unchanged.
    color
        Histogram color. If ``None``, a metric-specific default is used when available.
    show_median
        If ``True``, add a dashed median line and annotate the median.
    show_std
        If ``True``, include the standard deviation in the annotation box.

    Returns
    -------
    :class:`matplotlib.axes.Axes` containing the histogram.
        Customize axis labels on the returned axes with ``ax.set(...)``.

    See Also
    --------
    harpy.tb.pl.histograms
    harpy.tb.pl.scatter

    Examples
    --------
    .. code-block:: python

        import harpy as hp

        sdata = hp.datasets.xenium_human_ovarian_cancer(
            subset=True,
            processed=True,
        )

        hp.tb.pl.histogram(
            sdata.tables["table_transcriptomics_preprocessed"],
            "total_counts",
            region="nucleus_segmentation_mask",
            dataframe="obs",
            quantile_range=(0.1, 0.99),
        )
    """
    rows = _region_rows(adata, region)
    resolved_dataframe = _resolve_dataframe(adata, column=column, dataframe=dataframe)
    _check_var_selection(resolved_dataframe, rows)

    values = getattr(adata, resolved_dataframe)[column]
    if resolved_dataframe == "obs" and rows is not None:
        values = values.loc[rows]

    if not pd.api.types.is_numeric_dtype(values):
        raise TypeError(
            f"Column '{column}' in 'adata.{resolved_dataframe}' is not numeric and cannot be visualized as a histogram."
        )

    values = values.dropna()
    if values.empty:
        raise ValueError(f"Column '{column}' in 'adata.{resolved_dataframe}' does not contain any non-null values.")
    display_name = display_column if display_column is not None else _format_display_name(column)
    return _plot_histogram(
        values,
        median=float(values.median()),
        std=float(values.std()),
        ax=ax,
        bins=bins,
        range=range,
        quantile_range=quantile_range,
        histplot_kwargs=histplot_kwargs,
        median_line_kwargs=median_line_kwargs,
        median_text_kwargs=median_text_kwargs,
        figsize=figsize,
        title=title,
        color=color if color is not None else _DEFAULT_COLUMN_COLORS.get(column, "#4C78A8"),
        show_median=show_median,
        show_std=show_std,
        xlabel=display_name,
        count_ylabel=_default_ylabel(resolved_dataframe),
        percent_ylabel="Percentage of instances (%)" if resolved_dataframe == "obs" else "Percentage of features (%)",
    )


def histograms(
    adata: AnnData,
    metrics: Sequence[tuple[Literal["obs", "var"], str]] = (
        ("obs", "total_counts"),
        ("obs", "n_genes_by_counts"),
        ("var", "total_counts"),
        ("var", "log1p_total_counts"),
        ("var", "n_cells_by_counts"),
        ("var", "mean_counts"),
    ),
    *,
    region: str | Sequence[str] | None = None,
    ax: np.ndarray | Sequence[Axes] | None = None,
    bins: int | str | Sequence[int | str] = "auto",
    range: tuple[float, float] | None = None,
    quantile_range: tuple[float, float] | None = None,
    histplot_kwargs: Mapping[str, Any] = MappingProxyType({}),
    median_line_kwargs: Mapping[str, Any] = MappingProxyType({}),
    median_text_kwargs: Mapping[str, Any] = MappingProxyType({}),
    figsize: tuple[float, float] | None = None,
    ncols: int = 3,
    subplot_width: float = 5.5,
    subplot_height: float = 4.5,
    sharex: bool = False,
    sharey: bool = False,
    title: str | None = None,
    display_column: str | Sequence[str | None] | None = None,
    color: str | Sequence[str] | None = None,
    show_median: bool = True,
    show_std: bool = True,
) -> np.ndarray:
    """
    Plot a panel of histograms of numeric ``.obs`` and ``.var`` columns of a table.

    By default, the standard panel of QC metrics that :func:`scanpy.pp.calculate_qc_metrics`
    adds. The table is not modified.

    Parameters
    ----------
    adata
        The table, for example ``sdata.tables["table"]``.
    metrics
        Sequence of ``(dataframe, column)`` tuples to plot. Defaults to a standard transcript QC panel obtained
        through :func:`scanpy.pp.calculate_qc_metrics`.
    region
        ``None``, the default, plots every row. Otherwise, one or more regions
        of the table's SpatialData annotation, as in :func:`harpy.tb.pl.histogram`;
        a ``.var`` column cannot be plotted for a subset of the regions.
    ax
        Array-like collection of axes to draw on. If ``None``, subplot axes are created.
    bins
        Histogram bin specification passed to :func:`seaborn.histplot`.
        If a sequence is provided, it must match the length of ``metrics`` and each value is applied to the
        corresponding panel.
    range
        Lower and upper bounds of the histogram x-axis applied to all panels.
    quantile_range
        Quantile interval used to derive the histogram x-axis automatically when ``range`` is ``None``.
    histplot_kwargs
        Plotting options forwarded through :func:`harpy.tb.pl.histogram`.
        ``stat="percent"`` uses each metric's full selected, non-null
        population before display filtering; clipped bars may sum to less
        than 100%. Data selection, grouping, and weights cannot be overridden.
    median_line_kwargs
        Keyword arguments passed to :meth:`matplotlib.axes.Axes.axvline`.
    median_text_kwargs
        Keyword arguments passed to :meth:`matplotlib.axes.Axes.text`.
    figsize
        Figure size used when ``ax`` is ``None``. If ``None``, a size is inferred from ``ncols`` and the number of metrics.
    ncols
        Number of subplot columns when ``ax`` is ``None``.
    subplot_width
        Width of each subplot when ``figsize`` is not provided.
    subplot_height
        Height of each subplot when ``figsize`` is not provided.
    sharex
        Whether to share x-axes across subplots when ``ax`` is ``None``.
    sharey
        Whether to share y-axes across subplots when ``ax`` is ``None``.
    title
        Figure title applied when ``ax`` is ``None``.
    display_column
        Display name override for the plotted metrics. If a single string is provided, it is applied to all panels.
        If a sequence is provided, it must match the length of ``metrics`` and each value is applied to the
        corresponding panel. Entries set to ``None`` fall back to a readable label derived from the metric name.
    color
        Histogram color override. If a single string is provided, it is applied to all panels. If a sequence is
        provided, it must match the length of ``metrics`` and each value is applied to the corresponding panel.
    show_median
        If ``True``, add a dashed median line and annotate the median.
    show_std
        If ``True``, include the standard deviation in the annotation box.

    Returns
    -------
    :class:`numpy.ndarray` containing the histogram axes.

    See Also
    --------
    harpy.tb.pl.histogram

    Examples
    --------
    .. code-block:: python

        import harpy as hp

        sdata = hp.datasets.xenium_human_ovarian_cancer(
            subset=True,
            processed=True,
        )

        hp.tb.pl.histograms(
            sdata.tables["table_transcriptomics_preprocessed"],
            quantile_range=(0.1, 0.99),
        )
    """
    if len(metrics) == 0:
        raise ValueError("Parameter 'metrics' must contain at least one (dataframe, column) tuple.")

    if isinstance(bins, Sequence) and not isinstance(bins, str):
        bins_per_metric = list(bins)
        if len(bins_per_metric) != len(metrics):
            raise ValueError(
                f"Parameter 'bins' has length {len(bins_per_metric)}, but 'metrics' has length {len(metrics)}."
            )
    else:
        bins_per_metric = [bins] * len(metrics)

    display_columns_per_metric = _expand_per_metric_option(
        display_column,
        n_metrics=len(metrics),
        parameter_name="display_column",
    )
    colors_per_metric = _expand_per_metric_option(
        color,
        n_metrics=len(metrics),
        parameter_name="color",
    )

    if ax is None:
        nrows = int(np.ceil(len(metrics) / ncols))
        if figsize is None:
            figsize = (subplot_width * min(ncols, len(metrics)), subplot_height * nrows)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=sharex, sharey=sharey)
        axes = np.asarray(axes)
    else:
        axes = np.asarray(ax)
        fig = np.ravel(axes)[0].figure

    axes_flat = np.ravel(axes)
    if len(axes_flat) < len(metrics):
        raise ValueError(
            f"Received {len(axes_flat)} axes for {len(metrics)} metrics. Please provide enough axes or set 'ax=None'."
        )

    for axis, (dataframe, column), bins_value, display_column_value, color_value in zip(
        axes_flat[: len(metrics)],
        metrics,
        bins_per_metric,
        display_columns_per_metric,
        colors_per_metric,
        strict=True,
    ):
        histogram(
            adata,
            column,
            region=region,
            dataframe=dataframe,
            display_column=display_column_value,
            ax=axis,
            bins=bins_value,
            range=range,
            quantile_range=quantile_range,
            histplot_kwargs=histplot_kwargs,
            median_line_kwargs=median_line_kwargs,
            median_text_kwargs=median_text_kwargs,
            color=color_value,
            show_median=show_median,
            show_std=show_std,
        )

    for axis in axes_flat[len(metrics) :]:
        fig.delaxes(axis)

    if title is not None:
        fig.suptitle(title, weight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        fig.tight_layout()

    return axes


def _expand_per_metric_option[T](
    value: T | Sequence[T] | None,
    *,
    n_metrics: int,
    parameter_name: str,
) -> list[T | None]:
    if isinstance(value, Sequence) and not isinstance(value, str):
        values = list(value)
        if len(values) != n_metrics:
            raise ValueError(
                f"Parameter '{parameter_name}' has length {len(values)}, but 'metrics' has length {n_metrics}."
            )
        return values
    return [value] * n_metrics


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


def _check_var_selection(dataframe: Literal["obs", "var"], rows: np.ndarray | None) -> None:
    """Raise for ``.var`` columns with a subset of the regions: they describe the whole table."""
    if dataframe == "var" and rows is not None and not rows.all():
        raise ValueError(
            "A '.var' column describes the whole table, so it cannot be plotted for a subset of its regions without "
            "recomputing it. Plot a table that holds the subset's own '.var' columns, or use dataframe='obs'."
        )


def _format_display_name(column: str) -> str:
    return column.replace("_", " ").strip().title()


def _default_ylabel(dataframe: Literal["obs", "var"]) -> str:
    return "Number of instances" if dataframe == "obs" else "Number of features"
