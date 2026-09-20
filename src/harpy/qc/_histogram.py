"""Table and spatial-bin histograms with a shared numerical renderer."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger as log
from matplotlib.axes import Axes
from spatialdata import SpatialData

from harpy.qc._summarize_points import PointsSummary
from harpy.table._table import ProcessTable

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

_WARNED_DEPRECATED_ATTRIBUTES: set[str] = set()


def __getattr__(name: str) -> object:
    aliases = {"metric_histogram": table_histogram, "metrics_histogram": table_histograms}
    if name in aliases:
        function = aliases[name]
        if name not in _WARNED_DEPRECATED_ATTRIBUTES:
            _WARNED_DEPRECATED_ATTRIBUTES.add(name)
            log.warning(f"`harpy.qc.{name}` is deprecated. Import and use `harpy.qc.{function.__name__}` instead.")
        return function
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def spatial_bin_histogram(
    summary: PointsSummary,
    *,
    feature_class: str,
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
    """Plot one feature class's point counts across retained spatial bins.

    Each retained bin contributes one value, including zeros for the selected
    class. Bins were retained when any class selected during summary computation
    had points; choosing a class here never changes that population. Bin
    inclusion is based on detected points, not a tissue annotation. No source
    points are read or summaries recalculated.

    Parameters
    ----------
    summary
        Result of :func:`harpy.qc.summarize_points` computed with ``bin_size``. Uses
        ``spatial_bins.per_bin`` for counts, ``spatial_bins.per_class`` for
        median/SD annotations. Source and geometry context remain available
        in ``metadata`` for caller-supplied titles or figure captions.
    feature_class
        Exact class name present in the summary; no classes are pooled.
    ax
        Axes to reuse, or None to create a figure.
    bins
        Histogram intervals along the point-count axis, not spatial bin size.
    range
        Inclusive display bounds. Values outside them are hidden, but still
        contribute to the percentage denominator and median/SD annotations.
    quantile_range
        Quantiles defining display bounds when ``range`` is None; for example
        ``(0.1, 0.99)``. Filtering affects only the display.
    histplot_kwargs
        Seaborn histogram options, including ``kde``, ``alpha``, and ``color``.
        Defaults to a borderless filled-step histogram with ``alpha=0.5``,
        ``stat="count"`` (number of spatial bins), and a KDE of line width 2.
        ``stat="percent"`` uses all retained bins as the denominator, so
        clipped bars may sum to less than 100%. Other statistics, data selection,
        grouping, and weights cannot be supplied here. KDE is omitted for
        constant or insufficient displayed data. Counts are not area-normalized.
    median_line_kwargs, median_text_kwargs
        Matplotlib styling for the median line and annotation box.
    figsize
        Figure size when creating axes.
    title
        Optional plot title. None leaves the axes title unchanged; no title
        is generated automatically.
    color
        Histogram color, unless overridden by ``histplot_kwargs``.
    show_median
        Draw the full retained-population median and annotation.
    show_std
        Include sample SD in the median annotation; fewer than two bins shows N/A.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the histogram, or an empty-state message when no bins
        are retained. Input data and plotting-option mappings remain unchanged.
        Customize axis labels on the returned axes with ``ax.set(...)``.

    See Also
    --------
    harpy.qc.summarize_points
    harpy.qc.table_histogram

    Examples
    --------
    .. code-block:: python

        summary = hp.qc.summarize_points(sdata, "transcripts", bin_size=100)
        ax = hp.qc.spatial_bin_histogram(
            summary, feature_class="Negative", quantile_range=(0.1, 0.99)
        )
    """
    if not isinstance(summary, PointsSummary):
        raise TypeError("summary must be a PointsSummary returned by hp.qc.summarize_points().")
    spatial_bins = summary.spatial_bins
    if spatial_bins is None:
        raise ValueError("Spatial-bin measurements are missing; call hp.qc.summarize_points() with bin_size.")
    statistics = spatial_bins.per_class.loc[spatial_bins.per_class["feature_class"] == feature_class]
    if statistics.empty:
        raise ValueError(f"Feature class {feature_class!r} is absent from the spatial-bin summary.")
    if "std_points_per_bin" not in statistics:
        raise ValueError("Spatial-bin SD is missing; recompute the summary with hp.qc.summarize_points().")
    if histplot_kwargs.get("stat", "count") not in {"count", "percent"}:
        raise ValueError("Spatial-bin histograms support stat='count' or stat='percent'.")
    values = spatial_bins.per_bin.loc[spatial_bins.per_bin["feature_class"] == feature_class, "n_points"]
    row = statistics.iloc[0]
    return _plot_histogram(
        values,
        median=float(row["median_points_per_bin"]),
        std=float(row["std_points_per_bin"]),
        ax=ax,
        bins=bins,
        range=range,
        quantile_range=quantile_range,
        histplot_kwargs=histplot_kwargs,
        median_line_kwargs=median_line_kwargs,
        median_text_kwargs=median_text_kwargs,
        figsize=figsize,
        title=title,
        color=color if color is not None else "#4C78A8",
        show_median=show_median,
        show_std=show_std,
        xlabel=f"{feature_class} points per spatial bin",
        count_ylabel="Number of spatial bins",
        percent_ylabel="Percentage of spatial bins (%)",
    )


def table_histogram(
    sdata: SpatialData,
    table_name: str,
    labels_name: str | Iterable[str] | None = None,
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
) -> Axes:
    """
    Plot a QC metric histogram for an :class:`~anndata.AnnData` table.

    This function is read-only and expects QC metrics to already be present on the selected table,
    typically after running :func:`scanpy.pp.calculate_qc_metrics` during preprocessing.

    Parameters
    ----------
    sdata
        :class:`~spatialdata.SpatialData` object containing the table.
    table_name
        Table element in ``sdata.tables``.
    labels_name
        Labels element or elements used to subset the selected table via :class:`~harpy.table._table.ProcessTable`.
        If ``None``, all observations in ``table_name`` are used.
    column
        QC metric column to plot. The column is searched in ``.obs`` and/or ``.var`` depending on ``dataframe``.
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
        metric population, before display filtering, as its denominator.
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

    Examples
    --------
    .. code-block:: python

        import harpy as hp

        sdata = hp.datasets.xenium_human_ovarian_cancer(
            subset=True,
            processed=True,
        )

        hp.qc.table_histogram(
            sdata,
            table_name="table_transcriptomics_preprocessed",
            labels_name="nucleus_segmentation_mask",
            column="total_counts",
            dataframe="obs",
            quantile_range=(0.1, 0.99),
        )
    """
    process_table = ProcessTable(sdata, labels_name=labels_name, table_name=table_name)
    adata = sdata.tables[table_name]
    obs_mask = None
    if process_table.labels_name is not None:
        obs_mask = adata.obs[process_table.region_key].isin(process_table.labels_name).to_numpy()

    resolved_dataframe = _resolve_dataframe(adata, column=column, dataframe=dataframe)
    if resolved_dataframe == "var" and obs_mask is not None and not obs_mask.all():
        raise ValueError(
            "Plotting '.var' QC metrics for a subset of 'labels_name' is not supported without recomputing QC metrics. "
            "Please plot a table element that already contains the desired subset-specific QC metrics, or use dataframe='obs'."
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
        percent_ylabel="Percentage of cells (%)" if resolved_dataframe == "obs" else "Percentage of genes (%)",
    )


def table_histograms(
    sdata: SpatialData,
    table_name: str,
    labels_name: str | Iterable[str] | None = None,
    metrics: Sequence[tuple[Literal["obs", "var"], str]] = (
        ("obs", "total_counts"),
        ("obs", "n_genes_by_counts"),
        ("var", "total_counts"),
        ("var", "log1p_total_counts"),
        ("var", "n_cells_by_counts"),
        ("var", "mean_counts"),
    ),
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
    Plot a standard panel of QC metric histograms for an :class:`~anndata.AnnData` table.

    This function is read-only and expects QC metrics to already be present on the selected table,
    typically after running :func:`scanpy.pp.calculate_qc_metrics` during preprocessing.

    Parameters
    ----------
    sdata
        :class:`~spatialdata.SpatialData` object containing the table.
    table_name
        Table element in ``sdata.tables``.
    labels_name
        Labels element or elements used to subset the selected table via :class:`~harpy.table._table.ProcessTable`.
    metrics
        Sequence of ``(dataframe, column)`` tuples to plot. Defaults to a standard transcript QC panel obtained through :func:`scanpy.pp.calculate_qc_metrics`.
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
        Plotting options forwarded through :func:`table_histogram`.
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

    Examples
    --------
    .. code-block:: python

        import harpy as hp

        sdata = hp.datasets.xenium_human_ovarian_cancer(
            subset=True,
            processed=True,
        )

        hp.qc.table_histograms(
            sdata,
            table_name="table_transcriptomics_preprocessed",
            labels_name="nucleus_segmentation_mask",
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
        table_histogram(
            sdata=sdata,
            table_name=table_name,
            labels_name=labels_name,
            column=column,
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
        ax.text(0.5, 0.5, "No retained spatial bins", ha="center", va="center", transform=ax.transAxes)
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


def _format_display_name(column: str) -> str:
    return column.replace("_", " ").strip().title()


def _default_ylabel(dataframe: Literal["obs", "var"]) -> str:
    return "Number of cells" if dataframe == "obs" else "Number of genes"
