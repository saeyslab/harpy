"""Spatial-bin and instance histograms, and the deprecated table histograms of ``hp.qc``."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from spatialdata import SpatialData

from harpy.plot._renderers import _plot_histogram
from harpy.qc._summary_schema import _FEATURE_KEY, _N_POINTS_KEY
from harpy.qc._summary_statistics import _points_statistics
from harpy.qc.points._spatial_bin_measures import SpatialBinMeasure, _spatial_bin_values
from harpy.qc.points._summarize_points import PointsSummary
from harpy.qc.points._summarize_points_by_feature import FeaturePointsSummary
from harpy.qc.table._instance_measures import InstanceMeasure, _instance_values
from harpy.qc.table._summarize_table import TableSummary
from harpy.qc.table._summarize_table_by_feature import FeatureTableSummary
from harpy.table.pl._histogram import histogram, histograms
from harpy.utils._deprecation import _deprecated_function


def spatial_bin_histogram(
    summary: PointsSummary,
    *,
    feature_class: str,
    measure: SpatialBinMeasure = "points",
    normalization: Literal["per_panel_feature"] | None = None,
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
    """Plot one feature class's measure across retained spatial bins.

    Each retained bin contributes one value, including zeros for the selected
    class. Bins were retained when any class selected during summary computation
    had points; choosing one here never changes that population. Bin
    inclusion is based on detected points, not a tissue annotation. No source
    points are read or summaries recalculated.

    Parameters
    ----------
    summary
        Result of :func:`harpy.qc.summarize_points`, computed with ``bin_size``.
        Uses ``summary.spatial_bins.per_bin``. Source and geometry context
        remain available in ``metadata`` for caller-supplied titles or figure
        captions.
    feature_class
        Required exact class name present in the summary; classes are not pooled.
    measure
        What to plot for the class, per spatial bin: ``"points"``, its points;
        or ``"pct_points"``, its share of the bin's points across the selected
        classes, in percent, with two selected classes or more. They read the
        columns ``n_points`` and ``pct_points`` of
        ``summary.spatial_bins.per_bin``.
        The share is not corrected for panel size: a class with more features
        collects more points, also from background. Compare it across bins or
        samples of the same panel, or compare classes with
        ``normalization="per_panel_feature"`` on ``"points"``.
        Detected features per bin, a measure of
        :func:`harpy.qc.instance_histogram`, are not computed.
    normalization
        ``None`` plots the measure as stored. ``"per_panel_feature"`` divides
        each bin's count, and the median and SD annotated on the figure, by the
        class's number of panel features, features without detections
        included, as :func:`harpy.pl.plot_points_density` does. It applies to
        ``measure="points"`` only. Only the displayed values are divided;
        ``summary`` is not modified. The result is the class's points per panel
        feature per bin, comparable across classes with different panel sizes.
        ``range`` is then given in that unit: for a class with 10 panel
        features, the view ``range=(0, 50)`` of raw counts becomes
        ``range=(0, 5)``.
    ax
        Axes to reuse, or None to create a figure.
    bins
        Histogram intervals along the measure's axis, not spatial bin size.
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

    Raises
    ------
    TypeError
        If ``summary`` is not a :class:`~harpy.qc.PointsSummary`.
    ValueError
        If the summary was computed without ``bin_size``; if
        ``feature_class`` or ``measure`` is unknown, ``"detected_features"``
        is requested, or ``"pct_points"`` from a summary of one class; or if
        ``normalization`` is unknown or applied to a measure other than
        ``"points"``.

    Notes
    -----
    The median and SD are computed from the plotted values, as
    ``"pct_points"`` has no stored statistics. For ``"points"``, they equal
    the class's ``median_points_per_bin`` and ``std_points_per_bin`` in
    ``summary.spatial_bins.per_class``.

    See Also
    --------
    harpy.qc.summarize_points
    harpy.qc.spatial_bin_histogram_by_feature
    harpy.qc.instance_histogram
    harpy.tb.pl.histogram

    Examples
    --------
    .. code-block:: python

        summary = hp.qc.summarize_points(sdata, "transcripts", bin_size=100)
        ax = hp.qc.spatial_bin_histogram(
            summary, feature_class="Negative", quantile_range=(0.1, 0.99)
        )

        # Points per negative probe per bin, comparable with other classes.
        ax = hp.qc.spatial_bin_histogram(
            summary, feature_class="Negative", normalization="per_panel_feature"
        )

        # The share of each bin's points that are negative probes.
        ax = hp.qc.spatial_bin_histogram(summary, feature_class="Negative", measure="pct_points")
    """
    values, label = _spatial_bin_values(
        summary, feature_class=feature_class, measure=measure, normalization=normalization
    )
    return _plot_spatial_bin_histogram(
        values,
        xlabel=label,
        ax=ax,
        bins=bins,
        range=range,
        quantile_range=quantile_range,
        histplot_kwargs=histplot_kwargs,
        median_line_kwargs=median_line_kwargs,
        median_text_kwargs=median_text_kwargs,
        figsize=figsize,
        title=title,
        color=color,
        show_median=show_median,
        show_std=show_std,
    )


def spatial_bin_histogram_by_feature(
    summary: FeaturePointsSummary,
    *,
    feature: str,
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
    """Plot one feature's point counts across retained spatial bins.

    Each retained bin contributes one value, including zeros for the selected
    feature. Bins were retained when any feature requested during summary
    computation had points; choosing one here never changes that population.
    Bin inclusion is based on detected points, not a tissue annotation.
    No source points are read or summaries recalculated.

    Parameters
    ----------
    summary
        Result of :func:`harpy.qc.summarize_points_by_feature`, computed from a
        binned :class:`~harpy.qc.PointsSummary`. Uses
        ``summary.spatial_bins.per_bin``. Source and geometry context remain
        available in ``metadata`` for caller-supplied titles or figure captions.
    feature
        Required exact feature name present in the summary; features are not pooled.
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

    Notes
    -----
    The median and SD are computed from the plotted values, as in
    :func:`harpy.qc.spatial_bin_histogram`. They equal the feature's
    ``median_points_per_bin`` and ``std_points_per_bin`` in
    ``summary.spatial_bins.per_feature``.

    See Also
    --------
    harpy.qc.summarize_points_by_feature
    harpy.qc.spatial_bin_histogram
    harpy.tb.pl.histogram

    Examples
    --------
    .. code-block:: python

        summary = hp.qc.summarize_points(sdata, "transcripts", bin_size=100)
        features = hp.qc.summarize_points_by_feature(sdata, summary=summary, features=["EPCAM", "VIM"])
        ax = hp.qc.spatial_bin_histogram_by_feature(features, feature="EPCAM")
    """
    if not isinstance(summary, FeaturePointsSummary):
        raise TypeError("summary must be a FeaturePointsSummary; use spatial_bin_histogram() for a PointsSummary.")
    spatial_bins = summary.spatial_bins
    if spatial_bins is None:
        raise ValueError(
            "Spatial-bin measurements are missing; call hp.qc.summarize_points() with bin_size, "
            "then pass that summary to hp.qc.summarize_points_by_feature()."
        )
    # per_feature has a row for every requested feature, also when no bin is retained and per_bin is empty.
    features = [str(name) for name in spatial_bins.per_feature[_FEATURE_KEY]]
    if feature not in features:
        raise ValueError(f"Unknown feature {feature!r}; choose from {features}.")
    per_bin = spatial_bins.per_bin
    return _plot_spatial_bin_histogram(
        per_bin.loc[per_bin[_FEATURE_KEY] == feature, _N_POINTS_KEY],
        xlabel=f"{feature} points per spatial bin",
        ax=ax,
        bins=bins,
        range=range,
        quantile_range=quantile_range,
        histplot_kwargs=histplot_kwargs,
        median_line_kwargs=median_line_kwargs,
        median_text_kwargs=median_text_kwargs,
        figsize=figsize,
        title=title,
        color=color,
        show_median=show_median,
        show_std=show_std,
    )


def _plot_spatial_bin_histogram(
    values: pd.Series,
    *,
    xlabel: str,
    ax: Axes | None,
    bins: int | str,
    range: tuple[float, float] | None,
    quantile_range: tuple[float, float] | None,
    histplot_kwargs: Mapping[str, Any],
    median_line_kwargs: Mapping[str, Any],
    median_text_kwargs: Mapping[str, Any],
    figsize: tuple[float, float],
    title: str | None,
    color: str | None,
    show_median: bool,
    show_std: bool,
) -> Axes:
    """Annotate the median and SD of one measure over the retained bins, then share the histogram renderer.

    Shared by :func:`spatial_bin_histogram` and :func:`spatial_bin_histogram_by_feature`,
    which select the values.
    """
    if histplot_kwargs.get("stat", "count") not in {"count", "percent"}:
        raise ValueError("Spatial-bin histograms support stat='count' or stat='percent'.")
    # The shared statistics helper keeps the conventions of the summaries: SD with ddof=1, NaN below two values.
    statistics = _points_statistics(values.to_numpy(dtype=np.float64), unit="bin")
    return _plot_histogram(
        values,
        median=statistics["median_points_per_bin"],
        std=statistics["std_points_per_bin"],
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
        xlabel=xlabel,
        count_ylabel="Number of spatial bins",
        percent_ylabel="Percentage of spatial bins (%)",
    )


def instance_histogram(
    summary: TableSummary,
    *,
    feature_class: str,
    measure: InstanceMeasure = "points",
    normalization: Literal["per_panel_feature"] | None = None,
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
    """Plot one class's measure across the retained instances of a table summary.

    Each retained instance contributes one value, including zeros for the
    selected class. Instances were retained when any class read during summary
    computation had points; choosing one here never changes that population.
    No table matrix is read or summary recalculated; nothing is written to the
    table.

    Parameters
    ----------
    summary
        Result of :func:`harpy.qc.summarize_table`. Uses
        ``summary.per_instance``; source context remains available in
        ``metadata`` for caller-supplied titles or figure captions.
    feature_class
        A class of the summary, one of ``summary.feature_classes``: the label
        of a matrix read, for example ``"x"`` or ``"negative"``.
    measure
        What to plot for the class, per instance: ``"points"``, its points;
        ``"detected_features"``, its features with at least one point; or
        ``"pct_points"``, its share of the instance's points across the classes
        read, in percent, with two classes or more. They read the columns
        ``n_<class>_points``, ``n_<class>_detected_features`` and
        ``pct_<class>_points`` of ``summary.per_instance``. The share is not
        corrected for panel size: a class with more features collects more
        points, also from background. Compare it across instances or samples
        of the same panel, or compare classes with
        ``normalization="per_panel_feature"`` on ``"points"``.
    normalization
        ``None`` plots the measure as stored. ``"per_panel_feature"`` divides
        each instance's points, and the annotated median and SD, by the class's
        number of features, ``summary.panel_feature_counts``, as
        :func:`harpy.qc.spatial_bin_histogram` does, so that classes with
        different panel sizes become comparable. It applies to
        ``measure="points"`` only. Only the displayed values are divided;
        ``summary`` is not modified.
    ax
        Axes to reuse, or None to create a figure.
    bins
        Histogram intervals along the measure's axis.
    range
        Inclusive display bounds. Values outside them are hidden, but still
        contribute to the percentage denominator and median/SD annotations.
    quantile_range
        Quantiles defining display bounds when ``range`` is None; for example
        ``(0.1, 0.99)``. Filtering affects only the display.
    histplot_kwargs
        Seaborn histogram options, including ``kde``, ``alpha``, and ``color``.
        Defaults to a borderless filled-step histogram with ``alpha=0.5``,
        ``stat="count"`` (number of instances), and a KDE of line width 2.
        ``stat="percent"`` uses all retained instances as the denominator, so
        clipped bars may sum to less than 100%. Other statistics, data
        selection, grouping, and weights cannot be supplied here.
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
        Draw the median of the plotted values and its annotation.
    show_std
        Include the sample SD in the median annotation; fewer than two
        instances shows N/A.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the histogram, or an empty-state message when no
        instances are retained. Input data and plotting-option mappings remain
        unchanged. Customize axis labels on the returned axes with ``ax.set(...)``.

    Raises
    ------
    TypeError
        If ``summary`` is not a :class:`~harpy.qc.TableSummary`.
    ValueError
        If ``feature_class`` or ``measure`` is unknown, ``"pct_points"`` is
        requested from a summary of one class, or ``normalization`` is unknown
        or applied to a measure other than ``"points"``.

    Notes
    -----
    The median and SD are computed from the plotted values, as some measures,
    such as ``"pct_points"``, have no stored statistics. For ``"points"`` and
    ``"detected_features"``, they equal the class's ``median_…_per_instance``
    and ``std_…_per_instance`` in ``summary.per_class``.

    See Also
    --------
    harpy.qc.summarize_table
    harpy.qc.instance_scatter
    harpy.qc.spatial_bin_histogram

    Examples
    --------
    .. code-block:: python

        summary = hp.qc.summarize_table(sdata, "table", obsm_keys=["negative", "system_control"])
        ax = hp.qc.instance_histogram(summary, feature_class="x", measure="pct_points")

        # Negative points per negative probe per instance, comparable with other classes.
        ax = hp.qc.instance_histogram(summary, feature_class="negative", normalization="per_panel_feature")
    """
    values, label = _instance_values(summary, feature_class=feature_class, measure=measure, normalization=normalization)
    return _plot_instance_histogram(
        values,
        xlabel=label,
        ax=ax,
        bins=bins,
        range=range,
        quantile_range=quantile_range,
        histplot_kwargs=histplot_kwargs,
        median_line_kwargs=median_line_kwargs,
        median_text_kwargs=median_text_kwargs,
        figsize=figsize,
        title=title,
        color=color,
        show_median=show_median,
        show_std=show_std,
    )


def instance_histogram_by_feature(
    summary: FeatureTableSummary,
    *,
    feature: str,
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
    """Plot one feature's points across the retained instances of a feature table summary.

    Each retained instance of the reference table summary contributes one
    value, including zeros for the selected feature: choosing a feature never
    changes that population. No table matrix is read or summary recalculated.

    Parameters
    ----------
    summary
        Result of :func:`harpy.qc.summarize_table_by_feature`. Uses
        ``summary.per_instance``.
    feature
        The name of one of the features passed to
        :func:`harpy.qc.summarize_table_by_feature`, listed in
        ``summary.per_feature["feature"]``; features are not pooled.
    ax
        Axes to reuse, or None to create a figure.
    bins
        Histogram intervals along the points axis.
    range
        Inclusive display bounds. Values outside them are hidden, but still
        contribute to the percentage denominator and median/SD annotations.
    quantile_range
        Quantiles defining display bounds when ``range`` is None; for example
        ``(0.1, 0.99)``. Filtering affects only the display.
    histplot_kwargs
        Seaborn histogram options, including ``kde``, ``alpha``, and ``color``.
        Defaults to a borderless filled-step histogram with ``alpha=0.5``,
        ``stat="count"`` (number of instances), and a KDE of line width 2.
        ``stat="percent"`` uses all retained instances as the denominator, so
        clipped bars may sum to less than 100%. Other statistics, data
        selection, grouping, and weights cannot be supplied here.
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
        Draw the median of the plotted values and its annotation.
    show_std
        Include the sample SD in the median annotation; fewer than two
        instances shows N/A.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the histogram, or an empty-state message when no
        instances are retained. Input data and plotting-option mappings remain
        unchanged. Customize axis labels on the returned axes with ``ax.set(...)``.

    Raises
    ------
    TypeError
        If ``summary`` is not a :class:`~harpy.qc.FeatureTableSummary`.
    ValueError
        If ``feature`` is not one of the summary's requested features.

    Notes
    -----
    The median and SD are computed from the plotted values, as in
    :func:`harpy.qc.instance_histogram`. They equal the feature's
    ``median_points_per_instance`` and ``std_points_per_instance`` in
    ``summary.per_feature``.

    See Also
    --------
    harpy.qc.summarize_table_by_feature
    harpy.qc.instance_histogram
    harpy.qc.spatial_bin_histogram_by_feature

    Examples
    --------
    .. code-block:: python

        summary = hp.qc.summarize_table(sdata, "table", obsm_keys=["negative"])
        features = hp.qc.summarize_table_by_feature(sdata, summary=summary, features=["Negative01", "GeneA"])
        ax = hp.qc.instance_histogram_by_feature(features, feature="Negative01")
    """
    if not isinstance(summary, FeatureTableSummary):
        raise TypeError("summary must be a FeatureTableSummary; use instance_histogram() for a TableSummary.")
    features = [str(name) for name in summary.per_feature[_FEATURE_KEY]]
    if feature not in features:
        raise ValueError(f"Unknown feature {feature!r}; choose from {features}.")
    per_instance = summary.per_instance
    return _plot_instance_histogram(
        per_instance.loc[per_instance[_FEATURE_KEY] == feature, _N_POINTS_KEY],
        xlabel=f"{feature} points per instance",
        ax=ax,
        bins=bins,
        range=range,
        quantile_range=quantile_range,
        histplot_kwargs=histplot_kwargs,
        median_line_kwargs=median_line_kwargs,
        median_text_kwargs=median_text_kwargs,
        figsize=figsize,
        title=title,
        color=color,
        show_median=show_median,
        show_std=show_std,
    )


def _plot_instance_histogram(
    values: pd.Series,
    *,
    xlabel: str,
    ax: Axes | None,
    bins: int | str,
    range: tuple[float, float] | None,
    quantile_range: tuple[float, float] | None,
    histplot_kwargs: Mapping[str, Any],
    median_line_kwargs: Mapping[str, Any],
    median_text_kwargs: Mapping[str, Any],
    figsize: tuple[float, float],
    title: str | None,
    color: str | None,
    show_median: bool,
    show_std: bool,
) -> Axes:
    """Annotate the median and SD of one measure over the retained instances, then share the histogram renderer.

    Shared by :func:`instance_histogram` and :func:`instance_histogram_by_feature`,
    which select the values.
    """
    if histplot_kwargs.get("stat", "count") not in {"count", "percent"}:
        raise ValueError("Instance histograms support stat='count' or stat='percent'.")
    # The shared statistics helper keeps the conventions of the summaries: SD with ddof=1, NaN below two values.
    statistics = _points_statistics(values.to_numpy(dtype=np.float64), unit="instance")
    return _plot_histogram(
        values,
        median=statistics["median_points_per_instance"],
        std=statistics["std_points_per_instance"],
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
        xlabel=xlabel,
        count_ylabel="Number of instances",
        percent_ylabel="Percentage of instances (%)",
        empty_message="No retained instances",
    )


def _table_histogram(
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

    .. warning::
       Deprecated, and will be removed in a future release. Use :func:`harpy.tb.pl.histogram` on
       ``sdata.tables[table_name]``, with ``region`` for ``labels_name``; it takes the other arguments of this
       function.
    """
    return histogram(
        sdata.tables[table_name],
        column,
        region=_region(labels_name),
        display_column=display_column,
        dataframe=dataframe,
        ax=ax,
        bins=bins,
        range=range,
        quantile_range=quantile_range,
        histplot_kwargs=histplot_kwargs,
        median_line_kwargs=median_line_kwargs,
        median_text_kwargs=median_text_kwargs,
        figsize=figsize,
        title=title,
        color=color,
        show_median=show_median,
        show_std=show_std,
    )


def _table_histograms(
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

    .. warning::
       Deprecated, and will be removed in a future release. Use :func:`harpy.tb.pl.histograms` on
       ``sdata.tables[table_name]``, with ``region`` for ``labels_name``; it takes the other arguments of this
       function.
    """
    return histograms(
        sdata.tables[table_name],
        metrics,
        region=_region(labels_name),
        ax=ax,
        bins=bins,
        range=range,
        quantile_range=quantile_range,
        histplot_kwargs=histplot_kwargs,
        median_line_kwargs=median_line_kwargs,
        median_text_kwargs=median_text_kwargs,
        figsize=figsize,
        ncols=ncols,
        subplot_width=subplot_width,
        subplot_height=subplot_height,
        sharex=sharex,
        sharey=sharey,
        title=title,
        display_column=display_column,
        color=color,
        show_median=show_median,
        show_std=show_std,
    )


def _region(labels_name: str | Iterable[str] | None) -> str | list[str] | None:
    """Return the labels_name of the deprecated plots as a region of :mod:`harpy.tb.pl`."""
    return labels_name if labels_name is None or isinstance(labels_name, str) else list(labels_name)


_TABLE_HISTOGRAM_USE = "harpy.tb.pl.histogram on sdata.tables[table_name], with region for labels_name"
_TABLE_HISTOGRAMS_USE = "harpy.tb.pl.histograms on sdata.tables[table_name], with region for labels_name"
table_histogram = _deprecated_function(_table_histogram, "harpy.qc.table_histogram", use=_TABLE_HISTOGRAM_USE)
table_histograms = _deprecated_function(_table_histograms, "harpy.qc.table_histograms", use=_TABLE_HISTOGRAMS_USE)
metric_histogram = _deprecated_function(_table_histogram, "harpy.qc.metric_histogram", use=_TABLE_HISTOGRAM_USE)
metrics_histogram = _deprecated_function(_table_histograms, "harpy.qc.metrics_histogram", use=_TABLE_HISTOGRAMS_USE)
