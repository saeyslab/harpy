from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any, Literal

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from spatialdata import SpatialData

from harpy.qc._histogram import _format_display_name, _style_qc_axis
from harpy.qc.table._instance_measures import InstanceMeasure, _instance_values
from harpy.qc.table._summarize_table import TableSummary
from harpy.table._table import ProcessTable
from harpy.utils._keys import _CELLSIZE_KEY


def obs_scatter(
    sdata: SpatialData,
    table_name: str,
    labels_name: str | Iterable[str] | None = None,
    column_x: str = _CELLSIZE_KEY,
    column_y: str = "total_counts",
    ax: Axes | None = None,
    figsize: tuple[float, float] = (6, 4),
    title: str | None = None,
    display_column_x: str | None = None,
    display_column_y: str | None = None,
    cmap: str | None = None,
    histplot_kwargs: Mapping[str, Any] = MappingProxyType({}),
    show_regplot: bool = True,
    regplot_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> Axes:
    """
    Plot the relationship between two observation-level columns.

    Parameters
    ----------
    sdata
        :class:`~spatialdata.SpatialData` object containing the table.
    table_name
        Table element in ``sdata.tables``.
    labels_name
        Labels element or elements used to subset the selected table via :class:`~harpy.table._table.ProcessTable`.
        If ``None``, all observations in ``table_name`` are used.
    column_x
        Observation-level column in ``adata.obs`` to plot on the x-axis.
    column_y
        Observation-level column in ``adata.obs`` to plot on the y-axis.
    ax
        Matplotlib axes to draw on. If ``None``, a new figure and axes are created.
    figsize
        Figure size used when ``ax`` is ``None``.
    title
        Plot title. Defaults to ``"{x column} vs {y column}"``.
    display_column_x
        Display label for ``column_x``. If ``None``, a readable label is inferred from the column name.
    display_column_y
        Display label for ``column_y``. If ``None``, a readable label is inferred from the column name.
    cmap
        Colormap passed to :func:`seaborn.histplot`. If ``None``, seaborn's default is used.
    histplot_kwargs
        Keyword arguments passed to :func:`seaborn.histplot`.
    show_regplot
        Whether to overlay :func:`seaborn.regplot`. Enabled by default.
    regplot_kwargs
        Keyword arguments passed to :func:`seaborn.regplot` when ``show_regplot=True``.

    Returns
    -------
    :class:`matplotlib.axes.Axes` containing the relationship plot.

    Examples
    --------
    .. code-block:: python

        import harpy as hp

        sdata = hp.datasets.xenium_human_ovarian_cancer(
            subset=True,
            processed=True,
        )

        hp.qc.obs_scatter(
            sdata,
            table_name="table_transcriptomics_preprocessed",
            column_x="shapeSize",
            column_y="total_counts",
        )
    """
    process_table = ProcessTable(sdata, labels_name=labels_name, table_name=table_name)
    adata = sdata.tables[table_name]

    for obs_column in (column_x, column_y):
        if obs_column not in adata.obs.columns:
            raise ValueError(f"Column '{obs_column}' not found in 'adata.obs'.")
        if not pd.api.types.is_numeric_dtype(adata.obs[obs_column]):
            raise TypeError(f"Column '{obs_column}' in 'adata.obs' is not numeric and cannot be visualized.")

    values = adata.obs[[column_x, column_y]].copy()
    if process_table.labels_name is not None:
        obs_mask = adata.obs[process_table.region_key].isin(process_table.labels_name).to_numpy()
        values = values.loc[obs_mask]

    values = values.dropna()
    if values.empty:
        raise ValueError(
            f"Columns '{column_x}' and '{column_y}' in 'adata.obs' do not contain any paired non-null values."
        )

    return _plot_scatter(
        values[column_x],
        values[column_y],
        x_label=display_column_x if display_column_x is not None else _format_display_name(column_x),
        y_label=display_column_y if display_column_y is not None else _format_display_name(column_y),
        ax=ax,
        figsize=figsize,
        title=title,
        cmap=cmap,
        histplot_kwargs=histplot_kwargs,
        show_regplot=show_regplot,
        regplot_kwargs=regplot_kwargs,
    )


def instance_scatter(
    summary: TableSummary,
    *,
    x: tuple[str, InstanceMeasure],
    y: tuple[str, InstanceMeasure],
    normalization: Literal["per_panel_feature"] | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (6, 4),
    title: str | None = None,
    cmap: str | None = None,
    histplot_kwargs: Mapping[str, Any] = MappingProxyType({}),
    show_regplot: bool = True,
    regplot_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> Axes:
    """Plot two measures against each other across the retained instances of a table summary.

    Each retained instance contributes one point, including zeros. Instances
    were retained when any class read during summary computation had points.
    No table matrix is read or summary recalculated; nothing is written to the
    table.

    Parameters
    ----------
    summary
        Result of :func:`harpy.qc.summarize_table`. Uses ``summary.per_instance``.
    x, y
        Each axis as a pair ``(feature_class, measure)``: a class of
        ``summary.feature_classes``, for example ``"negative"``, and a measure
        as in :func:`harpy.qc.instance_histogram`: ``"points"``,
        ``"detected_features"`` or ``"pct_points"``.
    normalization
        ``None`` plots the measures as stored. ``"per_panel_feature"`` divides
        each axis by its own class's number of features,
        ``summary.panel_feature_counts``, so that classes with different panel
        sizes become comparable; it then requires both axes to use
        ``"points"``. ``summary`` is not modified.
    ax
        Matplotlib axes to draw on. If ``None``, a new figure and axes are created.
    figsize
        Figure size used when ``ax`` is ``None``.
    title
        Optional plot title. None leaves the axes title unchanged.
    cmap
        Colormap passed to :func:`seaborn.histplot`. If ``None``, seaborn's default is used.
    histplot_kwargs
        Keyword arguments passed to :func:`seaborn.histplot`.
    show_regplot
        Whether to overlay :func:`seaborn.regplot`. Enabled by default.
    regplot_kwargs
        Keyword arguments passed to :func:`seaborn.regplot` when ``show_regplot=True``.

    Returns
    -------
    :class:`matplotlib.axes.Axes` containing the relationship plot. Customize
    axis labels on the returned axes with ``ax.set(...)``.

    Raises
    ------
    TypeError
        If ``summary`` is not a :class:`~harpy.qc.TableSummary`.
    ValueError
        If ``x`` or ``y`` is not a ``(feature_class, measure)`` pair, names an
        unknown class or measure, or ``"pct_points"`` of a summary of one class;
        if ``normalization`` is unknown or an axis does not use ``"points"``
        with it; or if the summary retains no instances.

    See Also
    --------
    harpy.qc.summarize_table
    harpy.qc.instance_histogram
    harpy.qc.obs_scatter

    Examples
    --------
    .. code-block:: python

        summary = hp.qc.summarize_table(sdata, "table", obsm_keys=["negative", "system_control"])
        ax = hp.qc.instance_scatter(
            summary,
            x=("negative", "points"),
            y=("system_control", "points"),
            normalization="per_panel_feature",
        )
    """
    axes = {}
    for name, pair in (("x", x), ("y", y)):
        if not isinstance(pair, tuple) or len(pair) != 2 or not all(isinstance(part, str) for part in pair):
            raise ValueError(f"{name} must be a pair (feature_class, measure), found {pair!r}.")
        feature_class, measure = pair
        axes[name] = _instance_values(
            summary, feature_class=feature_class, measure=measure, normalization=normalization
        )
    (x_values, x_label), (y_values, y_label) = axes["x"], axes["y"]
    if x_values.empty:
        raise ValueError("The summary retains no instances to plot.")
    return _plot_scatter(
        x_values,
        y_values,
        x_label=x_label,
        y_label=y_label,
        ax=ax,
        figsize=figsize,
        title=title,
        cmap=cmap,
        histplot_kwargs=histplot_kwargs,
        show_regplot=show_regplot,
        regplot_kwargs=regplot_kwargs,
    )


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

    Shared by :func:`obs_scatter` and :func:`instance_scatter`: callers select
    the values and remove missing ones. No source data or summary is accessed
    here, and the option mappings are not modified.
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
