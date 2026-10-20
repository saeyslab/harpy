"""The instance scatter of a table summary, and the deprecated table scatter of ``hp.qc``."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any, Literal

from matplotlib.axes import Axes
from spatialdata import SpatialData

from harpy.plot._renderers import _plot_scatter
from harpy.qc.table._instance_measures import InstanceMeasure, _instance_values
from harpy.qc.table._summarize_table import TableSummary
from harpy.table.pl._scatter import scatter
from harpy.utils._deprecation import _deprecated_function
from harpy.utils._keys import _CELLSIZE_KEY


def _obs_scatter(
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

    .. warning::
       Deprecated, and will be removed in a future release. Use :func:`harpy.tb.pl.scatter` on
       ``sdata.tables[table_name]``, with ``region`` for ``labels_name``; it takes the other arguments of this
       function.
    """
    return scatter(
        sdata.tables[table_name],
        column_x,
        column_y,
        region=labels_name if labels_name is None or isinstance(labels_name, str) else list(labels_name),
        dataframe="obs",
        ax=ax,
        figsize=figsize,
        title=title,
        display_column_x=display_column_x,
        display_column_y=display_column_y,
        cmap=cmap,
        histplot_kwargs=histplot_kwargs,
        show_regplot=show_regplot,
        regplot_kwargs=regplot_kwargs,
    )


obs_scatter = _deprecated_function(
    _obs_scatter,
    "harpy.qc.obs_scatter",
    use="harpy.tb.pl.scatter on sdata.tables[table_name], with region for labels_name",
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
    """Plot two measures against each other across the instances of a table summary.

    Each instance contributes one point, including zeros. With
    ``"pct_points"`` on an axis, instances without points in any class read
    have no share and are skipped. No table matrix is read or summary
    recalculated; nothing is written to the table.

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
        with it; or if no instance is left to plot.

    See Also
    --------
    harpy.qc.summarize_table
    harpy.qc.instance_histogram
    harpy.tb.pl.scatter

    Examples
    --------
    .. code-block:: python

        summary = hp.qc.summarize_table(sdata, "table", obsm_keys=["negative", "system_control"])
        # Complexity: detected genes (y) against gene points (x), scanpy's n_genes_by_counts and total_counts.
        ax = hp.qc.instance_scatter(summary, x=("x", "points"), y=("x", "detected_features"))

        # Background-dominated instances: a high share of negative-probe points (y) at few gene points (x),
        # scanpy's pct_counts_mt and total_counts.
        ax = hp.qc.instance_scatter(summary, x=("x", "points"), y=("negative", "pct_points"))
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
    if "pct_points" in (x[1], y[1]):
        # A point needs both values: drop an instance whose pct_points (share) is NaN on either axis, as it has
        # no points in any class read.
        complete = x_values.notna() & y_values.notna()
        x_values, y_values = x_values[complete], y_values[complete]
    if x_values.empty:
        raise ValueError("No instances to plot.")
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
