"""The relationship between two numeric ``.obs`` columns of an AnnData table, as a 2D histogram."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

import pandas as pd
from anndata import AnnData
from matplotlib.axes import Axes

from harpy.plot._renderers import _plot_scatter
from harpy.table._utils import _region_rows
from harpy.table.pl._histogram import _format_display_name
from harpy.utils._keys import _CELLSIZE_KEY


def scatter(
    adata: AnnData,
    column_x: str = _CELLSIZE_KEY,
    column_y: str = "total_counts",
    *,
    region: str | Sequence[str] | None = None,
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
    Plot the relationship between two numeric ``.obs`` columns of a table.

    A 2D histogram of the rows, with a lowess line, unlike the dot plot of
    :func:`scanpy.pl.scatter`. The table is not modified.

    Parameters
    ----------
    adata
        The table, for example ``sdata.tables["table"]``.
    column_x
        Column in ``adata.obs`` to plot on the x-axis.
    column_y
        Column in ``adata.obs`` to plot on the y-axis.
    region
        ``None``, the default, plots every row. Otherwise, one or more regions
        of the table's SpatialData annotation, ``uns["spatialdata_attrs"]["region"]``,
        whose rows are plotted, whatever the type of the annotated elements.
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

    See Also
    --------
    harpy.tb.pl.histogram
    harpy.qc.instance_scatter

    Examples
    --------
    .. code-block:: python

        import harpy as hp

        sdata = hp.datasets.xenium_human_ovarian_cancer(
            subset=True,
            processed=True,
        )

        hp.tb.pl.scatter(
            sdata.tables["table_transcriptomics_preprocessed"],
            column_x="shapeSize",
            column_y="total_counts",
        )
    """
    rows = _region_rows(adata, region)
    for obs_column in (column_x, column_y):
        if obs_column not in adata.obs.columns:
            raise ValueError(f"Column '{obs_column}' not found in 'adata.obs'.")
        if not pd.api.types.is_numeric_dtype(adata.obs[obs_column]):
            raise TypeError(f"Column '{obs_column}' in 'adata.obs' is not numeric and cannot be visualized.")

    values = adata.obs[[column_x, column_y]].copy()
    if rows is not None:
        values = values.loc[rows]

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
