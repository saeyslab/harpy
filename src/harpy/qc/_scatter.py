from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from spatialdata import SpatialData

from harpy.qc._histogram import _format_display_name, _style_qc_axis
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

    x_label = display_column_x if display_column_x is not None else _format_display_name(column_x)
    y_label = display_column_y if display_column_y is not None else _format_display_name(column_y)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    histplot_kwargs = dict(histplot_kwargs)
    histplot_kwargs.setdefault("bins", 60)
    histplot_kwargs.setdefault("cbar", True)
    histplot_kwargs.setdefault("pmax", 0.98)
    if cmap is not None:
        histplot_kwargs.setdefault("cmap", cmap)
    sns.histplot(x=values[column_x], y=values[column_y], ax=ax, **histplot_kwargs)

    if show_regplot:
        regplot_kwargs = dict(regplot_kwargs)
        regplot_kwargs.setdefault("scatter", False)
        regplot_kwargs.setdefault("lowess", True)
        regplot_kwargs.setdefault("line_kws", {"color": "#1F3B4D", "lw": 1.5})
        sns.regplot(x=values[column_x], y=values[column_y], ax=ax, **regplot_kwargs)

    if title is not None:
        ax.set_title(title, weight="bold")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    _style_qc_axis(ax)
    return ax
