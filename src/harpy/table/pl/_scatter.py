"""The relationship between two numeric ``.obs`` or ``.var`` columns of an AnnData table, as a 2D histogram."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Literal

import pandas as pd
from anndata import AnnData
from matplotlib.axes import Axes

from harpy.plot._renderers import _plot_scatter
from harpy.table._utils import _region_rows
from harpy.table.pl._histogram import _check_var_selection, _format_display_name
from harpy.utils._keys import _CELLSIZE_KEY


def scatter(
    adata: AnnData,
    column_x: str = _CELLSIZE_KEY,
    column_y: str = "total_counts",
    *,
    region: str | Sequence[str] | None = None,
    dataframe: Literal["obs", "var", "auto"] = "auto",
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
    Plot the relationship between two numeric columns of a table's ``.obs`` or ``.var``.

    A 2D histogram of the rows, with a lowess line, unlike the dot plot of
    :func:`scanpy.pl.scatter`: of the instances, such as cells, for two
    ``.obs`` columns, and of the features, such as genes or channels, for two
    ``.var`` columns. The table is not modified.

    Parameters
    ----------
    adata
        The table, for example ``sdata.tables["table"]``.
    column_x
        Column to plot on the x-axis.
    column_y
        Column to plot on the y-axis, of the same dataframe as ``column_x``:
        each point is one instance, such as a cell, for two ``.obs`` columns,
        or one feature, such as a gene or a channel, for two ``.var`` columns.
    region
        ``None``, the default, plots every row. Otherwise, one or more regions
        of the table's SpatialData annotation, ``uns["spatialdata_attrs"]["region"]``,
        whose rows of ``.obs`` are plotted, whatever the type of the annotated
        elements. ``.var`` columns describe the whole table, so they cannot be
        plotted for a subset of the regions.
    dataframe
        Which annotation dataframe holds both columns. With ``"auto"``, the
        one of ``adata.obs`` and ``adata.var`` that holds both; it raises if
        neither or both do. For example, after
        :func:`scanpy.pp.calculate_qc_metrics`, ``total_counts`` is in both,
        and ``shapeSize`` with ``total_counts`` selects ``.obs``.
    ax
        Matplotlib axes to draw on. If ``None``, a new figure and axes are created.
    figsize
        Figure size used when ``ax`` is ``None``.
    title
        Optional plot title. None leaves the axes title unchanged.
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

        # Gene-level QC metrics of scanpy.pp.calculate_qc_metrics.
        hp.tb.pl.scatter(
            sdata.tables["table_transcriptomics_preprocessed"],
            column_x="mean_counts",
            column_y="pct_dropout_by_counts",
            dataframe="var",
        )
    """
    rows = _region_rows(adata, region)
    resolved_dataframe = _resolve_pair_dataframe(adata, column_x, column_y, dataframe=dataframe)
    _check_var_selection(resolved_dataframe, rows)
    frame = getattr(adata, resolved_dataframe)
    for column in (column_x, column_y):
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise TypeError(
                f"Column '{column}' in 'adata.{resolved_dataframe}' is not numeric and cannot be visualized."
            )

    values = frame[[column_x, column_y]].copy()
    if resolved_dataframe == "obs" and rows is not None:
        values = values.loc[rows]

    values = values.dropna()
    if values.empty:
        raise ValueError(
            f"Columns '{column_x}' and '{column_y}' in 'adata.{resolved_dataframe}' do not contain any paired "
            "non-null values."
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


def _resolve_pair_dataframe(
    adata: AnnData, column_x: str, column_y: str, *, dataframe: Literal["obs", "var", "auto"]
) -> Literal["obs", "var"]:
    """Return the dataframe that holds both columns: the one requested, or for ``"auto"`` the only one holding both.

    A point is one row, an instance of ``.obs`` or a feature of ``.var``, so both columns come from one dataframe.
    """
    if dataframe in {"obs", "var"}:
        missing = [column for column in (column_x, column_y) if column not in getattr(adata, dataframe).columns]
        if missing:
            raise ValueError(f"Columns {missing} not found in 'adata.{dataframe}'.")
        return dataframe
    holding = [name for name in ("obs", "var") if {column_x, column_y} <= set(getattr(adata, name).columns)]
    if not holding:
        raise ValueError(
            f"Columns '{column_x}' and '{column_y}' are not both in 'adata.obs', nor both in 'adata.var'. A point is "
            "one instance, a row of 'adata.obs', or one feature, a row of 'adata.var', so plot two columns of one "
            "dataframe."
        )
    if len(holding) > 1:
        raise ValueError(
            f"Columns '{column_x}' and '{column_y}' are both in 'adata.obs' and in 'adata.var'. Please set "
            "'dataframe' explicitly."
        )
    return holding[0]
