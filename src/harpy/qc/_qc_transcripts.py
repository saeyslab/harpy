from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger as log
from matplotlib.axes import Axes
from scipy.stats import pearsonr
from spatialdata import SpatialData
from spatialdata.models import TableModel

from harpy.image._image import _get_boundary, _get_spatial_element
from harpy.table._table import ProcessTable
from harpy.utils._keys import _CELLSIZE_KEY, _GENES_KEY, _RAW_COUNTS_KEY
from harpy.utils._transformations import _identity_check_transformations_points

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

_T = TypeVar("_T")


def analyse_genes_left_out(
    sdata: SpatialData,
    labels_layer: str,
    table_layer: str,
    points_layer: str = "transcripts",
    to_coordinate_system: str = "global",
    name_x: str = "x",
    name_y: str = "y",
    name_gene_column: str = _GENES_KEY,
    output: str | Path | None = None,
) -> pd.DataFrame:
    """
    Analyse and visualize the proportion of genes that could not be assigned to an instance during allocation step.

    Parameters
    ----------
    sdata
        Data containing spatial information for plotting.
    labels_layer
        The layer in `sdata` that contains the segmentation masks.
        This layer is used to calculate the crd (region of interest) that was used in the segmentation step,
        otherwise transcript counts in `points_layer` of `sdata` (containing all transcripts)
        and the counts obtained via `sdata.tables[ table_layer ]` are not comparable.
        It is also used to select the cells in `sdata.tables[table_layer]` that are linked to this `labels_layer` via the region key.
    table_layer
        The table layer in `sdata` on which to perform analysis.
    points_layer
        The layer in `sdata` containing transcript information.
    to_coordinate_system
        The coordinate system that holds `labels_layer` and `points_layer`.
        This should be the intrinsic coordinate system in pixels.
    name_x
        The column name representing the x-coordinate in `points_layer`.
    name_y
        The column name representing the y-coordinate in `points_layer`.
    name_gene_column
        The column name representing the gene name in `points_layer`.
    output
        The path to save the generated plots. If None, plots will be shown directly using plt.show().

    Returns
    -------
    :class:`pandas.DataFrame` containing information about the proportion of transcripts kept for each gene,
    raw counts (i.e. obtained from `points_layer` of `sdata`), and the log of raw counts.

    Raises
    ------
    AttributeError
        If the provided `sdata` does not contain the necessary attributes (i.e., 'labels' or 'points').

    Notes
    -----
    This function produces two plots:
        - A scatter plot of the log of raw gene counts vs. the proportion of transcripts kept.
        - A regression plot for the same data with Pearson correlation coefficients.

    The function also prints the ten genes with the highest proportion of transcripts filtered out.

    See Also
    --------
    harpy.tb.allocate

    Examples
    --------
    .. code-block:: python

        import harpy as hp

        sdata = hp.datasets.xenium_human_ovarian_cancer(subset=True)
        hp.qc.analyse_genes_left_out(
            sdata,
            labels_layer="cell_labels_global",
            points_layer="transcripts_global",
            table_layer="table_global",
        )
    """
    if not hasattr(sdata, "labels"):
        raise AttributeError(
            "Provided SpatialData object does not have the attribute 'labels', please run segmentation step before using this function."
        )

    if not hasattr(sdata, "points"):
        raise AttributeError(
            "Provided SpatialData object does not have the attribute 'points', please run allocation step before using this function."
        )

    if not np.issubdtype(sdata.tables[table_layer].X.dtype, np.integer):
        log.warning(
            f"The count matrix of the provided table layer '{table_layer}', seems to be of type '{sdata.tables[table_layer].X.dtype}', "
            "which could indicate that the analysis is being run on normalized counts, "
            "please consider running this analysis before the counts in the AnnData object "
            "are normalized (i.e. on the raw counts)."
        )

    if labels_layer not in [*sdata.labels]:
        raise ValueError(f"labels_layer '{labels_layer}' is not a labels layer in `sdata`.")

    se = _get_spatial_element(sdata, layer=labels_layer)
    crd = _get_boundary(se, to_coordinate_system=to_coordinate_system)

    region_key = sdata.tables[table_layer].uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY]
    adata = sdata.tables[table_layer][sdata.tables[table_layer].obs[region_key] == labels_layer]

    ddf = sdata.points[points_layer]
    _identity_check_transformations_points(ddf, to_coordinate_system=to_coordinate_system)
    ddf = ddf.query(f"{crd[0]} <= {name_x} < {crd[1]} and {crd[2]} <= {name_y} < {crd[3]}")

    _raw_counts = ddf.groupby(name_gene_column, observed=True).size().compute()
    missing_indices = adata.var.index.difference(_raw_counts.index)

    if not missing_indices.empty:
        raise ValueError(
            f"There are genes found in '.var' of table layer '{table_layer}' that are not found in the points layer '{points_layer}'. Please verify that allocation '(harpy.tb.allocation)' is performed using the correct points layer."
        )

    raw_counts = _raw_counts[adata.var.index]

    filtered = pd.DataFrame(np.array(adata.X.sum(axis=0)).flatten() / raw_counts)
    filtered = filtered.rename(columns={0: "proportion_kept"})
    filtered[_RAW_COUNTS_KEY] = raw_counts
    filtered[f"log_{_RAW_COUNTS_KEY}"] = np.log(filtered[_RAW_COUNTS_KEY])

    sns.scatterplot(data=filtered, y="proportion_kept", x=f"log_{_RAW_COUNTS_KEY}")
    plt.axvline(filtered[f"log_{_RAW_COUNTS_KEY}"].median(), color="green", linestyle="dashed")
    plt.axhline(filtered["proportion_kept"].median(), color="red", linestyle="dashed")
    plt.xlim(left=-0.5, right=filtered[f"log_{_RAW_COUNTS_KEY}"].quantile(0.99))

    if output:
        plt.savefig(f"{output}_0", bbox_inches="tight")
    else:
        plt.show()
    plt.close()

    r, p = pearsonr(filtered[f"log_{_RAW_COUNTS_KEY}"], filtered["proportion_kept"])
    sns.regplot(x=f"log_{_RAW_COUNTS_KEY}", y="proportion_kept", data=filtered)
    ax = plt.gca()
    ax.text(0.7, 0.9, f"r={r:.2f}, p={p:.2g}", transform=ax.transAxes)
    plt.axvline(filtered[f"log_{_RAW_COUNTS_KEY}"].median(), color="green", linestyle="dashed")
    plt.axhline(filtered["proportion_kept"].median(), color="red", linestyle="dashed")

    if output:
        plt.savefig(f"{output}_1", bbox_inches="tight")
    else:
        plt.show()
    plt.close()

    log.info(
        f"The ten genes with the highest proportion of transcripts filtered out in the "
        f"region of interest ([x_min,x_max,y_min,y_max]={crd}):\n"
        f"{filtered.sort_values(by='proportion_kept').iloc[0:10, 0:2]}"
    )

    return filtered


def metric_histogram(
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
    :class:`matplotlib.axes.Axes` containing the histogram.

    Examples
    --------
    .. code-block:: python

        import harpy as hp

        sdata = hp.datasets.xenium_human_ovarian_cancer(
            subset=True,
            processed=True,
        )

        hp.qc.metric_histogram(
            sdata,
            table_layer="table_transcriptomics_preprocessed",
            labels_layer="nucleus_segmentation_mask",
            column="total_counts",
            dataframe="obs",
            quantile_range=(0.1, 0.99),
        )
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


def metrics_histogram(
    sdata: SpatialData,
    table_layer: str,
    labels_layer: str | Iterable[str] | None = None,
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
    table_layer
        Table layer in ``sdata.tables``.
    labels_layer
        Label layer or layers used to subset the selected table via :class:`~harpy.table._table.ProcessTable`.
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
        Keyword arguments passed to :func:`seaborn.histplot`.
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

        hp.qc.metrics_histogram(
            sdata,
            table_layer="table_transcriptomics_preprocessed",
            labels_layer="nucleus_segmentation_mask",
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
        metric_histogram(
            sdata=sdata,
            table_layer=table_layer,
            labels_layer=labels_layer,
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


def obs_scatter(
    sdata: SpatialData,
    table_layer: str,
    labels_layer: str | Iterable[str] | None = None,
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
    table_layer
        Table layer in ``sdata.tables``.
    labels_layer
        Label layer or layers used to subset the selected table via :class:`~harpy.table._table.ProcessTable`.
        If ``None``, all observations in ``table_layer`` are used.
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
            table_layer="table_transcriptomics_preprocessed",
            column_x="shapeSize",
            column_y="total_counts",
        )
    """
    process_table = ProcessTable(sdata, labels_layer=labels_layer, table_layer=table_layer)
    adata = sdata.tables[table_layer]

    for obs_column in (column_x, column_y):
        if obs_column not in adata.obs.columns:
            raise ValueError(f"Column '{obs_column}' not found in 'adata.obs'.")
        if not pd.api.types.is_numeric_dtype(adata.obs[obs_column]):
            raise TypeError(f"Column '{obs_column}' in 'adata.obs' is not numeric and cannot be visualized.")

    values = adata.obs[[column_x, column_y]].copy()
    if process_table.labels_layer is not None:
        obs_mask = adata.obs[process_table.region_key].isin(process_table.labels_layer).to_numpy()
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


def _expand_per_metric_option(
    value: _T | Sequence[_T] | None,
    *,
    n_metrics: int,
    parameter_name: str,
) -> list[_T | None]:
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
