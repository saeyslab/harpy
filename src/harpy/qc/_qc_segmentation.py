from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import dask.array as da
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger as log
from matplotlib.axes import Axes
from spatialdata import SpatialData

from harpy.image._image import get_dataarray
from harpy.utils._aggregate import get_instance_size
from harpy.utils._keys import _CELLSIZE_KEY, _INSTANCE_KEY
from harpy.utils.utils import _da_unique


def segmentation_coverage(
    sdata: SpatialData,
    labels_name: str,
    microns_per_pixel: float = 1,
) -> pd.DataFrame:
    """
    Calculate coverage statistics for a segmentation labels layer.

    This function summarizes how much of the image area is covered by labeled pixels and
    how many segmented instances are present in the selected labels layer.

    Parameters
    ----------
    sdata
        SpatialData object containing the segmentation labels layer.
    labels_name
        Name of the labels layer in ``sdata`` for which coverage statistics are computed.
    microns_per_pixel
        Pixel size used to convert areas from pixels to square microns. When set to ``1``,
        area values are reported in pixels.

    Returns
    -------
    :class:`pandas.DataFrame` containing the labels layer name, the total number of instances,
    the total image area, the covered area, and the covered area percentage.

    Examples
    --------
    .. code-block:: python

        import harpy as hp

        sdata = hp.datasets.pixie_example()

        hp.qc.segmentation_coverage(
            sdata,
            labels_name="label_nuclear_fov0",
            microns_per_pixel=1,
        )
    """
    array = get_dataarray(sdata, layer=labels_name).data
    array = array[None, ...] if array.ndim == 2 else array

    total_instances = _da_unique(array, run_on_gpu=False)

    total_instances = int((total_instances != 0).sum())
    total_area = float(array.size) * (microns_per_pixel**2)
    covered_area = float(da.count_nonzero(array).compute().item()) * (microns_per_pixel**2)
    covered_area_percentage = (covered_area / total_area) * 100 if total_area else 0.0

    unit = "μm²" if microns_per_pixel != 1 else "pixels"
    return pd.DataFrame(
        {
            "labels_name": [labels_name],
            "total_instances": [total_instances],
            f"total_area_{unit}": [total_area],
            f"covered_area_{unit}": [covered_area],
            "covered_area_percentage": [covered_area_percentage],
        }
    )


def segmentation_histogram(
    sdata: SpatialData,
    labels_name: str,
    microns_per_pixel: float = 1,
    ax: Axes | None = None,
    bins: int | str = "auto",
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
    Plot a histogram of segmented instance sizes for a labels layer.

    The histogram is computed from per-instance areas extracted from the selected segmentation
    mask. Areas can be reported either in pixels or in square microns depending on
    ``microns_per_pixel``.

    Parameters
    ----------
    sdata
        SpatialData object containing the segmentation labels layer.
    labels_name
        Name of the labels layer in ``sdata`` for which instance sizes are visualized.
    microns_per_pixel
        Pixel size used to convert areas from pixels to square microns. When set to ``1``,
        instance sizes are reported in pixels.
    ax
        Matplotlib axes to draw on. If ``None``, a new figure and axes are created.
    bins
        Histogram bin specification passed to :func:`seaborn.histplot`.
    histplot_kwargs
        Keyword arguments passed to :func:`seaborn.histplot`.
    median_line_kwargs
        Keyword arguments passed to :meth:`matplotlib.axes.Axes.axvline` for the median guide line.
    median_text_kwargs
        Keyword arguments passed to :meth:`matplotlib.axes.Axes.text` for the median annotation.
    figsize
        Figure size used when ``ax`` is ``None``.
    title
        Plot title. If ``None``, no title is added.
    color
        Histogram color. If ``None``, a default color is used.
    show_median
        If ``True``, add a dashed median line and annotate the median.
    show_std
        If ``True``, include the standard deviation in the annotation box.
    ylabel
        Y-axis label. If ``None``, ``"Number of instances"`` is used.

    Returns
    -------
    :class:`matplotlib.axes.Axes` containing the histogram.

    Raises
    ------
    ValueError
        If the selected labels layer does not contain any labeled instances.

    Examples
    --------
    .. code-block:: python

        import harpy as hp

        sdata = hp.datasets.pixie_example()

        hp.qc.segmentation_histogram(
            sdata,
            labels_name="label_nuclear_fov0",
            microns_per_pixel=1,
        )
    """
    array = get_dataarray(sdata, layer=labels_name).data
    array = array[None, ...] if array.ndim == 2 else array

    log.info(f"Calculating cell size for labels layer '{labels_name}'.")
    instance_sizes = get_instance_size(mask=array, run_on_gpu=False)
    instance_sizes = instance_sizes.loc[instance_sizes[_INSTANCE_KEY] != 0, _CELLSIZE_KEY].dropna()
    if microns_per_pixel != 1:
        log.info(f"Converting cell area from pixels to square microns using microns_per_pixel={microns_per_pixel}.")
    instance_sizes = instance_sizes * (microns_per_pixel**2)
    unit = "μm²" if microns_per_pixel != 1 else "pixels"

    if instance_sizes.empty:
        raise ValueError(f"No labeled instances found in labels layer '{labels_name}'.")

    median_size = float(instance_sizes.median())
    std_size = float(instance_sizes.std())

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    histplot_kwargs = dict(histplot_kwargs)
    histplot_kwargs.setdefault("kde", True)
    histplot_kwargs.setdefault("stat", "count")
    histplot_kwargs.setdefault("edgecolor", "white")
    histplot_kwargs.setdefault("linewidth", 0.8)
    histplot_kwargs.setdefault("alpha", 0.9)
    histplot_kwargs.setdefault("color", color if color is not None else "#4C78A8")
    if bins is not None:
        histplot_kwargs.setdefault("bins", bins)

    sns.histplot(instance_sizes, ax=ax, **histplot_kwargs)

    if show_median:
        line_kwargs = dict(median_line_kwargs)
        line_kwargs.setdefault("color", "black")
        line_kwargs.setdefault("linestyle", "--")
        line_kwargs.setdefault("linewidth", 1.5)
        ax.axvline(median_size, **line_kwargs)

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
        annotation_text = f"{'Median':<6}: {median_size:.2f}"
        if show_std:
            annotation_text += f"\n{'SD':<6}: {std_size:.2f}"
        ax.text(0.02, 0.95, annotation_text, **text_kwargs)

    if title is not None:
        ax.set_title(title, weight="bold")
    ax.set_xlabel(f"Instance Size ({unit})")
    ax.set_ylabel("Number of instances" if ylabel is None else ylabel)
    return ax
