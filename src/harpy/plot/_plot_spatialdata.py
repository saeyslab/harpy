from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import pandas as pd
from matplotlib.axes import Axes
from spatialdata import SpatialData, bounding_box_query

from harpy.utils._keys import _REGION_KEY
from harpy.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


try:
    import spatialdata_plot  # noqa: F401
except ImportError:
    log.warning(
        "Module 'spatialdata-plot' not installed, please install 'spatialdata-plot' if you want to use 'hp.pl.plot_spatialdata'."
    )


def plot_spatialdata(
    sdata: SpatialData,
    img_layer: str,
    channel: str | list[str],
    labels_layer: str | None = None,
    table_layer: str | None = None,
    color: str | None = None,
    crd: tuple[int, int, int, int] | None = None,
    to_coordinate_system: str = "global",
    region_key: str = _REGION_KEY,
    render_images_kwargs: Mapping[str, Any] = MappingProxyType({}),
    render_labels_kwargs: Mapping[str, Any] = MappingProxyType({}),
    show_kwargs: Mapping[str, Any] = MappingProxyType({}),
    ax: Axes | None = None,
) -> Axes:
    """
    Light wrapper around `spatialdata-plot` to plot a SpatialData object.

    Parameters
    ----------
    sdata
        SpatialData object.
    img_layer
        Image layer to plot from `sdata.images`.
    channel
        Channel(s) to visualize, passed to `.pl.render_images`.
    labels_layer
        Labels layer to plot from `sdata.labels`.
    table_layer
        Table layer from `sdata.tables` used to color instances of the `labels_layer`.
        If specified, the table layer should be annotated by `labels_layer` via `region_key`.
        Ignored if `color` is `None`.
    color
        Column from `sdata[table_layer].obs` or name from `sdata[table_layer].var_names` to color the instances from `labels_layer`.
        A ValueError is raised when color is specified and `table_layer` is `None`.
        Set to `None` if you do not want to color the labels, and only want to visualise the labels in `labels_layer`.
    crd
        The coordinates for the region of interest in the format `(xmin, xmax, ymin, ymax)`.
    to_coordinate_system
        Coordinate system to plot. Only necessary to specify if `crd` is not `None`.
    region_key
        Column in `sdata[table_layer].obs` that contains the labels layer that annotates the `table_layer`.
        Ignored if `table_layer` or `labels_layer` is None.
    render_images_kwargs
        Keyword arguments passed to `.pl.render_images()`.
    render_labels_kwargs
        Keyword arguments passed to `.pl.render_labels()`.
        Ignored if `labels_layer` is `None`.
    show_kwargs
        Keyword arguments passed to `.pl.show()`.
    ax:
       Matplotlib axes object to plot on.

    Raises
    ------
    ValueError
        If `table_layer` is not None and `labels_layer` is None.
    ValueError
        If `color` is not None and `table_layer` is None.


    Examples
    --------
    >>> from spatialdata.datasets import blobs
    >>> import matplotlib.pyplot as plt
    >>> from matplotlib.colors import Normalize
    >>> import harpy as hp
    >>>
    >>> # Load example spatial dataset
    >>> sdata = blobs()
    >>>
    >>> # Configure rendering options
    >>> render_images_kwargs = {
    ...     "palette": ["red", "blue", "green"],
    ...     "scale": "scale2",
    ... }
    >>> render_labels_kwargs = {
    ...     "scale": "scale2",
    ...     "fill_alpha": 0.8,
    ...     "outline_alpha": 0.3,
    ... }
    >>> show_kwargs = {
    ...     "title": "my_multichannel_figure",
    ...     "colorbar": False,
    ...     "dpi": 200,
    ...     "figsize": (20, 20),
    ... }
    >>>
    >>> fig, ax = plt.subplots(1, 2)
    >>>
    >>> # Plot multichannel image with labels and table annotations
    >>> hp.pl.plot_spatialdata(
    ...     sdata,
    ...     img_layer="blobs_multiscale_image",
    ...     channel=[0, 1, 2],
    ...     labels_layer="blobs_labels",
    ...     table_layer="table",
    ...     color="channel_1_sum",
    ...     region_key="region",
    ...     render_images_kwargs=render_images_kwargs,
    ...     render_labels_kwargs=render_labels_kwargs,
    ...     show_kwargs=show_kwargs,
    ...     ax=ax[0],
    ... )
    >>>
    >>> # Plot a single channel with intensity normalization and crop
    >>> norm = Normalize(vmin=0.1, vmax=0.2, clip=True)
    >>> render_images_kwargs = {
    ...     "cmap": "grey",
    ...     "scale": "full",
    ...     "norm": norm,
    ... }
    >>> show_kwargs = {
    ...     "title": "my_channel_1",
    ...     "colorbar": False,
    ...     "dpi": 200,
    ...     "figsize": (20, 20),
    ... }
    >>>
    >>> hp.pl.plot_spatialdata(
    ...     sdata,
    ...     img_layer="blobs_multiscale_image",
    ...     channel=1,
    ...     crd=[100, 300, 100, 300],
    ...     render_images_kwargs=render_images_kwargs,
    ...     render_labels_kwargs=render_labels_kwargs,
    ...     show_kwargs=show_kwargs,
    ...     ax=ax[1],
    ... )
    """
    if table_layer is not None and labels_layer is None:
        raise ValueError(
            f"Please specify a labels layer (which annotates the table layer '{table_layer}') if 'table_layer' is specified."
        )
    if color is not None and table_layer is None:
        raise ValueError(
            f"Please specify a 'table_layer' if 'color' is specified. "
            f"Choose from {[*sdata.tables]}, and make sure the table layer is annotated by '{labels_layer}' via '{region_key}'."
        )

    if table_layer is not None:
        adata = sdata[table_layer]
        # sanity check
        mask = adata.obs[region_key] == labels_layer
        if not mask.any():
            raise ValueError(
                f"The labels layer '{labels_layer}' does not seem to annotate the table layer '{table_layer}."
            )

    sdata_to_plot = sdata
    queried = False
    if crd is not None:
        queried = True
        sdata_to_plot = bounding_box_query(
            sdata,
            axes=["x", "y"],
            min_coordinate=[crd[0], crd[2]],
            max_coordinate=[crd[1], crd[3]],
            target_coordinate_system=to_coordinate_system,
            filter_table=True
            if table_layer is not None
            else False,  # table do not need to be filtered if table layer is not specified
        )

    if labels_layer is None:
        ax = sdata_to_plot.pl.render_images(
            img_layer,
            channel=channel,
            **render_images_kwargs,
        ).pl.show(
            **show_kwargs,
            ax=ax,
            return_ax=True,
        )

    else:
        if "table_name" in render_labels_kwargs.keys():
            raise ValueError(
                "Please specify 'table_name' via the keyword argument 'table_layer' of 'hp.pl.plot_spatialdata.'"
            )
        if "color" in render_labels_kwargs.keys():
            raise ValueError("Please specify 'color' via the keyword argument 'color' of 'hp.pl.plot_spatialdata.'")
        # workaround for https://github.com/scverse/spatialdata-plot/issues/414, also see
        # https://github.com/ArneDefauw/spatialdata-plot/blob/5af65aa118f7abf87e47470038ecdbddb27ef1ca/tests/pl/test_render_labels.py#L250
        # I.e. by making a query (via crd), it could happen that not all categories are still present in the query.
        # if we do not remove unused categories, spatialdata plot will use different colors to color the cells than
        # compared to case without query.
        if queried:
            if (
                color is not None
                and color in sdata_to_plot[table_layer].obs
                and pd.api.types.is_categorical_dtype(sdata_to_plot[table_layer].obs[color])
            ):
                sdata_to_plot[table_layer].obs[color] = (
                    sdata_to_plot[table_layer].obs[color].cat.remove_unused_categories()
                )

        ax = (
            sdata_to_plot.pl.render_images(
                img_layer,
                channel=channel,
                **render_images_kwargs,
            )
            .pl.render_labels(
                labels_layer,
                table_name=table_layer,
                color=color,
                **render_labels_kwargs,
            )
            .pl.show(
                **show_kwargs,
                ax=ax,
                return_ax=True,
            )
        )

    return ax
