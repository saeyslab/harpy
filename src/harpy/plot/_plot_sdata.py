from __future__ import annotations

import uuid
from collections.abc import Mapping
from copy import deepcopy
from types import MappingProxyType
from typing import Any

import pandas as pd
from matplotlib.axes import Axes
from spatialdata import SpatialData, bounding_box_query
from spatialdata.models import TableModel

from harpy.plot._utils import _get_distinct_colors
from harpy.utils._keys import _GENES_KEY
from harpy.utils.pylogger import get_pylogger
from harpy.utils.utils import _make_list

log = get_pylogger(__name__)


try:
    import spatialdata_plot

    _ = spatialdata_plot  # prevent precommit to complain about unused imports
except ImportError:
    log.warning(
        "Module 'spatialdata-plot' not installed, please install 'spatialdata-plot' if you want to use 'harpy.pl.plot_sdata' or 'harpy.pl.plot_sdata_genes'."
    )


def plot_sdata(
    sdata: SpatialData,
    img_layer: str,
    channel: str | list[str],
    labels_layer: str | None = None,
    table_layer: str | None = None,
    color: str | None = None,
    crd: tuple[int, int, int, int] | None = None,
    to_coordinate_system: str = "global",
    render_images_kwargs: Mapping[str, Any] = MappingProxyType({}),
    render_labels_kwargs: Mapping[str, Any] = MappingProxyType({}),
    show_kwargs: Mapping[str, Any] = MappingProxyType({}),
    ax: Axes | None = None,
) -> Axes:
    """
    Light wrapper around `spatialdata-plot` to plot a :class:`~spatialdata.SpatialData` object.

    Parameters
    ----------
    sdata
        SpatialData object.
    img_layer
        Image layer to plot from `sdata.images`.
    channel
        Channel(s) to visualize, passed to `.pl.render_images()`.
    labels_layer
        Labels layer to plot from `sdata.labels`.
    table_layer
        Table layer from `sdata.tables` used to color instances of the `labels_layer`.
        If specified, the table layer should be annotated by `labels_layer`.
        Ignored if `color` is `None`.
    color
        Column from `sdata[table_layer].obs` or name from `sdata[table_layer].var_names` to color the instances from `labels_layer`.
        A ValueError is raised when `color` is specified and `table_layer` is `None`.
        Set to `None` if you do not want to color the labels, and only want to visualise the labels in `labels_layer`.
    crd
        The coordinates for the region of interest in the format `(xmin, xmax, ymin, ymax)`, in the coordinate system `to_coordinate_system`.
    to_coordinate_system
        Coordinate system to plot.
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
    ValueError
        If `coordinate_systems`in `show_kwargs`. Please pass coordinate system to plot via `to_coordinate_system`.


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
    >>> hp.pl.plot_sdata(
    ...     sdata,
    ...     img_layer="blobs_multiscale_image",
    ...     channel=[0, 1, 2],
    ...     labels_layer="blobs_labels",
    ...     table_layer="table",
    ...     color="channel_1_sum",
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
    >>> hp.pl.plot_sdata(
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
            f"Choose from {[*sdata.tables]}, and make sure the table layer is annotated by '{labels_layer}'."
        )

    if table_layer is not None:
        adata = sdata.tables[table_layer]
        region_key = adata.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY]
        # sanity check
        mask = adata.obs[region_key] == labels_layer
        if not mask.any():
            raise ValueError(
                f"The labels layer '{labels_layer}' does not seem to annotate the table layer '{table_layer}'."
            )

    if "coordinate_systems" in show_kwargs.keys():
        raise ValueError(
            "'coordinate_systems' found as key in 'show_kwargs'"
            " Please specify the coordinate sytem to plot via the parameter 'to_coordinate_system', not via 'show_kwargs'."
        )

    if isinstance(show_kwargs, MappingProxyType):
        show_kwargs = {}
    if not isinstance(show_kwargs, dict):
        raise ValueError("Please specify 'show_kwargs' as a dict.")
    show_kwargs = deepcopy(show_kwargs)  # otherwise inplace update of show_kwargs
    show_kwargs["coordinate_systems"] = [to_coordinate_system]

    sdata_to_plot = sdata
    queried = False
    if crd is not None:
        queried = True
        sdata_to_plot = bounding_box_query(
            sdata_to_plot,
            axes=["x", "y"],
            min_coordinate=[crd[0], crd[2]],
            max_coordinate=[crd[1], crd[3]],
            target_coordinate_system=to_coordinate_system,
            filter_table=True
            if table_layer is not None
            else False,  # table do not need to be filtered if table layer is not specified
        )
        if img_layer not in sdata_to_plot.images:
            raise ValueError(
                f"After applying the bounding-box query with coordinates {crd!r} "
                f"(xmin, xmax, ymin, ymax), the image layer '{img_layer}' is no longer present "
                "in the resulting SpatialData object. Please try different parameters for 'crd'."
            )
        if labels_layer is not None and labels_layer not in sdata_to_plot.labels:
            raise ValueError(
                f"After applying the bounding-box query with coordinates {crd!r} "
                f"(xmin, xmax, ymin, ymax), the labels layer '{labels_layer}' is no longer present "
                "in the resulting SpatialData object. Please try different parameters for 'crd'."
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
                "Please specify 'table_name' via the keyword argument 'table_layer' of 'hp.pl.plot_sdata.'"
            )
        if "color" in render_labels_kwargs.keys():
            raise ValueError("Please specify 'color' via the keyword argument 'color' of 'hp.pl.plot_sdata.'")
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


def plot_sdata_genes(
    sdata: SpatialData,
    points_layer: str,
    img_layer: str | None = None,
    channel: str | list[str] | None = None,  # ignored if img_layer is None
    name_gene_column: str | None = _GENES_KEY,
    genes: str | list[str] = None,  #
    palette: str | list[str] | None = None,
    color: str = "cornflowerblue",  # ignored if genes is not None
    frac: float | None = None,
    size: int | float = 5,  # size of the points
    alpha: float = 1.0,  # alpha of the points (transparency)
    crd: tuple[int, int, int, int] | None = None,
    to_coordinate_system: str = "global",
    render_images_kwargs: Mapping[str, Any] = MappingProxyType({}),
    show_kwargs: Mapping[str, Any] = MappingProxyType({}),
    ax: Axes | None = None,
) -> Axes:
    """
    Light wrapper around `spatialdata-plot` to visualize gene expression from a :class:`~spatialdata.SpatialData` object.

    Parameters
    ----------
    sdata
        SpatialData object.
    points_layer
        Points layer to plot from ``sdata.points``. The associated table is expected
        to contain gene information in ``name_gene_column``.
    img_layer
        Optional image layer to plot from ``sdata.images`` as background.
        If ``None``, no image is rendered and ``channel`` is ignored.
    channel
        Channel(s) of ``img_layer`` to visualize, passed to ``.pl.render_images()``.
        Ignored if ``img_layer`` is ``None``.
    name_gene_column
        Column in the ``points_layer`` that stores the gene
        names for each point (e.g. ``"gene"``).
    genes
        Gene or list of genes to visualize.
        * If a list is provided, only points in ``points_layer`` from this list (via ``name_gene_column``) are plotted
        (as categories).
        * If ``None``, points are plotted without gene-specific coloring and
        ``color`` is used instead.
    palette
        Colors used to color each gene in ``genes``. If ``None``, defaults to  :mod:`scanpy`’s standard color palette.
        Ignored if ``genes`` is ``None``.
    color
        Color used to plot the points when ``genes`` is ``None``. Ignored if
        ``genes`` is not ``None``.
    frac
        Fraction of points to randomly sample for plotting. If ``None``, all points
        in ``points_layer`` are visualized.
    size
        Size of the points.
    alpha
        Transparency of the points.
    crd
        The coordinates for the region of interest in the format
        ``(xmin, xmax, ymin, ymax)``, in the coordinate system
        ``to_coordinate_system``.
    to_coordinate_system
        Coordinate system to plot.
    render_images_kwargs
        Keyword arguments passed to ``.pl.render_images()`` for rendering
        ``img_layer``.
    show_kwargs
        Keyword arguments passed to ``.pl.show()``.
    ax
        Matplotlib axes object to plot on. If ``None``, a new axes is created by
        the underlying plotting function.

    Returns
    -------
    Matplotlib Axes.

    Examples
    --------
    Plot gene expression for multiple genes on top of an image:

    >>> from spatialdata.datasets import blobs
    >>> import matplotlib.pyplot as plt
    >>> import harpy as hp
    >>>
    >>> sdata = blobs()
    >>> fig, ax = plt.subplots()
    >>>
    >>> hp.pl.plot_sdata_genes(
    ...     sdata,
    ...     points_layer="blobs_points",
    ...     img_layer="blobs_image",
    ...     name_gene_column="genes",
    ...     genes=["gene_b", "gene_a"],
    ...     ax=ax,
    ... )

    Plot a single gene:

    >>> fig, ax = plt.subplots()
    >>> hp.pl.plot_sdata_genes(
    ...     sdata,
    ...     points_layer="blobs_points",
    ...     img_layer="blobs_image",
    ...     name_gene_column="genes",
    ...     genes="gene_b",
    ...     ax=ax,
    ... )

    Plot points without categorical genes, using a single color:

    >>> fig, ax = plt.subplots()
    >>> hp.pl.plot_sdata_genes(
    ...     sdata,
    ...     points_layer="blobs_points",
    ...     img_layer="blobs_image",
    ...     name_gene_column="genes",
    ...     frac=0.5,
    ...     genes=None,
    ...     color="cornflowerblue",
    ...     ax=ax,
    ... )
    """
    df = sdata.points[points_layer]
    if name_gene_column not in df.columns:
        raise ValueError(
            f"Column '{name_gene_column}' not found in 'sdata.points[{points_layer}].columns'. "
            "Please specify the column in the points layer that contains the gene name via "
            "the parameter 'name_gene_column'."
        )
    # if genes is not None, we want the name_gene_column to be ploth as categorical.
    if genes is not None and df[name_gene_column].dtype != "category":
        log.info(
            f"Column '{name_gene_column}' of 'sdata.points[{points_layer}]' is not of dtype categorical, while 'genes' is not 'None'. "
            "We proceed with categorizing the column, so spatialdata-plot can plot the genes as categories. In case of a backed SpatialData object, "
            f"this will not affect the underlying zarr store, only the in-memory representation of 'sdata.points[{points_layer}][{name_gene_column}]'."
        )
        df = df.categorize(columns=[name_gene_column])
        sdata[points_layer] = df
    # if genes is None, we want the name_gene_column to NOT be plot as categorical (otherwise all genes are plot as categories, resulting in hundreds of categories,)
    if genes is None and df[name_gene_column].dtype == "category":
        log.info(
            f"Column '{name_gene_column}' of 'sdata.points[{points_layer}]' is of dtype categorical, while 'genes' is 'None'. "
            "We proceed with converting to dtype object, to prevent spatialdata-plot to plot all genes as categories. In case of a backed SpatialData object, "
            f"this will not affect the underlying zarr store, only the in-memory representation of 'sdata.points[{points_layer}][{name_gene_column}]'."
        )
        df[name_gene_column] = df[name_gene_column].astype(str)
        sdata[points_layer] = df

    # we work with the palette, to prevent spatialdata-plot to calculate color from total number of categories in the dask dataframe, which can be hundreds,
    # which results in spatialdata-plot setting all genes to grey (if nr of categories >= 103)
    if genes is not None:
        genes = _make_list(genes)
        if palette is None:
            palette = _get_distinct_colors(len(genes))
        palette = _make_list(palette)
        if len(palette) != len(genes):
            raise ValueError(
                f"The number of genes specified via 'genes' ({len(genes)}) differs "
                f"from the number of colors in 'palette' ({len(palette)})."
            )
    else:
        if palette is not None:
            log.info(
                "'palette' is not 'None', while 'genes' is 'None'. Will proceed with setting 'palette' to 'None'. "
                "To set the color when 'genes' is 'None', please set 'color'."
            )
            palette = None

    if "coordinate_systems" in show_kwargs.keys():
        raise ValueError(
            "'coordinate_systems' found as key in 'show_kwargs'"
            " Please specify the coordinate sytem to plot via the parameter 'to_coordinate_system', not via 'show_kwargs'."
        )

    if isinstance(show_kwargs, MappingProxyType):
        show_kwargs = {}
    if not isinstance(show_kwargs, dict):
        raise ValueError("Please specify 'show_kwargs' as a dict.")
    show_kwargs = deepcopy(show_kwargs)  # otherwise inplace update of show_kwargs
    show_kwargs["coordinate_systems"] = [to_coordinate_system]

    # Note, we sample, before query.
    sampled = False
    if frac is not None:
        if frac < 0 or frac > 1:
            raise ValueError(f"Please set 'frac' to a value between 0 and 1; received {frac}.")
        df = df.sample(frac=frac, random_state=42)
        sampled_points_layer = f"{points_layer}_sample_{uuid.uuid4()}"
        sdata[sampled_points_layer] = df
        points_layer = sampled_points_layer
        sampled = True

    sdata_to_plot = sdata
    if crd is not None:
        sdata_to_plot = bounding_box_query(
            sdata,
            axes=["x", "y"],
            min_coordinate=[crd[0], crd[2]],
            max_coordinate=[crd[1], crd[3]],
            target_coordinate_system=to_coordinate_system,
        )
        if points_layer not in sdata_to_plot.points:
            raise ValueError(
                f"After applying the bounding-box query with coordinates {crd!r} "
                f"(xmin, xmax, ymin, ymax), the points layer '{points_layer}' is no longer present "
                "in the resulting SpatialData object. Please try different parameters for 'crd'."
            )
        if img_layer is not None and img_layer not in sdata_to_plot.images:
            raise ValueError(
                f"After applying the bounding-box query with coordinates {crd!r} "
                f"(xmin, xmax, ymin, ymax), the image layer '{img_layer}' is no longer present "
                "in the resulting SpatialData object. Please try different parameters for 'crd'."
            )

    if genes is not None:
        log.info(f"Plotting column {name_gene_column} of 'sdata.points[{points_layer}]' as categorical.")
        color = name_gene_column
    if img_layer is None:
        ax = sdata_to_plot.pl.render_points(
            points_layer,
            color=color,
            alpha=alpha,
            palette=palette,
            size=size,
            groups=genes,
        ).pl.show(
            **show_kwargs,
            ax=ax,
            return_ax=True,
        )
    else:
        ax = (
            sdata_to_plot.pl.render_images(
                img_layer,
                channel=channel,
                **render_images_kwargs,
            )
            .pl.render_points(
                points_layer,
                color=color,
                alpha=alpha,
                palette=palette,
                size=size,
                groups=genes,
            )
            .pl.show(
                **show_kwargs,
                ax=ax,
                return_ax=True,
            )
        )

    if sampled:
        del sdata[points_layer]

    return ax
