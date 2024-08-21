from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from geopandas.geodataframe import GeoDataFrame
from geopandas.geoseries import GeoSeries
from scipy.sparse import issparse
from shapely.affinity import translate
from spatialdata import SpatialData
from spatialdata.transformations import get_transformation

from sparrow.image._image import (
    _apply_transform,
    _get_boundary,
    _get_spatial_element,
    _unapply_transform,
)
from sparrow.shape import intersect_rectangles
from sparrow.utils._keys import _INSTANCE_KEY, _REGION_KEY
from sparrow.utils._transformations import _get_translation_values
from sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def plot_image(
    sdata: SpatialData,
    img_layer: str = "raw_image",
    channel: int | str | Iterable[int | str] | None = None,
    z_slice: float | None = None,
    crd: tuple[int, int, int, int] | None = None,
    to_coordinate_system: str = "global",
    output: str | Path | None = None,
    **kwargs: dict[str, Any],
) -> None:
    """
    Plot an image based on given parameters.

    Parameters
    ----------
    sdata
        Data containing spatial information for plotting.
    img_layer
        Image layer to be plotted. Default is "raw_image".
    channel
        Channel(s) to be displayed from the image.
    z_slice
        The z_slice to visualize in case of 3D (c,z,y,x) image.
    crd
        The coordinates for the region of interest in the format (xmin, xmax, ymin, ymax). If None, the entire image is considered, by default None.
    to_coordinate_system
        Coordinate system to plot.
    output
        Path to save the plot. If not provided, plot will be displayed.
    **kwargs
        Additional arguments to be passed to the `sp.pl.plot_shapes` function.

    See Also
    --------
    sparrow.pl.plot_shapes
    """
    plot_shapes(
        sdata,
        img_layer=img_layer,
        shapes_layer=None,
        channel=channel,
        z_slice=z_slice,
        crd=crd,
        to_coordinate_system=to_coordinate_system,
        output=output,
        **kwargs,
    )


def plot_labels(
    sdata: SpatialData,
    labels_layer: str = "segmentation_mask",
    z_slice: float | None = None,
    crd: tuple[int, int, int, int] | None = None,
    to_coordinate_system: str = "global",
    output: str | Path | None = None,
    **kwargs: dict[str, Any],
) -> None:
    """
    Plot a labels layer (masks) based on given parameters.

    Parameters
    ----------
    sdata
        Data containing spatial information for plotting.
    labels_layer
        Labels layer to be plotted. Default is "segmentation_mask".
    z_slice
        The z_slice to visualize in case of 3D (c,z,y,x) labels.
    crd
        The coordinates for the region of interest in the format (xmin, xmax, ymin, ymax). If None, the entire image is considered, by default None.
    to_coordinate_system
        Coordinate system to plot.
    output
        Path to save the plot. If not provided, plot will be displayed.
    **kwargs
        Additional arguments to be passed to the `sp.pl.plot_shapes` function.

    See Also
    --------
    sparrow.pl.plot_shapes
    """
    plot_shapes(
        sdata,
        labels_layer=labels_layer,
        shapes_layer=None,
        z_slice=z_slice,
        crd=crd,
        to_coordinate_system=to_coordinate_system,
        output=output,
        **kwargs,
    )


def plot_shapes(
    sdata: SpatialData,
    img_layer: str | Iterable[str] | None = None,
    labels_layer: str | Iterable[str] | None = None,
    shapes_layer: str | Iterable[str] | None = None,
    table_layer: str | None = None,
    column: str | None = None,
    region: str | None = None,
    cmap: str | None = "magma",
    linewidth: float = 1.0,
    channel: int | str | Iterable[int] | Iterable[str] | None = None,
    z_slice: float | None = None,
    alpha: float = 0.5,
    crd: tuple[int, int, int, int] | None = None,
    to_coordinate_system: str = "global",
    vmin: float | None = None,
    vmax: float | None = None,
    vmin_img: float | None = None,
    vmax_img: float | None = None,
    shapes_layer_filtered: str | Iterable[str] | None = None,
    img_title: bool = False,
    shapes_title: bool = False,
    channel_title: bool = True,
    aspect: str = "equal",
    figsize: tuple[int, int] | None = None,
    output: str | Path | None = None,
) -> None:
    """
    Plot shapes and/or images/labels from a SpatialData object.

    The number of provided `img_layer` or `labels_layer` and `shapes_layer` should be equal if both are iterables and if their length is greater than 1.

    Examples
    --------
    1. For `img_layer=['raw_image', 'clahe']` and `shapes_layer=['segmentation_mask_boundaries', 'expanded_cells20']`:
    Subplots:
    - Column 1: 'raw_image' with 'segmentation_mask_boundaries'
    - Column 2: 'clahe' with 'expanded_cells20'

    2. For `img_layer=['raw_image', 'clahe']` and `shapes_layer='segmentation_mask_boundaries'`:
    Subplots:
    - Column 1: 'raw_image' with 'segmentation_mask_boundaries'
    - Column 2: 'clahe' with 'segmentation_mask_boundaries'

    3. For `img_layer=['raw_image', 'clahe']` and `shapes_layer=['segmentation_mask_boundaries']` (which behaves the same as the previous example):
    Subplots:
    - Column 1: 'raw_image' with 'segmentation_mask_boundaries'
    - Column 2: 'clahe' with 'segmentation_mask_boundaries'

    4. For `img_layer=['raw_image']` and `shapes_layer=['segmentation_mask_boundaries', 'expanded_cells20' ]`:
    Subplots:
    - Column 1: 'raw_image' with 'segmentation_mask_boundaries'
    - Column 2: 'raw_image' with 'expanded_cells20'

    5. For `img_layer=['raw_image', 'clahe']` and `shapes_layer=None`:
    Subplots:
    - Column 1: 'raw_image'
    - Column 2: 'clahe'

    When multiple channels are supplied as an Iterable, they will be displayed as rows in the image

    Parameters
    ----------
    sdata
        Data containing spatial information for plotting.
    img_layer
        Image layer(s) to be plotted. If not provided, and labels_layer is also not provided, the last added image layer is plotted.
        Displayed as columns in the plot, if multiple are provided.
    labels_layer
        Labels layer(s) to be plotted.
        Displayed as columns in the plot, if multiple are provided.
    shapes_layer
        Specifies which shapes to plot. If set to None, no shapes_layer is plotted.
        Displayed as columns in the plot, if multiple are provided.
    table_layer
        Table layer to be plotted (i.e. to base cell colors on) if `column` is specified.
    column
        Column in `sdata.tables[table_layer].obs` or name in `sdata.tables[table_layer].var.index` to base cell colors on. If none provided, default color is used.
    region
        If `table_layer` and `column` is specified, this specifies the region in `sdata.tables[table_layer]` to be plotted (via `sdata.tables[table_layer].obs[_REGION_KEY]`).
    cmap
        Colormap for column. Ignored if column is None, or if column + "_colors" is in `sdata.tables[table_layer].uns`.
    linewidth
        linewidth of the shapes layer to be plotted.
    channel
        Channel(s) to be displayed from the image. Displayed as rows in the plot.
        If channel is None, get the number of channels from the first img_layer given as input.
        Ignored if img_layer is None and labels_layer is specified.
    z_slice
        The z_slice to visualize in case of 3D (c,z,y,x) image/polygons.
        If no z_slice is specified and `img_layer` or `labels_layer` is 3D, a max projection along the z-axis will be performed.
        If no z_slice is specified and `shapes_layer` is 3D, all polygons in all z-stacks will be plotted.
    crd
        The coordinates for the region of interest in the format (xmin, xmax, ymin, ymax). If None, the entire image is considered, by default None.
    to_coordinate_system
        Coordinate system to plot.
    vmin
        Lower bound for color scale for continuous data (i.e. a column). Given as a percentile. Ignored if column is None.
    vmax
        Upper bound for color scale for continuous data (i.e. a column). Given as a percentile. Ignored if column is None.
    vmin_img
        Lower bound for plotting of `img_layer` or `labels_layer`.
    vmax_img
        Upper bound for plotting of `img_layer` or `labels_layer`.
    shapes_layer_filtered
        Extra shapes layers to plot. E.g. shapes filtered out in previous preprocessing steps.
    img_title
        A flag indicating whether the image layer's name should be added to the title of the plot.
    shapes_title
        A flag indicating whether the shapes layer's name should be added to the title of the plot.
    channel_title
        A flag indicating whether the channel's name should be added to the title of the plot.
        Ignored if img_layer is None and labels_layer is specified.
    aspect
        Aspect ratio for the plot.
    figsize
        Size of the figure for plotting. If not provided, a default size is used based on the number of columns and rows.
    output
        Path to save the plot. If not provided, plot will be displayed.

    Raises
    ------
    ValueError
        If both `img_layer` and `labels_layer` are specified.
    ValueError
        If `img_layer` or `labels_layer` is specified, and they are not found in `sdata.images` respectively `sdata.labels`.
    ValueError
        If z_slice is specified, and it is not a z_slice in specified `img_layer` or `labels_layer`.
    ValueError
        If a `column` is specified, but no `table_layer`.
    ValueError
        If `table_layer` is specified, but `table_layer` is not a table in `sdata.tables`.
    ValueError
        If `sdata.tables[table_layer].obs[_REGION_KEY].cat.categories` contains more than on element, but `region` is not specified.
    ValueError
        If both `table_layer`, `column` and `region` are specified, but `region` is not in `sdata.tables[table_layer].obs[_REGION_KEY].cat.categories`.

    Notes
    -----
    - This function offers advanced visualization options for `sdata` with support for multiple image layers, labels layers shape layers, and channels.
    - Either `img_layer` or `labels_layer` should be specified, not both.
    """
    if img_layer is not None and labels_layer is not None:
        raise ValueError(
            "Both img_layer and labels_layer is not None. " "Please specify either img_layer or labels_layer, not both."
        )

    if column is not None and table_layer is None:
        raise ValueError("Please specify a 'table_layer' if a 'column' is specified.")

    # Choose the appropriate layer or default to the last image layer if none is specified.
    if img_layer is not None:
        layer = img_layer
        img_layer_type = True
    elif labels_layer is not None:
        layer = labels_layer
        img_layer_type = False
    else:
        layer = [*sdata.images][-1]
        img_layer_type = True
        log.warning(
            f"No image layer or labels layer specified. "
            f"Plotting last image layer '{layer}' of the provided SpatialData object."
        )

    if table_layer is not None:
        if table_layer not in [*sdata.tables]:
            raise ValueError(f"table layer '{table_layer}' not found in 'sdata.tables'")

    # Make code also work if user would provide another iterable than List
    layer = list(layer) if isinstance(layer, Iterable) and not isinstance(layer, str) else [layer]
    shapes_layer = (
        list(shapes_layer)
        if isinstance(shapes_layer, Iterable) and not isinstance(shapes_layer, str)
        else [shapes_layer]
    )
    if channel is not None:
        channel = list(channel) if isinstance(channel, Iterable) and not isinstance(channel, str) else [channel]

    # if multiple shapes are provided, and one img_layer, then len(shapes_layer) subfigures with same img_layer beneath are plotted.
    if len(layer) == 1 and shapes_layer != 1:
        layer = layer * len(shapes_layer)
    # if multiple img_layers are provided, and one shapes_layer, then len(img_layer) subfigures with same shapes_layer above are plotted.
    if len(shapes_layer) == 1 and layer != 1:
        shapes_layer = shapes_layer * len(layer)

    if isinstance(layer, list) and isinstance(shapes_layer, list) and len(layer) != len(shapes_layer):
        raise ValueError(f"Length of '{layer}' is not equal to the length of shapes_layer '{shapes_layer}'.")

    nr_of_columns = max(len(layer), len(shapes_layer))

    if img_layer_type:
        # if channel is None, get the number of channels from the first img_layer given, maybe print a message about this.
        if channel is None:
            se = _get_spatial_element(sdata, layer=layer[0])
            channels = se.c.data
        else:
            channels = channel

    else:
        channels = [None]  # for labels_layer type, there are no channels

    nr_of_rows = len(channels)

    if figsize is None:
        figsize = (
            10 * nr_of_columns,
            10 * nr_of_rows,
        )
    fig, axes = plt.subplots(nr_of_rows, nr_of_columns, figsize=figsize)

    # Flattening axes to make iteration easier
    if nr_of_rows == 1 and nr_of_columns == 1:
        axes = np.array([axes])
    elif nr_of_rows == 1 or nr_of_columns == 1:
        axes = axes.reshape(-1)  # make it 1D if it isn't
    else:
        axes = axes.ravel()  # flatten the 2D array to 1D

    idx = 0
    for _channel in channels:
        for _layer, _shapes_layer in zip(layer, shapes_layer):
            _plot(
                sdata,
                axes[idx],
                img_layer=_layer if img_layer_type else None,
                labels_layer=_layer if not img_layer_type else None,
                shapes_layer=_shapes_layer,
                table_layer=table_layer,
                column=column,
                region=region,
                cmap=cmap,
                linewidth=linewidth,
                channel=_channel,
                z_slice=z_slice,
                alpha=alpha,
                crd=crd,
                to_coordinate_system=to_coordinate_system,
                vmin=vmin,
                vmax=vmax,
                vmin_img=vmin_img,
                vmax_img=vmax_img,
                shapes_layer_filtered=shapes_layer_filtered,
                img_title=img_title,
                shapes_title=shapes_title,
                channel_title=channel_title,
                aspect=aspect,
            )
            idx += 1

    plt.tight_layout()
    # Save the plot to output
    if output:
        fig.savefig(output)
    else:
        plt.show()
    plt.close()


def _plot(
    sdata: SpatialData,
    ax: plt.Axes,
    img_layer: str | None = None,
    labels_layer: str | None = None,
    shapes_layer: str | None = "segmentation_mask_boundaries",
    table_layer: str | None = None,
    column: str | None = None,
    region: str | None = None,
    cmap: str | None = "magma",
    linewidth: float = 1.0,
    channel: int | str | None = None,
    z_slice: float | None = None,
    alpha: float = 0.5,
    crd: tuple[int, int, int, int] | None = None,
    to_coordinate_system: str = "global",
    vmin: float | None = None,
    vmax: float | None = None,
    vmin_img: float | None = None,
    vmax_img: float | None = None,
    shapes_layer_filtered: str | Iterable[str] | None = None,
    img_title: bool = False,
    shapes_title: bool = False,
    channel_title: bool = True,
    aspect: str = "equal",
) -> plt.Axes:
    """
    Plots a SpatialData object.

    Parameters
    ----------
    sdata
        Data containing spatial information for plotting.
    ax
        Axes object to plot on.
    img_layer
        Image layer to be plotted. By default, the last added image layer is plotted.
    labels_layer
        Labels layer to be plotted.
    shapes_layer
        Specifies which shapes to plot. Default is 'segmentation_mask_boundaries'. If set to None, no shapes_layer is plot.
    table_layer
        Table layer to be plotted (i.e. to base cell colors on) if `column` is specified.
    column
        Column in `sdata.tables[table_layer].obs` or name in `sdata.tables[table_layer].var.index` to base cell colors on. If none provided, default color is used for plotting shapes.
    region
        If `table_layer` and `column` is specified, this specifies the region in `sdata.tables[table_layer]` to be plotted (via `sdata.tables[table_layer].obs[_REGION_KEY]`).
    cmap
        Colormap for column. Ignored if column is None, or if column + "_colors" is in `sdata.tables[table_layer].uns`.
    linewidth
        linewidth of the shapes layer to be plotted.
    channel
        Channel to display from the image. If none provided, or if provided channel could not be found, first channel is plot.
        Ignored if img_layer is None and labels_layer is specified.
    z_slice
        The z_slice to visualize in case of 3D (c,z,y,x) image/polygons.
        If no z_slice is specified and `img_layer` or `labels_layer` is 3D, a max projection along the z-axis will be performed.
        If no z_slice is specified and `shapes_layer` is 3D, all polygons in all z-stacks will be plotted.
    alpha
        Transparency level for the cells, given by the alpha parameter of matplotlib.
    crd
        The coordinates for the region of interest in the format (xmin, xmax, ymin, ymax). If None, the entire image is considered, by default None.
    to_coordinate_system
        Coordinate system to plot.
    vmin
        Lower bound for color scale for continuous data (i.e. a column). Given as a percentile. Ignored if column is None.
    vmax
        Upper bound for color scale for continuous data (i.e. a column). Given as a percentile. Ignored if column is None.
    vmin_img
        Lower bound for plotting of `img_layer` or `labels_layer`.
    vmax_img
        Upper bound for plotting of `img_layer` or `labels_layer`.
    shapes_layer_filtered
        Extra shapes layers to plot. E.g. shapes filtered out in previous preprocessing steps.
    img_title
        A flag indicating whether the image layer's name should be added to the title of the plot.
    shapes_title
        A flag indicating whether the shapes layer's name should be added to the title of the plot.
    channel_title
        A flag indicating whether the channel's name should be added to the title of the plot.
        Ignored if img_layer is None and labels_layer is specified.
    aspect
        Aspect ratio for the plot.

    Returns
    -------
    The axes with the plotted SpatialData.

    Raises
    ------
    ValueError
        If both `img_layer` and `labels_layer` are specified.
    ValueError
        If `img_layer` or `labels_layer` is specified, and they are not found in `sdata.images` respectively `sdata.labels`.
    ValueError
        If z_slice is specified, and it is not a z_slice in specified `img_layer` or `labels_layer`.
    ValueError
        If a `column` is specified, but no `table_layer`.
    ValueError
        If `table_layer` is specified, but `table_layer` is not a table in `sdata.tables`.
    ValueError
        If `sdata.tables[table_layer].obs[_REGION_KEY].cat.categories` contains more than on element, but `region` is not specified.
    ValueError
        If both `table_layer`, `column` and `region` are specified, but `region` is not in `sdata.tables[table_layer].obs[_REGION_KEY].cat.categories`.

    Notes
    -----
    The function supports various visualization options such as image layers, shape layers, channels, color mapping, and custom regions.
    """
    if img_layer is not None and labels_layer is not None:
        raise ValueError(
            "Both img_layer and labels_layer is not None. " "Please specify either img_layer or labels_layer, not both."
        )

    if column is not None and table_layer is None:
        raise ValueError("Please specify a 'table_layer' if a 'column' is specified.")

    # Choose the appropriate layer or default to the last image layer if none is specified.
    if img_layer is not None:
        layer = img_layer
        if layer not in sdata.images:
            raise ValueError(f"Provided layer '{layer}' is not an image layer in 'sdata'.")
        img_layer_type = True
    elif labels_layer is not None:
        layer = labels_layer
        img_layer_type = False
        if layer not in sdata.labels:
            raise ValueError(f"Provided layer '{layer}' is not a labels layer in 'sdata'.")
    else:
        layer = [*sdata.images][-1]
        img_layer_type = True
        log.warning(
            f"No image layer or labels layer specified. "
            f"Plotting last image layer '{layer}' of the provided SpatialData object."
        )

    if table_layer is not None:
        if table_layer not in [*sdata.tables]:
            raise ValueError(f"table layer '{table_layer}' not found in 'sdata.tables'")

    if shapes_layer_filtered is not None:
        shapes_layer_filtered = (
            list(shapes_layer_filtered)
            if isinstance(shapes_layer_filtered, Iterable) and not isinstance(shapes_layer_filtered, str)
            else [shapes_layer_filtered]
        )

    se = _get_spatial_element(sdata, layer=layer)

    # Update coords
    se, x_coords_orig, y_coords_orig = _apply_transform(se, to_coordinate_system=to_coordinate_system)

    image_boundary = _get_boundary(se, to_coordinate_system=to_coordinate_system)

    if crd is not None:
        _crd = crd
        crd = intersect_rectangles(crd, image_boundary)
        if crd is None:
            log.warning(
                f"Provided crd '{_crd}' and image_boundary '{image_boundary}' do not have any overlap. "
                f"Please provide a crd that has some overlap with the image. Skipping."
            )
            return
    else:
        crd = image_boundary

    z_index = None
    if z_slice is not None:
        if "z" in se.dims:
            if z_slice not in se.z.data:
                raise ValueError(
                    f"z_slice {z_slice} not a z slice in layer '{layer}' of `sdata`. "
                    f"Please specify a z_slice from the list '{se.z.data}'."
                )
            z_index = np.where(se.z.data == z_slice)[0][0]

    polygons = None
    if shapes_layer is not None and not sdata.shapes[shapes_layer].empty:
        # copy is necessary, otherwise, in memory shapes layer altered by performing a plot.
        polygons = _translate_polygons(sdata.shapes[shapes_layer].copy(), to_coordinate_system=to_coordinate_system)
        polygons = polygons.cx[crd[0] : crd[1], crd[2] : crd[3]]
        if z_index is not None:
            polygons = _get_z_slice_polygons(polygons, z_index=z_index)

    if polygons is not None and column is not None:
        if not polygons.empty:
            adata_view = sdata.tables[table_layer]
            # do some checks on adata regarding the region.
            regions_in_table = adata_view.obs[_REGION_KEY].cat.categories.to_list()
            if len(regions_in_table) > 1:
                if region is None:
                    raise ValueError(
                        f"'sdata.tables[{table_layer}]' contains more than one region in 'sdata.tables[{table_layer}].obs[ {_REGION_KEY} ]', please specify 'region'. Choose from the list '{regions_in_table}'."
                    )
            if region is not None:
                if region not in regions_in_table:
                    raise ValueError(
                        f"Provided 'region' ({region}) is not one of the regions in 'sdata.tables[{table_layer}].obs[ {_REGION_KEY} ]'. Please choose a region from the list '({regions_in_table})'."
                    )
                else:
                    adata_view = adata_view[adata_view.obs[_REGION_KEY] == region]

            mask = adata_view.obs[_INSTANCE_KEY].isin(set(polygons.index.astype(int)))
            adata_view = adata_view[mask]
            # sort both adata and polygons on _INSTANCE_KEY
            sorted_index = adata_view.obs[_INSTANCE_KEY].sort_values().index
            adata_view = adata_view[sorted_index]

            # sort polygons (their index corresponds to the _INSTANCE_KEY):
            polygons.index = polygons.index.astype(int)
            polygons = polygons.sort_index()

            # could be that polygons contains more elements than adata_view. So we also filter in that direction, but with raising a warning
            mask_polygons = polygons.index.isin(adata_view.obs[_INSTANCE_KEY])
            if (~mask_polygons).any():
                log.warning(
                    f"There are '{sum( ~mask_polygons )}' cells in provided shapes_layer '{shapes_layer}' not found in 'sdata.tables[{table_layer}]' (linked through '{_INSTANCE_KEY}'), these cells will not be plotted."
                )
                polygons = polygons[mask_polygons]

            if column + "_colors" in adata_view.uns:
                cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                    "new_map",
                    adata_view.uns[column + "_colors"],
                    N=len(adata_view.uns[column + "_colors"]),
                )
            if column in adata_view.obs.columns:
                column = adata_view.obs[[column]].values.flatten()
            elif column in adata_view.var.index:
                column = adata_view.X[:, np.where(adata_view.var.index == column)[0][0]]
                column = column.toarray().flatten() if issparse(column) else column
            else:
                log.info(
                    f"The column '{column}' is not a column in the dataframe 'sdata.tables[{table_layer}].obs', "
                    f"nor is it a gene/channel name (sdata.tables[{table_layer}].var.index). The plot is made without taking into account this value."
                )
                column = None
                cmap = None

            if column is not None or cmap is not None:
                # sanity checks
                assert adata_view.shape[0] == polygons.shape[0], (
                    f"The number of observations in 'sdata.tables[{table_layer}]' (for which 'sdata.tables[{table_layer}].obs[ {_REGION_KEY} ] == {region}') "
                    f"is different than number of observation in 'sdata.shapes[{shapes_layer}]'."
                )

                assert np.array_equal(adata_view.obs[_INSTANCE_KEY].values, polygons.index.values), (
                    f"'{_INSTANCE_KEY}'s of shapes layer 'sdata.shapes[{shapes_layer}]' are not the same as "
                    f"the '{_INSTANCE_KEY}'s in 'sdata.tables[{table_layer}].obs' (for which 'sdata.tables[{table_layer}].obs[ {_REGION_KEY} ] == {region}')."
                )

        else:
            log.warning(f"Shapes layer '{shapes_layer}' was empty for crd {crd}.")
    else:
        cmap = None
    if vmin is not None:
        vmin = np.percentile(column, vmin)
    if vmax is not None:
        vmax = np.percentile(column, vmax)

    if img_layer_type:
        if channel is None:
            # if channel is None, plot the first channel
            channel = se.c.data[0]
            # if channel not in spatialelement, plot the first channel
        elif channel not in se.c.data:
            _channel = channel
            channel = se.c.data[0]
            log.warning(
                f"Provided channel '{_channel}' not in list of available channels '{se.c.data}' "
                f"for provided img_layer '{layer}'. Falling back to plotting first available channel '{channel}' for this img_layer."
            )

        channel_name = se.c.name
        channel_idx = list(se.c.data).index(channel)
        _se = se.isel(c=channel_idx)
        cmap_layer = "gray"
    else:
        _se = se
        cmap_layer = "viridis"

    if z_slice is not None:
        if "z" in _se.dims:
            _se = _se.sel(z=z_slice)
    else:
        if "z" in _se.dims:
            if img_layer_type:
                log.info(
                    f"Layer '{layer}' has 3 spatial dimensions, but no z-slice was specified. "
                    f"will perform a max projection along the z-axis."
                )
                _se = _se.max(dim="z")
            else:
                log.info(
                    f"Layer '{layer}' has 3 spatial dimensions, but no z-slice was specified. "
                    f"By default the z-slice located at the midpoint of the z-dimension ({_se.shape[0]//2}) will be utilized."
                )
                _se = _se[_se.shape[0] // 2, ...]

        _se = _se.squeeze()

    _se.sel(x=slice(crd[0], crd[1]), y=slice(crd[2], crd[3])).plot.imshow(
        cmap=cmap_layer,
        robust=True,
        ax=ax,
        add_colorbar=False,
        vmin=vmin_img,
        vmax=vmax_img,
    )

    if polygons is not None:
        if not polygons.empty:
            polygons.plot(
                ax=ax,
                edgecolor="white",
                column=column,
                linewidth=linewidth,
                alpha=alpha,
                legend=True,
                aspect=1,
                cmap=cmap,
                vmax=vmax,  # np.percentile(column,vmax),
                vmin=vmin,  # np.percentile(column,vmin)
            )
        else:
            log.warning(f"Shapes layer {shapes_layer} was empty for crd {crd}.")
        if shapes_layer_filtered is not None:
            for i in shapes_layer_filtered:
                if not sdata.shapes[i].empty:
                    polygons = _translate_polygons(sdata.shapes[i].copy(), to_coordinate_system=to_coordinate_system)
                    polygons = polygons.cx[crd[0] : crd[1], crd[2] : crd[3]]
                    if z_index is not None:
                        polygons = _get_z_slice_polygons(polygons, z_index=z_index)
                    if not polygons.empty:
                        polygons.plot(
                            ax=ax,
                            edgecolor="red",
                            linewidth=linewidth,
                            alpha=alpha,
                            legend=True,
                            aspect=1,
                            cmap="gray",
                        )
                else:
                    log.warning(f"Shapes layer {i} was empty for crd {crd}.")
    ax.axes.set_aspect(aspect)
    ax.set_xlim(crd[0], crd[1])
    ax.set_ylim(crd[2], crd[3])
    ax.invert_yaxis()
    titles = []
    if channel_title and img_layer_type:
        titles.append(f"{channel_name}={channel}")
    if img_title:
        titles.append(layer)
    if shapes_title and shapes_layer:
        titles.append(shapes_layer)
    title = ", ".join(titles)
    ax.set_title(title)
    # ax.axes.xaxis.set_visible(False)
    # ax.axes.yaxis.set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Restore coords
    se = _unapply_transform(se, x_coords_orig, y_coords_orig)

    return ax


def _get_z_slice_polygons(polygons: GeoDataFrame, z_index: int) -> GeoDataFrame:
    def _get_z_slice(geometry: GeoSeries, z_value) -> bool:
        # return original geometry if geometry does not has z dimension
        if not geometry.has_z:
            return True

        if geometry.geom_type == "Polygon":
            for _x, _y, z in geometry.exterior.coords:
                if z == z_value:
                    return True

        elif geometry.geom_type == "MultiPolygon":
            for polygon in geometry.geoms:
                for _x, _y, z in polygon.exterior.coords:
                    if z == z_value:
                        return True

        return False

    if polygons.empty:
        return polygons

    return polygons[polygons["geometry"].apply(_get_z_slice, args=(z_index,))]


def _translate_polygons(polygons: GeoDataFrame, to_coordinate_system: str = "global") -> GeoDataFrame:
    # get the transformation defined on "global"
    transformations = get_transformation(polygons, get_all=True)
    if to_coordinate_system not in [*transformations]:
        raise ValueError(
            f"'Coordinate system {to_coordinate_system}' does not appear to be a coordinate system of the spatial element. "
            f"Please choose a coordinate system from this list: {[*transformations]}."
        )
    transformation = transformations[to_coordinate_system]
    x_translation, y_translation = _get_translation_values(transformation)
    polygons["geometry"] = polygons["geometry"].apply(
        lambda geom: translate(geom, xoff=x_translation, yoff=y_translation)
    )

    return polygons
