from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dask.dataframe.core import DataFrame as DaskDataFrame
from shapely.affinity import translate
from spatialdata import SpatialData

from sparrow.image._image import (
    _apply_transform,
    _get_boundary,
    _get_spatial_element,
    _unapply_transform,
)
from sparrow.plot._plot import _get_translation_values_shapes, _get_z_slice_polygons
from sparrow.shape import intersect_rectangles
from sparrow.shape._shape import _extract_boundaries_from_geometry_collection
from sparrow.utils._keys import _GENES_KEY
from sparrow.utils._transformations import _identity_check_transformations_points
from sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def sanity_plot_transcripts_matrix(
    sdata: SpatialData,
    img_layer: str | None = None,
    labels_layer: str | None = None,
    points_layer: str = "transcripts",
    shapes_layer: str | None = None,
    channel: int | str | None = None,
    z_slice: float | None = None,
    plot_cell_number: bool = False,
    n_sample: int | None = None,
    name_x: str = "x",
    name_y: str = "y",
    name_z: str = "z",
    name_gene_column: str = _GENES_KEY,
    gene: str | None = None,
    radius: str | None = None,
    crd: tuple[int, int, int, int] | None = None,
    to_coordinate_system: str = "global",
    figsize: tuple[int, int] | None = None,
    output: Path | str | None = None,
) -> None:
    """
    Produce a sanity plot to visualize spatial transcriptomics data on top of an image.

    This function plots a spatial image (e.g. microscopy or histology image) and overlays transcripts
    and optionally cell boundaries to visually inspect the spatial alignment of the data.
    This can be particularly useful to check data registration and alignment in spatial transcriptomics.

    Parameters
    ----------
    sdata
        The SpatialData object containing the spatial image and transcripts data.
    img_layer
        The layer of the SpatialData object to be plotted. Defaults to the last layer if not provided.
    labels_layer
        Labels layer to be plotted.
    points_layer
        The points layer in the SpatialData object representing transcripts.
    shapes_layer
        The layer in the SpatialData object representing cell boundaries. If None, no cell boundaries are plotted.
    channel
        Channel to display from the img_layer. If none provided, or if provided channel could not be found, first channel is plot.
    z_slice
        The z_slice to visualize in case of 3D (c,z,y,x) image/polygons. For transcripts, if the z_slice is specified,
        the transcripts at index corresponding to the z_slice in the image layer will be plotted.
        If no z_slice is specified and `img_layer` or `labels_layer` is 3D, a max projection along the z-axis will be performed.
        If no z_slice is specified and `shapes_layer` is 3D, all polygons in all z-stacks will be plotted.
        If no z-slice is specified and `points_layer` is 3D, all transcripts in all z-stacks will be plotted.
    plot_cell_number
        Whether to annotate cells with their numbers on the plot.
    n_sample
        The number of transcripts to sample for plotting. Useful for large datasets. Ignored if `gene` is specified.
    name_x
        Column name in the points_layer representing x-coordinates of transcripts.
    name_y
        Column name in the points_layer representing y-coordinates of transcripts.
    name_z
        Column name in the points_layer representing z-coordinates of transcripts.
    name_gene_column
        Column name in the points_layer representing gene information.
    gene
        Specific gene to filter and plot. If None, all genes are plotted.
    radius
        Column in the `shapes_layer` specifying the radius. The radius will be applied using `geometry.buffer` before plotting `shapes_layer`.
        Useful when the `geometry` of the `shapes_layer` contains points instead of polygons.
    crd
        Coordinates to define a rectangular region for plotting as (xmin, xmax, ymin, ymax).
        If None, the entire image boundary is used.
    to_coordinate_system
        Coordinate system to plot.
    figsize
        Size of the figure for plotting.
    output
        Filepath to save the generated plot. If not provided, the plot will be displayed using plt.show().

    Returns
    -------
    Either displays the plot or saves it to the specified output location.

    Raises
    ------
    ValueError
        If both `img_layer` and `labels_layer` are specified.
    ValueError
        If `img_layer` or `labels_layer` is specified, and they are not found in `sdata.images` respectively `sdata.labels`.
    AttributeError
        If `sdata` does not contain a `points_layer`.
    Warning
        If provided coordinates (crd) and image boundary do not have overlap.
    Warning
        If provided shapes_layer is not present in the SpatialData object.

    Notes
    -----
    - If a specific gene is provided, only transcripts corresponding to that gene are plotted.
    - If `plot_cell_number` is set to True, cells are annotated with their numbers.
    - If the `output` parameter is provided, the plot is saved to the specified location, otherwise, it's displayed.
    """
    if img_layer is not None and labels_layer is not None:
        raise ValueError(
            "Both img_layer and labels_layer is not None. " "Please specify either img_layer or labels_layer, not both."
        )

    # Choose the appropriate layer or default to the last image layer if none is specified.
    if img_layer is not None:
        layer = img_layer
        if layer not in sdata.images:
            raise ValueError(f"Provided layer '{layer}' is not an image layer in 'sdata'.")
        img_layer_type = True
    elif labels_layer is not None:
        layer = labels_layer
        if layer not in sdata.labels:
            raise ValueError(f"Provided layer '{layer}' is not a labels layer in 'sdata'.")
        img_layer_type = False
    else:
        layer = [*sdata.images][-1]
        img_layer_type = True
        log.warning(
            f"No image layer or labels layer specified. "
            f"Plotting last image layer {layer} of the provided SpatialData object."
        )

    se = _get_spatial_element(sdata, layer=layer)

    _, ax = plt.subplots(figsize=(10, 10) if figsize is None else figsize)

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
    # if crd is None, set crd equal to image_boundary
    else:
        crd = image_boundary

    se, x_coords_orig, y_coords_orig = _apply_transform(se, to_coordinate_system=to_coordinate_system)

    z_index = None
    if z_slice is not None:
        if "z" in se.dims:
            if z_slice not in se.z.data:
                raise ValueError(
                    f"z_slice {z_slice} not a z slice in layer '{layer}' of `sdata`. "
                    f"Please specify a z_slice from the list '{se.z.data}'."
                )
            z_index = np.where(se.z.data == z_slice)[0][0]

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
                    f"Layer '{layer}' has 3 spatial dimensions, but no z-slice was added. "
                    f"will perform a max projection along the z-axis."
                )
                _se = _se.max(dim="z")
            else:
                log.info(
                    f"Layer '{layer}' has 3 spatial dimensions, but no z-slice was added. "
                    f"By default the z-slice located at the midpoint of the z-dimension ({_se.shape[0]//2}) will be utilized."
                )
                _se = _se[_se.shape[0] // 2, ...]
        _se = _se.squeeze()

    _se.sel(x=slice(crd[0], crd[1]), y=slice(crd[2], crd[3])).plot.imshow(
        cmap=cmap_layer, robust=True, ax=ax, add_colorbar=False
    )

    se = _unapply_transform(se, x_coords_orig, y_coords_orig)

    if not hasattr(sdata, "points"):
        raise AttributeError("Please first read transcripts in SpatialData object.")

    _identity_check_transformations_points(sdata.points[points_layer], to_coordinate_system=to_coordinate_system)

    in_df = sdata.points[points_layer]

    # query first and then slicing gene is faster than vice versa
    in_df = in_df.query(f"{crd[0]} <= {name_x} < {crd[1]} and {crd[2]} <= {name_y} < {crd[3]}")

    if z_index is not None:
        in_df = in_df.query(f"{name_z} == {z_index}")

    if gene:
        in_df = in_df[in_df[name_gene_column] == gene]

    # we do not sample a fraction of the transcripts if a specific gene is given
    else:
        size = len(in_df)

        log.info(f"size before sampling is {size}")

        if n_sample is not None and size > n_sample:
            fraction = n_sample / size
            in_df = in_df.sample(frac=fraction)

    if isinstance(in_df, DaskDataFrame):
        in_df = in_df.compute()

    log.info(f"Plotting {in_df.shape[0]} transcripts.")

    if gene:
        alpha = 0.5
    else:
        alpha = 0.2

    ax.scatter(in_df[name_x], in_df[name_y], color="r", s=8, alpha=alpha)

    if shapes_layer is not None:
        polygons = sdata.shapes[shapes_layer]
    else:
        polygons = None

    if polygons is not None:
        log.info("Selecting boundaries")

        if not polygons.empty:
            if radius is not None:
                if radius in polygons.columns:
                    polygons = polygons.copy()
                    polygons["geometry"] = polygons.geometry.buffer(polygons[radius])
                else:
                    log.warning(
                        f"radius parameter was specified as '{radius}', but could not be found as a column of shapes layer '{shapes_layer}'. Will proceed "
                        "plotting while ignoring radius parameter."
                    )
            x_translation, y_translation = _get_translation_values_shapes(
                polygons, to_coordinate_system=to_coordinate_system
            )
            _crd_shapes = [
                crd[0] - x_translation,
                crd[1] - x_translation,
                crd[2] - y_translation,
                crd[3] - y_translation,
            ]
            polygons = polygons.cx[_crd_shapes[0] : _crd_shapes[1], _crd_shapes[2] : _crd_shapes[3]]
            if x_translation != 0 or y_translation != 0:
                polygons["geometry"] = polygons["geometry"].apply(
                    lambda geom, x_trans=x_translation, y_trans=y_translation: translate(
                        geom, xoff=x_trans, yoff=y_trans
                    )
                )
            if z_index is not None:
                polygons = _get_z_slice_polygons(polygons, z_index=z_index)

        if not polygons.empty:
            polygons["boundaries"] = polygons["geometry"].apply(_extract_boundaries_from_geometry_collection)
            exploded_boundaries = polygons.explode("boundaries")
            exploded_boundaries["geometry"] = exploded_boundaries["boundaries"]
            exploded_boundaries = exploded_boundaries.drop(columns=["boundaries"])

            log.info("Plotting boundaries")

            # Plot the polygon boundaries
            exploded_boundaries.plot(
                ax=ax,
                aspect=1,
            )

            log.info("End plotting boundaries")

            # Plot the values inside the polygons
            cell_numbers_plotted = set()
            if plot_cell_number:
                for _, row in polygons.iterrows():
                    centroid = row.geometry.centroid
                    value = row.name
                    if value in cell_numbers_plotted:
                        # plot cell number only once for 3D stack of polygons
                        continue
                    cell_numbers_plotted.add(value)
                    ax.annotate(
                        value,
                        (centroid.x, centroid.y),
                        color="green",
                        fontsize=20,
                        ha="center",
                        va="center",
                    )
        else:
            log.warning(f"Shapes layer {shapes_layer} was empty for crd {crd}.")

    ax.set_xlim(crd[0], crd[1])
    ax.set_ylim(crd[2], crd[3])

    ax.invert_yaxis()

    ax.axis("on")
    if img_layer_type:
        ax.set_title(f"{channel_name}={channel}")

    if gene:
        ax.set_title(f"Transcripts and cell boundaries for gene: {gene}.")

    if output:
        plt.savefig(output)
    else:
        plt.show()
    plt.close()
