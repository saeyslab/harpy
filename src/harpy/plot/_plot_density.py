import matplotlib.pyplot as plt
import numpy as np
from loguru import logger as log
from matplotlib.axes import Axes
from scipy.ndimage import gaussian_filter
from spatialdata import SpatialData
from spatialdata.transformations import get_transformation

from harpy.table._table import ProcessTable
from harpy.utils._keys import _GENES_KEY, _SPATIAL
from harpy.utils.utils import _affine_transform

_MAX_POINTS_IN_MEMORY = 1_000_000
_MAX_HEATMAP_CELLS = 1_000_000


def _plot_density_from_coordinates(
    coords: np.ndarray,
    bin_size: float,
    extent: tuple[float, float, float, float] | None = None,
    smooth_sigma: float | None = None,
    cmap: str = "viridis",
    figsize: tuple = (8, 8),
    colorbar: bool = True,
    ax: Axes | None = None,
    label: str = "Count",
    heatmap_warning_suffix: str = "",
) -> Axes:
    if coords.shape[0] == 0:
        raise ValueError("No data available for plotting.")

    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("Coordinates must be a 2D array with at least two columns for x and y.")

    x = coords[:, 0]
    y = coords[:, 1]

    if extent is not None:
        xmin, xmax, ymin, ymax = extent
    else:
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()

    x_edges = np.arange(xmin, xmax + bin_size, bin_size)
    y_edges = np.arange(ymin, ymax + bin_size, bin_size)
    n_x_bins = len(x_edges) - 1
    n_y_bins = len(y_edges) - 1
    n_heatmap_cells = n_x_bins * n_y_bins

    if n_heatmap_cells > _MAX_HEATMAP_CELLS:
        log.warning(
            f"Creating a density grid with {n_x_bins} x-bins and {n_y_bins} y-bins ({n_heatmap_cells} total cells); "
            f"consider increasing 'bin_size'{heatmap_warning_suffix}.",
        )

    heatmap, xedges, yedges = np.histogram2d(x, y, bins=[x_edges, y_edges])

    if smooth_sigma is not None:
        heatmap = gaussian_filter(heatmap, sigma=smooth_sigma)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_ax = True
    else:
        fig = ax.figure
        created_ax = False

    im = ax.imshow(
        heatmap.T,
        origin="lower",
        cmap=cmap,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    )

    if colorbar:
        cbar = fig.colorbar(im, ax=ax, shrink=0.75)
        cbar.set_label(label, fontsize=12)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    if created_ax:
        fig.tight_layout()
    return ax


def plot_transcript_density(
    sdata: SpatialData,
    bin_size: float,
    points_layer: str,
    name_gene_column: str | None = _GENES_KEY,
    genes: str | list[str] | None = None,
    z_plane: int | float | None = None,
    smooth_sigma: float | None = None,
    cmap: str = "viridis",
    frac: float | None = None,
    figsize: tuple = (8, 8),
    crd: tuple[int, int, int, int] | None = None,
    to_coordinate_system: str = "global",  # only relevant for crd
    colorbar: bool = True,
    ax: Axes | None = None,
) -> Axes:
    """
    Plot a transcript density heatmap from a :class:`~spatialdata.SpatialData` object.

    Parameters
    ----------
    sdata
        :class:`~spatialdata.SpatialData` object.
    bin_size
        Width of a histogram bin in the units of ``to_coordinate_system``.
    points_layer
        Points layer to plot from ``sdata.points``.
    name_gene_column
        Column in the ``points_layer`` that stores gene identities.
    genes
        Gene or list of genes to visualize. If ``None``, all points are used.
    z_plane
        If provided, filter the points layer to rows with ``z == z_plane``.
        This requires a ``"z"`` column on the points layer.
    smooth_sigma
        Gaussian smoothing sigma applied to the histogram. If ``None``, no smoothing is applied.
    cmap
        Colormap passed to :func:`matplotlib.axes.Axes.imshow`.
    frac
        Fraction of points to randomly sample for plotting. If ``None``, all
        points in ``points_layer`` are visualized.
    figsize
        Figure size used when ``ax`` is not provided.
    crd
        The coordinates for the region of interest in the format
        ``(xmin, xmax, ymin, ymax)``, in the coordinate system
        ``to_coordinate_system``.
    to_coordinate_system
        Coordinate system in which ``crd``, ``bin_size``, and the plotted axes
        are interpreted.
    colorbar
        If ``True``, add a colorbar to the figure.
    ax
        :class:`matplotlib.axes.Axes` object to plot on. If ``None``, a new axes is created via
        :func:`matplotlib.pyplot.subplots`.

    Returns
    -------
    :class:`matplotlib.axes.Axes` object.
    """
    ddf = sdata.points[points_layer]
    # Dask dataframe operations can drop SpatialData metadata stored in .attrs.
    points_attrs = dict(ddf.attrs)
    ddf_attrs = dict(ddf.attrs)

    name_x = "x"  # NOTE: spatialdata always uses the names "x" and "y" as the name of the coordinates for points
    name_y = "y"

    if genes is not None:
        if name_gene_column is None:
            raise ValueError("name_gene_column must be provided if filtering genes.")

        if isinstance(genes, str):
            genes = [genes]

        ddf = ddf[ddf[name_gene_column].isin(genes)]
        ddf.attrs.update(points_attrs)

    if z_plane is not None:
        if "z" not in ddf.columns:
            raise ValueError(
                f"Parameter 'z_plane' was set to {z_plane}, but points layer '{points_layer}' does not contain a 'z' column."
            )
        ddf = ddf[ddf["z"] == z_plane]
        ddf.attrs.update(points_attrs)

    if frac is not None:
        if frac < 0 or frac > 1:
            raise ValueError(f"Please set 'frac' to a value between 0 and 1; received {frac}.")
        ddf = ddf.sample(frac=frac, random_state=42)
        ddf.attrs.update(points_attrs)

    if crd is not None:
        # Query points in the intrinsic point coordinate system after mapping the requested box back.
        coords = np.array([[crd[0], crd[2]], [crd[1], crd[3]]])
        ddf.attrs.update(points_attrs)
        transform_matrix = (
            get_transformation(ddf, to_coordinate_system=to_coordinate_system)
            .inverse()
            .to_affine_matrix(input_axes=["x", "y"], output_axes=["x", "y"])
        )
        coords = _affine_transform(coords=coords, transform_matrix=transform_matrix)
        x_query = f"{coords[0, 0].item()} <={name_x} < {coords[1, 0]}"
        y_query = f"{coords[0, 1].item()} <={name_y} < {coords[1, 1]}"
        ddf = ddf.query(f"{y_query} and {x_query}")
        ddf.attrs.update(ddf_attrs)
        ddf.attrs.update(points_attrs)

    n_points = len(ddf)
    if n_points == 0:
        if genes is not None:
            raise ValueError("No transcripts found for specified gene(s).")
        if crd is not None:
            raise ValueError(
                f"After applying the bounding-box query with coordinates {crd!r} "
                f"(xmin, xmax, ymin, ymax), the points layer '{points_layer}' is no longer present "
                "in the resulting SpatialData object. Please try different parameters for 'crd'."
            )
        raise ValueError("No data available for plotting.")

    if n_points > _MAX_POINTS_IN_MEMORY:
        log.warning(
            f"Computing {n_points} points into memory for plotting; consider using 'genes', 'frac', or 'crd' to reduce this.",
        )

    df = ddf.compute()
    ddf.attrs.update(points_attrs)
    transform_matrix = get_transformation(ddf, to_coordinate_system=to_coordinate_system).to_affine_matrix(
        input_axes=["x", "y"], output_axes=["x", "y"]
    )
    transformed_coords = _affine_transform(df[[name_x, name_y]].to_numpy(), transform_matrix=transform_matrix)

    if crd is not None:
        xmin, xmax, ymin, ymax = crd
    else:
        xmin, xmax = transformed_coords[:, 0].min(), transformed_coords[:, 0].max()
        ymin, ymax = transformed_coords[:, 1].min(), transformed_coords[:, 1].max()

    x = transformed_coords[:, 0]
    y = transformed_coords[:, 1]

    return _plot_density_from_coordinates(
        coords=np.column_stack([x, y]),
        bin_size=bin_size,
        extent=(xmin, xmax, ymin, ymax),
        smooth_sigma=smooth_sigma,
        cmap=cmap,
        figsize=figsize,
        colorbar=colorbar,
        ax=ax,
        label="Transcript Count",
        heatmap_warning_suffix=" or restricting 'crd'",
    )


def plot_instance_density(
    sdata: SpatialData,
    labels_layer: str | list[str] | None,
    table_layer: str,
    spatial_key: str = _SPATIAL,
    bin_size: float = 100,
    smooth_sigma: float | None = None,
    cmap: str = "viridis",
    figsize: tuple = (8, 8),
    colorbar: bool = True,
    ax: Axes | None = None,
) -> Axes:
    """
    Plot an instance density heatmap from centroids stored in ``sdata.tables[table_layer].obsm[spatial_key]``.

    Parameters
    ----------
    sdata
        :class:`~spatialdata.SpatialData` object.
    labels_layer
        Labels layer(s) used to select the instances from ``table_layer`` via the table region key.
        If ``None``, all observations from ``table_layer`` are used.
    table_layer
        Table layer to plot from ``sdata.tables``.
    spatial_key
        Key in ``adata.obsm`` containing instance centroid coordinates.
    bin_size
        Width of a histogram bin in the coordinate units stored in ``adata.obsm[spatial_key]``.
    smooth_sigma
        Gaussian smoothing sigma applied to the histogram. If ``None``, no smoothing is applied.
    cmap
        Colormap passed to :func:`matplotlib.axes.Axes.imshow`.
    figsize
        Figure size used when ``ax`` is not provided.
    colorbar
        If ``True``, add a colorbar to the figure.
    ax
        :class:`matplotlib.axes.Axes` object to plot on. If ``None``, a new axes is created via
        :func:`matplotlib.pyplot.subplots`.

    Returns
    -------
    :class:`matplotlib.axes.Axes` object.
    """
    process_table = ProcessTable(sdata, labels_layer=labels_layer, table_layer=table_layer)
    adata = sdata.tables[table_layer]

    if spatial_key not in adata.obsm:
        raise ValueError(
            f"Key '{spatial_key}' not found in 'sdata.tables[\"{table_layer}\"].obsm'. "
            f"Choose from {list(adata.obsm.keys())}."
        )

    # Avoid ProcessTable._get_adata() here because it makes a full AnnData copy,
    # while this plotting path only needs the selected coordinates from .obsm.
    coords = adata.obsm[spatial_key]
    if process_table.labels_layer is not None:
        mask = adata.obs[process_table.region_key].isin(process_table.labels_layer).to_numpy()
        coords = coords[mask]

    coords = np.asarray(coords)

    if coords.shape[0] == 0:
        raise ValueError("No instances found for the specified labels layer(s).")

    return _plot_density_from_coordinates(
        coords=coords,
        bin_size=bin_size,
        smooth_sigma=smooth_sigma,
        cmap=cmap,
        figsize=figsize,
        colorbar=colorbar,
        ax=ax,
        label="Instance Count",
    )
