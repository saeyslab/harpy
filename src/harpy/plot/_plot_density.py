import matplotlib.pyplot as plt
import numpy as np
from loguru import logger as log
from matplotlib.axes import Axes
from scipy.ndimage import gaussian_filter
from spatialdata import SpatialData

from harpy.table._table import ProcessTable
from harpy.utils._keys import _SPATIAL

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
    ax.set_aspect("equal")
    ax.invert_yaxis()

    if created_ax:
        fig.tight_layout()
    return ax


def plot_instance_density(
    sdata: SpatialData,
    table_name: str,
    labels_name: str | list[str] | None = None,
    spatial_key: str = _SPATIAL,
    bin_size: float = 100,
    smooth_sigma: float | None = None,
    cmap: str = "cividis",
    figsize: tuple = (8, 8),
    colorbar: bool = True,
    ax: Axes | None = None,
) -> Axes:
    """
    Plot an instance density heatmap from centroids stored in ``sdata.tables[table_name].obsm[spatial_key]``.

    Parameters
    ----------
    sdata
        :class:`~spatialdata.SpatialData` object.
    table_name
        Table element to plot from ``sdata.tables``.
    labels_name
        Labels element(s) used to select the instances from ``table_name`` via the table region key.
        If ``None``, all observations from ``table_name`` are used.
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

    Examples
    --------
    >>> import harpy as hp
    >>> sdata = hp.datasets.xenium_human_ovarian_cancer(
    ...     subset=True,
    ... )
    >>> hp.pl.plot_instance_density(
    ...     sdata,
    ...     labels_name="cell_labels_global",
    ...     table_name="table_global",
    ... )
    """
    process_table = ProcessTable(sdata, labels_name=labels_name, table_name=table_name)
    adata = sdata.tables[table_name]

    if spatial_key not in adata.obsm:
        raise ValueError(
            f"Key '{spatial_key}' not found in 'sdata.tables[\"{table_name}\"].obsm'. "
            f"Choose from {list(adata.obsm.keys())}."
        )

    # Avoid ProcessTable._get_adata() here because it makes a full AnnData copy,
    # while this plotting path only needs the selected coordinates from .obsm.
    coords = adata.obsm[spatial_key]
    if process_table.labels_name is not None:
        mask = adata.obs[process_table.region_key].isin(process_table.labels_name).to_numpy()
        coords = coords[mask]

    coords = np.asarray(coords)

    if coords.shape[0] == 0:
        raise ValueError("No instances found for the specified labels element(s).")

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
