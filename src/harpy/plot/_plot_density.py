import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger as log
from matplotlib.axes import Axes
from scipy.ndimage import gaussian_filter
from spatialdata import SpatialData

from harpy.utils._keys import _GENES_KEY

_MAX_POINTS_IN_MEMORY = 1_000_000
_MAX_HEATMAP_CELLS = 1_000_000


def plot_density(
    sdata: SpatialData,
    bin_size: float,
    points_layer: str,
    name_x: str = "x",
    name_y: str = "y",
    name_gene_column: str | None = _GENES_KEY,
    genes: str | list[str] | None = None,
    smooth_sigma: float | None = None,
    cmap: str = "viridis",
    frac: float | None = None,
    figsize: tuple = (8, 8),
    crd: tuple[int, int, int, int] | None = None,
    to_coordinate_system: str = "global",  # only relevant for crd
    ax: Axes | None = None,
) -> Axes:
    ddf = sdata.points[points_layer]
    # Dask dataframe operations can drop SpatialData metadata stored in .attrs.
    points_attrs = dict(ddf.attrs)

    if genes is not None:
        if name_gene_column is None:
            raise ValueError("name_gene_column must be provided if filtering genes.")

        if isinstance(genes, str):
            genes = [genes]

        ddf = ddf[ddf[name_gene_column].isin(genes)]
        ddf.attrs.update(points_attrs)

    if frac is not None:
        if frac < 0 or frac > 1:
            raise ValueError(f"Please set 'frac' to a value between 0 and 1; received {frac}.")
        ddf = ddf.sample(frac=frac, random_state=42)
        ddf.attrs.update(points_attrs)

    if crd is not None:
        xmin, xmax, ymin, ymax = crd
        ddf = ddf[(ddf[name_x] >= xmin) & (ddf[name_x] <= xmax) & (ddf[name_y] >= ymin) & (ddf[name_y] <= ymax)]
        ddf.attrs.update(points_attrs)

    n_points = len(ddf)
    if n_points > _MAX_POINTS_IN_MEMORY:
        log.warning(
            f"Computing {n_points} points into memory for plotting; consider using 'genes', 'frac', or 'crd' to reduce this.",
        )

    df = ddf.compute()

    if genes is not None and df.empty:
        raise ValueError("No transcripts found for specified gene(s).")

    if crd is not None:
        if df.empty:
            raise ValueError("No data found in specified region.")
    else:
        xmin, xmax = df[name_x].min(), df[name_x].max()
        ymin, ymax = df[name_y].min(), df[name_y].max()

    x = df[name_x].to_numpy()
    y = df[name_y].to_numpy()

    label = "Transcript Count"
    title = "Transcript Density"

    # Create 2D histogram
    x_edges = np.arange(xmin, xmax + bin_size, bin_size)
    y_edges = np.arange(ymin, ymax + bin_size, bin_size)
    n_x_bins = len(x_edges) - 1
    n_y_bins = len(y_edges) - 1
    n_heatmap_cells = n_x_bins * n_y_bins

    if n_heatmap_cells > _MAX_HEATMAP_CELLS:
        log.warning(
            f"Creating a density grid with {n_x_bins} x-bins and {n_y_bins} y-bins ({n_heatmap_cells} total cells); "
            "consider increasing 'bin_size' or restricting 'crd'.",
        )

    heatmap, xedges, yedges = np.histogram2d(x, y, bins=[x_edges, y_edges])

    if smooth_sigma is not None:
        heatmap = gaussian_filter(heatmap, sigma=smooth_sigma)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    im = ax.imshow(
        heatmap.T,
        origin="lower",
        cmap=cmap,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.75)
    cbar.set_label(label, fontsize=12)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    fig.tight_layout()
    return ax


def plot_density_deprecated(
    sdata: SpatialData,
    points_layer: str = None,
    name_gene_column: str | None = _GENES_KEY,
    genes: str | list[str] | None = None,
    table_layer: str = None,
    bin_size: float | None = None,
    smooth_sigma: float | None = None,
    cmap: str = "viridis",
    figsize: tuple = (8, 8),
    crd: tuple[int, int, int, int] | None = None,
):
    """
    Plot transcript density from a SpatialData points layer or plot cell density from a SpatialData table layer. Exactly one of `points_layer` or `table_layer` must be provided.

    Parameters
    ----------
    sdata
        SpatialData object containing the points layer or table layer.
    points_layer
        Points layer to plot from `sdata.points`.
    name_gene_column
        Column in the `points_layer` that stores the genes.
    genes
        Gene or list of genes to visualize. If not provide, will plot a density map of all transcripts.
    table_layer
        Table layer to plot from `sdata.tables`. Must contain obsm["spatial"].
    bin_size
        Width of a bin
    smooth_sigma
        Gaussian smoothing sigma. Note that this slightly misrepresents the data. If None, no smoothing applied.
    cmap
        Matplotlib colormap.
    crd
        The coordinates for a region of interest in the format `(xmin, xmax, ymin, ymax)`.
    """
    if bin_size is None:
        raise ValueError("`bin_size` must be specified.")

    if (points_layer is None and table_layer is None) or (points_layer is not None and table_layer is not None):
        raise ValueError("Specify exactly one of `points_layer` or `table_layer`.")

    # Load transcripts
    if points_layer is not None:
        # TODO we should first subset the genes and then run compute
        df = sdata.points[points_layer].compute()

        if genes is not None:
            if name_gene_column is None:
                raise ValueError("name_gene_column must be provided if filtering genes.")

            if isinstance(genes, str):
                genes = [genes]

            df = df[df[name_gene_column].isin(genes)]

            if df.empty:
                raise ValueError("No transcripts found for specified gene(s).")

        label = "Transcript Count"
        title = "Transcript Density"

    elif table_layer is not None:
        coords = sdata.tables[table_layer].obsm["spatial"]
        df = pd.DataFrame(coords, columns=["x", "y"])

        label = "Cell Count"
        title = "Cell Density"

    if crd is not None:
        xmin, xmax, ymin, ymax = crd
        df = df[(df["x"] >= xmin) & (df["x"] <= xmax) & (df["y"] >= ymin) & (df["y"] <= ymax)]
        if df.empty:
            raise ValueError("No data found in specified region.")
    else:
        xmin, xmax = df["x"].min(), df["x"].max()
        ymin, ymax = df["y"].min(), df["y"].max()

    x = df["x"].to_numpy()
    y = df["y"].to_numpy()

    # Create 2D histogram
    x_edges = np.arange(xmin, xmax + bin_size, bin_size)
    y_edges = np.arange(ymin, ymax + bin_size, bin_size)

    heatmap, xedges, yedges = np.histogram2d(x, y, bins=[x_edges, y_edges])

    if smooth_sigma is not None:
        heatmap = gaussian_filter(heatmap, sigma=smooth_sigma)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        heatmap.T,
        origin="lower",
        cmap=cmap,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.75)
    cbar.set_label(label, fontsize=12)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.show()
