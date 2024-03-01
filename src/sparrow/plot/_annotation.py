from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from spatialdata import SpatialData

from sparrow.image._image import _get_boundary, _get_spatial_element
from sparrow.plot._plot import plot_shapes
from sparrow.utils._keys import _ANNOTATION_KEY, _CLEANLINESS_KEY


def score_genes(
    sdata: SpatialData,
    scoresper_cluster: pd.DataFrame,
    img_layer: Optional[str] = None,
    shapes_layer: str = "segmentation_mask_boundaries",
    crd=None,
    filter_index: Optional[int] = None,
    output: Optional[str] = None,
) -> None:
    """
    Function generates following plots:

    - umap of assigned celltype next to umap of calculated cleanliness.
    - umap of assigned celltype next to umap of assigned leiden cluster.
    - assigned celltype for all cells in region of interest (crd).
    - a heatmap of the assigned leiden cluster for each cell type.
    - a heatmap of the assigned leiden cluster for each cell type, with leiden cluster >= filter_index.

    Parameters
    ----------
    sdata : SpatialData
        Data containing spatial information for plotting.
    scoresper_cluster : pd.DataFrame
        Index:
            cells: The index corresponds to indivdual cells ID's.
        Columns:
            celltypes (as provided via the markers file).
        Values:
            Score obtained using the scanpy's score_genes function for each celltype and for each cell.
    img_layer : str, optional
        Image layer to be plotted. If not provided, the last image layer in `sdata` will be used.
    shapes_layer : str, optional
        Name of the layer containing segmentation mask boundaries, by default "segmentation_mask_boundaries".
    crd : tuple of int, optional
        The coordinates for a region of interest in the format (xmin, xmax, ymin, ymax). Only used for plotting purposes.
    filter_index : int or None, optional
        Index used to filter leiden clusters when plotting the heatmap. Only leiden clusters >= filter index will be plotted.
    output : str or None, optional
        Filepath to save the plots. If not provided, plots will be displayed without being saved.

    Returns
    -------
    None

    Notes
    -----
    This function uses `scanpy` for plotting and may save multiple plots based on the output parameter.
    """
    if img_layer is None:
        img_layer = [*sdata.images][-1]

    if crd is None:
        se = _get_spatial_element(sdata, layer=img_layer)
        crd = _get_boundary(se)

    # Custom colormap:
    colors = np.concatenate((plt.get_cmap("tab20c")(np.arange(20)), plt.get_cmap("tab20b")(np.arange(20))))
    colors = [mpl.colors.rgb2hex(colors[j * 4 + i]) for i in range(4) for j in range(10)]

    # Plot cleanliness and leiden next to annotation
    sc.pl.umap(sdata.table, color=[_CLEANLINESS_KEY, _ANNOTATION_KEY], show=False)

    if output:
        plt.savefig(output + f"_{_CLEANLINESS_KEY}_{_ANNOTATION_KEY}", bbox_inches="tight")
    else:
        plt.show()
    plt.close()

    sc.pl.umap(sdata.table, color=["leiden", _ANNOTATION_KEY], show=False)

    if output:
        plt.savefig(output + f"_leiden_{_ANNOTATION_KEY}", bbox_inches="tight")
    else:
        plt.show()
    plt.close()

    # Plot annotation and cleanliness columns of sdata.table (AnnData) object
    sdata.table.uns[f"{_ANNOTATION_KEY}_colors"] = colors
    plot_shapes(
        sdata=sdata,
        column=_ANNOTATION_KEY,
        crd=crd,
        img_layer=img_layer,
        shapes_layer=shapes_layer,
        output=output + f"_{_ANNOTATION_KEY}" if output else None,
    )

    # Plot heatmap of celltypes and filtered celltypes based on filter index
    sc.pl.heatmap(
        sdata.table,
        var_names=scoresper_cluster.columns.values,
        groupby="leiden",
        show=False,
    )

    if output:
        plt.savefig(output + "_leiden_heatmap", bbox_inches="tight")
    else:
        plt.show()
    plt.close()

    if filter_index:
        sc.pl.heatmap(
            sdata.table[
                sdata.table.obs.leiden.isin([str(index) for index in range(filter_index, len(sdata.table.obs.leiden))])
            ],
            var_names=scoresper_cluster.columns.values,
            groupby="leiden",
            show=False,
        )

        if output:
            plt.savefig(
                output + f"_leiden_heatmap_filtered_{filter_index}",
                bbox_inches="tight",
            )
        else:
            plt.show()
        plt.close()
