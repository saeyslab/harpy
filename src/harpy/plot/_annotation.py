from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from spatialdata import SpatialData

from harpy.plot._plot import plot_shapes
from harpy.utils._keys import _ANNOTATION_KEY, _CLEANLINESS_KEY, _UNKNOWN_CELLTYPE_KEY


def score_genes(
    sdata: SpatialData,
    table_layer: str,
    celltypes: list[str],
    img_layer: str | None = None,
    shapes_layer: str = "segmentation_mask_boundaries",
    crd: tuple[int, int, int, int] | None = None,
    filter_index: int | None = None,
    output: str | None = None,
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
    sdata
        Data containing spatial information for plotting.
    table_layer
        The table layer in `sdata` to visualize.
    celltypes: list[str]
        list of celltypes to plot.
    img_layer
        Image layer to be plotted. If not provided, the last image layer in `sdata` will be used.
    shapes_layer
        Name of the layer containing segmentation mask boundaries, by default "segmentation_mask_boundaries".
    crd
        The coordinates for a region of interest in the format (xmin, xmax, ymin, ymax). Only used for plotting purposes.
    filter_index
        Index used to filter leiden clusters when plotting the heatmap. Only leiden clusters >= filter index will be plotted.
    output
        Filepath to save the plots. If not provided, plots will be displayed without being saved.

    Returns
    -------
    None

    Notes
    -----
    This function uses `scanpy` for plotting and may save multiple plots based on the output parameter.
    """
    celltypes = [element for element in celltypes if element != _UNKNOWN_CELLTYPE_KEY]

    if img_layer is None:
        img_layer = [*sdata.images][-1]

    # Custom colormap:
    colors = np.concatenate((plt.get_cmap("tab20c")(np.arange(20)), plt.get_cmap("tab20b")(np.arange(20))))
    colors = [mpl.colors.rgb2hex(colors[j * 4 + i]) for i in range(4) for j in range(10)]

    # Plot cleanliness and leiden next to annotation
    sc.pl.umap(sdata.tables[table_layer], color=[_CLEANLINESS_KEY, _ANNOTATION_KEY], show=False)

    if output:
        plt.savefig(output + f"_{_CLEANLINESS_KEY}_{_ANNOTATION_KEY}", bbox_inches="tight")
    else:
        plt.show()
    plt.close()

    sc.pl.umap(sdata.tables[table_layer], color=["leiden", _ANNOTATION_KEY], show=False)

    if output:
        plt.savefig(output + f"_leiden_{_ANNOTATION_KEY}", bbox_inches="tight")
    else:
        plt.show()
    plt.close()

    # Plot annotation and cleanliness columns of sdata.tables[table_layer] (AnnData) object
    sdata.tables[table_layer].uns[f"{_ANNOTATION_KEY}_colors"] = colors
    plot_shapes(
        sdata=sdata,
        img_layer=img_layer,
        shapes_layer=shapes_layer,
        table_layer=table_layer,
        column=_ANNOTATION_KEY,
        crd=crd,
        output=output + f"_{_ANNOTATION_KEY}" if output else None,
    )

    # Plot heatmap of celltypes and filtered celltypes based on filter index
    sc.pl.heatmap(
        sdata.tables[table_layer],
        var_names=celltypes,
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
            sdata.tables[table_layer][
                sdata.tables[table_layer].obs.leiden.isin(
                    [str(index) for index in range(filter_index, len(sdata.tables[table_layer].obs.leiden))]
                )
            ],
            var_names=celltypes,
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
