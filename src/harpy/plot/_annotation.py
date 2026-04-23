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
    table_name: str,
    celltypes: list[str],
    image_name: str | None = None,
    shapes_name: str = "segmentation_mask_boundaries",
    crd: tuple[int, int, int, int] | None = None,
    filter_index: int | None = None,
    celltype_column: str = _ANNOTATION_KEY,
    unknown_celltype_key: str = _UNKNOWN_CELLTYPE_KEY,
    cleanliness_key: str = _CLEANLINESS_KEY,
    output: str | None = None,
) -> None:
    """
    Function generates following plots:

    - umap of assigned celltype next to umap of calculated cleanliness.
    - umap of assigned celltype next to umap of assigned leiden cluster.
    - assigned celltype for all cells in region of interest (crd).
    - a heatmap of the assigned leiden cluster for each cell type.
    - a heatmap of the assigned leiden cluster for each cell type, with leiden cluster >= filter_index.

    This function is deprecated and will be removed in a future version.

    Parameters
    ----------
    sdata
        Data containing spatial information for plotting.
    table_name
        The table element in `sdata` to visualize.
    celltypes: list[str]
        list of celltypes to plot.
    image_name
        Image element to be plotted. If not provided, the last image element in `sdata` will be used.
    shapes_name
        Name of the shapes element containing segmentation mask boundaries, by default "segmentation_mask_boundaries".
    crd
        The coordinates for a region of interest in the format (xmin, xmax, ymin, ymax). Only used for plotting purposes.
    filter_index
        Index used to filter leiden clusters when plotting the heatmap. Only leiden clusters >= filter index will be plotted.
    celltype_column
        The column name in the `.obs` attribute of the :class:`anndata.AnnData` table where the cell types are stored.
    unknown_celltype_key
        The name reserved for cells that could not be assigned a specific cell type.
    cleanliness_key
        The column name in the `.obs` attribute of the :class:`anndata.AnnData` table where a score for the cleanliness of the predicted cell type is stored.
    output
        Filepath to save the plots. If not provided, plots will be displayed without being saved.

    Returns
    -------
    None

    Notes
    -----
    This function uses `scanpy` for plotting and may save multiple plots based on the output parameter.
    """
    celltypes = [element for element in celltypes if element != unknown_celltype_key]

    if image_name is None:
        image_name = [*sdata.images][-1]

    # Custom colormap:
    colors = np.concatenate((plt.get_cmap("tab20c")(np.arange(20)), plt.get_cmap("tab20b")(np.arange(20))))
    colors = [mpl.colors.rgb2hex(colors[j * 4 + i]) for i in range(4) for j in range(10)]

    # Plot cleanliness and leiden next to annotation
    sc.pl.umap(sdata.tables[table_name], color=[cleanliness_key, celltype_column], show=False)

    if output:
        plt.savefig(output + f"_{cleanliness_key}_{celltype_column}", bbox_inches="tight")
    else:
        plt.show()
    plt.close()

    sc.pl.umap(sdata.tables[table_name], color=["leiden", celltype_column], show=False)

    if output:
        plt.savefig(output + f"_leiden_{celltype_column}", bbox_inches="tight")
    else:
        plt.show()
    plt.close()

    # Plot annotation and cleanliness columns of sdata.tables[table_name] (AnnData) object
    sdata.tables[table_name].uns[f"{celltype_column}_colors"] = colors
    plot_shapes(
        sdata=sdata,
        image_name=image_name,
        shapes_name=shapes_name,
        table_name=table_name,
        column=celltype_column,
        crd=crd,
        output=output + f"_{celltype_column}" if output else None,
    )

    # Plot heatmap of celltypes and filtered celltypes based on filter index
    sc.pl.heatmap(
        sdata.tables[table_name],
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
            sdata.tables[table_name][
                sdata.tables[table_name].obs.leiden.isin(
                    [str(index) for index in range(filter_index, len(sdata.tables[table_name].obs.leiden))]
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
