from __future__ import annotations

import matplotlib.pyplot as plt
import scanpy as sc
from spatialdata import SpatialData

from harpy.plot._plot import plot_shapes
from harpy.utils._keys import _ANNOTATION_KEY


def cluster_cleanliness(
    sdata: SpatialData,
    table_layer: str,
    img_layer: str | None = None,
    shapes_layer: str = "segmentation_mask_boundaries",
    crd: tuple[int, int, int, int] | None = None,
    color_dict: dict | None = None,
    celltype_column: str = _ANNOTATION_KEY,
    output: str | None = None,
) -> None:
    """
    Generate plots that allow assessing the "cleanliness" or accuracy of the cell clustering:

    - a barplot with a bar for each cluster, showing the composition by cell type of that cluster;
    - a UMAP with cells colored by cell type;
    - an image of the tissue with cells colored by cell type.

    Parameters
    ----------
    sdata
        SpatialData object containing the spatial data and annotations.
    table_layer
        The table layer in `sdata` to visualize.
    img_layer
        Name of the imgage layer in `sdata` (default is None).
    shapes_layer
        Name of the shapes layer in `sdata` object (default is "segmentation_mask_boundaries").
    crd
        An optional rectangle [xmin, xmax, ymin, ymax] (default is None).
        If specified, the tissue image will be cropped to this rectangle,
        otherwise the full image will be displayed.
    color_dict
        Custom colormap dictionary for coloring cell types in the barplot.
    celltype_column
        Name of the column in `sdata.tables[table_layer]` containing cell type annotations (default is `_ANNOTATION_KEY`).
    output
        The file path prefix for the plots (default is None).
        If provided, the plots will be saved to the specified output file path with "_barplot.png",
        "_{celltype_column}.png", "_{celltype_column}_crop.png" and "_{celltype_column}_umap.png" as suffixes.
        If None, the plots will be displayed directly without saving.

    Returns
    -------
    None
    """
    # Barplot with cell type composition of the clusters.
    stacked = (
        sdata.tables[table_layer]
        .obs.groupby(["leiden", celltype_column], as_index=False)
        .size()
        .pivot(index="leiden", columns=celltype_column)
        .fillna(0)
    )
    stacked_norm = stacked.div(stacked.sum(axis=1), axis=0)
    stacked_norm.columns = list(sdata.tables[table_layer].obs[_ANNOTATION_KEY].cat.categories)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    if color_dict:
        stacked_norm.plot(kind="bar", stacked=True, ax=fig.gca(), color=color_dict)
    else:
        stacked_norm.plot(kind="bar", stacked=True, ax=fig.gca())
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_yaxis().set_ticks([])
    plt.xlabel("Clusters")
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize="large")

    if output:
        fig.savefig(output + "_barplot.png", bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)

    # Tissue image with cells colored by cell type.
    plot_shapes(
        sdata=sdata,
        table_layer=table_layer,
        img_layer=img_layer,
        column=celltype_column,
        alpha=0.8,
        shapes_layer=shapes_layer,
        output=output + f"_{celltype_column}" if output else None,
    )

    plot_shapes(
        sdata=sdata,
        table_layer=table_layer,
        img_layer=img_layer,
        column=celltype_column,
        crd=crd,
        alpha=0.8,
        shapes_layer=shapes_layer,
        output=output + f"_{celltype_column}_crop" if output else None,
    )

    # UMAP plot with cells colored by cell type.
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    sc.pl.umap(
        sdata.tables[table_layer],
        color=[celltype_column],
        ax=ax,
        show=not output,
        size=300000 / sdata.tables[table_layer].shape[0],
    )
    ax.axis("off")

    if output:
        fig.savefig(output + f"_{celltype_column}_umap.png", bbox_inches="tight")
    else:
        plt.show()
    plt.close()
