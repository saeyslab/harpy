from typing import Optional

import matplotlib.pyplot as plt
import scanpy as sc
from spatialdata import SpatialData


def cluster(sdata: SpatialData, table_layer: str, key_added: str = "leiden", output: Optional[str] = None) -> None:
    """
    Visualize clusters.

    Plot the Leiden clusters on a UMAP (using `scanpy.pl.umap`),
    and show the most differentially expressed genes/channels for each cluster on a second plot (using `scanpy.pl.rank_genes_group`), if "rank_genes_groups" is in `sdata.tables[table_layer].uns.keys()`.

    Parameters
    ----------
    sdata : SpatialData
        The SpatialData object containing the analyzed data.
    table_layer: str
        The table layer in `sdata` to visualize.
    key_added: str, optional
        name of the column in `sdata.tables[table_layer].obs` that contains the cluster id.
    output : str or None, optional
        The file path prefix for the plots (default is None).
        If provided, the plots will be saved to the specified output file path with "_umap.png"
        and "_rank_genes_groups.png" as suffixes.
        If None, the plots will be displayed directly without saving.

    Returns
    -------
    None
    """
    # Plot clusters on a UMAP
    sc.pl.umap(sdata.tables[table_layer], color=[key_added], show=not output)
    if output:
        plt.savefig(output + "_umap.png", bbox_inches="tight")
        plt.close()

    # Plot the highly differential genes for each cluster
    if "rank_genes_groups" in sdata.tables[table_layer].uns.keys():
        sc.pl.rank_genes_groups(sdata.tables[table_layer], n_genes=8, sharey=False, show=False)
        if output:
            plt.savefig(output + "_rank_genes_groups.png", bbox_inches="tight")
            plt.close()
