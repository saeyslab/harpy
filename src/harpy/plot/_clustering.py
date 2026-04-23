from __future__ import annotations

import matplotlib.pyplot as plt
import scanpy as sc
from spatialdata import SpatialData


def cluster(sdata: SpatialData, table_name: str, key_added: str = "leiden", output: str | None = None) -> None:
    """
    Visualize clusters.

    .. deprecated:: 0.3.0
       `harpy.pl.cluster` is deprecated and will be removed in 0.4.0.

    Plot the clusters on a UMAP (using :func:`~scanpy.pl.umap`),
    and shows the most differentially expressed genes/channels for each cluster on a second plot (using :func:`~scanpy.pl.rank_genes_group`), if "rank_genes_groups" is in `sdata.tables[table_name].uns.keys()`.

    Parameters
    ----------
    sdata
        The SpatialData object containing the analyzed data.
    table_name
        The table element in `sdata` to visualize.
    key_added
        name of the column in `sdata.tables[table_name].obs` that contains the cluster id.
    output
        The file path prefix for the plots (default is None).
        If provided, the plots will be saved to the specified output file path with "_umap.png"
        and "_rank_genes_groups.png" as suffixes.
        If None, the plots will be displayed directly without saving.

    Returns
    -------
    None

    See Also
    --------
    harpy.tb.leiden: leiden clustering
    harpy.tb.kmeans: kmeans clustering
    """
    # Plot clusters on a UMAP
    sc.pl.umap(sdata.tables[table_name], color=[key_added], show=not output)
    if output:
        plt.savefig(output + "_umap.png", bbox_inches="tight")
        plt.close()

    # Plot the highly differential genes for each cluster
    if "rank_genes_groups" in sdata.tables[table_name].uns.keys():
        sc.pl.rank_genes_groups(sdata.tables[table_name], n_genes=8, sharey=False, show=False)
        if output:
            plt.savefig(output + "_rank_genes_groups.png", bbox_inches="tight")
            plt.close()
