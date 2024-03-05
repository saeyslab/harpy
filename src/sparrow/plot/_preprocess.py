from typing import Optional

import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns
from spatialdata import SpatialData

from sparrow.utils._keys import _CELLSIZE_KEY


def preprocess_transcriptomics(sdata: SpatialData, output: Optional[str] = None) -> None:
    """Function plots the size of the nucleus/cell related to the counts."""
    sc.pl.pca(
        sdata.table,
        color="total_counts",
        show=False,
        title="PC plot colored by total counts",
    )
    if output:
        plt.savefig(output + "_total_counts_pca.png")
        plt.close()
    else:
        plt.show()
    plt.close()
    sc.pl.pca(
        sdata.table,
        color=_CELLSIZE_KEY,
        show=False,
        title="PC plot colored by object size",
    )
    if output:
        plt.savefig(output + f"_{_CELLSIZE_KEY}_pca.png")
        plt.close()
    else:
        plt.show()
    plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(15, 4))
    sns.histplot(sdata.table.obs["total_counts"], kde=False, ax=axs[0])
    sns.histplot(sdata.table.obs["n_genes_by_counts"], kde=False, bins=55, ax=axs[1])
    if output:
        plt.savefig(output + "_histogram.png")
    else:
        plt.show()
    plt.close()

    fig, ax = plt.subplots()
    plt.scatter(sdata.table.obs[_CELLSIZE_KEY], sdata.table.obs["total_counts"])
    ax.set_title(f"{_CELLSIZE_KEY} vs Transcripts Count")
    ax.set_xlabel(_CELLSIZE_KEY)
    ax.set_ylabel("Total Counts")
    if output:
        plt.savefig(output + "_size_count.png")
    else:
        plt.show()
    plt.close()
