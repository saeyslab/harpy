from __future__ import annotations

import matplotlib.pyplot as plt
from spatialdata import SpatialData

from harpy.plot._qc_transcripts import qc_metric_histogram, qc_obs_scatter
from harpy.utils._keys import _CELLSIZE_KEY


def preprocess_transcriptomics(
    sdata: SpatialData,
    table_layer: str = "table_transcriptomics",
    instance_size_key: str = _CELLSIZE_KEY,
    bins_total_counts: int | None = 55,
    bins_n_genes_by_counts: int | None = 55,
    output: str | None = None,
) -> None:
    """
    Plot transcriptomics preprocessing QC figures.

    Parameters
    ----------
    sdata
        SpatialData object containing the spatial data and annotations.
    table_layer
        The table layer in `sdata`.
    instance_size_key
        The key in the :class:`~anndata.AnnData` table `.obs` that holds the size of the instances.
    bins_total_counts
        Number of bins for the ``total_counts`` histogram. If `None`, seaborn chooses the bins automatically.
    bins_n_genes_by_counts
        Number of bins for the ``n_genes_by_counts`` histogram. If `None`, seaborn chooses the bins automatically.
    output
        The file path prefix for the plots (default is None).

    See Also
    --------
    harpy.tb.preprocess_transcriptomics: preprocess.
    """
    _, axs = plt.subplots(1, 2, figsize=(10, 4))
    qc_metric_histogram(
        sdata,
        table_layer=table_layer,
        column="total_counts",
        display_column="Total Counts per Cell",
        ax=axs[0],
        bins=bins_total_counts,
        dataframe="obs",
        histplot_kwargs={"kde": False},
        title=None,
        show_median=True,
    )
    qc_metric_histogram(
        sdata,
        table_layer=table_layer,
        column="n_genes_by_counts",
        display_column="Detected Genes per Cell",
        dataframe="obs",
        ax=axs[1],
        bins=bins_n_genes_by_counts,
        histplot_kwargs={"kde": False},
        show_median=True,
        title=None,
    )
    plt.tight_layout()
    if output:
        plt.savefig(output + "_histogram.png")
    else:
        plt.show()
    plt.close()

    qc_obs_scatter(
        sdata,
        table_layer=table_layer,
        instance_size_key=instance_size_key,
        column="total_counts",
        display_column="Total Counts",
        display_instance_size_key=instance_size_key,
    )
    plt.tight_layout()
    if output:
        plt.savefig(output + "_size_count.png")
    else:
        plt.show()
    plt.close()
