from __future__ import annotations

import matplotlib.pyplot as plt
from spatialdata import SpatialData

from harpy.qc._qc_transcripts import metric_histogram, obs_scatter
from harpy.utils._keys import _CELLSIZE_KEY


def preprocess_transcriptomics(
    sdata: SpatialData,
    table_name: str = "table_transcriptomics",
    instance_size_key: str = _CELLSIZE_KEY,
    bins_total_counts: int | None = 55,
    bins_n_genes_by_counts: int | None = 55,
    output: str | None = None,
) -> None:
    """
    Plot transcriptomics preprocessing QC figures.

    This function is read-only and expects QC metrics to already be present on the selected table,
    typically after running :func:`scanpy.pp.calculate_qc_metrics` during preprocessing.

    Parameters
    ----------
    sdata
        SpatialData object containing the spatial data and annotations.
    table_name
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
    metric_histogram(
        sdata,
        table_name=table_name,
        column="total_counts",
        display_column="Total Counts per Cell",
        ax=axs[0],
        bins=bins_total_counts,
        dataframe="obs",
        histplot_kwargs={"kde": False},
        title=None,
        show_median=True,
    )
    metric_histogram(
        sdata,
        table_name=table_name,
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

    obs_scatter(
        sdata,
        table_name=table_name,
        column_x=instance_size_key,
        column_y="total_counts",
        display_column_x=instance_size_key,
        display_column_y="Total Counts",
    )
    plt.tight_layout()
    if output:
        plt.savefig(output + "_size_count.png")
    else:
        plt.show()
    plt.close()
