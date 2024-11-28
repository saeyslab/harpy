from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import squidpy as sq
from spatialdata import SpatialData

from sparrow.utils._keys import _ANNOTATION_KEY


def nhood_enrichment(
    sdata: SpatialData,
    table_layer: str,
    celltype_column: str = _ANNOTATION_KEY,
    output: str | None = None,
) -> None:
    """
    Plot the neighborhood enrichment across cell-type annotations.

    Enrichment is shown in a hierarchically clustered heatmap. Each entry in the heatmap indicates
    if the corresponding cluster pair (or cell-type pair) is over-represented or over-depleted for node-node
    interactions in the spatial connectivity graph.

    Parameters
    ----------
    sdata
        The SpatialData object containing the data for analysis.
    table_layer
        The table layer in `sdata` to visualize.
    celltype_column
        The column name in the SpatialData object's table that specifies the cell type annotations.
        The default value is `_ANNOTATION_KEY`.
    output
        If provided, the plot will be displayed and also saved to a file with the specified filename.
        If None, the plot will be displayed directly without saving.

    Returns
    -------
    None

    Notes
    -----
    See https://www.nature.com/articles/s41592-021-01358-2 for details on the permutation-based
    neighborhood enrichment score.

    See Also
    --------
    sparrow.tb.nhood_enrichment : Calculate neighborhood enrichment.
    """
    # remove 'nan' values.
    tmp = sdata.tables[table_layer].uns[f"{celltype_column}_nhood_enrichment"]["zscore"]
    sdata.tables[table_layer].uns[f"{celltype_column}_nhood_enrichment"]["zscore"] = np.nan_to_num(tmp)

    sq.pl.nhood_enrichment(sdata.tables[table_layer], cluster_key=celltype_column, method="ward")

    # Save the plot to ouput
    if output:
        plt.savefig(output, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
