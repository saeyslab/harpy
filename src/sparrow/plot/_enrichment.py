from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import squidpy as sq
from spatialdata import SpatialData

from sparrow.table._table import _back_sdata_table_to_zarr
from sparrow.utils._keys import _ANNOTATION_KEY


def nhood_enrichment(
    sdata: SpatialData,
    celltype_column: str = _ANNOTATION_KEY,
    output: Optional[str] = None,
) -> None:
    """
    Plot the neighborhood enrichment across cell-type annotations.

    Enrichment is shown in a hierarchically clustered heatmap. Each entry in the heatmap indicates
    if the corresponding cluster pair (or cell-type pair) is over-represented or over-depleted for node-node
    interactions in the spatial connectivity graph.

    Parameters
    ----------
    sdata : SpatialData
        The SpatialData object containing the data for analysis.
    celltype_column : str, optional
        The column name in the SpatialData object's table that specifies the cell type annotations.
        The default value is `_ANNOTATION_KEY`.
    output : str or None, optional
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
    - tb.nhood_enrichment : Calculate neighborhood enrichment.
    """
    # remove 'nan' values from "adata.uns['annotation_nhood_enrichment']['zscore']"
    tmp = sdata.table.uns[f"{celltype_column}_nhood_enrichment"]["zscore"]
    sdata.table.uns[f"{celltype_column}_nhood_enrichment"]["zscore"] = np.nan_to_num(tmp)
    _back_sdata_table_to_zarr(sdata=sdata)

    sq.pl.nhood_enrichment(sdata.table, cluster_key=celltype_column, method="ward")

    # Save the plot to ouput
    if output:
        plt.savefig(output, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
