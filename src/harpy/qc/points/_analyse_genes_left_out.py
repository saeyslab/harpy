from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger as log
from scipy.stats import pearsonr
from spatialdata import SpatialData
from spatialdata.models import TableModel

from harpy.image._image import _get_boundary, _get_spatial_element
from harpy.transformations._transformations import _identity_check_transformations_points
from harpy.utils._keys import _GENES_KEY, _RAW_COUNTS_KEY


def analyse_genes_left_out(
    sdata: SpatialData,
    labels_name: str,
    table_name: str,
    points_name: str = "transcripts",
    to_coordinate_system: str = "global",
    name_x: str = "x",
    name_y: str = "y",
    name_gene_column: str = _GENES_KEY,
    output: str | Path | None = None,
) -> pd.DataFrame:
    """
    Analyse and visualize the proportion of genes that were not assigned to an instance during point aggregation.

    Parameters
    ----------
    sdata
        Data containing spatial information for plotting.
    labels_name
        The labels element in `sdata` that contains the segmentation masks.
        This labels element is used to calculate the crd (region of interest) that was used in the segmentation step,
        otherwise transcript counts in `points_name` of `sdata` (containing all transcripts)
        and the counts obtained via `sdata.tables[ table_name ]` are not comparable.
        It is also used to select the cells in `sdata.tables[table_name]` that are linked to this `labels_name` via the region key.
    table_name
        The table element in `sdata` on which to perform analysis.
    points_name
        The points element in `sdata` containing transcript information.
    to_coordinate_system
        The coordinate system that holds `labels_name` and `points_name`.
        This should be the intrinsic coordinate system in pixels.
    name_x
        The column name representing the x-coordinate in `points_name`.
    name_y
        The column name representing the y-coordinate in `points_name`.
    name_gene_column
        The column name representing the gene name in `points_name`.
    output
        The path to save the generated plots. If None, plots will be shown directly using plt.show().

    Returns
    -------
    :class:`pandas.DataFrame` containing information about the proportion of transcripts kept for each gene,
    raw counts (i.e. obtained from `points_name` of `sdata`), and the log of raw counts.

    Raises
    ------
    AttributeError
        If the provided `sdata` does not contain the necessary attributes (i.e., 'labels' or 'points').

    Notes
    -----
    This function produces two plots:
        - A scatter plot of the log of raw gene counts vs. the proportion of transcripts kept.
        - A regression plot for the same data with Pearson correlation coefficients.

    The function also prints the ten genes with the highest proportion of transcripts filtered out.

    See Also
    --------
    harpy.tb.aggregate_points

    Examples
    --------
    .. code-block:: python

        import harpy as hp

        sdata = hp.datasets.xenium_human_ovarian_cancer(subset=True)
        hp.qc.analyse_genes_left_out(
            sdata,
            labels_name="cell_labels_global",
            points_name="transcripts_global",
            table_name="table_global",
        )
    """
    if not hasattr(sdata, "labels"):
        raise AttributeError(
            "Provided SpatialData object does not have the attribute 'labels', please run segmentation step before using this function."
        )

    if not hasattr(sdata, "points"):
        raise AttributeError(
            "Provided SpatialData object does not have the attribute 'points', please run point aggregation before using this function."
        )

    if not np.issubdtype(sdata.tables[table_name].X.dtype, np.integer):
        log.warning(
            f"The count matrix of the provided table element '{table_name}', seems to be of type '{sdata.tables[table_name].X.dtype}', "
            "which could indicate that the analysis is being run on normalized counts, "
            "please consider running this analysis before the counts in the AnnData object "
            "are normalized (i.e. on the raw counts)."
        )

    if labels_name not in [*sdata.labels]:
        raise ValueError(f"labels_name '{labels_name}' is not a labels element in `sdata`.")

    se = _get_spatial_element(sdata, element_name=labels_name)
    crd = _get_boundary(se, to_coordinate_system=to_coordinate_system)

    region_key = sdata.tables[table_name].uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY]
    adata = sdata.tables[table_name][sdata.tables[table_name].obs[region_key] == labels_name]

    ddf = sdata.points[points_name]
    _identity_check_transformations_points(ddf, to_coordinate_system=to_coordinate_system)
    ddf = ddf.query(f"{crd[0]} <= {name_x} < {crd[1]} and {crd[2]} <= {name_y} < {crd[3]}")

    _raw_counts = ddf.groupby(name_gene_column, observed=True).size().compute()
    missing_indices = adata.var.index.difference(_raw_counts.index)

    if not missing_indices.empty:
        raise ValueError(
            f"There are genes found in '.var' of table element '{table_name}' that are not found in the points element '{points_name}'. Please verify that 'harpy.tb.aggregate_points' was called with the correct points element."
        )

    raw_counts = _raw_counts[adata.var.index]

    filtered = pd.DataFrame(np.array(adata.X.sum(axis=0)).flatten() / raw_counts)
    filtered = filtered.rename(columns={0: "proportion_kept"})
    filtered[_RAW_COUNTS_KEY] = raw_counts
    filtered[f"log_{_RAW_COUNTS_KEY}"] = np.log(filtered[_RAW_COUNTS_KEY])

    sns.scatterplot(data=filtered, y="proportion_kept", x=f"log_{_RAW_COUNTS_KEY}")
    plt.axvline(filtered[f"log_{_RAW_COUNTS_KEY}"].median(), color="green", linestyle="dashed")
    plt.axhline(filtered["proportion_kept"].median(), color="red", linestyle="dashed")
    plt.xlim(left=-0.5, right=filtered[f"log_{_RAW_COUNTS_KEY}"].quantile(0.99))

    if output:
        plt.savefig(f"{output}_0", bbox_inches="tight")
    else:
        plt.show()
    plt.close()

    r, p = pearsonr(filtered[f"log_{_RAW_COUNTS_KEY}"], filtered["proportion_kept"])
    sns.regplot(x=f"log_{_RAW_COUNTS_KEY}", y="proportion_kept", data=filtered)
    ax = plt.gca()
    ax.text(0.7, 0.9, f"r={r:.2f}, p={p:.2g}", transform=ax.transAxes)
    plt.axvline(filtered[f"log_{_RAW_COUNTS_KEY}"].median(), color="green", linestyle="dashed")
    plt.axhline(filtered["proportion_kept"].median(), color="red", linestyle="dashed")

    if output:
        plt.savefig(f"{output}_1", bbox_inches="tight")
    else:
        plt.show()
    plt.close()

    log.info(
        f"The ten genes with the highest proportion of transcripts filtered out in the "
        f"region of interest ([x_min,x_max,y_min,y_max]={crd}):\n"
        f"{filtered.sort_values(by='proportion_kept').iloc[0:10, 0:2]}"
    )

    return filtered
