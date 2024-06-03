from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from spatialdata import SpatialData

from sparrow.image._image import _get_boundary, _get_spatial_element
from sparrow.plot import plot_shapes
from sparrow.utils._keys import _RAW_COUNTS_KEY, _REGION_KEY
from sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def analyse_genes_left_out(
    sdata: SpatialData,
    labels_layer: str,
    table_layer: str,
    points_layer: str = "transcripts",
    name_x: str = "x",
    name_y: str = "y",
    name_gene_column: str = "gene",
    output: str | Path | None = None,
) -> pd.DataFrame:
    """
    Analyse and visualize the proportion of genes that could not be assigned to a cell during allocation step.

    Parameters
    ----------
    sdata
        Data containing spatial information for plotting.
    labels_layer : str
        The layer in `sdata` that contains the segmentation masks, by default "segmentation_mask".
        This layer is used to calculate the crd (region of interest) that was used in the segmentation step,
        otherwise transcript counts in `points_layer` of `sdata` (containing all transcripts)
        and the counts obtained via `sdata.tables[ table_layer ]` are not comparable.
        It is also used to select the cells in `sdata.tables[table_layer]` that are linked to this `labels_layer` via the _REGION_KEY.
    table_layer: str
        The table layer in `sdata` on which to perform analysis.
    points_layer : str, optional
        The layer in `sdata` containing transcript information, by default "transcripts".
    name_x : str, optional
        The column name representing the x-coordinate in `points_layer`, by default "x".
    name_y
        The column name representing the y-coordinate in `points_layer`, by default "y".
    name_gene_column
        The column name representing the gene name in `points_layer`, by default "gene".
    output
        The path to save the generated plots. If None, plots will be shown directly using plt.show().

    Returns
    -------
    A DataFrame containing information about the proportion of transcripts kept for each gene,
    raw counts (i.e. obtained from `points_layer` of `sdata`), and the log of raw counts.

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
    sparrow.tb.allocate
    """
    # we need the segmentation_mask to calculate crd used during allocation step,
    # otherwise transcript counts in points layer of sdata (containing all transcripts)
    # and the counts obtained via sdata.tables[ table_layer ] are not comparable.
    if not hasattr(sdata, "labels"):
        raise AttributeError(
            "Provided SpatialData object does not have the attribute 'labels', please run segmentation step before using this function."
        )

    if not hasattr(sdata, "points"):
        raise AttributeError(
            "Provided SpatialData object does not have the attribute 'points', please run allocation step before using this function."
        )

    if sdata.tables[table_layer].raw is not None:
        log.warning(
            f"'sdata.tables[{table_layer}].raw' is not None. It seems that analysis is being run on an AnnData object containing normalized counts, "
            "please consider running this analysis before the counts in the AnnData object "
            "are normalized (i.e. on the raw counts)."
        )
    if labels_layer is None:
        labels_layer = [*sdata.labels][-1]

    if labels_layer not in [*sdata.labels]:
        raise ValueError(f"labels_layer '{labels_layer}' is not a labels layer in `sdata`.")

    se = _get_spatial_element(sdata, layer=labels_layer)
    crd = _get_boundary(se)

    adata = sdata.tables[table_layer][sdata.tables[table_layer].obs[_REGION_KEY] == labels_layer]

    ddf = sdata.points[points_layer]

    ddf = ddf.query(f"{crd[0]} <= {name_x} < {crd[1]} and {crd[2]} <= {name_y} < {crd[3]}")

    _raw_counts = ddf.groupby(name_gene_column).size().compute()

    missing_indices = adata.var.index.difference(_raw_counts.index)

    if not missing_indices.empty:
        raise ValueError(
            f"There are genes found in '.var' of table layer '{table_layer}' that are not found in the points layer '{points_layer}'. Please verify that allocation '(sp.tb.allocation)' is performed using the correct points layer."
        )

    raw_counts = _raw_counts[adata.var.index]

    filtered = pd.DataFrame(adata.X.sum(axis=0) / raw_counts)

    filtered = filtered.rename(columns={0: "proportion_kept"})
    filtered[_RAW_COUNTS_KEY] = raw_counts
    filtered[f"log_{_RAW_COUNTS_KEY}"] = np.log(filtered[_RAW_COUNTS_KEY])

    # first plot:

    sns.scatterplot(data=filtered, y="proportion_kept", x=f"log_{_RAW_COUNTS_KEY}")
    plt.axvline(filtered[f"log_{_RAW_COUNTS_KEY}"].median(), color="green", linestyle="dashed")
    plt.axhline(filtered["proportion_kept"].median(), color="red", linestyle="dashed")
    plt.xlim(left=-0.5, right=filtered[f"log_{_RAW_COUNTS_KEY}"].quantile(0.99))

    if output:
        plt.savefig(f"{output}_0", bbox_inches="tight")
    else:
        plt.show()
    plt.close()

    # second plot:

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


def transcript_density(
    sdata: SpatialData,
    img_layer: tuple[str, str] = ["raw_image", "transcript_density"],
    channel: int = 0,
    crd: tuple[int, int, int, int] | None = None,
    figsize: tuple[int, int] | None = None,
    output: str | Path | None = None,
) -> None:
    """
    Visualize the transcript density layer.

    This function wraps around the `plot_shapes` function to showcase transcript density.

    Parameters
    ----------
    sdata
        Data containing spatial information for plotting.
    img_layer
        A tuple where the first element indicates the base image layer and
        the second element indicates the transcript density.
    channel
        The channel of the image to be visualized.
        If the channel not in one of the images, the first available channel of the image will be plotted
    crd
        The coordinates for the region of interest in the format (xmin, xmax, ymin, ymax). If None, the entire image is considered, by default None.
    figsize
        The figure size for the visualization. If None, a default size will be used.
    output
        Path to save the output image. If None, the image will not be saved and will be displayed instead.

    Returns
    -------
    None

    Examples
    --------
    >>> sdata = SpatialData(...)
    >>> transcript_density(sdata, img_layer=["raw_img", "density"], crd=(2000,4000,2000,4000))

    See Also
    --------
    sparrow.im.transcript_density
    """
    plot_shapes(
        sdata,
        img_layer=img_layer,
        shapes_layer=None,
        channel=channel,
        crd=crd,
        figsize=figsize,
        output=output,
    )
