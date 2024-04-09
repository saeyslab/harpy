from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame
from scipy.stats import pearsonr
from spatialdata import SpatialData

from sparrow.image._image import _get_boundary, _get_spatial_element
from sparrow.plot import plot_shapes
from sparrow.utils._keys import _REGION_KEY
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
) -> DataFrame:
    """
    Analyse and visualize the proportion of genes that could not be assigned to a cell during allocation step.

    Parameters
    ----------
    sdata : SpatialData
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
    name_y : str, optional
        The column name representing the y-coordinate in `points_layer`, by default "y".
    name_gene_column : str, optional
        The column name representing the gene name in `points_layer`, by default "gene".
    output : str or Path, optional
        The path to save the generated plots. If None, plots will be shown directly using plt.show().

    Returns
    -------
    DataFrame
        A DataFrame containing information about the proportion of transcripts kept for each gene,
        raw counts (i.e. obtained from `points_layer` of `sdata`), and the log of raw counts.

    Raises
    ------
    AttributeError
        If the provided `sdata` does not contain the necessary attributes (i.e., "labels" or "points").

    Notes
    -----
    This function produces two plots:
    1. A scatter plot of the log of raw gene counts vs. the proportion of transcripts kept.
    2. A regression plot for the same data with Pearson correlation coefficients.

    The function also prints the ten genes with the highest proportion of transcripts filtered out.

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
            "It seems that analysis is being run on AnnData object (sdata.tables[ table_layer ]) containing normalized counts, "
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

    raw_counts = ddf.groupby(name_gene_column).size().compute()[adata.var.index]

    filtered = pd.DataFrame(adata.X.sum(axis=0) / raw_counts)

    filtered = filtered.rename(columns={0: "proportion_kept"})
    filtered["raw_counts"] = raw_counts
    filtered["log_raw_counts"] = np.log(filtered["raw_counts"])

    # first plot:

    sns.scatterplot(data=filtered, y="proportion_kept", x="log_raw_counts")
    plt.axvline(filtered["log_raw_counts"].median(), color="green", linestyle="dashed")
    plt.axhline(filtered["proportion_kept"].median(), color="red", linestyle="dashed")
    plt.xlim(left=-0.5, right=filtered["log_raw_counts"].quantile(0.99))

    if output:
        plt.savefig(f"{output}_0", bbox_inches="tight")
    else:
        plt.show()
    plt.close()

    # second plot:

    r, p = pearsonr(filtered["log_raw_counts"], filtered["proportion_kept"])
    sns.regplot(x="log_raw_counts", y="proportion_kept", data=filtered)
    ax = plt.gca()
    ax.text(0.7, 0.9, f"r={r:.2f}, p={p:.2g}", transform=ax.transAxes)

    plt.axvline(filtered["log_raw_counts"].median(), color="green", linestyle="dashed")
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
    sdata: SpatialData
        Data containing spatial information for plotting.
    img_layer: Tuple[str, str], default=["raw_image", "transcript_density"]
        A tuple where the first element indicates the base image layer and
        the second element indicates the transcript density.
    channel: int, default=0
        The channel of the image to be visualized.
        If the channel not in one of the images, the first available channel of the image will be plotted
    crd: Optional[Tuple[int, int, int, int]], default=None
        The coordinates for the region of interest in the format (xmin, xmax, ymin, ymax). If None, the entire image is considered, by default None.
    figsize: Optional[Tuple[int, int]], default=None
        The figure size for the visualization. If None, a default size will be used.
    output: Optional[str | Path], default=None
        Path to save the output image. If None, the image will not be saved and will be displayed instead.

    Returns
    -------
    None

    Example:
    -------
    >>> sdata = SpatialData(...)
    >>> transcript_density(sdata, img_layer=["raw_img", "density"], crd=(2000,4000,2000,4000))
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
