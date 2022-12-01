""" This file holds all the general functions that are used to build up the pipeline and the notebooks. The functions are odered by their occurence in the pipeline from top to bottom."""

# %load_ext autoreload
# %autoreload 2
import warnings
from itertools import chain
from typing import List, Optional, Tuple

import cv2
import geopandas
import matplotlib
import matplotlib.colors as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import scanpy as sc
import seaborn as sns
import shapely
import squidpy as sq
import torch
from anndata import AnnData
from basicpy import BaSiC
from cellpose import models
from rasterio import features
from scipy import ndimage


def tilingCorrection(
    img: sq.im.ImageContainer,
    left_corner: Tuple[int, int] = None,
    size: Tuple[int, int] = None,
    tile_size: int = 2144,
) -> Tuple[sq.im.ImageContainer, np.ndarray]:
    """Returns the corrected image and the flatfield array

    This function corrects for the tiling effect that occurs in some image data for example the resolve dataset.
    The illumination within the tiles is adjusted, afterwards the tiles are connected as a whole image by inpainting the lines between the tiles.
    """

    # Create the tiles
    tiles = img.generate_equal_crops(size=2144, as_array="image")
    tiles = np.array([tile + 1 if ~np.any(tile) else tile for tile in tiles])

    # Measure the filters
    # BaSiC has no support for gpu devices, see https://github.com/peng-lab/BaSiCPy/issues/101
    basic = BaSiC(epsilon=1e-06)

    basic.fit(tiles)
    flatfield = basic.flatfield
    tiles_corrected = basic.transform(tiles)
    tiles_corrected = np.array(
        [tile + 1 if ~np.any(tile) else tile for tile in tiles_corrected]
    )

    # Stitch the tiles back together
    i_new = np.block(
        [
            list(tiles_corrected[i : i + (img.shape[1] // tile_size)])
            for i in range(0, len(tiles_corrected), img.shape[1] // tile_size)
        ]
    ).astype(np.uint16)

    img = sq.im.ImageContainer(i_new, layer="image")
    img.add_img(
        img.apply(
            lambda array: (array == 0).astype(np.uint8), new_layer="mask", copy=True
        ),
        layer="mask",
    )

    if size is not None and left_corner is not None:
        img = img.crop_corner(*left_corner, size)

    # Perform inpainting
    img = img.apply(
        {"0": cv2.inpaint},
        layer="image",
        drop=True,
        channel=0,
        copy=True,
        fn_kwargs={
            "inpaintMask": img.data.mask.squeeze().to_numpy(),
            "inpaintRadius": 55,
            "flags": cv2.INPAINT_NS,
        },
    )

    return img, flatfield


def tilingCorrectionPlot(
    img: np.ndarray, flatfield, img_orig: np.ndarray, output: str = None
) -> None:
    """Creates the plots based on the correction overlay and the original and corrected images."""

    # disable interactive mode
    if output:
        plt.ioff()

    # Tile correction overlay
    fig1, ax1 = plt.subplots(1, 1, figsize=(20, 10))
    ax1.imshow(flatfield, cmap="gray")
    ax1.set_title("Correction performed per tile")

    # Save the plot to ouput
    if output:
        plt.close(fig1)
        fig1.savefig(output + "0.png")

    # Original and corrected image
    fig2, ax2 = plt.subplots(1, 2, figsize=(20, 10))
    ax2[0].imshow(img, cmap="gray")
    ax2[0].set_title("Corrected image")
    ax2[1].imshow(img_orig, cmap="gray")
    ax2[1].set_title("Original image")

    # Save the plot to ouput
    if output:
        plt.close(fig2)
        fig2.savefig(output + "1.png")


def preprocessImage(
    img: sq.im.ImageContainer,
    contrast_clip: float = 2.5,
    size_tophat: int = None,
) -> sq.im.ImageContainer:
    """Returns the new image

    This function performs the preprocessing of the image.
    Contrast_clip indicates the input to the create_CLAHE function for histogram equalization.
    Size_tophat indicates the tophat filter size. If no tophat lfiter size is given, no tophat filter is applied. The recommendable size is 45.
    Small_size_vis indicates the coordinates of an optional zoom in plot to check the processing better.
    """

    # Apply tophat filter
    if size_tophat is not None:
        minimum_t = ndimage.minimum_filter(
            img.data.image.squeeze().to_numpy(), size_tophat
        )
        max_of_min_t = ndimage.maximum_filter(minimum_t, size_tophat)
        img = img.apply(
            {"0": lambda array: array - max_of_min_t},
            layer="image",
            drop=True,
            channel=0,
            copy=True,
        )

    # Enhance the contrast
    clahe = cv2.createCLAHE(clipLimit=contrast_clip, tileGridSize=(8, 8))
    img = img.apply({"0": clahe.apply}, layer="image", drop=True, channel=0, copy=True)

    return img


def preprocessImagePlot(
    img: np.ndarray,
    img_orig: np.ndarray,
    small_size_vis: List[int] = None,
    output: str = None,
) -> None:
    """Creates the plots based on the original and preprocessed image."""

    # disable interactive mode
    if output:
        plt.ioff()

    # Original and preprocessed image
    fig1, ax1 = plt.subplots(1, 2, figsize=(20, 10))
    ax1[0].imshow(img, cmap="gray")
    ax1[0].set_title("Corrected image")
    ax1[1].imshow(img_orig, cmap="gray")
    ax1[1].set_title("Original image")

    # Save the plot to ouput
    if output:
        plt.close(fig1)
        fig1.savefig(output + "0.png")

    # Plot small part of the images
    if small_size_vis is not None:
        fig2, ax2 = plt.subplots(1, 2, figsize=(20, 10))
        ax2[0].imshow(
            img[
                small_size_vis[0] : small_size_vis[1],
                small_size_vis[2] : small_size_vis[3],
            ],
            cmap="gray",
        )
        ax2[0].set_title("Corrected image")
        ax2[1].imshow(
            img_orig[
                small_size_vis[0] : small_size_vis[1],
                small_size_vis[2] : small_size_vis[3],
            ],
            cmap="gray",
        )
        ax2[1].set_title("Original image")

        # Save the plot to ouput
        if output:
            plt.close(fig2)
            fig2.savefig(output + "1.png")


def segmentation(
    img: sq.im.ImageContainer,
    device: str = "cpu",
    min_size: int = 80,
    flow_threshold: float = 0.6,
    diameter: int = 55,
    cellprob_threshold: int = 0,
    model_type: str = "nuclei",
    channels: List[int] = [0, 0],
) -> Tuple[np.ndarray, np.ndarray, geopandas.GeoDataFrame, sq.im.ImageContainer]:
    """Returns the segmentation masks, the image masks and the polygons

    This function segments the data, using the cellpose algorithm.
    Img is the input image.
    You can define your device by setting the device parameter.
    Min_size indicates the minimal amount of pixels in a mask.
    The flow_threshold indicates someting about the shape of the masks, if you increase it, more masks with less orund shapes will be accepted.
    The diameter is a very important parameter to estimate, in the best case you estimate it yourself. It indicates the mean expected diameter of your dataset.
    If you put None in diameter, the model will estimate it automatically.
    Mask_threshold indicates how many of the possible masks are kept. Making it smaller (up to -6), will give you more masks, bigger is less masks.
    When an RGB image is given an input, the R channel is expected to have the nuclei, and the blue channel the membranes.
    When whole cell segmentation needs to be performed, model_type=cyto, otherwise, model_type=nuclei.
    """

    channels = np.array(channels)

    # Perform cellpose segmentation
    model = models.Cellpose(device=torch.device(device), model_type=model_type)
    masks, _, _, _ = model.eval(
        img.data.image.squeeze().to_numpy(),
        diameter=diameter,
        channels=channels,
        min_size=min_size,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )
    mask_i = np.ma.masked_where(masks == 0, masks)

    # Create the polygon shapes of the different cells
    polygons = mask_to_polygons_layer(masks)
    polygons["border_color"] = polygons.geometry.map(border_color)
    polygons["linewidth"] = polygons.geometry.map(linewidth)
    polygons["color"] = polygons.geometry.map(color)
    polygons["cells"] = polygons.index
    polygons = polygons.dissolve(by="cells")

    img.add_img(masks, layer="segment_cellpose")

    return masks, mask_i, polygons, img


def segmentationPlot(
    img: np.ndarray,
    mask_i: np.ndarray,
    polygons: geopandas.GeoDataFrame,
    channels: List[int] = [0, 0],
    small_size_vis: List[int] = None,
    output: str = None,
) -> None:
    """Creates the plots based on the original image as well as the image masks."""

    # disable interactive mode
    if output:
        plt.ioff()

    # Select correct layer of the image
    if sum(channels) != 0:
        img = img[0, :, :]

    # Show only small part of the image
    if small_size_vis is not None:
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].imshow(
            img[
                small_size_vis[0] : small_size_vis[1],
                small_size_vis[2] : small_size_vis[3],
            ],
            cmap="gray",
        )

        ax[1].imshow(
            img[
                small_size_vis[0] : small_size_vis[1],
                small_size_vis[2] : small_size_vis[3],
            ],
            cmap="gray",
        )
        ax[1].imshow(
            mask_i[
                small_size_vis[0] : small_size_vis[1],
                small_size_vis[2] : small_size_vis[3],
            ],
            cmap="jet",
        )

    # Show the full image
    else:
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].imshow(img, cmap="gray")
        ax[1].imshow(img, cmap="gray")
        polygons.plot(
            ax=ax[1],
            edgecolor="white",
            linewidth=polygons.linewidth,
            alpha=0.5,
            legend=True,
            color="red",
        )

    # Save the plot to ouput
    if output:
        plt.close(fig)
        fig.savefig(output + ".png")


def mask_to_polygons_layer(mask: np.ndarray) -> geopandas.GeoDataFrame:
    """Returns the polygons as GeoDataFrame

    This function converts the mask to polygons.
    https://rocreguant.com/convert-a-mask-into-a-polygon-for-images-using-shapely-and-rasterio/1786/
    """

    all_polygons = []
    all_values = []

    # Extract the polygons from the mask
    for shape, value in features.shapes(
        mask.astype(np.int16),
        mask=(mask > 0),
        transform=rasterio.Affine(1.0, 0, 0, 0, 1.0, 0),
    ):
        all_polygons.append(shapely.geometry.shape(shape))
        all_values.append(int(value))

    return geopandas.GeoDataFrame(dict(geometry=all_polygons), index=all_values)


def color(_) -> matplotlib.colors.Colormap:
    """Select random color from set1 colors."""
    return plt.get_cmap("Set1")(np.random.choice(np.arange(0, 18)))


def border_color(r: bool) -> matplotlib.colors.Colormap:
    """Select border color from tab10 colors or preset color (1, 1, 1, 1) otherwise."""
    return plt.get_cmap("tab10")(3) if r else (1, 1, 1, 1)


def linewidth(r: bool) -> float:
    """Select linewidth 1 if true else 0.5."""
    return 1 if r else 0.5


def create_adata_quick(
    path: str, ic: sq.im.ImageContainer, masks: np.ndarray, library_id: str = "melanoma"
) -> AnnData:
    """Returns the AnnData object with transcript and polygon data.

    This function creates the polygon shapes from the mask and adjusts the colors and linewidth.
    The transcripts are read from the csv file in path, all transcripts within cells are assigned.
    Only cells with transcripts are retained.
    """

    # Create the polygon shapes of the different cells
    polygons = mask_to_polygons_layer(masks)
    polygons["geometry"] = polygons["geometry"].translate(
        float(ic.data.attrs["coords"].x0), float(ic.data.attrs["coords"].y0)
    )

    polygons["border_color"] = polygons.geometry.map(border_color)
    polygons["linewidth"] = polygons.geometry.map(linewidth)
    polygons["color"] = polygons.geometry.map(color)
    polygons["cells"] = polygons.index
    polygons = polygons.dissolve(by="cells")

    # Allocate the transcripts
    in_df = pd.read_csv(path, delimiter="\t", header=None)
    # Changed for subset
    df = in_df[
        (ic.data.attrs["coords"].y0 < in_df[1])
        & (in_df[1] < masks.shape[0] + ic.data.attrs["coords"].y0)
        & (ic.data.attrs["coords"].x0 < in_df[0])
        & (in_df[0] < masks.shape[1] + ic.data.attrs["coords"].x0)
    ]

    df["cells"] = masks[
        df[1].values - ic.data.attrs["coords"].y0,
        df[0].values - ic.data.attrs["coords"].x0,
    ]

    # Calculate the mean of the transcripts for every cell
    coordinates = df.groupby(["cells"]).mean().iloc[:, [0, 1]]
    cell_counts = df.groupby(["cells", 3]).size().unstack(fill_value=0)

    # Create the anndata object
    adata = AnnData(cell_counts[cell_counts.index != 0])
    coordinates.index = coordinates.index.map(str)
    adata.obsm["spatial"] = coordinates[coordinates.index != "0"].values

    # Add the polygons to the anndata object
    polygons_f = polygons[
        np.isin(polygons.index.values, list(map(int, adata.obs.index.values)))
    ]
    polygons_f.index = list(map(str, polygons_f.index))
    adata.obsm["polygons"] = polygons_f
    adata.obs["cell_ID"] = [int(x) for x in adata.obsm["polygons"].index]

    # Add the figure to the anndata object
    adata.uns["spatial"] = {library_id: {}}
    adata.uns["spatial"][library_id]["images"] = {}
    adata.uns["spatial"][library_id]["images"] = {
        "hires": ic.data.image.squeeze().to_numpy()
    }
    adata.uns["spatial"][library_id]["scalefactors"] = {
        "tissue_hires_scalef": 1,
        "spot_diameter_fullres": 75,
    }
    adata.uns["spatial"][library_id]["segmentation"] = masks.astype(np.uint16)
    adata.uns["spatial"][library_id]["points"] = AnnData(in_df.values[:, 0:2])
    adata.uns["spatial"][library_id]["points"].obs = pd.DataFrame(
        {"gene": in_df.values[:, 3]}
    )

    return adata


def plot_shapes(
    adata: AnnData,
    column: str = None,
    cmap: str = "magma",
    alpha: float = 0.5,
    crd: List[int] = None,
    output: str = None,
) -> None:
    """This function plots the anndata on the shapes of the cells."""

    # disable interactive mode
    if output:
        plt.ioff()

    # Only plot specific column
    if column is not None:
        if column + "_colors" in adata.uns:
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                "new_map",
                adata.uns[column + "_colors"],
                N=len(adata.uns[column + "_colors"]),
            )

        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        ax.imshow(
            adata.uns["spatial"]["melanoma"]["images"]["hires"],
            cmap="gray",
        )
        adata.obsm["polygons"].plot(
            ax=ax,
            column=adata.obs[column],
            edgecolor="white",
            linewidth=adata.obsm["polygons"].linewidth,
            alpha=alpha,
            legend=True,
            cmap=cmap,
        )

    # Plot full AnnData object
    else:
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        ax.imshow(
            adata.uns["spatial"]["melanoma"]["images"]["hires"],
            cmap="gray",
        )
        adata.obsm["polygons"].plot(
            ax=ax,
            edgecolor="white",
            linewidth=adata.obsm["polygons"].linewidth,
            alpha=alpha,
            legend=True,
            color="blue",
        )

    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Plot small part of the image
    if crd is not None:
        ax.set_xlim(crd[0], crd[1])
        ax.set_ylim(crd[2], crd[3])

    # Save the plot to ouput
    if output:
        plt.close(fig)
        fig.savefig(output + ".png")


def preprocessAdata(
    adata: AnnData, mask: np.ndarray, nuc_size_norm: bool = True, n_comps: int = 50
) -> Tuple[AnnData, AnnData]:
    """Returns the new and original AnnData objects

    This function calculates the QC metrics.
    All cells with les then 10 genes and all genes with less then 5 cells are removed.
    Normalization is performed based on the size of the nucleus in nuc_size_norm.
    """

    # Calculate QC Metrics
    sc.pp.calculate_qc_metrics(adata, inplace=True, percent_top=[2, 5])
    adata_orig = adata

    # Filter cells and genes
    sc.pp.filter_cells(adata, min_counts=10)
    sc.pp.filter_genes(adata, min_cells=5)
    adata.raw = adata

    # Normalize nucleus size
    if nuc_size_norm:
        _, counts = np.unique(mask, return_counts=True)
        adata.obs["nucleusSize"] = [counts[int(index)] for index in adata.obs.index]
        adata.X = (adata.X.T / adata.obs.nucleusSize.values).T

        sc.pp.log1p(adata)
        sc.pp.scale(adata, max_value=10)
    else:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)

    sc.tl.pca(adata, svd_solver="arpack", n_comps=n_comps)
    adata.obsm["polygons"] = geopandas.GeoDataFrame(
        adata.obsm["polygons"], geometry=adata.obsm["polygons"].geometry
    )

    return adata, adata_orig


def preprocesAdataPlot(adata: AnnData, adata_orig: AnnData, output: str = None) -> None:
    """This function plots the size of the nucleus related to the counts."""

    # disable interactive mode
    if output:
        plt.ioff()

    sc.pl.pca(adata, color="total_counts", show=not output)

    fig, axs = plt.subplots(1, 2, figsize=(15, 4))
    sns.histplot(adata_orig.obs["total_counts"], kde=False, ax=axs[0])
    sns.histplot(adata_orig.obs["n_genes_by_counts"], kde=False, bins=55, ax=axs[1])

    plt.scatter(adata.obs["nucleusSize"], adata.obs["total_counts"])
    plt.title = "cellsize vs cellcount"

    # Save the plot to ouput
    if output:
        plt.close()
        plt.savefig(output + "0.png")
        plt.close(fig)
        fig.savefig(output + "1.png")


def filter_on_size(
    adata: AnnData, min_size: int = 100, max_size: int = 100000
) -> Tuple[AnnData, int]:
    """Returns a tuple with the AnnData object and the number of filtered cells.

    All cells outside of the min and max size range are removed.
    If the distance between the location of the transcript and the center of the polygon is large, the cell is deleted.
    """

    start = adata.shape[0]

    # Calculate center of the cell and distance between transcript and polygon center
    adata.obsm["polygons"]["X"] = adata.obsm["polygons"].centroid.x
    adata.obsm["polygons"]["Y"] = adata.obsm["polygons"].centroid.y
    adata.obs["distance"] = np.sqrt(
        np.square(adata.obsm["polygons"]["X"] - adata.obsm["spatial"][:, 0])
        + np.square(adata.obsm["polygons"]["Y"] - adata.obsm["spatial"][:, 1])
    )

    # Filter cells based on size and distance
    adata = adata[adata.obs["nucleusSize"] < max_size, :]
    adata = adata[adata.obs["nucleusSize"] > min_size, :]
    adata = adata[adata.obs["distance"] < 70, :]
    adata.obsm["polygons"] = geopandas.GeoDataFrame(
        adata.obsm["polygons"], geometry=adata.obsm["polygons"].geometry
    )
    filtered = start - adata.shape[0]

    return adata, filtered


def extract(ic: sq.im.ImageContainer, adata: AnnData) -> AnnData:
    """This function performs segmenation feature extraction and adds cell area and mean intensity to the annData object under obsm segmentation_features."""
    sq.im.calculate_image_features(
        adata,
        ic,
        layer="image",
        features="segmentation",
        key_added="segmentation_features",
        features_kwargs={
            "segmentation": {
                "label_layer": "segment_cellpose",
                "props": ["label", "area", "mean_intensity"],
                "channels": [0],
            }
        },
    )

    return adata


def clustering(
    adata: AnnData, pcs: int, neighbors: int, cluster_resolution: float = 0.8
) -> AnnData:
    """Returns the AnnData object.

    Performs neighborhood analysis, Leiden clustering and UMAP.
    Provides option to save the plots to output.
    """

    # Neighborhood analysis
    sc.pp.neighbors(adata, n_neighbors=neighbors, n_pcs=pcs)
    sc.tl.umap(adata)

    # Leiden clustering
    sc.tl.leiden(adata, resolution=cluster_resolution)
    sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon")

    return adata


def clustering_plot(adata: AnnData, output: str = None) -> None:
    """This function plots the clusters and genes ranking"""

    # disable interactive mode
    if output:
        plt.ioff()

    # Leiden clustering
    sc.pl.umap(adata, color=["leiden"], show=not output)

    # Save the plot to ouput
    if output:
        plt.savefig(output + "0.png", bbox_inches="tight")
        plt.close()
        sc.pl.rank_genes_groups(adata, n_genes=8, sharey=False, show=False)
        plt.savefig(output + "1.png", bbox_inches="tight")
        plt.close()

    # Display plot
    else:
        sc.pl.rank_genes_groups(adata, n_genes=8, sharey=False)


def scoreGenes(
    adata: AnnData,
    path_marker_genes: str,
    row_norm: bool = False,
    repl_columns: dict[str, str] = None,
    del_genes: List[str] = None,
) -> Tuple[dict, pd.DataFrame]:
    """Returns genes dict and the score sper cluster

    Load the marker genes from csv file in path_marker_genes.
    repl_columns holds the column names that should be replaced the in the marker genes.
    del_genes holds the marker genes that should be deleted from the marker genes and genes dict.
    """

    # Load marker genes from csv
    df_markers = pd.read_csv(path_marker_genes, index_col=0)

    # Replace column names in marker genes
    if repl_columns:
        for column, replace in repl_columns.items():
            df_markers.columns = df_markers.columns.str.replace(column, replace)

    # Create genes dict with all marker genes for every celltype
    genes_dict = {}
    for i in df_markers:
        genes = []
        for row, value in enumerate(df_markers[i]):
            if value > 0:
                genes.append(df_markers.index[row])
        genes_dict[i] = genes

    # Score all cells for all celltypes
    for key, value in genes_dict.items():
        try:
            sc.tl.score_genes(adata, value, score_name=key)
        except ValueError:
            warnings.warn(
                f"Markergenes {value} not present in region, celltype {key} not found"
            )

    # Delete genes from marker genes and genes dict
    if del_genes:
        for gene in del_genes:
            del df_markers[gene]
            del genes_dict[gene]

    scoresper_cluster = adata.obs[
        [col for col in adata.obs if col in df_markers.columns]
    ]

    # Row normalization for visualisation purposes
    if row_norm:
        row_norm = scoresper_cluster.sub(
            scoresper_cluster.mean(axis=1).values, axis="rows"
        ).div(scoresper_cluster.std(axis=1).values, axis="rows")
        adata.obs[scoresper_cluster.columns.values] = row_norm
        temp = pd.DataFrame(np.sort(row_norm)[:, -2:])
    else:
        temp = pd.DataFrame(np.sort(scoresper_cluster)[:, -2:])

    scores = (temp[1] - temp[0]) / ((temp[1] + temp[0]) / 2)
    adata.obs["Cleanliness"] = scores.values
    adata.obs["maxScores"] = scoresper_cluster.idxmax(axis=1)

    return genes_dict, scoresper_cluster


def scoreGenesPlot(
    adata: AnnData,
    scoresper_cluster: pd.DataFrame,
    filter_index: int = 5,
    output: str = None,
) -> None:
    """This function plots the cleanliness and the leiden score next to the maxscores."""

    # Custom colormap:
    colors = np.concatenate(
        (plt.get_cmap("tab20c")(np.arange(20)), plt.get_cmap("tab20b")(np.arange(20)))
    )
    colors = [mpl.rgb2hex(colors[j * 4 + i]) for i in range(4) for j in range(10)]

    # disable interactive mode
    if output:
        plt.ioff()

    # Plot cleanliness and leiden next to maxscores
    sc.pl.umap(adata, color=["Cleanliness", "maxScores"], show=not output)

    # Save the plot to ouput
    if output:
        plt.savefig(output + "0.png", bbox_inches="tight")
        plt.close()
        sc.pl.umap(adata, color=["leiden", "maxScores"], show=False)
        plt.savefig(output + "1.png", bbox_inches="tight")
        plt.close()

    # Display plot
    else:
        sc.pl.umap(adata, color=["leiden", "maxScores"])

    # Plot maxScores and cleanliness columns of AnnData object
    adata.uns["maxScores_colors"] = colors
    plot_shapes(adata, column="maxScores", output=output + "2" if output else None)
    plot_shapes(adata, column="Cleanliness", output=output + "3" if output else None)

    # Plot heatmap of celltypes and filtered celltypes based on filter index
    sc.pl.heatmap(
        adata,
        var_names=scoresper_cluster.columns.values,
        groupby="leiden",
        show=not output,
    )

    # Save the plot to ouput
    if output:
        plt.savefig(output + "4.png", bbox_inches="tight")
        plt.close()
        sc.pl.heatmap(
            adata[
                adata.obs.leiden.isin(
                    [str(index) for index in range(filter_index, len(adata.obs.leiden))]
                )
            ],
            var_names=scoresper_cluster.columns.values,
            groupby="leiden",
            show=False,
        )
        plt.savefig(output + "5.png", bbox_inches="tight")
        plt.close()

    # Display plot
    else:
        sc.pl.heatmap(
            adata[
                adata.obs.leiden.isin(
                    [str(index) for index in range(filter_index, len(adata.obs.leiden))]
                )
            ],
            var_names=scoresper_cluster.columns.values,
            groupby="leiden",
        )


def correct_marker_genes(
    adata: AnnData, genes: dict[str, Tuple[float, float]]
) -> AnnData:
    """Returns the new AnnData object.

    Corrects marker genes that are higher expessed by dividing them.
    The genes has as keys the genes that should be corrected and as values the threshold and the divider.
    """

    # Correct for all the genes
    for gene, values in genes.items():
        for i in range(0, len(adata.obs)):
            if adata.obs[gene].iloc[i] < values[0]:
                adata.obs[gene].iloc[i] = adata.obs[gene].iloc[i] / values[1]
    return adata


def annotate_maxscore(types: str, indexes: dict, adata: AnnData) -> AnnData:
    """Returns the AnnData object.

    Adds types to the Anndata maxscore category.
    """
    adata.obs.maxScores = adata.obs.maxScores.cat.add_categories([types])
    for i, val in enumerate(adata.obs.maxScores):
        if val in indexes[types]:
            adata.obs.maxScores[i] = types
    return adata


def remove_celltypes(types: str, indexes: dict, adata: AnnData) -> AnnData:
    """Returns the AnnData object."""
    for index in indexes[types]:
        if index in adata.obs.maxScores.cat.categories:
            adata.obs.maxScores = adata.obs.maxScores.cat.remove_categories(index)
    return adata


def clustercleanliness(
    adata: AnnData,
    genes: List[str],
    gene_indexes: dict[str, int] = None,
    colors: List[str] = None,
) -> Tuple[AnnData, Optional[dict]]:
    """Returns a tuple with the AnnData object and the color dict."""
    celltypes = np.array(sorted(genes), dtype=str)
    color_dict = None

    adata.obs["maxScores"] = adata.obs[
        [col for col in adata.obs if col in celltypes]
    ].idxmax(axis=1)
    adata.obs.maxScores = adata.obs.maxScores.astype("category")

    # Create custom colormap for clusters
    if not colors:
        color = np.concatenate(
            (
                plt.get_cmap("tab20c")(np.arange(20)),
                plt.get_cmap("tab20b")(np.arange(20)),
            )
        )
        colors = [mpl.rgb2hex(color[j * 4 + i]) for i in range(4) for j in range(10)]

    adata.uns["maxScores_colors"] = colors

    if gene_indexes:
        adata.obs["maxScoresSave"] = adata.obs.maxScores  # CHECKME: does not seem to get used elsewhere, needed?
        gene_celltypes = {}

        for key, value in gene_indexes.items():
            gene_celltypes[key] = celltypes[value]

        for gene, indexes in gene_indexes.items():
            adata = annotate_maxscore(gene, gene_celltypes, adata)

        for gene, indexes in gene_indexes.items():
            adata = remove_celltypes(gene, gene_celltypes, adata)

        celltypes_f = np.delete(celltypes, list(chain(*gene_indexes.values())))  # type: ignore
        celltypes_f = np.append(celltypes_f, list(gene_indexes.keys()))
        color_dict = dict(zip(celltypes_f, adata.uns["maxScores_colors"]))

    else:
        color_dict = dict(zip(celltypes, adata.uns["maxScores_colors"]))

    for i, name in enumerate(color_dict.keys()):
        color_dict[name] = colors[i]
    adata.uns["maxScores_colors"] = list(
        map(color_dict.get, adata.obs.maxScores.cat.categories.values)
    )

    return adata, color_dict


def clustercleanlinessPlot(
    adata: AnnData,
    crop_coord: List[int] = [0, 2000, 0, 2000],
    color_dict: dict = None,
    output: str = None,
) -> None:
    """This function plots the clustercleanliness as barplots, the images with colored celltypes and the clusters."""

    # disable interactive mode
    if output:
        plt.ioff()

    # Create the barplot
    stacked = (
        adata.obs.groupby(["leiden", "maxScores"], as_index=False)
        .size()
        .pivot("leiden", "maxScores")
        .fillna(0)
    )
    stacked_norm = stacked.div(stacked.sum(axis=1), axis=0)
    stacked_norm.columns = list(adata.obs.maxScores.cat.categories)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Use custom colormap
    if color_dict:
        stacked_norm.plot(kind="bar", stacked=True, ax=fig.gca(), color=color_dict)
    else:
        stacked_norm.plot(kind="bar", stacked=True, ax=fig.gca())
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_yaxis().set_ticks([])
    plt.xlabel("Clusters")
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize="large")

    # Save the barplot to ouput
    if output:
        plt.close(fig)
        fig.savefig(output + "0.png", bbox_inches="tight")
    else:
        plt.show()

    # Plot images with colored celltypes
    plot_shapes(
        adata, column="maxScores", alpha=0.8, output=output + "1" if output else None
    )
    plot_shapes(
        adata,
        column="maxScores",
        crd=crop_coord,
        alpha=0.8,
        output=output + "2" if output else None,
    )

    # Plot clusters
    _, ax = plt.subplots(1, 1, figsize=(15, 10))
    sc.pl.umap(adata, color=["maxScores"], ax=ax, size=60, show=not output)
    ax.axis("off")

    # Save the plot to ouput
    if output:
        fig.savefig(output + "3.png", bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def enrichment(adata: AnnData) -> AnnData:
    """Returns the AnnData object.

    Performs some adaptations to save the data.
    Calculate the nhood enrichment"
    """

    # Adaptations for saving
    adata.raw.var.index.names = ["genes"]
    adata.var.index.names = ["genes"]
    # TODO: not used since napari spatialdata
    # adata.obsm["spatial"] = adata.obsm["spatial"].rename({0: "X", 1: "Y"}, axis=1)

    # Calculate nhood enrichment
    sq.gr.spatial_neighbors(adata, coord_type="generic")
    sq.gr.nhood_enrichment(adata, cluster_key="maxScores")
    return adata


def enrichment_plot(adata: AnnData, output: str = None) -> None:
    """This function plots the nhood enrichment between different celltypes."""

    # disable interactive mode
    if output:
        plt.ioff()

    sq.pl.nhood_enrichment(adata, cluster_key="maxScores", method="ward")

    # Save the plot to ouput
    if output:
        plt.savefig(output + ".png", bbox_inches="tight")


def save_data(adata: AnnData, output_geojson: str, output_h5ad: str):
    """Saves the ploygons to output_geojson as GeoJson object and the rest of the AnnData object to output_h5ad as h5ad file."""

    # Save polygons to geojson
    del adata.obsm["polygons"]["color"]
    adata.obsm["polygons"]["geometry"].to_file(output_geojson, driver="GeoJSON")
    adata.obsm["polygons"] = pd.DataFrame(
        {
            "linewidth": adata.obsm["polygons"]["linewidth"],
            "X": adata.obsm["polygons"]["X"],
            "Y": adata.obsm["polygons"]["Y"],
        }
    )

    # Write AnnData object to h5ad file
    adata.write(output_h5ad)
