import json
import os
from pathlib import Path

import pandas as pd
import pooch
from spatialdata import SpatialData, read_zarr
from spatialdata.models import TableModel
from spatialdata.transformations import Scale, Sequence, get_transformation, set_transformation
from spatialdata_io import xenium_aligned_image
from spatialdata_io._constants._constants import XeniumKeys

import harpy as hp
from harpy.datasets.registry import get_registry
from harpy.io._merscope import merscope
from harpy.io._visium_hd import visium_hd
from harpy.io._xenium import xenium
from harpy.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def resolve_example(path: str | Path | None = None) -> SpatialData:
    """
    Example transcriptomics dataset.

    Mouse liver spatial transcriptomics data generated using Resolve Biosciences’ Molecular Cartography technology.

    Parameters
    ----------
    path
        If ``None``, the example data will be downloaded into the default cache
        directory for your OS. Provide a custom path to change this behavior.
    """
    # Fetch and unzip the file
    registry = get_registry(path)
    unzip_path = registry.fetch("transcriptomics/resolve/mouse/sdata_transcriptomics.zarr.zip", processor=pooch.Unzip())
    sdata = read_zarr(os.path.commonpath(unzip_path))
    sdata.path = None
    return sdata


def resolve_example_multiple_coordinate_systems(path: str | Path | None = None) -> SpatialData:
    """
    Example transcriptomics dataset (liver mouse )

    Mouse liver spatial transcriptomics data generated using Resolve Biosciences’ Molecular Cartography technology.

    Parameters
    ----------
    path
        If ``None``, the example data will be downloaded into the default cache
        directory for your OS. Provide a custom path to change this behavior.
    """
    registry = get_registry(path)
    unzip_path = registry.fetch(
        "transcriptomics/resolve/mouse/sdata_transcriptomics_coordinate_systems_unit_test.zarr.zip",
        processor=pooch.Unzip(),
    )
    sdata = read_zarr(os.path.commonpath(unzip_path))
    sdata.path = None
    return sdata


def merscope_mouse_liver(
    output: str | Path | None = None, path: str | Path | None = None, transcripts: bool = True
) -> SpatialData:
    """
    Example transcriptomics dataset

    MERFISH mouse liver spatial transcriptomics dataset, downloaded from (accessed 1/10/2024):
    https://info.vizgen.com/mouse-liver-data

    Parameters
    ----------
    output
        The path where the resulting `SpatialData` object will be backed. If None, it will not be backed to a zarr store.
        Note that setting `output` to `None` will persist the transcripts in memory.
        We recommend specifying `output`.
    path
        If ``None``, the example data will be downloaded into the default cache
        directory for your OS. Provide a custom path to change this behavior.
    transcripts
        Whether to read transcripts.

    Returns
    -------
    A SpatialData object.
    """
    if output is None and transcripts:
        log.warning("Setting 'output' to None will persist the detected transcripts in memory.")
    registry = get_registry(path)

    _ = registry.fetch("transcriptomics/vizgen/mouse/Liver1Slice1/images/mosaic_DAPI_z3.tif")
    _ = registry.fetch("transcriptomics/vizgen/mouse/Liver1Slice1/images/mosaic_PolyT_z3.tif")
    _ = registry.fetch("transcriptomics/vizgen/mouse/Liver1Slice1/images/micron_to_mosaic_pixel_transform.csv")
    path_transcripts = registry.fetch("transcriptomics/vizgen/mouse/Liver1Slice1/detected_transcripts.csv")

    input_path = os.path.dirname(path_transcripts)

    sdata = merscope(
        path=input_path,
        to_coordinate_system="global",
        z_layers=[
            3,
        ],
        backend=None,
        transcripts=transcripts,
        mosaic_images=True,
        do_3D=False,
        z_projection=False,
        table=False,
        cell_boundaries=False,
        image_models_kwargs={"scale_factors": [2, 2, 2, 2]},
        output=output,
    )

    return sdata


def xenium_human_lung_cancer(output: str | Path | None = None, path: str | Path | None = None) -> SpatialData:
    """
    Example transcriptomics dataset

    Data downloaded from 10x Genomics (accessed 01/10/2024):
    https://www.10xgenomics.com/datasets/preview-data-ffpe-human-lung-cancer-with-xenium-multimodal-cell-segmentation-1-standard

    Parameters
    ----------
    output
        The path where the resulting `SpatialData` object will be backed. If `None`, it will not be backed to a zarr store.
        We recommend specifying `output`.
    path
        If `None`, the example data will be downloaded into the default cache
        directory for your OS. Provide a custom path to change this behavior.

    Returns
    -------
    A SpatialData object.
    """
    registry = get_registry(path)
    path_unzipped = registry.fetch(
        "transcriptomics/xenium/Xenium_V1_humanLung_Cancer_FFPE/Xenium_V1_humanLung_Cancer_FFPE_outs.zip",
        processor=pooch.Unzip(extract_dir="."),
    )
    _ = registry.fetch(
        "transcriptomics/xenium/Xenium_V1_humanLung_Cancer_FFPE/Xenium_V1_humanLung_Cancer_FFPE_he_image.ome.tif"
    )
    _ = registry.fetch(
        "transcriptomics/xenium/Xenium_V1_humanLung_Cancer_FFPE/Xenium_V1_humanLung_Cancer_FFPE_he_imagealignment.csv"
    )
    input_path = os.path.commonpath(path_unzipped)
    sdata = xenium(
        input_path,
        to_coordinate_system="global",
        aligned_images=True,
        cells_table=True,
        nucleus_labels=True,
        cells_labels=True,
        filter_gene_names=["Unassigned", "NegControl"],
        output=output,
    )

    return sdata


def xenium_human_ovarian_cancer(output: str | Path = None, path: str | Path | None = None) -> SpatialData:
    """
    Example transcriptomics dataset

    Data downloaded from 10x Genomics (accessed 25/11/2025):
    https://www.10xgenomics.com/datasets/xenium-prime-ffpe-human-ovarian-cancer

    The downloaded data (~ 80GB), combined with the generated `SpatialData` object (~ 20GB),
    requires approximately 100 GB of disk storage.

    Parameters
    ----------
    output
        The path where the resulting `SpatialData` object will be backed. If `None`, it will not be backed to a Zarr store.
        We recommend specifying `output`.
    path
        If `None`, the example data will be downloaded into the default cache
        directory for your OS. Provide a custom path to change this behavior.

    Returns
    -------
    A SpatialData object.
    """
    to_coordinate_system = "global_ROI1"

    # fetch the data
    registry = get_registry(path)
    path_unzipped = registry.fetch(
        "transcriptomics/xenium/Xenium_human_ovarian_cancer/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_outs.zip",
        processor=pooch.Unzip(extract_dir="."),
    )
    _ = registry.fetch(
        "transcriptomics/xenium/Xenium_human_ovarian_cancer/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_he_image.ome.tif"
    )

    path_he_annotated = registry.fetch(
        "transcriptomics/xenium/Xenium_human_ovarian_cancer/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_he_annotated_image.ome.tif"
    )

    path_alignment_file = registry.fetch(
        "transcriptomics/xenium/Xenium_human_ovarian_cancer/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_he_imagealignment.csv"
    )

    path_cell_groups = registry.fetch(
        "transcriptomics/xenium/Xenium_human_ovarian_cancer/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_cell_groups.csv"
    )

    path_gene_groups = registry.fetch(
        "transcriptomics/xenium/Xenium_human_ovarian_cancer/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_gene_groups.csv"
    )

    # 1) read everything from path_unzipped, and the aligned images
    input_path = os.path.commonpath(path_unzipped)
    image_model_kwargs = {"scale_factors": [2, 2, 2, 2], "chunks": (1, 4096, 4096)}
    labels_model_kwargs = {"scale_factors": [2, 2, 2, 2], "chunks": (4096, 4096)}
    sdata = hp.io.xenium(
        input_path,
        to_coordinate_system=to_coordinate_system,
        aligned_images=True,
        cells_table=True,
        nucleus_labels=True,
        cells_labels=True,
        filter_gene_names=["Unassigned", "NegControl"],
        image_models_kwargs=image_model_kwargs,
        labels_models_kwargs=labels_model_kwargs,
        output=output,
    )

    # 2) add extra aligned he image
    se = xenium_aligned_image(
        image_path=path_he_annotated,
        alignment_file=path_alignment_file,
        rgba=True,
        image_models_kwargs=image_model_kwargs,
    )

    # add a micron coordinate system to the annotated he image, and rename coordinate system "global" to "to_coordinate_system"
    with open(os.path.join(input_path, XeniumKeys.XENIUM_SPECS)) as f:
        specs = json.load(f)
        pixel_size = specs["pixel_size"]

    scale_pixels_to_micron = Scale(axes=("x", "y"), scale=[pixel_size, pixel_size])

    transformation = get_transformation(
        se, to_coordinate_system="global"
    )  # spatialdata by default adds the image to the "global" coordinate system when calling 'xenium_aligned_image'

    transformations = {
        to_coordinate_system: transformation,
        f"{to_coordinate_system}_micron": Sequence([transformation, scale_pixels_to_micron]),
    }
    set_transformation(se, transformation=transformations, set_all=True)

    sdata[f"he_image_annotated_{to_coordinate_system}"] = se
    sdata.write_element(f"he_image_annotated_{to_coordinate_system}")
    sdata = read_zarr(sdata.path)

    # 3. add cell and gene groups to the AnnData table

    table_layer = f"table_{to_coordinate_system}"  # default table name when calling hp.io.xenium

    # index of adata is XeniumKeys.CELL_ID
    adata = sdata[table_layer]
    if adata.obs.index.name != XeniumKeys.CELL_ID:
        raise ValueError(
            f"Name of the index of the AnnData table with name '{table_layer}' should be {XeniumKeys.CELL_ID}, please report this bug."
        )
    adata.obs.reset_index(inplace=True)

    cell_groups = pd.read_csv(path_cell_groups)
    gene_groups = pd.read_csv(path_gene_groups)

    # i) check which cells in adata.obs could be matched to a group
    missing = set(adata.obs[XeniumKeys.CELL_ID]) - set(cell_groups[XeniumKeys.CELL_ID])
    if missing:
        log.info(
            f"{len(missing)} {XeniumKeys.CELL_ID} values in .obs not found in cell groups (e.g. {list(missing)[:5]}). These will be removed from the AnnData table."
        )

    # ii) check that there are no duplicates in cell_groups dataframe. If there would be duplicates in cell_groups, then rows in adata.obs would be multiplied, we want to catch this early
    duplicates = cell_groups[cell_groups.duplicated(subset=XeniumKeys.CELL_ID, keep=False)]
    if not duplicates.empty:
        raise ValueError(
            f"Duplicate {XeniumKeys.CELL_ID} values found in cell groups (e.g. {duplicates[XeniumKeys.CELL_ID].unique()[:5]})"
        )

    # merge cell groups with adata
    adata.obs = pd.merge(adata.obs, cell_groups, how="left", on=XeniumKeys.CELL_ID)
    adata.obs.set_index(XeniumKeys.CELL_ID, inplace=True)

    adata = adata[~adata.obs["group"].isna()].copy()  # copy because we can not pop on a view

    # 3) there are duplicates in the gene_groups dataframe. Therefore we can not add them to adata.var, so we store them in .uns
    duplicates = gene_groups[gene_groups.duplicated(subset="gene", keep=False)]
    if not duplicates.empty:
        log.info(f"Duplicate 'gene' values found in gene groups (e.g. {duplicates['gene'].unique()[:5]})")

    adata.uns["gene_groups"] = gene_groups

    region = adata.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY]
    adata.uns.pop(TableModel.ATTRS_KEY)
    # back to zarr
    sdata = hp.tb.add_table_layer(
        sdata,
        adata=adata,
        output_layer=table_layer,
        region=region,
        overwrite=True,
    )

    return sdata


def merscope_mouse_liver_segmentation_mask(path: str | Path | None = None) -> SpatialData:
    """
    Example transcriptomics dataset, but with segmentation masks generated using :func:`harpy.im.segment`.

    MERFISH mouse liver spatial transcriptomics dataset, downloaded from (accessed 1/10/2024)
    https://info.vizgen.com/mouse-liver-data

    path
        If `None`, the example data will be downloaded into the default cache
        directory for your OS. Provide a custom path to change this behavior.
    """
    # Fetch and unzip the file
    registry = get_registry(path)
    unzip_path = registry.fetch("transcriptomics/vizgen/mouse/_sdata_2D.zarr.zip", processor=pooch.Unzip())
    sdata = read_zarr(os.path.commonpath(unzip_path))
    sdata.path = None
    return sdata


def visium_hd_example(
    bin_size: int | list[int] = 16, output: str | Path | None = None, path: str | Path | None = None
) -> SpatialData:
    """Example transcriptomics dataset

    Parameters
    ----------
    bin_size
        When specified, load the data of a specific bin size, or a list of bin sizes. By default, it loads all the
        available bin sizes.
    output
        The path where the resulting `SpatialData` object will be backed. If None, it will not be backed to a zarr store.
        We recommend specifying `output`.
    path
        If `None`, the example data will be downloaded into the default cache
        directory for your OS. Provide a custom path to change this behavior.

    Returns
    -------
    A SpatialData object.
    """
    registry = get_registry(path)
    unzip_path = registry.fetch(
        "transcriptomics/visium_hd/mouse/visium_hd_mouse_small_intestine.zip",
        processor=pooch.Unzip(),
    )

    path = os.path.commonpath(unzip_path)

    sdata = visium_hd(
        path=path, bin_size=bin_size, dataset_id="Visium_HD_Mouse_Small_Intestine", bins_as_squares=True, output=output
    )

    return sdata


def visium_hd_example_custom_binning(path: str | Path | None = None) -> SpatialData:
    """
    Example transcriptomics dataset

    Parameters
    ----------
    path
        If `None`, the example data will be downloaded into the default cache
        directory for your OS. Provide a custom path to change this behavior.

    Returns
    -------
    A SpatialData object.
    """
    # Fetch and unzip the file
    registry = get_registry(path)
    unzip_path = registry.fetch(
        "transcriptomics/visium_hd/mouse/sdata_custom_binning_visium_hd_unit_test.zarr.zip", processor=pooch.Unzip()
    )
    sdata = read_zarr(os.path.commonpath(unzip_path))
    sdata.path = None
    return sdata
