import os
from pathlib import Path

import pooch
from spatialdata import SpatialData, read_zarr

from sparrow.datasets.registry import get_registry
from sparrow.io._merscope import merscope
from sparrow.io._visium_hd import visium_hd
from sparrow.io._xenium import xenium
from sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def resolve_example() -> SpatialData:
    """Example transcriptomics dataset"""
    # Fetch and unzip the file
    registry = get_registry()
    unzip_path = registry.fetch("transcriptomics/resolve/mouse/sdata_transcriptomics.zarr.zip", processor=pooch.Unzip())
    sdata = read_zarr(os.path.commonpath(unzip_path))
    sdata.path = None
    return sdata


def resolve_example_multiple_coordinate_systems() -> SpatialData:
    """Example transcriptomics dataset"""
    registry = get_registry()
    unzip_path = registry.fetch(
        "transcriptomics/resolve/mouse/sdata_transcriptomics_coordinate_systems_unit_test.zarr.zip",
        processor=pooch.Unzip(),
    )
    sdata = read_zarr(os.path.commonpath(unzip_path))
    sdata.path = None
    return sdata


def merscope_example(output: str | Path = None, transcripts: bool = True) -> SpatialData:
    """
    Example transcriptomics dataset

    Parameters
    ----------
    output
        The path where the resulting `SpatialData` object will be backed. If None, it will not be backed to a zarr store.
        Note that setting `output` to `None` will persist the transcripts in memory.
    transcripts
        Whether to read transcripts.

    Returns
    -------
    A SpatialData object.
    """
    if output is None and transcripts:
        log.warning("Setting 'output' to None will persist the detected transcripts in memory.")
    registry = get_registry()

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
        image_models_kwargs={"scale_factors": [2, 2, 2, 2]},
        output=output,
    )

    return sdata


def xenium_example(output: str | Path = None) -> SpatialData:
    """
    Example transcriptomics dataset

    Parameters
    ----------
    output
        The path where the resulting `SpatialData` object will be backed. If None, it will not be backed to a zarr store.

    Returns
    -------
    A SpatialData object.
    """
    registry = get_registry()
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
        output=output,
    )

    return sdata


def merscope_segmentation_masks_example() -> SpatialData:
    """Example transcriptomics dataset"""
    # Fetch and unzip the file
    registry = get_registry()
    unzip_path = registry.fetch("transcriptomics/vizgen/mouse/_sdata_2D.zarr.zip", processor=pooch.Unzip())
    sdata = read_zarr(os.path.commonpath(unzip_path))
    sdata.path = None
    return sdata


def visium_hd_example(bin_size: int | list[int] = 16, output=None) -> SpatialData:
    """Example transcriptomics dataset

    Parameters
    ----------
    bin_size
        When specified, load the data of a specific bin size, or a list of bin sizes. By default, it loads all the
        available bin sizes.
    output
        The path where the resulting `SpatialData` object will be backed. If None, it will not be backed to a zarr store.

    Returns
    -------
    A SpatialData object.
    """
    registry = get_registry()
    unzip_path = registry.fetch(
        "transcriptomics/visium_hd/mouse/visium_hd_mouse_small_intestine.zip",
        processor=pooch.Unzip(),
    )

    path = os.path.commonpath(unzip_path)

    sdata = visium_hd(
        path=path, bin_size=bin_size, dataset_id="Visium_HD_Mouse_Small_Intestine", bins_as_squares=True, output=output
    )

    return sdata


def visium_hd_example_custom_binning() -> SpatialData:
    """Example transcriptomics dataset"""
    # Fetch and unzip the file
    registry = get_registry()
    unzip_path = registry.fetch(
        "transcriptomics/visium_hd/mouse/sdata_custom_binning_visium_hd_unit_test.zarr.zip", processor=pooch.Unzip()
    )
    sdata = read_zarr(os.path.commonpath(unzip_path))
    sdata.path = None
    return sdata
