import os
from pathlib import Path

import pooch
from spatialdata import SpatialData, read_zarr

from sparrow.datasets.registry import get_registry
from sparrow.io._visium_hd import visium_hd
from sparrow.io._xenium import xenium


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


def xenium_example(output: str | Path = None) -> SpatialData:
    """
    Example transcriptomics dataset

    Parameters
    ----------
    output
        The path where the resulting `SpatialData` object will be backed. If None, it will not be backed to a zarr store.
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


def vizgen_example() -> SpatialData:
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
