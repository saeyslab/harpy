import os

import pooch
from spatialdata import SpatialData, read_zarr

from sparrow.datasets.registry import get_registry


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


def visium_hd_example_custom_binning() -> SpatialData:
    """Example transcriptomics dataset"""
    # Fetch and unzip the file
    registry = get_registry()
    unzip_path = registry.fetch(
        "transcriptomics/visium_hd/mouse/sdata_custom_binning_visium_hd.zarr.zip", processor=pooch.Unzip()
    )
    sdata = read_zarr(os.path.commonpath(unzip_path))
    sdata.path = None
    return sdata
