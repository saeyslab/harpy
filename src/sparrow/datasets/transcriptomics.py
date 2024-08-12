import os

import pooch
from spatialdata import SpatialData, read_zarr

from sparrow.datasets.registry import registry


def resolve_example() -> SpatialData:
    """Example transcriptomics dataset"""
    # Fetch and unzip the file
    unzip_path = registry.fetch("transcriptomics/resolve/mouse/sdata_transcriptomics.zarr.zip", processor=pooch.Unzip())
    return read_zarr(os.path.commonpath(unzip_path))


def resolve_example_multiple_coordinate_systems() -> SpatialData:
    """Example transcriptomics dataset"""
    unzip_path = registry.fetch(
        "transcriptomics/resolve/mouse/sdata_transcriptomics_coordinate_systems_unit_test.zarr.zip",
        processor=pooch.Unzip(),
    )
    return read_zarr(os.path.commonpath(unzip_path))
