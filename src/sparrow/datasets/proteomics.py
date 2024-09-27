import os

import pooch
from spatialdata import SpatialData, read_zarr

from sparrow.datasets.registry import get_registry, get_spatialdata_registry


def mibi_example() -> SpatialData:
    """Example proteomics dataset"""
    # Fetch and unzip the file
    registry = get_registry()
    unzip_path = registry.fetch("proteomics/mibi_tof/sdata_multi_channel.zarr.zip", processor=pooch.Unzip())
    sdata = read_zarr(os.path.commonpath(unzip_path))
    sdata.path = None
    return sdata


def macsima_example() -> SpatialData:
    """Example proteomics dataset"""
    # Fetch and unzip the file
    registry = get_registry()
    unzip_path = registry.fetch("proteomics/macsima/sdata_multi_channel.zarr.zip", processor=pooch.Unzip())
    sdata = read_zarr(os.path.commonpath(unzip_path))
    sdata.path = None
    return sdata


def imc_example():
    """Example IMC dataset"""
    # Fetch and unzip the file
    registry = get_spatialdata_registry()
    unzip_path = registry.fetch("spatialdata-sandbox/steinbock_io.zip", processor=pooch.Unzip(), progressbar=True)
    sdata = read_zarr(os.path.commonpath(unzip_path))
    sdata.path = None
    # Add extra metadata
    sdata.table.var_names = sdata.table.var_names.astype("string")
    # set channel names to be the same as var_names
    for image in list(sdata.images):
        sdata[image].coords["c"] = sdata.table.var_names.astype("string")
    # sample_id is image without the suffix
    sdata.table.obs["sample_id"] = sdata.table.obs["image"].str.split(".").str[0].astype("category")
    # get first part of image name as patient_id
    sdata.table.obs["patient_id"] = sdata.table.obs["image"].str.split("_").str[0].astype("category")
    # get second part of image name as ROI, without the suffix
    sdata.table.obs["ROI"] = sdata.table.obs["image"].str.split("_").str[1].str.split(".").str[0].astype("category")
    # map patient_id to indication using sample_metadata at https://zenodo.org/records/5949116
    sdata.table.obs["indication"] = (
        sdata.table.obs["patient_id"]
        .map(
            {
                "Patient1": "SCCHN",
                "Patient2": "BCC",
                "Patient3": "NSCLC",
                "Patient4": "CRC",
            }
        )
        .astype("category")
    )
    return sdata
