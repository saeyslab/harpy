from __future__ import annotations

import os
from pathlib import Path

import pooch
import spatialdata as sd
import tifffile
from numpy.typing import NDArray
from spatialdata import SpatialData, read_zarr
from spatialdata.transformations import get_transformation

from harpy.datasets.registry import get_ome_registry, get_registry, get_spatialdata_registry
from harpy.image._image import add_image_layer
from harpy.io._macsima import macsima


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
    c_coords = sdata["HumanLiverH35"].c.data
    new_c_coords = []
    counts = {}
    # fix duplicate names
    for name in c_coords:
        if name in counts:
            counts[name] += 1
            new_c_coords.append(f"{name}_{counts[name]}")
        else:
            counts[name] = 0
            new_c_coords.append(name)
    sdata = add_image_layer(
        sdata,
        arr=sdata["HumanLiverH35"].data,
        output_layer="HumanLiverH35",
        c_coords=new_c_coords,
        transformations=get_transformation(sdata["HumanLiverH35"], get_all=True),
        overwrite=True,
    )
    assert len(new_c_coords) == len(set(new_c_coords))
    return sdata


def macsima_tonsil(filter_regex: str | None = "None|FITC|PE|APC") -> SpatialData:
    """Tonsil proteomics dataset"""
    # Fetch and unzip the file
    registry = get_registry()
    unzip_path = registry.fetch("proteomics/macsima/tonsil_all.zarr.zip", processor=pooch.Unzip())
    sdata = read_zarr(os.path.commonpath(unzip_path))
    sdata.path = None

    # filter channels if name is contained in filter_regex
    if filter_regex:
        for image in list(sdata.images):
            sdata[image] = sdata[image].drop_isel(c=sdata[image].coords["c"].str.contains(filter_regex, regex=True))

    return sdata


def macsima_colorectal_carcinoma(subset: bool = True, path: str | Path | None = None, **kwargs) -> SpatialData:
    """
    Load the Colorectal Carcinoma MACSima dataset as a :class:`spatialdata.SpatialData` object.

    This function provides access to a highly multiplexed MACSima imaging dataset
    of colorectal carcinoma tissue. By default, a standardized subset of the data
    is returned for convenience and faster loading (5-plex), but the full dataset
    (63-plex) can be loaded if desired (requires ~50 GB of storage).

    Parameters
    ----------
    subset
        Whether to load a pre-defined subset of the dataset (5-plex). Set to
        ``False`` to load the full dataset (63-plex, ~50 GB).
    path
        If ``None``, the example data will be downloaded into the default cache
        directory for your OS. Provide a custom path to change this behavior.
    **kwargs
        Additional keyword arguments passed to :func:`harpy.io.macsima`.

    Returns
    -------
    A :class:`spatialdata.SpatialData` object containing the MACSima colorectal carcinoma
    imaging data.

    See Also
    --------
    harpy.io.macsima : Reader for MACSima data.
    """
    registry = get_registry(path)
    if subset:
        files = [
            "C-000_S-000_S_DAPI_R-02_W-A-1_ROI-01_A-DAPI.tif",
            "C-034_S-000_S_DAPI_R-02_W-A-1_ROI-01_A-DAPI.tif",
            "C-002_S-000_S_FITC_R-02_W-A-1_ROI-01_A-CD8a_C-REA1024.tif",
            "C-004_S-000_S_FITC_R-02_W-A-1_ROI-01_A-CD15_C-VIMC6.tif",
            "C-004_S-000_S_PE_R-02_W-A-1_ROI-01_A-CD45_C-5B1.tif",
        ]
        for _file in files:
            # fetch the files and file paths
            image_path = registry.fetch(os.path.join("proteomics/macsima/REAscreen_IO_CRC", _file))
        return macsima(os.path.dirname(image_path), **kwargs)
    else:
        unzip_path = registry.fetch(
            "proteomics/macsima/REAscreen_IO_CRC/REAscreen_IO_CRC_fixed.zip", processor=pooch.Unzip()
        )
        return macsima(
            os.path.commonpath(unzip_path),
            **kwargs,
        )


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


def vectra_example():
    """Example proteomics dataset LuCa-7color_[13860,52919]_1x1 from Perkin Elmer"""
    # Fetch and unzip the file
    registry = get_ome_registry()
    path = registry.fetch("Vectra-QPTIFF/perkinelmer/PKI_fields/LuCa-7color_%5b13860,52919%5d_1x1component_data.tif")
    input_data, physical_pixel_size_x, physical_pixel_size_y = read_tifffile(path)
    assert physical_pixel_size_x == physical_pixel_size_y
    # TODO use pixel metadata to set the pixel size
    sdata = sd.SpatialData(images={"image": sd.models.Image2DModel.parse(input_data, dims="cyx")})
    sdata.path = None
    return sdata


def codex_example():
    """Example annotated codex dataset (cHL maps dataset), Shaban, M. et al. Data for MAPS: Pathologist-level Cell Type Annotation from Tissue Images through Machine Learning. https://zenodo.org/records/10067010. (2023)."""
    registry = get_registry()

    unzip_path = registry.fetch(
        "proteomics/codex/chl_maps_dataset/sdata_codex_zarr_with_annotations.zarr.zip", processor=pooch.Unzip()
    )
    sdata = read_zarr(os.path.commonpath(unzip_path))
    sdata.path = None
    return sdata


def read_tifffile(path: str | Path) -> tuple[NDArray, float, float]:
    """Read tifffile and extract physical pixel size in microns."""
    # Open the TIFF file
    with tifffile.TiffFile(path) as tif:
        image_data = tif.asarray()

        tags = tif.pages[0].tags
        x_resolution_tag = tags.get("XResolution")
        y_resolution_tag = tags.get("YResolution")
        resolution_unit_tag = tags.get("ResolutionUnit")

        def _get_resolution(resolution_tag):
            if resolution_tag is not None:
                res_value = resolution_tag.value
                if isinstance(res_value, tuple | list):
                    res = res_value[0] / res_value[1]
                else:
                    res = float(res_value)
                return res
            else:
                return None

        x_res = _get_resolution(x_resolution_tag)
        y_res = _get_resolution(y_resolution_tag)

        if resolution_unit_tag is not None:
            resolution_unit = resolution_unit_tag.value
        else:
            raise ValueError("No resolution unit tag found.")

        # Map the resolution unit to micrometers
        if resolution_unit == 1:
            unit_in_micrometers = 1e6  # No unit; assume meter (1 meter = 1,000,000 µm)
        elif resolution_unit == 2:
            unit_in_micrometers = 25.4e3  # Inch to micrometers (1 inch = 25,400 µm)
        elif resolution_unit == 3:
            unit_in_micrometers = 10e3  # Centimeter to micrometers (1 cm = 10,000 µm)
        else:
            raise ValueError("Unknown unit.")

        if x_res is not None:
            physical_pixel_size_x = (1 / x_res) * unit_in_micrometers  # In micrometers
        else:
            physical_pixel_size_x = None

        if y_res is not None:
            physical_pixel_size_y = (1 / y_res) * unit_in_micrometers  # In micrometers
        else:
            physical_pixel_size_y = None

    return image_data, physical_pixel_size_x, physical_pixel_size_y
