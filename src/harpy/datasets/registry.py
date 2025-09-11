from __future__ import annotations

import hashlib
from pathlib import Path

import pooch
from pooch import Pooch

# from harpy import __version__
__version__ = "0.0.1"  # Do not automatically update dataset Cache for each release, because it downloads a lot of data.

BASE_URL = "https://objectstor.vib.be/spatial-hackathon-public/sparrow/public_datasets"


def get_registry(path: str | Path | None = None) -> Pooch:
    """
    Get the Pooch registry

    Parameters
    ----------
    path
        If None, example data will be downloaded in the default cache folder of your os. Set this to a custom path, to change this behaviour.

    Returns
    -------
    Pooch registry.
    """
    registry = pooch.create(
        path=pooch.os_cache("sparrow") if path is None else path,
        base_url=BASE_URL,
        version=__version__,
        registry={
            "transcriptomics/resolve/mouse/20272_slide1_A1-1_DAPI.tiff": "831e5e7ee30d5aa56a21ed30bafd14a45ee667eae937de27ed0caaa7fa6df6f0",
            "transcriptomics/resolve/mouse/20272_slide1_A1-2_DAPI.tiff": "7c917f517533033a1ff97a1665f83c8a34b6a8e483e013cdb40ef8e53e55dc96",
            "transcriptomics/resolve/mouse/20272_slide1_A1-1_results.txt": "6a83d5725afab88dabd769e2d6fec0357206e8fbef275e5bd9abe9be4263d8a6",
            "transcriptomics/resolve/mouse/20272_slide1_A1-2_results.txt": "9f1d8818b0de2c99d53b2df696f0b4d4948e4877948ef8bbec3c67c147a87de4",
            "transcriptomics/resolve/mouse/20272_slide1_A1-1_DAPI_4288_2144.tiff": "b5a1a16d033634125fb7ca36773b65f34fb5099e97249c38ff0d7e215aecd378",
            "transcriptomics/resolve/mouse/20272_slide1_A1-1_results_4288_2144.txt": "1ac0cc9a8e6cb741cc9634fafb6d1a5c0c8ba6593755fa98ba94f2549bed8f4d",
            "transcriptomics/resolve/mouse/dummy_markers.csv": "3d62c198f5dc1636f2d4faf3c564fad4a3313026f561d2e267e2061a2356432c",
            "transcriptomics/resolve/mouse/markerGeneListMartinNoLow.csv": "1ffefe7d4e72e05ef158ee1e73919b50882a97b6590f4ae977041d6b8b66a459",
            "transcriptomics/resolve/mouse/sdata_transcriptomics.zarr.zip": "30a5649b8a463a623d4e573716f8c365df8c5eed3e77b3e81abf0acaf5ffd1f3",
            "transcriptomics/resolve/mouse/sdata_transcriptomics_coordinate_systems_unit_test.zarr.zip": "ef2ba1c0f6cc9aebe4cf394d1ee00e0622ea4f9273fedd36feb9c7a2363e41a7",
            "transcriptomics/vizgen/mouse/_sdata_2D.zarr.zip": "e1f36061e97e74ad131eb709ca678658829dc4385a444923ef74835e783d63bc",
            "transcriptomics/vizgen/mouse/Liver1Slice1/images/mosaic_DAPI_z3.tif": "943f785d41cb34558349cd2a185c2d1798a788b1d0415e2a0027182d86284175",
            "transcriptomics/vizgen/mouse/Liver1Slice1/images/mosaic_PolyT_z3.tif": "fff162ccdb08359a30f4c578f48aa35e7ee07c432c25c951deac691d2370b26f",
            "transcriptomics/vizgen/mouse/Liver1Slice1/images/micron_to_mosaic_pixel_transform.csv": "17e9ca8560307094b65a364c484b580ea17a6cc776ae882d0b6515bfe6194830",
            "transcriptomics/vizgen/mouse/Liver1Slice1/detected_transcripts.csv": "ceb73b4dcbcfc9d201b19e5f50e98a4704d7d114d254ca434e878e1c63792f69",
            "transcriptomics/visium_hd/mouse/masks.geojson": "a02377ce9924662b440fd7ab91da95e51344e82cda5f27d698ca820030fbfbf3",
            "transcriptomics/visium_hd/mouse/sdata_custom_binning_visium_hd_unit_test.zarr.zip": "346597ca5c85a6ab81239e5b7dbcd11c7715f7a4208cd4912ac78738bd3ed092",
            "transcriptomics/visium_hd/mouse/visium_hd_mouse_small_intestine.zip": "791938dc972d4b42b255673c08dcb3948ebb66c60eabd1483c2fdb67f001256b",
            "transcriptomics/xenium/Xenium_V1_humanLung_Cancer_FFPE/Xenium_V1_humanLung_Cancer_FFPE_he_image.ome.tif": "0348c3aeac4be3770fd3385a0cd99c3948b25b8d5e2d853add6c95e07a8baaf1",
            "transcriptomics/xenium/Xenium_V1_humanLung_Cancer_FFPE/Xenium_V1_humanLung_Cancer_FFPE_he_imagealignment.csv": "9b8e47bf81ca1aa447dc01a476ba6315cdf9ea66d7abb357d91e06ccf67735a7",
            "transcriptomics/xenium/Xenium_V1_humanLung_Cancer_FFPE/Xenium_V1_humanLung_Cancer_FFPE_outs.zip": "865c3805a959b51514555300df61328fc05055b939b9bece43522b8918314e1d",
            "proteomics/pixie/sdata_pixie.zarr.zip": "cdcb17e4e18165d5cfca2c07a899a0e5b19efea26c25826370ef2bf77ea6ad13",
            "proteomics/mibi_tof/sdata_multi_channel.zarr.zip": "930fd2574666b90d5d6660ad8b52d47afffc9522704b9e6fef39d11c9cfff06e",
            "proteomics/macsima/sdata_multi_channel.zarr.zip": "26187fe62b75c3411f948cbcea52abd24b318992509e768c2ed2f55ddcb99f28",
            "proteomics/macsima/tonsil_all.zarr.zip": "f3444332c9decb318c6eff03e719ee35df6d7399da2a04a17dbed5833bfd9ed7",
        },
    )
    return registry


def get_spatialdata_registry(path: str | Path | None = None) -> Pooch:
    """
    Get the Pooch SpatialData registry

    Parameters
    ----------
    path
        If None, example data will be downloaded in the default cache folder of your os. Set this to a custom path, to change this behaviour.

    Returns
    -------
    Pooch registry.
    """
    registry = pooch.create(
        path=pooch.os_cache("sparrow") if path is None else path,
        base_url="https://s3.embl.de/spatialdata",
        version=__version__,
        registry={
            "spatialdata-sandbox/steinbock_io.zip": None,
        },
    )
    return registry


def get_ome_registry(path: str | Path | None = None) -> Pooch:
    """
    Get the Pooch SpatialData registry

    Parameters
    ----------
    path
        If None, example data will be downloaded in the default cache folder of your os. Set this to a custom path, to change this behaviour.

    Returns
    -------
    Pooch registry.
    """
    registry = pooch.create(
        path=pooch.os_cache("sparrow") if path is None else path,
        base_url="https://downloads.openmicroscopy.org/images",
        version=__version__,
        registry={
            "Vectra-QPTIFF/perkinelmer/PKI_fields/LuCa-7color_%5b13860,52919%5d_1x1component_data.tif": "50c3cc12b4e644467cb752d3e5cc778bb7c43209b99f3cac0ba5f44bbbb28fcc",
        },
    )
    return registry


def _calculate_sha256(file_path):
    """Helper function to calculate the hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read the file in chunks to avoid memory issues with large files
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()
