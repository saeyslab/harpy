import pooch

from sparrow import __version__

BASE_URL = "https://objectstor.vib.be/spatial-hackathon-public/sparrow/public_datasets"

registry = pooch.create(
    path=pooch.os_cache("sparrow"),
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
        "transcriptomics/resolve/mouse/sdata_transcriptomics.zarr.zip": "32b369195f924f31e57de66a4daa10f34ae049611cfbface46c928fbe40b8fbb",
        "proteomics/mibi_tof/sdata_multi_channel.zarr.zip": "930fd2574666b90d5d6660ad8b52d47afffc9522704b9e6fef39d11c9cfff06e",
    },
)
