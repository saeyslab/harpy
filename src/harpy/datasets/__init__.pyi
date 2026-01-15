from .cluster_blobs import cluster_blobs, multisample_blobs
from .pixie_example import pixie_example
from .proteomics import (
    codex_example,
    macsima_colorectal_carcinoma,
    macsima_example,
    macsima_tonsil,
    macsima_tonsil_benchmark,
    mibi_example,
    vectra_example,
)
from .registry import get_ome_registry, get_registry, get_spatialdata_registry
from .transcriptomics import (
    merscope_mouse_liver,
    merscope_mouse_liver_segmentation_mask,
    resolve_example,
    resolve_example_multiple_coordinate_systems,
    visium_hd_example,
    visium_hd_example_custom_binning,
    xenium_human_lung_cancer,
    xenium_human_ovarian_cancer,
)

__all__ = [
    "cluster_blobs",
    "multisample_blobs",
    "pixie_example",
    "macsima_example",
    "macsima_tonsil",
    "macsima_tonsil_benchmark",
    "codex_example",
    "macsima_colorectal_carcinoma",
    "mibi_example",
    "vectra_example",
    "get_ome_registry",
    "get_registry",
    "get_spatialdata_registry",
    "merscope_mouse_liver",
    "merscope_mouse_liver_segmentation_mask",
    "resolve_example",
    "resolve_example_multiple_coordinate_systems",
    "visium_hd_example",
    "visium_hd_example_custom_binning",
    "xenium_human_lung_cancer",
    "xenium_human_ovarian_cancer",
]
