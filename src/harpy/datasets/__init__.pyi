from .cluster_blobs import cluster_blobs, multisample_blobs
from .pixie_example import pixie_example
from .proteomics import macsima_example, macsima_tonsil, mibi_example, vectra_example
from .registry import get_ome_registry, get_registry, get_spatialdata_registry
from .transcriptomics import (
    merscope_example,
    merscope_segmentation_masks_example,
    resolve_example,
    resolve_example_multiple_coordinate_systems,
    visium_hd_example,
    visium_hd_example_custom_binning,
    xenium_example,
)

__all__ = [
    "cluster_blobs",
    "multisample_blobs",
    "pixie_example",
    "macsima_example",
    "macsima_tonsil",
    "mibi_example",
    "vectra_example",
    "get_ome_registry",
    "get_registry",
    "get_spatialdata_registry",
    "merscope_example",
    "merscope_segmentation_masks_example",
    "resolve_example",
    "resolve_example_multiple_coordinate_systems",
    "visium_hd_example",
    "visium_hd_example_custom_binning",
    "xenium_example",
]
