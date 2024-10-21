from .cluster_blobs import cluster_blobs, multisample_blobs
from .pixie_example import pixie_example
from .proteomics import macsima_example, mibi_example, vectra_example
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
