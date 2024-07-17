from sparrow.utils.pylogger import get_pylogger

from ._apply import map_channels_zstacks
from ._combine import combine
from ._contrast import enhance_contrast
from ._filters import gaussian_filtering, min_max_filtering
from ._image import add_image_layer, add_labels_layer
from ._shapes_to_labels import add_labels_layer_from_shapes_layer
from ._transcripts import transcript_density
from .pixel_clustering._preprocess import pixel_clustering_preprocess
from .segmentation._align_masks import align_labels_layers
from .segmentation._apply import apply_labels_layers
from .segmentation._expand_masks import expand_labels_layer
from .segmentation._filter_masks import filter_labels_layer
from .segmentation._merge_masks import (
    mask_to_original,
    merge_labels_layers,
    merge_labels_layers_nuclei,
)
from .segmentation._segmentation import segment, segment_points

log = get_pylogger(__name__)

try:
    import basicpy
    import jax
    import jaxlib

    from ._tiling import tiling_correction
except ImportError:
    log.warning("'jax' or 'basicpy' not installed, 'sp.im.tiling_correction' will not be available.")

try:
    from .pixel_clustering._clustering import flowsom
except ImportError:
    log.warning("'flowsom' not installed, 'sp.im.flowsom' will not be available.")
