from sparrow.utils.pylogger import get_pylogger

from ._apply import apply
from ._combine import combine
from ._contrast import enhance_contrast
from ._image import _add_image_layer, _add_label_layer
from ._minmax import min_max_filtering
from ._shapes_to_labels import add_label_layer_from_shapes_layer
from ._transcripts import transcript_density
from .segmentation._align_masks import align_labels_layers
from .segmentation._apply import apply_labels_layers
from .segmentation._expand_masks import expand_labels_layer
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
