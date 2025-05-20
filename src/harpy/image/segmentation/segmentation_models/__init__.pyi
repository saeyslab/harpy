from .segmentation.segmentation_models._cellpose import cellpose_callable
from .segmentation.segmentation_models._instanseg import instanseg_callable

__all__ = [
    "cellpose_callable",
    "instanseg_callable",
]
