"""Export all the widgets for the napari plugin."""
__version__ = "0.0.1"

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "_allocate_widget": ["allocate_widget"],
        "_annotate_widget": ["annotate_widget"],
        "_clean_widget": ["clean_widget"],
        "_load_widget": ["load_widget"],
        "_segment_widget": ["segment_widget"],
        "_wizard_widget": ["wizard_widget"],
    },
)