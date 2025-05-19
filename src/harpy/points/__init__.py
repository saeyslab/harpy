# from ._points import add_points_layer

import lazy_loader as lazy

# __getattr__, __dir__, __all__ = lazy.attach(
#     __name__,
#     submod_attrs={
#         '_points': ['add_points_layer'],
#     }
# )

# this assumes there is a `.pyi` file adjacent to this module
__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
