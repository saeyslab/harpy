# from ._cell_expansion import create_voronoi_boundaries
# from ._shape import add_shapes_layer, filter_shapes_layer, intersect_rectangles, vectorize

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)

# __getattr__, __dir__, __all__ = lazy.attach(
#     __name__,
#     submod_attrs={
#         '_cell_expansion': ['create_voronoi_boundaries'],
#         '_shape': ['add_shapes_layer', 'filter_shapes_layer', 'intersect_rectangles', 'vectorize'],
#     }
# )