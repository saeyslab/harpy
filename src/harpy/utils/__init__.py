# from harpy.utils._aggregate import RasterAggregator
# from harpy.utils._query import bounding_box_query
# from harpy.utils.pylogger import get_pylogger
# from harpy.utils.utils import _export_config, _get_polygons_in_napari_format, _get_raster_multiscale
import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)

# __getattr__, __dir__, __all__ = lazy.attach(
#     __name__,
#     submod_attrs={
#         '_aggregate': ['RasterAggregator'],
#         '_query': ['bounding_box_query'],
#         'pylogger': ['get_pylogger'],
#         'utils': [
#             '_export_config',
#             '_get_polygons_in_napari_format',
#             '_get_raster_multiscale',
#         ],
#     }
# )
