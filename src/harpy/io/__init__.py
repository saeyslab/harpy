# from ._merscope import merscope
# from ._spatial_data import create_sdata
# from ._transcripts import (
#     read_merscope_transcripts,
#     read_resolve_transcripts,
#     read_stereoseq_transcripts,
#     read_transcripts,
# )
# from ._visium_hd import visium_hd
# from ._xenium import xenium

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)

# __getattr__, __dir__, __all__ = lazy.attach(
#     __name__,
#     submod_attrs={
#         '_merscope': ['merscope'],
#         '_spatial_data': ['create_sdata'],
#         '_transcripts': [
#             'read_merscope_transcripts',
#             'read_resolve_transcripts',
#             'read_stereoseq_transcripts',
#             'read_transcripts',
#         ],
#         '_visium_hd': ['visium_hd'],
#         '_xenium': ['xenium'],
#     }
# )
