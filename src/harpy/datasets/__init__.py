import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)

# __getattr__, __dir__, __all__ = lazy.attach(
#     __name__,
#     submod_attrs={
#         'cluster_blobs': ['cluster_blobs', 'multisample_blobs'],
#         'pixie_example': ['pixie_example'],
#         'proteomics': ['macsima_example', 'macsima_tonsil', 'mibi_example', 'vectra_example'],
#         'registry': ['get_ome_registry', 'get_registry', 'get_spatialdata_registry'],
#         'transcriptomics': [
#             'merscope_example',
#             'merscope_segmentation_masks_example',
#             'resolve_example',
#             'resolve_example_multiple_coordinate_systems',
#             'visium_hd_example',
#             'visium_hd_example_custom_binning',
#             'xenium_example'
#         ]
#     }
# )