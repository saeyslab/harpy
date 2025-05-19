import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)

# __getattr__, __dir__, __all__ = lazy.attach(
#     __name__,
#     submod_attrs={
#         '_align_masks': ['align_labels_layers'],
#         '_expand_masks': ['expand_labels_layer'],
#         '_filter_masks': ['filter_labels_layer'],
#         '_grid': ['add_grid_labels_layer'],
#         '_map': ['map_labels'],
#         '_merge_masks': ['mask_to_original', 'merge_labels_layers', 'merge_labels_layers_nuclei'],
#         '_segmentation': ['segment', 'segment_points'],
#         'segmentation_models': ['cellpose_callable', 'instanseg_callable'],
#     }
# )