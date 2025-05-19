
# from ._annotation import score_genes
# from ._cluster_cleanliness import cluster_cleanliness
# from ._clustering import cluster
# from ._enrichment import nhood_enrichment
# from ._flowsom import pixel_clusters, pixel_clusters_heatmap
# from ._plot import plot, plot_image, plot_labels, plot_shapes
# from ._preprocess import preprocess_transcriptomics
# from ._qc_cells import plot_adata, ridgeplot_channel, ridgeplot_channel_sample
# from ._qc_image import (
#     calculate_mean_norm,
#     calculate_snr_ratio,
#     clustermap,
#     get_hexes,
#     group_snr_ratio,
#     histogram,
#     make_cols_colors,
#     marker_supervenn,
#     signal_clustermap,
#     snr_clustermap,
#     snr_ratio,
#     supervenn_of_images,
# )
# from ._qc_segmentation import (
#     calculate_segmentation_coverage,
#     calculate_segments_per_area,
#     segmentation_coverage,
#     segmentation_size_boxplot,
#     segments_per_area,
# )
# from ._sanity import sanity
# from ._segmentation import segment
# from ._tiling_correction import flatfield, tiling_correction
# from ._transcripts import analyse_genes_left_out, transcript_density

from harpy.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

import lazy_loader as lazy

# this assumes there is a `.pyi` file adjacent to this module
__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)

# __getattr__, __dir__, __all__ = lazy.attach(
#     __name__,
#     submod_attrs={
#         '_annotation': ['score_genes'],
#         '_cluster_cleanliness': ['cluster_cleanliness'],
#         '_clustering': ['cluster'],
#         '_enrichment': ['nhood_enrichment'],
#         '_flowsom': ['pixel_clusters', 'pixel_clusters_heatmap'],
#         '_plot': ['plot', 'plot_image', 'plot_labels', 'plot_shapes'],
#         '_preprocess': ['preprocess_transcriptomics'],
#         '_qc_cells': ['plot_adata', 'ridgeplot_channel', 'ridgeplot_channel_sample'],
#         '_qc_image': [
#             'calculate_mean_norm',
#             'calculate_snr_ratio',
#             'clustermap',
#             'get_hexes',
#             'group_snr_ratio',
#             'histogram',
#             'make_cols_colors',
#             'marker_supervenn',
#             'signal_clustermap',
#             'snr_clustermap',
#             'snr_ratio',
#             'supervenn_of_images'
#         ],
#         '_qc_segmentation': [
#             'calculate_segmentation_coverage',
#             'calculate_segments_per_area',
#             'segmentation_coverage',
#             'segmentation_size_boxplot',
#             'segments_per_area'
#         ],
#         '_sanity': ['sanity'],
#         '_segmentation': ['segment'],
#         '_tiling_correction': ['flatfield', 'tiling_correction'],
#         '_transcripts': ['analyse_genes_left_out', 'transcript_density'],
#     }
# )
