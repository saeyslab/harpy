from harpy.utils.pylogger import get_pylogger

# from ._allocation import allocate, bin_counts
# from ._allocation_intensity import allocate_intensity
# from ._annotation import cluster_cleanliness, score_genes, score_genes_iter
# from ._clustering import kmeans, leiden
# from ._enrichment import nhood_enrichment
# from ._preprocess import preprocess_proteomics, preprocess_transcriptomics
# from ._regionprops import add_regionprop_features
# from ._table import add_table_layer, correct_marker_genes, filter_on_size
# from .cell_clustering._clustering import flowsom
# from .cell_clustering._preprocess import cell_clustering_preprocess
# from .cell_clustering._weighted_channel_expression import weighted_channel_expression
# from .pixel_clustering._cluster_intensity import cluster_intensity
# from .pixel_clustering._neighbors import spatial_pixel_neighbors

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)


# __getattr__, __dir__, __all__ = lazy.attach(
#     __name__,
#     submod_attrs={
#         '_allocation': ['allocate', 'bin_counts'],
#         '_allocation_intensity': ['allocate_intensity'],
#         '_annotation': ['cluster_cleanliness', 'score_genes', 'score_genes_iter'],
#         '_clustering': ['kmeans', 'leiden'],
#         '_enrichment': ['nhood_enrichment'],
#         '_preprocess': ['preprocess_proteomics', 'preprocess_transcriptomics'],
#         '_regionprops': ['add_regionprop_features'],
#         '_table': ['add_table_layer', 'correct_marker_genes', 'filter_on_size'],
#         'cell_clustering._clustering': ['flowsom'],
#         'cell_clustering._preprocess': ['cell_clustering_preprocess'],
#         'cell_clustering._weighted_channel_expression': ['weighted_channel_expression'],
#         'pixel_clustering._cluster_intensity': ['cluster_intensity'],
#         'pixel_clustering._neighbors': ['spatial_pixel_neighbors'],
#     }
# )