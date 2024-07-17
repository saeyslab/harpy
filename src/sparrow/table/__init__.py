from sparrow.utils.pylogger import get_pylogger

from ._allocation import allocate
from ._allocation_intensity import allocate_intensity
from ._annotation import cluster_cleanliness, score_genes, score_genes_iter
from ._clustering import kmeans, leiden
from ._enrichment import nhood_enrichment
from ._preprocess import preprocess_proteomics, preprocess_transcriptomics
from ._regionprops import add_regionprop_features
from ._table import add_table_layer, correct_marker_genes, filter_on_size
from .cell_clustering._preprocess import cell_clustering_preprocess
from .cell_clustering._weighted_channel_expression import weighted_channel_expression
from .pixel_clustering._cluster_intensity import cluster_intensity

log = get_pylogger(__name__)

try:
    from .cell_clustering._clustering import flowsom
except ImportError:
    log.warning("'flowsom' not installed, 'sp.tb.flowsom' will not be available.")
