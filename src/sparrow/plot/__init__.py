from sparrow.utils.pylogger import get_pylogger

from ._annotation import score_genes
from ._cluster_cleanliness import cluster_cleanliness
from ._clustering import cluster
from ._enrichment import nhood_enrichment
from ._plot import plot_image, plot_labels, plot_shapes
from ._preprocess import preprocess_transcriptomics
from ._qc_cells import plot_adata
from ._qc_image import (
    calculate_mean_norm,
    calculate_snr_ratio,
    clustermap,
    get_hexes,
    make_cols_colors,
    signal_clustermap,
    snr_clustermap,
)
from ._qc_segmentation import (
    calculate_segmentation_coverage,
    calculate_segments_per_area,
    segmentation_coverage,
    segmentation_size_boxplot,
    segments_per_area,
)
from ._sanity import sanity_plot_transcripts_matrix
from ._segmentation import segment
from ._tiling_correction import flatfield, tiling_correction
from ._transcripts import analyse_genes_left_out, transcript_density

log = get_pylogger(__name__)

try:
    import joypy

    from ._qc_cells import ridgeplot_channel, ridgeplot_channel_sample

except ImportError:
    log.warning(
        "'joypy' not installed, 'sp.pl.ridgeplot_channel' and 'sp.pl.ridgeplot_channel_sample' will not be available."
    )

try:
    import textalloc

    from ._qc_image import (
        group_snr_ratio,
        snr_ratio,
    )
except ImportError:
    log.warning("'textalloc' not installed, 'sp.pl.group_snr_ratio' and 'sp.pl.snr_ratio' will not be available.")
