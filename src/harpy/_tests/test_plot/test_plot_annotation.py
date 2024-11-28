import os

from sparrow.plot import score_genes as score_genes_plot
from sparrow.table import score_genes


def test_score_genes(sdata_transcripts_no_backed, path_dataset_markers, tmp_path):
    sdata_transcripts_no_backed, celltypes_scored, celltypes_all = score_genes(
        sdata_transcripts_no_backed,
        labels_layer="segmentation_mask",
        table_layer="table_transcriptomics_cluster",
        output_layer="table_transcriptomics_score_genes",
        path_marker_genes=path_dataset_markers,
        delimiter=",",
        row_norm=False,
        del_celltypes=["dummy_20"],  # celltypes that will not be considered for annotation.
        overwrite=True,
    )

    score_genes_plot(
        sdata_transcripts_no_backed,
        table_layer="table_transcriptomics_score_genes",
        celltypes=celltypes_scored,
        img_layer="raw_image",
        shapes_layer="segmentation_mask_boundaries",
        output=os.path.join(tmp_path, "score_genes"),
    )
