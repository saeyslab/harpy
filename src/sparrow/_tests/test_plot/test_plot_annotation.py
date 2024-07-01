import os

from sparrow.plot import score_genes as score_genes_plot
from sparrow.shape import _add_shapes_layer
from sparrow.table import score_genes


def test_score_genes(sdata_transcripts, path_dataset_markers, tmp_path):
    sdata_transcripts, celltypes_scored, celltypes_all = score_genes(
        sdata_transcripts,
        labels_layer="segmentation_mask",
        table_layer="table_transcriptomics_cluster",
        output_layer="table_transcriptomics_score_genes",
        path_marker_genes=path_dataset_markers,
        delimiter=",",
        row_norm=False,
        del_celltypes=["dummy_20"],  # celltypes that will not be considered for annotation.
        overwrite=True,
    )

    # TODO: add shapes layer to test object.
    sdata_transcripts = _add_shapes_layer(
        sdata_transcripts,
        input=sdata_transcripts.labels["segmentation_mask"].data,
        output_layer="segmentation_mask_boundaries",
        overwrite=True,
    )

    score_genes_plot(
        sdata_transcripts,
        table_layer="table_transcriptomics_score_genes",
        celltypes=celltypes_scored,
        img_layer="raw_image",
        shapes_layer="segmentation_mask_boundaries",
        output=os.path.join(tmp_path, "score_genes"),
    )
