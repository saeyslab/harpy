import os

from sparrow.plot._transcripts import analyse_genes_left_out


def test_analyse_genes_left_out(sdata_transcripts, tmp_path):
    df = analyse_genes_left_out(
        sdata_transcripts,
        labels_layer="segmentation_mask",
        table_layer="table_transcriptomics",
        points_layer="transcripts",
        output=os.path.join(tmp_path, "labels_nucleus"),
    )

    assert df.shape == (96, 3)
