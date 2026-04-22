import os

from harpy.qc._qc_transcripts import analyse_genes_left_out


def test_analyse_genes_left_out(sdata_transcripts_no_backed, tmp_path):
    df = analyse_genes_left_out(
        sdata_transcripts_no_backed,
        labels_name="segmentation_mask",
        table_name="table_transcriptomics",
        points_name="transcripts",
        output=os.path.join(tmp_path, "labels_nucleus"),
    )

    assert df.shape == (96, 3)
