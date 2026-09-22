import os

from harpy.qc import point_retention


def test_point_retention(sdata_transcripts_no_backed, tmp_path):
    df = point_retention(
        sdata_transcripts_no_backed,
        labels_name="segmentation_mask",
        table_name="table_transcriptomics",
        points_name="transcripts",
        output=os.path.join(tmp_path, "labels_nucleus"),
    )

    assert df.shape == (96, 3)


def test_deprecated_point_retention_alias(monkeypatch):
    from harpy.qc.points import _point_retention as module

    warnings = []
    monkeypatch.setattr(module, "_WARNED_DEPRECATED_ATTRIBUTES", set())
    monkeypatch.setattr(module.log, "warning", warnings.append)

    from harpy.qc import analyse_genes_left_out

    assert analyse_genes_left_out is point_retention
    assert warnings == [
        "`harpy.qc.analyse_genes_left_out` is deprecated. Import and use `harpy.qc.point_retention` instead."
    ]
