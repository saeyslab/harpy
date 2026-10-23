import os

import pytest

import harpy.qc as qc
from harpy.qc import point_retention
from harpy.qc.points import _point_retention as point_retention_module


def test_point_retention(sdata_transcripts_no_backed, tmp_path):
    df = point_retention(
        sdata_transcripts_no_backed,
        labels_name="segmentation_mask",
        table_name="table_transcriptomics",
        points_name="transcripts",
        output=os.path.join(tmp_path, "labels_nucleus"),
    )

    assert df.shape == (96, 3)


def test_deprecated_point_retention_alias():
    message = (
        r"harpy\.qc\.analyse_genes_left_out is deprecated .* Use harpy\.qc\.point_retention, which takes the same "
        r"arguments\."
    )
    for namespace in (qc, point_retention_module):
        with pytest.warns(FutureWarning, match=message) as record:
            assert namespace.analyse_genes_left_out is point_retention
        assert [warning.filename for warning in record] == [__file__]
