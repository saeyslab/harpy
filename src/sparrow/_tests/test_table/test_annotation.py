from anndata import AnnData

from sparrow.table import score_genes, score_genes_iter
from sparrow.utils._keys import _ANNOTATION_KEY


def test_score_genes(sdata_transcripts, path_dataset_markers):
    sdata_transcripts, celltypes_scored, celltypes_all = score_genes(
        sdata=sdata_transcripts,
        labels_layer="segmentation_mask",
        table_layer="table_transcriptomics_cluster",
        output_layer="table_transcriptomics_score_genes",
        path_marker_genes=path_dataset_markers,
        delimiter=",",
        row_norm=False,
        del_celltypes=["dummy_20"],  # celltypes that will not be considered for annotation.
        overwrite=True,
    )

    assert isinstance(sdata_transcripts["table_transcriptomics_score_genes"], AnnData)

    annotated_celltypes = sdata_transcripts["table_transcriptomics_score_genes"].obs[_ANNOTATION_KEY].cat.categories

    for celltype in annotated_celltypes:
        assert celltype in celltypes_scored
        assert celltype in celltypes_all

    assert "dummy_20" not in annotated_celltypes
    assert "dummy_20" not in celltypes_scored
    assert "dummy_20" not in celltypes_all
    assert "dummy_33" not in celltypes_scored  # because this celltypes has no matches in the tissue
    assert "dummy_33" in celltypes_all


def test_score_genes_iter(sdata_transcripts, path_dataset_markers, tmpdir):
    sdata_transcripts, celltypes_scored, celltypes_all = score_genes_iter(
        sdata=sdata_transcripts,
        labels_layer="segmentation_mask",
        table_layer="table_transcriptomics_cluster",
        output_layer="table_transcriptomics_score_genes",
        path_marker_genes=path_dataset_markers,
        delimiter=",",
        overwrite=True,
        output_dir=tmpdir,
        n_iter=5,
    )

    assert "dummy_33" not in celltypes_scored  # because this celltypes has no matches in the tissue
    assert "dummy_33" in celltypes_all
    assert isinstance(sdata_transcripts["table_transcriptomics_score_genes"], AnnData)
