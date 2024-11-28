import pytest

from harpy.table._preprocess import preprocess_proteomics, preprocess_transcriptomics
from harpy.utils._keys import _CELLSIZE_KEY, _REGION_KEY


@pytest.mark.parametrize("q", [None, 0.999])
def test_preprocess_proteomics(sdata_multi_c_no_backed, q):
    sdata_multi_c_no_backed = preprocess_proteomics(
        sdata_multi_c_no_backed,
        labels_layer="masks_whole",
        table_layer="table_intensities",
        output_layer="table_intensities_preprocessed",
        q=q,
        overwrite=True,
    )

    assert sdata_multi_c_no_backed["table_intensities"].shape == (1299, 22)
    assert sdata_multi_c_no_backed["table_intensities_preprocessed"].shape == (674, 22)

    adata = sdata_multi_c_no_backed.tables["table_intensities_preprocessed"]
    assert (adata.raw is None) if q is None else (adata.raw is not None)
    assert len(adata.obs[_REGION_KEY].cat.categories) == 1
    assert "masks_whole" in adata.obs[_REGION_KEY].cat.categories
    assert _CELLSIZE_KEY in adata.obs.columns
    assert "log1p" in adata.uns.keys()


def test_preprocess_proteomics_multiple_samples(sdata_multi_c_no_backed):
    # instead of one labels layer, user could give up multiple labels layers (linked to sdata.tables[table_layer] via the _REGION_KEY),
    # which could represent multiple samples that need to be preprocessed together.
    # in this dummy example, we preprocess cell masks and corresponding nuclear masks together, which does not make much sense in practice.
    sdata_multi_c_no_backed = preprocess_proteomics(
        sdata_multi_c_no_backed,
        labels_layer=["masks_whole", "masks_nuclear_aligned"],
        table_layer="table_intensities",
        output_layer="table_intensities_preprocessed",
        overwrite=True,
    )

    assert sdata_multi_c_no_backed["table_intensities"].shape == (1299, 22)
    assert sdata_multi_c_no_backed["table_intensities_preprocessed"].shape == (1299, 22)

    adata = sdata_multi_c_no_backed.tables["table_intensities_preprocessed"]
    assert len(adata.obs[_REGION_KEY].cat.categories) == 2
    assert "masks_whole" in adata.obs[_REGION_KEY].cat.categories
    assert "masks_nuclear_aligned" in adata.obs[_REGION_KEY].cat.categories
    assert _CELLSIZE_KEY in adata.obs.columns
    assert "log1p" in adata.uns.keys()


def test_preprocess_proteomics_overwrite(sdata_multi_c_no_backed):
    sdata_multi_c_no_backed = preprocess_proteomics(
        sdata_multi_c_no_backed,
        labels_layer="masks_whole",
        table_layer="table_intensities",
        output_layer="table_intensities",
        overwrite=True,
    )
    # running preprocess takes cells corresponding to certain labels_layer from sdata.tables[table_layer].
    adata = sdata_multi_c_no_backed.tables["table_intensities"]
    assert adata.shape == (674, 22)
    assert len(adata.obs[_REGION_KEY].cat.categories) == 1
    assert "masks_whole" in adata.obs[_REGION_KEY].cat.categories
    assert _CELLSIZE_KEY in adata.obs.columns
    assert "log1p" in adata.uns.keys()


@pytest.mark.parametrize("highly_variable_genes", [False, True])
def test_preprocess_transcriptomics(sdata_transcripts_no_backed, highly_variable_genes):
    sdata_transcripts_no_backed = preprocess_transcriptomics(
        sdata_transcripts_no_backed,
        labels_layer="segmentation_mask",
        table_layer="table_transcriptomics",
        output_layer="table_transcriptomics",
        highly_variable_genes=highly_variable_genes,
        overwrite=True,
    )
    adata = sdata_transcripts_no_backed.tables["table_transcriptomics"]
    if highly_variable_genes:
        assert adata.shape == (616, 17)
        assert adata.raw.to_adata().shape == (616, 87)
        assert adata.layers["raw_counts"].shape == (616, 17)
    else:
        assert adata.shape == (616, 87)
        assert adata.raw.to_adata().shape == (616, 87)
        assert adata.layers["raw_counts"].shape == (616, 87)
    assert len(adata.obs[_REGION_KEY].cat.categories) == 1
    assert "segmentation_mask" in adata.obs[_REGION_KEY].cat.categories
    assert _CELLSIZE_KEY in adata.obs.columns
    assert "log1p" in adata.uns.keys()
    assert "pca" in adata.uns.keys()
