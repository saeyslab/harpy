from sparrow.table._preprocess import preprocess_proteomics, preprocess_transcriptomics
from sparrow.utils._keys import _CELLSIZE_KEY, _REGION_KEY


def test_preprocess_proteomics(sdata_multi_c):
    sdata_multi_c = preprocess_proteomics(
        sdata_multi_c,
        labels_layer="masks_whole",
        table_layer="table_intensities",
        output_layer="table_intensities_preprocessed",
        overwrite=True,
    )

    assert sdata_multi_c["table_intensities"].shape == (1299, 22)
    assert sdata_multi_c["table_intensities_preprocessed"].shape == (674, 22)

    adata = sdata_multi_c.tables["table_intensities_preprocessed"]
    assert len(adata.obs[_REGION_KEY].cat.categories) == 1
    assert "masks_whole" in adata.obs[_REGION_KEY].cat.categories
    assert _CELLSIZE_KEY in adata.obs.columns
    assert "log1p" in adata.uns.keys()


def test_preprocess_proteomics_multiple_samples(sdata_multi_c):
    # instead of one labels layer, user could give up multiple labels layers (linked to sdata.tables[table_layer] via the _REGION_KEY),
    # which could represent multiple samples that need to be preprocessed together.
    # in this dummy example, we preprocess cell masks and corresponding nuclear masks together, which does not make much sense in practice.
    sdata_multi_c = preprocess_proteomics(
        sdata_multi_c,
        labels_layer=["masks_whole", "masks_nuclear_aligned"],
        table_layer="table_intensities",
        output_layer="table_intensities_preprocessed",
        overwrite=True,
    )

    assert sdata_multi_c["table_intensities"].shape == (1299, 22)
    assert sdata_multi_c["table_intensities_preprocessed"].shape == (1299, 22)

    adata = sdata_multi_c.tables["table_intensities_preprocessed"]
    assert len(adata.obs[_REGION_KEY].cat.categories) == 2
    assert "masks_whole" in adata.obs[_REGION_KEY].cat.categories
    assert "masks_nuclear_aligned" in adata.obs[_REGION_KEY].cat.categories
    assert _CELLSIZE_KEY in adata.obs.columns
    assert "log1p" in adata.uns.keys()


def test_preprocess_proteomics_overwrite(sdata_multi_c):
    sdata_multi_c = preprocess_proteomics(
        sdata_multi_c,
        labels_layer="masks_whole",
        table_layer="table_intensities",
        output_layer="table_intensities",
        overwrite=True,
    )
    # running preprocess takes cells corresponding to certain labels_layer from sdata.table.
    adata = sdata_multi_c.tables["table_intensities"]
    assert adata.shape == (674, 22)
    assert len(adata.obs[_REGION_KEY].cat.categories) == 1
    assert "masks_whole" in adata.obs[_REGION_KEY].cat.categories
    assert _CELLSIZE_KEY in adata.obs.columns
    assert "log1p" in adata.uns.keys()


def test_preprocess_transcriptomics(sdata_transcripts):
    sdata_transcripts = preprocess_transcriptomics(
        sdata_transcripts,
        labels_layer="segmentation_mask",
        table_layer="table_transcriptomics",
        output_layer="table_transcriptomics",
        overwrite=True,
    )

    adata = sdata_transcripts.tables["table_transcriptomics"]
    assert len(adata.obs[_REGION_KEY].cat.categories) == 1
    assert "segmentation_mask" in adata.obs[_REGION_KEY].cat.categories
    assert _CELLSIZE_KEY in adata.obs.columns
    assert "log1p" in adata.uns.keys()
    assert "pca" in adata.uns.keys()
