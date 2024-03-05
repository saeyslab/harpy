from sparrow.table._preprocess import preprocess_proteomics, preprocess_transcriptomics
from sparrow.utils._keys import _CELLSIZE_KEY, _REGION_KEY


def test_preprocess_proteomics(sdata_multi_c):
    sdata_multi_c = preprocess_proteomics(sdata_multi_c, labels_layer="masks_whole")
    # running preprocess takes cells corresponding to certain labels_layer from sdata.table.
    assert len(sdata_multi_c.table.obs[_REGION_KEY].cat.categories) == 1
    assert "masks_whole" in sdata_multi_c.table.obs[_REGION_KEY].cat.categories
    assert _CELLSIZE_KEY in sdata_multi_c.table.obs.columns
    assert "log1p" in sdata_multi_c.table.uns.keys()


def test_preprocess_transcriptomics(sdata_multi_c):
    sdata_multi_c = preprocess_transcriptomics(sdata_multi_c, labels_layer="masks_whole")
    assert len(sdata_multi_c.table.obs[_REGION_KEY].cat.categories) == 1
    assert "masks_whole" in sdata_multi_c.table.obs[_REGION_KEY].cat.categories
    assert _CELLSIZE_KEY in sdata_multi_c.table.obs.columns
    assert "log1p" in sdata_multi_c.table.uns.keys()
    assert "pca" in sdata_multi_c.table.uns.keys()
