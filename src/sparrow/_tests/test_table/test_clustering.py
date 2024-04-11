import importlib.util

import pytest

from sparrow.table._clustering import kmeans, leiden
from sparrow.table._preprocess import preprocess_proteomics


@pytest.mark.skipif(not importlib.util.find_spec("sklearn"), reason="requires the scikit-learn library")
def test_sklearn(sdata_multi_c):
    from sklearn.cluster import KMeans

    X = sdata_multi_c.tables["table_intensities"].X
    result = KMeans(n_clusters=2, random_state=0).fit(X)
    assert len(result.labels_) == len(X)


def test_leiden(sdata_multi_c):
    assert "leiden" not in sdata_multi_c.tables["table_intensities"].obs.columns
    sdata_multi_c = leiden(
        sdata_multi_c,
        labels_layer="masks_whole",
        table_layer="table_intensities",
        output_layer="table_intensities_clustered",
        key_added="leiden",
        index_names_var=["0", "2", "5", "20"],
        random_state=100,
        overwrite=True,
    )
    assert "leiden" in sdata_multi_c.tables["table_intensities_clustered"].obs.columns
    assert sdata_multi_c["table_intensities_clustered"].shape == (674, 4)
    assert sdata_multi_c["table_intensities_clustered"].var.index.to_list() == ["0", "2", "5", "20"]

    sdata_multi_c = leiden(
        sdata_multi_c,
        labels_layer="masks_whole",
        table_layer="table_intensities",
        output_layer="table_intensities_clustered",
        key_added="leiden",
        index_positions_var=[0, 2, 5],
        random_state=100,
        overwrite=True,
    )
    assert "leiden" in sdata_multi_c.tables["table_intensities_clustered"].obs.columns
    assert sdata_multi_c["table_intensities_clustered"].shape == (674, 3)
    assert sdata_multi_c["table_intensities_clustered"].var.index.to_list() == ["0", "2", "5"]


def test_kmeans(sdata_multi_c):
    assert "kmeans" not in sdata_multi_c.tables["table_intensities"].obs.columns
    sdata_multi_c = kmeans(
        sdata_multi_c,
        labels_layer="masks_whole",
        table_layer="table_intensities",
        output_layer="table_intensities_clustered",
        key_added="kmeans",
        random_state=100,
        overwrite=True,
    )
    assert "kmeans" in sdata_multi_c.tables["table_intensities_clustered"].obs.columns


def test_integration_clustering(sdata_multi_c):
    assert "leiden" not in sdata_multi_c.tables["table_intensities"].obs.columns

    sdata_multi_c = preprocess_proteomics(
        sdata_multi_c,
        labels_layer="masks_whole",
        table_layer="table_intensities",
        output_layer="table_intensities_preprocessed",
        overwrite=True,
    )

    sdata_multi_c = leiden(
        sdata_multi_c,
        labels_layer="masks_whole",
        table_layer="table_intensities_preprocessed",
        output_layer="table_intensities_preprocessed",
        key_added="leiden",
        random_state=100,
        overwrite=True,
    )

    assert "leiden" in sdata_multi_c.tables["table_intensities_preprocessed"].obs.columns


@pytest.mark.skipif(not importlib.util.find_spec("flowsom"), reason="requires the flowSOM library")
def test_flowsom(sdata_multi_c):
    from flowsom import FlowSOM

    adata = sdata_multi_c.tables["table_intensities"]
    adata.var["channel"] = adata.var.index
    adata.var["marker"] = adata.var.index
    fsom = FlowSOM(adata, cols_to_use=[0, 1], xdim=10, ydim=10, n_clus=10)

    assert len(fsom.mudata) == len(adata)
