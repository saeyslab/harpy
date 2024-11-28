import importlib.util

import pytest

from harpy.table._clustering import kmeans, leiden
from harpy.table._preprocess import preprocess_proteomics


@pytest.mark.skipif(not importlib.util.find_spec("sklearn"), reason="requires the scikit-learn library")
def test_sklearn(sdata_multi_c_no_backed):
    from sklearn.cluster import KMeans

    X = sdata_multi_c_no_backed.tables["table_intensities"].X
    result = KMeans(n_clusters=2, random_state=0).fit(X)
    assert len(result.labels_) == len(X)


def test_leiden(sdata_multi_c_no_backed):
    assert "leiden" not in sdata_multi_c_no_backed.tables["table_intensities"].obs.columns
    sdata_multi_c_no_backed = leiden(
        sdata_multi_c_no_backed,
        labels_layer="masks_whole",
        table_layer="table_intensities",
        output_layer="table_intensities_clustered",
        key_added="leiden",
        index_names_var=["0", "2", "5", "20"],
        random_state=100,
        overwrite=True,
    )
    assert "leiden" in sdata_multi_c_no_backed.tables["table_intensities_clustered"].obs.columns
    assert sdata_multi_c_no_backed["table_intensities_clustered"].shape == (674, 4)
    assert sdata_multi_c_no_backed["table_intensities_clustered"].var.index.to_list() == ["0", "2", "5", "20"]

    sdata_multi_c_no_backed = leiden(
        sdata_multi_c_no_backed,
        labels_layer="masks_whole",
        table_layer="table_intensities",
        output_layer="table_intensities_clustered",
        key_added="leiden",
        index_positions_var=[0, 2, 5],
        random_state=100,
        overwrite=True,
    )
    assert "leiden" in sdata_multi_c_no_backed.tables["table_intensities_clustered"].obs.columns
    assert sdata_multi_c_no_backed["table_intensities_clustered"].shape == (674, 3)
    assert sdata_multi_c_no_backed["table_intensities_clustered"].var.index.to_list() == ["0", "2", "5"]


def test_kmeans(sdata_multi_c_no_backed):
    assert "kmeans" not in sdata_multi_c_no_backed.tables["table_intensities"].obs.columns
    sdata_multi_c_no_backed = kmeans(
        sdata_multi_c_no_backed,
        labels_layer="masks_whole",
        table_layer="table_intensities",
        output_layer="table_intensities_clustered",
        key_added="kmeans",
        random_state=100,
        overwrite=True,
    )
    assert "kmeans" in sdata_multi_c_no_backed.tables["table_intensities_clustered"].obs.columns


def test_integration_clustering(sdata_multi_c_no_backed):
    assert "leiden" not in sdata_multi_c_no_backed.tables["table_intensities"].obs.columns

    sdata_multi_c_no_backed = preprocess_proteomics(
        sdata_multi_c_no_backed,
        labels_layer="masks_whole",
        table_layer="table_intensities",
        output_layer="table_intensities_preprocessed",
        overwrite=True,
    )

    sdata_multi_c_no_backed = leiden(
        sdata_multi_c_no_backed,
        labels_layer="masks_whole",
        table_layer="table_intensities_preprocessed",
        output_layer="table_intensities_preprocessed",
        key_added="leiden",
        random_state=100,
        overwrite=True,
    )

    assert "leiden" in sdata_multi_c_no_backed.tables["table_intensities_preprocessed"].obs.columns
