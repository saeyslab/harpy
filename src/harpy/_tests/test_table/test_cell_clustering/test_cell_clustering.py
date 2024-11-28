import importlib

import dask.array as da
import pandas as pd
import pytest

from sparrow.utils._keys import ClusteringKey


@pytest.mark.skipif(not importlib.util.find_spec("flowsom"), reason="requires the flowSOM library")
def test_cell_clustering(sdata_blobs):
    """Integration test for cell clustering using flowsom"""
    import flowsom as fs

    from sparrow.image.pixel_clustering._clustering import flowsom as flowsom_pixel
    from sparrow.table.cell_clustering._clustering import flowsom as flowsom_cell
    from sparrow.table.cell_clustering._weighted_channel_expression import weighted_channel_expression
    from sparrow.table.pixel_clustering._cluster_intensity import cluster_intensity

    img_layer = "blobs_image"
    labels_layer = "blobs_labels"
    table_layer = "table_cell_clustering"
    table_layer_intensity = "counts_clusters"
    channels = ["lineage_0", "lineage_1", "lineage_5", "lineage_9"]
    fraction = 0.1

    sdata_blobs, _, mapping = flowsom_pixel(
        sdata_blobs,
        img_layer=[img_layer],
        output_layer_clusters=[f"{img_layer}_clusters"],
        output_layer_metaclusters=[f"{img_layer}_metaclusters"],
        channels=channels,
        fraction=fraction,
        n_clusters=20,
        random_state=100,
        chunks=(1, 200, 200),
        overwrite=True,
    )

    sdata_blobs, fsom = flowsom_cell(
        sdata_blobs,
        labels_layer_cells=[labels_layer],
        labels_layer_clusters=[f"{img_layer}_metaclusters"],
        output_layer=table_layer,
        chunks=(200, 200),
        overwrite=True,
    )

    assert table_layer in [*sdata_blobs.tables]

    # check if table_layer is of correct shape
    unique_labels = da.unique(sdata_blobs[labels_layer].data).compute()
    unique_labels = unique_labels[unique_labels != 0]

    unique_clusters = da.unique(sdata_blobs[f"{img_layer}_metaclusters"].data).compute()
    unique_clusters = unique_clusters[unique_clusters != 0]

    assert unique_labels.shape[0] == sdata_blobs.tables[table_layer].shape[0]
    assert unique_clusters.shape[0] == sdata_blobs.tables[table_layer].shape[1]

    assert isinstance(fsom, fs.FlowSOM)
    # check that flowsom adds metaclusters and clusters to table
    assert ClusteringKey._METACLUSTERING_KEY.value in sdata_blobs.tables[table_layer].obs
    assert ClusteringKey._CLUSTERING_KEY.value in sdata_blobs.tables[table_layer].obs

    # check that averages are also added
    assert ClusteringKey._METACLUSTERING_KEY.value in sdata_blobs.tables[table_layer].uns
    assert ClusteringKey._CLUSTERING_KEY.value in sdata_blobs.tables[table_layer].uns

    # check that metacluster and cluster key are of categorical type, needed for visualization in napari-spatialdata
    assert isinstance(sdata_blobs[table_layer].obs[ClusteringKey._METACLUSTERING_KEY.value].dtype, pd.CategoricalDtype)
    assert isinstance(sdata_blobs[table_layer].obs[ClusteringKey._CLUSTERING_KEY.value].dtype, pd.CategoricalDtype)

    # calculate average cluster intensity both for the metaclusters and clusters
    sdata_blobs = cluster_intensity(
        sdata_blobs,
        mapping=mapping,
        img_layer=[img_layer],
        labels_layer=[f"{img_layer}_clusters"],
        output_layer=table_layer_intensity,
        channels=channels,
        overwrite=True,
    )

    sdata_blobs = weighted_channel_expression(
        sdata_blobs,
        table_layer_cell_clustering=table_layer,
        table_layer_pixel_cluster_intensity=table_layer_intensity,
        output_layer=table_layer,
        clustering_key=ClusteringKey._METACLUSTERING_KEY,
        overwrite=True,
    )

    # check that average marker expression for each cell weighted by pixel cluster count are added to .obs
    assert set(channels).issubset(sdata_blobs.tables[table_layer].obs.columns)
    # and average over cell clusters is added to .uns
    assert (
        f"{ClusteringKey._CLUSTERING_KEY.value}_{sdata_blobs[ table_layer_intensity ].var_names.name}"
        in sdata_blobs.tables[table_layer].uns
    )
    assert (
        f"{ClusteringKey._METACLUSTERING_KEY.value}_{sdata_blobs[ table_layer_intensity ].var_names.name}"
        in sdata_blobs.tables[table_layer].uns
    )
