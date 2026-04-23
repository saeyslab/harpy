import importlib

import dask
import dask.array as da
import numpy as np
import pandas as pd
import pytest
from loguru import logger
from spatialdata.transformations import get_transformation

from harpy.image._image import add_labels
from harpy.table.cell_clustering._preprocess import cell_clustering_preprocess
from harpy.utils._keys import _INSTANCE_KEY, ClusteringKey


@pytest.mark.skipif(not importlib.util.find_spec("flowsom"), reason="requires the flowSOM library")
def test_cell_clustering(sdata_blobs):
    """Integration test for cell clustering using flowsom"""
    import flowsom as fs

    from harpy.image.pixel_clustering._clustering import flowsom as flowsom_pixel
    from harpy.table.cell_clustering._clustering import flowsom as flowsom_cell
    from harpy.table.cell_clustering._weighted_channel_expression import weighted_channel_expression
    from harpy.table.pixel_clustering._cluster_intensity import cluster_intensity_SOM

    batch_model = fs.models.BatchFlowSOMEstimator

    image_name = "blobs_image"
    labels_name = "blobs_labels"
    table_name = "table_cell_clustering"
    table_name_intensity = "counts_clusters"
    channels = ["lineage_0", "lineage_1", "lineage_5", "lineage_9"]
    fraction = 0.1

    with dask.config.set(scheduler="threads"):
        sdata_blobs, _, mapping = flowsom_pixel(
            sdata_blobs,
            image_name=[image_name],
            output_cluster_labels_name=[f"{image_name}_clusters"],
            output_metacluster_labels_name=[f"{image_name}_metaclusters"],
            channels=channels,
            fraction=fraction,
            n_clusters=20,
            random_state=100,
            chunks=(1, 200, 200),
            model=batch_model,
            overwrite=True,
        )

    sdata_blobs, fsom = flowsom_cell(
        sdata_blobs,
        cells_labels_name=[labels_name],
        cluster_labels_name=[f"{image_name}_metaclusters"],
        output_table_name=table_name,
        chunks=(200, 200),
        overwrite=True,
    )

    assert table_name in [*sdata_blobs.tables]

    # check if table_name is of correct shape
    unique_labels = da.unique(sdata_blobs[labels_name].data).compute()
    unique_labels = unique_labels[unique_labels != 0]

    unique_clusters = da.unique(sdata_blobs[f"{image_name}_metaclusters"].data).compute()
    unique_clusters = unique_clusters[unique_clusters != 0]

    assert unique_labels.shape[0] == sdata_blobs.tables[table_name].shape[0]
    assert unique_clusters.shape[0] == sdata_blobs.tables[table_name].shape[1]

    assert isinstance(fsom, fs.FlowSOM)
    # check that flowsom adds metaclusters and clusters to table
    assert ClusteringKey._METACLUSTERING_KEY.value in sdata_blobs.tables[table_name].obs
    assert ClusteringKey._CLUSTERING_KEY.value in sdata_blobs.tables[table_name].obs

    # check that averages are also added
    assert ClusteringKey._METACLUSTERING_KEY.value in sdata_blobs.tables[table_name].uns
    assert ClusteringKey._CLUSTERING_KEY.value in sdata_blobs.tables[table_name].uns

    # check that metacluster and cluster key are of categorical type, needed for visualization in napari-spatialdata
    assert isinstance(sdata_blobs[table_name].obs[ClusteringKey._METACLUSTERING_KEY.value].dtype, pd.CategoricalDtype)
    assert isinstance(sdata_blobs[table_name].obs[ClusteringKey._CLUSTERING_KEY.value].dtype, pd.CategoricalDtype)

    # calculate average cluster intensity both for the metaclusters and clusters
    sdata_blobs = cluster_intensity_SOM(
        sdata_blobs,
        mapping=mapping,
        image_name=[image_name],
        labels_name=[f"{image_name}_clusters"],
        output_table_name=table_name_intensity,
        channels=channels,
        overwrite=True,
    )

    sdata_blobs = weighted_channel_expression(
        sdata_blobs,
        cell_clustering_table_name=table_name,
        table_name_pixel_cluster_intensity=table_name_intensity,
        output_table_name=table_name,
        clustering_key=ClusteringKey._METACLUSTERING_KEY,
        overwrite=True,
    )

    # check that average marker expression for each cell weighted by pixel cluster count are added to .obs
    assert set(channels).issubset(sdata_blobs.tables[table_name].obs.columns)
    # and average over cell clusters is added to .uns
    assert (
        f"{ClusteringKey._CLUSTERING_KEY.value}_{sdata_blobs[table_name_intensity].var_names.name}"
        in sdata_blobs.tables[table_name].uns
    )
    assert (
        f"{ClusteringKey._METACLUSTERING_KEY.value}_{sdata_blobs[table_name_intensity].var_names.name}"
        in sdata_blobs.tables[table_name].uns
    )


def test_cell_clustering_preprocess_logs_removed_no_overlap_cells(sdata_blobs):
    cells_labels_name = "blobs_labels"
    cluster_labels_name = "blobs_metaclusters_for_logging_test"
    output_table_name = "table_cell_clustering_logging_test"

    labels = np.asarray(sdata_blobs[cells_labels_name].data)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]
    dropped_instance_id = int(unique_labels[0])

    # Make one cell overlap only with background/unassigned cluster id 0.
    # clusters is a dummy pixel cluster labels layer we create, for which instance with instance id 1 has no overlap with any pixel cluster.
    clusters = da.from_array(
        np.where((labels != 0) & (labels != dropped_instance_id), 1, 0).astype(np.uint32),
        chunks=labels.shape,
    )
    transformations = get_transformation(sdata_blobs[cells_labels_name], get_all=True)
    sdata_blobs = add_labels(
        sdata_blobs,
        arr=clusters,
        output_labels_name=cluster_labels_name,
        transformations=transformations,
        overwrite=True,
    )

    records = []
    sink_id = logger.add(
        records.append,
        format="{message}",
        filter=lambda record: record["name"] == "harpy.table.cell_clustering._preprocess",
    )
    try:
        sdata_blobs = cell_clustering_preprocess(
            sdata_blobs,
            cells_labels_name=[cells_labels_name],
            cluster_labels_name=[cluster_labels_name],
            output_table_name=output_table_name,
            q=None,
            overwrite=True,
        )
    finally:
        logger.remove(sink_id)

    messages = [record.record["message"] for record in records]

    assert any(
        msg == f"Removing 1 cells with no overlap with any pixel cluster from table '{output_table_name}'."
        for msg in messages
    )
    assert any(
        msg == (f"Removed 1 no-overlap cells for region 'blobs_labels' (instance ids: [{dropped_instance_id}]).")
        for msg in messages
    )

    assert dropped_instance_id not in sdata_blobs.tables[output_table_name].obs[_INSTANCE_KEY].values
