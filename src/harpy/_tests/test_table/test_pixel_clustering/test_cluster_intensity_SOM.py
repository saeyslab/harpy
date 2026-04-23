import importlib.util

import dask
import pytest

from harpy.table.pixel_clustering._cluster_intensity import _export_to_ark_format, cluster_intensity_SOM
from harpy.utils._keys import ClusteringKey


@pytest.mark.skipif(not importlib.util.find_spec("flowsom"), reason="requires the flowSOM library")
def test_cluster_intensity(sdata_blobs):
    import flowsom as fs

    from harpy.image.pixel_clustering._clustering import flowsom

    batch_model = fs.models.BatchFlowSOMEstimator

    image_name = "blobs_image"
    channels = ["lineage_0", "lineage_1", "lineage_5", "lineage_9"]
    fraction = 0.1

    with dask.config.set(scheduler="threads"):
        sdata_blobs, fsom, mapping = flowsom(
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

    sdata_blobs = cluster_intensity_SOM(
        sdata_blobs,
        mapping=mapping,
        image_name=image_name,
        labels_name=f"{image_name}_clusters",
        output_table_name="counts_clusters",
        overwrite=True,
    )

    assert isinstance(fsom, fs.FlowSOM)
    assert "counts_clusters" in sdata_blobs.tables
    # avg intensity per metacluster saved in .uns
    assert ClusteringKey._METACLUSTERING_KEY.value in sdata_blobs.tables["counts_clusters"].uns
    df = _export_to_ark_format(sdata_blobs["counts_clusters"], output=None)
    assert df.shape[0] == sdata_blobs.tables["counts_clusters"].shape[0]
