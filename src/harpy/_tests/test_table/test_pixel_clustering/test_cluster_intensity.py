import importlib.util

import pytest

from sparrow.table.pixel_clustering._cluster_intensity import _export_to_ark_format, cluster_intensity
from sparrow.utils._keys import ClusteringKey


@pytest.mark.skipif(not importlib.util.find_spec("flowsom"), reason="requires the flowSOM library")
def test_cluster_intensity(sdata_blobs):
    import flowsom as fs

    from sparrow.image.pixel_clustering._clustering import flowsom

    img_layer = "blobs_image"
    channels = ["lineage_0", "lineage_1", "lineage_5", "lineage_9"]
    fraction = 0.1

    sdata_blobs, fsom, mapping = flowsom(
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

    sdata_blobs = cluster_intensity(
        sdata_blobs,
        mapping=mapping,
        img_layer=img_layer,
        labels_layer=f"{img_layer}_clusters",
        output_layer="counts_clusters",
        overwrite=True,
    )

    assert isinstance(fsom, fs.FlowSOM)
    assert "counts_clusters" in sdata_blobs.tables
    # avg intensity per metacluster saved in .uns
    assert ClusteringKey._METACLUSTERING_KEY.value in sdata_blobs.tables["counts_clusters"].uns
    df = _export_to_ark_format(sdata_blobs["counts_clusters"], output=None)
    assert df.shape[0] == sdata_blobs.tables["counts_clusters"].shape[0]
