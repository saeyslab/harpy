import importlib
import os

import matplotlib
import matplotlib.pyplot as plt
import pytest

from harpy.plot._flowsom import pixel_clusters, pixel_clusters_heatmap


@pytest.mark.skipif(not importlib.util.find_spec("flowsom"), reason="requires the flowSOM library")
def test_plot_pixel_clusters(sdata_blobs, tmp_path):
    from harpy.image.pixel_clustering._clustering import flowsom
    from harpy.table.pixel_clustering._cluster_intensity import cluster_intensity

    matplotlib.use("Agg")

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

    assert f"{img_layer}_clusters" in sdata_blobs.labels
    assert f"{img_layer}_metaclusters" in sdata_blobs.labels
    assert fsom.model._is_fitted

    pixel_clusters(
        sdata_blobs,
        labels_layer=f"{img_layer}_metaclusters",
        figsize=(10, 10),
        coordinate_systems="global",
        crd=(100, 300, 100, 300),
        output=os.path.join(tmp_path, "pixel_metaclusters.png"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(20, 5))

    pixel_clusters(
        sdata_blobs,
        labels_layer=f"{img_layer}_clusters",
        crd=(100, 300, 100, 300),
        ax=axes[0],
    )

    pixel_clusters(
        sdata_blobs,
        labels_layer=f"{img_layer}_metaclusters",
        crd=(100, 300, 100, 300),
        ax=axes[1],
    )

    fig.savefig(os.path.join(tmp_path, "pixel_clusters_metaclusters"), bbox_inches="tight")
    plt.close()

    sdata_blobs = cluster_intensity(
        sdata_blobs,
        mapping=mapping,
        img_layer=[img_layer],
        labels_layer=[f"{img_layer}_clusters"],
        to_coordinate_system=["global"],
        output_layer="counts_clusters",
        chunks="auto",
        overwrite=True,
    )

    fig, axes = plt.subplots(2, 1, figsize=(30, 40))

    pixel_clusters_heatmap(
        sdata_blobs,
        table_layer="counts_clusters",
        ax=axes[0],
        metaclusters=True,
        linewidths=0.01,
        z_score=True,
    )

    pixel_clusters_heatmap(
        sdata_blobs,
        table_layer="counts_clusters",
        ax=axes[1],
        metaclusters=False,
        linewidths=0.01,
        z_score=True,
    )

    fig.savefig(os.path.join(tmp_path, "pixel_clusters_heatmap"), bbox_inches="tight")
    plt.close()
