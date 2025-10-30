import os

import numpy as np

from harpy.plot._cluster_intensity import cluster_intensity_heatmap
from harpy.table._allocation_intensity import allocate_intensity
from harpy.table._cluster_intensity import cluster_intensity
from harpy.utils._keys import _CELLSIZE_KEY


def test_cluster_intensity_heatmap(sdata_pixie, tmp_path):
    sdata = sdata_pixie

    img_layer = ["raw_image_fov0", "raw_image_fov1"]
    labels_layer = ["label_whole_fov0", "label_whole_fov1"]
    table_layer = "table_intensities"
    to_coordinate_system = ["fov0", "fov1"]
    cluster_key = "cluster_id"

    for _img_layer, _labels_layer, _to_coordinate_system in zip(
        img_layer, labels_layer, to_coordinate_system, strict=True
    ):
        sdata = allocate_intensity(
            sdata,
            img_layer=_img_layer,
            labels_layer=_labels_layer,
            output_layer=table_layer,
            mode="mean",
            to_coordinate_system=_to_coordinate_system,
            append=True,
            overwrite=True,
        )

    # add a dummy cluster id
    n_obs = sdata[table_layer].shape[0]
    RNG = np.random.default_rng(seed=42)
    sdata[table_layer].obs[cluster_key] = RNG.choice(range(10), size=n_obs)
    sdata[table_layer].obs[cluster_key] = sdata[table_layer].obs[cluster_key].astype("category")

    sdata = cluster_intensity(
        sdata,
        table_layer=table_layer,
        labels_layer=labels_layer,
        cluster_key=cluster_key,
        cluster_key_uns=f"{cluster_key}_weighted_intensity",
        output_layer=table_layer,
        instance_size_key=_CELLSIZE_KEY,
    )

    fig_kwargs = {
        "dpi": 100,
    }

    cluster_intensity_heatmap(
        sdata,
        table_layer=table_layer,
        cluster_key=cluster_key,
        cluster_key_uns=f"{cluster_key}_weighted_intensity",
        z_score=True,
        figsize=(10, 6),
        fig_kwargs=fig_kwargs,
        output=os.path.join(tmp_path, "weighted_intensity_heatmap.png"),
    )
