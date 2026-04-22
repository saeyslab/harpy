import os

import numpy as np

from harpy.plot._cluster_intensity import cluster_intensity_heatmap
from harpy.table._allocation_intensity import allocate_intensity
from harpy.table._cluster_intensity import cluster_intensity
from harpy.utils._keys import _CELLSIZE_KEY


def test_cluster_intensity_heatmap(sdata_pixie, tmp_path):
    sdata = sdata_pixie

    image_name = ["raw_image_fov0", "raw_image_fov1"]
    labels_name = ["label_whole_fov0", "label_whole_fov1"]
    table_name = "table_intensities"
    to_coordinate_system = ["fov0", "fov1"]
    cluster_key = "cluster_id"

    for _img_layer, _labels_layer, _to_coordinate_system in zip(
        image_name, labels_name, to_coordinate_system, strict=True
    ):
        sdata = allocate_intensity(
            sdata,
            image_name=_img_layer,
            labels_name=_labels_layer,
            output_table_name=table_name,
            mode="mean",
            to_coordinate_system=_to_coordinate_system,
            append=True,
            overwrite=True,
        )

    # add a dummy cluster id
    n_obs = sdata[table_name].shape[0]
    RNG = np.random.default_rng(seed=42)
    sdata[table_name].obs[cluster_key] = RNG.choice(range(10), size=n_obs)
    sdata[table_name].obs[cluster_key] = sdata[table_name].obs[cluster_key].astype("category")

    sdata = cluster_intensity(
        sdata,
        table_name=table_name,
        labels_name=labels_name,
        cluster_key=cluster_key,
        cluster_key_uns=f"{cluster_key}_weighted_intensity",
        output_table_name=table_name,
        instance_size_key=_CELLSIZE_KEY,
    )

    fig_kwargs = {
        "dpi": 100,
    }

    cluster_intensity_heatmap(
        sdata,
        table_name=table_name,
        cluster_key=cluster_key,
        cluster_key_uns=f"{cluster_key}_weighted_intensity",
        z_score=True,
        figsize=(10, 6),
        fig_kwargs=fig_kwargs,
        output=os.path.join(tmp_path, "weighted_intensity_heatmap.png"),
    )
