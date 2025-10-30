import numpy as np

from harpy.table._allocation_intensity import allocate_intensity
from harpy.table._cluster_intensity import cluster_intensity
from harpy.utils._keys import _CELLSIZE_KEY


def test_cluster_intensity(sdata_pixie):
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

    assert f"{cluster_key}_weighted_intensity" in sdata[table_layer].uns

    df_calculated = sdata[table_layer].uns[f"{cluster_key}_weighted_intensity"]

    expected_columns = set(sdata[table_layer].var_names) | {cluster_key}
    assert set(df_calculated.columns) == expected_columns
    assert df_calculated.shape == (
        sdata[table_layer].obs[cluster_key].cat.categories.size,
        sdata[table_layer].shape[1] + 1,  # +1 because cluster id also in columns
    )

    # check if we calculated weighted average correctly
    df_test = sdata[table_layer].to_df()
    df_test[_CELLSIZE_KEY] = sdata[table_layer].obs[_CELLSIZE_KEY]
    df_test[cluster_key] = sdata[table_layer].obs[cluster_key]
    data = df_test[df_test[cluster_key] == 5]["HLADR"].values
    weight = df_test[df_test[cluster_key] == 5][_CELLSIZE_KEY].values

    assert np.average(data, weights=weight) == df_calculated[df_calculated[cluster_key] == 5]["HLADR"].item()

    """
    fig_kwargs = {
        "dpi": 100,
    }

    hp.pl.cluster_intensity_heatmap(
        sdata,
        table_layer=table_layer,
        cluster_key=cluster_key,
        z_score=True,
        figsize=(10, 6),
        fig_kwargs=fig_kwargs,
    )
    """
