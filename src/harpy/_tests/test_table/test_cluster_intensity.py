import numpy as np

from harpy.table._allocation_intensity import allocate_intensity
from harpy.table._cluster_intensity import cluster_intensity
from harpy.utils._keys import _CELLSIZE_KEY


def test_cluster_intensity(sdata_pixie):
    sdata = sdata_pixie

    image_name = ["raw_image_fov0", "raw_image_fov1"]
    labels_name = ["label_whole_fov0", "label_whole_fov1"]
    table_name = "table_intensities"
    to_coordinate_system = ["fov0", "fov1"]
    cluster_key = "cluster_id"

    for _image_name, _labels_name, _to_coordinate_system in zip(
        image_name, labels_name, to_coordinate_system, strict=True
    ):
        sdata = allocate_intensity(
            sdata,
            image_name=_image_name,
            labels_name=_labels_name,
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

    assert f"{cluster_key}_weighted_intensity" in sdata[table_name].uns

    df_calculated = sdata[table_name].uns[f"{cluster_key}_weighted_intensity"]

    expected_columns = set(sdata[table_name].var_names) | {cluster_key}
    assert set(df_calculated.columns) == expected_columns
    assert df_calculated.shape == (
        sdata[table_name].obs[cluster_key].cat.categories.size,
        sdata[table_name].shape[1] + 1,  # +1 because cluster id also in columns
    )

    # check if we calculated weighted average correctly
    df_test = sdata[table_name].to_df()
    df_test[_CELLSIZE_KEY] = sdata[table_name].obs[_CELLSIZE_KEY]
    df_test[cluster_key] = sdata[table_name].obs[cluster_key]
    data = df_test[df_test[cluster_key] == 5]["HLADR"].values
    weight = df_test[df_test[cluster_key] == 5][_CELLSIZE_KEY].values

    assert np.average(data, weights=weight) == df_calculated[df_calculated[cluster_key] == 5]["HLADR"].item()

    """
    fig_kwargs = {
        "dpi": 100,
    }

    hp.pl.cluster_intensity_heatmap(
        sdata,
        table_name=table_name,
        cluster_key=cluster_key,
        z_score=True,
        figsize=(10, 6),
        fig_kwargs=fig_kwargs,
    )
    """
