import numpy as np
from spatialdata import read_zarr

from harpy.table._feature_matrix import feature_matrix


def test_feature_matrix_creates_new_table(sdata_multi_c_no_backed):
    sdata_multi_c_no_backed = feature_matrix(
        sdata_multi_c_no_backed,
        labels_layer="masks_whole",
        img_layer="raw_image",
        table_layer=None,
        output_layer="table_features",
        feature_key="cell_features",
        features=["mean", "area"],
        channels=[0, 4],
        overwrite_output_layer=True,
    )

    adata = sdata_multi_c_no_backed.tables["table_features"]

    assert adata.n_obs == 674
    assert adata.X is None
    assert adata.obsm["cell_features"].shape == (adata.n_obs, 3)
    assert np.isfinite(adata.obsm["cell_features"]).all()

    metadata = adata.uns["harpy_feature_matrices"]["cell_features"]
    assert metadata["features"] == ["mean", "area"]
    assert metadata["feature_columns"] == ["mean__0", "mean__4", "area"]
    assert metadata["source_label"] == "masks_whole"
    assert metadata["source_image"] == "raw_image"


def test_feature_matrix_creates_intensity_stats_table(sdata_multi_c_no_backed):
    sdata_multi_c_no_backed = feature_matrix(
        sdata_multi_c_no_backed,
        labels_layer="masks_whole",
        img_layer="raw_image",
        table_layer=None,
        output_layer="table_intensity_stats",
        feature_key="intensity_stats",
        features=["mean", "var"],
        channels=[0],
        overwrite_output_layer=True,
    )

    adata = sdata_multi_c_no_backed.tables["table_intensity_stats"]

    assert adata.obsm["intensity_stats"].shape == (adata.n_obs, 2)
    assert np.isfinite(adata.obsm["intensity_stats"]).all()
    assert adata.uns["harpy_feature_matrices"]["intensity_stats"]["feature_columns"] == ["mean__0", "var__0"]


def test_feature_matrix_existing_table_preserves_other_regions(sdata_pixie_intensities):
    sdata_pixie_intensities = feature_matrix(
        sdata_pixie_intensities,
        labels_layer="label_whole_fov0",
        img_layer=None,
        table_layer="table_intensities",
        feature_key="morphology_features",
        features=["area"],
        to_coordinate_system="fov0",
        overwrite_feature_key=True,
    )
    sdata_pixie_intensities = feature_matrix(
        sdata_pixie_intensities,
        labels_layer="label_whole_fov1",
        img_layer=None,
        table_layer="table_intensities",
        feature_key="morphology_features",
        features=["area"],
        to_coordinate_system="fov1",
        overwrite_feature_key=True,
    )

    adata = sdata_pixie_intensities.tables["table_intensities"]
    region_key = adata.uns["spatialdata_attrs"]["region_key"]
    matrix = adata.obsm["morphology_features"]

    fov0_mask = adata.obs[region_key] == "label_whole_fov0"
    fov1_mask = adata.obs[region_key] == "label_whole_fov1"

    assert np.isfinite(matrix[fov0_mask]).all()
    assert np.isfinite(matrix[fov1_mask]).all()
    assert adata.uns["harpy_feature_matrices"]["morphology_features"]["feature_columns"] == ["area"]


def test_feature_matrix_persists_backed_updates(sdata_multi_c):
    sdata_multi_c = feature_matrix(
        sdata_multi_c,
        labels_layer="masks_whole",
        img_layer=None,
        table_layer=None,
        output_layer="table_feature_matrix",
        feature_key="area_features",
        features=["area"],
        overwrite_output_layer=True,
    )

    reloaded = read_zarr(sdata_multi_c.path)
    adata = reloaded.tables["table_feature_matrix"]

    assert "area_features" in adata.obsm
    assert adata.obsm["area_features"].shape == (adata.n_obs, 1)
    assert adata.uns["harpy_feature_matrices"]["area_features"]["feature_columns"].tolist() == ["area"]
