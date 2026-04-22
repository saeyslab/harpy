import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from spatialdata import read_zarr

from harpy.table._add_feature_matrix import add_feature_matrix


def test_add_feature_matrix_creates_new_table(sdata_multi_c_no_backed):
    sdata_multi_c_no_backed = add_feature_matrix(
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

    metadata = adata.uns["feature_matrices"]["cell_features"]
    assert metadata["features"] == ["mean", "area"]
    assert metadata["feature_columns"] == ["mean__0", "mean__4", "area"]
    assert metadata["source_label"] == "masks_whole"
    assert metadata["source_image"] == "raw_image"


def test_add_feature_matrix_creates_intensity_stats_table(sdata_multi_c_no_backed):
    sdata_multi_c_no_backed = add_feature_matrix(
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
    assert adata.uns["feature_matrices"]["intensity_stats"]["feature_columns"] == ["mean__0", "var__0"]


def test_add_feature_matrix_supports_2d_eccentricity_with_intensity_features(sdata_multi_c_no_backed):
    sdata_multi_c_no_backed = add_feature_matrix(
        sdata_multi_c_no_backed,
        labels_layer="masks_whole",
        img_layer="raw_image",
        table_layer=None,
        output_layer="table_mixed_features",
        feature_key="mixed_features",
        features=["mean", "eccentricity"],
        channels=[0],
        overwrite_output_layer=True,
    )

    adata = sdata_multi_c_no_backed.tables["table_mixed_features"]
    instance_key = adata.uns["spatialdata_attrs"]["instance_key"]
    feature_columns = adata.uns["feature_matrices"]["mixed_features"]["feature_columns"]

    expected = pd.DataFrame(
        regionprops_table(
            label_image=sdata_multi_c_no_backed["masks_whole"].data.compute(),
            properties=["label", "eccentricity"],
        )
    )
    expected[instance_key] = expected["label"].astype(int)
    expected = expected.set_index(instance_key).loc[adata.obs[instance_key], "eccentricity"].to_numpy()

    assert adata.obsm["mixed_features"].shape == (adata.n_obs, 2)
    assert feature_columns == ["mean__0", "eccentricity"]
    eccentricity_index = feature_columns.index("eccentricity")
    assert np.allclose(adata.obsm["mixed_features"][:, eccentricity_index], expected)


def test_add_feature_matrix_supports_custom_metadata_key(sdata_multi_c_no_backed):
    sdata_multi_c_no_backed = add_feature_matrix(
        sdata_multi_c_no_backed,
        labels_layer="masks_whole",
        img_layer=None,
        table_layer=None,
        output_layer="table_custom_metadata",
        feature_key="area_features",
        features=["area"],
        feature_matrices_key="custom_feature_matrices",
        overwrite_output_layer=True,
    )

    adata = sdata_multi_c_no_backed.tables["table_custom_metadata"]

    assert "custom_feature_matrices" in adata.uns
    assert "feature_matrices" not in adata.uns
    assert adata.uns["custom_feature_matrices"]["area_features"]["feature_columns"] == ["area"]


def test_add_feature_matrix_existing_table_preserves_other_regions(sdata_pixie_intensities):
    sdata_pixie_intensities = add_feature_matrix(
        sdata_pixie_intensities,
        labels_layer="label_whole_fov0",
        img_layer=None,
        table_layer="table_intensities",
        feature_key="morphology_features",
        features=["area"],
        to_coordinate_system="fov0",
        overwrite_feature_key=True,
    )
    sdata_pixie_intensities = add_feature_matrix(
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
    assert adata.uns["feature_matrices"]["morphology_features"]["feature_columns"] == ["area"]


def test_add_feature_matrix_persists_backed_updates(sdata_multi_c):
    sdata_multi_c = add_feature_matrix(
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
    assert adata.uns["feature_matrices"]["area_features"]["feature_columns"].tolist() == ["area"]
