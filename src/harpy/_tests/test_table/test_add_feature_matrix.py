import numpy as np
import pandas as pd
import pytest
from skimage.measure import regionprops_table
from spatialdata import read_zarr

from harpy.table._add_feature_matrix import add_feature_matrix


def test_add_feature_matrix_creates_new_table(sdata_multi_c_no_backed):
    sdata_multi_c_no_backed = add_feature_matrix(
        sdata_multi_c_no_backed,
        labels_name="masks_whole",
        image_name="raw_image",
        table_name=None,
        output_table_name="table_features",
        feature_key="cell_features",
        features=["mean", "area"],
        channels=[0, 4],
        overwrite_output_table=True,
    )

    adata = sdata_multi_c_no_backed.tables["table_features"]

    assert adata.n_obs == 674
    assert adata.X is None
    assert adata.obsm["cell_features"].shape == (adata.n_obs, 3)
    assert np.isfinite(adata.obsm["cell_features"]).all()

    metadata = adata.uns["feature_matrices"]["cell_features"]
    assert metadata["features"] == ["mean", "area"]
    assert metadata["feature_columns"] == ["mean__0", "mean__4", "area"]
    assert metadata["source_label"] == ["masks_whole"]
    assert metadata["source_image"] == ["raw_image"]
    assert metadata["source_channels"] == ["0", "4"]


def test_add_feature_matrix_creates_intensity_stats_table(sdata_multi_c_no_backed):
    sdata_multi_c_no_backed = add_feature_matrix(
        sdata_multi_c_no_backed,
        labels_name="masks_whole",
        image_name="raw_image",
        table_name=None,
        output_table_name="table_intensity_stats",
        feature_key="intensity_stats",
        features=["mean", "var"],
        channels=[0],
        overwrite_output_table=True,
    )

    adata = sdata_multi_c_no_backed.tables["table_intensity_stats"]

    assert adata.obsm["intensity_stats"].shape == (adata.n_obs, 2)
    assert np.isfinite(adata.obsm["intensity_stats"]).all()
    assert adata.uns["feature_matrices"]["intensity_stats"]["feature_columns"] == ["mean__0", "var__0"]
    assert adata.uns["feature_matrices"]["intensity_stats"]["source_channels"] == ["0"]


def test_add_feature_matrix_rechunks_labels_when_chunks_differ(sdata_multi_c_no_backed):
    # Reference matrix computed when image and labels share chunks.
    reference = (
        add_feature_matrix(
            sdata_multi_c_no_backed,
            labels_name="masks_whole",
            image_name="raw_image",
            table_name=None,
            output_table_name="table_reference",
            feature_key="intensity_stats",
            features=["mean", "var"],
            channels=[0],
            overwrite_output_table=True,
        )
        .tables["table_reference"]
        .obsm["intensity_stats"]
    )

    # Give the labels a different spatial chunk size than the image.
    sdata_multi_c_no_backed["masks_whole"] = sdata_multi_c_no_backed["masks_whole"].chunk({"y": 256, "x": 256})

    # Previously this raised in RasterAggregator ("Please rechunk"); now labels are
    # auto-aligned onto the image's spatial chunks.
    sdata_multi_c_no_backed = add_feature_matrix(
        sdata_multi_c_no_backed,
        labels_name="masks_whole",
        image_name="raw_image",
        table_name=None,
        output_table_name="table_mismatched_chunks",
        feature_key="intensity_stats",
        features=["mean", "var"],
        channels=[0],
        overwrite_output_table=True,
    )

    matrix = sdata_multi_c_no_backed.tables["table_mismatched_chunks"].obsm["intensity_stats"]

    assert np.isfinite(matrix).all()
    # Rechunking must not change the computed values.
    assert np.allclose(matrix, reference)


def test_add_feature_matrix_stores_all_source_channels_when_channels_is_none(sdata_multi_c_no_backed):
    sdata_multi_c_no_backed = add_feature_matrix(
        sdata_multi_c_no_backed,
        labels_name="masks_whole",
        image_name="raw_image",
        table_name=None,
        output_table_name="table_all_channels",
        feature_key="all_channels",
        features=["mean"],
        channels=None,
        overwrite_output_table=True,
    )

    adata = sdata_multi_c_no_backed.tables["table_all_channels"]
    expected_channels = [str(channel) for channel in sdata_multi_c_no_backed["raw_image"].c.data]
    metadata = adata.uns["feature_matrices"]["all_channels"]

    assert metadata["source_channels"] == expected_channels
    assert metadata["feature_columns"] == [f"mean__{channel}" for channel in expected_channels]


def test_add_feature_matrix_supports_2d_eccentricity_with_intensity_features(sdata_multi_c_no_backed):
    sdata_multi_c_no_backed = add_feature_matrix(
        sdata_multi_c_no_backed,
        labels_name="masks_whole",
        image_name="raw_image",
        table_name=None,
        output_table_name="table_mixed_features",
        feature_key="mixed_features",
        features=["mean", "eccentricity"],
        channels=[0],
        overwrite_output_table=True,
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
        labels_name="masks_whole",
        image_name=None,
        table_name=None,
        output_table_name="table_custom_metadata",
        feature_key="area_features",
        features=["area"],
        feature_matrices_key="custom_feature_matrices",
        overwrite_output_table=True,
    )

    adata = sdata_multi_c_no_backed.tables["table_custom_metadata"]

    assert "custom_feature_matrices" in adata.uns
    assert "feature_matrices" not in adata.uns
    assert adata.uns["custom_feature_matrices"]["area_features"]["feature_columns"] == ["area"]
    assert adata.uns["custom_feature_matrices"]["area_features"]["source_channels"] is None


def test_add_feature_matrix_existing_table_preserves_other_regions(sdata_pixie_intensities):
    sdata_pixie_intensities = add_feature_matrix(
        sdata_pixie_intensities,
        labels_name="label_whole_fov0",
        image_name=None,
        table_name="table_intensities",
        feature_key="morphology_features",
        features=["area"],
        to_coordinate_system="fov0",
        overwrite_feature_key=True,
    )
    sdata_pixie_intensities = add_feature_matrix(
        sdata_pixie_intensities,
        labels_name="label_whole_fov1",
        image_name=None,
        table_name="table_intensities",
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


def test_add_feature_matrix_multiple_pairs_share_flat_source_channels(sdata_pixie_intensities):
    sdata_pixie_intensities = add_feature_matrix(
        sdata_pixie_intensities,
        labels_name=["label_whole_fov0", "label_whole_fov1"],
        image_name=["raw_image_fov0", "raw_image_fov1"],
        table_name=None,
        output_table_name="table_multi_pair",
        feature_key="mean_features",
        features=["mean"],
        channels=["CD14"],
        to_coordinate_system=["fov0", "fov1"],
        overwrite_output_table=True,
    )

    metadata = sdata_pixie_intensities.tables["table_multi_pair"].uns["feature_matrices"]["mean_features"]

    # Channels are shared across samples, so the metadata is a single flat list, not a list of lists.
    assert metadata["source_channels"] == ["CD14"]
    assert metadata["feature_columns"] == ["mean__CD14"]
    assert metadata["source_label"] == ["label_whole_fov0", "label_whole_fov1"]


def test_add_feature_matrix_rejects_mismatched_channels_across_pairs(sdata_pixie_intensities):
    image = sdata_pixie_intensities["raw_image_fov1"]
    sdata_pixie_intensities["raw_image_fov1"] = image.assign_coords(c=[f"alt_{name}" for name in image.c.data])

    with pytest.raises(ValueError, match="same channels"):
        add_feature_matrix(
            sdata_pixie_intensities,
            labels_name=["label_whole_fov0", "label_whole_fov1"],
            image_name=["raw_image_fov0", "raw_image_fov1"],
            table_name=None,
            output_table_name="table_mismatched_channels",
            feature_key="mean_features",
            features=["mean"],
            channels=None,
            to_coordinate_system=["fov0", "fov1"],
            overwrite_output_table=True,
        )


def test_add_feature_matrix_persists_backed_updates(sdata_multi_c):
    sdata_multi_c = add_feature_matrix(
        sdata_multi_c,
        labels_name="masks_whole",
        image_name=None,
        table_name=None,
        output_table_name="table_feature_matrix",
        feature_key="area_features",
        features=["area"],
        overwrite_output_table=True,
    )
    sdata_multi_c = add_feature_matrix(
        sdata_multi_c,
        labels_name="masks_whole",
        image_name="raw_image",
        table_name=None,
        output_table_name="table_intensity_feature_matrix",
        feature_key="mean_features",
        features=["mean"],
        channels=[0],
        overwrite_output_table=True,
    )

    reloaded = read_zarr(sdata_multi_c.path)
    adata = reloaded.tables["table_feature_matrix"]
    intensity_adata = reloaded.tables["table_intensity_feature_matrix"]

    assert "area_features" in adata.obsm
    assert adata.obsm["area_features"].shape == (adata.n_obs, 1)
    assert adata.uns["feature_matrices"]["area_features"]["feature_columns"].tolist() == ["area"]
    assert adata.uns["feature_matrices"]["area_features"]["source_channels"] is None
    assert intensity_adata.uns["feature_matrices"]["mean_features"]["source_channels"].tolist() == ["0"]
