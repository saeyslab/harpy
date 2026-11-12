import dask.array as da
import numpy as np
import pandas as pd
import pytest
from scipy import ndimage
from skimage.measure import regionprops_table
from spatialdata import read_zarr

from harpy.image._image import add_image, add_labels
from harpy.table._add_feature_matrix import add_feature_matrix
from harpy.utils._keys import _INSTANCE_KEY


@pytest.mark.parametrize(
    ("labels_name", "image_name", "features"),
    [
        (["masks_whole", "masks_empty"], "raw_image", ["mean", "area"]),
        ("masks_empty", None, ["area"]),
        ("masks_empty", None, ["eccentricity"]),
    ],
)
def test_add_feature_matrix_rejects_labels_without_instances(
    sdata_multi_c_no_backed, labels_name, image_name, features
):
    shape = sdata_multi_c_no_backed["masks_whole"].shape
    sdata_multi_c_no_backed = add_labels(
        sdata_multi_c_no_backed, da.zeros(shape, dtype=np.int32, chunks=256), output_labels_name="masks_empty"
    )

    with pytest.raises(ValueError, match=r"Labels element 'masks_empty' contains no instances, only background"):
        add_feature_matrix(
            sdata_multi_c_no_backed,
            labels_name=labels_name,
            image_name=image_name,
            table_name=None,
            output_table_name="table_features",
            feature_key="features",
            features=features,
            channels=[0],
        )
    assert "table_features" not in sdata_multi_c_no_backed.tables


def test_add_feature_matrix_3d_labels(sdata_multi_c_no_backed):
    # two identical z-planes: every instance has twice the 2D area and the same 2D mean
    mask = sdata_multi_c_no_backed["masks_whole"].data
    image = sdata_multi_c_no_backed["raw_image"].data
    sdata_multi_c_no_backed = add_labels(
        sdata_multi_c_no_backed, da.stack([mask, mask]), output_labels_name="masks_3d", dims=("z", "y", "x")
    )
    sdata_multi_c_no_backed = add_image(
        sdata_multi_c_no_backed,
        da.stack([image, image], axis=1),
        output_image_name="image_3d",
        dims=("c", "z", "y", "x"),
        c_coords=[str(channel) for channel in sdata_multi_c_no_backed["raw_image"].c.data],
    )

    def features(features, image_name):
        sdata = add_feature_matrix(
            sdata_multi_c_no_backed,
            labels_name="masks_3d",
            image_name=image_name,
            table_name=None,
            output_table_name="table_3d",
            feature_key="features",
            features=features,
            channels=[0],
        )
        return sdata.tables["table_3d"]

    adata = features(["mean", "area"], "image_3d")
    labels = adata.obs[_INSTANCE_KEY].to_numpy()
    mask_2d, image_2d = mask.compute(), image[0].compute()
    expected = np.column_stack(
        [ndimage.mean(image_2d, mask_2d, labels), 2 * ndimage.sum_labels(np.ones(mask_2d.shape), mask_2d, labels)]
    )
    np.testing.assert_allclose(np.asarray(adata.obsm["features"]), expected, rtol=1e-6)

    with pytest.raises(ValueError, match=r"\['perimeter'\] are not supported for 3D labels data"):
        features(["area", "perimeter"], None)


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
    assert set(metadata) == {
        "schema_version",
        "source_kind",
        "feature_columns",
        "source_channels",
        "features",
        "regions",
    }
    assert metadata["features"] == ["mean", "area"]
    assert metadata["feature_columns"] == ["mean__0", "mean__4", "area"]
    assert metadata["source_kind"] == "harpy_add_feature_matrix"
    assert metadata["regions"] == {"masks_whole": {"image_element": "raw_image", "coordinate_system": "global"}}
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


def test_add_feature_matrix_rechunks_labels_with_other_chunk_boundaries(sdata_multi_c_no_backed):
    def intensity_stats(output_table_name):
        sdata = add_feature_matrix(
            sdata_multi_c_no_backed,
            labels_name="masks_whole",
            image_name="raw_image",
            table_name=None,
            output_table_name=output_table_name,
            feature_key="intensity_stats",
            features=["mean", "var"],
            channels=[0],
            overwrite_output_table=True,
        )
        return sdata.tables[output_table_name].obsm["intensity_stats"]

    reference = intensity_stats("table_reference")

    # Same chunk size, other chunk boundaries, as a crop that does not start on the chunk grid gives.
    shifted = (100, 256, 156)
    sdata_multi_c_no_backed["raw_image"] = sdata_multi_c_no_backed["raw_image"].chunk({"y": shifted, "x": shifted})
    sdata_multi_c_no_backed["masks_whole"] = sdata_multi_c_no_backed["masks_whole"].chunk({"y": 256, "x": 256})

    np.testing.assert_allclose(intensity_stats("table_shifted_chunks"), reference, rtol=1e-6)


def test_add_feature_matrix_intensity_features_follow_feature_and_channel_order(sdata_multi_c_no_backed):
    # The values of the statistics are tested in test_moments.py; this checks where add_feature_matrix puts
    # them: which channels, which columns of .obsm, and which rows. Channels out of order catch a sorted order.
    features = ["max", "mean"]
    channels = [4, 0]
    sdata_multi_c_no_backed = add_feature_matrix(
        sdata_multi_c_no_backed,
        labels_name="masks_whole",
        image_name="raw_image",
        table_name=None,
        output_table_name="table_features",
        feature_key="intensity",
        features=features,
        channels=channels,
        overwrite_output_table=True,
    )
    adata = sdata_multi_c_no_backed.tables["table_features"]
    feature_columns = adata.uns["feature_matrices"]["intensity"]["feature_columns"]
    assert feature_columns == ["max__4", "max__0", "mean__4", "mean__0"]

    image = sdata_multi_c_no_backed["raw_image"].data.compute()
    mask = sdata_multi_c_no_backed["masks_whole"].data.compute()
    labels = adata.obs[_INSTANCE_KEY].to_numpy()
    reference = {"max": ndimage.maximum, "mean": ndimage.mean}
    expected = np.column_stack(
        [reference[feature](image[channel], labels=mask, index=labels) for feature in features for channel in channels]
    )

    np.testing.assert_allclose(np.asarray(adata.obsm["intensity"]), expected, rtol=1e-6)


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


@pytest.mark.parametrize(
    ("image_name", "features"),
    [(None, ["area"]), ("raw_image", ["mean", "area"])],
)
def test_add_feature_matrix_area_matches_regionprops(sdata_multi_c_no_backed, image_name, features):
    sdata_multi_c_no_backed = add_feature_matrix(
        sdata_multi_c_no_backed,
        labels_name="masks_whole",
        image_name=image_name,
        table_name=None,
        output_table_name="table_area",
        feature_key="area_features",
        features=features,
        channels=[0],
        overwrite_output_table=True,
    )

    adata = sdata_multi_c_no_backed.tables["table_area"]
    feature_columns = adata.uns["feature_matrices"]["area_features"]["feature_columns"]
    expected = pd.DataFrame(
        regionprops_table(
            label_image=sdata_multi_c_no_backed["masks_whole"].data.compute(), properties=["label", "area"]
        )
    )
    expected = expected.set_index("label").loc[adata.obs[_INSTANCE_KEY], "area"].to_numpy()

    np.testing.assert_array_equal(adata.obsm["area_features"][:, feature_columns.index("area")], expected)


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
    assert adata.uns["custom_feature_matrices"]["area_features"]["source_kind"] == "harpy_add_feature_matrix"
    assert adata.uns["custom_feature_matrices"]["area_features"]["source_channels"] is None


def test_add_feature_matrix_rejects_table_rows_missing_from_labels(sdata_multi_c_no_backed):
    sdata_multi_c_no_backed = add_feature_matrix(
        sdata_multi_c_no_backed,
        labels_name="masks_whole",
        image_name=None,
        table_name=None,
        output_table_name="table_features",
        feature_key="area_features",
        features=["area"],
    )
    # remove two instances from the labels, so that the table and the labels are out of sync
    removed = sdata_multi_c_no_backed.tables["table_features"].obs[_INSTANCE_KEY].to_numpy()[[3, 7]]
    labels = sdata_multi_c_no_backed["masks_whole"]
    sdata_multi_c_no_backed["masks_whole"] = labels.copy(data=da.where(da.isin(labels.data, removed), 0, labels.data))

    message = (
        rf"2 rows of table 'table_features' have instance IDs that do not occur in labels element 'masks_whole' "
        rf"\(e\.g\. {removed[0]}, {removed[1]}\)\. The labels and the table are out of sync"
    )
    with pytest.raises(ValueError, match=message):
        add_feature_matrix(
            sdata_multi_c_no_backed,
            labels_name="masks_whole",
            image_name="raw_image",
            table_name="table_features",
            feature_key="mean_features",
            features=["mean"],
            channels=[0],
        )
    assert "mean_features" not in sdata_multi_c_no_backed.tables["table_features"].obsm


def test_add_feature_matrix_existing_table_keeps_its_rows(sdata_multi_c_no_backed):
    # Instances of the labels without a table row, e.g. after a QC filter, get no row.
    sdata_multi_c_no_backed = add_feature_matrix(
        sdata_multi_c_no_backed,
        labels_name="masks_whole",
        image_name=None,
        table_name=None,
        output_table_name="table_features",
        feature_key="area_features",
        features=["area"],
    )
    table = sdata_multi_c_no_backed.tables["table_features"]
    sdata_multi_c_no_backed.tables["table_features"] = table[np.arange(table.n_obs) % 3 != 0].copy()
    kept_obs_names = sdata_multi_c_no_backed.tables["table_features"].obs_names.copy()

    sdata_multi_c_no_backed = add_feature_matrix(
        sdata_multi_c_no_backed,
        labels_name="masks_whole",
        image_name="raw_image",
        table_name="table_features",
        feature_key="mean_features",
        features=["mean"],
        channels=[0],
    )

    adata = sdata_multi_c_no_backed.tables["table_features"]
    pd.testing.assert_index_equal(adata.obs_names, kept_obs_names)
    labels = adata.obs[_INSTANCE_KEY].to_numpy()
    expected = ndimage.mean(
        sdata_multi_c_no_backed["raw_image"].data[0].compute(),
        sdata_multi_c_no_backed["masks_whole"].data.compute(),
        labels,
    )
    np.testing.assert_allclose(np.asarray(adata.obsm["mean_features"])[:, 0], expected, rtol=1e-6)


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
    assert metadata["regions"] == {
        "label_whole_fov0": {"image_element": "raw_image_fov0", "coordinate_system": "fov0"},
        "label_whole_fov1": {"image_element": "raw_image_fov1", "coordinate_system": "fov1"},
    }


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


@pytest.mark.parametrize("feature_key", ["spatial_canonical", "spatial"])
def test_add_feature_matrix_rejects_a_reserved_feature_key(sdata_multi_c_no_backed, feature_key):
    """``.obsm`` keys that tools read as coordinates cannot hold a feature matrix."""
    tables = set(sdata_multi_c_no_backed.tables)

    with pytest.raises(ValueError, match="reserved .obsm key"):
        add_feature_matrix(
            sdata_multi_c_no_backed,
            labels_name="masks_whole",
            image_name=None,
            table_name=None,
            output_table_name="table_reserved",
            feature_key=feature_key,
            features=["area"],
        )

    assert set(sdata_multi_c_no_backed.tables) == tables


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
    assert adata.uns["feature_matrices"]["area_features"]["source_kind"] == "harpy_add_feature_matrix"
    assert adata.uns["feature_matrices"]["area_features"]["source_channels"] is None
    # A morphology matrix has no image; read back from the store, its None stays None.
    assert adata.uns["feature_matrices"]["area_features"]["regions"] == {
        "masks_whole": {"image_element": None, "coordinate_system": "global"}
    }
    assert intensity_adata.uns["feature_matrices"]["mean_features"]["source_kind"] == "harpy_add_feature_matrix"
    assert intensity_adata.uns["feature_matrices"]["mean_features"]["source_channels"].tolist() == ["0"]
