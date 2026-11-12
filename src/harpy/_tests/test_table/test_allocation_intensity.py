import numpy as np
import pytest
import zarr
from anndata import AnnData
from scipy import ndimage
from spatialdata import SpatialData
from spatialdata.models import TableModel
from xrspatial import zonal_stats

import harpy as hp
import harpy._storage._anndata as anndata_storage
import harpy.table._allocation_intensity as image_aggregation_module
from harpy.image.segmentation._align_masks import align_labels
from harpy.table._allocation_intensity import aggregate_image
from harpy.table._regionprops import add_regionprops
from harpy.utils._keys import _CELLSIZE_KEY, _INSTANCE_KEY, _REGION_KEY, _SPATIAL


def test_allocate_intensity_import_warns_and_resolves_to_aggregate_image():
    message = r"harpy\.tb\.allocate_intensity is deprecated .* Use harpy\.tb\.aggregate_image, which takes the same arguments\."
    for namespace in (hp.tb, image_aggregation_module):
        with pytest.warns(FutureWarning, match=message) as record:
            assert namespace.allocate_intensity is aggregate_image
        assert [warning.filename for warning in record] == [__file__]


def test_integration_aggregate_image(sdata_multi_c_no_backed: SpatialData):
    # Integration test for aligning masks, aggregating image intensities, and adding region properties to
    # sdata.tables["table_intensities"].obs

    sdata_multi_c_no_backed = align_labels(
        sdata_multi_c_no_backed,
        labels_name_1="masks_nuclear",
        labels_name_2="masks_whole",
        output_labels_name="masks_nuclear_aligned",
        output_shapes_name=None,
        overwrite=True,
        chunks=256,
        depth=100,
    )

    assert "masks_nuclear_aligned" in sdata_multi_c_no_backed.labels

    sdata_multi_c_no_backed = aggregate_image(
        sdata_multi_c_no_backed,
        image_name="raw_image",
        labels_name="masks_whole",
        output_table_name="table_intensities",
        obs_stats=("max"),
        chunks=100,
        append=False,
        overwrite=True,
    )

    sdata_multi_c_no_backed = aggregate_image(
        sdata_multi_c_no_backed,
        image_name="raw_image",
        labels_name="masks_nuclear_aligned",
        output_table_name="table_intensities",
        obs_stats=("max"),
        chunks=100,
        append=True,
        overwrite=True,
    )

    sdata_multi_c_no_backed = add_regionprops(
        sdata_multi_c_no_backed,
        labels_name=["masks_whole", "masks_nuclear_aligned"],
        table_name="table_intensities",
        output_table_name="table_intensities",
        overwrite=True,
    )

    assert isinstance(sdata_multi_c_no_backed, SpatialData)

    assert isinstance(sdata_multi_c_no_backed.tables["table_intensities"], AnnData)

    assert sdata_multi_c_no_backed.tables["table_intensities"].shape == (1299, 22)

    channel_0 = sdata_multi_c_no_backed["raw_image"].c.data[0]
    assert f"max_{channel_0}" in sdata_multi_c_no_backed.tables["table_intensities"].obs.columns


def test_aggregate_image(sdata_multi_c_no_backed: SpatialData):
    sdata_multi_c_no_backed = aggregate_image(
        sdata_multi_c_no_backed,
        image_name="raw_image",
        labels_name="masks_whole",
        output_table_name="table_intensities",
        mode="mean",
        chunks=100,
        append=False,
        channels=[0, 4, 5],
        overwrite=True,
    )

    assert isinstance(sdata_multi_c_no_backed, SpatialData)

    # check if calculated values are same as the ones obtained via zonal_stats (used by spatialdata)
    # ``zonal_stats`` is much slower than the ``aggregate_image`` implementation.
    df = zonal_stats(
        sdata_multi_c_no_backed["masks_whole"],
        sdata_multi_c_no_backed["raw_image"][0],
        stats_funcs=["mean"],
    ).compute()
    assert np.allclose(df["mean"].values[1:], sdata_multi_c_no_backed["table_intensities"].to_df()["0"].values)

    assert isinstance(sdata_multi_c_no_backed.tables["table_intensities"], AnnData)


def test_aggregate_image_rechunks_labels_when_chunks_differ(sdata_multi_c_no_backed: SpatialData):
    # Reference values computed when image and labels share chunks.
    reference = (
        aggregate_image(
            sdata_multi_c_no_backed,
            image_name="raw_image",
            labels_name="masks_whole",
            output_table_name="table_reference",
            mode="mean",
            append=False,
            channels=[0],
            overwrite=True,
        )
        .tables["table_reference"]
        .to_df()["0"]
        .values
    )

    # Give the labels a different spatial chunk size than the image.
    sdata_multi_c_no_backed["masks_whole"] = sdata_multi_c_no_backed["masks_whole"].chunk({"y": 256, "x": 256})

    # Previously this raised in RasterAggregator ("Please rechunk"); now the aggregator's
    # labels are auto-aligned onto the image's spatial chunks.
    sdata_multi_c_no_backed = aggregate_image(
        sdata_multi_c_no_backed,
        image_name="raw_image",
        labels_name="masks_whole",
        output_table_name="table_mismatched_chunks",
        mode="mean",
        append=False,
        channels=[0],
        overwrite=True,
    )

    values = sdata_multi_c_no_backed.tables["table_mismatched_chunks"].to_df()["0"].values

    # Rechunking must not change the computed values.
    assert np.allclose(values, reference)


def test_aggregate_image_rechunks_labels_with_other_chunk_boundaries(sdata_multi_c_no_backed: SpatialData):
    reference = (
        aggregate_image(
            sdata_multi_c_no_backed,
            image_name="raw_image",
            labels_name="masks_whole",
            output_table_name="table_reference",
            channels=[0],
            overwrite=True,
        )
        .tables["table_reference"]
        .to_df()["0"]
        .values
    )

    # Same chunk size, other chunk boundaries, as a crop that does not start on the chunk grid gives.
    shifted = (100, 256, 156)
    sdata_multi_c_no_backed["raw_image"] = sdata_multi_c_no_backed["raw_image"].chunk({"y": shifted, "x": shifted})
    sdata_multi_c_no_backed["masks_whole"] = sdata_multi_c_no_backed["masks_whole"].chunk({"y": 256, "x": 256})

    sdata_multi_c_no_backed = aggregate_image(
        sdata_multi_c_no_backed,
        image_name="raw_image",
        labels_name="masks_whole",
        output_table_name="table_shifted_chunks",
        channels=[0],
        overwrite=True,
    )

    values = sdata_multi_c_no_backed.tables["table_shifted_chunks"].to_df()["0"].values
    np.testing.assert_allclose(values, reference, rtol=1e-6)


def test_aggregate_image_places_statistics_and_centers(sdata_multi_c_no_backed: SpatialData):
    # The values of the statistics are tested in test_moments.py; this checks where aggregate_image puts them:
    # which channels, which columns of .X and .obs, which rows, and the (x, y) order in .obsm. Channels out of
    # order catch a sorted order; "max" first in obs_stats checks that max and min come last in .obs.
    channels = [4, 0]
    sdata_multi_c_no_backed = aggregate_image(
        sdata_multi_c_no_backed,
        image_name="raw_image",
        labels_name="masks_whole",
        output_table_name="table_intensities",
        channels=channels,
        mode="mean",
        obs_stats=["max", "count", "sum"],
        chunks=100,
        overwrite=True,
    )
    adata = sdata_multi_c_no_backed.tables["table_intensities"]
    assert list(adata.var_names) == ["4", "0"]
    statistics = [_CELLSIZE_KEY, "sum_4", "sum_0", "max_4", "max_0"]
    assert [column for column in adata.obs.columns if column in statistics] == statistics

    image = sdata_multi_c_no_backed["raw_image"].data.compute()
    mask = sdata_multi_c_no_backed["masks_whole"].data.compute()
    labels = adata.obs[_INSTANCE_KEY].to_numpy()

    def per_channel(statistic):
        return np.column_stack([statistic(image[channel], labels=mask, index=labels) for channel in channels])

    np.testing.assert_allclose(np.asarray(adata.X), per_channel(ndimage.mean), rtol=1e-6)
    np.testing.assert_allclose(adata.obs[["sum_4", "sum_0"]], per_channel(ndimage.sum_labels), rtol=1e-6)
    np.testing.assert_array_equal(adata.obs[["max_4", "max_0"]], per_channel(ndimage.maximum))
    np.testing.assert_array_equal(adata.obs[_CELLSIZE_KEY], ndimage.sum_labels(np.ones(mask.shape), mask, labels))
    # .obsm holds (x, y); ndimage gives (y, x)
    centers = np.array(ndimage.center_of_mass(np.ones(mask.shape), mask, labels))
    np.testing.assert_allclose(adata.obsm[_SPATIAL], centers[:, ::-1], rtol=1e-12)


@pytest.mark.parametrize(
    ("obs_stats", "expected_columns"),
    [(["mean", "count"], [_CELLSIZE_KEY]), (["mean"], []), ([], [])],
)
def test_aggregate_image_obs_stats_containing_the_mode(
    sdata_multi_c_no_backed: SpatialData, obs_stats: list[str], expected_columns: list[str]
):
    # the mode goes to .X only; the other statistics in obs_stats still go to .obs
    sdata_multi_c_no_backed = aggregate_image(
        sdata_multi_c_no_backed,
        image_name="raw_image",
        labels_name="masks_whole",
        output_table_name="table_intensities",
        mode="mean",
        obs_stats=obs_stats,
        channels=[0],
        overwrite=True,
    )
    obs = sdata_multi_c_no_backed.tables["table_intensities"].obs
    assert [column for column in obs.columns if column not in (_INSTANCE_KEY, _REGION_KEY)] == expected_columns
    if expected_columns:
        mask = sdata_multi_c_no_backed["masks_whole"].data.compute()
        assert obs[_CELLSIZE_KEY].sum() == np.count_nonzero(mask)


def test_aggregate_image_overwrite(sdata_multi_c: SpatialData):
    sdata_multi_c = aggregate_image(
        sdata_multi_c,
        image_name="raw_image",
        labels_name="masks_whole",
        output_table_name="table_intensities",
        append=False,
        overwrite=True,
    )

    with pytest.raises(
        ValueError,
        # match=r"Attempting to overwrite 'sdata\.tables\[\\"table_intensities\\"\]', but overwrite is set to False\. Set overwrite to True to overwrite the \.zarr store\.",
        match=r'Attempting to overwrite \'sdata\.tables\["table_intensities"\]\', but overwrite is set to False. Set overwrite to True to overwrite the \.zarr store.',
    ):
        # unit test with append to True, and overwrite to False, which should not be allowed
        sdata_multi_c = aggregate_image(
            sdata_multi_c,
            image_name="raw_image",
            labels_name="masks_nuclear_aligned",
            output_table_name="table_intensities",
            chunks=512,
            append=True,
            overwrite=False,
        )


def test_aggregate_image_stores_intensities_in_row_only_chunks(sdata_multi_c: SpatialData, monkeypatch):
    """aggregate_image writes through add_table, so its dense table gets row-only stored chunks."""
    monkeypatch.setattr(anndata_storage, "_STORED_CHUNK_BYTES", 1024)

    sdata_multi_c = aggregate_image(
        sdata_multi_c,
        image_name="raw_image",
        labels_name="masks_whole",
        output_table_name="table_intensities",
        append=False,
        overwrite=True,
    )

    matrix = zarr.open_group(str(sdata_multi_c.path / "tables" / "table_intensities"), mode="r")["X"]
    bytes_per_row = matrix.shape[1] * matrix.dtype.itemsize
    assert matrix.shape[0] > 1024 // bytes_per_row
    assert matrix.chunks == (max(1024 // bytes_per_row, 1), matrix.shape[1])


def test_aggregate_image_raises_instance_key(sdata_pixie: SpatialData):
    instance_key = "my_instance_key"
    region_key = "my_region_key"
    instance_size_key = "instance_size"
    sdata_pixie = aggregate_image(
        sdata_pixie,
        image_name="raw_image_fov0",
        labels_name="label_whole_fov0",
        to_coordinate_system="fov0",
        output_table_name="my_table",
        mode="sum",
        obs_stats="count",
        region_key=region_key,
        instance_key=instance_key,
        instance_size_key=instance_size_key,
        overwrite=True,
    )

    assert instance_key == sdata_pixie.tables["my_table"].uns[TableModel.ATTRS_KEY][TableModel.INSTANCE_KEY]
    assert region_key == sdata_pixie.tables["my_table"].uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY]
    assert instance_size_key in sdata_pixie.tables["my_table"].obs.columns

    instance_key = f"{instance_key}_test"

    with pytest.raises(
        ValueError,
        match=f"Provided instance key '{instance_key}' is different than the instance key of the AnnData object you wish to append to.*This is not allowed",
    ):
        sdata_pixie = aggregate_image(
            sdata_pixie,
            image_name="raw_image_fov1",
            labels_name="label_whole_fov1",
            to_coordinate_system="fov1",
            output_table_name="my_table",
            mode="sum",
            obs_stats="count",
            region_key=region_key,
            instance_key=instance_key,
            append=True,
            overwrite=True,
        )


def test_aggregate_image_raises_region_key(sdata_pixie: SpatialData):
    instance_key = "my_instance_key"
    region_key = "my_region_key"
    instance_size_key = "instance_size"
    sdata_pixie = aggregate_image(
        sdata_pixie,
        image_name="raw_image_fov0",
        labels_name="label_whole_fov0",
        to_coordinate_system="fov0",
        output_table_name="my_table",
        mode="sum",
        obs_stats="count",
        region_key=region_key,
        instance_key=instance_key,
        instance_size_key=instance_size_key,
        overwrite=True,
    )

    assert instance_key == sdata_pixie.tables["my_table"].uns[TableModel.ATTRS_KEY][TableModel.INSTANCE_KEY]
    assert region_key == sdata_pixie.tables["my_table"].uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY]
    assert instance_size_key in sdata_pixie.tables["my_table"].obs.columns

    region_key = f"{region_key}_test"

    with pytest.raises(
        ValueError,
        match=f"Provided region key '{region_key}' is different than the region key of the AnnData object you wish to append to.*This is not allowed",
    ):
        sdata_pixie = aggregate_image(
            sdata_pixie,
            image_name="raw_image_fov1",
            labels_name="label_whole_fov1",
            to_coordinate_system="fov1",
            output_table_name="my_table",
            mode="sum",
            obs_stats="count",
            region_key=region_key,
            instance_key=instance_key,
            instance_size_key=instance_size_key,
            append=True,
            overwrite=True,
        )
