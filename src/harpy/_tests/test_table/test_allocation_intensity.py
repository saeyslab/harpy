import numpy as np
import pytest
from anndata import AnnData
from spatialdata import SpatialData
from spatialdata.models import TableModel
from xrspatial import zonal_stats

from harpy.image.segmentation._align_masks import align_labels_layers
from harpy.table._allocation_intensity import (
    allocate_intensity,
)
from harpy.table._regionprops import add_regionprop_features


def test_integration_allocate_intensity(sdata_multi_c_no_backed: SpatialData):
    # integration test for process of aligning masks, allocate intensities and add regionprop features to
    # sdata.tables["table_intensities"].obs

    sdata_multi_c_no_backed = align_labels_layers(
        sdata_multi_c_no_backed,
        labels_layer_1="masks_nuclear",
        labels_layer_2="masks_whole",
        output_labels_layer="masks_nuclear_aligned",
        output_shapes_layer=None,
        overwrite=True,
        chunks=256,
        depth=100,
    )

    assert "masks_nuclear_aligned" in sdata_multi_c_no_backed.labels

    sdata_multi_c_no_backed = allocate_intensity(
        sdata_multi_c_no_backed,
        img_layer="raw_image",
        labels_layer="masks_whole",
        output_layer="table_intensities",
        obs_stats=("max"),
        chunks=100,
        append=False,
        overwrite=True,
    )

    sdata_multi_c_no_backed = allocate_intensity(
        sdata_multi_c_no_backed,
        img_layer="raw_image",
        labels_layer="masks_nuclear_aligned",
        output_layer="table_intensities",
        obs_stats=("max"),
        chunks=100,
        append=True,
        overwrite=True,
    )

    sdata_multi_c_no_backed = add_regionprop_features(
        sdata_multi_c_no_backed,
        labels_layer=["masks_whole", "masks_nuclear_aligned"],
        table_layer="table_intensities",
        output_layer="table_intensities",
        overwrite=True,
    )

    assert isinstance(sdata_multi_c_no_backed, SpatialData)

    assert isinstance(sdata_multi_c_no_backed.tables["table_intensities"], AnnData)

    assert sdata_multi_c_no_backed.tables["table_intensities"].shape == (1299, 22)

    channel_0 = sdata_multi_c_no_backed["raw_image"].c.data[0]
    assert f"max_{channel_0}" in sdata_multi_c_no_backed.tables["table_intensities"].obs.columns


def test_allocate_intensity(sdata_multi_c_no_backed: SpatialData):
    sdata_multi_c_no_backed = allocate_intensity(
        sdata_multi_c_no_backed,
        img_layer="raw_image",
        labels_layer="masks_whole",
        output_layer="table_intensities",
        mode="mean",
        chunks=100,
        append=False,
        channels=[0, 4, 5],
        overwrite=True,
    )

    assert isinstance(sdata_multi_c_no_backed, SpatialData)

    # check if calculated values are same as the ones obtained via zonal_stats (used by spatialdata)
    # note zonal_stats is much slower than allocate_intensity implementation
    df = zonal_stats(
        sdata_multi_c_no_backed["masks_whole"],
        sdata_multi_c_no_backed["raw_image"][0],
        stats_funcs=["mean"],
    ).compute()
    assert np.allclose(df["mean"].values[1:], sdata_multi_c_no_backed["table_intensities"].to_df()["0"].values)

    assert isinstance(sdata_multi_c_no_backed.tables["table_intensities"], AnnData)


def test_allocate_intensity_overwrite(sdata_multi_c: SpatialData):
    sdata_multi_c = allocate_intensity(
        sdata_multi_c,
        img_layer="raw_image",
        labels_layer="masks_whole",
        output_layer="table_intensities",
        append=False,
        overwrite=True,
    )

    with pytest.raises(
        ValueError,
        # match=r"Attempting to overwrite 'sdata\.tables\[\\"table_intensities\\"\]', but overwrite is set to False\. Set overwrite to True to overwrite the \.zarr store\.",
        match=r'Attempting to overwrite \'sdata\.tables\["table_intensities"\]\', but overwrite is set to False. Set overwrite to True to overwrite the \.zarr store.',
    ):
        # unit test with append to True, and overwrite to False, which should not be allowed
        sdata_multi_c = allocate_intensity(
            sdata_multi_c,
            img_layer="raw_image",
            labels_layer="masks_nuclear_aligned",
            output_layer="table_intensities",
            chunks=512,
            append=True,
            overwrite=False,
        )


def test_allocate_intensity_raises_instance_key(sdata_pixie: SpatialData):
    instance_key = "my_instance_key"
    region_key = "my_region_key"
    instance_size_key = "instance_size"
    sdata_pixie = allocate_intensity(
        sdata_pixie,
        img_layer="raw_image_fov0",
        labels_layer="label_whole_fov0",
        to_coordinate_system="fov0",
        output_layer="my_table",
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
        sdata_pixie = allocate_intensity(
            sdata_pixie,
            img_layer="raw_image_fov1",
            labels_layer="label_whole_fov1",
            to_coordinate_system="fov1",
            output_layer="my_table",
            mode="sum",
            obs_stats="count",
            region_key=region_key,
            instance_key=instance_key,
            append=True,
            overwrite=True,
        )


def test_allocate_intensity_raises_region_key(sdata_pixie: SpatialData):
    instance_key = "my_instance_key"
    region_key = "my_region_key"
    instance_size_key = "instance_size"
    sdata_pixie = allocate_intensity(
        sdata_pixie,
        img_layer="raw_image_fov0",
        labels_layer="label_whole_fov0",
        to_coordinate_system="fov0",
        output_layer="my_table",
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
        sdata_pixie = allocate_intensity(
            sdata_pixie,
            img_layer="raw_image_fov1",
            labels_layer="label_whole_fov1",
            to_coordinate_system="fov1",
            output_layer="my_table",
            mode="sum",
            obs_stats="count",
            region_key=region_key,
            instance_key=instance_key,
            instance_size_key=instance_size_key,
            append=True,
            overwrite=True,
        )
