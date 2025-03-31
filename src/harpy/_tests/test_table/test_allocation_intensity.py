import numpy as np
import pytest
from anndata import AnnData
from spatialdata import SpatialData
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
        sdata_multi_c_no_backed, labels_layer="masks_whole", table_layer="table_intensities"
    )

    sdata_multi_c_no_backed = add_regionprop_features(
        sdata_multi_c_no_backed, labels_layer="masks_nuclear_aligned", table_layer="table_intensities"
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
