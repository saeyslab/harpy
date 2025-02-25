import re

import dask.array as da
import numpy as np
import pytest
from scipy import ndimage
from scipy.stats import kurtosis, skew
from skimage.measure import regionprops_table
from xrspatial import zonal_stats

from harpy.utils._aggregate import RasterAggregator, _get_center_of_mass, _get_mask_area, _region_radii_and_axes
from harpy.utils._keys import _CELLSIZE_KEY, _INSTANCE_KEY


def test_aggregate_stats(sdata):
    se_image = sdata["blobs_image"]
    se_labels = sdata["blobs_labels"]

    image = se_image.data[:, None, ...]
    mask = se_labels.data[None, ...]

    aggregator = RasterAggregator(
        mask_dask_array=mask.rechunk(512),
        image_dask_array=image.rechunk(512),
    )
    stats_funcs = ("sum", "mean", "count", "var", "kurtosis", "skew")
    dfs = aggregator.aggregate_stats(stats_funcs=stats_funcs)

    assert len(dfs) == len(stats_funcs)

    for df in dfs:
        assert df.shape == (len(aggregator._labels), image.shape[0] + 1)  # +1 for the _INSTANCE_KEY column

    stats_funcs = ("sum", "wrong_stat")

    with pytest.raises(
        AssertionError,
        match=r"^Invalid statistic function",
    ):
        dfs = aggregator.aggregate_stats(stats_funcs=stats_funcs)


def test_aggregate_sum_dask_array():
    chunk_size = (2, 2)

    mask_dask_array = da.from_array(
        np.array([[3, 0, 0, 0], [0, 3, 1, 1], [1, 0, 1, 1], [1, 1, 0, 0]]),
        chunks=chunk_size,
    )
    mask_dask_array = mask_dask_array[None, ...]

    float_dask_array = da.from_array(
        np.array(
            [
                [0.5, 1.5, 2.5, 3.5],
                [4.5, 5.5, 6.5, 7.5],
                [8.5, 9.5, 10.5, 11.5],
                [12.5, 13.5, 14.5, 15.5],
            ]
        ),
        chunks=chunk_size,
    )

    float_dask_array = float_dask_array[None, None, ...]

    aggregator = RasterAggregator(mask_dask_array=mask_dask_array, image_dask_array=float_dask_array)

    df_sum = aggregator.aggregate_sum()

    expected_result = np.array([51.5, 70.5, 6.0])

    assert np.array_equal(df_sum[0].values, expected_result)


def test_aggregate_sum(sdata):
    se_image = sdata["blobs_image"]
    se_labels = sdata["blobs_labels"]

    image = se_image.data[:, None, ...]
    mask = se_labels.data[None, ...]

    aggregator = RasterAggregator(
        mask_dask_array=mask.rechunk(512),
        image_dask_array=image.rechunk(512),
    )
    df_sum = aggregator.aggregate_sum()

    assert df_sum.shape[1] - 1 == image.shape[0]

    # check if we get same result as scipy

    scipy_sum = ndimage.sum_labels(input=image[0].compute(), labels=mask.compute(), index=da.unique(mask).compute())

    assert np.allclose(df_sum[0].values, scipy_sum, rtol=0, atol=1e-5)

    scipy_sum = ndimage.sum_labels(input=image[2].compute(), labels=mask.compute(), index=da.unique(mask).compute())

    assert np.allclose(df_sum[2].values, scipy_sum, rtol=0, atol=1e-5)

    # check if we get same results as xrspatial

    xrspatial_sum = zonal_stats(
        values=se_image[0],
        zones=se_labels,
        stats_funcs=["sum"],
    )

    xrspatial_sum = xrspatial_sum.compute()

    assert np.allclose(df_sum[0].values, xrspatial_sum["sum"].values, rtol=0, atol=1e-5)


def test_aggregate_min_max(sdata):
    se_image = sdata["blobs_image"]
    se_labels = sdata["blobs_labels"]

    image = se_image.data[:, None, ...]
    mask = se_labels.data[None, ...]

    aggregator = RasterAggregator(
        mask_dask_array=mask.rechunk(512),
        image_dask_array=image.rechunk(512),
    )
    df_max = aggregator.aggregate_max()

    assert df_max.shape[1] - 1 == image.shape[0]

    scipy_max = ndimage.labeled_comprehension(
        input=image[0].compute(),
        labels=mask.compute(),
        index=da.unique(mask).compute(),
        func=np.max,
        out_dtype=image.dtype,
        default=-np.inf,
    )

    assert np.allclose(df_max[0].values, scipy_max, rtol=0, atol=1e-5)

    df_min = aggregator.aggregate_min()

    assert df_min.shape[1] - 1 == image.shape[0]

    scipy_min = ndimage.labeled_comprehension(
        input=image[2].compute(),
        labels=mask.compute(),
        index=da.unique(mask).compute(),
        func=np.min,
        out_dtype=image.dtype,
        default=np.inf,
    )

    assert np.allclose(df_min[2].values, scipy_min, rtol=0, atol=1e-5)


def test_aggregate_mean(sdata):
    se_image = sdata["blobs_image"]
    se_labels = sdata["blobs_labels"]

    image = se_image.data[:, None, ...]
    mask = se_labels.data[None, ...]

    aggregator = RasterAggregator(
        mask_dask_array=mask.rechunk(512),
        image_dask_array=image.rechunk(512),
    )
    df_mean = aggregator.aggregate_mean()

    assert df_mean.shape[1] - 1 == image.shape[0]

    scipy_mean = ndimage.labeled_comprehension(
        input=image[0].compute(),
        labels=mask.compute(),
        index=da.unique(mask).compute(),
        func=np.mean,
        out_dtype=image.dtype,
        default=0,
    )

    assert np.allclose(df_mean[0].values, scipy_mean, rtol=0, atol=1e-5)

    scipy_mean = ndimage.labeled_comprehension(
        input=image[2].compute(),
        labels=mask.compute(),
        index=da.unique(mask).compute(),
        func=np.mean,
        out_dtype=image.dtype,
        default=0,
    )

    assert np.allclose(df_mean[2].values, scipy_mean, rtol=0, atol=1e-5)


def test_aggregate_var(sdata):
    se_image = sdata["blobs_image"]
    se_labels = sdata["blobs_labels"]

    image = se_image.data[:, None, ...]
    mask = se_labels.data[None, ...]

    aggregator = RasterAggregator(
        mask_dask_array=mask.rechunk(512),
        image_dask_array=image.rechunk(512),
    )
    df_var = aggregator.aggregate_var()

    assert df_var.shape[1] - 1 == image.shape[0]

    scipy_var = ndimage.variance(
        input=image[0].compute(),
        labels=mask.compute(),
        index=da.unique(mask).compute(),
    )

    assert np.allclose(df_var[0].values, scipy_var, rtol=0, atol=1e-5)

    scipy_var = ndimage.variance(
        input=image[2].compute(),
        labels=mask.compute(),
        index=da.unique(mask).compute(),
    )

    assert np.allclose(df_var[2].values, scipy_var, rtol=0, atol=1e-5)


def test_aggregate_kurtosis(sdata):
    se_image = sdata["blobs_image"]
    se_labels = sdata["blobs_labels"]

    image = se_image.data[:, None, ...]
    mask = se_labels.data[None, ...]

    aggregator = RasterAggregator(
        mask_dask_array=mask.rechunk(100),
        image_dask_array=image.rechunk(100),
    )
    df_kurt = aggregator.aggregate_kurtosis()

    assert df_kurt.shape[1] - 1 == image.shape[0]

    kurtosis_scipy = kurtosis(image[0].squeeze()[mask.squeeze() == 1].compute(), fisher=True, bias=True)

    assert np.allclose(df_kurt[df_kurt[_INSTANCE_KEY] == 1][0].item(), kurtosis_scipy, rtol=0, atol=1e-5)


def test_aggregate_skewness(sdata):
    se_image = sdata["blobs_image"]
    se_labels = sdata["blobs_labels"]

    image = se_image.data[:, None, ...]
    mask = se_labels.data[None, ...]

    aggregator = RasterAggregator(
        mask_dask_array=mask.rechunk(100),
        image_dask_array=image.rechunk(100),
    )
    df_skew = aggregator.aggregate_skew()

    assert df_skew.shape[1] - 1 == image.shape[0]

    skew_scipy = skew(image[0].squeeze()[mask.squeeze() == 1].compute(), bias=True)

    assert np.allclose(df_skew[df_skew[_INSTANCE_KEY] == 1][0].item(), skew_scipy, rtol=0, atol=1e-5)


def test_aggregate_quantiles(sdata_multi_c_no_backed):
    se_image = sdata_multi_c_no_backed["raw_image"]
    se_labels = sdata_multi_c_no_backed["masks_whole"]

    image = se_image.data[:, None, ...].rechunk(210)
    mask = se_labels.data[None, ...].rechunk(210)

    aggregator = RasterAggregator(
        mask_dask_array=mask,
        image_dask_array=image,
    )

    quantiles = [0.3, 0.5]
    dfs = aggregator.aggregate_quantiles(depth=200, quantiles=quantiles, quantile_background=True)

    channel_id = 3
    cell_id = 670
    quantile = 0.3
    assert (
        np.quantile(image[channel_id].compute()[mask.compute() == cell_id], q=quantile)
        == dfs[quantiles.index(quantile)][dfs[quantiles.index(quantile)][_INSTANCE_KEY] == cell_id][channel_id].item()
    )

    channel_id = 1
    cell_id = 0
    quantile = 0.5
    assert np.isclose(
        np.quantile(image[channel_id].compute()[mask.compute() == cell_id], q=quantile),
        dfs[quantiles.index(quantile)][dfs[quantiles.index(quantile)][_INSTANCE_KEY] == cell_id][channel_id].item(),
        rtol=0,
        atol=1e-6,
    )


def test_aggregate_radii_and_axes(sdata_multi_c_no_backed):
    se_labels = sdata_multi_c_no_backed["masks_whole"]

    mask = se_labels.data[None, ...].rechunk(210)

    aggregator = RasterAggregator(
        mask_dask_array=mask,
        image_dask_array=None,
    )

    df = aggregator.aggregate_radii_and_axes(depth=200, calculate_axes=True)

    assert df.shape == (
        len(aggregator._labels) - 1,  # radii for background (0), not computed.
        mask.ndim + mask.ndim**2 + 1,  # 3 radii, 3*3 axes, 1 cell ID
    )

    cell_id = 2

    radii, axes = _region_radii_and_axes(mask, label=cell_id)

    assert np.allclose(radii, df[df[_INSTANCE_KEY] == cell_id][range(3)].values.flatten())
    assert np.allclose(axes.flatten(), df[df[_INSTANCE_KEY] == cell_id][range(3, 12)].values.flatten())

    # check that we get same results for other chunksize
    mask = se_labels.data[None, ...].rechunk(100)

    aggregator = RasterAggregator(
        mask_dask_array=mask,
        image_dask_array=None,
    )

    df_other_chunksize = aggregator.aggregate_radii_and_axes(depth=200, calculate_axes=True)

    assert np.allclose(df_other_chunksize.values, df.values)


def test_aggregate_var_3D(sdata):
    se_image = sdata["blobs_image"]
    se_labels = sdata["blobs_labels"]

    image = se_image.data[:, None, ...]
    mask = se_labels.data[None, ...]

    # make artificial 3D mask and image
    mask = da.concatenate([mask, mask + mask])
    image = da.concatenate(
        [
            image,
            image + image,
        ],
        axis=1,
    )

    aggregator = RasterAggregator(
        mask_dask_array=mask.rechunk((1, 512, 512)),
        image_dask_array=image.rechunk((1, 1, 512, 512)),
    )
    df_var = aggregator.aggregate_var()

    assert df_var.shape[1] - 1 == image.shape[0]

    scipy_var = ndimage.variance(
        input=image[0].compute(),
        labels=mask.compute(),
        index=da.unique(mask).compute(),
    )

    assert np.allclose(df_var[0].values, scipy_var, rtol=0, atol=1e-5)

    scipy_var = ndimage.variance(
        input=image[2].compute(),
        labels=mask.compute(),
        index=da.unique(mask).compute(),
    )

    assert np.allclose(df_var[2].values, scipy_var, rtol=0, atol=1e-5)


def test_get_mask_area(sdata):
    se_labels = sdata["blobs_labels"]
    mask = se_labels.data[None, ...].rechunk(512)
    df = _get_mask_area(mask)

    mask_compute = mask.compute()
    area = ndimage.sum_labels(input=np.ones(mask_compute.shape), labels=mask_compute, index=np.unique(mask_compute))

    assert np.array_equal(df[_CELLSIZE_KEY].values, area)


def test_get_mask_area_subset(sdata):
    se_labels = sdata["blobs_labels"]
    mask = se_labels.data[None, ...].rechunk(512)
    mask_compute = mask.compute()

    index = np.unique(mask_compute)
    subset_index = [
        index[4],
        index[2],
        index.max() + 10,
    ]  # pick subset + an index not in mask. It will return 0 for this index.
    df = _get_mask_area(mask, index=subset_index)

    area = ndimage.sum_labels(input=np.ones(mask_compute.shape), labels=mask_compute, index=subset_index)

    assert np.array_equal(df[_CELLSIZE_KEY].values, area)


def test_get_center_of_mask(sdata):
    se_labels = sdata["blobs_labels"]
    mask = se_labels.data[None, ...].rechunk(512)
    df = _get_center_of_mass(mask)

    mask_compute = mask.compute()
    scipy_center_of_mass = np.array(
        ndimage.center_of_mass(input=mask_compute, labels=mask_compute, index=np.unique(mask_compute))
    )

    assert np.array_equal(df[[0, 1, 2]].values, np.array(scipy_center_of_mass), equal_nan=True)


def test_aggregate_custom_channel(sdata_multi_c_no_backed):
    def _calculate_intensity_mean(mask_block, image_block):
        table = regionprops_table(label_image=mask_block, intensity_image=image_block, properties=["intensity_mean"])
        return table["intensity_mean"]

    se_image = sdata_multi_c_no_backed["raw_image"]
    se_labels = sdata_multi_c_no_backed["masks_whole"]

    image = se_image.data[:, None, ...]
    mask = se_labels.data[None, ...]

    aggregator = RasterAggregator(
        mask_dask_array=mask,
        image_dask_array=image,
    )

    intensity_mean = aggregator._aggregate_custom_channel(
        image=image[0].rechunk(210),
        mask=mask.rechunk(210),
        depth=200,
        fn=_calculate_intensity_mean,
    )

    intensity_mean_skimage = _calculate_intensity_mean(mask.compute(), image[0].compute()).astype(np.float32)

    assert np.array_equal(intensity_mean.flatten(), intensity_mean_skimage.flatten())


def test_aggregate_custom_channel_fails(sdata_multi_c_no_backed):
    def _calculate_centroid_weighted(mask_block, image_block):
        table = regionprops_table(label_image=mask_block, intensity_image=image_block, properties=["centroid_weighted"])
        return table["centroid_weighted-1"]

    se_image = sdata_multi_c_no_backed["raw_image"]
    se_labels = sdata_multi_c_no_backed["masks_whole"]

    image = se_image.data[:, None, ...]
    mask = se_labels.data[None, ...]

    aggregator = RasterAggregator(
        mask_dask_array=mask,
        image_dask_array=image,
    )

    centroid = aggregator._aggregate_custom_channel(
        image=image[0].rechunk(210),
        mask=mask.rechunk(210),
        depth=200,
        fn=_calculate_centroid_weighted,
    )

    centroid_skimage = _calculate_centroid_weighted(mask.compute(), image[0].compute()).astype(np.float32)

    # Does not work for calculation of centroid, because this requires global information.
    # could work if we would also pass block info to custom callable fn.
    with pytest.raises(AssertionError):
        assert np.array_equal(centroid.flatten(), centroid_skimage.flatten())


def test_aggregate_custom_channel_mask(sdata_multi_c_no_backed):
    # also works for euler,..etc
    def _calculate_eccentricity(mask_block):
        table = regionprops_table(label_image=mask_block[0], intensity_image=None, properties=["eccentricity"])
        return table["eccentricity"]

    se_image = sdata_multi_c_no_backed["raw_image"]
    se_labels = sdata_multi_c_no_backed["masks_whole"]

    image = se_image.data[:, None, ...]
    mask = se_labels.data[None, ...]

    aggregator = RasterAggregator(
        mask_dask_array=mask,
        image_dask_array=image,
    )
    eccentricity = aggregator._aggregate_custom_channel(
        image=None,
        mask=mask.rechunk(210),
        depth=200,
        fn=_calculate_eccentricity,
        dtype=np.float32,
    )
    eccentricity_skimage = _calculate_eccentricity(mask.compute()).astype(np.float32)

    assert np.array_equal(eccentricity.flatten(), eccentricity_skimage.flatten())


def test_aggregate_custom_channel_multiple_features(sdata_multi_c_no_backed):
    def _calculate_intensity_mean_area(mask_block, image_block):
        table = regionprops_table(
            label_image=mask_block, intensity_image=image_block, properties=["area", "intensity_mean"]
        )
        return np.stack([table["intensity_mean"], table["area"]], axis=1)

    se_image = sdata_multi_c_no_backed["raw_image"]
    se_labels = sdata_multi_c_no_backed["masks_whole"]

    image = se_image.data[:, None, ...]
    mask = se_labels.data[None, ...]

    aggregator = RasterAggregator(
        mask_dask_array=mask,
        image_dask_array=image,
    )

    intensity_mean_area = aggregator._aggregate_custom_channel(
        image=image[0].rechunk(210),
        mask=mask.rechunk(210),
        depth=200,
        fn=_calculate_intensity_mean_area,
        features=2,
    )

    intensity_mean_area_skimage = _calculate_intensity_mean_area(mask.compute(), image[0].compute()).astype(np.float32)

    assert np.array_equal(intensity_mean_area.reshape(-1, 2), intensity_mean_area_skimage.reshape(-1, 2))


def test_aggregate_custom_channel_multiple_features_sdata(sdata):
    def _calculate_intensity_mean_area(mask_block, image_block):
        table = regionprops_table(
            label_image=mask_block, intensity_image=image_block, properties=["area", "intensity_mean"]
        )
        return np.stack([table["intensity_mean"], table["area"]], axis=1)

    se_image = sdata["blobs_image"]
    se_labels = sdata["blobs_labels"]

    image = se_image.data[:, None, ...]
    mask = se_labels.data[None, ...]

    aggregator = RasterAggregator(
        mask_dask_array=mask,
        image_dask_array=image,
    )

    # sdata contains masks that span almost complete tissue, so an assertion error will be raised,
    # because we have masks with greater diameter than the depth.
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "We expect exactly one non-NaN element per row (each column corresponding to a chunk of 'mask'). Please consider increasing 'depth' parameter."
        ),
    ):
        aggregator._aggregate_custom_channel(
            image=image[0].rechunk(512),
            mask=mask.rechunk(512),
            depth=200,
            fn=_calculate_intensity_mean_area,
            features=2,
        )


def test_region_radii_and_axes():
    mask = np.array(
        [
            [1, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )

    mask = mask[None, ...]

    radii, axis = _region_radii_and_axes(mask=mask, label=1)

    assert np.array_equal(np.array([1.0, 0.0, 0.0]), radii)

    assert np.allclose(np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), axis)
