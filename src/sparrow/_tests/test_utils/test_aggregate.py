import dask.array as da
import numpy as np
from scipy import ndimage
from xrspatial import zonal_stats

from sparrow.utils._aggregate import Aggregator


def test_aggregate_sum(sdata):
    se_image = sdata["blobs_image"]
    se_labels = sdata["blobs_labels"]

    image = se_image.data[:, None, ...]
    mask = se_labels.data[None, ...]

    aggregator = Aggregator(
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

    aggregator = Aggregator(
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

    aggregator = Aggregator(
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
