import dask.array as da
import numpy as np
import pytest
from scipy import ndimage

from harpy.experimental import DistributionAggregator
from harpy.experimental._order_statistics import sort_within_segments

Q = (0.1, 0.25, 0.9)
STATS = ("median", "quantiles", "iqr", "mad")


def _synthetic_mask(shape=(1, 160, 200), n_seeds=150, radius=7, seed=0) -> np.ndarray:
    """Voronoi-like cells, plus a bar spanning several chunks and a ring whose bounding-box midpoint is outside it."""
    rng = np.random.default_rng(seed)
    seeds = np.zeros(shape, dtype=np.int32)
    idx = tuple(rng.integers(0, s, n_seeds) for s in shape)
    seeds[idx] = np.arange(1, n_seeds + 1, dtype=np.int32)
    distance, indices = ndimage.distance_transform_edt(seeds == 0, return_indices=True)
    mask = seeds[tuple(indices)]
    mask[distance > radius] = 0
    z = slice(None)
    mask[z, 100:103, 5:190] = 1000  # bar crossing many chunks along x
    yy, xx = np.ogrid[: shape[1], : shape[2]]
    ring = (np.hypot(yy - 60, xx - 60) >= 25) & (np.hypot(yy - 60, xx - 60) < 28)
    mask[:, ring] = 2000
    return mask


def _reference(mask: np.ndarray, image: np.ndarray, q=Q) -> dict[str, np.ndarray]:
    ids = np.unique(mask)
    ids = ids[ids != 0]
    out = {name: np.empty((ids.size, image.shape[0])) for name in ("median", *[f"q{v:g}" for v in q], "iqr", "mad")}
    for i, instance in enumerate(ids):
        values = image[:, mask == instance].astype(np.float64)  # (c, n)
        median = np.nanmedian(values, axis=1)
        out["median"][i] = median
        for v in q:
            out[f"q{v:g}"][i] = np.nanquantile(values, v, axis=1)
        out["iqr"][i] = np.nanquantile(values, 0.75, axis=1) - np.nanquantile(values, 0.25, axis=1)
        out["mad"][i] = np.nanmedian(np.abs(values - median[:, None]), axis=1)
    return ids, out


def _image(mask_shape, n_channels, dtype, seed=1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shape = (n_channels, *mask_shape)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return rng.integers(max(info.min, -3000), min(info.max, 3000), size=shape, endpoint=True).astype(dtype)
    image = rng.normal(100, 30, size=shape).astype(dtype)
    image[rng.random(shape) < 0.02] = np.nan  # NaN pixels are ignored
    return image


def _assert_matches(dfs, ids, expected, instance_key="cell_ID", rtol=1e-6):
    for name, values in expected.items():
        df = dfs[name]
        np.testing.assert_array_equal(df[instance_key].to_numpy(), ids)
        actual = df.drop(columns=instance_key).to_numpy()
        np.testing.assert_allclose(actual, values, rtol=rtol, atol=1e-4, err_msg=name)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16, np.int32, np.float32, np.float64])
def test_matches_numpy_across_chunk_borders(dtype):
    mask = _synthetic_mask()
    image = _image(mask.shape, n_channels=3, dtype=dtype)
    aggregator = DistributionAggregator(
        mask_dask_array=da.from_array(mask, chunks=(1, 48, 48)),
        image_dask_array=da.from_array(image, chunks=(2, 1, 48, 48)),
    )
    dfs = aggregator.aggregate_stats(stats_funcs=STATS, q=Q)
    ids, expected = _reference(mask, image)
    _assert_matches(dfs, ids, expected)
    assert not aggregator.instance_table["truncated"].any()


def test_3d_chunked_in_z_and_image_chunks_differ_from_mask():
    mask = _synthetic_mask(shape=(6, 90, 110), n_seeds=80)
    image = _image(mask.shape, n_channels=4, dtype=np.uint16)
    aggregator = DistributionAggregator(
        mask_dask_array=da.from_array(mask, chunks=(3, 40, 40)),
        image_dask_array=da.from_array(image, chunks=(1, 2, 25, 70)),
    )
    dfs = aggregator.aggregate_stats(stats_funcs=STATS, q=Q)
    ids, expected = _reference(mask, image)
    _assert_matches(dfs, ids, expected)


def test_2d_mask():
    mask = _synthetic_mask()[0]
    image = _image(mask.shape, n_channels=2, dtype=np.float32)
    aggregator = DistributionAggregator(
        mask_dask_array=da.from_array(mask, chunks=(50, 64)),
        image_dask_array=da.from_array(image, chunks=(2, 50, 64)),
    )
    dfs = aggregator.aggregate_stats(stats_funcs=("median", "mad"))
    ids, expected = _reference(mask, image)
    _assert_matches(dfs, ids, {k: expected[k] for k in ("median", "mad")})


def test_depth_cap_truncates_only_large_instances():
    mask = _synthetic_mask()
    image = _image(mask.shape, n_channels=2, dtype=np.uint16)
    aggregator = DistributionAggregator(
        mask_dask_array=da.from_array(mask, chunks=(1, 48, 48)),
        image_dask_array=da.from_array(image, chunks=(2, 1, 48, 48)),
        depth=10,
    )
    table = aggregator.instance_table
    truncated = set(table.loc[table["truncated"], "cell_ID"])
    assert 1000 in truncated  # the bar spans far more than a tile plus 10 px
    assert truncated <= {1000, 2000}

    dfs = aggregator.aggregate_stats(stats_funcs=("median",))
    ids, expected = _reference(mask, image)
    complete = ~np.isin(ids, list(truncated))
    actual = dfs["median"].drop(columns="cell_ID").to_numpy()
    np.testing.assert_allclose(actual[complete], expected["median"][complete])


def test_index_subset_and_lazy_output():
    mask = _synthetic_mask()
    image = _image(mask.shape, n_channels=2, dtype=np.uint16)
    aggregator = DistributionAggregator(
        mask_dask_array=da.from_array(mask, chunks=(1, 48, 48)),
        image_dask_array=da.from_array(image, chunks=(1, 1, 48, 48)),
    )
    index = np.array([2000, 3, 1000])
    instance_ids, stats, names = aggregator.aggregate_stats_lazy(stats_funcs=("quantiles",), q=(0.5,), index=index)
    assert names == ("q0.5",)
    assert stats.shape == (3, 2, 1)
    assert sorted(instance_ids.tolist()) == [3, 1000, 2000]
    computed = stats.compute()
    for row, instance in enumerate(instance_ids):
        expected = np.quantile(image[:, mask == instance].astype(np.float64), 0.5, axis=1)
        np.testing.assert_allclose(computed[row, :, 0], expected)


def test_invalid_arguments():
    mask = da.from_array(_synthetic_mask(), chunks=(1, 48, 48))
    image = da.from_array(_image(mask.shape, n_channels=1, dtype=np.uint8), chunks=(1, 1, 48, 48))
    with pytest.raises(ValueError, match="integer dtype"):
        DistributionAggregator(mask.astype(np.float32), image)
    with pytest.raises(ValueError, match="expected"):
        DistributionAggregator(mask, image[0])
    aggregator = DistributionAggregator(mask, image)
    with pytest.raises(ValueError, match="Unsupported statistic"):
        aggregator.aggregate_stats(stats_funcs=("mean",))
    with pytest.raises(ValueError, match="'q' is None"):
        aggregator.aggregate_stats(stats_funcs=("quantiles",))
    with pytest.raises(ValueError, match="background"):
        aggregator.aggregate_stats(index=np.array([0, 1]))
    with pytest.raises(ValueError, match="not in the mask"):
        aggregator.aggregate_stats(index=np.array([123456]))


def test_empty_mask():
    mask = da.zeros((1, 32, 32), dtype=np.int32, chunks=16)
    image = da.ones((2, 1, 32, 32), dtype=np.uint16, chunks=16)
    dfs = DistributionAggregator(mask, image).aggregate_stats(stats_funcs=("median",))
    assert dfs["median"].shape == (0, 3)


@pytest.mark.parametrize("dtype", [np.bool_, np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.float16])
def test_packed_sort_is_exact(dtype):
    rng = np.random.default_rng(0)
    segments = rng.integers(0, 50, 5000)
    if dtype == np.bool_:
        values = rng.random(5000) < 0.5
    elif np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        values = rng.integers(info.min, info.max, 5000, endpoint=True, dtype=dtype)
    else:
        values = rng.normal(size=5000).astype(dtype)
    expected = values[np.lexsort((values, segments))]
    np.testing.assert_array_equal(sort_within_segments(segments, values), expected)


def test_packed_sort_float32_special_values():
    values = np.array([np.nan, -np.inf, np.inf, -0.0, 0.0, -1.5, 1.5, np.nan, -np.nan], dtype=np.float32)
    segments = np.zeros(values.size, dtype=np.intp)
    out = sort_within_segments(segments, values)
    np.testing.assert_array_equal(out[:6], [-np.inf, -1.5, -0.0, 0.0, 1.5, np.inf])
    assert np.isnan(out[6:]).all()
