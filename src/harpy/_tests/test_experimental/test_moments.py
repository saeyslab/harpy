import dask.array as da
import numpy as np
import pandas as pd
import pytest
from scipy import ndimage

from harpy.experimental import MomentAggregator
from harpy.experimental._moments import (
    _channel_batches,
    _chunk_partial,
    _group_labels,
    _merge,
    _merge_numba,
    _Partial,
)
from harpy.utils import RasterAggregator

STATS = ("sum", "mean", "count", "var", "kurtosis", "skew")


def _mask(shape=(1, 120, 150), n_seeds=120, radius=6, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    seeds = np.zeros(shape, dtype=np.int32)
    seeds[tuple(rng.integers(0, s, n_seeds) for s in shape)] = rng.permutation(np.arange(1, 10 * n_seeds))[:n_seeds]
    distance, indices = ndimage.distance_transform_edt(seeds == 0, return_indices=True)
    mask = seeds[tuple(indices)]
    mask[distance > radius] = 0
    return mask


def _reference(mask: np.ndarray, image: np.ndarray, ids: np.ndarray) -> dict[str, np.ndarray]:
    n_channels = image.shape[0]
    out = {name: np.zeros((ids.size, n_channels)) for name in STATS}
    out["min"] = np.zeros((ids.size, n_channels), dtype=image.dtype)
    out["max"] = np.zeros((ids.size, n_channels), dtype=image.dtype)
    out["center_of_mass"] = np.zeros((ids.size, 3))
    for i, label in enumerate(ids):
        values = image[:, mask == label].astype(np.float64)
        n = values.shape[1]
        mean = values.mean(axis=1)
        d = values - mean[:, None]
        m2, m3, m4 = ((d**p).mean(axis=1) for p in (2, 3, 4))
        with np.errstate(divide="ignore", invalid="ignore"):
            out["skew"][i] = np.nan_to_num(m3 / m2**1.5, nan=0)
            out["kurtosis"][i] = np.nan_to_num(m4 / m2**2 - 3, nan=0)
        out["sum"][i], out["mean"][i], out["count"][i], out["var"][i] = values.sum(axis=1), mean, n, m2
        out["min"][i], out["max"][i] = image[:, mask == label].min(axis=1), image[:, mask == label].max(axis=1)
        out["center_of_mass"][i] = np.argwhere(mask == label).mean(axis=0)
    out["count"] = out["count"][:, :1]
    return out


def _raise(block: np.ndarray) -> np.ndarray:
    raise RuntimeError("the image was read")


def _check(dfs, expected, stats_funcs, ids, rtol=1e-5):
    assert len(dfs) == len(stats_funcs)
    for df, name in zip(dfs, stats_funcs, strict=True):
        np.testing.assert_array_equal(df["cell_ID"].to_numpy(), ids)
        np.testing.assert_allclose(
            df.drop(columns="cell_ID").to_numpy(), expected[name], rtol=rtol, atol=1e-4, err_msg=name
        )


@pytest.mark.parametrize("engine", ["numba", "numpy"])
@pytest.mark.parametrize(
    ("dtype", "offset"), [(np.uint8, 0), (np.uint16, 0), (np.int16, 0), (np.float32, 0), (np.float32, 1e5)]
)
def test_matches_numpy(dtype, offset, engine):
    mask = _mask()
    rng = np.random.default_rng(1)
    image = (rng.integers(0, 200, size=(3, *mask.shape)) + offset).astype(dtype)
    if np.issubdtype(dtype, np.floating):
        image += rng.normal(size=image.shape).astype(dtype)
    aggregator = MomentAggregator(
        mask_dask_array=da.from_array(mask, chunks=(1, 40, 50)),
        image_dask_array=da.from_array(image, chunks=(2, 1, 40, 50)),
        engine=engine,
    )
    ids = np.unique(mask)
    _check(aggregator.aggregate_stats(STATS), _reference(mask, image, ids), STATS, ids)


@pytest.mark.parametrize("engine", ["numba", "numpy"])
@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.int32, np.float32, np.float64])
def test_min_max_center_of_mass_match_numpy(dtype, engine):
    mask = _mask(shape=(3, 70, 90))
    rng = np.random.default_rng(5)
    image = rng.integers(-100 if np.dtype(dtype).kind != "u" else 0, 200, size=(3, *mask.shape)).astype(dtype)
    if np.issubdtype(dtype, np.floating):
        image += rng.normal(size=image.shape).astype(dtype)
    aggregator = MomentAggregator(
        da.from_array(mask, chunks=(2, 30, 40)), da.from_array(image, chunks=(2, 2, 30, 40)), engine=engine
    )
    ids = np.unique(mask)
    expected = _reference(mask, image, ids)
    stats_funcs = ("max", "mean", "center_of_mass", "min")
    df_max, df_mean, df_com, df_min = aggregator.aggregate_stats(stats_funcs)
    for df, name in ((df_min, "min"), (df_max, "max")):
        values = df.drop(columns="cell_ID").to_numpy()
        assert values.dtype == image.dtype
        np.testing.assert_array_equal(values, expected[name], err_msg=name)
    assert list(df_com.columns) == [0, 1, 2, "cell_ID"]
    np.testing.assert_allclose(df_com.drop(columns="cell_ID").to_numpy(), expected["center_of_mass"], rtol=1e-12)
    np.testing.assert_allclose(df_mean.drop(columns="cell_ID").to_numpy(), expected["mean"], rtol=1e-5)


@pytest.mark.parametrize("engine", ["numba", "numpy"])
def test_min_max_propagate_nan(engine):
    mask = _mask()
    image = np.random.default_rng(6).normal(size=(2, *mask.shape)).astype(np.float32)
    label = mask[0, 60, 75]
    first, second = np.argwhere(mask[0] == label)[[0, -1]]
    image[0, 0, *first] = np.nan  # channel 0: NaN on the first pixel of the label
    image[1, 0, *second] = np.nan  # channel 1: NaN on the last pixel of the label
    aggregator = MomentAggregator(
        da.from_array(mask, chunks=(1, 40, 50)), da.from_array(image, chunks=(1, 1, 40, 50)), engine=engine
    )
    df_min, df_max = aggregator.aggregate_stats(("min", "max"))
    row = np.flatnonzero(df_min["cell_ID"].to_numpy() == label)[0]
    for df in (df_min, df_max):
        assert np.isnan(df.iloc[row, :2].to_numpy(dtype=np.float32)).all()
        assert not np.isnan(df.drop(index=row).drop(columns="cell_ID").to_numpy()).any()


@pytest.mark.parametrize("stats_funcs", [("mean",), ("sum", "var"), ("skew", "count")])
def test_lower_orders_match_numpy(stats_funcs):
    mask = _mask()
    image = np.random.default_rng(4).integers(0, 500, size=(2, *mask.shape)).astype(np.uint16)
    aggregator = MomentAggregator(da.from_array(mask, chunks=(1, 40, 50)), da.from_array(image, chunks=(2, 1, 40, 50)))
    ids = np.unique(mask)
    _check(aggregator.aggregate_stats(stats_funcs), _reference(mask, image, ids), stats_funcs, ids)


def test_matches_raster_aggregator():
    mask = _mask(shape=(4, 80, 90))
    image = np.random.default_rng(2).integers(0, 4000, size=(5, *mask.shape)).astype(np.uint16)
    mask_da = da.from_array(mask, chunks=(2, 32, 32))
    image_da = da.from_array(image, chunks=(2, 2, 32, 32))
    index = np.unique(mask)
    raster = RasterAggregator(mask_da, image_da, run_on_gpu=False)
    expected = raster.aggregate_stats(STATS, index=index)
    actual = MomentAggregator(mask_da, image_da).aggregate_stats(STATS, index=index)
    for e, a, name in zip(expected, actual, STATS, strict=True):
        np.testing.assert_array_equal(a["cell_ID"].to_numpy(), e["cell_ID"].to_numpy())
        # RasterAggregator accumulates per-chunk sums in float32, so near-zero skew/kurtosis differ by ~1e-7
        np.testing.assert_allclose(
            a.drop(columns="cell_ID"), e.drop(columns="cell_ID"), rtol=1e-5, atol=1e-6, err_msg=name
        )

    # a subset with a label absent from the mask: both fill min/max with the dtype extremes and the center with NaN
    index = np.r_[index[::3], mask.max() + 1]
    expected = [
        raster.aggregate_min(index=index),
        raster.aggregate_max(index=index),
        raster.center_of_mass(index=index),
    ]
    actual = MomentAggregator(mask_da, image_da).aggregate_stats(("min", "max", "center_of_mass"), index=index)
    for e, a, name in zip(expected, actual, ("min", "max", "center_of_mass"), strict=True):
        pd.testing.assert_frame_equal(a, e, check_dtype=name != "center_of_mass", obj=name)


def test_index_subset_order_and_missing_labels():
    mask = _mask()
    image = np.random.default_rng(3).integers(0, 100, size=(2, *mask.shape)).astype(np.uint16)
    aggregator = MomentAggregator(da.from_array(mask, chunks=(1, 40, 50)), da.from_array(image, chunks=(1, 1, 40, 50)))
    present = np.unique(mask)[1:4]
    missing = mask.max() + 1
    index = np.array([present[2], 0, missing, present[0]])
    df_mean, df_count, df_sum = aggregator.aggregate_stats(("mean", "count", "sum"), index=index)
    np.testing.assert_array_equal(df_mean["cell_ID"].to_numpy(), index)
    assert np.isnan(df_mean.iloc[2, :2]).all()
    assert df_count.iloc[2, 0] == 0
    assert (df_sum.iloc[2, :2] == 0).all()
    expected = _reference(mask, image, index[[0, 1, 3]])
    np.testing.assert_allclose(df_mean.drop(columns="cell_ID").to_numpy()[[0, 1, 3]], expected["mean"], rtol=1e-6)


@pytest.mark.parametrize("engine", ["numba", "numpy"])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32, np.int32, np.int64, np.uint64])
def test_group_labels_matches_unique(dtype, engine):
    block = np.random.default_rng(7).choice(np.array([0, 3, 7, 200], dtype=dtype), size=(2, 30, 40))
    labels, counts, segments = _group_labels(block, engine)
    expected_labels, expected_segments, expected_counts = np.unique(block, return_inverse=True, return_counts=True)
    assert labels.dtype == block.dtype
    np.testing.assert_array_equal(labels, expected_labels)
    np.testing.assert_array_equal(counts, expected_counts)
    np.testing.assert_array_equal(segments, expected_segments.reshape(-1))


@pytest.mark.parametrize("layout", ["noise", "stripes"])
def test_engines_agree_on_short_and_long_runs(layout):
    # the numba kernel accumulates over runs of equal labels: check runs of one pixel and runs spanning rows
    rng = np.random.default_rng(8)
    shape = (2, 40, 50)
    if layout == "noise":
        mask = rng.integers(0, 30, size=shape).astype(np.int32)
    else:
        mask = np.repeat(np.arange(shape[0] * shape[1] // 4), 4 * shape[2]).reshape(shape).astype(np.int32)
    image = rng.integers(0, 1000, size=(3, *shape)).astype(np.uint16)
    stats = (*STATS, "min", "max", "center_of_mass")
    results = [
        MomentAggregator(
            da.from_array(mask, chunks=(1, 20, 25)), da.from_array(image, chunks=(2, 1, 20, 25)), engine=engine
        ).aggregate_stats(stats)
        for engine in ("numba", "numpy")
    ]
    for numba_df, numpy_df, name in zip(*results, stats, strict=True):
        pd.testing.assert_frame_equal(numba_df, numpy_df, check_exact=False, rtol=1e-5, atol=1e-6, obj=name)


def test_channel_batches():
    assert _channel_batches((1, 1, 1, 1, 1), channel_nbytes=10, budget=25) == [[0, 1], [2, 3], [4]]
    assert _channel_batches((2, 2), channel_nbytes=10, budget=40) == [[0, 1]]
    # a channel chunk above the budget forms a batch on its own
    assert _channel_batches((3, 1, 1), channel_nbytes=10, budget=25) == [[0], [1, 2]]


def test_channel_batches_follow_max_task_bytes(monkeypatch):
    mask = _mask()
    image = np.random.default_rng(9).integers(0, 1000, size=(5, *mask.shape)).astype(np.uint16)
    mask_da = da.from_array(mask, chunks=(1, 40, 50))
    image_da = da.from_array(image, chunks=(1, 1, 40, 50))
    stats = (*STATS, "min", "max", "center_of_mass")
    expected = MomentAggregator(mask_da, image_da).aggregate_stats(stats)

    blocks_per_task = []

    def record(mask_block, offset, request, engine, *image_blocks):
        blocks_per_task.append(len(image_blocks))
        return _chunk_partial(mask_block, offset, request, engine, *image_blocks)

    monkeypatch.setattr("harpy.experimental._moments._chunk_partial", record)
    # one channel of a spatial chunk is 40 * 50 * 2 bytes, so a task gets at most two channel chunks
    actual = MomentAggregator(mask_da, image_da, max_task_bytes="8000B").aggregate_stats(stats)
    assert sorted(blocks_per_task) == [1] * 9 + [2] * 18  # 9 spatial chunks, channel batches [0, 1], [2, 3], [4]
    for e, a, name in zip(expected, actual, stats, strict=True):
        pd.testing.assert_frame_equal(a, e, check_exact=True, obj=name)


@pytest.mark.parametrize(
    "fields",
    [
        ("mean", "m2", "m3", "m4", "minimum", "maximum", "coordinate_sum"),
        ("mean",),
        ("mean", "m2"),
        ("maximum", "coordinate_sum"),
        (),
    ],
)
@pytest.mark.parametrize("dtype", [np.uint16, np.float32])
def test_merge_engines_agree(fields, dtype):
    rng = np.random.default_rng(10)
    partials = []
    for _ in range(5):
        # 80 of 200 labels per partial, so most labels occur in several partials
        labels = np.sort(rng.choice(200, 80, replace=False)).astype(np.uint32)
        k = labels.size
        values = {name: rng.normal(size=(k, 3)) for name in ("mean", "m2", "m3", "m4")}
        values["coordinate_sum"] = rng.random((k, 3))
        values["minimum"], values["maximum"] = rng.integers(0, 1000, size=(2, k, 3)).astype(dtype)
        if np.issubdtype(dtype, np.floating):
            values["minimum"][:2, 0] = values["maximum"][:2, 0] = np.nan
        partials.append(_Partial(labels, rng.integers(1, 50, k), **{name: values[name] for name in fields}))
    expected, actual = _merge(*partials), _merge_numba(*partials)
    for name in ("labels", "counts", "minimum", "maximum"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name), err_msg=name)
    if np.issubdtype(dtype, np.floating) and "maximum" in fields:
        assert np.isnan(actual.maximum[:, 0]).any()
    for name in ("mean", "m2", "m3", "m4", "coordinate_sum"):
        e, a = getattr(expected, name), getattr(actual, name)
        assert (a is None) == (e is None), name
        if e is not None:
            np.testing.assert_allclose(a, e, rtol=1e-12, atol=1e-12 * np.abs(e).max(), err_msg=name)


def test_mask_statistics_skip_image_and_validation():
    mask = _mask()
    image = da.zeros((1, *mask.shape), dtype=np.uint8, chunks=(1, 1, 40, 50)).map_blocks(_raise, dtype=np.uint8)
    mask_da = da.from_array(mask, chunks=(1, 40, 50))
    df_count, df_com = MomentAggregator(mask_da, image).aggregate_stats(("count", "center_of_mass"))
    ids, counts = np.unique(mask, return_counts=True)
    np.testing.assert_array_equal(df_count[0].to_numpy(), counts)
    np.testing.assert_array_equal(df_count["cell_ID"].to_numpy(), ids)
    expected = _reference(mask, np.zeros((1, *mask.shape)), ids)["center_of_mass"]
    np.testing.assert_allclose(df_com.drop(columns="cell_ID").to_numpy(), expected, rtol=1e-12)
    with pytest.raises(ValueError, match="Unsupported statistic"):
        MomentAggregator(mask_da, image).aggregate_stats(("median",))
    with pytest.raises(ValueError, match="chunks"):
        MomentAggregator(mask_da, image.rechunk((1, 1, 30, 50)))
    with pytest.raises(ValueError, match="max_task_bytes"):
        MomentAggregator(mask_da, image, max_task_bytes=0)
