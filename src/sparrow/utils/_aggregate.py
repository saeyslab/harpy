# write general functions to do an aggregation between an image layer or points layer and a labels layer.
from collections import defaultdict
from functools import partial
from typing import Callable

import dask
import dask.array as da
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import ndimage

from sparrow.utils._keys import _CELLSIZE_KEY, _INSTANCE_KEY


# maybe support DataArray as input instead of dask arrays.
class Aggregator:
    def __init__(self, mask_dask_array: da.Array, image_dask_array: da.Array):
        self._labels = (
            da.unique(mask_dask_array).compute()
        )  # calculate this one time during initialization, otherwise we would need to calculate this multiple times.
        assert image_dask_array.ndim == 4
        assert mask_dask_array.ndim == 3
        assert image_dask_array.shape[1:] == mask_dask_array.shape
        self._mask = mask_dask_array
        self._image = image_dask_array

    def aggregate_sum(
        self,
    ) -> pd.DataFrame:
        return self._aggregate(aggregate_func=partial(self._aggregate_stats_channel, stats_funcs=("sum")))

    def aggregate_mean(
        self,
    ) -> pd.DataFrame:
        return self._aggregate(aggregate_func=partial(self._aggregate_stats_channel, stats_funcs=("mean")))

    def aggregate_var(
        self,
    ) -> pd.DataFrame:
        return self._aggregate(aggregate_func=partial(self._aggregate_stats_channel, stats_funcs=("var")))

    def aggregate_max(
        self,
    ) -> pd.DataFrame:
        return self._aggregate(aggregate_func=self._aggregate_max_channel)

    def aggregate_min(
        self,
    ) -> pd.DataFrame:
        return self._aggregate(aggregate_func=self._aggregate_min_channel)

    def aggregate_area(self) -> pd.DataFrame:
        return _get_mask_area(self._mask, index=self._labels)

    def _aggregate(self, aggregate_func: Callable[[da.Array], pd.DataFrame]) -> pd.DataFrame:
        _result = []
        for _c_image in self._image:
            _result.append(aggregate_func(_c_image, self._mask))
        _result = np.concatenate(_result, axis=1)

        df = pd.DataFrame(_result)

        df[_INSTANCE_KEY] = self._labels
        return df

    # this calculates sum, count, mean and var
    def _aggregate_stats_channel(
        self,
        image: da.Array,
        mask: da.Array,
        stats_funcs: tuple[str, ...] = ("sum", "mean", "count", "var"),
    ) -> NDArray:
        # add an assert that checks that stats_funcs is in the list that is given.
        # first calculate the sum.
        if isinstance(stats_funcs, str):
            stats_funcs = (stats_funcs,)

        if "sum" in stats_funcs or "mean" in stats_funcs or "var" in stats_funcs:

            def _calculate_sum_per_chunk(mask_block: NDArray, image_block: NDArray) -> NDArray:
                unique_labels, new_labels = np.unique(mask_block, return_inverse=True)
                new_labels = np.reshape(new_labels, (-1,))  # flatten, since it may be >1-D
                idxs = np.searchsorted(unique_labels, self._labels)
                # make all of idxs valid
                idxs[idxs >= unique_labels.size] = 0
                found = unique_labels[idxs] == self._labels
                sums = np.bincount(new_labels, weights=image_block.ravel())
                sums = sums[idxs]
                sums[~found] = 0
                return sums.reshape(-1, 1)

            chunk_sum = da.map_blocks(
                lambda m, f: _calculate_sum_per_chunk(m, f),
                mask,
                image,
                dtype=image.dtype,
                chunks=(len(self._labels), 1),
                drop_axis=0,
            )

            dask_chunks = [
                da.from_delayed(_chunk, shape=(len(self._labels), 1), dtype=image.dtype)
                for _chunk in chunk_sum.to_delayed().flatten()
            ]

            # dask_array is an array of shape (len(index), nr_of_chunks in image/mask )
            dask_array = da.concatenate(dask_chunks, axis=1)

            sum = da.sum(dask_array, axis=1).compute().reshape(-1, 1)

        # then calculate the mean
        # i) first calculate the area
        if "mean" in stats_funcs or "count" in stats_funcs or "var" in stats_funcs:
            count = _calculate_area(mask, index=self._labels)

        # ii) then calculate the mean
        if "mean" in stats_funcs or "var" in stats_funcs:
            mean = sum / count

        if "var" in stats_funcs:
            # calculate the sum of squares per cell
            def _calculate_sum_c_per_chunk(mask_block: NDArray, image_block: NDArray) -> NDArray:
                def _sum_centered(labels):
                    # `labels` is expected to be an ndarray with the same shape as `input`.
                    # It must contain the label indices (which are not necessarily the labels
                    # themselves).
                    centered_input = image_block - mean_found.flatten()[labels]
                    # bincount expects 1-D inputs, so we ravel the arguments.
                    bc = np.bincount(labels.ravel(), weights=(centered_input * centered_input.conjugate()).ravel())
                    return bc

                unique_labels, new_labels = np.unique(mask_block, return_inverse=True)
                new_labels = np.reshape(new_labels, (-1,))  # flatten, since it may be >1-D
                idxs = np.searchsorted(unique_labels, self._labels)
                # make all of idxs valid
                idxs[idxs >= unique_labels.size] = 0
                found = unique_labels[idxs] == self._labels
                mean_found = mean[
                    found
                ]  # mean is the total mean calculated in previous step, but we only select the ones that are found
                sums_c = _sum_centered(new_labels.reshape(mask_block.shape))
                sums_c = sums_c[idxs]
                sums_c[~found] = 0
                return sums_c.reshape(-1, 1)

            chunk_sum_c = da.map_blocks(
                lambda m, f: _calculate_sum_c_per_chunk(m, f),
                mask,
                image,
                dtype=image.dtype,
                chunks=(len(self._labels), 1),
                drop_axis=0,
            )

            dask_chunks = [
                da.from_delayed(_chunk, shape=(len(self._labels), 1), dtype=image.dtype)
                for _chunk in chunk_sum_c.to_delayed().flatten()
            ]

            # dask_array is an array of shape (len(index), nr_of_chunks in image/mask )
            dask_array = da.concatenate(dask_chunks, axis=1)

            sum_c = da.sum(dask_array, axis=1).compute().reshape(-1, 1)

        to_return = {}
        if "sum" in stats_funcs:
            to_return["sum"] = sum
        if "mean" in stats_funcs:
            to_return["mean"] = mean
        if "count" in stats_funcs:
            to_return["count"] = count
        if "var" in stats_funcs:
            to_return["var"] = sum_c / count

        to_return = [to_return[func] for func in stats_funcs if func in to_return]

        return to_return[0] if len(to_return) == 1 else to_return

    def _aggregate_max_channel(
        self,
        image: da.Array,
        mask: da.Array,
    ):
        return self._min_max_channel(image, mask, min_or_max="max")

    def _aggregate_min_channel(
        self,
        image: da.Array,
        mask: da.Array,
    ):
        return self._min_max_channel(image, mask, min_or_max="min")

    def _min_max_channel(
        self,
        image: da.Array,
        mask: da.Array,
        min_or_max: str,
    ) -> NDArray:
        assert (
            image.numblocks == mask.numblocks
        ), "Dask arrays must have same number of blocks. Please rechunk arrays `image` and `mask` with same chunks size."

        assert min_or_max in ["max", "min"], "Please choose from [ 'min', 'max' ]."

        min_dtype, max_dtype = _get_min_max_dtype(image)

        def _calculate_min_max_per_chunk(mask_block: NDArray, image_block: NDArray) -> NDArray:
            max = ndimage.labeled_comprehension(
                image_block,
                mask_block,
                self._labels,
                func=np.max if min_or_max == "max" else np.min,
                out_dtype=image_block.dtype,
                default=min_dtype if min_or_max == "max" else max_dtype,
            )  # also works if we have a lot of labels. scipy makes sure it only searches for labels of self._labels that are in mask_block

            return max.reshape(-1, 1)

        chunk_min_max = da.map_blocks(
            lambda m, f: _calculate_min_max_per_chunk(m, f),
            mask,
            image,
            dtype=image.dtype,
            chunks=(len(self._labels), 1),
            drop_axis=0,
        )

        dask_chunks = [
            da.from_delayed(_chunk, shape=(len(self._labels), 1), dtype=image.dtype)
            for _chunk in chunk_min_max.to_delayed().flatten()
        ]

        # dask_array is an array of shape (len(self._labels), nr_of_chunks in image/mask )
        dask_array = da.concatenate(dask_chunks, axis=1)

        min_max_func = da.max if min_or_max == "max" else da.min

        return min_max_func(dask_array, axis=1).compute().reshape(-1, 1)


# util function to get the area of each label in mask
def _get_mask_area_deprecated(mask: da.Array, calculate_background_area: bool = False) -> pd.DataFrame:
    """
    Calculate area of each label in mask. Return as pd.Series.

    Deprecated, because using scipy to calculate area will scale better for large masks image.
    """

    @dask.delayed
    def calculate_area(mask_chunk: np.ndarray) -> tuple:
        unique, counts = np.unique(mask_chunk, return_counts=True)

        return unique, counts

    delayed_results = [calculate_area(chunk) for chunk in mask.to_delayed().flatten()]

    results = dask.compute(*delayed_results)

    combined_counts = defaultdict(int)

    # aggregate
    for unique, counts in results:
        for label, count in zip(unique, counts):
            if not calculate_background_area and label == 0:
                continue
            combined_counts[int(label)] += count

    combined_counts = pd.Series(combined_counts)
    combined_counts.index.name = _INSTANCE_KEY

    combined_counts.name = _CELLSIZE_KEY
    combined_counts = combined_counts.to_frame().reset_index()

    return combined_counts


def _get_mask_area(mask: da.Array, index: NDArray | None = None) -> pd.DataFrame:
    if index is None:
        index = da.unique(mask).compute()
    _result = _calculate_area(mask, index=index)
    return pd.DataFrame({_INSTANCE_KEY: index, _CELLSIZE_KEY: _result.flatten()})


def _calculate_area(mask: da.Array, index: NDArray | None = None) -> NDArray:
    if index is None:
        index = da.unique(mask).compute()

    def _calculate_count_per_chunk(mask_block: NDArray) -> NDArray:
        # fix labels, so we do not need to calculate for all
        unique_labels, new_labels = np.unique(mask_block, return_inverse=True)
        new_labels = np.reshape(new_labels, (-1,))  # flatten, since it may be >1-D
        idxs = np.searchsorted(unique_labels, index)
        # make all of idxs valid
        idxs[idxs >= unique_labels.size] = 0
        found = unique_labels[idxs] == index
        # calculate counts
        counts = np.bincount(new_labels)
        counts = counts[idxs]
        counts[~found] = 0
        return counts.reshape(-1, 1)

    chunk_count = da.map_blocks(
        _calculate_count_per_chunk,
        mask,
        dtype=mask.dtype,
        chunks=(len(index), 1),
        drop_axis=0,
    )

    dask_chunks = [
        da.from_delayed(_chunk, shape=(len(index), 1), dtype=mask.dtype)
        for _chunk in chunk_count.to_delayed().flatten()
    ]

    # dask_array is an array of shape (len(index), nr_of_chunks in image/mask )
    dask_array = da.concatenate(dask_chunks, axis=1)

    return da.sum(dask_array, axis=1).compute().reshape(-1, 1)


def _get_min_max_dtype(array):
    dtype = array.dtype
    if np.issubdtype(dtype, np.integer):
        return np.iinfo(dtype).min, np.iinfo(dtype).max
    elif np.issubdtype(dtype, np.floating):
        return np.finfo(dtype).min, np.finfo(dtype).max
    else:
        raise TypeError("Unsupported dtype")
