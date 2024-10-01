from functools import partial
from types import MappingProxyType
from typing import Any, Callable, Mapping

import dask
import dask.array as da
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import ndimage

from sparrow.utils._keys import _CELLSIZE_KEY, _INSTANCE_KEY


# TODO maybe support DataArray as input instead of dask arrays.
class RasterAggregator:
    """Helper class to calulate aggregated 'sum', 'mean', 'var', 'area', 'min' or 'max' of image and labels using Dask."""

    def __init__(self, mask_dask_array: da.Array, image_dask_array: da.Array | None):
        if not np.issubdtype(mask_dask_array.dtype, np.integer):
            raise ValueError(f"'mask_dask_array' should contains chunks of type {np.integer}.")
        self._labels = (
            da.unique(mask_dask_array).compute()
        )  # calculate this one time during initialization, otherwise we would need to calculate this multiple times.
        if image_dask_array is not None:
            assert image_dask_array.ndim == 4, "Currently only 4D image arrays are supported ('c', 'z', 'y', 'x')."
            assert (
                image_dask_array.shape[1:] == mask_dask_array.shape
            ), "The mask and the image should have the same spatial dimensions ('z', 'y', 'x')."
            assert (
                image_dask_array.chunksize[1:] == mask_dask_array.chunksize
            ), "Provided mask ('mask_dask_array') and image ('image_dask_array') do not have the same chunksize in ( 'z', 'y', 'x' ). Please rechunk."
            self._image = image_dask_array
        assert mask_dask_array.ndim == 3, "Currently only 3D masks are supported ('z', 'y', 'x')."

        self._mask = mask_dask_array

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

    def _aggregate_custom_channel(
        self,
        image: da.Array | None,
        mask: da.Array,
        depth: int,  # choose depth > estimated diameter of largest cell
        fn: Callable[[NDArray[np.int_], NDArray[np.int_ | np.float_] | None], NDArray[np.float_]],
        fn_kwargs: Mapping[str, Any] = MappingProxyType(
            {}
        ),  # fn is a callable that returns a 1D array with len == nr of unique labels in the mask passed to fn excluding 0
        dtype: np.dtype = np.float32,  # output dtype
        features: int = 1,
    ) -> NDArray:
        """
        Aggregates a custom operation over a masked region of an image, with the option to pass additional parameters to a custom function.

        Parameters
        ----------
        image
            The input image array. If None, the function will only process the `mask`. The array is expected
            to be a dask array (da.Array). If not None, the function will apply the operation to the image
            based on the mask regions.
        mask
            A dask array representing the mask. Each unique non-zero value in the mask identifies
            a separate region of interest (ROI), typically a cell. The mask array must have integer values corresponding to
            different cells in the image.
        depth
            depth is passed as `depth` to `dask.array.map_overlap`, where `depth` must be greater than the estimated
            diameter of the largest region of interest in the `mask`. This value ensures the appropriate
            neighborhood is considered when applying the function `fn`.
        fn
            A custom function that performs operations on a mask and an optional image array. The function should accept
            one or two NumPy arrays depending whether image is None or not: the first is the mask as an integer array,
            and the second is the image as a float or integer array. It must return a 2D NumPy array of type `np.float_`, where
            the first dimension of the returned array matches the number of unique labels in the mask passed to `fn` (excluding label `0`),
            ordered by its correponding label number, and second dimension are the number of features calculated by `fn`.
            This should match `features`.
            User warning: the number of unique labels in the mask passed to `fn` is not equal to the number of
            unique labels in `mask` due to the use of `dask.array.map_overlap`.
        fn_kwargs
            Additional keyword arguments to be passed to the function `fn`. The default is an empty `MappingProxyType`.
        dtype
            The data type of the output array. By default, this is `np.float32`. It can be changed to any valid
            NumPy data type if necessary.
        features
            The number of features `fn` calculates.

        Returns
        -------
        A 2D NumPy array containing the aggregated results of the custom operation `fn`, applied to the regions defined by the `mask`.
        The array has the same number of elements as the unique labels in the `mask` excluding `0`, and the results are ordered based on the ordered labels.
        The shape of the 2D Numpy array is thus (`len( self._labels[[self._labels!=0]]), features)`, with `self._labels = dask.array.unique( mask ).compute()`.
        """
        assert mask.numblocks[0] == 1, "mask can not be chunked in z-dimension. Please rechunk."
        depth = (0, depth, depth)
        _labels = self._labels[self._labels != 0]
        if image is not None:
            assert image.numblocks[0] == 1, "image can not be chunked in z-dimension. Please rechunk."
            arrays = [mask, image]
        else:
            arrays = [mask]
        dask_chunks = da.map_overlap(
            lambda *arrays, block_info=None, **kw: _aggregate_custom_block(*arrays, block_info=block_info, **kw),
            *arrays,
            dtype=dtype,
            chunks=(len(_labels), features),
            trim=False,
            drop_axis=0,
            boundary=0,
            depth=depth,
            index=_labels,
            _depth=depth,
            fn=fn,  # callable.
            fn_kwargs=fn_kwargs,  # keywords of the callable
            features=features,
        )
        dask_chunks = [
            da.from_delayed(_chunk, shape=(len(_labels), features), dtype=dtype).reshape(-1, 1)
            for _chunk in dask_chunks.to_delayed().flatten()
        ]  # put all features under each other

        dask_array = da.concatenate(dask_chunks, axis=1)
        # this gives you dask array of shape (features*len(_labels), nr_of_chunks in mask ) with chunksize (features*len(_labels), 1)

        sanity = da.all((~da.isnan(dask_array)).sum(axis=1) == 1)
        # da.nansum ignores np.nan added by _aggregate_custom_block
        results = da.nansum(dask_array, axis=1).reshape(-1, features)

        sanity, results = dask.compute(*[sanity, results])

        assert sanity, "We expect exactly one non-NaN element per row (each column corresponding to a chunk of 'mask'). Please consider increasing 'depth' parameter."

        return results


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


def _aggregate_custom_block(
    *arrays,
    index: NDArray,
    block_info,
    _depth,
    fn: Callable,
    fn_kwargs: Mapping[str, Any] = MappingProxyType({}),
    features: int = 1,
) -> NDArray:
    mask_block = arrays[0]
    if len(arrays) == 2:
        image_block = arrays[1]
        assert mask_block.shape == image_block.shape
    if len(arrays) > 2:
        raise ValueError("Only accepts one or two arrays.")
    assert 0 not in index
    total_nr_of_blocks = block_info[0]["num-chunks"]
    block_location = block_info[0]["chunk-location"]
    # check if chunk is on border of larger dask array
    y_upper_border = block_location[1] + 1 == total_nr_of_blocks[1]
    x_upper_border = block_location[2] + 1 == total_nr_of_blocks[2]
    y_lower_border = block_location[1] == 0
    x_lower_border = block_location[2] == 0

    border_labels = set()
    if not y_upper_border:
        # you do not only extract the ones on border, but in the overlap region that is in the current block,
        # e.g. you go from _depth[1] : _depth[1] * 2
        # otherwise you could miss masks that are crossing the border, but are non-continuous and do not overlap with the border.
        # we still assume diameter < depth
        border_labels.update(set(np.unique(mask_block[:, -(_depth[1] * 2) : -(_depth[1]), _depth[2] : -_depth[2]])))
    if not x_upper_border:
        border_labels.update(set(np.unique(mask_block[:, _depth[1] : -_depth[1], -(_depth[2] * 2) : -(_depth[2])])))
    if not y_lower_border:
        border_labels.update(set(np.unique(mask_block[:, _depth[1] : _depth[1] * 2, _depth[2] : -_depth[2]])))
    if not x_lower_border:
        border_labels.update(set(np.unique(mask_block[:, _depth[1] : -_depth[1], _depth[2] : _depth[2] * 2])))
    if 0 in border_labels:
        border_labels.remove(0)

    border_labels = list(border_labels)
    center_of_mass_border_labels = ndimage.center_of_mass(input=mask_block, labels=mask_block, index=border_labels)

    def _isin_original(center: tuple[float, float, float]):
        return (
            center[1] >= _depth[1]
            and center[1] < (mask_block.shape[1] - _depth[1])
            and center[2] >= _depth[2]
            and center[2] < (mask_block.shape[2] - _depth[2])
        )

    border_labels_in_original_block = []
    for _center in center_of_mass_border_labels:
        if _isin_original(_center):
            border_labels_in_original_block.append(True)
        else:
            border_labels_in_original_block.append(False)

    # get the border labels not to consider
    if border_labels:
        border_labels_not_to_consider = np.array(border_labels)[~np.array(border_labels_in_original_block)]

    # Set all masks that are fully outside the region to zero, they will be covered by other chunks
    subset = mask_block[:, _depth[1] : -_depth[1], _depth[2] : -_depth[2]]
    # Unique masks gives you all masks that are at least partially in 'original' array (i.e. without depth added)
    unique_masks = np.unique(subset)
    # remove masks that are on border, but are covered by other chunks, because center of mass is in other chunk
    if border_labels:
        unique_masks = unique_masks[~np.isin(unique_masks, border_labels_not_to_consider)]

    # Create a mask for labels that are NOT in unique_masks
    mask = ~np.isin(mask_block, unique_masks)
    mask_block[mask] = 0

    unique_masks = unique_masks[unique_masks != 0]
    index = index[index != 0]

    # if no labels in the block, there is nothing to calculate,
    # so return 1D array containing nan at each position.
    if len(unique_masks) == 0:
        return np.full((index.shape[0], features), np.nan)

    idxs = np.searchsorted(unique_masks, index)
    idxs[idxs >= unique_masks.size] = 0
    found = unique_masks[idxs] == index

    result = fn(*arrays, **fn_kwargs)  # fn can either take in a mask + image, or only a mask
    result = result.reshape(-1, features)
    assert (
        result.shape[0] == unique_masks.shape[0]
    ), "Callable 'fn' should return an array with length equal to the number of non zero labels in the provided mask."
    assert np.issubdtype(result.dtype, np.floating), "Callable 'fn' should return an array of dtype 'float'."
    if any(np.isnan(result).flatten()):
        raise AssertionError("Result of callable 'fn' is not allowed to contain NaN.")
    result = result[idxs]
    result[~found] = np.nan
    return result
