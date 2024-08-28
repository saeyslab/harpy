# write general functions to do an aggregation between an image layer or points layer and a labels layer.
from collections import defaultdict
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
        )  # calculate this one time during initialization, otherwise we would need to calculate this again and again.
        assert image_dask_array.ndim == 4
        assert mask_dask_array.ndim == 3
        assert image_dask_array.shape[1:] == mask_dask_array.shape
        self._mask = mask_dask_array
        self._image = image_dask_array
        self._df_area = None  # This to avoid recomputation of the area in mask_dask_array

    def aggregate_sum(
        self,
    ) -> pd.DataFrame:
        return self._aggregate(aggregate_func=self._aggregate_sum_channel)

    def aggregate_max(
        self,
    ) -> pd.DataFrame:
        return self._aggregate(aggregate_func=self._aggregate_max_channel)

    def aggregate_min(
        self,
    ) -> pd.DataFrame:
        return self._aggregate(aggregate_func=self._aggregate_min_channel)

    def aggregate_mean(
        self,
    ) -> pd.DataFrame:
        df = self.aggregate_sum()
        if self._df_area is None:
            self._df_area = self.aggregate_area()
        df = pd.merge(df, self._df_area, on=_INSTANCE_KEY)
        columns_to_divide = df.columns.difference([_INSTANCE_KEY, _CELLSIZE_KEY])
        df[columns_to_divide] = df[columns_to_divide].div(df[_CELLSIZE_KEY], axis=0)
        df = df.drop(columns=[_CELLSIZE_KEY])

        return df

    def aggregate_area(self) -> pd.DataFrame:
        return _get_mask_area(self._mask, calculate_background_area=True)

    def _aggregate(self, aggregate_func: Callable[[da.Array], pd.DataFrame]) -> pd.DataFrame:
        _result = []
        for _c_image in self._image:
            _result.append(aggregate_func(_c_image, self._mask))
        _result = np.concatenate(_result, axis=1)

        df = pd.DataFrame(_result)

        df[_INSTANCE_KEY] = self._labels
        return df

    def _aggregate_sum_channel(
        self,
        image: da.Array,
        mask: da.Array,
    ) -> NDArray:
        # lazy computation of pixel intensities on one channel for each label in mask_dask_array
        # result is an array of shape (len(unique(mask_dask_array).compute(), 1 ), so be aware that if
        # some labels are missing, e.g. unique(mask_dask_array).compute()=np.array([ 0,1,3,4 ]), resulting
        # array will hold at postion 2 the intensity for cell with index 3.
        assert (
            image.numblocks == mask.numblocks
        ), "Dask arrays must have same number of blocks. Please rechunk arrays `image` and `mask` with same chunks size."

        def _calculate_intensity_per_chunk_custom(mask_block: NDArray, image_block: NDArray) -> NDArray:
            sums = np.bincount(mask_block.ravel(), weights=image_block.ravel())

            num_padding = (max(self._labels) + 1) - len(sums)

            sums = np.pad(sums, (0, num_padding), "constant", constant_values=(0))

            sums = sums[self._labels]

            sums = sums.reshape(-1, 1)

            return sums

        def _calculate_intensity_per_chunk(mask_block: NDArray, image_block: NDArray) -> NDArray:
            sums = ndimage.sum_labels(input=image_block, labels=mask_block, index=self._labels)

            sums = sums.reshape(-1, 1)

            return sums

        chunk_sum = da.map_blocks(
            lambda m, f: _calculate_intensity_per_chunk(m, f),
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

        # dask_array is an array of shape (len(self._labels), nr_of_chunks in image/mask )
        dask_array = da.concatenate(dask_chunks, axis=1)

        return da.sum(dask_array, axis=1).compute().reshape(-1, 1)

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
def _get_mask_area(mask: da.Array, calculate_background_area: bool = False) -> pd.DataFrame:
    """Calculate area of each label in mask. Return as pd.Series."""

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


def _get_mask_area_update(mask: da.Array, index: NDArray | None = None):
    if index is None:
        index = da.unique(mask).compute()

    def _calculate_mask_area_per_chunk(mask_block: NDArray) -> NDArray:
        area = ndimage.sum_labels(
            input=np.ones(mask_block.shape), labels=mask_block, index=index
        )  # use this in a map_blocks
        # area is 0 if label of unique_labels is not in mask_block. So it will not contribute to the sum

        return area.reshape(-1, 1)

    chunk_area = da.map_blocks(
        _calculate_mask_area_per_chunk,
        mask,
        dtype=mask.dtype,
        chunks=(len(index), 1),
        drop_axis=0,
    )

    dask_chunks = [
        da.from_delayed(_chunk, shape=(len(index), 1), dtype=mask.dtype) for _chunk in chunk_area.to_delayed().flatten()
    ]

    # dask_array is an array of shape (len(self._labels), nr_of_chunks in image/mask )
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
