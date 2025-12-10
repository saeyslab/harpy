from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from types import MappingProxyType
from typing import Any, Literal

import dask
import dask.array as da
import numpy as np
import pandas as pd
from dask_image import ndmeasure
from loguru import logger as log
from numpy.typing import NDArray
from scipy import ndimage
from sklearn.decomposition import PCA

from harpy.utils._keys import _CELLSIZE_KEY, _INSTANCE_KEY

# NOTE:
# aggregate works by computing statistics per chunk. For each chunk, it produces
# a matrix with shape (i, c, z, y, x), where:
#   - i = total number of labels in the global mask
#   - c = number of channels in the chunk
#   - z = 1
#   - y = 1
#   - x = 1
#
# We deliberately avoid the optimization where i is set to the maximum number of
# labels in any chunk, because that would require an extra pass over the global
# mask to count labels per chunk (as done in Featurizer).
#
# Since i is on the order of thousands to millions, and we only compute a single
# feature (e.g. mean), each chunk of the aggregated matrix stays below ~50 MB
# (assuming chunksize c = 1), even if the global mask contains ~10M labels.
#
# By chunking over the (c, z, y, x) dimensions, the user can control RAM usage.
# As a rule of thumb, choose chunk sizes of about z,y,x ≈ 4000 and c ≈ 5,
# adjusting as needed based on available memory.


class RasterAggregator:
    """
    Helper class to calulate aggregated 'sum', 'mean', 'var', 'kurtosis', 'skew', 'area', 'min', 'max' 'quantiles', 'center of mass', 'radii' or 'principal_axes' of image and labels using Dask.

    Parameters
    ----------
    mask_dask_array
        A 3D Dask array of integer labels representing segmented regions.
        Expected shape is ('z', 'y', 'x'). Each unique integer value represents a separate label.
    image_dask_array
        A 4D Dask array representing the image data with shape ('c', 'z', 'y', 'x'),
        where 'c' is the number of channels. Can be `None` if only mask-based computations
        (e.g., count or center of mass) are required.
    instance_key
        name of the instance key
    instance_size_key
        name of the instance size key

    Raises
    ------
    ValueError
        If `mask_dask_array` does not contain an integer dtype.
    AssertionError
        If `mask_dask_array` is not 3D.
    AssertionError
        If `image_dask_array` is provided but is not 4D.
    AssertionError
        If spatial dimensions of `image_dask_array` and `mask_dask_array` do not match.
    AssertionError
        If chunk sizes of spatial dimensions do not match between image and mask.

    Notes
    -----
    The aggregate operation computes statistics per chunk.
    For each chunk in `mask_dask_array` and `image_dask_array`, it
    produces a matrix with shape (i, c, z, y, x), where:

    - i: total number of labels in the global mask
    - c: number of channels in the chunk
    - z = 1, y = 1, x = 1

    These matrices (chunks) are then aggregated, to obtain statistics for the
    global mask and image.

    We intentionally avoid the optimization of setting i to the maximum number of
    labels in any chunk, because this would require an additional pass over the
    global mask to count labels per chunk (as done in :class:`harpy.utils._featurize.Featurizer`).

    Because i typically ranges from thousands to millions, and because only a single
    feature (e.g., a mean statistic) is computed, each chunk of the aggregated
    matrix remains under ~50 MB (for a chunksize of c = 1), even when the global
    mask contains around 10 million labels.

    By chunking the underlying dask arrays along the (c, z, y, x) dimensions in the
    on-disk Zarr store, the user can effectively control RAM usage during
    aggregation. As a practical guideline, choose chunk sizes of roughly
    z, y, x ≈ 4096 and c ≈ 5, adjusting these values based on the available memory.
    """

    def __init__(
        self,
        mask_dask_array: da.Array,
        image_dask_array: da.Array | None,
        instance_key: str = _INSTANCE_KEY,
        instance_size_key: str = _CELLSIZE_KEY,
    ):
        if not np.issubdtype(mask_dask_array.dtype, np.integer):
            raise ValueError(f"'mask_dask_array' should contains chunks of type {np.integer}.")
        self._labels = (
            da.unique(mask_dask_array).compute()
        )  # calculate this one time during initialization, otherwise we would need to calculate this multiple times.
        if image_dask_array is not None:
            assert image_dask_array.ndim == 4, "Currently only 4D image arrays are supported ('c', 'z', 'y', 'x')."
            assert image_dask_array.shape[1:] == mask_dask_array.shape, (
                "The mask and the image should have the same spatial dimensions ('z', 'y', 'x')."
            )
            assert image_dask_array.chunksize[1:] == mask_dask_array.chunksize, (
                "Provided mask ('mask_dask_array') and image ('image_dask_array') do not have the same chunksize in ( 'z', 'y', 'x' ). Please rechunk."
            )
            self._image = image_dask_array
        assert mask_dask_array.ndim == 3, "Currently only 3D masks are supported ('z', 'y', 'x')."

        self._mask = mask_dask_array
        self._instance_key = instance_key
        self._instance_size_key = instance_size_key
        # where area will be saved
        self._count = None

    def aggregate_stats(
        self,
        stats_funcs: tuple[str, ...] = ("sum", "mean", "count", "var", "kurtosis", "skew"),
    ) -> list[pd.DataFrame]:
        """
        Computes multiple statistical metrics for each label in the mask, across all image channels.

        Parameters
        ----------
        stats_funcs
            A tuple of statistical functions to apply. Supported values include: "sum", "mean",
            "count", "var", "kurtosis", and "skew". Defaults to all.

        Returns
        -------
        A list of DataFrames, each corresponding to one of the requested statistics.
        Each DataFrame contains one row per label a column per image channel and a column with the label ID,
        except for stat "count", which only contains a column with the counts and a column with the label ID.
        """
        if isinstance(stats_funcs, str):
            stats_funcs = (stats_funcs,)
        results = self._aggregate_stats(stats_funcs=stats_funcs)
        dfs = []
        for _result in results:
            df = pd.DataFrame(_result)
            df[self._instance_key] = self._labels
            dfs.append(df)

        return dfs

    def aggregate_sum(
        self,
    ) -> pd.DataFrame:
        """
        Computes the sum of pixel values within each labeled region for all image channels.

        Returns
        -------
        DataFrame where rows represent labels and columns represent channels.
        """
        return self._aggregate(aggregate_func=partial(self._aggregate_stats, stats_funcs=("sum")))

    def aggregate_mean(
        self,
    ) -> pd.DataFrame:
        """
        Computes the mean of pixel values within each labeled region for all image channels.

        Returns
        -------
        DataFrame where rows represent labels and columns represent channels.
        """
        return self._aggregate(aggregate_func=partial(self._aggregate_stats, stats_funcs=("mean")))

    def aggregate_var(
        self,
    ) -> pd.DataFrame:
        """
        Computes the variance of pixel values within each labeled region for all image channels.

        Returns
        -------
        DataFrame where rows represent labels and columns represent channels.
        """
        return self._aggregate(aggregate_func=partial(self._aggregate_stats, stats_funcs=("var")))

    def aggregate_kurtosis(
        self,
    ) -> pd.DataFrame:
        """
        Computes the kurtosis of pixel values within each labeled region for all image channels.

        Returns
        -------
        DataFrame where rows represent labels and columns represent channels.
        """
        return self._aggregate(aggregate_func=partial(self._aggregate_stats, stats_funcs=("kurtosis")))

    def aggregate_skew(
        self,
    ) -> pd.DataFrame:
        """
        Computes the skewness of pixel values within each labeled region for all image channels.

        Returns
        -------
        DataFrame where rows represent labels and columns represent channels.
        """
        return self._aggregate(aggregate_func=partial(self._aggregate_stats, stats_funcs=("skew")))

    def aggregate_max(
        self,
    ) -> pd.DataFrame:
        """
        Computes the maximum pixel value within each labeled region for all image channels.

        Returns
        -------
        DataFrame where rows represent labels and columns represent channels.
        """
        return self._aggregate(aggregate_func=self._aggregate_max)

    def aggregate_min(
        self,
    ) -> pd.DataFrame:
        """
        Computes the minimum pixel value within each labeled region for all image channels.

        Returns
        -------
        DataFrame where rows represent labels and columns represent channels.
        """
        return self._aggregate(aggregate_func=self._aggregate_min)

    def aggregate_area(self) -> pd.DataFrame:
        """
        Computes the area (number of pixels) for each labeled region in the mask.

        Returns
        -------
        A DataFrame with one column for area and one for label ID.
        """
        return _get_mask_area(
            self._mask, index=self._labels, instance_key=self._instance_key, instance_size_key=self._instance_size_key
        )

    def center_of_mass(self) -> pd.DataFrame:
        """
        Computes the center of mass for each labeled region in the mask.

        Note that we use scipy.ndimage.center_of_mass, which loads the mask into memory.

        Returns
        -------
        A DataFrame with columns for spatial coordinates (z,y,x) and label ID.
        """
        return _get_center_of_mass(self._mask, index=self._labels, instance_key=self._instance_key)

    def aggregate_quantiles(
        self,
        depth: int,
        quantiles: list[float] | NDArray | None = None,
        quantile_background: bool = False,
    ) -> list[pd.DataFrame]:
        """
        Computes quantiles of pixel values for each label in the mask, across all channels.

        Parameters
        ----------
        depth
            Depth to apply during Dask's `map_overlap` operation. Set depth > estimated diameter of labels.
        quantiles
            List of quantile values to compute (between 0 and 1). Defaults to [0.1, ..., 0.9].
        quantile_background
            Whether to include background label (0) in computation. Defaults to False.

        Returns
        -------
        List of DataFrames, one per quantile, with rows as labels and columns as channels.
        """
        # Returns a list of pandas DataFrames, one for per quantile.
        # Each DataFrame contains cells as rows and channels as columns.
        if quantiles is None:
            quantiles = np.linspace(0.1, 0.9, 9)
        # return a list of dataframes, one dataframe per quantile
        if quantile_background:
            _labels = self._labels
        else:
            _labels = self._labels[self._labels != 0]

        result = np.full((len(self._image), len(_labels), len(quantiles)), np.nan, dtype=np.float32)
        for i, _c_image in enumerate(self._image):
            result[i] = self._aggregate_quantiles_channel(
                image=_c_image,
                mask=self._mask,
                depth=depth,
                quantiles=quantiles,
                quantile_background=quantile_background,
            )

        result = result.transpose(2, 1, 0)  # shape after transpose: (nr_of_quantiles, nr_of_labels, nr_of_channels)
        dfs = [pd.DataFrame(_result) for _result in result]

        for _df in dfs:
            _df[self._instance_key] = _labels

        return dfs

    def aggregate_radii_and_axes(self, depth: int, calculate_axes: bool = True) -> pd.DataFrame:
        """
        Computes and aggregates radii and principal axes for segmented regions in the mask.

        This method returns a pandas DataFrame where each row represents a segmented cell.
        The DataFrame includes a column for the cell ID, three columns for the radii (sorted
        from largest to smallest), and optionally, nine columns (3*3) for the principal axes.

        Parameters
        ----------
        depth
            The depth at which the aggregation is performed, passed to `dask.array.map_overlap`.
            Please set `depth` > expected diameter of the labels.
        calculate_axes
            If True, the DataFrame will include the principal axes (default is True).

        Returns
        -------
        A DataFrame where:
        - Each row corresponds to a segmented cell, sorted by cell ID.
        - The next three columns contain the radii (sorted from highest to lowest).
        - If `calculate_axes` is True, the next nine columns contain the corresponding principal axes (flattened 3*3 matrix).
        - One column contains the cell ID.
        """
        results = self.aggregate_custom_channel(
            image=None,
            mask=self._mask,
            depth=depth,
            features=self._mask.ndim + self._mask.ndim * self._mask.ndim
            if calculate_axes
            else self._mask.ndim,  # returns 3 radii and 3 axis with (z,y,x) coordinates.
            dtype=np.float32,
            fn=_all_region_radii_and_axes,
            fn_kwargs={"calculate_axes": calculate_axes},
        )
        df = pd.DataFrame(results)
        df[self._instance_key] = self._labels[self._labels != 0]
        return df

    def _aggregate(self, aggregate_func: Callable[[da.Array], pd.DataFrame]) -> pd.DataFrame:
        results = aggregate_func()
        assert len(results) == 1
        df = pd.DataFrame(results[0])
        df[self._instance_key] = self._labels
        return df

    # this calculates "sum", "count", "mean", "var", "kurtosis" and "skew"
    def _aggregate_stats(
        self,
        stats_funcs: tuple[str, ...] = ("sum", "mean", "count", "var", "kurtosis", "skew"),
    ) -> list[NDArray]:
        # add an assert that checks that stats_funcs is in the list that is given.
        # first calculate the sum.
        if isinstance(stats_funcs, str):
            stats_funcs = (stats_funcs,)

        allowed_funcs = {"sum", "mean", "count", "var", "kurtosis", "skew"}
        invalid_funcs = [func for func in stats_funcs if func not in allowed_funcs]
        assert not invalid_funcs, (
            f"Invalid statistic function(s): '{invalid_funcs}'. Allowed functions: '{allowed_funcs}'."
        )

        if (
            "sum" in stats_funcs
            or "mean" in stats_funcs
            or "var" in stats_funcs
            or "kurtosis" in stats_funcs
            or "skew" in stats_funcs
        ):
            # calculate the sum
            def _calculate_sum_per_chunk(*arrays: NDArray) -> NDArray:
                assert len(arrays) == 2
                mask_block = arrays[0]
                image_block = arrays[1]
                unique_labels, new_labels = np.unique(mask_block, return_inverse=True)
                new_labels = np.reshape(new_labels, (-1,))  # flatten, since it may be >1-D
                idxs = np.searchsorted(unique_labels, self._labels)
                # make all of idxs valid
                idxs[idxs >= unique_labels.size] = 0
                found = unique_labels[idxs] == self._labels

                n_unique = unique_labels.size

                # NOTE: doing it without a for loop, e.g. with one bincount,
                # takes a lot of RAM when there are many channels (>100)
                # C = image.shape[0]
                # encoded_labels = new_labels[None, :] + n_unique * np.arange(C)[:, None]  # shape (c,i)
                # sums = np.bincount(
                #    encoded_labels.ravel(),
                #    weights=image_block.reshape(C, -1).ravel(),
                #    minlength=C
                #    * n_unique,
                # ).reshape(C, n_unique)

                sums = []
                for _c_image_block in image_block:
                    sums.append(
                        np.bincount(new_labels.ravel(), _c_image_block.ravel(), minlength=n_unique),
                    )
                sums = np.stack(sums)
                sums = sums[:, idxs]
                sums[:, ~found] = 0
                # sums is of shape (c,i), with i = len(self._labels)
                # we make it i,c,z,y,x
                sums = sums.T
                return sums[..., None, None, None].astype(np.float32)

            # add dummy C dimension for the mask, so we can pass it to map_blocks
            arrays = [self._mask[None, ...], self._image]

            meta = np.empty((0, 0, 0, 0, 0), dtype=np.float32)
            chunk_sum = da.map_blocks(
                _calculate_sum_per_chunk,
                *arrays,
                dtype=np.float32,  # for background pixels, theoretically this could overflow for float32
                chunks=(
                    (len(self._labels),),
                    self._image.chunks[0],
                    (1,) * self._image.numblocks[1],
                    (1,) * self._image.numblocks[2],
                    (1,) * self._image.numblocks[3],
                ),
                new_axis=0,  # add the i dimension
                meta=meta,
            )

            # chunk_sum is an array of shape (i, c, num_blocks_z,  num_blocks_y, num_blocks_x), with i=nr of unique labels
            sum = chunk_sum.reshape(len(self._labels), self._image.shape[0], -1).sum(axis=-1).compute()

        # then calculate the mean
        # i) first calculate the area
        if (
            "mean" in stats_funcs
            or "count" in stats_funcs
            or "var" in stats_funcs
            or "kurtosis" in stats_funcs
            or "skew" in stats_funcs
        ):
            if self._count is None:
                self._count = _calculate_area(self._mask, index=self._labels)

        # ii) then calculate the mean
        if "mean" in stats_funcs or "var" in stats_funcs or "kurtosis" in stats_funcs or "skew" in stats_funcs:
            self._mean = sum / self._count

        def sum_of_n(n: int) -> NDArray:
            # calculate the sum of n (e.g. squares if n=2) per cell
            def _calculate_sum_c_per_chunk(mask_block: NDArray, image_block: NDArray, block_info=None) -> NDArray:
                unique_labels, new_labels = np.unique(mask_block, return_inverse=True)
                new_labels = np.reshape(new_labels, (-1,))  # flatten, since it may be >1-D
                idxs = np.searchsorted(unique_labels, self._labels)
                # make all of idxs valid
                idxs[idxs >= unique_labels.size] = 0
                found = unique_labels[idxs] == self._labels

                # i) self._mean contains the mean over all channels (global), i.e. it is of shape (I,C),
                #  we only need the channels that are in the current block
                # ii) self._mean is the mean for all i, but we only select the ones that are in current block
                img_info = block_info[1]
                c_start, c_stop = img_info["array-location"][0]
                mean_found = self._mean[found, c_start:c_stop]

                n_unique = unique_labels.size
                C = image_block.shape[0]

                # NOTE: doing it without a for loop, e.g. with one bincount,
                # takes a lot of RAM when there are many channels (>100)
                # encoded_labels = new_labels[None, :] + n_unique * np.arange(C)[:, None]  # shape (c,i)
                # sums_c = np.bincount(
                #    encoded_labels.ravel(),
                #    weights=weights.ravel(),
                #    minlength=C
                #    * n_unique,  # NOTE: specifying minlength not really necessary here, keep it for documentation
                # ).reshape(C, n_unique)

                mean_per_pixel = mean_found.T[
                    :, new_labels
                ]  # creates an array of shape (c,image_block.shape[1]*image_block.shape[2]*image_block.shape[3])
                centered_weights = (image_block.reshape(C, -1) - mean_per_pixel) ** n
                sums_c = []
                for _c_weights in centered_weights:
                    sums_c.append(
                        np.bincount(new_labels.ravel(), _c_weights.ravel(), minlength=n_unique),
                    )
                sums_c = np.stack(sums_c)
                sums_c = sums_c[:, idxs]
                sums_c[:, ~found] = 0
                # sums_c is of shape (c,i), with i = len(self._labels)
                # we make it i,c,z,y,x
                sums_c = sums_c.T
                return sums_c[..., None, None, None].astype(np.float32)

            arrays = [self._mask[None, ...], self._image]

            meta = np.empty((0, 0, 0, 0, 0), dtype=np.float32)
            chunk_sum_c = da.map_blocks(
                _calculate_sum_c_per_chunk,
                *arrays,
                dtype=np.float32,
                chunks=(
                    (len(self._labels),),  # i: labels
                    self._image.chunks[0],  # c: channels
                    (1,) * self._image.numblocks[1],  # z-block index
                    (1,) * self._image.numblocks[2],  # y-block index
                    (1,) * self._image.numblocks[3],  # x-block index
                ),
                new_axis=0,  # add the i dimension
                meta=meta,
            )
            # chunk_sum_c is an array of shape (i, c, num_blocks_z,  num_blocks_y, num_blocks_x),
            # with i=nr of unique labels, and num_blocks the number of blocks in z,y,x of the image and the mask.
            sum_c_n = (
                chunk_sum_c.reshape(len(self._labels), self._image.shape[0], -1)
                .sum(axis=-1)  # sum over all chunks
                .compute()
            )

            return sum_c_n

        if "var" in stats_funcs or "kurtosis" in stats_funcs or "skew" in stats_funcs:
            sum_square = sum_of_n(n=2)
        if "kurtosis" in stats_funcs:
            sum_fourth = sum_of_n(n=4)
        if "skew" in stats_funcs:
            sum_third = sum_of_n(n=3)

        to_return = {}
        if "sum" in stats_funcs:
            to_return["sum"] = sum
        if "mean" in stats_funcs:
            to_return["mean"] = self._mean
        if "count" in stats_funcs:
            to_return["count"] = self._count
        if "var" in stats_funcs:
            to_return["var"] = sum_square / self._count
        if "kurtosis" in stats_funcs:
            # fisher kurtosis
            kurtosis = ((sum_fourth / self._count) / ((sum_square / self._count) ** 2)) - 3
            if np.isnan(kurtosis).any():
                log.warning("Replacing NaN values in 'kurtosis' with 0 for affected instances.")
                kurtosis = np.nan_to_num(kurtosis, nan=0)
            to_return["kurtosis"] = kurtosis
        if "skew" in stats_funcs:
            skewness = (sum_third / self._count) / (np.sqrt(sum_square / self._count)) ** 3
            if np.isnan(skewness).any():
                log.warning("Replacing NaN values in 'skewness' with 0 for affected instances.")
                skewness = np.nan_to_num(skewness, nan=0)

            to_return["skew"] = skewness

        to_return = [to_return[func] for func in stats_funcs if func in to_return]

        return to_return

    def _aggregate_max(
        self,
    ):
        return self._min_max(min_or_max="max")

    def _aggregate_min(
        self,
    ):
        return self._min_max(min_or_max="min")

    def _min_max(
        self,
        min_or_max: Literal["max", "min"],
    ) -> list[NDArray]:
        assert min_or_max in ["max", "min"], "Please choose from [ 'min', 'max' ]."

        min_dtype, max_dtype = _get_min_max_dtype(self._image)

        def _calculate_min_max_per_chunk(*arrays: NDArray) -> NDArray:
            assert len(arrays) == 2
            mask_block = arrays[0]
            image_block = arrays[1]
            min_or_max_array = []
            for _c_image_block in image_block:
                min_or_max_c = ndimage.labeled_comprehension(
                    _c_image_block,
                    mask_block,
                    self._labels,
                    func=np.max if min_or_max == "max" else np.min,
                    out_dtype=image_block.dtype,
                    default=min_dtype
                    if min_or_max == "max"
                    else max_dtype,  # set the default for labels in self._labels not found in current mask_block
                )  # also works if we have a lot of labels. scipy makes sure it only searches for labels of self._labels that are in mask_block
                min_or_max_array.append(min_or_max_c)
            min_or_max_array = np.stack(min_or_max_array)
            # max is (c,i)
            # make it (i,c,z,y,x)
            min_or_max_array = min_or_max_array.T
            return min_or_max_array[..., None, None, None]

        arrays = [self._mask[None, ...], self._image]
        meta = np.empty((0, 0, 0, 0, 0), dtype=self._image.dtype)

        chunk_min_max = da.map_blocks(
            _calculate_min_max_per_chunk,
            *arrays,
            dtype=self._image.dtype,
            chunks=(
                (len(self._labels),),
                self._image.chunks[0],
                (1,) * self._image.numblocks[1],
                (1,) * self._image.numblocks[2],
                (1,) * self._image.numblocks[3],
            ),
            new_axis=0,  # add the i dimension
            meta=meta,
        )

        # chunk_min_max is an array of shape (i, c, num_blocks_z,  num_blocks_y, num_blocks_x),
        # with i=nr of unique labels, and num_blocks the number of blocks in z,y,x of the image and the mask.
        chunk_min_max = chunk_min_max.reshape(len(self._labels), self._image.shape[0], -1)

        dask_min_max_func = da.max if min_or_max == "max" else da.min

        # return a list, so it is in line with self._aggregate_stats()
        return [dask_min_max_func(chunk_min_max, axis=-1).compute()]

    def _aggregate_quantiles_channel(
        self,
        image: da.Array,
        mask: da.Array,
        depth: int,
        quantiles: list[float] | None,
        quantile_background: bool = False,
    ) -> NDArray:
        if quantiles is None:
            quantiles = np.linspace(0.1, 0.9, 9)

        fn_kwargs = {
            "quantiles": quantiles,
        }

        results = self.aggregate_custom_channel(
            image=image,
            mask=mask,
            depth=depth,
            features=len(quantiles),
            dtype=np.float32,
            fn=_quantile_intensity_distribution,
            fn_kwargs=fn_kwargs,
        )

        def _quantile_background(
            image: da.Array,
            mask: da.Array,
            q: float = None,
            internal_method: str = "tdigest",
            background_label: int = 0,
        ) -> float:
            # calculate the quantile of the background
            q = q * 100
            image = image.flatten()
            mask = mask.flatten()
            background_non_nan_mask = (mask == background_label) & (~da.isnan(image))

            array = da.compress(background_non_nan_mask, image)

            return da.percentile(array, q=q, internal_method=internal_method).astype(np.float32)[0].compute()

        if quantile_background and quantiles is not None:
            results_background = np.array([_quantile_background(image, mask, q=_quantile) for _quantile in quantiles])
            results = np.vstack((np.array(results_background), results))

        return results

    def aggregate_custom_channel(
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
            to be a dask array. If not None, the function will apply the operation to the image
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
            A custom function that processes a mask and, optionally, an image array.
            The function must accept either one or two NumPy arrays:

            - The first argument is the mask, provided as an integer array.
            - The second argument (optional) is the image, given as a float or integer array.

            The function must return a 2D NumPy array of type `np.float`, where:

            - The first dimension corresponds to the number of unique labels in the mask (excluding label `0`), sorted by label number.
            - The second dimension represents the number of features computed by `fn`.

            The output of `fn` must **not** contain `NaN` values.
            User warning: the number of unique labels in the mask passed to `fn` is not equal to the number of
            unique labels from the global `mask` due to the use of `dask.array.map_overlap`.
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
            if mask.ndim != 3 or image.ndim != 3 or mask.ndim != image.ndim:
                raise ValueError(
                    f"mask and image must both be 3D (z, y, x). Got mask.shape={mask.shape}, image.shape={image.shape}"
                )
            arrays = [mask, image]
        else:
            arrays = [mask]
            if mask.ndim != 3:
                raise ValueError(f"mask must be 3D (z, y, x), but got {mask.ndim}D with shape {mask.shape}")
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
        # da.nansum ignores np.nan added by _featurize_custom_block
        results = da.nansum(dask_array, axis=1, dtype=dtype).reshape(-1, features)

        sanity, results = dask.compute(*[sanity, results])

        assert sanity, (
            "We expect exactly one non-NaN element per row (each column corresponding to a chunk of 'mask'). Please consider increasing 'depth' parameter."
        )

        return results


def _get_mask_area(
    mask: da.Array,
    index: NDArray | None = None,
    instance_key: str = _INSTANCE_KEY,
    instance_size_key: str = _CELLSIZE_KEY,
) -> pd.DataFrame:
    assert mask.ndim == 3, "Currently only 3D masks are supported ('z','y','x')."
    if index is None:
        index = da.unique(mask).compute()
    _result = _calculate_area(mask, index=index)
    return pd.DataFrame({instance_key: index, instance_size_key: _result.ravel()})


def _calculate_area(mask: da.Array, index: NDArray | None = None) -> NDArray:
    assert mask.ndim == 3, "Currently only 3D masks are supported ('z','y','x')."

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
        counts = np.bincount(
            new_labels, minlength=unique_labels.size
        )  # NOTE: specifying minlength not really necessary here, we keep it for documentation
        counts = counts[idxs]
        counts[~found] = 0
        # counts is an array of shape len(self._lables)
        # we make it i,z,y,x
        return counts[
            :, None, None, None
        ].astype(
            np.float32
        )  # TODO, potential overflow problem, should we remove background, or just cast to np.float64 (which it is already)

    meta = np.empty((0, 0, 0, 0, 0), dtype=np.float32)
    chunk_count = da.map_blocks(
        _calculate_count_per_chunk,
        mask,
        dtype=np.float32,
        chunks=(
            (len(index)),
            (1,) * mask.numblocks[0],
            (1,) * mask.numblocks[1],
            (1,) * mask.numblocks[2],
        ),
        new_axis=0,  # i, new axis contains the sum for each label (in the chunk)
        meta=meta,
    )
    sum = chunk_count.reshape(len(index), -1).sum(axis=-1).compute()
    return sum.reshape(-1, 1)


def _get_center_of_mass(
    mask: da.Array, index: NDArray | None = None, instance_key: str = _INSTANCE_KEY
) -> pd.DataFrame:
    assert mask.ndim == 3, "Currently only 3D masks are supported."
    if index is None:
        index = da.unique(mask).compute()

    # dask image center of mass for masks seems bugged (very slow), use in memory scipy.ndimage.center_of_mass.
    in_memory = True
    if not in_memory:
        coordinates = ndmeasure.center_of_mass(
            image=mask,
            label_image=mask,
            index=index,
        )
        coordinates = coordinates.compute()

    else:
        mask_in_memory = mask.compute()
        coordinates = np.array(
            ndimage.center_of_mass(
                input=mask_in_memory,
                labels=mask_in_memory,
                index=index,
            )
        )

    return pd.DataFrame(
        {
            instance_key: index,
            0: coordinates[:, 0],
            1: coordinates[:, 1],
            2: coordinates[:, 2],
        }
    )


def _quantile_intensity_distribution(
    mask: NDArray, image: NDArray, quantiles: list[float] | NDArray | None = None
) -> NDArray:
    """
    Calculate the quantile intensity distribution for each object in the mask.

    Parameters
    ----------
    image
        intensity values.
    mask
       should have same shape as image, with integer cell IDs (0 for background).
    quantiles
        list of quantiles to compute (e.g., [0.1, 0.2, ..., 0.9]).

    Returns
    -------
        The computed quantiles as a numpy array of shape `(nr of non zero labels, len(quantiles))`.
    """
    if quantiles is None:
        quantiles = np.linspace(0.1, 0.9, 9)

    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels != 0]

    if len(unique_labels) == 0:
        # No objects in the mask, return an empty array with shape (0, len(quantiles))
        return np.empty((0, len(quantiles)))

    result = np.full((len(unique_labels), len(quantiles)), np.nan, dtype=np.float32)

    for i, label in enumerate(unique_labels):
        object_intensities = image[mask == label]
        if object_intensities.size > 0:
            result[i] = np.quantile(object_intensities, quantiles)

    return result


def _all_region_radii_and_axes(mask: NDArray, calculate_axes: bool = True) -> NDArray:
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels != 0]

    nr_of_features = mask.ndim + mask.ndim**2 if calculate_axes else mask.ndim

    if len(unique_labels) == 0:
        return np.empty((0, nr_of_features))

    result = np.full((len(unique_labels), nr_of_features), np.nan, dtype=np.float32)

    for i, label in enumerate(unique_labels):
        radii, axes = _region_radii_and_axes(mask=mask, label=label)
        assert radii.shape == (mask.ndim,), f"Unexpected radii shape: {radii.shape}"
        assert axes.shape == (mask.ndim, mask.ndim), f"Unexpected axes shape: {axes.shape}"
        result[i] = np.concatenate((radii, axes.flatten())) if calculate_axes else radii
    return result


def _region_radii_and_axes(mask: NDArray, label: int) -> tuple[NDArray, NDArray]:
    """
    Compute the principal axes and radii of an object in a mask using PCA.

    This function extracts the coordinates of all pixels belonging to a given label in a segmentation mask,
    performs Principal Component Analysis (PCA) on those coordinates, and returns the radii (square roots
    of the eigenvalues) and the principal axes (eigenvectors).

    Parameters
    ----------
    mask : NDArray
        A binary or labeled mask where each object is represented by a unique integer.
    label : int
        The integer label of the object whose principal axes and radii are to be computed.

    Returns
    -------
    A tuple containing:
        - radii: A 1D numpy array of shape `(ndim,)` representing the spread of the object along each principal axis.
        - axes: A 2D numpy array of shape `(ndim, ndim)`, where each row is a principal axis (eigenvector).
    """
    _ndim = mask.ndim

    coords = np.column_stack(np.where(mask == label))

    if len(coords) < _ndim:
        radii = np.zeros(_ndim)
        return radii, np.eye(_ndim)

    pca = PCA(n_components=_ndim)
    pca.fit(coords)

    eigenvalues = pca.explained_variance_
    radii = np.sqrt(eigenvalues)

    axes = pca.components_

    # sort radii AND axes together
    # sort from largest to smallest eigenvalue
    # sklearn PCA returns sorted radii, but sorting ensures consistency across implementations
    sorted_indices = np.argsort(radii)[::-1]
    radii = radii[sorted_indices]
    axes = axes[sorted_indices]

    return radii, axes


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
    assert result.shape[0] == unique_masks.shape[0], (
        "Callable 'fn' should return an array with length equal to the number of non zero labels in the provided mask."
    )
    assert np.issubdtype(result.dtype, np.floating), "Callable 'fn' should return an array of dtype 'float'."
    if any(np.isnan(result).ravel()):
        raise AssertionError("Result of callable 'fn' is not allowed to contain NaN.")
    result = result[idxs]
    result[~found] = np.nan
    return result
