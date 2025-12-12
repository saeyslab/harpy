from __future__ import annotations

import inspect
import os
import shutil
import time
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import dask
import dask.array as da
import numpy as np
import pandas as pd
from loguru import logger as log
from numpy.typing import NDArray

from harpy.image.segmentation._utils import _add_depth_to_chunks_size
from harpy.utils._keys import _INSTANCE_KEY
from harpy.utils.utils import _dummy_embedding, _make_list


class Featurizer:
    """
    Helper class to featurize images and labels using Dask.

    Parameters
    ----------
    mask_dask_array
        A 3D Dask array of integer labels representing segmented regions.
        Expected shape is ('z', 'y', 'x'). Each unique integer value represents a separate label.
    image_dask_array
        A 4D Dask array representing the image data with shape ('c', 'z', 'y', 'x'),
        where 'c' is the number of channels. Can be `None` if only mask-based computations
        (e.g., count or center of mass) are required.

    Raises
    ------
    ValueError
        If `mask_dask_array` does not contain an integer dtype.
    AssertionError
        If `mask_dask_array` is not 3D.
    AssertionError
        If `image_dask_array` is is not 4D.
    AssertionError
        If spatial dimensions of `image_dask_array` and `mask_dask_array` do not match.
    AssertionError
        If chunk sizes of spatial dimensions do not match between image and mask.

    Notes
    -----
    - Unique labels are computed once at initialization for efficiency.
    - Image and mask must be aligned in spatial dimensions and chunking to ensure accurate and efficient featurization.
    """

    def __init__(self, mask_dask_array: da.Array, image_dask_array: da.Array | None = None):
        if not np.issubdtype(mask_dask_array.dtype, np.integer):
            raise ValueError(f"'mask_dask_array' should contains chunks of type {np.integer}.")
        log.info("Calculating unique labels in the mask.")
        self._labels = (
            da.unique(mask_dask_array).compute()
        )  # calculate this one time during initialization, otherwise we would need to calculate this multiple times.
        log.info("Finished calculating unique labels in the mask.")
        if image_dask_array is not None:
            assert image_dask_array.ndim == 4, "Currently only 4D image arrays are supported ('c', 'z', 'y', 'x')."
            assert image_dask_array.shape[1:] == mask_dask_array.shape, (
                "The mask and the image should have the same spatial dimensions ('z', 'y', 'x')."
            )
            assert image_dask_array.chunksize[1:] == mask_dask_array.chunksize, (
                "Provided mask ('mask_dask_array') and image ('image_dask_array') do not have the same chunksize in ( 'z', 'y', 'x' ). Please rechunk."
            )
        assert mask_dask_array.ndim == 3, "Currently only 3D masks are supported ('z', 'y', 'x')."

        self._image = image_dask_array
        self._mask = mask_dask_array

    def featurize(
        self,
        depth: int,
        embedding_dimension: int,
        diameter: int | None = None,
        remove_background: bool = True,
        zarr_output_path: str
        | Path
        | None = None,  # if zarr_output_path is specified, we compute the graph, otherwise we return a non-computed graph
        store_intermediate: bool = False,
        model: Callable[..., NDArray] = _dummy_embedding,
        batch_size: int | None = None,
        model_kwargs: Mapping[str, Any] = MappingProxyType({}),
        dtype: np.dtype = np.float32,
        **kwargs: Any,
    ) -> tuple[NDArray, da.Array]:
        """
        Extract per-instance feature vectors from the image/mask using a user-provided embedding `model`.

        See `hp.tb.featurize` for a full description.

        Parameters
        ----------
        zarr_output_path
            If a filesystem path (string or ``Path``) is provided, the feature Dask array is
            **computed** and materialized to a Zarr store at that location. The returned object will
            still be a Dask array backed by the written data, but all computations necessary to
            populate the store will have been executed. If `None` (default), no data are written and
            the method returns a **lazy** (not yet computed) Dask array.
        store_intermediate
            If `True`, intermediate `.zarr` data will be written to disk to reduce peak RAM usage.
            This is useful for large datasets. If `zarr_output_path` is not specified, it is
            not allowed to set `store_intermediate=True`.
            It is preferred to set `store_intermediate=False`, and work with a Dask client,
            so Dask can spill to disk.
        **kwargs
            Additional keyword arguments forwarded to `map_blocks`. Use with care.

        Returns
        -------
        tuple
            - A NumPy array of instance (label) indices, shape `(i,)`, where `i` equals the total
            number of non-zero labels in the mask, matching the rows in the feature matrix.
            - A Dask array (feature matrix) of features with shape `(i, embedding_dimension)`. If `zarr_output_path`
            is provided, this array points to the computed Zarr store; otherwise it is lazy.

        Examples
        --------
        >>> fe = Featurizer(mask_dask_array=mask, image_dask_array=img)

        # Lazy graph: generate embeddings with default dummy model
        >>> instance_ids, feats = fe.featurize(
        ...     depth=100,
        ...     embedding_dimension=128,
        ...     diameter=75,
        ... )
        >>> feats    # inspect shape/chunks without computing
        dask.array<...>

        # Use a custom model with arguments, and persist to Zarr on disk (computes now)
        >>> def my_model(batch, *, normalize: bool = True) -> np.ndarray:
        ...     # batch: (b, c, z, y, x) -> return (b, d)
        ...     vecs = batch.reshape(batch.shape[0], -1).astype(np.float32)
        ...     if normalize:
        ...         norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
        ...         vecs = vecs / norms
        ...     # project to desired dim (toy example)
        ...     W = np.random.RandomState(0).randn(vecs.shape[1], 64).astype(np.float32)
        ...     return vecs @ W
        ...
        >>> instance_ids, feats = fe.featurize(
        ...     depth=96,
        ...     embedding_dimension=64,
        ...     diameter=192,
        ...     model=my_model,
        ...     model_kwargs={"normalize": True},
        ...     batch_size=64,
        ...     zarr_output_path="features.zarr",
        ... )

        See Also
        --------
        harpy.tb.featurize : featurize instances in labels layer from an image layer using an embedding model.
        """
        if store_intermediate and zarr_output_path is None:
            raise ValueError("Please specify a 'zarr_output_path' if 'store_intermediate' is 'True'.")

        if zarr_output_path is not None and store_intermediate:
            intermediate_zarr_output_path = os.path.join(
                os.path.dirname(zarr_output_path), f"instances_{uuid.uuid4()}.zarr"
            )
        else:
            intermediate_zarr_output_path = None
        instance_ids, dask_chunks = self.extract_instances(
            depth=depth,
            diameter=diameter,
            remove_background=remove_background,
            zarr_output_path=intermediate_zarr_output_path,
            store_intermediate=store_intermediate,
            batch_size=batch_size,
            extract_mask=False,
        )

        assert "embedding_dimension" in inspect.signature(model).parameters, (
            f"Callable '{model.__name__}' must include the parameter 'embedding_dimension'."
        )

        # remove the masks, located at first dimension
        # dask_chunks = dask_chunks[:, 1:, ...]  # in self.extract_instances we already pass extract_mask==False

        # dask_chunks is array of dimension (i,c,z,y,x)
        if batch_size is not None:
            chunks = (
                batch_size,
                dask_chunks.shape[1],  # we do not allow chunking in c-dimension
                dask_chunks.shape[2],
                dask_chunks.shape[3],
                dask_chunks.shape[4],
            )
        else:
            chunks = (
                dask_chunks.chunks[0],
                dask_chunks.shape[1],  # we do not allow chunking in c-dimension
                dask_chunks.shape[2],
                dask_chunks.shape[3],
                dask_chunks.shape[4],
            )

        # this rechunk is potentially computationally demanding.
        dask_chunks = dask_chunks.rechunk(
            chunks
        )  # this is a no-op if batch_size is None, and if self._image/self._mask is not chunked in c dimension.
        embedded_dask_chunks = da.map_blocks(
            model,
            dask_chunks,
            chunks=(dask_chunks.chunks[0], embedding_dimension),
            drop_axis=[2, 3, 4],  # we do not allow chunking in 2,3 and 4, and we drop them
            dtype=dtype,  # FIXME, would it be possible to remove the dtype here
            **kwargs,
            embedding_dimension=embedding_dimension,
            **model_kwargs,
        )

        if zarr_output_path is not None:
            embedded_dask_chunks.rechunk(embedded_dask_chunks.chunksize).to_zarr(zarr_output_path)
            embedded_dask_chunks = da.from_zarr(zarr_output_path)

        if store_intermediate:
            log.info(f"Deleting intermediate zarr store {intermediate_zarr_output_path}")
            if Path(intermediate_zarr_output_path).suffix == ".zarr":
                shutil.rmtree(intermediate_zarr_output_path)

        return instance_ids, embedded_dask_chunks

    def calculate_instance_statistics(
        self,
        depth: int,
        statistic_dimension: int,
        diameter: int | None = None,
        zarr_output_path: str
        | Path
        | None = None,  # if zarr_output_path is specified, we compute the graph, otherwise we return a non-computed graph
        store_intermediate: bool = False,
        fn: Callable[
            ..., NDArray
        ] = _dummy_embedding,  # FIXME update this default to a dummy statistic to illustrate its use
        batch_size: int | None = None,
        fn_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any,
    ) -> tuple[NDArray, da.Array]:
        # FIXME write docstring
        # calculate single cell statistics
        if store_intermediate and zarr_output_path is None:
            raise ValueError("Please specify a 'zarr_output_path' if 'store_intermediate' is 'True'.")

        if zarr_output_path is not None and store_intermediate:
            intermediate_zarr_output_path = os.path.join(
                os.path.dirname(zarr_output_path), f"instances_{uuid.uuid4()}.zarr"
            )
        else:
            intermediate_zarr_output_path = None
        instance_ids, dask_chunks = self.extract_instances(
            depth=depth,
            diameter=diameter,
            remove_background=True,  #
            zarr_output_path=intermediate_zarr_output_path,
            store_intermediate=store_intermediate,
            batch_size=batch_size,
            extract_mask=True,  # always extract the mask, we need it; the user can decide if wants to calculate statistic on the mask, or on the image
        )

        # transpose dask chunks, so it is c,i,z,y,x, that way we can allow chunking in c dimension
        dask_chunks = dask_chunks.transpose(1, 0, 2, 3, 4)
        if self._image is not None:
            arrays = [dask_chunks[0][None], dask_chunks[1:]]  # the mask and the images
        else:
            arrays = [dask_chunks[0][None]]  # only the masks (and add dummy c dimension for consistency)

        c_chunks = arrays[1].chunks[0] if self._image is not None else arrays[0].chunks[0]

        # we allow chunking in c dimension, because statistic can typically be calculated on each channel separately, so no rechunk necessary
        calculated_statistic = da.map_blocks(
            _calculate_statistic_image_block if self._image is not None else _calculate_statistic_mask_block,
            *arrays,
            chunks=(c_chunks, arrays[0].chunks[1], statistic_dimension),  # c,i,statistic_dimension
            drop_axis=[
                3,
                4,
            ],  # we do not allow chunking in 3 and 4 (spatial dimensions of the instance windows), so we drop them
            dtype=np.float32 if self._image is not None else self._mask.dtype,
            fn=fn,
            statistic_dimension=statistic_dimension,
            fn_kwargs=fn_kwargs,
            **kwargs,
        )

        if zarr_output_path is not None:
            calculated_statistic.rechunk(calculated_statistic.chunksize).to_zarr(zarr_output_path)
            calculated_statistic = da.from_zarr(zarr_output_path)

        if store_intermediate:
            log.info(f"Deleting intermediate zarr store {intermediate_zarr_output_path}")
            if Path(intermediate_zarr_output_path).suffix == ".zarr":
                shutil.rmtree(intermediate_zarr_output_path)

        # make it (i,c,statistic_dimension)
        calculated_statistic = calculated_statistic.transpose(1, 0, 2)

        return instance_ids, calculated_statistic

        # FIXME write proper func for np.quantile case. then write for dummy case; then for radii and axes (only mask needed), and then for only mask (most occuring element)

    def extract_instances(
        self,
        depth: int,  # ~max_diameter/2, depth in y and x,
        diameter: int
        | None = None,  # will be dimension of resulting chunks in y and x. Can be set to value < max_diameter to optimize performance
        remove_background: bool = True,
        extract_mask: bool = False,
        zarr_output_path: str
        | Path
        | None = None,  # if zarr_output_path is specified, we compute the graph, otherwise we return a non-computed graph
        store_intermediate: bool = False,
        batch_size: int | None = None,
    ) -> tuple[NDArray, da.Array]:
        """
        Extract per-label instance windows from the mask and image of size `diameter` in `y` and `x` using :func:`dask.array.map_overlap` and :func:`dask.array.map_blocks`.

        See :func:`harpy.tb.extract_instances` for a full description.

        Parameters
        ----------
        store_intermediate
            If `True`, and intermediate `.zarr` file is written to disk.
            Setting this to `True` will decrease ram usage.
            If `zarr_output_path` is not specified, it is not allowed to set
            `store_intermediate` to `True`.
            It is recommended to set `store_intermediate=False`, and work with a Dask client,
            so Dask can spill to disk.

        Returns
        -------
        tuple:

            - a numpy array containing indices of extracted labels, shape `(i,)`.
            Dimension of `i` will be equal to the total number of non-zero labels in the mask.

            - a Dask array of dimension `(i,c+1,z,y,x)`, with dimension of `c` the number of channels in the image array.
            At channel index 0 of each instance, is the corresponding mask.
            dimension of `y` and `x` equal to `diameter`, or 2*`depth` if `diameter` is not specified.

        Examples
        --------
        ### example 1
        >>> import dask
        >>> import dask.array as da
        >>> import matplotlib.pyplot as plt
        >>> import hp as hp
        >>> sdata = hp.datasets.resolve_example()
        >>> mask_array = sdata[ "segmentation_mask" ].data[ None, ... ].rechunk( 1024 )
        >>> image_array = sdata[ "raw_image" ].data[ :, None, ... ].rechunk( 1024 )
        >>> fe = Featurizer(mask_dask_array=mask_array, image_dask_array=image_array)
        >>> instance_ids, instances = fe.extract_instances(depth=100, diameter=75)            # lazy graph
        >>> instances                                                             # inspect shape/chunks
        dask.array<...>
        # Persist to Zarr on disk (computes instances now)
        >>> instance_ids, instances = fe.extract_instances(
        ...     depth=100,
        ...     diameter=75,
        ...     zarr_output_path="instances.zarr",
        ... )
        # Keep full window content instead of masking to the instance
        >>> instance_ids, instances = fe.extract_instances(depth=100, diameter=75 remove_background=False)

        ### example 2 (with a visual sanity check of the extracted instances)
        >>> import dask
        >>> import dask.array as da
        >>> import matplotlib.pyplot as plt
        >>> import hp as hp

        >>> sdata = hp.datasets.resolve_example()
        >>> mask_array = sdata["segmentation_mask"].data[None, ...]

        >>> fe = Featurizer(
        ...     mask_dask_array=mask_array,
        ...     image_dask_array=None,
        ... )
        >>> instance_ids, instances = fe.extract_instances(
        ...     depth=100,
        ...     diameter=200,
        ...     batch_size=500,
        ...     extract_mask=True,
        ... )

        >>> instances = instances.compute()

        >>> instance_id = 23
        >>> mask = instances[instance_ids == instance_id][0][0][0]
        >>> plt.imshow(mask)

        >>> mask_array_remove = da.where(mask_array == instance_id, mask_array, 0)

        >>> _, y_, x_ = da.where(mask_array == instance_id)
        >>> y_, x_ = dask.compute(y_, x_)

        >>> plt.imshow(
        ...     mask_array_remove[
        ...         0, y_.min():y_.max(), x_.min():x_.max()
        ...     ]
        ... )

        See Also
        --------
        harpy.tb.extract_instances : extract instances in labels layer from an image layer.
        """
        if diameter is None:
            diameter = 2 * depth
        if diameter > 2 * depth:
            log.info("Diameter is set to a value > 2*depth. Consider decreasing diameter value for performance.")
        if store_intermediate and zarr_output_path is None:
            raise ValueError("Please specify a 'zarr_output_path' if 'store_intermediate' is 'True'.")
        _depth = {0: 0, 1: 0, 2: depth, 3: depth}
        if self._image is None and not extract_mask:
            log.info(
                "No image available and 'extract_mask' is False; forcing 'extract_mask=True' since nothing can be extracted otherwise."
            )
            extract_mask = True

        array_mask = self._mask[None, ...]  # add trivial channel dimension
        array_image = self._image

        if array_image is not None and array_image.numblocks[1] != 1:
            raise ValueError("Currently we do not allow chunking in z dimension.")

        array_mask = _transpose_chunks(array_mask, depth=_depth)
        array_image = _transpose_chunks(array_image, depth=_depth) if array_image is not None else None

        if store_intermediate:
            _dirname_zarr = os.path.dirname(zarr_output_path)
            array_mask_intermediate_store = os.path.join(_dirname_zarr, f"array_mask_{uuid.uuid4()}.zarr")
            log.info(f"Writing to intermediate zarr store {array_mask_intermediate_store}")
            array_mask.to_zarr(array_mask_intermediate_store)
            array_mask = da.from_zarr(array_mask_intermediate_store)
            array_image_intermediate_store = os.path.join(_dirname_zarr, f"array_image_{uuid.uuid4()}.zarr")
            log.info(f"Writing to intermediate zarr store {array_image_intermediate_store}")
            array_image.to_zarr(array_image_intermediate_store)
            array_image = da.from_zarr(array_image_intermediate_store)

        N = 500  # guess for nr of labels per block.
        # This guess does not need to be exact, because we do a dask.compute() on labels_per_chunk and then Dask does not need exact chunk sizes
        labels_per_chunk = da.map_blocks(
            _labels_per_block,
            array_mask,
            chunks=(
                (1,),  # trivial c dimension
                (1,) * len(array_mask.chunks[1]),
                N,  # N is a guess, add this step you do not know size of resulting chunks.
                (1,) * len(array_mask.chunks[3]),
            ),  # e.g. ((1,),(1, 1, 1,), (1, 1,),),
            dtype=array_mask.dtype,
            _depth=_depth,
            index=self._labels[self._labels != 0],
        )

        log.info("Calculating instance numbers per chunk. This could take a few minutes for large images.")
        labels_per_chunk = dask.compute(*labels_per_chunk.to_delayed().flatten())
        log.info("Finished calculating instance numbers per chunks.")
        labels_per_chunk = [_item.flatten() for _item in labels_per_chunk]
        counts = [len(_item) for _item in labels_per_chunk]

        instances_ids = np.concatenate(labels_per_chunk)

        unique_instance_ids, idx, _returned_counts = np.unique(instances_ids, return_index=True, return_counts=True)
        duplicates = unique_instance_ids[_returned_counts > 1]

        # This case can also happen if depth> max_diameter/2, e.g. mask consisting of two points, one in each chunk.
        if duplicates.size:
            log.info(
                f"There are {len(duplicates)} instances that are assigned to more than one chunk (instance id's: {duplicates}). "
                "If 'depth' is already set to a value > maximum expected diameter//2, this message can be ignored, "
                "else consider increasing depth. "
                "We will only keep the first occurence. "
            )

        # instances that are not assigned to any chunk. This should not happen if depth>max_diameter/2.
        _diff = np.setdiff1d(
            self._labels[self._labels != 0],
            unique_instance_ids,
        )

        if _diff.size:
            log.info(
                f"There are {len(_diff)} labels that could not be assigned to a chunk. "
                "Consider increasing the 'depth' parameter. "
                "Some labels may not be assigned to a chunk even at high depth values. "
                "This number should remain very small compared to the total number of instances. "
                f"(Instance ids: {_diff}.)"
            )

        arrays = [array_mask, array_image] if array_image is not None else [array_mask]

        if array_image is None:
            c_chunks = (1,)
        else:
            c_chunks = array_image.chunks[0]
            c_chunks = tuple([c_chunks[0] + 1] + list(c_chunks[1:])) if extract_mask else c_chunks
        # we concat mask to first c channel chunk
        # It is computationally more optimal to set extract_labels==False, and only extract the instances, that way we prevent a potential
        # computational intens rechunk along channel dimension due to adding mask to first chunk.
        # (chunks are now e.g. (2,1,1,1,1), and we would get chunksize=2, so rechunk by chunksize 2, leads to rechunk along channel dimension)
        # returns c,z,i,y,x tensor
        if extract_mask:
            if array_image is not None:
                output_dtype = np.result_type(array_image.dtype, array_mask.dtype)
            else:
                output_dtype = array_mask.dtype
        else:
            output_dtype = array_image.dtype
        _func = _featurize_mask_block if array_image is None else _featurize_block

        dask_chunks = da.map_blocks(
            lambda *arrays, block_info=None, **kw: _func(*arrays, block_info=block_info, **kw),
            *arrays,
            dtype=output_dtype,
            chunks=(
                c_chunks,  # e.g. (3+1,1) # do allow chunking in c.
                array_mask.chunks[1],
                tuple(counts),
                (diameter,),
                (diameter,),
            ),
            new_axis=4,
            _depth=_depth,
            diameter=diameter,
            index=self._labels[self._labels != 0],
            remove_background=remove_background,
            extract_mask=extract_mask,
        )
        # make it i,c,z,y,x
        dask_chunks = dask_chunks.transpose(2, 0, 1, 3, 4)

        chunksize = dask_chunks.chunksize
        if batch_size is not None:
            chunksize = (batch_size, chunksize[1], chunksize[2], chunksize[3], chunksize[4])
        # Correct for non unique instances in instance_ids.
        if len(idx) < len(instances_ids):  # equivalent to 'if duplicates.size:'
            log.info("Removing duplicates.")
            indices_to_keep = np.sort(idx)
            instances_ids = instances_ids[indices_to_keep]
            dask_chunks = dask_chunks[indices_to_keep]
            log.info("Finished removing duplicates.")

        # removing instances messes up the chunksize, so rechunk.
        dask_chunks = dask_chunks.rechunk(chunksize)

        if zarr_output_path is not None:
            dask_chunks.to_zarr(zarr_output_path)
            dask_chunks = da.from_zarr(zarr_output_path)

        if store_intermediate:
            log.info(f"Deleting intermediate zarr store {array_image_intermediate_store}")
            log.info(f"Deleting intermediate zarr store {array_mask_intermediate_store}")
            if Path(array_image_intermediate_store).suffix == ".zarr":
                shutil.rmtree(array_image_intermediate_store)
            if Path(array_mask_intermediate_store).suffix == ".zarr":
                shutil.rmtree(array_mask_intermediate_store)

        # Note that instance_ids are not sorted.
        # It is recommended not to do so (otherwise the dask_chunks array needs to be sorted, which is not optimal)
        return instances_ids, dask_chunks

    # need to make this a general function, that calculates various statistics in one shot.
    def quantiles(
        self,
        q: float | list[float] | NDArray,
        diameter: int,  # estimated max diameter of cell in y, x,
        depth: int | None = None,
        batch_size: int | None = None,
        instance_key: str = _INSTANCE_KEY,
    ) -> list[pd.DataFrame]:
        # FIXME write docstring
        if depth is None:
            depth = diameter // 2 + 1
            log.info(f"Parameter depth not provided; using default depth={depth} (computed from diameter={diameter})")
        # need to add a check to see if q is provided as a list, and if it is float, make it a list.
        # quantiles_lazy is a lazy dask array
        q = _make_list(q)
        fn_kwargs = {"q": q}
        instance_ids, quantiles_lazy = self.calculate_instance_statistics(
            depth=depth,
            statistic_dimension=len(q),
            diameter=diameter,
            zarr_output_path=None,
            store_intermediate=False,
            batch_size=batch_size,
            fn=_quantile,
            fn_kwargs=fn_kwargs,
        )

        quantiles = quantiles_lazy.compute().transpose(2, 0, 1)  # shape after transpose (statistic_dimension, i, c)
        dfs = [pd.DataFrame(_quantile) for _quantile in quantiles]

        for _df in dfs:
            _df[instance_key] = instance_ids

        return dfs

    def _mean(
        self,
        diameter: int,  # estimated max diameter of cell in y, x
        instance_key: str = _INSTANCE_KEY,
    ) -> pd.DataFrame:
        """Function calculates mean intensity. Please use the optimized RasterAggregator.aggregate_mean() function."""
        # this is dummy function to illustrate working of ._extract_instances, please use optimized RasterAggregator
        instances_ids, dask_chunks = self.extract_instances(
            depth=diameter // 2 + 1,
            diameter=diameter,
            zarr_output_path=None,
            extract_mask=True,
            store_intermediate=False,
        )

        mask = dask_chunks[
            :, 0:1, ...
        ]  # the first 'channel' dimension of (i,c,z,y,x) vector is the mask if extract_mask==True
        image = dask_chunks[:, 1:, ...]
        mask = mask != 0
        sum_nonmask = (image * mask).sum(axis=(2, 3, 4))
        count_nonzero = mask.sum(axis=(2, 3, 4))

        avg_intensity = da.where(count_nonzero > 0, sum_nonmask / count_nonzero, 0)

        df = pd.DataFrame(avg_intensity)

        df[instance_key] = instances_ids

        return df.sort_values(by=instance_key).reset_index(drop=True)


def _transpose_chunks(array: da.Array, depth: dict[int, int]):
    def return_block(block):
        # dummy function, to do a map_overlap
        return block

    # we only support regular chunked arrays, so rechunk first
    array = array.rechunk(array.chunksize)

    for i in range(len(depth)):
        if depth[i] != 0:
            if depth[i] > array.chunksize[i]:
                raise ValueError(
                    f"Depth for dimension {i} exceeds chunk size. Consider decreasing depth, or increase the chunk size."
                )

    _, _, Y_c, X_c = array.chunksize

    # only the last chunk can be different than the chunk size
    y_rest = Y_c - array.chunks[2][-1]
    x_rest = X_c - array.chunks[3][-1]

    array = da.pad(array, ((0, 0), (0, 0), (0, y_rest), (0, x_rest))).rechunk(array.chunksize)
    # no need to to a rechunk_overlap due to the padding

    ### PAD TO multiple of the chunk size, make sure to rechunk also with chunksize before, so it also supports irregular chunks as input
    # array = _rechunk_overlap(array, depth=depth, chunks=array.chunks)
    output_chunks = _add_depth_to_chunks_size(array.chunks, depth)

    # map an overlap
    array = da.map_overlap(
        return_block,
        array,
        depth=depth,
        chunks=output_chunks,
        allow_rechunk=False,
        dtype=array.dtype,
        trim=False,
        boundary=0,
    )
    _, _, chunksize_y, chunksize_x = array.chunksize

    array_list = []
    for _c_chunksize, _array in zip(array.chunks[0], array.to_delayed(), strict=True):
        array_list.append(
            da.concatenate(
                [
                    da.from_delayed(_item, shape=(_c_chunksize, 1, chunksize_y, chunksize_x), dtype=array.dtype)
                    for _item in _array.flatten()
                ],
                axis=2,
            )
        )  # for now assume z_dim==1
    array = da.concatenate(array_list, axis=0)
    return array


def _featurize_block(
    *arrays,
    index: NDArray,
    block_info: dict,
    _depth: dict[int, int],
    diameter: int,
    remove_background: bool,
    extract_mask: bool,
) -> NDArray:
    mask_block = arrays[0]
    assert len(arrays) == 2
    image_block = arrays[1]
    assert mask_block.shape[3:] == image_block.shape[3:]
    assert mask_block.ndim == image_block.ndim == 4
    if len(arrays) > 2:
        raise ValueError("Only accepts one or two arrays.")
    assert 0 not in index

    # do a copy of mask_block, because we alter mask_block inside _mask_center_of_mass_outside
    mask_block, _ = _mask_center_of_mass_outside(mask_block=mask_block.copy(), _depth=_depth)

    # i,c,z,y,x tensor
    # concat the mask instances to the block at channel location 0.
    c_location_block = block_info[1]["chunk-location"][0]
    _concat_mask = False
    if c_location_block == 0 and extract_mask:
        _concat_mask = True

    instances = _extract_instances(
        mask=mask_block,
        image=image_block,
        size=(
            image_block.shape[1],
            diameter,
            diameter,
        ),  # no chunking in z dimension, so size in z is image_block.shape[1]
        concat_mask=_concat_mask,
        remove_background=remove_background,
    )

    # return c,z,i,y,x tensor
    return instances.transpose(1, 2, 0, 3, 4)


def _featurize_mask_block(
    *arrays,
    index: NDArray,
    block_info: dict,  # placeholder
    _depth: dict[int, int],
    diameter: int,
    remove_background: bool,
    extract_mask: bool,  # placeholder
) -> NDArray:
    mask_block = arrays[0]
    assert len(arrays) == 1
    assert mask_block.ndim == 4
    assert 0 not in index

    # do a copy of mask_block, because we alter mask_block inside _mask_center_of_mass_outside
    mask_block, _ = _mask_center_of_mass_outside(mask_block=mask_block.copy(), _depth=_depth)

    instances = _extract_instances(
        mask=mask_block,
        image=None,
        size=(
            mask_block.shape[1],
            diameter,
            diameter,
        ),  # no chunking in z dimension, so size in z is image_block.shape[1]
        concat_mask=True,
        remove_background=remove_background,
    )

    # return 1,z,i,y,x tensor
    return instances.transpose(1, 2, 0, 3, 4)


def _labels_per_block(
    mask_block: NDArray,
    index: NDArray,
    _depth,
) -> NDArray:
    assert mask_block.ndim == 4
    assert 0 not in index

    _, unique_masks_inside = _mask_center_of_mass_outside(mask_block=mask_block.copy(), _depth=_depth)

    return unique_masks_inside.reshape(1, 1, -1, 1)


def _extract_instances(
    mask: NDArray,
    image: NDArray | None,
    size: tuple[int, int, int] = (1, 100, 100),
    remove_background=True,
    concat_mask: bool = True,
) -> NDArray:
    if image is None and not concat_mask:
        raise ValueError("'concat_mask' should be set to True if 'image' is None.")
    start = time.time()
    log.info("Extracting instances.")
    if image is not None:
        assert image.ndim == 4
    assert mask.ndim == 4
    assert mask.shape[0] == 1
    assert len(size) == 3
    # catch an edge case where mask id's would be >2**53
    if image is not None and concat_mask and np.issubdtype(image.dtype, np.floating) and np.max(mask) > 2**53:
        raise ValueError(f"Cannot safely cast to float (float64). Maximum value allowed in mask is ({2**53}).")

    if image is not None:
        C, Z, Y, X = image.shape
    else:
        C, Z, Y, X = mask.shape
    size_z, size_y, size_x = size

    # foreground coords once (O(V))
    fg = mask != 0
    if not np.any(fg):
        out_shape = (
            (0, C + (1 if concat_mask else 0), size_z, size_y, size_x)
            if concat_mask
            else (0, C, size_z, size_y, size_x)
        )
        log.info("No instances found.")
        return np.empty(out_shape, dtype=np.float32)

    # Order of coords matches order of mask[fg], so inv indices align
    _, zz, yy, xx = np.nonzero(fg)
    labels = mask[fg]  # shape (N,)

    # sanity check
    assert 0 not in labels

    uniq, inv = np.unique(labels, return_inverse=True)

    L = uniq.size

    # per-label bbox (O(N))
    zmin = np.full(L, Z, dtype=np.int64)
    ymin = np.full(L, Y, dtype=np.int64)
    xmin = np.full(L, X, dtype=np.int64)
    zmax = np.full(L, -1, dtype=np.int64)
    ymax = np.full(L, -1, dtype=np.int64)
    xmax = np.full(L, -1, dtype=np.int64)

    np.minimum.at(zmin, inv, zz)
    np.minimum.at(ymin, inv, yy)
    np.minimum.at(xmin, inv, xx)
    np.maximum.at(zmax, inv, zz)
    np.maximum.at(ymax, inv, yy)
    np.maximum.at(xmax, inv, xx)

    instance_list = []
    instance_mask_list = []

    for i in range(L):
        zs, ze = int(zmin[i]), int(zmax[i]) + 1
        ys, ye = int(ymin[i]), int(ymax[i]) + 1
        xs, xe = int(xmin[i]), int(xmax[i]) + 1

        # If we want to keep background, we need to extend the bbox to size_z,size_y,size_x
        if not remove_background:
            zl = ze - zs
            yl = ye - ys
            xl = xe - xs
            if zl < size_z:
                zs = max(0, zs - (size_z - zl) // 2)
                ze = min(Z, ze + ((size_z - zl) // 2) + 1)  # +1 to account for rounding when //2
            if yl < size_y:
                ys = max(0, ys - (size_y - yl) // 2)
                ye = min(Y, ye + ((size_y - yl) // 2) + 1)
            if xl < size_x:
                xs = max(0, xs - (size_x - xl) // 2)
                xe = min(X, xe + ((size_x - xl) // 2) + 1)

        # crop
        if image is not None:
            inst_img = image[:, zs:ze, ys:ye, xs:xe]  # .copy() # no copy needed, because we use np.where later
        inst_mask = mask[:, zs:ze, ys:ye, xs:xe]  # .copy()

        # keep only this label in the mask, zero everywhere else
        lbl = uniq[i]
        inst_mask = np.where(inst_mask == lbl, inst_mask, 0)
        if remove_background and image is not None:
            # zero image outside the instance
            inst_img = np.where(inst_mask != 0, inst_img, 0)

        if image is not None:
            inst_img = _pad_array(inst_img, size=[size_z, size_y, size_x])
        inst_mask = _pad_array(inst_mask, size=[size_z, size_y, size_x])

        if image is not None:
            instance_list.append(inst_img)
        instance_mask_list.append(inst_mask)

    # create i,c,z,y,x
    mask_out = np.stack(instance_mask_list)
    if image is not None:
        img_out = np.stack(instance_list)

    log.info(
        f"Finished extracting instances, took {time.time() - start:.3f}s "
        f"({L / max(1e-9, (time.time() - start)):.2f} instances/s)."
    )

    if image is None:
        return mask_out

    return np.concatenate([mask_out, img_out], axis=1) if concat_mask else img_out


def _pad_array(arr: NDArray, size: tuple[int, int, int]) -> NDArray:
    """
    Resize a numpy array to shape (..., size, size, size).

    - Crop if dimensions exceed 'size'.
    - Pad with zeros if smaller.

    Raises
    ------
    ValueError
        If the input array has ndim < 3.
    """
    if arr.ndim < 3:
        raise ValueError("Array must have at least 3 dimensions (z, y, x).")
    if len(size) != 3:
        raise ValueError("'size' must be a tuple of length 3.")

    size_z, size_y, size_x = size

    # crop in the center
    # arr = arr[..., :size_z, :size_y, :size_x]

    z, y, x = arr.shape[-3:]

    # central crop if array is too large
    start_z = max((z - size_z) // 2, 0)
    start_y = max((y - size_y) // 2, 0)
    start_x = max((x - size_x) // 2, 0)

    end_z = start_z + min(size_z, z)
    end_y = start_y + min(size_y, y)
    end_x = start_x + min(size_x, x)

    arr = arr[..., start_z:end_z, start_y:end_y, start_x:end_x]

    # padding for smaller arrays
    z, y, x = arr.shape[-3:]

    pad_z = max(size_z - z, 0)
    pad_y = max(size_y - y, 0)
    pad_x = max(size_x - x, 0)

    pad_z_left = pad_z // 2
    pad_z_right = pad_z - pad_z_left

    pad_y_left = pad_y // 2
    pad_y_right = pad_y - pad_y_left

    pad_x_left = pad_x // 2
    pad_x_right = pad_x - pad_x_left

    pad_width = [(0, 0)] * (arr.ndim - 3) + [
        (pad_z_left, pad_z_right),
        (pad_y_left, pad_y_right),
        (pad_x_left, pad_x_right),
    ]

    arr = np.pad(arr, pad_width=pad_width, mode="constant", constant_values=0)

    return arr


def _mask_center_of_mass_outside(mask_block: NDArray, _depth):
    assert len(_depth) == 4
    assert mask_block.ndim == 4
    # Set all masks that are fully outside the region to zero, they will be covered by other chunks
    DY = _depth[2]
    DX = _depth[3]
    subset = mask_block[:, :, DY:-DY, DX:-DX]
    # Unique masks gives you all masks that are at least partially in 'original' array (i.e. without depth added)
    unique_masks = np.unique(subset)
    unique_masks = unique_masks[unique_masks != 0]
    # Create a mask for labels that are NOT in unique_masks
    mask = ~np.isin(mask_block, unique_masks)
    mask_block[mask] = 0

    slabs = [
        mask_block[:, :, -(2 * DY) : -DY, DX:-DX],  # +Y upper band
        mask_block[:, :, DY:-DY, -(2 * DX) : -DX],  # +X right band
        mask_block[:, :, DY : 2 * DY, DX:-DX],  # -Y lower band
        mask_block[:, :, DY:-DY, DX : 2 * DX],  # -X left band
    ]

    slabs = [s for s in slabs if s.size]

    if slabs:
        border_labels = np.unique(np.concatenate([s.ravel() for s in slabs]))
        border_labels = border_labels[border_labels != 0]  # drop background
    else:
        border_labels = np.array([], dtype=mask_block.dtype)

    if not border_labels.size:
        # if no border labels, nothing to do
        return mask_block, unique_masks[unique_masks != 0]

    """
    border_labels = list(border_labels)
    center_of_mass_border_labels = ndimage.center_of_mass(input=mask_block, labels=mask_block, index=border_labels)

    centers = np.array(center_of_mass_border_labels).astype(np.float32)  # shape (N, 4)
    shape = np.array(mask_block.shape)

    border_labels_in_original_block = (
        (centers[:, 2] >= _depth[2])
        & (centers[:, 2] < shape[2] - _depth[2])
        & (centers[:, 3] >= _depth[3])
        & (centers[:, 3] < shape[3] - _depth[3])
    )

    # get the border labels not to consider
    if border_labels:
        border_labels_not_to_consider = np.array(border_labels)[~np.array(border_labels_in_original_block)]
        unique_masks = unique_masks[~np.isin(unique_masks, border_labels_not_to_consider)]
    """
    border_labels_in_original_block = _is_inside(
        mask_block.squeeze(0),
        instance_ids=border_labels,
        DY=_depth[2],
        DX=_depth[3],
    )

    border_labels_not_to_consider = np.array(border_labels)[~np.array(border_labels_in_original_block)]
    unique_masks = unique_masks[~np.isin(unique_masks, border_labels_not_to_consider)]

    unique_masks = unique_masks[unique_masks != 0]

    mask = ~np.isin(mask_block, unique_masks)
    mask_block[mask] = 0

    return mask_block, unique_masks


def _is_inside(mask_array: NDArray, instance_ids: NDArray, DY: int, DX: int) -> NDArray:
    """
    Check for each id in instance_ids that area is in DY:-DY,DX:-DX

    Count pixels (areas) for each cell_id within nine regions of mask_array.

    Parameters
    ----------
    mask_array : (Y, X) uint ndarray
    instance_ids : 1D array-like of uint32
    DY, DX : int
        Border thicknesses. Assumes 0 < DY < Y and 0 < DX < X.

    Returns
    -------
    Instance ids that are in DY:-DY,DX:-DX.
    """
    assert mask_array.ndim == 3
    assert instance_ids.ndim == 1
    _, Y, X = mask_array.shape
    assert DY < Y
    assert DX < X

    assert 0 not in instance_ids  # we assume these do not contain 0, are sorted, and unique
    # ids = ids[ ids!=0 ]

    # Define convenient slices
    top = slice(0, DY)
    middle_y = slice(DY, Y - DY)
    bottom = slice(Y - DY, Y)

    left = slice(0, DX)
    middle_x = slice(DX, X - DX)
    right = slice(X - DX, X)

    regions = {
        0: (middle_y, middle_x),  # [DY:-DY, DX:-DX]
        1: (bottom, left),  # [-DY:Y, 0:DX]
        2: (bottom, middle_x),  # [-DY:Y, DX:-DX]
        3: (bottom, right),  # [-DY:Y, -DX:X]
        4: (middle_y, right),  # [DY:-DY, -DX:X]
        5: (top, right),  # [0:DY, -DX:X]
        6: (top, middle_x),  # [0:DY, DX:-DX]
        7: (top, left),  # [0:DY, 0:DX]
        8: (middle_y, left),  # [DY:-DY, 0:DX]
    }

    def counts_for(view: NDArray, ids: NDArray) -> NDArray:
        # Flatten region labels, find positions in ids_sorted via searchsorted
        vals = view.ravel()
        idx = np.searchsorted(ids, vals)
        # make idx valid
        idx[idx >= ids.size] = 0
        # bincount
        hits = ids[idx] == vals
        return np.bincount(idx[hits], minlength=ids.size)

    stacked = np.empty((9, instance_ids.size), dtype=np.uint32)
    for r in range(9):
        ysl, xsl = regions[r]
        view = mask_array[:, ysl, xsl]
        cnts = counts_for(view=view, ids=instance_ids)
        stacked[r] = cnts

    max_rows = np.argmax(stacked, axis=0)

    # True when the max is in row 0 ->inside
    mask = max_rows == 0

    return mask


def _calculate_statistic_mask_block(
    *arrays: NDArray,
    statistic_dimension: int,
    func: Callable[..., NDArray],
    func_kwargs: Mapping[str, Any] = MappingProxyType({}),
):
    # array should be an array of shape 1,i,z,y,x
    assert len(arrays) == 2
    mask = arrays[0]
    assert mask.ndim == 5
    mask = mask[0]  # make it i,z,y,x

    # FIXME-> finish this
    # here it is probably best not to flatten the array (i.e. you want to calculate some regional properties about the mask, radii, axis,...)


def _calculate_statistic_image_block(
    *arrays: NDArray,
    statistic_dimension: int,
    fn: Callable[
        ..., NDArray
    ],  # callable that expects shape=(c, number of pixels corresponding to non zero mask for instance i) and returns shape=(c,statistic_dimension)
    fn_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> NDArray:
    # array should be an array of shape c,i,z,y,x
    assert len(arrays) == 2
    mask = arrays[0]
    image = arrays[1]
    assert mask.ndim == 5
    assert image.ndim == 5

    mask = mask.transpose(1, 0, 2, 3, 4)
    image = image.transpose(1, 0, 2, 3, 4)  # make it i,c,z,y,c

    I, C, _, _, _ = image.shape

    mask_flat = (mask != 0).reshape(I, -1)  # i,z*y*x

    vals_flat = image.reshape(I, C, -1)  # i,c,z*y*x

    calculated_statistic = np.full(
        (I, C, statistic_dimension), np.nan, dtype=np.float32
    )  # set statistic to nan when there is no mask found

    for i in range(I):
        m = mask_flat[i]  # (z*y*x)
        if not np.any(m):
            # this could happen for edge cases, i.e. very small masks
            log.info(
                "Instance found with no non-zero corresponding mask in the instance window. "
                "This could happen for very small masks."
            )  # skip instances with no non zero mask
            continue

        v = vals_flat[
            i, :, m
        ]  # shape of v=(number of pixels corresponding to non zero mask for instance i,c)  # v gives you all pixels in image for which corresponding mask is non zero -> now we can apply our statistic

        #  pass v.T to fn, shape of v.T=(c, number of pixels corresponding to non zero mask for instance i)
        _result = fn(v.T, **fn_kwargs)  # shape of result=(c, statistic_dimension)
        calculated_statistic[i] = _result

    # calculated statistic is of shape ( i,c,statistic_dimension )
    # so we transpose to (c,i,statistic_dimension)
    return calculated_statistic.astype(np.float32).transpose(1, 0, 2)


def _quantile(
    array: NDArray,
    q: list[float] | NDArray | None = None,
) -> NDArray:
    assert array.ndim == 2
    # shape of array=(c, number of pixels corresponding to non zero mask for instance i)
    if q is None:  # maybe leave this fallback out
        q = np.linspace(0.1, 0.9, 9)
    result = np.quantile(array, q=q, axis=1)  # result of shape ( statistic_dimension, c)
    result = result.T
    # sanity check
    assert result.shape[0] == array.shape[0]
    assert result.shape[1] == len(q)
    return result  # of shape ( c, statistic_dimension)


def _spread():
    # Q3 - Q1
    # to implement
    pass
