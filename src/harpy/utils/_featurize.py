from __future__ import annotations

import os
import shutil
import time
import uuid
from pathlib import Path

import dask
import dask.array as da
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from harpy.image.segmentation._utils import _add_depth_to_chunks_size
from harpy.utils._keys import _INSTANCE_KEY
from harpy.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

"""
try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    log.warning(
        "Module 'torch' not installed, please install 'torch' if you want to use the callable 'harpy.im.cellpose_callable' as model for 'harpy.im.segment'."
    )
"""


class Featurizer:
    """
    Helper class to featurize images and labels using Dask and PyTorch.

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

    def __init__(self, mask_dask_array: da.Array, image_dask_array: da.Array):
        if not np.issubdtype(mask_dask_array.dtype, np.integer):
            raise ValueError(f"'mask_dask_array' should contains chunks of type {np.integer}.")
        log.info("Calculating unique labels in the mask.")
        self._labels = (
            da.unique(mask_dask_array).compute()
        )  # calculate this one time during initialization, otherwise we would need to calculate this multiple times.
        assert image_dask_array.ndim == 4, "Currently only 4D image arrays are supported ('c', 'z', 'y', 'x')."
        assert mask_dask_array.ndim == 3, "Currently only 3D masks are supported ('z', 'y', 'x')."
        assert image_dask_array.shape[1:] == mask_dask_array.shape, (
            "The mask and the image should have the same spatial dimensions ('z', 'y', 'x')."
        )
        assert image_dask_array.chunksize[1:] == mask_dask_array.chunksize, (
            "Provided mask ('mask_dask_array') and image ('image_dask_array') do not have the same chunksize in ( 'z', 'y', 'x' ). Please rechunk."
        )
        self._image = image_dask_array
        self._mask = mask_dask_array

    def extract_instances(
        self,
        depth: int,  # ~max_diameter/2, depth in y and x,
        diameter: int
        | None = None,  # will be dimension of resulting chunks in y and x. Can be set to value < max_diameter to optimize performance
        remove_background: bool = True,
        zarr_output_path: str
        | Path
        | None = None,  # if zarr_output_path is specified, we compute the graph, otherwise we return a non-computed graph
        store_intermediate: bool = False,
    ) -> da.Array:
        """
        Extract per-label instance windows from the mask and image of size `diameter` in `y` and `x` using `dask.array.map_overlap` and `dask.array.map_blocks`.

        For every non-zero label in the mask, this method builds a Dask graph that
        slices out a centered, square window in the `y`,`x` plane around that instance.
        The `z` dimension is preserved from the source arrays. The corresponding image data
        are gathered for each instance (aligned in `z`, `y` and `x`, with channels preserved)
        Note that decreasing the chunk size of the provided image and mask dask array will lead to decreased
        consumption of ram. A good first guess for chunk size is `(5,1,2048,2048)`.

        Parameters
        ----------
        depth
            Passed to `dask.map_overlap`. Please set depth `~ max_diameter / 2`).
        diameter
            Optional explicit side length of the resulting `y`, `x` window for every
            instance. If not provided `diameter` is set to 2 times `depth`.
        remove_background
            If `True` (default), pixels outside the instance label within each
            window are set to background (e.g., zero) so that only the object remains
            inside the cutout. If ``False``, the entire window content is kept.
        zarr_output_path
            If a filesystem path (string or ``Path``) is provided, the extracted
            instances are **computed** and materialized to a Zarr store at that
            location. The returned object will still be a Dask array pointing at the
            written data, but all computations necessary to populate the store will
            have been executed. If `None` (default), no data are written and the
            method returns a **lazy** (not yet computed) Dask array.
        store_intermediate
            If `True`, and intermediate `.zarr` file is written to disk.
            Setting this to `True` will decrease ram usage.
            If `zarr_output_path` is not specified, it is not allowed to set
            `store_intermediate` to `True`.

        Returns
        -------
        tuple:

            - a dask array containing indices of extracted labels, shape `(i,)`.
            Dimension of `i` will be equal to the total number of non-zero labels in the mask.

            - a dask array of dimension `(i,c,z,y,x)`, with dimension of y and x equal to `diameter`,
             or 2*`depth` if `diameter` is not specified.

        Examples
        --------
        >>> fe = Featurizer(mask_dask_array=mask, image_dask_array=img)
        >>> instance_ids, instances = fe.extract_instances(depth=100, diameter=75)            # lazy graph
        >>> instances                                                             # inspect shape/chunks
        dask.array<...>

        # Persist to Zarr on disk (computes now)
        >>> instance_ids, instances = fe.extract_instances(
        ...     depth=100,
        ...     diameter=75,
        ...     zarr_output_path="instances.zarr",
        ... )

        # Keep full window content instead of masking to the instance
        >>> inst = fe.extract_instances(depth=100, diameter=75 remove_background=False)

        """
        if diameter is None:
            diameter = 2 * depth
        if diameter > 2 * depth:
            log.info("Diameter is set to a value > 2*depth. Consider decreasing diameter value for performance.")
        if store_intermediate and zarr_output_path is None:
            raise ValueError("Please specify a 'zarr_output_path' if 'store_intermediate' is 'True'.")
        _depth = {0: 0, 1: 0, 2: depth, 3: depth}

        array_mask = self._mask[None, ...]  # add trivial channel dimension
        array_image = self._image

        if array_image.numblocks[1] != 1:
            raise ValueError("Currently we do not allow chunking in z dimension.")

        array_mask = _transpose_chunks(array_mask, depth=_depth)
        array_image = _transpose_chunks(array_image, depth=_depth)

        if store_intermediate:
            _dirname_zarr = os.path.dirname(zarr_output_path)
            array_image_intermediate_store = os.path.join(_dirname_zarr, f"array_image_{uuid.uuid4()}.zarr")
            array_mask_intermediate_store = os.path.join(_dirname_zarr, f"array_mask_{uuid.uuid4()}.zarr")
            log.info(f"Writing to intermediate zarr store {array_image_intermediate_store}")
            log.info(f"Writing to intermediate zarr store {array_mask_intermediate_store}")
            array_mask.to_zarr(array_mask_intermediate_store)
            array_image.to_zarr(array_image_intermediate_store)
            array_mask = da.from_zarr(array_mask_intermediate_store)
            array_image = da.from_zarr(array_image_intermediate_store)

        N = 500  # guess for nr of labels per block.
        # This guess does not need to be exact, because we do a dask.compute() on labels_per_chunk and then dask does not need exact chunk sizes
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

        # This case should not happen if depth> max_diameter/2.
        if duplicates.size:
            log.info(
                f"There are {len(duplicates)} instances that are assigned to more than one chunk (instance id's: {duplicates}). "
                "Consider increasing depth. "
                "We will only keep the first occurence. "
            )

        # instances that are not assigned to any chunk. Can happen for edge cases, or if depth is too small.
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

        arrays = [array_mask, array_image]

        c_chunks = array_image.chunks[0]
        c_chunks = tuple([c_chunks[0] + 1] + list(c_chunks[1:]))  # we concat mask to first c channel chunk
        # returns c,z,i,y,x tensor
        dask_chunks = da.map_blocks(
            lambda *arrays, block_info=None, **kw: _featurize_block(*arrays, block_info=block_info, **kw),
            *arrays,
            dtype=np.float32,  # images and mask will be cast to dtype, if dtype==np.float32, max label supported is 2**24. # TODO cast to float64 if max label>2**24
            chunks=(
                c_chunks,  # e.g. (3+1,1) # do allow chunking in c.
                array_image.chunks[1],
                tuple(counts),
                (diameter,),
                (diameter,),
            ),
            new_axis=4,
            _depth=_depth,
            diameter=diameter,
            index=self._labels[self._labels != 0],
            remove_background=remove_background,
        )
        # make it i,c,z,y,x
        dask_chunks = dask_chunks.transpose(2, 0, 1, 3, 4)

        # Correct for non unique instances in instance_ids. Case should not happen for depth>max_diameter/2
        if len(idx) < len(instances_ids):  # equivalent to 'if duplicates.size:'
            log.info("Removing duplicates.")
            indices_to_keep = np.sort(idx)
            instances_ids = instances_ids[indices_to_keep]
            dask_chunks = dask_chunks[indices_to_keep]
            log.info("Finished removing duplicates.")

        if zarr_output_path is not None:
            dask_chunks.rechunk(dask_chunks.chunksize).to_zarr(zarr_output_path)
            dask_chunks = da.from_zarr(zarr_output_path)

        if store_intermediate:
            log.info(f"Deleting intermediate zarr store {array_image_intermediate_store}")
            log.info(f"Deleting intermediate zarr store {array_mask_intermediate_store}")
            shutil.rmtree(array_mask_intermediate_store)
            shutil.rmtree(array_image_intermediate_store)

        # Note that instance_ids are not sorted.
        # It is recommended not to do so (otherwise the dask_chunks array needs to be sorted, which is not optimal)
        return instances_ids, dask_chunks

    def _mean(
        self,
        diameter: int,  # estimated max diameter of cell in y, x
    ) -> pd.DataFrame:
        """Function calculates mean intensity. Please use optimized RasterAggregator.aggregate_mean() function."""
        # this is dummy function to illustrate working of ._extract_instances, please use optimized RasterAggregator
        instances_ids, dask_chunks = self.extract_instances(
            depth=diameter // 2 + 1,
            diameter=diameter,
            zarr_output_path=None,
            store_intermediate=False,
        )

        mask = dask_chunks[:, 0:1, ...]  # the first one is the mask
        image = dask_chunks[:, 1:, ...]
        mask = mask != 0
        sum_nonmask = (image * mask).sum(axis=(2, 3, 4))
        count_nonzero = mask.sum(axis=(2, 3, 4))

        avg_intensity = da.where(count_nonzero > 0, sum_nonmask / count_nonzero, 0)

        df = pd.DataFrame(avg_intensity)

        df[_INSTANCE_KEY] = instances_ids

        return df.sort_values(by=_INSTANCE_KEY).reset_index(drop=True)


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
    if c_location_block == 0:
        concat_mask = True
    else:
        concat_mask = False

    instances = _extract_instances(
        mask_block,
        image_block,
        size=(
            image_block.shape[1],
            diameter,
            diameter,
        ),  # no chunking in z dimension, so size in z is image_block.shape[1]
        concat_mask=concat_mask,
        remove_background=remove_background,
    )

    # return c,z,i,y,x tensor
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
    image: NDArray,
    size: tuple[int, int, int] = (1, 100, 100),
    remove_background=True,
    concat_mask: bool = True,
) -> NDArray:
    start = time.time()
    log.info("Extracting instances.")

    assert mask.ndim == image.ndim == 4
    assert mask.shape[0] == 1
    assert len(size) == 3
    if np.max(mask) > 2**24:
        raise ValueError(f"Maximum value allowed in mask is ({2**24}).")
    if not np.issubdtype(image.dtype, np.floating):
        if np.max(image) > 2**24:
            raise ValueError(f"Cannot safely cast image to float32. Please clip the image to <= {2**24}")
    else:
        if not np.issubdtype(image.dtype, np.float32):
            raise ValueError("Currently only images of dtype int and float32 are supported.")

    C, Z, Y, X = image.shape
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
        inst_img = image[:, zs:ze, ys:ye, xs:xe]  # .copy() # no copy needed, because we use np.where later
        inst_mask = mask[:, zs:ze, ys:ye, xs:xe]  # .copy()

        # keep only this label in the mask, zero everywhere else
        lbl = uniq[i]
        inst_mask = np.where(inst_mask == lbl, inst_mask, 0)
        if remove_background:
            # zero image outside the instance
            inst_img = np.where(inst_mask != 0, inst_img, 0)

        inst_img = _pad_array(inst_img, size=[size_z, size_y, size_x])
        inst_mask = _pad_array(inst_mask, size=[size_z, size_y, size_x])

        instance_list.append(inst_img)
        instance_mask_list.append(inst_mask)

    # create i,c,z,y,x
    mask_out = np.stack(instance_mask_list).astype(np.float32)
    img_out = np.stack(instance_list).astype(np.float32)

    log.info(
        f"Finished extracting instances, took {time.time() - start:.3f}s "
        f"({L / max(1e-9, (time.time() - start)):.2f} instances/s)."
    )

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


def _is_inside(mask_array: NDArray, instance_ids: NDArray, DY: int, DX: int):
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
