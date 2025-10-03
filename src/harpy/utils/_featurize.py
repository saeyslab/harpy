from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

import dask
import dask.array as da
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from scipy import ndimage

from harpy.utils._keys import _INSTANCE_KEY
from harpy.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    log.warning(
        "Module 'torch' not installed, please install 'torch' if you want to use the callable 'harpy.im.cellpose_callable' as model for 'harpy.im.segment'."
    )


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
        If `image_dask_array` is provided but is not 4D.
    AssertionError
        If spatial dimensions of `image_dask_array` and `mask_dask_array` do not match.
    AssertionError
        If chunk sizes of spatial dimensions do not match between image and mask.

    Notes
    -----
    - Unique labels are computed once at initialization for efficiency.
    - Image and mask must be aligned in spatial dimensions and chunking to ensure accurate and efficient featurization.
    """

    def __init__(self, mask_dask_array: da.Array, image_dask_array: da.Array | None):
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

    def mean(
        self,
        diameter: int,  # estimated max diameter of cell in y, x
        device: str = "cpu",
    ) -> pd.DataFrame:
        fn_kwargs = {
            "size": (self._image.shape[1], diameter, diameter),
            "device": device,
        }

        _result = self.featurize(
            depth=diameter * 2,
            features=1,
            dtype=np.float32,
            fn=_stats,
            fn_kwargs=fn_kwargs,
        )

        df = pd.DataFrame(_result)

        df[_INSTANCE_KEY] = self._labels[self._labels != 0]

        return df

    def featurize(
        self,
        depth: int,  # choose depth > estimated diameter of largest cell
        fn: Callable[[NDArray[np.int_], NDArray[np.int_ | np.float_] | None], NDArray[np.float_]],
        fn_kwargs: Mapping[str, Any] = MappingProxyType(
            {}
        ),  # fn is a callable that returns a 1D array with len == nr of unique labels in the mask passed to fn excluding 0
        dtype: np.dtype = np.float32,  # output dtype
        features: int = 1,
    ) -> NDArray:
        mask = self._mask
        image = self._image
        assert mask.numblocks[0] == 1, "masks can not be chunked in z-dimension. Please rechunk."
        depth = (0, 0, depth, depth)
        _labels = self._labels[self._labels != 0]
        if image is not None:
            assert image.numblocks[0] == 1, (
                "image can not be chunked in c-dimension. Please rechunk."
            )  # TODO check if also works if chunked in c-dimension. Probably it does.
            assert image.numblocks[1] == 1, "image can not be chunked in z-dimension. Please rechunk."
            if mask.ndim != 3 or image.ndim != 4:
                raise ValueError(
                    f"mask must be 3D (z, y, x) and image 4D (c,z,y,x). Got mask.shape={mask.shape}, image.shape={image.shape}"
                )
            arrays = [mask[None, ...], image]  # add trivial channel dimension
        else:
            if mask.ndim != 3:
                raise ValueError(f"mask must be 3D (z, y, x), but got {mask.ndim}D with shape {mask.shape}")
            arrays = [mask[None, ...]]  # add trivial channel dimension

        dask_chunks = da.map_overlap(
            lambda *arrays, block_info=None, **kw: _featurize_custom_block(*arrays, block_info=block_info, **kw),
            *arrays,
            dtype=dtype,
            chunks=(len(_labels), features),
            trim=False,
            drop_axis=[0, 1],
            boundary=0,
            depth=depth,
            index=_labels,
            _depth=depth,
            fn=fn,  # callable.
            fn_kwargs=fn_kwargs,  # keywords of the callable
            features=features,
        )

        dask_chunks = [
            da.from_delayed(_chunk, shape=(len(self._labels) - 1, features), dtype=dtype).reshape(-1, 1)
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


def _featurize_custom_block(
    *arrays,
    index: NDArray,
    block_info,
    _depth,
    fn: Callable,
    fn_kwargs: Mapping[str, Any] = MappingProxyType({}),
    features: int = 1,
) -> NDArray:
    # same as _aggregate_custom_block, but now dimension of mask_block and image_block needs to be 4, and image_block contains all channels, need to put assert there
    mask_block = arrays[0]
    assert len(_depth) == 4
    if len(arrays) == 2:
        image_block = arrays[1]
        assert mask_block.shape[1:] == image_block.shape[1:]
        assert mask_block.ndim == image_block.ndim == 4
    else:
        image_block = None
    if len(arrays) > 2:
        raise ValueError("Only accepts one or two arrays.")
    assert 0 not in index
    total_nr_of_blocks = block_info[0]["num-chunks"]
    block_location = block_info[0]["chunk-location"]
    # check if chunk is on border of larger dask array
    y_upper_border = block_location[2] + 1 == total_nr_of_blocks[2]
    x_upper_border = block_location[3] + 1 == total_nr_of_blocks[3]
    y_lower_border = block_location[2] == 0
    x_lower_border = block_location[3] == 0

    border_labels = set()
    if not y_upper_border:
        # you do not only extract the ones on border, but in the overlap region that is in the current block,
        # e.g. you go from _depth[2] : _depth[2] * 2
        # otherwise you could miss masks that are crossing the border, but are non-continuous and do not overlap with the border.
        # we still assume diameter < depth
        border_labels.update(set(np.unique(mask_block[:, :, -(_depth[2] * 2) : -(_depth[2]), _depth[3] : -_depth[3]])))
    if not x_upper_border:
        border_labels.update(set(np.unique(mask_block[:, :, _depth[2] : -_depth[2], -(_depth[3] * 2) : -(_depth[3])])))
    if not y_lower_border:
        border_labels.update(set(np.unique(mask_block[:, :, _depth[2] : _depth[2] * 2, _depth[3] : -_depth[3]])))
    if not x_lower_border:
        border_labels.update(set(np.unique(mask_block[:, :, _depth[2] : -_depth[2], _depth[3] : _depth[3] * 2])))
    if 0 in border_labels:
        border_labels.remove(0)

    border_labels = list(border_labels)
    center_of_mass_border_labels = ndimage.center_of_mass(input=mask_block, labels=mask_block, index=border_labels)

    def _isin_original(center: tuple[float, float, float, float]):
        return (
            center[2] >= _depth[2]
            and center[2] < (mask_block.shape[2] - _depth[2])
            and center[3] >= _depth[3]
            and center[3] < (mask_block.shape[3] - _depth[3])
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
    subset = mask_block[:, :, _depth[2] : -_depth[2], _depth[3] : -_depth[3]]
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
    if image_block is not None:
        c_dim = image_block.shape[0]
    else:
        c_dim = 1
    result = result.reshape(-1, features * c_dim)
    assert result.shape[0] == unique_masks.shape[0], (
        "Callable 'fn' should return an array with length equal to the number of non zero labels in the provided mask."
    )
    assert np.issubdtype(result.dtype, np.floating), "Callable 'fn' should return an array of dtype 'float'."
    if any(np.isnan(result).flatten()):
        raise AssertionError("Result of callable 'fn' is not allowed to contain NaN.")
    result = result[idxs]
    result[~found] = np.nan
    return result


def _stats(mask: NDArray, image: NDArray, size: tuple[int, int, int], device="cpu") -> NDArray:
    instances_mask, instances = _extract_instances(
        mask, image, size=size, device="cpu"
    )  # extracting instances on cpu is fastest
    # can be any complex function here
    instances_mask = instances_mask.to(device)
    instances = instances.to(device)
    mean = _masked_mean(instances, instances_mask, dim=[2, 3, 4])
    return mean.cpu().numpy()


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, dim=None, keepdim=False, nan_if_empty=True) -> torch.Tensor:
    mask = mask != 0  # bool mask
    x = x.to(torch.float32)  # ensure float division

    cnt = mask.sum(dim=dim, keepdim=keepdim)
    s = x.masked_fill(~mask, 0).sum(dim=dim, keepdim=keepdim)
    mean = s / cnt.clamp_min(1)

    # note that in our case cnt==0 will not happen, because we only feed 'real' masks
    if nan_if_empty:
        mean = torch.where(cnt == 0, torch.zeros_like(mean), mean)
    return mean


def _extract_instances(
    mask: NDArray,
    image: NDArray,
    size: tuple[int, int, int] = (1, 100, 100),
    device: str = "cpu",  # this is fastest on cpu, so maybe rewrite to numpy again
) -> tuple[torch.Tensor, torch.Tensor]:
    # returns two tensors of shape i,c,z,y,x
    #
    start = time.time()
    log.info("Extracting instances.")
    # rewrite this in torch
    assert mask.ndim == image.ndim == 4
    assert len(size) == 3

    if np.max(mask) > torch.iinfo(torch.int32).max:
        raise ValueError(f"Maximum value allowed in mask is {torch.iinfo(torch.int32)}.")

    mask = torch.from_numpy(mask).to(torch.int32).to(device)

    if not np.issubdtype(image.dtype, np.floating):
        if np.max(image) > 2**24:
            raise ValueError(f"Cannot savely cast image to float32. Please clip the image to values <={2**24}")
    else:  # case where it is floating -> we only support np.float32
        if not np.issubdtype(image.dtype, np.float32):
            raise ValueError("Currently only images of dtype int and float32 are supported.")

    image = torch.from_numpy(image).to(torch.float32).to(device)

    unique_labels = torch.unique(mask)
    unique_labels = unique_labels[unique_labels != 0]

    instance_list = []
    instance_mask_list = []

    for _label in unique_labels:
        if _label == 0:
            continue
        size_z, size_y, size_x = size  # ~ diameter of the cell ~2*depth ( 2* depth+1 )
        # _, z_mask, y_mask, x_mask = torch.where(mask == _label)
        vox_idx = (mask == _label).nonzero(as_tuple=False)
        z_min = int(vox_idx[:, 1].amin().item())
        z_max = int(vox_idx[:, 1].amax().item())
        y_min = int(vox_idx[:, 2].amin().item())
        y_max = int(vox_idx[:, 2].amax().item())
        x_min = int(vox_idx[:, 3].amin().item())
        x_max = int(vox_idx[:, 3].amax().item())

        # z_max, z_min = max(z_mask), min(z_mask)
        # y_max, y_min = max(y_mask), min(y_mask)
        # x_max, x_min = max(x_mask), min(x_mask)

        instance = image[:, z_min : z_max + 1, y_min : y_max + 1, x_min : x_max + 1]
        instance_mask = mask[:, z_min : z_max + 1, y_min : y_max + 1, x_min : x_max + 1]
        # instance_mask = torch.where(instance_mask == _label, instance_mask, 0)
        instance = instance * (instance_mask != 0).to(instance.dtype)
        instance = torch.where(instance_mask != 0, instance, 0)
        instance = _pad_tensor(instance, size=[size_z, size_y, size_x])
        instance_mask = _pad_tensor(instance_mask, size=[size_z, size_y, size_x])
        instance_list.append(instance)
        instance_mask_list.append(instance_mask)

    log.info(
        f"Finished extracting instances, took {time.time() - start} seconds ({len(unique_labels) / (time.time() - start)} instances/second)."
    )
    return torch.stack(instance_mask_list), torch.stack(instance_list)


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
    # Step 1: Crop to max size in z, y, x
    arr = arr[..., :size_z, :size_y, :size_x]

    # Step 2: Compute padding for z, y, x
    z, y, x = arr.shape[-3:]
    pad_z = size_z - z
    pad_y = size_y - y
    pad_x = size_x - x

    # Pad only at the end of each dimension
    pad_width = [(0, 0)] * (arr.ndim - 3) + [(0, pad_z), (0, pad_y), (0, pad_x)]
    arr = np.pad(arr, pad_width=pad_width, mode="constant", constant_values=0)

    return arr


def _pad_tensor(t: torch.Tensor, size: tuple[int, int, int]) -> torch.Tensor:
    """
    Resize a torch tensor to shape (size_z, size_y, size_x) or (..., size_z, size_y, size_x).

    - Crops if dimensions exceed target size.
    - Pads with zeros if smaller.

    Raises
    ------
    ValueError
        If the input tensor has ndim < 3.
    """
    if t.ndim < 3:
        raise ValueError("Tensor must have at least 3 dimensions (z, y, x).")

    size_z, size_y, size_x = size

    t = t[..., :size_z, :size_y, :size_x]

    z, y, x = t.shape[-3:]
    pad_z = size_z - z
    pad_y = size_y - y
    pad_x = size_x - x

    # Order for F.pad: (x_left, x_right, y_left, y_right, z_left, z_right, ...)
    pad = (0, pad_x, 0, pad_y, 0, pad_z)

    if any(p < 0 for p in pad):
        raise ValueError(f"Target size {size} is smaller than tensor shape {t.shape[-3:]}")

    t = F.pad(t, pad, mode="constant", value=0)
    return t
