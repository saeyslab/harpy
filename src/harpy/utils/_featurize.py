from __future__ import annotations

import time

import dask.array as da
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from scipy import ndimage

from harpy.image.segmentation._utils import _add_depth_to_chunks_size, _rechunk_overlap
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

    def extract_instances(
        self,
        depth: int,  # ~diameter/2, depth in y and x
    ) -> da.Array:
        _depth = {0: 0, 1: 0, 2: depth, 3: depth}

        array_mask = self._mask[None, ...]  # add trivial channel dimension
        array_image = self._image

        if array_image.numblocks[1] != 1:
            raise ValueError("Currently we do not allow chunking in z dimension.")

        array_mask = _tranpose_chunks(
            array_mask, depth=_depth
        )  # TODO best make option to save these as a temp zarr store to optimize the dask graph
        array_image = _tranpose_chunks(array_image, depth=_depth)

        # probably not even necessary to do this count_custom block, dask seems to help himself well if these are not exactly specified in featurize custom_block call below
        counts = da.map_blocks(
            _count_custom_block,
            array_mask,
            chunks=(
                (1,),  # trivial c dimension
                (1,) * len(array_mask.chunks[1]),
                (1,) * len(array_mask.chunks[2]),
                (1,) * len(array_mask.chunks[3]),
            ),  # e.g. ((1,),(1, 1, 1,), (1, 1,),),
            dtype=array_mask.dtype,
            _depth=_depth,
            index=self._labels[self._labels != 0],
        )

        log.info("Calculating instances per chunk. This could take a few minutes for large images.")
        counts = counts.compute().flatten()

        instances_not_assigned = len(self._labels[self._labels != 0]) - sum(counts)
        if instances_not_assigned != 0:
            log.info(
                f"The number of total labels in the mask differs from the number of labels counted per chunk "
                f"(difference is {instances_not_assigned}). Consider increasing depth."
            )

        arrays = [array_mask, array_image]

        c_chunks = array_image.chunks[0]
        c_chunks = tuple([c_chunks[0] + 1] + list(c_chunks[1:]))  # we concat mask to first c channel chunk
        # returns c,z,i,y,x tensor
        dask_chunks = da.map_blocks(
            lambda *arrays, block_info=None, **kw: _featurize_custom_block(*arrays, block_info=block_info, **kw),
            *arrays,
            dtype=np.float32,  # images and mask will be cast to dtype, if dtype==np.float32, max label supported is 2**24. # TODO cast to float64 if max label>2**24
            chunks=(
                c_chunks,  # e.g. (3+1,1) # do allow chunking in c.
                array_image.chunks[1],
                tuple(counts.tolist()),
                (_depth[2],),
                (_depth[3],),
            ),
            new_axis=4,
            _depth=_depth,
            index=self._labels[self._labels != 0],
        )
        # make it i,c,z,y,x
        dask_chunks = dask_chunks.transpose(2, 0, 1, 3, 4)

        return dask_chunks

    def mean(
        self,
        diameter: int,  # estimated max diameter of cell in y, x
        device: str = "cpu",
    ) -> pd.DataFrame:
        fn_kwargs = {
            "size": (self._image.shape[1], diameter, diameter),
            "device": device,
        }

        return self.featurize(
            depth=diameter * 2,
            features=2,
            dtype=np.float32,
            fn=_stats,
            fn_kwargs=fn_kwargs,
        )

        # df = pd.DataFrame(_result)

        # df[_INSTANCE_KEY] = self._labels[self._labels != 0]

        # return df


def _stats(mask: NDArray, image: NDArray, size: tuple[int, int, int], device="cpu") -> NDArray:
    instances_mask, instances = _extract_instances(
        mask, image, size=size, device="cpu"
    )  # extracting instances on cpu is fastest
    # can be any complex function here
    instances_mask = instances_mask.to(device)
    instances = instances.to(device)
    mean = _masked_mean(instances, instances_mask, dim=[2, 3, 4])
    return np.stack([mean.cpu().numpy(), mean.cpu().numpy()])
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
    concat_mask: bool = True,
) -> NDArray:
    start = time.time()
    log.info("Extracting instances (NumPy).")

    # --- validations (kept from your original) ---
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

    # --- foreground coords once (O(V)) ---
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
    cz, zz, yy, xx = np.nonzero(fg)
    labels = mask[fg]  # shape (N,)

    # sanity check
    assert 0 not in labels

    uniq, inv = np.unique(labels, return_inverse=True)

    L = uniq.size

    # per-label bbox via segment reductions (O(N))
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

        # crop
        inst_img = image[:, zs:ze, ys:ye, xs:xe]  # .copy() # no copy needed, because we use np.where later
        inst_mask = mask[:, zs:ze, ys:ye, xs:xe]  # .copy()

        # keep only this label in the mask, zero everywhere else
        lbl = uniq[i]
        inst_mask = np.where(inst_mask == lbl, inst_mask, 0)
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

    arr = arr[..., :size_z, :size_y, :size_x]

    z, y, x = arr.shape[-3:]

    pad_z = size_z - z
    pad_y = size_y - y
    pad_x = size_x - x

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


def _featurize_custom_block(
    *arrays,
    index: NDArray,
    block_info,
    _depth,
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
            _depth[2],
            _depth[3],
        ),  # no chunking in z dimension, so size in z is image_block.shape[1]
        concat_mask=concat_mask,
    )

    # return c,z,i,y,x tensor
    return instances.transpose(1, 2, 0, 3, 4)


def _count_custom_block(
    mask_block: NDArray,
    index: NDArray,
    _depth,
) -> NDArray:
    assert mask_block.ndim == 4
    assert 0 not in index

    _, unique_masks_inside = _mask_center_of_mass_outside(mask_block=mask_block.copy(), _depth=_depth)

    return np.array([len(unique_masks_inside)]).reshape(-1, 1, 1, 1)


def _mask_center_of_mass_outside(mask_block: NDArray, _depth):
    assert len(_depth) == 4
    assert mask_block.ndim == 4
    # Set all masks that are fully outside the region to zero, they will be covered by other chunks
    # (NOTE: theoretically possible that mask fully outside chunk, but center of mass inside chunks
    # to solve, we should not construct the mask below, and we should consider all masks from 0:2DX and -2DX:X
    # for performance we ignore this edge case)
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

    unique_masks = unique_masks[unique_masks != 0]

    mask = ~np.isin(mask_block, unique_masks)
    mask_block[mask] = 0

    return mask_block, unique_masks


def _tranpose_chunks(array: da.Array, depth: dict[int, int]):
    def return_block(block):
        # dummy function, to do a map_overlap
        return block

    array = _rechunk_overlap(array, depth=depth, chunks=array.chunks)
    output_chunks = _add_depth_to_chunks_size(array.chunks, depth)

    # map an overlap
    array = da.map_overlap(
        array,
        return_block,
        depth=depth,
        chunks=output_chunks,
        allow_rechunk=False,
        dtype=array.dtype,
        trim=False,
        boundary=0,
    )
    _, _, chunksize_y, chunksize_x = array.chunksize

    y_rest = chunksize_y - array.chunks[2][-1]
    x_rest = chunksize_x - array.chunks[3][-1]

    array = da.pad(array, ((0, 0), (0, 0), (0, y_rest), (0, x_rest))).rechunk(array.chunksize)

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


def _count_custom_block_fast(
    *arrays,
    index,
    _depth,
):
    mask_block = arrays[0]
    if len(arrays) == 2:
        image_block = arrays[1]
        assert mask_block.shape == image_block.shape
    if len(arrays) > 2:
        raise ValueError("Only accepts one or two arrays.")
    assert 0 not in index

    D0, D1, D2 = _depth
    Z, Y, X = mask_block.shape

    # Build border "overlap region" once, then unique
    slabs = [
        mask_block[:, -(D1 * 2) : -D1, D2:-D2],  # near +Y border (inner overlap band)
        mask_block[:, D1:-D1, -(D2 * 2) : -D2],  # near +X border
        mask_block[:, D1 : D1 * 2, D2:-D2],  # near -Y border
        mask_block[:, D1:-D1, D2 : D2 * 2],  # near -X border
    ]
    border_labels = np.unique(np.concatenate([s.ravel() for s in slabs]))
    border_labels = border_labels[border_labels != 0]  # drop background early

    # If no border labels, just count uniques in the inner subset and return
    subset = mask_block[:, D1 : Y - D1, D2 : X - D2]
    unique_masks = np.unique(subset)
    if border_labels.size == 0:
        unique_masks = unique_masks[unique_masks != 0]
        return np.array([unique_masks.size]).reshape(-1, 1, 1)

    # ----- compute centroids for border labels (no SciPy) -----
    # Select voxels that belong to border labels
    # (np.isin is vectorized in C; much faster than Python loops)
    sel = np.isin(mask_block, border_labels)
    if not np.any(sel):
        # Fallback: no selected voxels (degenerate), just count uniques in subset
        unique_masks = unique_masks[unique_masks != 0]
        return np.array([unique_masks.size]).reshape(-1, 1, 1)

    # Coordinates of selected voxels
    zz, yy, xx = np.nonzero(sel)

    """
    labs = mask_block[sel]

    # Map labs -> [0..k-1] for bincounts
    lbls_sorted = np.sort(border_labels)
    # searchsorted requires labels to be in the sorted list
    # positions give compact indices for bincount
    idx = np.searchsorted(lbls_sorted, labs)

    counts = np.bincount(idx)
    sum_y = np.bincount(idx, weights=yy.astype(np.float64))
    sum_x = np.bincount(idx, weights=xx.astype(np.float64))

    cy = sum_y / counts
    cx = sum_x / counts
    # We don't actually need cz for the inside test, but compute the same way if you do:
    # sum_z = np.bincount(idx, weights=zz.astype(np.float64))
    # cz = sum_z / counts

    # vectorized inside-original check (Y,X only)
    inside_y = (cy >= D1) & (cy < (Y - D1))
    inside_x = (cx >= D2) & (cx < (X - D2))
    inside = inside_y & inside_x
    """
    labs = mask_block[sel]  # labels present in selection
    labs_unique, inv = np.unique(labs, return_inverse=True)
    counts = np.bincount(inv)
    sum_y = np.bincount(inv, weights=yy.astype(np.float64))
    sum_x = np.bincount(inv, weights=xx.astype(np.float64))
    cy = sum_y / counts
    cx = sum_x / counts

    inside = (cy >= D1) & (cy < (Y - D1)) & (cx >= D2) & (cx < (X - D2))
    labels_inside = labs_unique[inside]

    # labels to exclude = border labels that are NOT inside
    border_labels_not_to_consider = border_labels[~np.isin(border_labels, labels_inside)]

    # Labels whose COM is OUTSIDE the original block (to be excluded)
    # border_labels_not_to_consider = lbls_sorted[~inside]

    # Unique masks that intersect the inner subset
    unique_masks = unique_masks[unique_masks != 0]
    if border_labels_not_to_consider.size:
        # Remove those to be ignored
        unique_masks = unique_masks[~np.isin(unique_masks, border_labels_not_to_consider)]

    return np.array([unique_masks.size]).reshape(-1, 1, 1)


def _extract_instances_slow(
    mask: NDArray,
    image: NDArray,
    size: tuple[int, int, int] = (1, 100, 100),
    concat_mask: bool = True,
) -> tuple[NDArray, NDArray]:
    # returns two tensors of shape i,c,z,y,x
    start = time.time()
    log.info("Extracting instances.")
    # rewrite this in torch
    assert mask.ndim == image.ndim == 4
    assert len(size) == 3

    if np.max(mask) > 2**24:
        raise ValueError(f"Maximum value allowed in mask is ({2**24}).")
    if not np.issubdtype(image.dtype, np.floating):
        if np.max(image) > 2**24:
            raise ValueError(f"Cannot savely cast image to float32. Please clip the image to values <={2**24}")
    else:  # case where it is floating -> we only support np.float32
        if not np.issubdtype(image.dtype, np.float32):
            raise ValueError("Currently only images of dtype int and float32 are supported.")

    # image = torch.from_numpy(image).to(torch.float32).to(device)

    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels != 0]

    instance_list = []
    instance_mask_list = []

    total_inner = 0
    for _label in unique_labels:
        if _label == 0:
            continue
        size_z, size_y, size_x = size  # ~ diameter of the cell ~2*depth ( 2* depth+1 )
        # _, z_mask, y_mask, x_mask = torch.where(mask == _label)
        start_inner = time.time()
        vox_idx = (mask == _label).nonzero()  # this one makes it slow
        total_inner += time.time() - start_inner
        z_min = int(vox_idx[1].min().item())
        z_max = int(vox_idx[1].max().item())
        y_min = int(vox_idx[2].min().item())
        y_max = int(vox_idx[2].max().item())
        x_min = int(vox_idx[3].min().item())
        x_max = int(vox_idx[3].max().item())

        instance = image[:, z_min : z_max + 1, y_min : y_max + 1, x_min : x_max + 1]
        instance_mask = mask[:, z_min : z_max + 1, y_min : y_max + 1, x_min : x_max + 1]

        instance_mask = instance_mask * (instance_mask == _label)  # put to zero everywhere there is no feature

        # instance_mask = torch.where(instance_mask == _label, instance_mask, 0)
        # instance = instance * (instance_mask != 0)
        # put to zero everywhere there is no feature
        instance = np.where(instance_mask != 0, instance, 0)
        instance = _pad_array(instance, size=[size_z, size_y, size_x])
        instance_mask = _pad_array(instance_mask, size=[size_z, size_y, size_x])
        instance_list.append(instance)
        instance_mask_list.append(instance_mask)

    log.info(
        f"Finished extracting instances, took {time.time() - start} seconds ({len(unique_labels) / (time.time() - start)} instances/second)."
    )

    mask = np.stack(instance_mask_list).astype(np.float32)
    image = np.stack(instance_list).astype(np.float32)

    print(total_inner)

    if concat_mask:
        return np.concatenate([mask, image], axis=1)
    else:
        return image


def _extract_instances_torch(
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

        instance_mask = instance_mask * (instance_mask == _label)  # put to zero everywhere there is no _label

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
