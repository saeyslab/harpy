from __future__ import annotations

import warnings
from copy import deepcopy

import dask
import dask.array as da
import numpy as np
from dask.array import Array
from dask.array.overlap import ensure_minimum_chunksize
from dask_image.ndmeasure._utils import _label
from numpy.typing import NDArray
from sklearn import metrics as sk_metrics

from harpy.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

_SEG_DTYPE = np.uint32


def _rechunk_overlap(
    x: Array,
    depth: dict[int, int],
    chunks: str | int | tuple[int, ...] | None = None,
) -> Array:
    # rechunk, so that we ensure minimum overlap

    assert len(depth) == x.ndim, (
        f"Please provide depth value for every dimension of x ({x.ndim}). Provided depth was '{depth}'"
    )

    if chunks is not None:
        x = x.rechunk(chunks)

    # rechunk if new chunks are needed to fit depth in every chunk,
    # this allows us to send allow_rechunk=False with map_overlap,
    # and have control of chunk sizes of input dask array and output dask array

    for i in range(len(depth)):
        if depth[i] != 0:
            if depth[i] > x.chunksize[i]:
                log.warning(
                    f"Depth for dimension {i} exceeds chunk size. Adjusting to a quarter of chunk size: {x.chunksize[i] / 4}"
                )
                depth[i] = int(x.chunksize[i] // 4)

    new_chunks = tuple(ensure_minimum_chunksize(size + 1, c) for size, c in zip(depth.values(), x.chunks, strict=True))

    x = x.rechunk(new_chunks)  # this is a no-op if x.chunks == new_chunks

    return x


def _clean_up_masks(
    block: NDArray,
    block_id: tuple[int, int, int],
    block_info,
    depth: dict[int, int],
) -> NDArray:
    total_blocks = block_info[0]["num-chunks"]
    assert total_blocks[0] == 1, (
        "Dask arrays chunked in z dimension are not supported. Please only chunk in y and x dimensions."
    )
    total_blocks = total_blocks[1:]
    assert depth[0] == 0, "Depth not equal to 0 in z dimension is currently not supported."
    assert len(depth) == 3, "Please provide depth values for z,y and x."
    depth = deepcopy(depth)
    total_blocks = deepcopy(total_blocks)

    # remove z-dimension from depth
    depth[0] = depth[1]
    depth[1] = depth[2]
    del depth[2]

    # get the 'inside' region of the block, i.e. the original chunk without depth appended
    y_start, y_stop = depth[0], block.shape[1] - depth[0]
    x_start, x_stop = depth[1], block.shape[2] - depth[1]

    assert block_id[0] == 0, (
        "Dask arrays chunked in z dimension are not supported. Please only chunk in y and x dimensions."
    )
    block_id = block_id[1:]

    # get indices of all adjacent blocks
    adjacent_blocks = _get_ajdacent_block_ids(block_id, total_blocks)

    # get all masks id's that cross the boundary of original chunk (without depth appended)
    # masks that are on the boundary of the larger array (e.g. y==depth[0] axis are skipped)

    crossing_masks = set()

    if block_id[0] != 0:
        crossing_masks.update(np.unique(block[:, depth[0], :]))
    if block_id[1] != 0:
        crossing_masks.update(np.unique(block[:, :, depth[1]]))
    if block_id[0] != total_blocks[0] - 1:
        crossing_masks.update(np.unique(block[:, block.shape[1] - depth[0], :]))
    if block_id[1] != total_blocks[1] - 1:
        crossing_masks.update(np.unique(block[:, :, block.shape[2] - depth[1]]))

    def calculate_area(crd, mask_position):
        return np.sum(
            (crd[0] <= mask_position[0])
            & (mask_position[0] < crd[1])
            & (crd[2] <= mask_position[1])
            & (mask_position[1] < crd[3])
        )

    masks_to_reset = []
    for mask_label in crossing_masks:
        if mask_label == 0:
            continue
        mask_position = np.where(block == mask_label)
        # not interested in which z-slice these mask_positions are
        mask_position = mask_position[1:]

        inside_region = calculate_area((y_start, y_stop, x_start, x_stop), mask_position)

        for adjacent_block_id in adjacent_blocks:
            crd = _calculate_boundary_adjacent_block(block.shape[1:], depth, block_id, adjacent_block_id)

            outside_region = calculate_area(crd, mask_position)

            # if intersection with mask and region outside chunk is bigger than inside region, set values of chunk to 0 for this masks.
            # For edge case where inside region and outside region is the same, it will be assigned to both chunks.
            # Note that is better that both chunks claim the masks, than that no chunks are claiming the mask.
            if outside_region > inside_region:
                masks_to_reset.append(mask_label)
                break

    mask = np.isin(block, masks_to_reset)
    block[mask] = 0

    # Set all masks that are fully outside the region to zero, they will be covered by other chunks
    subset = block[:, depth[0] : block.shape[1] - depth[0], depth[1] : block.shape[2] - depth[1]]
    # Unique masks gives you all masks that are in 'original' array (i.e. without depth added)
    unique_masks = np.unique(subset)

    # Create a mask for labels that are NOT in the subset
    mask = ~np.isin(block, unique_masks)
    block[mask] = 0

    return block


def _merge_masks(
    array: NDArray,
    _depth: dict[int, int],
    num_blocks: tuple[int, int, int],
    block_id: tuple[int, int, int],
) -> NDArray:
    # helper function to merge the chunks

    assert num_blocks[0] == 1, (
        "Dask arrays chunked in z dimension are not supported. Please only chunk in y and x dimensions."
    )

    assert _depth[0] == 0, "Depth not equal to 0 in z dimension is currently not supported."
    assert len(_depth) == 3, "Please provide depth values for z,y and x."

    new_array = array[:, _depth[1] * 2 : -_depth[1] * 2, _depth[2] * 2 : -_depth[2] * 2]
    # y,x
    # upper ( y, x+1 )
    if block_id[2] + 1 != num_blocks[2]:
        overlap = array[:, _depth[1] * 2 : -_depth[1] * 2, -_depth[2] :]
        sliced_new_array = new_array[
            :,
            :,
            -_depth[2] :,
        ]
        non_zero_mask = (sliced_new_array == 0) & (overlap != 0)
        new_array[
            :,
            :,
            -_depth[2] :,
        ] = np.where(
            non_zero_mask,
            overlap,
            sliced_new_array,
        )
    # upper right ( y+1, x+1 )
    if block_id[1] + 1 != num_blocks[1] and block_id[2] + 1 != num_blocks[2]:
        overlap = array[:, -_depth[1] :, -_depth[2] :]
        sliced_new_array = new_array[
            :,
            -_depth[1] :,
            -_depth[2] :,
        ]
        non_zero_mask = (sliced_new_array == 0) & (overlap != 0)
        new_array[
            :,
            -_depth[1] :,
            -_depth[2] :,
        ] = np.where(
            non_zero_mask,
            overlap,
            sliced_new_array,
        )
    # right ( y+1, x )
    if block_id[1] + 1 != num_blocks[1]:
        overlap = array[:, -_depth[1] :, _depth[2] * 2 : -_depth[2] * 2]
        sliced_new_array = new_array[:, -_depth[1] :, :]
        non_zero_mask = (sliced_new_array == 0) & (overlap != 0)
        new_array[:, -_depth[1] :, :] = np.where(non_zero_mask, overlap, sliced_new_array)
    # under right ( y+1, x-1 )
    if block_id[1] + 1 != num_blocks[1] and block_id[2] != 0:
        overlap = array[:, -_depth[1] :, : _depth[2]]
        sliced_new_array = new_array[
            :,
            -_depth[1] :,
            : _depth[2],
        ]
        non_zero_mask = (sliced_new_array == 0) & (overlap != 0)
        new_array[
            :,
            -_depth[1] :,
            : _depth[2],
        ] = np.where(
            non_zero_mask,
            overlap,
            sliced_new_array,
        )
    # lower ( y, x-1 )
    if block_id[2] != 0:
        overlap = array[:, _depth[1] * 2 : -_depth[1] * 2, : _depth[2]]
        sliced_new_array = new_array[
            :,
            :,
            : _depth[2],
        ]
        non_zero_mask = (sliced_new_array == 0) & (overlap != 0)
        new_array[
            :,
            :,
            : _depth[2],
        ] = np.where(
            non_zero_mask,
            overlap,
            sliced_new_array,
        )
    # lower left ( y-1, x-1 )
    if block_id[1] != 0 and block_id[2] != 0:
        overlap = array[:, : _depth[1], : _depth[2]]
        sliced_new_array = new_array[
            :,
            : _depth[1],
            : _depth[2],
        ]
        non_zero_mask = (sliced_new_array == 0) & (overlap != 0)
        new_array[
            :,
            : _depth[1],
            : _depth[2],
        ] = np.where(
            non_zero_mask,
            overlap,
            sliced_new_array,
        )
    #  left ( y-1, x )
    if block_id[1] != 0:
        overlap = array[:, : _depth[1], _depth[2] * 2 : -_depth[2] * 2]
        sliced_new_array = new_array[
            :,
            : _depth[1],
            :,
        ]
        non_zero_mask = (sliced_new_array == 0) & (overlap != 0)
        new_array[
            :,
            : _depth[1],
            :,
        ] = np.where(non_zero_mask, overlap, sliced_new_array)
    # upper left ( y-1, x+1 )
    if block_id[1] != 0 and block_id[2] + 1 != num_blocks[2]:
        overlap = array[:, : _depth[1], -_depth[2] :]
        sliced_new_array = new_array[
            :,
            : _depth[1],
            -_depth[2] :,
        ]
        non_zero_mask = (sliced_new_array == 0) & (overlap != 0)
        new_array[
            :,
            : _depth[1],
            -_depth[2] :,
        ] = np.where(
            non_zero_mask,
            overlap,
            sliced_new_array,
        )

    return new_array


def _check_boundary(boundary: str) -> None:
    valid_boundaries = ["reflect", "periodic", "nearest"]

    if boundary not in valid_boundaries:
        raise ValueError(f"'{boundary}' is not a valid boundary. It must be one of {valid_boundaries}.")


def _add_depth_to_chunks_size(chunks: tuple[tuple[int, ...], ...], depth: dict[int, int, int]):
    result = []
    for i, item in enumerate(chunks):
        if i in depth:  # check if there's a corresponding depth value
            result.append(tuple(x + depth[i] * 2 for x in item))
        else:
            result.append(item)
    return tuple(result)


def _substract_depth_from_chunks_size(chunks: tuple[tuple[int, ...], ...], depth: dict[int, int]):
    result = []
    for i, item in enumerate(chunks):
        if i in depth:  # check if there's a corresponding depth value
            result.append(tuple(x - depth[i] * 2 for x in item))
        else:
            result.append(item)
    return tuple(result)


def _get_ajdacent_block_ids(block_id, total_blocks):
    y, x = block_id
    max_y, max_x = total_blocks

    potential_neighbors = [
        (y - 1, x - 1),  # top-left
        (y, x - 1),  # top
        (y + 1, x - 1),  # top-right
        (y - 1, x),  # left
        (y + 1, x),  # right
        (y - 1, x + 1),  # bottom-left
        (y, x + 1),  # bottom
        (y + 1, x + 1),  # bottom-right
    ]

    # Filter out neighbors that have negative IDs or exceed the total number of blocks
    neighbors = [neighbor for neighbor in potential_neighbors if 0 <= neighbor[0] < max_y and 0 <= neighbor[1] < max_x]
    return neighbors


def _calculate_boundary_adjacent_block(chunk_shape, depth, block_id, adjacent_block_id):
    if adjacent_block_id[0] > block_id[0]:
        y_start = chunk_shape[0] - depth[0]
        y_stop = chunk_shape[0]
    elif adjacent_block_id[0] == block_id[0]:
        y_start = depth[0]
        y_stop = chunk_shape[0] - depth[0]
    else:
        y_start = 0
        y_stop = depth[0]

    if adjacent_block_id[1] > block_id[1]:
        x_start = chunk_shape[1] - depth[1]
        x_stop = chunk_shape[1]
    elif adjacent_block_id[1] == block_id[1]:
        x_start = depth[1]
        x_stop = chunk_shape[1] - depth[1]
    else:
        x_start = 0
        x_stop = depth[1]

    return (y_start, y_stop, x_start, x_stop)


def _get_block_position(
    chunks: tuple[tuple[int, ...], ...], block_id: tuple[int, int, int]
) -> tuple[int, int, int, int]:
    """
    Given a block structure of a 4D Dask array and a block ID, return the start and stop positions in the full array for that block for the 1st (y) and 2nd (x) dimension.

    Parameters
    ----------
    block_structure: A tuple of tuples, where each inner tuple
                            represents the sizes of the blocks in that dimension.
    block_id: A tuple representing the position of the block in the
                     block structure.

    Returns
    -------
    A tuple (y_start, y_stop, x_start, x_stop)
    """
    y_structure, x_structure = chunks[1], chunks[2]
    _, i, j, _ = block_id

    y_start = sum(y_structure[:i])
    y_stop = y_start + y_structure[i]

    x_start = sum(x_structure[:j])
    x_stop = x_start + x_structure[j]

    return y_start, y_stop, x_start, x_stop


def _link_labels(block_labeled, total, depth, iou_threshold=1):
    """
    Link labels

    Build a label connectivity graph that groups labels across blocks,
    use this graph to find connected components, and then relabel each
    block according to those.
    """
    label_groups = _label_adjacency_graph(block_labeled, total, depth, iou_threshold)
    new_labeling = _label.connected_components_delayed(label_groups)
    return _label.relabel_blocks(block_labeled, new_labeling)


def _across_block_iou_delayed(face, axis, iou_threshold):
    """Delayed version of :func:`_across_block_label_grouping`."""
    _across_block_label_grouping_ = dask.delayed(_across_block_label_iou)
    grouped = _across_block_label_grouping_(face, axis, iou_threshold)
    return da.from_delayed(grouped, shape=(2, np.nan), dtype=np.int32)


def _across_block_label_iou(face, axis, iou_threshold):
    unique = np.unique(face)
    face0, face1 = np.split(face, 2, axis)

    warnings.filterwarnings("ignore", message="A single label was found in 'y_true' and 'y_pred'")
    intersection = sk_metrics.confusion_matrix(face0.reshape(-1), face1.reshape(-1), labels=unique)
    sum0 = intersection.sum(axis=0, keepdims=True)
    sum1 = intersection.sum(axis=1, keepdims=True)

    # Note that sum0 and sum1 broadcast to square matrix size.
    union = sum0 + sum1 - intersection

    # Ignore errors with divide by zero, which the np.where sets to zero.
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(intersection > 0, intersection / union, 0)

    labels0, labels1 = np.nonzero(iou >= iou_threshold)

    labels0_orig = unique[labels0]
    labels1_orig = unique[labels1]
    grouped = np.stack([labels0_orig, labels1_orig])

    valid = np.all(grouped != 0, axis=0)  # Discard any mappings with bg pixels
    return grouped[:, valid]


def get_slices_and_axes(chunks, shape, depth):
    ndim = len(shape)
    depth = da.overlap.coerce_depth(ndim, depth)
    slices = da.core.slices_from_chunks(chunks)
    slices_and_axes = []
    for ax in range(ndim):
        for sl in slices:
            if sl[ax].stop == shape[ax]:
                continue
            slice_to_append = list(sl)
            slice_to_append[ax] = slice(sl[ax].stop - 2 * depth[ax], sl[ax].stop + 2 * depth[ax])
            slices_and_axes.append((tuple(slice_to_append), ax))
    return slices_and_axes


def _label_adjacency_graph(labels, nlabels, depth, iou_threshold):
    all_mappings = [da.empty((2, 0), dtype=np.int32, chunks=1)]

    slices_and_axes = get_slices_and_axes(labels.chunks, labels.shape, depth)
    for face_slice, axis in slices_and_axes:
        face = labels[face_slice]
        mapped = _across_block_iou_delayed(face, axis, iou_threshold)
        all_mappings.append(mapped)

    i, j = da.concatenate(all_mappings, axis=1)

    result = _label._to_csr_matrix(i, j, nlabels + 1)
    return result
