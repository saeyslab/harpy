from __future__ import annotations

import uuid
from typing import Iterable

import dask.array as da
import numpy as np
import pandas as pd
from anndata import AnnData
from dask import delayed
from numpy.typing import NDArray
from spatialdata import SpatialData

from sparrow.image._image import _get_spatial_element, _get_transformation
from sparrow.table._preprocess import preprocess_proteomics
from sparrow.table._table import _add_table_layer
from sparrow.utils._keys import _CELL_INDEX, _INSTANCE_KEY, _REGION_KEY


def cell_clustering_preprocess(
    sdata: SpatialData,
    labels_layer_cells: str | Iterable[str],
    labels_layer_clusters: str | Iterable[str],
    output_layer: str,
    q: float | None = 0.999,
    chunks: str | int | tuple[int, ...] | None = None,
    overwrite: bool = False,
) -> SpatialData:
    """
    Preprocesses spatial data for cell clustering.

    This function prepares a SpatialData object for cell clustering by integrating cell segmentation masks (obtained via e.g. `sp.im.segment`) and SOM pixel/meta cluster (obtained via e.g. `sp.im.flosom`).
    The function calculates the cluster count (clusters provided via `labels_layer_clusters`) for each cell in `labels_layer_cells`, normalized by cell size, and optionally by quantile normalization if `q` is provided.
    The results are stored in a specified table layer within the `sdata` object of shape (#cells, #clusters).

    Parameters
    ----------
    sdata
        The input SpatialData object containing the spatial proteomics data.
    labels_layer_cells
        The labels layer(s) in `sdata` that contain cell segmentation masks. These masks should be previously generated using `sp.im.segment`.
    labels_layer_clusters
        The labels layer(s) in `sdata` that contain metacluster or cluster masks. These should be derived from `sp.im.flowsom`.
    output_layer
        The name of the table layer within `sdata` where the preprocessed data will be stored.
    q
        Quantile used for normalization. If specified, each pixel SOM/meta cluster column in `output_layer` is normalized by this quantile. Values are multiplied by 100 after normalization.
    chunks
        Chunk sizes for processing the data. If provided as a tuple, it should detail chunk sizes for each dimension `(z)`, `y`, `x`.
    overwrite
        If True, overwrites the existing data in the specified `output_layer` if it already exists.

    Returns
    -------
    The input `sdata` with a table layer added (`output_layer`).

    See Also
    --------
    sparrow.im.flowsom : flowsom pixel clustering.
    sparrow.tb.flowsom : flowsom cell clustering.
    """
    labels_layer_cells = (
        list(labels_layer_cells)
        if isinstance(labels_layer_cells, Iterable) and not isinstance(labels_layer_cells, str)
        else [labels_layer_cells]
    )
    labels_layer_clusters = (
        list(labels_layer_clusters)
        if isinstance(labels_layer_clusters, Iterable) and not isinstance(labels_layer_clusters, str)
        else [labels_layer_clusters]
    )

    assert (
        len(labels_layer_cells) == len(labels_layer_clusters)
    ), "The number of 'labels_layer_cells' specified should be the equal to the the number of 'labels_layer_clusters' specified."

    # first get total number of unique labels and total number of unique cluster id's over all FOV's
    _arr_list_labels = []
    _arr_list_clusters = []
    for i, (_labels_layer_cells, _labels_layer_clusters) in enumerate(zip(labels_layer_cells, labels_layer_clusters)):
        se_labels = _get_spatial_element(sdata, layer=_labels_layer_cells)
        se_clusters = _get_spatial_element(sdata, layer=_labels_layer_clusters)

        assert (
            se_labels.shape == se_clusters.shape
        ), f"Provided labels layers '{_labels_layer_cells}' and '{_labels_layer_clusters}' do not have the same shape."

        assert (
            _get_transformation(se_labels) == _get_transformation(se_clusters)
        ), f"Transformation on provided labels layers '{_labels_layer_cells}' and '{_labels_layer_clusters}' are not equal. This is currently not supported."

        if i == 0:
            _array_dim = se_labels.ndim
        else:
            assert (
                _array_dim == se_labels.ndim == se_clusters.ndim
            ), "Labels layer specified in 'labels_layer_cells' and 'labels_layer_cluster' should all have same number of dimensions."

        _array_labels = se_labels.data
        _array_clusters = se_clusters.data

        if chunks is not None:
            _array_labels = _array_labels.rechunk(chunks)
            _array_clusters = _array_clusters.rechunk(chunks)

        if _array_labels.ndim == 2:
            # add trivial z dimension for 2D case
            _array_labels = _array_labels[None, ...]
            _array_clusters = _array_clusters[None, ...]
        _arr_list_labels.append(_array_labels)
        _arr_list_clusters.append(_array_clusters)

    # should map on the same clusters, because predicted via same flowsom model,
    # but _labels_layer_clusters of one FOV could contain cluster ID's that is not in other _labels_layer_clusters correponding to other FOV, therefore get all cluster ID's across all FOVs
    _unique_clusters = da.unique(da.hstack([da.unique(_arr) for _arr in _arr_list_clusters])).compute()

    _results_sum_of_chunks = []
    _cells_id = []
    _region_keys = []
    for i in range(len(_arr_list_labels)):
        _array_labels = _arr_list_labels[i]
        _array_clusters = _arr_list_clusters[i]

        assert (
            _array_labels.numblocks == _array_clusters.numblocks
        ), f"Provided labels layers '{labels_layer_cells[i]}' and '{labels_layer_clusters[i]}' have different chunk sizes. Set 'chunk' parameter to fix this issue."

        _unique_mask = da.unique(_array_labels).compute()

        chunk_sum = da.map_blocks(
            lambda m, f, **kw: _cell_cluster_count(m, f, **kw),
            _array_labels,
            _array_clusters,
            dtype=np.int32,
            chunks=(1, len(_unique_mask), len(_unique_clusters)),
            unique_mask=_unique_mask,
            unique_clusters=_unique_clusters,
        )

        sum_of_chunks = np.zeros((len(_unique_mask), len(_unique_clusters)), dtype=int)

        num_chunks = chunk_sum.numblocks
        tasks = []

        # sum the result for each chunk
        for _i in range(num_chunks[0]):
            for _j in range(num_chunks[1]):
                for _k in range(num_chunks[2]):
                    current_chunk = chunk_sum.blocks[_i, _j, _k]
                    task = delayed(np.add)(sum_of_chunks, current_chunk)
                    tasks.append(task)

        total_sum_delayed = delayed(sum)(tasks)

        sum_of_chunks = total_sum_delayed.compute()
        _cells_id.append(_unique_mask.reshape(-1, 1))
        _results_sum_of_chunks.append(sum_of_chunks)
        _region_keys.extend(_unique_mask.shape[0] * [labels_layer_cells[i]])

    sum_of_chunks = np.row_stack(_results_sum_of_chunks)
    _cells_id = np.row_stack(_cells_id)

    var = pd.DataFrame(index=_unique_clusters)
    var.index = var.index.map(str)
    var.index.name = "pixel_cluster_id"

    cells = pd.DataFrame(index=_cells_id.squeeze(1))
    _uuid_value = str(uuid.uuid4())[:8]

    cells.index = [f"{idx}_{_region_keys[i]}_{_uuid_value}" for i, idx in enumerate(cells.index)]
    cells.index.name = _CELL_INDEX

    adata = AnnData(X=sum_of_chunks, obs=cells, var=var)

    adata.obs[_INSTANCE_KEY] = _cells_id.astype(int)
    adata.obs[_REGION_KEY] = pd.Categorical(_region_keys)

    # remove count for background (i.e. cluster id=='0' and label ==0)
    adata = adata[adata.obs[_INSTANCE_KEY] != 0].copy()
    adata = adata[:, ~adata.var_names.isin(["0"])].copy()

    # remove cells with no overlap with any pixel cluster
    adata = adata[~(adata.X == 0).all(axis=1)].copy()

    sdata = _add_table_layer(
        sdata,
        adata=adata,
        output_layer=output_layer,
        region=labels_layer_cells,
        overwrite=overwrite,
    )

    # for size normalization of counts by size of the labels; and quantile normalization
    sdata = preprocess_proteomics(
        sdata,
        labels_layer=labels_layer_cells,
        table_layer=output_layer,
        output_layer=output_layer,
        size_norm=True,
        log1p=False,
        scale=False,
        q=q,
        calculate_pca=False,
        overwrite=True,
    )

    return sdata


def _cell_cluster_count(
    mask_block: NDArray, cluster_block: NDArray, unique_mask: NDArray, unique_clusters: NDArray
) -> NDArray:
    result_array = np.zeros(
        (len(unique_mask), len(unique_clusters)), dtype=int
    )  # this output shape will be same for every chunk
    unique_mask_block = np.unique(mask_block)
    # Populate the result array
    for mask_id in unique_mask_block:
        mask_indices = mask_block == mask_id
        clusters_in_mask = cluster_block[mask_indices]
        unique_clusters_mask_id, counts = np.unique(clusters_in_mask, return_counts=True)

        mask_loc = np.searchsorted(unique_mask, mask_id)
        clusters_loc = np.searchsorted(unique_clusters, unique_clusters_mask_id)

        result_array[mask_loc, clusters_loc] = counts

    return result_array
