from __future__ import annotations

import uuid
from collections.abc import Iterable

import dask.array as da
import numpy as np
import pandas as pd
from anndata import AnnData
from loguru import logger as log
from numpy.typing import NDArray
from spatialdata import SpatialData
from spatialdata.transformations import get_transformation

from harpy.image._image import _get_spatial_element
from harpy.table._preprocess import preprocess_proteomics
from harpy.table._table import add_table
from harpy.utils._keys import _CELL_INDEX, _CELLSIZE_KEY, _INSTANCE_KEY, _RAW_COUNTS_KEY, _REGION_KEY


def cell_clustering_preprocess(
    sdata: SpatialData,
    cells_labels_name: str | Iterable[str],
    cluster_labels_name: str | Iterable[str],
    output_table_name: str,
    q: float | None = 0.999,
    chunks: str | int | tuple[int, ...] | None = None,
    region_key: str = _REGION_KEY,
    instance_key: str = _INSTANCE_KEY,
    cell_index_name: str = _CELL_INDEX,
    instance_size_key: str = _CELLSIZE_KEY,
    raw_counts_key: str = _RAW_COUNTS_KEY,
    overwrite: bool = False,
) -> SpatialData:
    """
    Preprocesses spatial data for cell clustering.

    This function prepares a SpatialData object for cell clustering by integrating cell segmentation masks (obtained via e.g. `harpy.im.segment`) and SOM pixel/meta cluster (obtained via e.g. `harpy.im.flosom`).
    The function calculates the cluster count (clusters provided via `cluster_labels_name`) for each cell in `cells_labels_name`, normalized by cell size, and optionally by quantile normalization if `q` is provided.
    The results are stored in a specified table layer within the `sdata` object of shape (#cells, #clusters).

    Parameters
    ----------
    sdata
        The input SpatialData object containing the spatial proteomics data.
    cells_labels_name
        The labels layer(s) in `sdata` that contain cell segmentation masks. These masks should be previously generated using `harpy.im.segment`.
    cluster_labels_name
        The labels layer(s) in `sdata` that contain metacluster or cluster masks. These should be derived from `harpy.im.flowsom`.
    output_table_name
        The name of the table layer within `sdata` where the preprocessed data will be stored.
    q
        Quantile used for normalization. If specified, each pixel SOM/meta cluster column in `output_table_name` is normalized by this quantile. Values are multiplied by 100 after normalization.
    chunks
        Chunk sizes for processing the data. If provided as a tuple, it should detail chunk sizes for each dimension `(z)`, `y`, `x`.
    instance_key
        Instance key. The name of the column in :class:`~anndata.AnnData` table `.obs` that will hold the instance ids.
    region_key
        Region key. The name of the column in  :class:`~anndata.AnnData` table `.obs` that will hold the name of the element(s) that are annotated by the resulting table.
    cell_index_name
        The name of the index of the resulting :class:`~anndata.AnnData` table.
    instance_size_key
        The key in the :class:`~anndata.AnnData` table `.obs` that will hold the size of the instances (obtained from `cells_labels_name`).
    raw_counts_key
        Name of the :class:`~anndata.AnnData` layer where the non-preprocessed counts will be stored.
    overwrite
        If True, overwrites the existing data in the specified `output_table_name` if it already exists.

    Returns
    -------
    The input `sdata` with a table layer added (`output_table_name`).

    See Also
    --------
    harpy.im.flowsom : flowsom pixel clustering.
    harpy.tb.flowsom : flowsom cell clustering.
    """
    cells_labels_name = (
        list(cells_labels_name)
        if isinstance(cells_labels_name, Iterable) and not isinstance(cells_labels_name, str)
        else [cells_labels_name]
    )
    cluster_labels_name = (
        list(cluster_labels_name)
        if isinstance(cluster_labels_name, Iterable) and not isinstance(cluster_labels_name, str)
        else [cluster_labels_name]
    )

    assert len(cells_labels_name) == len(cluster_labels_name), (
        "The number of 'cells_labels_name' specified should be the equal to the the number of 'cluster_labels_name' specified."
    )

    # first get total number of unique labels and total number of unique cluster id's over all FOV's
    _arr_list_labels = []
    _arr_list_clusters = []
    for i, (_cells_labels_name, _cluster_labels_name) in enumerate(
        zip(cells_labels_name, cluster_labels_name, strict=True)
    ):
        se_labels = _get_spatial_element(sdata, element_name=_cells_labels_name)
        se_clusters = _get_spatial_element(sdata, element_name=_cluster_labels_name)

        assert se_labels.shape == se_clusters.shape, (
            f"Provided labels layers '{_cells_labels_name}' and '{_cluster_labels_name}' do not have the same shape."
        )

        assert get_transformation(se_labels, get_all=True) == get_transformation(se_clusters, get_all=True), (
            f"Transformation on provided labels layers '{_cells_labels_name}' and '{_cluster_labels_name}' are not equal. This is currently not supported."
        )

        if i == 0:
            _array_dim = se_labels.ndim
        else:
            assert _array_dim == se_labels.ndim == se_clusters.ndim, (
                "Labels layer specified in 'cells_labels_name' and 'cluster_labels_name' should all have same number of dimensions."
            )

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
    # but _cluster_labels_name of one FOV could contain cluster IDs that are not in other
    # _cluster_labels_name values corresponding to other FOVs, therefore get all cluster IDs across all FOVs
    _unique_clusters = da.unique(da.hstack([da.unique(_arr) for _arr in _arr_list_clusters])).compute()

    _results_sum_of_chunks = []
    _cells_id = []
    _region_keys = []
    for i in range(len(_arr_list_labels)):
        _array_labels = _arr_list_labels[i]
        _array_clusters = _arr_list_clusters[i]

        assert _array_labels.numblocks == _array_clusters.numblocks, (
            f"Provided labels layers '{cells_labels_name[i]}' and '{cluster_labels_name[i]}' have different chunk sizes. Set 'chunk' parameter to fix this issue."
        )

        _unique_mask = da.unique(_array_labels).compute()

        chunk_sum = da.map_blocks(
            lambda m, f, **kw: _cell_cluster_count(m, f, **kw),
            _array_labels,
            _array_clusters,
            dtype=_array_labels.dtype,
            chunks=(len(_unique_mask), len(_unique_clusters)),
            drop_axis=0,
            unique_mask=_unique_mask,
            unique_clusters=_unique_clusters,
        )

        dask_chunks = [
            da.from_delayed(_chunk, shape=(len(_unique_mask), len(_unique_clusters)), dtype=_array_labels.dtype)
            for _chunk in chunk_sum.to_delayed().flatten()
        ]

        dask_array = da.stack(dask_chunks, axis=0)
        sum_of_chunks = da.sum(dask_array, axis=0).compute()

        _cells_id.append(_unique_mask.reshape(-1, 1))
        _results_sum_of_chunks.append(sum_of_chunks)
        _region_keys.extend(_unique_mask.shape[0] * [cells_labels_name[i]])

    sum_of_chunks = np.vstack(_results_sum_of_chunks)
    _cells_id = np.vstack(_cells_id)

    var = pd.DataFrame(index=_unique_clusters)
    var.index = var.index.map(str)
    var.index.name = "pixel_cluster_id"

    cells = pd.DataFrame(index=_cells_id.squeeze(1))
    _uuid_value = str(uuid.uuid4())[:8]

    cells.index = [f"{idx}_{_region_keys[i]}_{_uuid_value}" for i, idx in enumerate(cells.index)]
    cells.index.name = cell_index_name

    adata = AnnData(X=sum_of_chunks, obs=cells, var=var)

    adata.obs[instance_key] = _cells_id.astype(int)
    adata.obs[region_key] = pd.Categorical(_region_keys)

    # remove count for background (i.e. cluster id=='0' and label ==0)
    adata = adata[adata.obs[instance_key] != 0].copy()
    adata = adata[:, ~adata.var_names.isin(["0"])].copy()

    # remove cells with no overlap with any pixel cluster
    no_overlap_mask = np.asarray((adata.X == 0).all(axis=1)).ravel()
    if no_overlap_mask.any():
        removed_cells = adata.obs.loc[no_overlap_mask, [instance_key, region_key]].copy()
        removed_cells[instance_key] = removed_cells[instance_key].astype(int)
        log.info(
            f"Removing {int(no_overlap_mask.sum())} cells with no overlap with any pixel cluster from table '{output_table_name}'."
        )

        removed_per_region = removed_cells.groupby(region_key, observed=True)[instance_key].apply(list)
        for _region, _instance_ids in removed_per_region.items():
            if len(_instance_ids) <= 50:
                _instance_ids_str = ", ".join(str(_id) for _id in _instance_ids)
            else:
                _instance_ids_str = (
                    ", ".join(str(_id) for _id in _instance_ids[:50]) + f", ... ({len(_instance_ids) - 50} more)"
                )
            log.info(
                f"Removed {len(_instance_ids)} no-overlap cells for region '{_region}' (instance ids: [{_instance_ids_str}])."
            )
    adata = adata[~no_overlap_mask].copy()

    sdata = add_table(
        sdata,
        adata=adata,
        output_table_name=output_table_name,
        region=cells_labels_name,
        instance_key=instance_key,
        region_key=region_key,
        overwrite=overwrite,
    )

    # for size normalization of counts by size of the labels; and quantile normalization
    sdata = preprocess_proteomics(
        sdata,
        labels_name=cells_labels_name,
        table_name=output_table_name,
        output_table_name=output_table_name,
        size_norm=True,
        log1p=False,
        scale=False,
        q=q,
        calculate_pca=False,
        instance_size_key=instance_size_key,
        raw_counts_key=raw_counts_key,
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
