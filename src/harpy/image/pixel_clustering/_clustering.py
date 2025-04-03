from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Iterable
from pathlib import Path

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from anndata import AnnData
from dask.array import Array
from dask.distributed import Client
from numpy.typing import NDArray
from spatialdata import SpatialData, read_zarr
from spatialdata.models import Image3DModel
from spatialdata.models.models import ScaleFactors_t
from spatialdata.transformations import get_transformation

from harpy.image._image import _get_spatial_element, add_labels_layer
from harpy.utils._keys import _INSTANCE_KEY, _REGION_KEY, _SPATIAL, ClusteringKey
from harpy.utils.pylogger import get_pylogger
from harpy.utils.utils import _get_uint_dtype

log = get_pylogger(__name__)

try:
    import flowsom as fs

    from harpy.utils._flowsom import _flowsom
except ImportError:
    log.warning(
        "'flowsom' not installed, to use 'harpy.im.flowsom', please install this library (https://git@github.com/saeyslab/FlowSOM_Python)."
    )


def flowsom(
    sdata: SpatialData,
    img_layer: str | Iterable[str],
    output_layer_clusters: str | Iterable[str],  # these are labels layers containing predicted clusters
    output_layer_metaclusters: str | Iterable[str],
    channels: int | str | Iterable[int] | Iterable[str] | None = None,
    fraction: float | None = 0.1,
    n_clusters: int = 5,
    random_state: int = 100,
    chunks: str | int | tuple[int, ...] | None = None,
    scale_factors: ScaleFactors_t | None = None,
    client: Client | None = None,
    persist_intermediate: bool = True,
    write_intermediate: bool = True,
    overwrite: bool = False,
    **kwargs,  # keyword arguments passed to _flowsom
) -> tuple[SpatialData, fs.FlowSOM, pd.Series]:
    """
    Applies flowsom clustering on image layer(s) of a SpatialData object.

    This function executes the flowsom clustering algorithm (via `fs.FlowSOM`) on spatial data encapsulated by a SpatialData object.
    The predited clusters and metaclusters are added as a labels layer to respectively `sdata.labels[output_layer_clusters]` and `sdata.labels[output_layer_metaclusters]`.

    Parameters
    ----------
    sdata
        The input SpatialData object.
    img_layer
        The image layer(s) of `sdata` on which flowsom is run. It is recommended to preprocess the data with :func:`harpy.im.pixel_clustering_preprocess`.
    output_layer_clusters
        The output labels layer in `sdata` to which labels layer with predicted flowsom SOM clusters are saved.
    output_layer_metaclusters
        The output labels layer in `sdata` to which labels layer with predicted flowsom metaclusters are saved.
    channels
        Specifies the channels to be included in the pixel clustering.
    fraction
        Fraction of the data to sample for training flowsom. Inference will be done on all pixels in `image_layer`.
    n_clusters
        The number of meta clusters to form.
    random_state
        A random state for reproducibility of the clustering and sampling.
    chunks
        Chunk sizes used for flowsom inference step on `img_layer`. If provided as a tuple, it should contain chunk sizes for `c`, `(z)`, `y`, `x`.
    scale_factors
        Scale factors to apply for multiscale
    client
        A Dask `Client` instance. If specified, during inference, the trained `fs.FlowSOM` model will be scattered (`client.scatter(...)`).
        This reduces the size of the task graph and can improve performance by minimizing data transfer overhead during computation.
        If not specified, Dask will use the default scheduler as configured on your system (e.g., single-threaded, multithreaded, or a global client if one is running).
    persist_intermediate
        If set to `True` will persit intermediate computation in memory. If `img_layer`, or one of the elements in `img_layer` is large, this could lead to increased ram usage.
        Set to `False` to write to intermediate zarr store instead, which will reduce ram usage, but will increase computation time slightly.
        We advice to set `persist_intermediate` to `True`, as it will only persist an array of dimension `(2,z,y,x)`, of dtype `numpy.uint8`.
        Ignored if `sdata` is not backed by a Zarr store.
    write_intermediate
        If set to `True`, an intermediate Zarr store will be used during sampling from `img_layer` for flowsom training.
        Enable this option to reduce RAM usage, especially if `img_layer` or any of its components is large.
        Ignored if `sdata` is not backed by a Zarr store.
    overwrite
        If `True`, overwrites the `output_layer_cluster` and/or `output_layer_metacluster` if it already exists in `sdata`.
    **kwargs
        Additional keyword arguments passed to `fs.FlowSOM`.

    Returns
    -------
    tuple:

        - The input `sdata` with the clustering results added.

        - FlowSOM object containing a `MuData` object and a trained `fs.models.FlowSOMEstimator`. `MuData` object will only contain the fraction (via the `fraction` parameter) of the data sampled from the `img_layer` on which the FlowSOM model is trained.

        - A pandas Series object containing a mapping between the clusters and the metaclusters.

    See Also
    --------
    harpy.im.pixel_clustering_preprocess : preprocess image layers before applying flowsom clustering.

    Warnings
    --------
    - The function is intended for use with spatial proteomics data. Input data should be appropriately preprocessed
      (e.g. via :func:`harpy.im.pixel_clustering_preprocess`) to ensure meaningful clustering results.
    - The cluster and metacluster ID's found in `output_layer_clusters` and `output_layer_metaclusters` count from 1, while they count from 0 in the `FlowSOM` object.
    """
    assert 0 < fraction <= 1, "Value must be between 0 and 1"

    def _fix_name(layer: str | Iterable[str]):
        return list(layer) if isinstance(layer, Iterable) and not isinstance(layer, str) else [layer]

    img_layer = _fix_name(img_layer)
    output_layer_clusters = _fix_name(output_layer_clusters)
    output_layer_metaclusters = _fix_name(output_layer_metaclusters)

    assert (
        len(output_layer_clusters) == len(output_layer_metaclusters) == len(img_layer)
    ), "The number of 'output_layer_clusters' and 'output_layer_metaclusters' specified should be the equal to the the number of 'img_layer' specified."

    se_image = _get_spatial_element(sdata, layer=img_layer[0])

    if channels is not None:
        channels = _fix_name(channels)
    else:
        if channels is None:
            channels = se_image.c.data

    # 1) Train flowsom on a sample
    results_arr_sampled = []
    _arr_list = []
    _transformations = []
    _region_keys = []
    log.info("Extracting random sample for FlowSOM training.")
    for i, _img_layer in enumerate(img_layer):
        se_image = _get_spatial_element(sdata, layer=_img_layer)
        _transformations.append(get_transformation(se_image, get_all=True))
        arr = se_image.sel(c=channels).data
        if i == 0:
            _array_dim = arr.ndim
        else:
            assert (
                _array_dim == arr.ndim
            ), "Image layers specified via parameter `img_layer` should all have same number of dimensions."

        to_squeeze = False
        if arr.ndim == 3:
            # add trivial z dimension for 2D case
            arr = arr[:, None, ...]
            to_squeeze = True
        _arr_list.append(arr)

        if sdata.is_backed() and write_intermediate:
            _temp_path = os.path.join(os.path.dirname(sdata.path), f"tmp_{uuid.uuid4()}")
        else:
            _temp_path = None

        # sample to train flowsom
        _arr_sampled = _sample_dask_array(
            _arr_list[i], fraction=fraction, remove_nan_columns=True, seed=random_state, temp_path=_temp_path
        )
        results_arr_sampled.append(_arr_sampled)
        _region_keys.extend(_arr_sampled.shape[0] * [img_layer[i]])

        # clean up
        if sdata.is_backed() and write_intermediate:
            shutil.rmtree(_temp_path)

    arr_sampled = np.row_stack(results_arr_sampled)

    # create anndata object
    var = pd.DataFrame(index=channels)
    var.index.name = "channels"
    var.index = var.index.map(str)

    adata = AnnData(X=arr_sampled[:, :-3], var=var)

    adata.obs[_INSTANCE_KEY] = np.arange(arr_sampled.shape[0])
    adata.obs[_REGION_KEY] = pd.Categorical(_region_keys)

    # add coordinates to anndata
    if to_squeeze:
        # 2D case, only save y,x position
        adata.obsm[_SPATIAL] = arr_sampled[:, -2:]
    else:
        # 3D case, save z,y,x position
        adata.obsm[_SPATIAL] = arr_sampled[:, -3:]

    xdim = kwargs.pop("xdim", 10)
    ydim = kwargs.pop("ydim", 10)
    dtype = _get_uint_dtype(value=xdim * ydim)
    log.info("Start FlowSOM training.")
    _, fsom = _flowsom(adata, n_clusters=n_clusters, seed=random_state, xdim=xdim, ydim=ydim, **kwargs)
    log.info("Finished FlowSOM training. Starting inference ")

    if client is not None:
        fsom_future = client.scatter(fsom)
    else:
        fsom_future = fsom

    assert len(img_layer) == len(_arr_list)
    # 2) apply fsom on all data
    for i, _array in enumerate(_arr_list):
        if chunks is not None:
            if to_squeeze:
                # fix chunks to account for fact that we added trivial z-dimension
                if isinstance(chunks, Iterable) and not isinstance(chunks, str):
                    chunks = (chunks[0], 1, chunks[1], chunks[2])
            _array = _array.rechunk(chunks)
        output_chunks = ((2,), _array.chunks[1], _array.chunks[2], _array.chunks[3])

        # predict flowsom clusters
        _labels_flowsom = da.map_blocks(
            _predict_flowsom_clusters_chunk,
            _array,  # can also be chunked in c dimension, drop_axis and new_axis take care of this
            dtype=dtype,
            chunks=output_chunks,
            drop_axis=0,
            new_axis=0,
            fsom=fsom_future,
        )

        # write to intermediate zarr slot or persist, otherwise dask will run the flowsom inference two times (once for clusters, once for metaclusters),
        # once for each time we call add_labels_layer.
        if sdata.is_backed() and not persist_intermediate:
            se_intermediate = Image3DModel.parse(_labels_flowsom)
            _labels_flowsom_name = f"labels_flowsom_{uuid.uuid4()}"
            sdata.images[_labels_flowsom_name] = se_intermediate
            sdata.write_element(_labels_flowsom_name)
            del sdata[_labels_flowsom_name]
            sdata_temp = read_zarr(sdata.path, selection=["images"])
            sdata[_labels_flowsom_name] = sdata_temp[_labels_flowsom_name]
            del sdata_temp
            _labels_flowsom = _get_spatial_element(sdata, layer=_labels_flowsom_name).data
        else:
            _labels_flowsom = _labels_flowsom.persist()

        _labels_flowsom_clusters, _labels_flowsom_metaclusters = _labels_flowsom

        # save the predicted clusters and metaclusters as a labels layer
        sdata = add_labels_layer(
            sdata,
            arr=_labels_flowsom_clusters.squeeze(0) if to_squeeze else _labels_flowsom_clusters,
            output_layer=output_layer_clusters[i],
            transformations=_transformations[i],
            scale_factors=scale_factors,
            overwrite=overwrite,
        )

        sdata = add_labels_layer(
            sdata,
            arr=_labels_flowsom_metaclusters.squeeze(0) if to_squeeze else _labels_flowsom_metaclusters,
            output_layer=output_layer_metaclusters[i],
            transformations=_transformations[i],
            scale_factors=scale_factors,
            overwrite=overwrite,
        )

        if sdata.is_backed() and not persist_intermediate:
            del sdata[_labels_flowsom_name]
            sdata.delete_element_from_disk(element_name=_labels_flowsom_name)

    # TODO decide on fix in flowsom to let clusters count from 1.
    # fsom cluster ID's count from 0, while labels layer cluster ID's count from 1.

    mapping = fsom.get_cluster_data().obs[ClusteringKey._METACLUSTERING_KEY.value].copy()
    # +1 because flowsom SOM clusters count from 0,
    mapping.index = mapping.index.astype(int) + 1
    mapping += 1

    return sdata, fsom, mapping


def _predict_flowsom_clusters_chunk(array: NDArray, fsom: fs.FlowSOM) -> NDArray:
    def _remove_nan_columns(array):
        # remove rows that contain NaN's, flowsom can not work with NaN's
        nan_mask = np.isnan(array[:, :-3]).any(axis=1)
        return array[~nan_mask]

    def _flatten_array(array: NDArray) -> NDArray:
        c, z, y, x = array.shape

        # Reshape the array to shape (z*y*x, c)
        reshaped_array = array.transpose(1, 2, 3, 0).reshape(-1, c)

        z_coords, y_coords, x_coords = np.meshgrid(np.arange(z), np.arange(y), np.arange(x), indexing="ij")
        coordinates = np.stack([z_coords.ravel(), y_coords.ravel(), x_coords.ravel()], axis=1)

        final_array = np.concatenate([reshaped_array, coordinates], axis=1)

        return _remove_nan_columns(final_array)

    assert array.ndim == 4

    final_array = _flatten_array(array)

    coordinates = final_array[:, -3:].astype(int)
    values = final_array[:, :-3]

    y_clusters = fsom.model.cluster_model.predict(values)
    y_codes = fsom.model._y_codes

    # do not do the latter, because it could lead to race condition
    # y_metaclusters=fsom.model.predict(values )
    # y_clusters=fsom.model.cluster_labels_

    # add +1 because we want labels to count from 1
    clusters_array = np.full(array.shape[1:], 0, dtype=np.uint32)
    clusters_array[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]] = y_clusters.astype(int) + 1

    meta_clusters_array = np.full(array.shape[1:], 0, dtype=np.uint32)
    meta_clusters_array[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]] = y_codes[y_clusters].astype(int) + 1

    return np.stack([clusters_array, meta_clusters_array], axis=0)


def _sample_dask_array(
    array: Array,
    fraction: float = 0.1,
    remove_nan_columns: bool = True,
    seed: int = 0,
    temp_path: str | Path | None = None,
) -> NDArray:
    """Function to sample from dask array and flatten"""
    assert array.ndim == 4

    c, z, y, x = array.shape

    def _remove_nan_columns(array):
        # remove rows for which all values are NaN along the channels
        nan_mask = da.isnan(array[:, :-3]).all(axis=1)
        return array[~nan_mask]

    # Reshape the array to shape (z*y*x, c)
    reshaped_array = array.transpose(1, 2, 3, 0).reshape(-1, c)

    z_coords, y_coords, x_coords = da.meshgrid(
        da.arange(z, dtype=np.uint32),
        da.arange(y, dtype=np.uint32),
        da.arange(x, dtype=np.uint32),
        indexing="ij",
    )

    coordinates = da.stack([z_coords.ravel(), y_coords.ravel(), x_coords.ravel()], axis=1)
    coordinates = coordinates.rechunk(reshaped_array.chunksize)

    # if we do not write to intermediate slot, unmanaged memory can become high (meshgrid needs to be pulled in memory)
    # causing pausing/termination of the workers.
    if temp_path is not None:
        coordinates.to_zarr(os.path.join(temp_path, "coordinates.zarr"))
        coordinates = da.from_zarr(os.path.join(temp_path, "coordinates.zarr"))

    final_array = da.concatenate([reshaped_array, coordinates], axis=1)

    if fraction is None or fraction == 1:
        if remove_nan_columns:
            return _remove_nan_columns(final_array.compute())
        else:
            return final_array.compute()

    # write to intermediate slot, to reduce unmanaged memory
    if temp_path is not None:
        final_array.rechunk(final_array.chunksize).to_zarr(os.path.join(temp_path, "final_array.zarr"))
        final_array = da.from_zarr(os.path.join(temp_path, "final_array.zarr"))

    log.info("Start sampling")
    sample = dd.from_array(final_array).sample(frac=fraction, replace=False, random_state=seed).values

    if remove_nan_columns:
        sample = _remove_nan_columns(sample)

    sample = sample.compute()

    # clean up
    if temp_path is not None:
        shutil.rmtree(os.path.join(temp_path, "coordinates.zarr"))
        shutil.rmtree(os.path.join(temp_path, "final_array.zarr"))

    return sample
