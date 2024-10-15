from __future__ import annotations

from typing import Iterable

import dask.array as da
import numpy as np
import pandas as pd
from anndata import AnnData
from dask.array import Array
from numpy.typing import NDArray
from spatialdata import SpatialData
from spatialdata.models.models import ScaleFactors_t
from spatialdata.transformations import get_transformation

from sparrow.image._image import _get_spatial_element, add_labels_layer
from sparrow.utils._keys import _INSTANCE_KEY, _REGION_KEY, _SPATIAL, ClusteringKey
from sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

try:
    import flowsom as fs

    from sparrow.utils._flowsom import _flowsom
except ImportError:
    log.warning("'flowsom' not installed, to use 'sp.im.flowsom', please install this library.")


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
        The image layer(s) of `sdata` on which flowsom is run. It is recommended to preprocess the data with `sp.im.pixel_clustering_preprocess`.
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
        Chunk sizes for processing. If provided as a tuple, it should contain chunk sizes for `c`, `(z)`, `y`, `x`.
    scale_factors
        Scale factors to apply for multiscale
    overwrite
        If True, overwrites the `output_layer_cluster` and/or `output_layer_metacluster` if it already exists in `sdata`.
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
    sparrow.im.pixel_clustering_preprocess : preprocess image layers before applying flowsom clustering.

    Warnings
    --------
    - The function is intended for use with spatial proteomics data. Input data should be appropriately preprocessed
      (e.g. via `sp.im.pixel_clustering_preprocess`) to ensure meaningful clustering results.
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

        if chunks is not None:
            arr = arr.rechunk(chunks)

        to_squeeze = False
        if arr.ndim == 3:
            # add trivial z dimension for 2D case
            arr = arr[:, None, ...]
            to_squeeze = True
        _arr_list.append(arr)

        # sample to train flowsom
        _arr_sampled = _sample_dask_array(
            _arr_list[i], fraction=fraction, remove_nan_columns=True, seed=random_state
        ).compute()
        results_arr_sampled.append(_arr_sampled)
        _region_keys.extend(_arr_sampled.shape[0] * [img_layer[i]])

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

    _, fsom = _flowsom(adata, n_clusters=n_clusters, seed=random_state, **kwargs)

    assert len(img_layer) == len(_arr_list)
    # 2) apply fsom on all data
    for i, _array in enumerate(_arr_list):
        output_chunks = ((2,), _array.chunks[1], _array.chunks[2], _array.chunks[3])

        # predict flowsom clusters
        _labels_flowsom = da.map_blocks(
            _predict_flowsom_clusters_chunk,
            _array,  # can also be chunked in c dimension, drop_axis and new_axis take care of this
            dtype=np.uint32,
            chunks=output_chunks,
            drop_axis=0,
            new_axis=0,
            fsom=fsom,
        )

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


def _sample_dask_array(array: Array, fraction: float = 0.1, remove_nan_columns: bool = True, seed: int = 0) -> Array:
    """Function to sample from dask array and flatten"""
    assert array.ndim == 4

    c, z, y, x = array.shape

    def _remove_nan_columns(array):
        # remove rows for which all values are NaN along the channels
        nan_mask = da.isnan(array[:, :-3]).all(axis=1)
        return array[~nan_mask]

    # Reshape the array to shape (z*y*x, c)
    reshaped_array = array.transpose(1, 2, 3, 0).reshape(-1, c)

    z_coords, y_coords, x_coords = da.meshgrid(da.arange(z), da.arange(y), da.arange(x), indexing="ij")
    coordinates = da.stack([z_coords.ravel(), y_coords.ravel(), x_coords.ravel()], axis=1)

    final_array = da.concatenate([reshaped_array, coordinates], axis=1)

    if fraction is None or fraction == 1:
        if remove_nan_columns:
            return _remove_nan_columns(final_array)
        else:
            return final_array

    num_samples = int(fraction * final_array.shape[0])

    rng = da.random.RandomState(seed)
    indices = rng.choice(
        final_array.shape[0],
        size=num_samples,
        replace=False,
    )

    if remove_nan_columns:
        return _remove_nan_columns(final_array[indices])
    else:
        return final_array[indices]
