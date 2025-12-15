from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import anndata as ad
import dask.array as da
import numpy as np
import pandas as pd
from loguru import logger as log
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from spatialdata import SpatialData

from harpy.image._image import _precondition
from harpy.table._table import ProcessTable, add_table_layer
from harpy.utils._featurize import Featurizer
from harpy.utils._keys import _CELL_INDEX, _INSTANCE_KEY, _REGION_KEY
from harpy.utils.utils import _dummy_embedding, _make_list


def extract_instances(
    sdata,
    img_layer: str,
    labels_layer: str,
    depth: int,
    diameter: int | None = None,
    remove_background: bool = True,
    extract_mask: bool = False,
    zarr_output_path: str | Path | None = None,
    batch_size: int | None = None,
    to_coordinate_system: str = "global",
) -> tuple[NDArray, da.Array]:
    """
    Extract per-label instance windows from `img_layer`/`labels_layer` of size `diameter` in `y` and `x` using :func:`dask.array.map_overlap` and :func:`dask.array.map_blocks`.

    For every non-zero label in the `labels_layer`, this method builds a Dask graph that
    slices out a centered, square window in the `y`, `x` plane around that instance (preserving
    the `z` dimension) both for the `img_layer` and `labels_layer`.

    Note that decreasing the chunk size on disk of the `image_layer` and `labels_layer` will lead to decreased
    consumption of RAM. A good first guess for chunk sizes is: `(c_chunksize, y_chunksize, x_chunksize)=(10, 2048, 2048)`.

    For optimal performance, configure :mod:`dask` to use `processes`, e.g. (`dask.config.set(scheduler="processes")`).

    Parameters
    ----------
    sdata
        SpatialData object.
    img_layer
        Name of the image layer.
    labels_layer
        Name of the labels layer.
    depth
        Passed to :func:`dask.array.map_overlap`.
        For correct results, choose depth to be roughly half of the estimated maximum diameter or larger.
    diameter
        Optional explicit side length of the resulting `y`, `x` window for every
        instance. If not provided `diameter` is set to 2 times `depth`.
    remove_background
        If `True`, pixels outside the instance label within each
        window are set to background (e.g., zero) so that only the object remains
        inside the cutout. If ``False``, the entire window content is kept.
    extract_mask
        If `True`, the corresponding mask (extracted from the `labels_layer`)
        will be added at channel index 0 for each extracted instance tensor.
    zarr_output_path
        If a filesystem path (string or ``Path``) is provided, the extracted
        instances are **computed** and materialized to a Zarr store at that
        location. The returned object will still be a Dask array pointing at the
        written data, but all computations necessary to populate the store will
        have been executed. If `None` (default), no data are written and the
        method returns a **lazy** (not yet computed) Dask array.
    batch_size
        Chunksize of the resulting dask array in the `i` dimension.
    to_coordinate_system
        The coordinate system that holds `img_layer` and `labels_layer`.

    Returns
    -------
    tuple:

        - a Numpy array containing indices of extracted labels, shape `(i,)`.
          Dimension of `i` will be equal to the total number of non-zero labels in the mask.

        - a Dask array of dimension `(i,c+1,z,y,x)` or `(i,c,z,y,x)`, with dimension of `c` the number of channels in `img_layer`.
          At channel index 0 of each instance, is the corresponding mask if `add_mask` is set to `True`.
          Dimension of `y` and `x` are equal to `diameter`, or 2*`depth` if `diameter` is not specified.

    Examples
    --------
    Extract instances directly from a SpatialData object:

    .. code-block:: python

        import harpy as hp
        import matplotlib.pyplot as plt

        sdata = hp.datasets.pixie_example()

        img_layer = "raw_image_fov0"
        labels_layer = "label_whole_fov0"

        # Persist to Zarr on disk (computes instances now)
        instance_ids, instances = hp.tb.extract_instances(
            sdata,
            img_layer=img_layer,
            labels_layer=labels_layer,
            depth=100,
            diameter=40,
            remove_background=True,
            extract_mask=False,
            zarr_output_path="instances.zarr",
            batch_size=64,
            to_coordinate_system="fov0",
        )

        mask_array = sdata[labels_layer].data[None, ...]

        instance_id = 23
        channel_idx = 20

        array = instances[instance_ids == instance_id][0][channel_idx][0]
        plt.imshow(array)
        plt.show()

    Or construct a lazy Dask graph:

    .. code-block:: python

        import harpy as hp
        import dask.array as da
        import matplotlib.pyplot as plt

        sdata = hp.datasets.pixie_example()

        img_layer = "raw_image_fov0"
        labels_layer = "label_whole_fov0"

        instance_ids, instances_lazy = hp.tb.extract_instances(
            sdata,
            img_layer=img_layer,
            labels_layer=labels_layer,
            depth=100,
            diameter=40,
            remove_background=True,
            extract_mask=False,
            zarr_output_path=None,
            batch_size=64,
            to_coordinate_system="fov0",
        )

        # compute instances now:
        instances_lazy.to_zarr( "instances.zarr" )
        instances = da.from_zarr( "instances.zarr" )
    """
    se_image, se_labels = _precondition(
        sdata, img_layer=img_layer, labels_layer=labels_layer, to_coordinate_system=to_coordinate_system
    )

    image_array = se_image.data
    mask_array = se_labels.data

    image_array = image_array[:, None, ...] if image_array.ndim == 3 else image_array
    mask_array = mask_array[None, ...] if mask_array.ndim == 2 else mask_array

    featurizer = Featurizer(mask_dask_array=mask_array, image_dask_array=image_array)

    instances_ids, instances = featurizer.extract_instances(
        depth=depth,
        diameter=diameter,
        remove_background=remove_background,
        extract_mask=extract_mask,
        zarr_output_path=zarr_output_path,
        store_intermediate=False,
        batch_size=batch_size,
    )

    return instances_ids, instances


def featurize(
    sdata: SpatialData,
    img_layer: str | list[str],
    labels_layer: str | list[str],
    table_layer: str | None,
    output_layer: str,
    depth: int,
    embedding_dimension: int,  # e.g. 384*matched_channels.shape[0]
    diameter: int | None = None,  # for kronos patch sizes must be multiples of 16
    remove_background: bool = True,
    model: Callable[..., NDArray] = _dummy_embedding,
    batch_size: int | None = None,
    model_kwargs: Mapping[str, Any] = MappingProxyType({}),
    embedding_obsm_key="embedding",
    to_coordinate_system: str | list[str] = "global",
    instance_key: str = _INSTANCE_KEY,
    region_key: str = _REGION_KEY,
    cell_index_name: str = _CELL_INDEX,
    dtype: np.dtype = np.float32,
    overwrite: bool = False,
    **kwargs: Any,
) -> SpatialData:
    """
    Extract per-instance feature vectors from `img_layer` and `labels_layer` using a user-provided embedding `model`.

    This method constructs a Dask graph that, for each non-zero label in `labels_layer`, extracts a
    centered `(y, x)` window (size set by `diameter` or `2 * depth`) from `img_layer`,
    optionally removes background pixels outside the labeled object (also in the corresponding `img_layer`),
    and feeds the resulting instance cutout (with preserved `z` and channel dimensions) through `model`
    to produce an embedding of size `embedding_dimension`.

    Internally, instance windows are generated lazily (via :func:`dask.array.map_overlap` and
    :func:`dask.array.map_blocks`) and then batched along the instance dimension to evaluate `model`
    in parallel. The output is a Dask array of shape `(i, d)`, where `i` is the number of
    non-zero labels and `d == embedding_dimension`.

    The resulting feature vectors are computed and added to
    `sdata[output_layer].obsm[embedding_obsm_key]` as a NumPy array.
    If `table_layer` is `None`, an empty table layer is created at `output_layer`.
    Otherwise, the feature vectors are sorted and filtered according to
    `sdata[table_layer].obs[_INSTANCE_KEY]`, and similarly added to
    `sdata[output_layer].obsm[embedding_obsm_key]`.

    For optimal performance, configure :mod:`dask` to use `processes`, e.g.:
    `dask.config.set(scheduler="processes")`.

    Note:
        Decreasing the chunk size of the provided image and mask arrays will reduce RAM usage.
        A good first guess for image/mask chunking is
        `(c_chunksize, y_chunksize, x_chunksize) = (10, 2048, 2048)`.

    Parameters
    ----------
    sdata
        SpatialData object.
    img_layer
        Name of the image layer.
    labels_layer
        Name of the labels layer.
    table_layer
        Name of the table layer. If `table_layer` is `None`, an empty `table_layer` will be created with
        the calculated embeddings at `.obsm[embedding_obsm_key]`, and annotated by `labels_layer`.
    output_layer
        Name of the output tables layer. Can be set equal to `table_layer` if overwrite is set to `True`.
    depth
        Passed to :func:`dask.array.map_overlap`.
        For correct results, choose depth to be roughly half of the estimated maximum diameter or larger.
    embedding_dimension
        The dimensionality `d` of the feature vectors returned by `model`. The returned Dask
        array will have shape `(i, embedding_dimension)`.
    diameter
        Optional explicit side length of the resulting `y`, `x` window for every
        instance. If not provided `diameter` is set to 2 times `depth`.
    remove_background
        If `True` (default), pixels outside the instance label within each window are set to
        background (e.g., zero) so that only the object remains inside the cutout. If `False`,
        the full window content is passed to `model`.
    model
        A callable that maps a batch of instance windows to embeddings:
        `(batch_size, c,z,y,x)->(batch_size, embedding_dimension)` , e.g.
        `model(batch, **model_kwargs) -> np.ndarray`.
        The callable should accept NumPy arrays; Dask will handle chunking and batching.
        The callable must include the parameter 'embedding_dimension'
    batch_size
        Chunk size of the resulting Dask array in the instance dimension `i` during model
        evaluation. Lower values can reduce (GPU) memory usage during model evaluation, but at the cost of more overhead (rechunking).
    model_kwargs
        Extra keyword arguments forwarded to `model` at call time (e.g., device selection,
        inference flags).
    embedding_obsm_key
        Name of the feature matrix added to `sdata[output_layer].obsm`.
    to_coordinate_system
        The coordinate system that holds `img_layer` and `labels_layer`.
    instance_key
        Instance key. The name of the column in `adata.obs` that will hold the instance ids.
        Ignored if `table_layer` is not None.
    region_key
        Region key. The name of the column in `adata.obs` that holds the name of the elements (`region`) that are annotated by the table layer.
        Ignored if `table_layer` is not None.
    cell_index_name
        The name of the index of the resulting :class:`~anndata.AnnData` table.
        Ignored if `table_layer` is not None.
    dtype
        Output dtype of `model`.
    overwrite
        If `True`, overwrites the `output_layer` if it already exists in `sdata`.
    **kwargs
        Additional keyword arguments forwarded to :func:`dask.array.map_blocks`. Use with care.

    Returns
    -------
    Spatialdata object.

    Examples
    --------
    Allocate intensity statistics and compute embeddings using a custom model:

    .. code-block:: python

        import numpy as np
        import harpy as hp

        sdata = hp.datasets.pixie_example()

        img_layer = "raw_image_fov0"
        labels_layer = "label_whole_fov0"

        # First, create an AnnData table by allocating intensity statistics
        # Note that this step is optional.
        sdata = hp.tb.allocate_intensity(
            sdata,
            img_layer=img_layer,
            labels_layer=labels_layer,
            to_coordinate_system="fov0",
            output_layer="my_table",
            mode="sum",
            obs_stats="count",  # cell size
            overwrite=True,
        )

        # Define a custom embedding model
        def my_model(
            batch,
            normalize: bool = True,
            embedding_dimension: int = 64,
        ) -> np.ndarray:
            # batch: (b, c, z, y, x) -> return (b, d)
            vecs = batch.reshape(batch.shape[0], -1).astype(np.float32)

            if normalize:
                norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
                vecs = vecs / norms

            # Project to desired embedding dimension (toy example)
            W = np.random.RandomState(0).randn(
                vecs.shape[1],
                embedding_dimension,
            ).astype(np.float32)

            return vecs @ W

        # Add embeddings to the table
        sdata = hp.tb.featurize(
            sdata,
            img_layer=img_layer,
            labels_layer=labels_layer,
            table_layer="my_table",
            output_layer="my_table",
            depth=96,
            embedding_dimension=64,
            diameter=192,
            model=my_model,
            model_kwargs={"normalize": True},
            batch_size=100,
            to_coordinate_system="fov0",
            embedding_obsm_key="embedding",
        )

        # Access the computed embedding for each instance
        sdata["my_table"].obsm["embedding"]
    """
    # if table_layer is None, we create an empty Anndata, that is annotated by labels layer
    # do it with dummy embedding first

    img_layer = _make_list(img_layer)
    labels_layer = _make_list(labels_layer)
    to_coordinate_system = _make_list(to_coordinate_system)

    instances_ids_list = []
    features_list = []

    for _img_layer, _labels_layer, _to_coordinate_system in zip(
        img_layer, labels_layer, to_coordinate_system, strict=True
    ):
        # currently this function will only work if img_layer and labels_layer have the same shape.
        # And are in same position, i.e. if one is translated, other should be translated with same offset

        se_image, se_labels = _precondition(
            sdata, img_layer=_img_layer, labels_layer=_labels_layer, to_coordinate_system=_to_coordinate_system
        )

        image_array = se_image.data
        mask_array = se_labels.data

        image_array = image_array[:, None, ...] if image_array.ndim == 3 else image_array
        mask_array = mask_array[None, ...] if mask_array.ndim == 2 else mask_array

        featurizer = Featurizer(mask_dask_array=mask_array, image_dask_array=image_array)

        instances_ids, features = featurizer.featurize(
            depth=depth,
            embedding_dimension=embedding_dimension,
            remove_background=remove_background,
            diameter=diameter,
            zarr_output_path=None,
            store_intermediate=False,
            model=model,
            batch_size=batch_size,
            model_kwargs=model_kwargs,
            dtype=dtype,
            **kwargs,
        )

        features = features.compute()
        # sanity check
        if features.shape[0] != instances_ids.shape[0]:
            raise RuntimeError(
                f"The first dimension of the feature matrix ({features.shape[0]}) "
                f"does not match with the number of instance ids {instances_ids.shape[0]}. Report this bug."
            )
        features_list.append(features)
        instances_ids_list.append(instances_ids)

    # now add the features to the table
    if table_layer is None:
        region_key = region_key
        instance_key = instance_key
        # create an anndata object with dummy count matrix.
        _region_keys = np.concatenate(
            [np.full(len(ids), label) for label, ids in zip(labels_layer, instances_ids_list, strict=True)]
        )
        instances_ids = np.concatenate(instances_ids_list, axis=0)

        index = [f"{id}_{region}" for id, region in zip(instances_ids, _region_keys, strict=True)]

        obs = pd.DataFrame({instance_key: instances_ids}, index=index)
        obs.index.name = cell_index_name
        # dummy count matrix
        count_matrix = csr_matrix((instances_ids.shape[0], 0))
        # create the anndata object
        adata = ad.AnnData(X=count_matrix, obs=obs)
        adata.obs[instance_key] = adata.obs[instance_key]

        adata.obs[region_key] = _region_keys
        adata.obs[region_key] = adata.obs[region_key].astype("category")
        # create an empty table, with all the layers in them i.e. set region key, instance key etc. instance keys are all unique ids in corresponding labels layer
    else:
        process_table_instance = ProcessTable(sdata, labels_layer=labels_layer, table_layer=table_layer)
        instance_key = process_table_instance.instance_key
        region_key = process_table_instance.region_key
        adata = process_table_instance._get_adata()

    for i, (_labels_layer, instances_ids, features) in enumerate(
        zip(labels_layer, instances_ids_list, features_list, strict=True)
    ):
        # get the instance_ids in adata, and sort features and instances_ids matrix in the same way.
        instances_ids_adata = adata[adata.obs[region_key] == _labels_layer].obs[instance_key].values
        _, features = _sort_features(
            instances_ids_adata=instances_ids_adata, instances_ids=instances_ids, features=features
        )
        features_list[i] = features
    adata.obsm[embedding_obsm_key] = np.concatenate(features_list, axis=0)

    sdata = add_table_layer(
        sdata,
        adata=adata,
        output_layer=output_layer,
        region=adata.obs[region_key].cat.categories.to_list(),  # equal to labels_layer
        instance_key=instance_key,
        region_key=region_key,
        overwrite=overwrite,
    )

    return sdata


def _sort_features(instances_ids_adata: NDArray, instances_ids: NDArray, features: NDArray):
    """Sort 'instances_ids' the way 'instances_ids_adata' is sorted, and sort 'features' similarly."""
    assert instances_ids_adata.ndim == 1
    assert instances_ids.ndim == 1
    assert features.ndim == 2
    assert features.shape[0] == instances_ids.shape[0]

    missing = np.setdiff1d(instances_ids_adata, instances_ids, assume_unique=True)
    if missing.size:
        raise ValueError(f"IDs in instances_ids_adata not found in instances_ids: {missing}")

    # Keep only IDs in instances_ids that are also in instances_id_adata (and mask the features matrix accordingly).
    mask = np.isin(instances_ids, instances_ids_adata, assume_unique=True)
    if ((~mask).sum()) != 0:
        log.info(
            "The following instance id's for which features where calculated, "
            f"are not in the AnnData object (and will therefore not be added to 'adata.obsm[embedding_key]'): {instances_ids[~mask]}. "
            "This is possible, if some instances were already filtered from the AnnData object prior to calling this function."
        )
        instances_ids = instances_ids[mask]
        features = features[mask]

    # sanity check
    if instances_ids.shape != instances_ids_adata.shape:
        raise RuntimeError(
            "After filtering, the number of instances in the AnnData object and the number of instances for which features were calculated differ. "
            "Please report this bug."
        )

    # Align/sort instances_ids (and features) to the order of instances_ids_adata
    # Build an indexer that tells, for each element of instances_ids, its position in instances_ids_adata
    sorter = np.argsort(instances_ids_adata)  # indices that would sort instances_ids_adata
    idx_in_instanes_ids_adata = sorter[np.searchsorted(instances_ids_adata, instances_ids, sorter=sorter)]

    instances_ids_sorted = np.empty_like(instances_ids)
    instances_ids_sorted[idx_in_instanes_ids_adata] = instances_ids
    features_sorted = np.empty_like(features)
    features_sorted[idx_in_instanes_ids_adata] = features

    # above 4 lines are equivalent to
    # order = np.argsort(idx_in_instanes_ids_adata)
    # instances_ids = instances_ids[order]
    # features = features[order]

    # sanity check, to see if we sorted correctly
    assert np.array_equal(instances_ids_adata, instances_ids_sorted)

    return instances_ids_sorted, features_sorted
