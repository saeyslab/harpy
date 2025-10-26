from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from spatialdata import SpatialData

from harpy.image._image import _get_spatial_element
from harpy.table._table import ProcessTable, add_table_layer
from harpy.utils._featurize import Featurizer
from harpy.utils._keys import _CELL_INDEX, _INSTANCE_KEY, _REGION_KEY
from harpy.utils.pylogger import get_pylogger
from harpy.utils.utils import _dummy_embedding, _make_list

log = get_pylogger(__name__)


def extract_instances(sdata):
    """Extract instances."""
    # sometimes you want to extract instances.
    # place holder, TODO, implement
    instances_ids = None
    instances = None
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
    overwrite: bool = False,
    **kwargs: Any,
) -> SpatialData:
    """Featurize"""
    # if table_layer is None, we create an empty Anndata, that is annotated by labels layer
    # do it with dummy embedding first

    img_layer = _make_list(img_layer)
    labels_layer = _make_list(labels_layer)

    instances_ids_list = []
    features_list = []

    for _img_layer, _labels_layer in zip(img_layer, labels_layer, strict=True):
        # TODO: raise valueError if translation defined on them that does not match
        image_array = _get_spatial_element(sdata, layer=_img_layer).data
        mask_array = _get_spatial_element(sdata, layer=_labels_layer).data

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
            batch_size=batch_size,  # 1m10  # 6m 60 MB max, chunksize 2048 # 4mins if you take markers_to_keep
            model_kwargs=model_kwargs,
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
        # create an anndata object with dummy count matrix.
        _region_keys = np.concatenate(
            [np.full(len(ids), label) for label, ids in zip(labels_layer, instances_ids_list, strict=True)]
        )
        instances_ids = np.concatenate(instances_ids_list, axis=0)

        index = [f"{id}_{region}" for id, region in zip(instances_ids, _region_keys, strict=True)]

        obs = pd.DataFrame({_INSTANCE_KEY: instances_ids}, index=index)
        obs.index.name = _CELL_INDEX
        # dummy count matrix
        count_matrix = csr_matrix((instances_ids.shape[0], 0))
        # create the anndata object
        adata = ad.AnnData(X=count_matrix, obs=obs)
        adata.obs[_INSTANCE_KEY] = adata.obs[_INSTANCE_KEY]

        adata.obs[_REGION_KEY] = _region_keys
        adata.obs[_REGION_KEY] = adata.obs[_REGION_KEY].astype("category")
        # create an empty table, with all the layers in them i.e. set region key, instance key etc. instance keys are all unique ids in corresponding labels layer
    else:
        process_table_instance = ProcessTable(sdata, labels_layer=labels_layer, table_layer=table_layer)
        adata = process_table_instance._get_adata()

    for i, (_labels_layer, instances_ids, features) in enumerate(
        zip(labels_layer, instances_ids_list, features_list, strict=True)
    ):
        # get the instance_ids in adata, and sort features and instances_ids matrix in the same way.
        instances_ids_adata = adata[adata.obs[_REGION_KEY] == _labels_layer].obs[_INSTANCE_KEY].values
        _, features = _sort_features(
            instances_ids_adata=instances_ids_adata, instances_ids=instances_ids, features=features
        )
        features_list[i] = features
    adata.obsm[embedding_obsm_key] = np.concatenate(features_list, axis=0)

    sdata = add_table_layer(
        sdata,
        adata=adata,
        output_layer=output_layer,
        region=adata.obs[_REGION_KEY].cat.categories.to_list(),  # equal to labels_layer
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
