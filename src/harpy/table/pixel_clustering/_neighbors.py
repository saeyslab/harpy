from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData
from numpy.typing import NDArray

from harpy.utils._keys import _SPATIAL
from harpy.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

try:
    import squidpy as sq
except ImportError:
    log.warning("'squidpy' not installed, to use 'harpy.tb.spatial_pixel_neighbors' please install this library.")


def spatial_pixel_neighbors(
    sdata,
    labels_layer: str,
    size: int = 20,
    subset: list[int] | None = None,
    spatial_neighbors_kwargs: Mapping[str, Any] = MappingProxyType({}),
    nhood_enrichment_kwargs: Mapping[str, Any] = MappingProxyType({}),
    seed: int = 0,
    key_added: str = "cluster_id",
) -> AnnData:
    """
    Computes spatial pixel neighbors and performs neighborhood enrichment analysis.

    This function extracts grid-based cluster labels from the specified labels layer of a SpatialData object,
    subdivides the spatial domain into a grid of a given size, and computes spatial neighbors along with
    neighborhood enrichment statistics. The resulting AnnData object stores the cluster labels as a categorical
    observation (under the key provided by `key_added`) and the corresponding spatial coordinates in its `.obsm`
    attribute. `squidpy` is used for the spatial neighbors computation and
    the neighborhood enrichment analysis (i.e. `squidpy.gr.spatial_neighbors` and `squidpy.gr.nhood_enrichment`).
    Results can then be visualized using e.g. `squidpy.pl.nhood_enrichment`.

    Parameters
    ----------
    sdata
        The input SpatialData object containing spatial data.
    labels_layer
        The key in `sdata.labels` from which the cluster label data is extracted.
        This labels layer is typically obtained using `harpy.im.flowsom`.
    size
        The grid size used to subdivide the spatial domain for extracting pixel-wise cluster labels.
    subset
        A list of labels to subset the analysis to, or None to include all labels in `labels_layer`.
    spatial_neighbors_kwargs
        Additional keyword arguments to be passed to `squidpy.gr.spatial_neighbors`.
    nhood_enrichment_kwargs
        Additional keyword arguments to be passed to `squidpy.gr.nhood_enrichment`.
    seed
        The random seed used for reproducibility in the neighborhood enrichment computation.
    key_added
        The key under which the extracted cluster labels will be stored in `.obs` of the returned AnnData object.

    Returns
    -------
    An AnnData object enriched with spatial neighbor information and neighborhood enrichment statistics.

    See Also
    --------
    harpy.im.flowsom : flowsom pixel clustering on image layers.
    """
    array = sdata.labels[labels_layer].data.compute()
    cluster_ids, coordinates = _get_values_grid(array=array, size=size, subset=subset)

    cluster_ids = cluster_ids.flatten()

    obs = pd.DataFrame({key_added: pd.Categorical(cluster_ids)})

    adata = AnnData(obs=obs)

    adata.obsm[_SPATIAL] = coordinates

    spatial_neighbors_kwargs = dict(spatial_neighbors_kwargs)
    coord_type = spatial_neighbors_kwargs.pop("coord_type", "grid")
    spatial_key = spatial_neighbors_kwargs.pop("spatial_key", _SPATIAL)

    nhood_enrichment_kwargs = dict(nhood_enrichment_kwargs)
    cluster_key = nhood_enrichment_kwargs.pop("cluster_key", key_added)
    seed = nhood_enrichment_kwargs.pop("seed", seed)

    sq.gr.spatial_neighbors(
        adata, spatial_key=spatial_key, coord_type=coord_type, copy=False, **spatial_neighbors_kwargs
    )
    sq.gr.nhood_enrichment(adata, cluster_key=cluster_key, seed=seed, copy=False, **nhood_enrichment_kwargs)

    return adata


def _get_values_grid(
    array: NDArray,
    size: int = 50,
    subset: list[int] | None = None,
) -> tuple[NDArray, NDArray]:
    # get values in a grid.
    assert array.ndim == 2, "Currently only 2D arrays are supported."

    y_coords = np.arange(0, array.shape[0], size)
    x_coords = np.arange(0, array.shape[1], size)

    y_grid, x_grid = np.meshgrid(y_coords, x_coords, indexing="ij")

    y_grid = y_grid.ravel()
    x_grid = x_grid.ravel()

    sampled_values = array[y_grid, x_grid]

    result = np.column_stack((sampled_values, y_grid, x_grid))

    if subset is not None:
        mask = np.isin(sampled_values, subset)
        result = result[mask]

    values = result[:, :1]
    coordinates = result[:, -2:]

    return values, coordinates
