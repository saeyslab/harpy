from __future__ import annotations

import uuid
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from numpy.typing import NDArray
from spatialdata import SpatialData
from spatialdata.transformations import get_transformation

from harpy.image._image import _get_spatial_element
from harpy.image.segmentation._grid import add_grid_labels_layer
from harpy.utils._aggregate import RasterAggregator
from harpy.utils._keys import _INSTANCE_KEY, _SPATIAL
from harpy.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

try:
    import squidpy as sq
except ImportError:
    log.warning("'squidpy' not installed, to use 'harpy.tb.spatial_pixel_neighbors' please install this library.")


def spatial_pixel_neighbors(
    sdata: SpatialData,
    labels_layer: str,
    size: int = 20,
    mode: Literal["most_frequent", "center"] = "most_frequent",
    grid_type: Literal["hexagon", "square"] = "hexagon",  # ignored if mode is 'most_frequent'
    subset: list[int] | None = None,
    spatial_neighbors_kwargs: Mapping[str, Any] = MappingProxyType({}),
    nhood_enrichment_kwargs: Mapping[str, Any] = MappingProxyType({}),
    seed: int = 0,
    key_added: str = "cluster_id",
) -> AnnData:
    """
    Computes spatial pixel neighbors and performs neighborhood enrichment analysis.

    This function extracts grid-based cluster labels from the specified labels layer of a SpatialData object,
    subdivides the spatial domain into a grid using a specified sampling interval, and computes spatial neighbors along with
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
        If `mode` is `"center"`, `size` determines the sampling interval for constructing the spatial grid.
        This value determines the distance (in pixels) between consecutive grid points along each axis. A smaller value produces a denser grid (higher resolution),
        while a larger value yields a sparser grid.
        If `mode` is `"most_frequent"`, this value is passed to `harpy.im.add_grid_labels_layer`.
    mode
        The method used to extract grid-based pixel cluster labels. Can be either `"most_frequent"` or `"center"`.
        - `"most_frequent"`: Assigns each grid point the most frequently occurring label within the surrounding
        neighborhood, determined by `size` and `grid_type`. This approach smooths local variations and
        provides a more representative cluster label for each region.
        - `"center"`: Assigns each grid point the label of the pixel at its exact center, without considering
        neighboring pixels. This method maintains high spatial precision but may be more sensitive to local noise.
        When using `"most_frequent"`, the `grid_type` parameter determines whether a hexagonal or square
        grid is used for sampling. If `"center"` is selected, `grid_type` is ignored.
    grid_type
        The type of grid used when extracting pixel cluster labels from `labels_layer`. Can be either `"hexagon"` or `"square"`.
        This parameter is only relevant when `mode="most_frequent"` and is ignored when `mode="center"`.
        Passed to `harpy.im.add_grid_labels_layer`.
    subset
        A list of labels to subset the analysis to, or `None` to include all labels in `labels_layer`.
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
    harpy.im.add_grid_labels_layer : construct a grid.
    """
    if mode == "center":
        array = sdata.labels[labels_layer].data.compute()
        cluster_ids, coordinates = _get_values_grid_center(array=array, size=size, subset=subset)
    elif mode == "most_frequent":
        cluster_ids, coordinates = _get_values_grid_most_frequent(
            sdata, labels_layer=labels_layer, size=size, grid_type=grid_type, subset=subset
        )
    else:
        raise ValueError("Please set 'mode' to either 'center' or 'most_frequent'.")

    cluster_ids = cluster_ids.flatten()

    obs = pd.DataFrame({key_added: pd.Categorical(cluster_ids)})

    adata = AnnData(obs=obs)

    log.info(f"Using '{adata.shape[0]}' observations for neighborhood analysis.")

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


def _get_values_grid_center(
    array: NDArray,
    size: int = 50,
    subset: list[int] | None = None,
) -> tuple[NDArray, NDArray]:
    # get values in a grid.
    assert array.ndim == 2, "Currently only support for 2D ('y','x')."

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


def _get_values_grid_most_frequent(
    sdata: SpatialData,
    labels_layer: str,
    size: int = 20,
    grid_type: str = "hexagon",
    subset: list[int] | None = None,
) -> tuple[NDArray, NDArray]:
    assert _get_spatial_element(sdata, layer=labels_layer).data.ndim == 2, "Currently only support for 2D ('y','x')."
    _uuid = uuid.uuid4()
    # Make a grid, either hexagons or squares.
    sdata = add_grid_labels_layer(
        sdata,
        shape=sdata.labels[labels_layer].shape,
        size=size,
        output_labels_layer=f"labels_grid_{_uuid}",
        output_shapes_layer=f"shapes_grid_{_uuid}",
        grid_type=grid_type,
        chunks=_get_spatial_element(sdata, layer=labels_layer).data.chunksize[
            -1
        ],  # if chunksize in y would be different than in x, we rechunk, see below
        transformations=get_transformation(sdata[labels_layer], get_all=True),
        overwrite=True,
    )
    mask_grid = _get_spatial_element(sdata, layer=f"labels_grid_{_uuid}").data
    mask_pixel_clusters = _get_spatial_element(sdata, layer=labels_layer).data

    if mask_grid.chunksize != mask_pixel_clusters.chunksize:
        mask_pixel_clusters.rechunk(mask_grid.chunksize)

    mask_grid = mask_grid[None, ...]  # RasterAggregator only supports z,y,x
    mask_pixel_clusters = mask_pixel_clusters[None, None, ...]  # RasterAggregator only supports c,z,y,x

    aggregator = RasterAggregator(mask_dask_array=mask_grid, image_dask_array=mask_pixel_clusters)

    def _get_most_frequent_element_in_mask(mask_grid: NDArray, mask_pixel_clusters: NDArray) -> NDArray:
        unique_labels = np.unique(mask_grid)
        unique_labels = unique_labels[unique_labels != 0]
        results = []

        for _label in unique_labels:
            unique_elements, counts = np.unique(mask_pixel_clusters[mask_grid == _label], return_counts=True)
            results.append(unique_elements[np.argmax(counts)])

        return np.array(results).reshape(-1, 1).astype(np.float32)

    values = aggregator.aggregate_custom_channel(
        image=mask_pixel_clusters[0],
        mask=mask_grid,
        depth=size,
        fn=_get_most_frequent_element_in_mask,
        features=1,
        dtype=np.uint32,
    )

    center_of_mass = aggregator.center_of_mass()
    df = center_of_mass[center_of_mass[_INSTANCE_KEY] != 0]
    coordinates = df.values[:, -2:]  # gives you y,x coordinates

    if subset is not None:
        mask = np.isin(values, subset).flatten()
        values = values[mask]
        coordinates = coordinates[mask]

    # clean up
    for _layer in [f"labels_grid_{_uuid}", f"shapes_grid_{_uuid}"]:
        log.info(f"Removing layer '{_layer}' containing the spatial grid.")
        del sdata[_layer]
        if sdata.is_backed():
            sdata.delete_element_from_disk(_layer)

    return values, coordinates
