from __future__ import annotations

import itertools
from collections import namedtuple

import dask
import dask.array as da
import dask.dataframe as dd
import pandas as pd
import rasterio
import rasterio.features
import spatialdata
from affine import Affine
from anndata import AnnData
from spatialdata import SpatialData

from sparrow.image._image import _get_spatial_element, _get_translation
from sparrow.shape._shape import _filter_shapes_layer
from sparrow.utils._keys import _CELL_INDEX, _INSTANCE_KEY, _REGION_KEY
from sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def allocate(
    sdata: SpatialData,
    labels_layer: str = "segmentation_mask",
    shapes_layer: str | None = "segmentation_mask_boundaries",
    points_layer: str = "transcripts",
    allocate_from_shapes_layer: bool = True,
    chunks: str | tuple[int, ...] | int | None = 10000,
) -> SpatialData:
    """
    Allocates transcripts to cells via provided shapes_layer and points_layer and returns updated SpatialData augmented with a table attribute holding the AnnData object with cell counts.

    Parameters
    ----------
    sdata : SpatialData
        The SpatialData object.
    labels_layer : str, optional
        The layer in `sdata` that contains the masks corresponding to the shapes layer
        (possibly before performing operation on the shapes layer, such as calculating voronoi expansion).
        Only used for determining offset if `allocate_from_shapes_layer` is True.
    shapes_layer : str, optional
        The layer in `sdata` that contains the boundaries of the segmentation mask, by default "segmentation_mask_boundaries".
        Required if `allocate_from_shapes_layer` is True.
    points_layer: str, optional
        The layer in `sdata` that contains the transcripts.
    allocate_from_shapes_layer: bool, optional
        Whether to allocate transcripts using `shapes_layer` or `labels_layer`.
        Only supported if `shapes_layer` contains 2D polygons.
    chunks : Optional[str | int | tuple[int, ...]], default=10000
        Chunk sizes for processing. Can be a string, integer or tuple of integers.
        Consider setting the chunks to a relatively high value to speed up processing
        (>10000, or only chunk in z-dimension if data is 3D, and one z-slice fits in memory),
        taking into account the available memory of your system.

    Returns
    -------
        An updated SpatialData object with the added table attribute (AnnData object).
    """
    if shapes_layer is not None:
        sdata[shapes_layer].index = sdata[shapes_layer].index.astype("str")

    if labels_layer not in [*sdata.labels]:
        raise ValueError(
            f"Provided labels layer '{labels_layer}' not in 'sdata', please specify a labels layer from '{[*sdata.labels]}'"
        )

    # need to do this transformation,
    # because the polygons have same offset coords.x0 and coords.y0 as in segmentation_mask
    Coords = namedtuple("Coords", ["x0", "y0"])
    s_mask = _get_spatial_element(sdata, layer=labels_layer)
    coords = Coords(*_get_translation(s_mask))

    if allocate_from_shapes_layer:
        has_z = sdata.shapes[shapes_layer]["geometry"].apply(lambda geom: geom.has_z)
        if any(has_z):
            raise ValueError(
                "Allocating transcripts from a shapes layer is not supported "
                "for shapes layers containing 3D polygons. "
                "Please consider setting 'allocate_from_shapes_layer' to False, "
                "and passing the labels_layer corresponding to the shapes_layer."
            )

        if s_mask.ndim != 2:
            raise ValueError(
                "Allocating transcripts from a shapes layer is not supported "
                f"if corresponding labels_layer {labels_layer} is not 2D."
            )

        transform = Affine.translation(coords.x0, coords.y0)

        log.info("Creating masks from polygons.")
        masks = rasterio.features.rasterize(
            zip(
                sdata[shapes_layer].geometry,
                sdata[shapes_layer].index.values.astype(float),
            ),
            out_shape=[s_mask.shape[0], s_mask.shape[1]],
            dtype="uint32",
            fill=0,
            transform=transform,
        )
        log.info(f"Created masks with shape {masks.shape}.")

        masks = da.from_array(masks)

    else:
        masks = s_mask.data

    if chunks is not None:
        masks = masks.rechunk(chunks)
    else:
        masks = masks.rechunk(masks.chunksize)

    if masks.ndim == 2:
        masks = masks[None, ...]

    ddf = sdata[points_layer]

    log.info("Calculating cell counts.")

    def process_partition(index, chunk, chunk_coord):
        partition = ddf.get_partition(index).compute()

        z_start, y_start, x_start = chunk_coord

        if "z" in partition.columns:
            filtered_partition = partition[
                (coords.y0 + y_start <= partition["y"])
                & (partition["y"] < chunk.shape[1] + coords.y0 + y_start)
                & (coords.x0 + x_start <= partition["x"])
                & (partition["x"] < chunk.shape[2] + coords.x0 + x_start)
                & (z_start <= partition["z"])
                & (partition["z"] < chunk.shape[0] + z_start)
            ]

        else:
            filtered_partition = partition[
                (coords.y0 + y_start <= partition["y"])
                & (partition["y"] < chunk.shape[1] + coords.y0 + y_start)
                & (coords.x0 + x_start <= partition["x"])
                & (partition["x"] < chunk.shape[2] + coords.x0 + x_start)
            ]

        filtered_partition = filtered_partition.copy()

        if "z" in partition.columns:
            z_coords = filtered_partition["z"].values.astype(int) - z_start
        else:
            z_coords = 0

        y_coords = filtered_partition["y"].values.astype(int) - (int(coords.y0) + y_start)
        x_coords = filtered_partition["x"].values.astype(int) - (int(coords.x0) + x_start)

        filtered_partition.loc[:, _CELL_INDEX] = chunk[
            z_coords,
            y_coords,
            x_coords,
        ]

        return filtered_partition

    # Get the number of partitions in the Dask DataFrame
    num_partitions = ddf.npartitions

    chunk_coords = list(itertools.product(*[range(0, s, cs) for s, cs in zip(masks.shape, masks.chunksize)]))

    chunks = masks.to_delayed().flatten()

    # Process each partition using its index
    processed_partitions = []

    for _chunk, _chunk_coord in zip(chunks, chunk_coords):
        processed_partitions = processed_partitions + [
            dask.delayed(process_partition)(i, _chunk, _chunk_coord) for i in range(num_partitions)
        ]

    # Combine the processed partitions into a single DataFrame
    combined_partitions = dd.from_delayed(processed_partitions)

    if "z" in combined_partitions:
        coordinates = combined_partitions.groupby(_CELL_INDEX)["x", "y", "z"].mean()
    else:
        coordinates = combined_partitions.groupby(_CELL_INDEX)["x", "y"].mean()

    cell_counts = combined_partitions.groupby([_CELL_INDEX, "gene"]).size()

    coordinates, cell_counts = dask.compute(coordinates, cell_counts, scheduler="threads")

    cell_counts = cell_counts.unstack(fill_value=0)
    # convert dtype of columns to "object", otherwise error writing to zarr.
    cell_counts.columns = cell_counts.columns.astype(str)

    log.info("Finished calculating cell counts.")

    # make sure coordinates are sorted in same order as cell_counts
    index_order = cell_counts.index.argsort()
    coordinates = coordinates.loc[cell_counts.index[index_order]]
    cell_counts = cell_counts.sort_index()

    log.info("Creating AnnData object.")

    # Create the anndata object
    cell_counts.index = cell_counts.index.map(str)
    adata = AnnData(cell_counts[cell_counts.index != "0"])
    coordinates.index = coordinates.index.map(str)
    adata.obsm["spatial"] = coordinates[coordinates.index != "0"].values
    adata.obs[_INSTANCE_KEY] = adata.obs.index.astype(int)

    # currently only support adding only one field of view
    adata.obs[_REGION_KEY] = pd.Categorical([labels_layer] * len(adata.obs))

    if sdata.table:
        del sdata.table

    sdata.table = spatialdata.models.TableModel.parse(
        adata, region_key=_REGION_KEY, region=[labels_layer], instance_key=_INSTANCE_KEY
    )

    indexes_to_keep = sdata.table.obs[_INSTANCE_KEY].values.astype(int)
    sdata = _filter_shapes_layer(
        sdata,
        indexes_to_keep=indexes_to_keep,
        prefix_filtered_shapes_layer="filtered_segmentation",
    )

    return sdata
