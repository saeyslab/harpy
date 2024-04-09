from __future__ import annotations

import itertools
import uuid
from collections import namedtuple

import anndata as ad
import dask
import dask.array as da
import dask.dataframe as dd
import pandas as pd
import rasterio
import rasterio.features
from affine import Affine
from anndata import AnnData
from spatialdata import SpatialData

from sparrow.image._image import _get_spatial_element, _get_translation
from sparrow.shape._shape import _filter_shapes_layer
from sparrow.table._table import _add_table_layer
from sparrow.utils._keys import _CELL_INDEX, _INSTANCE_KEY, _REGION_KEY
from sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def allocate(
    sdata: SpatialData,
    labels_layer: str = "segmentation_mask",
    shapes_layer: str | None = "segmentation_mask_boundaries",
    points_layer: str = "transcripts",
    output_layer: str = "table_transcriptomics",
    allocate_from_shapes_layer: bool = True,
    chunks: str | tuple[int, ...] | int | None = 10000,
    append: bool = False,
    overwrite: bool = False,
) -> SpatialData:
    """
    Allocates transcripts to cells via provided shapes_layer/labels_layer and points_layer and returns updated SpatialData augmented with a table layer (`sdata.tables[output_layer]`) holding the AnnData object with cell counts.

    Parameters
    ----------
    sdata : SpatialData
        The SpatialData object.
    labels_layer : str, optional
        The labels layer (i.e. segmentation mask) in `sdata` to be used to allocate the transcripts to cells.
        If `allocate_from_shapes_layer` is True, it should have same offset as the provided `shapes_layer`,
        and it will only be used for determining this offset.
    shapes_layer : str, optional
        The layer in `sdata` that contains the boundaries of the segmentation mask, by default "segmentation_mask_boundaries".
        Required if `allocate_from_shapes_layer` is True, else it will be ignored.
    points_layer: str, optional
        The points layer in `sdata` that contains the transcripts.
    output_layer: str, optional
        The table layer in `sdata` in which to save the AnnData object with the transcripts counts per cell.
    allocate_from_shapes_layer: bool, optional
        Whether to allocate transcripts using `shapes_layer` or `labels_layer`.
        Only supported for `shapes_layer` that contain 2D polygons.
    chunks : Optional[str | int | tuple[int, ...]], default=10000
        Chunk sizes for processing. Can be a string, integer or tuple of integers.
        Consider setting the chunks to a relatively high value to speed up processing
        (>10000, or only chunk in z-dimension if data is 3D, and one z-slice fits in memory),
        taking into account the available memory of your system.
    append: bool, optional.
        If set to True, and the `labels_layer` does not yet exist as a `_REGION_KEY` in `sdata.tables[output_layer].obs`,
        the transcripts counts obtained during the current function call will be appended (along axis=0) to any existing transcript count values.
        within the SpatialData object's table attribute. If False, and overwrite is set to True any existing data in `sdata.tables[output_layer]` will be overwritten by the newly extracted transcripts counts.
    overwrite : bool, default=False
        If True, overwrites the `output_layer` if it already exists in `sdata`.

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
    _cells_id = cell_counts.index.astype(int)  # index of cell_counts should already be an int
    _uuid_value = str(uuid.uuid4())[:8]
    cell_counts.index = cell_counts.index.map(lambda x: f"{x}_{labels_layer}_{_uuid_value}")
    cell_counts.index.name = _CELL_INDEX

    adata = AnnData(cell_counts)
    adata.obs[_INSTANCE_KEY] = _cells_id
    adata.obs[_REGION_KEY] = pd.Categorical([labels_layer] * len(adata.obs))
    adata = adata[adata.obs[_INSTANCE_KEY] != 0]

    adata.obsm["spatial"] = coordinates[coordinates.index != 0].values

    if append:
        region = []
        if output_layer in [*sdata.tables]:
            if labels_layer in sdata.tables[output_layer].obs[_REGION_KEY].cat.categories:
                raise ValueError(
                    f"'{labels_layer}' already exists as a region in the 'sdata.tables[{output_layer}]' object. Please choose a different labels layer, choose a different 'output_layer' or set append to False and overwrite to True to overwrite the existing table."
                )
            adata = ad.concat([sdata.tables[output_layer], adata], axis=0)
            # get the regions already in sdata, and append the new one
            region = sdata.tables[output_layer].obs[_REGION_KEY].cat.categories.to_list()
        region.append(labels_layer)

    else:
        region = [labels_layer]

    sdata = _add_table_layer(
        sdata,
        adata=adata,
        output_layer=output_layer,
        region=region,
        overwrite=overwrite,
    )

    mask = sdata.tables[output_layer].obs[_REGION_KEY].isin(region)
    indexes_to_keep = sdata.tables[output_layer].obs[mask][_INSTANCE_KEY].values.astype(int)

    sdata = _filter_shapes_layer(
        sdata,
        indexes_to_keep=indexes_to_keep,
        prefix_filtered_shapes_layer="filtered_segmentation",
    )

    return sdata
