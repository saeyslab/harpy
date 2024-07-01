from __future__ import annotations

import uuid
from collections import namedtuple

import anndata as ad
import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse
from spatialdata import SpatialData

from sparrow.image._image import _get_spatial_element, _get_translation
from sparrow.shape._shape import _filter_shapes_layer
from sparrow.table._table import _add_table_layer
from sparrow.utils._keys import _CELL_INDEX, _GENES_KEY, _INSTANCE_KEY, _REGION_KEY
from sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def allocate(
    sdata: SpatialData,
    labels_layer: str,
    points_layer: str = "transcripts",
    output_layer: str = "table_transcriptomics",
    chunks: str | tuple[int, ...] | int | None = 10000,
    name_gene_column: str = _GENES_KEY,
    append: bool = False,
    overwrite: bool = False,
) -> SpatialData:
    # TODO: add update_shapes_layer as a parameter
    """
    Allocates transcripts to cells via provided `labels_layer` and `points_layer` and returns updated SpatialData object with a table layer (`sdata.tables[output_layer]`) holding the AnnData object with cell counts.

    Parameters
    ----------
    sdata
        The SpatialData object.
    labels_layer
        The labels layer (i.e. segmentation mask) in `sdata` to be used to allocate the transcripts to cells.
    points_layer
        The points layer in `sdata` that contains the transcripts.
    output_layer
        The table layer in `sdata` in which to save the AnnData object with the transcripts counts per cell.
    chunks
        Chunk sizes for processing. Can be a string, integer or tuple of integers.
        Consider setting the chunks to a relatively high value to speed up processing
        (>10000, or only chunk in z-dimension if data is 3D, and one z-slice fits in memory),
        taking into account the available memory of your system.
    name_gene_column
        Column name in the `points_layer` representing gene information.
    append
        If set to True, and the `labels_layer` does not yet exist as a `_REGION_KEY` in `sdata.tables[output_layer].obs`,
        the transcripts counts obtained during the current function call will be appended (along axis=0) to any existing transcript count values.
        within the SpatialData object's table attribute. If False, and overwrite is set to True any existing data in `sdata.tables[output_layer]` will be overwritten by the newly extracted transcripts counts.
    overwrite
        If True, overwrites the `output_layer` if it already exists in `sdata`.

    Returns
    -------
        An updated SpatialData object with an AnnData table added to `sdata.tables` at slot `output_layer`.
    """
    if labels_layer not in [*sdata.labels]:
        raise ValueError(
            f"Provided labels layer '{labels_layer}' not in 'sdata', please specify a labels layer from '{[*sdata.labels]}'"
        )

    Coords = namedtuple("Coords", ["x0", "y0"])
    se = _get_spatial_element(sdata, layer=labels_layer)
    coords = Coords(*_get_translation(se))

    arr = se.data

    if chunks is not None:
        arr = arr.rechunk(chunks)
    else:
        arr = arr.rechunk(arr.chunksize)

    if arr.ndim == 2:
        arr = arr[None, ...]

    ddf = sdata.points[points_layer]

    delayed_chunks = arr.to_delayed().flatten()

    # chunk info needed for querying
    chunk_info = []
    _chunks = arr.chunks

    # Iterate over each chunk and compute its coordinates and size, needed for query
    for i in range(delayed_chunks.shape[0]):
        z, y, x = np.unravel_index(i, [len(_chunks[0]), len(_chunks[1]), len(_chunks[2])])
        size = (_chunks[0][z], _chunks[1][y], _chunks[2][x])
        start_coords = (sum(_chunks[0][:z]), sum(_chunks[1][:y]), sum(_chunks[2][:x]))
        chunk_info.append((start_coords, size))

    log.info("Calculating cell counts.")

    @dask.delayed
    def _process_partition(_chunk, _chunk_info, ddf_partition):
        ddf_partition = ddf_partition.copy()

        z_start, y_start, x_start = _chunk_info[0]

        if "z" in ddf_partition.columns:
            z_coords = ddf_partition["z"].values.astype(int) - z_start
        else:
            z_coords = 0

        y_coords = ddf_partition["y"].values.astype(int) - (int(coords.y0) + y_start)
        x_coords = ddf_partition["x"].values.astype(int) - (int(coords.x0) + x_start)

        ddf_partition.loc[:, _CELL_INDEX] = _chunk[
            z_coords,
            y_coords,
            x_coords,
        ]

        return ddf_partition

    # Create a list to store delayed operations
    delayed_objects = []

    for _chunk, _chunk_info in zip(delayed_chunks, chunk_info):
        # Query the partition lazily without computing it
        z_start, y_start, x_start = _chunk_info[0]
        _chunk_shape = _chunk_info[1]

        y_query = f"{y_start + coords.y0 } <= y < {y_start + coords.y0 + _chunk_shape[1]}"
        x_query = f"{x_start + coords.x0 } <= x < {x_start + coords.x0 + _chunk_shape[2]}"
        query = f"{y_query} and {x_query}"

        if "z" in ddf.columns:
            z_query = f"{z_start} <= z < {z_start + _chunk_shape[0]}"
            query = f"{z_query} and {query}"

        ddf_partition = ddf.query(query)
        delayed_partition = _process_partition(_chunk, _chunk_info, ddf_partition)
        delayed_objects.append(delayed_partition)

    # Combine the delayed partitions into a single Dask DataFrame
    combined_partitions = dd.from_delayed(delayed_objects)

    if "z" in combined_partitions:
        coordinates = combined_partitions.groupby(_CELL_INDEX)["x", "y", "z"].mean()
    else:
        coordinates = combined_partitions.groupby(_CELL_INDEX)["x", "y"].mean()

    cell_counts = combined_partitions.groupby([_CELL_INDEX, name_gene_column]).size()

    cell_counts = cell_counts.map_partitions(lambda x: x.astype(np.uint32))

    coordinates, cell_counts = dask.compute(coordinates, cell_counts)

    cell_counts = cell_counts.to_frame(name="values")
    cell_counts = cell_counts.reset_index()

    cell_counts[name_gene_column] = cell_counts[name_gene_column].astype("object")
    cell_counts[name_gene_column] = pd.Categorical(cell_counts[name_gene_column])

    columns_categories = cell_counts[name_gene_column].cat.categories.to_list()
    columns_nodes = pd.Categorical(cell_counts[name_gene_column], categories=columns_categories, ordered=True)

    indices_of_aggregated_rows = np.array(cell_counts[_CELL_INDEX])
    rows_categories = np.unique(indices_of_aggregated_rows)

    rows_nodes = pd.Categorical(indices_of_aggregated_rows, categories=rows_categories, ordered=True)

    X = sparse.coo_matrix(
        (
            cell_counts["values"].values.ravel(),
            (rows_nodes.codes, columns_nodes.codes),
        ),
        shape=(len(rows_categories), len(columns_categories)),
    ).tocsr()

    adata = AnnData(
        X,
        obs=pd.DataFrame(index=rows_categories),
        var=pd.DataFrame(index=columns_categories),
        dtype=X.dtype,
    )

    coordinates.index = coordinates.index.map(str)

    # sanity check
    assert np.array_equal(np.unique(coordinates.index), np.unique(adata.obs.index))

    # make sure coordinates is in same order as adata
    coordinates = coordinates.reindex(adata.obs.index)

    adata = adata[adata.obs.index != "0"]

    adata.obsm["spatial"] = coordinates[coordinates.index != "0"].values
    adata.obs[_INSTANCE_KEY] = adata.obs.index.astype(int)

    adata.obs[_REGION_KEY] = pd.Categorical([labels_layer] * len(adata.obs))

    _uuid_value = str(uuid.uuid4())[:8]
    adata.obs.index = adata.obs.index.map(lambda x: f"{x}_{labels_layer}_{_uuid_value}")

    adata.obs.index.name = _CELL_INDEX

    if append:
        region = []
        if output_layer in [*sdata.tables]:
            if labels_layer in sdata.tables[output_layer].obs[_REGION_KEY].cat.categories:
                raise ValueError(
                    f"'{labels_layer}' already exists as a region in the 'sdata.tables[{output_layer}]' object. "
                    "Please choose a different labels layer, choose a different 'output_layer' or set append to False and overwrite to True to overwrite the existing table."
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
