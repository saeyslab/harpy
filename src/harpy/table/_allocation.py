from __future__ import annotations

import uuid
from collections import Counter, namedtuple

import anndata as ad
import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
from anndata import AnnData
from dask.dataframe import DataFrame as DaskDataFrame
from scipy import sparse
from scipy.sparse import issparse
from spatialdata import SpatialData
from spatialdata.models import PointsModel
from spatialdata.transformations import Identity
from xarray import DataArray

from harpy.image._image import _get_spatial_element, _get_translation
from harpy.shape._shape import filter_shapes_layer
from harpy.table._table import add_table_layer
from harpy.utils._keys import _CELL_INDEX, _GENES_KEY, _INSTANCE_KEY, _REGION_KEY, _SPATIAL
from harpy.utils._transformations import _identity_check_transformations_points
from harpy.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def allocate(
    sdata: SpatialData,
    labels_layer: str,
    points_layer: str = "transcripts",
    output_layer: str = "table_transcriptomics",
    to_coordinate_system: str = "global",
    chunks: str | tuple[int, ...] | int | None = 10000,
    name_gene_column: str = _GENES_KEY,
    append: bool = False,
    update_shapes_layers: bool = True,
    overwrite: bool = False,
) -> SpatialData:
    """
    Allocates transcripts to cells via provided `labels_layer` and `points_layer` and returns updated SpatialData object with a table layer (`sdata.tables[output_layer]`) holding the AnnData object with cell counts.

    It requires that `labels_layer` and `points_layer` are registered.
    Relation between `to_coordinate_system` and `points_layer` should be a `spatialdata.transformations.Identity` transformation.
    Relation between `to_coordinate_system` and `labels_layer` can be a `spatialdata.transformations.Identity`, `spatialdata.transformations.Translation`, or a `spatialdata.transformation.Sequence` of translations.

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
    to_coordinate_system
        The coordinate system that holds `labels_layer` and `points_layer`.
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
    update_shapes_layers
        Whether to filter the shapes layers associated with `labels_layer`.
        If set to `True`, cells that do not appear in resulting `output_layer` (with `_REGION_KEY` equal to `labels_layer`) will be removed from the shapes layers (via `_INSTANCE_KEY`) in the `sdata` object.
        Filtered shapes will be added to `sdata` with prefix 'filtered_segmentation'.
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
    ddf = sdata.points[points_layer]

    se = _get_spatial_element(sdata, layer=labels_layer)

    combined_partitions = _aggregate(
        se=se,
        ddf=ddf,
        value_key=name_gene_column,
        drop_coordinates=False,
        to_coordinate_system=to_coordinate_system,
        chunks=chunks,
    )

    if "z" in combined_partitions:
        coordinates = combined_partitions.groupby(_CELL_INDEX)["x", "y", "z"].mean()
    else:
        coordinates = combined_partitions.groupby(_CELL_INDEX)["x", "y"].mean()

    # make sure combined_partiions[ name_gene_column ] is not categorical,
    # because otherwise resulting cell_counts dataframe will contain zero counts for each gene for each cells (which would results in a huge dataframe)
    combined_partitions[name_gene_column] = combined_partitions[name_gene_column].astype("str")

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

    adata.obsm[_SPATIAL] = coordinates.values

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

    sdata = add_table_layer(
        sdata,
        adata=adata,
        output_layer=output_layer,
        region=region,
        overwrite=overwrite,
    )

    if update_shapes_layers:
        sdata = filter_shapes_layer(
            sdata,
            table_layer=output_layer,
            labels_layer=labels_layer,
            prefix_filtered_shapes_layer="filtered_segmentation",
        )

    return sdata


def bin_counts(
    sdata: SpatialData,
    table_layer: str,
    labels_layer: str,
    output_layer: str,
    to_coordinate_system: str = "global",
    chunks: str | tuple[int, ...] | int | None = 10000,
    append: bool = True,
    overwrite: bool = False,
) -> SpatialData:
    """
    Bins gene counts from barcodes to cells or regions defined in `labels_layer` and returns an updated SpatialData object with a table layer (`sdata.tables[output_layer]`) holding an AnnData object with the binned counts per cell or region.

    Parameters
    ----------
    sdata
        The SpatialData object.
    table_layer
        The table layer holding the counts. E.g. obtained using `harpy.io.visium_hd`.
        We assume that `sdata[table_layer].obsm[_SPATIAL]` contains a numpy array holding the barcode coordinates ('x', 'y').
        The relation of `sdata[table_layer].obsm[_SPATIAL]` to `to_coordinate_system` should be an identity transformation.
    labels_layer
        The labels layer (e.g., segmentation mask, or a grid generated by `harpy.im.add_grid_labels_layer`) in `sdata` used to bin barcodes (as specified via `table_layer`) into cells or regions.
    output_layer
        The table layer in `sdata` in which to save the AnnData object with the binned counts per cell or region defined by `labels_layer`.
    to_coordinate_system
        The coordinate system that holds `labels_layer`.
    chunks
        Chunk sizes for processing. Can be a string, integer, or tuple of integers.
        Consider setting the chunks to a relatively high value to speed up processing,
        taking into account the available memory of your system.
    append
        If set to `True`, and the `labels_layer` does not yet exist as a `_REGION_KEY` in `sdata.tables[output_layer].obs`,
        the binned counts obtained during the current function call will be appended (along axis=0) to `output_layer`.
        If `False`, and `overwrite` is set to `True`, any existing data in `sdata.tables[output_layer]` will be overwritten by the newly binned counts.
    overwrite
        If `True`, overwrites the `output_layer` if it already exists in `sdata`.

    Returns
    -------
        An updated SpatialData object with an AnnData table added to `sdata.tables` at slot `output_layer`.
    """
    se = _get_spatial_element(sdata, layer=labels_layer)

    # sdata[table_layer].obsm[_SPATIAL] contains the positions of the barcodes if visium reader is used 'harpy.io.visium_hd'
    name_x = "x"
    name_y = "y"
    df = pd.DataFrame(sdata[table_layer].obsm[_SPATIAL], columns=[name_x, name_y])
    name_barcode_id = "barcode_id"
    df[name_barcode_id] = sdata[table_layer].obs.index

    ddf = PointsModel.parse(
        df,
        transformations={to_coordinate_system: Identity()},
    )

    combined_partitions = _aggregate(
        se=se,
        ddf=ddf,
        chunks=chunks,
        to_coordinate_system=to_coordinate_system,
        name_x=name_x,
        name_y=name_y,
        drop_coordinates=False,
        value_key=name_barcode_id,
    )

    coordinates = combined_partitions.groupby(_CELL_INDEX)[name_x, name_y].mean()

    cell_counts = combined_partitions.groupby([name_barcode_id, _CELL_INDEX]).size()

    cell_counts = cell_counts.map_partitions(lambda x: x.astype(np.uint32))

    coordinates, cell_counts = dask.compute(coordinates, cell_counts)

    # Sanity check that every barcode that could be assigned to a bin is assigned exactly ones to a bin.
    _mask = cell_counts == 1
    assert _mask.all(), f"Some spots, given by 'sdata.tables[{table_layer}].obsm[{_SPATIAL}]', where assigned to more than one cell defined in '{labels_layer}'."
    cell_counts = cell_counts.reset_index(level=_CELL_INDEX)
    assert cell_counts.index.is_unique, "Spots should not be assigned to more than one cell."

    value_counts_counter = Counter(cell_counts.groupby(_CELL_INDEX).count()[0])
    value_counts_sorted = sorted(value_counts_counter.items())
    df = pd.DataFrame(value_counts_sorted, columns=["Number of spots per bin", "Frequency"])
    log.info(f"\n{df.to_string(index=False)}")
    # get adata
    adata_in = sdata.tables[table_layer].copy()  # should we do a copy here? otherwise in memory adata will be changed
    merged = pd.merge(adata_in.obs, cell_counts[_CELL_INDEX], left_index=True, right_index=True, how="inner")
    assert (
        merged.shape[0] != 0
    ), "Result after merging AnnData object, passed via 'table_layer' parameter with aggregated spots is empty."
    adata_in = adata_in[merged.index]
    adata_in.obs = merged

    group_labels = adata_in.obs[_CELL_INDEX].values
    unique_labels, group_indices = np.unique(group_labels, return_inverse=True)
    N_groups = len(unique_labels)

    assert issparse(adata_in.X), "Currently only AnnData objects with a sparse feature matrix are supported."

    # Extract the gene expression counts
    counts = adata_in.X

    rows = group_indices
    cols = np.arange(len(group_indices))
    data = np.ones(len(group_indices))
    group_indicator = sparse.csr_matrix((data, (rows, cols)), shape=(N_groups, counts.shape[0]))

    summed_counts = group_indicator.dot(counts)

    # exclude bins for which sum is zero (i.e. no genes detected)
    row_sums = np.array(summed_counts.sum(axis=1)).flatten()
    nonzero_rows = row_sums != 0
    summed_counts = summed_counts[nonzero_rows, :]
    unique_labels = unique_labels[nonzero_rows]

    adata = AnnData(
        X=summed_counts, obs=pd.DataFrame(unique_labels, columns=[_INSTANCE_KEY], index=unique_labels), var=adata_in.var
    )

    adata.obs[_REGION_KEY] = pd.Categorical([labels_layer] * len(adata.obs))

    _uuid_value = str(uuid.uuid4())[:8]
    adata.obs.index = adata.obs.index.map(lambda x: f"{x}_{labels_layer}_{_uuid_value}")
    adata.obs.index.name = _CELL_INDEX

    # now add the coordinates
    # coordinates are the average x,y coordinate of the assigned spots per bin/cell
    # adata.obs[ _INSTANCE_KEY ] is also sorted. And index of coordinates corresponds to _INSTANCE_KEY.
    adata.obsm[_SPATIAL] = coordinates[coordinates.index.isin(adata.obs[_INSTANCE_KEY])].sort_index().values

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

    sdata = add_table_layer(
        sdata,
        adata=adata,
        output_layer=output_layer,
        region=region,
        overwrite=overwrite,
    )

    return sdata


def _aggregate(
    se: DataArray,
    ddf: DaskDataFrame,
    value_key: str,
    drop_coordinates: bool = False,  # if set to True, will drop ((z),y,x) in resulting dask dataframe
    chunks: str | tuple[int, ...] | int | None = 10000,
    to_coordinate_system: str = "global",
    name_x: str = "x",
    name_y: str = "y",
    name_z: str = "z",
) -> DaskDataFrame:
    # helper function to do an aggregation between a dask array containing ints, and a dask dataframe containing coordinates ((z), y, x).
    assert np.issubdtype(se.data.dtype, np.integer), "Only integer arrays are supported."
    assert name_y in ddf and name_x in ddf, f"Dask Dataframe must contain '{name_y}' and '{name_x}' columns."
    Coords = namedtuple("Coords", ["x0", "y0"])
    coords = Coords(*_get_translation(se, to_coordinate_system=to_coordinate_system))
    _identity_check_transformations_points(ddf, to_coordinate_system=to_coordinate_system)

    value_keys = [name_x, name_y, name_z, value_key] if name_z in ddf.columns else [name_x, name_y, value_key]

    ddf = ddf[value_keys]

    arr = se.data

    if chunks is not None:
        arr = arr.rechunk(chunks)
    else:
        arr = arr.rechunk(arr.chunksize)

    if arr.ndim == 2:
        arr = arr[None, ...]

    ddf[name_x] = ddf[name_x].round().astype(int)
    ddf[name_y] = ddf[name_y].round().astype(int)
    if name_z in ddf.columns:
        ddf[name_z] = ddf[name_z].round().astype(int)

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

        if name_z in ddf_partition.columns:
            z_coords = ddf_partition[name_z].values.astype(int) - z_start
        else:
            z_coords = 0

        y_coords = ddf_partition[name_y].values.astype(int) - (int(coords.y0) + y_start)
        x_coords = ddf_partition[name_x].values.astype(int) - (int(coords.x0) + x_start)

        ddf_partition.loc[:, _CELL_INDEX] = _chunk[
            z_coords,
            y_coords,
            x_coords,
        ]

        return ddf_partition

    # Create a list to store delayed operations
    delayed_objects = []

    for _chunk, _chunk_info in zip(delayed_chunks, chunk_info, strict=True):
        # Query the partition lazily without computing it
        z_start, y_start, x_start = _chunk_info[0]
        _chunk_shape = _chunk_info[1]

        y_query = f"{y_start + coords.y0} <= {name_y} < {y_start + coords.y0 + _chunk_shape[1]}"
        x_query = f"{x_start + coords.x0} <= {name_x} < {x_start + coords.x0 + _chunk_shape[2]}"
        query = f"{y_query} and {x_query}"

        if name_z in ddf.columns:
            z_query = f"{z_start} <= {name_z} < {z_start + _chunk_shape[0]}"
            query = f"{z_query} and {query}"

        ddf_partition = ddf.query(query)
        delayed_partition = _process_partition(_chunk, _chunk_info, ddf_partition)
        delayed_objects.append(delayed_partition)

    # Combine the delayed partitions into a single Dask DataFrame
    combined_partitions = dd.from_delayed(delayed_objects)

    # remove background
    combined_partitions = combined_partitions[combined_partitions[_CELL_INDEX] != 0]

    if drop_coordinates:
        combined_partitions = combined_partitions[[value_key, _CELL_INDEX]]

    return combined_partitions
