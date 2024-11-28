from __future__ import annotations

import math
import uuid

import dask.array as da
import datashader as ds
import geopandas as gpd
import numpy as np
import shapely
import shapely.validation
from dask import delayed
from dask.distributed import Client
from shapely import GeometryCollection, MultiPolygon, Polygon
from shapely.geometry import box
from spatialdata import SpatialData
from spatialdata.models.models import ScaleFactors_t
from spatialdata.transformations import get_transformation

from sparrow.image._image import add_labels_layer
from sparrow.utils._keys import _INSTANCE_KEY
from sparrow.utils.utils import _get_uint_dtype


def rasterize(
    sdata: SpatialData,
    shapes_layer: str,
    output_layer: str,
    out_shape: tuple[int, int] | None = None,  # output shape in y, x.
    chunks: int | None = None,
    client: Client = None,
    scale_factors: ScaleFactors_t | None = None,
    overwrite: bool = False,
) -> SpatialData:
    """
    Given a shapes layer in a SpatialData object, corresponding masks are created, and added as a labels layer to the SpatialData object.

    The index of the shapes layer will be used as the label in the resulting labels layer (`output_layer`).

    Parameters
    ----------
    sdata
        The SpatialData object.
    shapes_layer
        The shapes layer to be converted to a labels layer.
    output_layer
        Name of the resulting labels layer that will be added to `sdata`.
    out_shape
        Output shape of the resulting labels layer `(y,x)`. Will be automatically calculated if set to None via `sdata[shapes_layer].geometry.total_bounds`.
        If `out_shape` is not `None`, with  `x_min, y_min, x_max, y_max = sdata[shapes_layer].geometry.total_bounds`,
        and `out_shape[1]<x_max` or `out_shape[0]<y_max`, then shapes with coordinates outside `out_shapes` will
        not be in resulting `output_layer`.
        For `shapes_layer` with large offset `(y_min, x_min)`,
        we recommend translating the shapes to the origin, and add the offset via a translation (`spatialdata.transformations.Translation`).
    chunks
        If provided, creation of the labels layer will be done in a chunked manner, with data divided into chunks for efficient computation.
    client
        A `Dask` client. If specified, a copy of `sdata[shapes_layer]` will be scattered across the workers, reducing the size of the task graph.
        If not specified, `Dask` will use the default scheduler as configured on your system.
    client
        A Dask `Client` instance. If specified, a copy of the GeoDataFrame (`sdata[shapes_layer]`) will be scattered across the workers.
        This reduces the size of the task graph and can improve performance by minimizing data transfer overhead during computation.
        If not specified, Dask will use the default scheduler as configured on your system (e.g., single-threaded, multithreaded, or a global client if one is running).
    scale_factors
        Scale factors to apply for multiscale.
    overwrite
        If True, overwrites the `output_layer` if it already exists in `sdata`.

    Returns
    -------
        An updated SpatialData object with the added labels layer.

    Raises
    ------
    ValueError
        If the provided `shapes_layer` contains 3D polygons.
    ValueError
        If the provided `shapes_layer` contains Points.
    ValueError
        If 0 is in the index of the `shapes_layer`. As 0 is used as background in the `output_layer`.
    TypeError
        If `chunks` is not `None` and not an instance of `int`.
    """
    # only 2D polygons are suported.
    has_z = sdata.shapes[shapes_layer]["geometry"].apply(lambda geom: geom.has_z)
    if any(has_z):
        raise ValueError("Shapes layer contains 3D polygons. " "This is currently not supported.")

    if any(sdata.shapes[shapes_layer].geometry.type == "Point"):
        raise ValueError(
            "Shapes layer contains Points. This is currently not supported. Please consider converting the Points to Polygons first using e.g. '.buffer( your_radius, cap_style=your_cap_style )"
        )

    if 0 in sdata[shapes_layer].index.astype(int):
        raise ValueError(
            "0 is in the index of the shapes layer. This is not allowed, because the label 0 is reserved for background. "
            "Either remove the item from the shapes layer or increase indices of shapes with 1."
        )
    if chunks is not None and not isinstance(chunks, int):
        raise TypeError("Parameter 'chunks' must be of type int if not None.")

    bounds = [int(round(coord)) for coord in sdata[shapes_layer].geometry.total_bounds]
    x_min, y_min, x_max, y_max = bounds
    y_min = y_min if y_min > 0 else 0
    x_min = x_min if x_min > 0 else 0

    assert (
        x_max > 0 and y_max > 0
    ), f"The maximum of the bounding box of the shapes layer {shapes_layer} is negative. This is not allowed."
    shapes = sdata[shapes_layer].copy()
    shapes.index = shapes.index.values.astype(int)
    # set index name to this value, because otherwise reset_index could cause error, if _INSTANCE_KEY column already exists in the shapes layer
    index_name = f"{_INSTANCE_KEY}_{uuid.uuid4()}"
    shapes.index.name = index_name
    shapes.reset_index(inplace=True)

    if out_shape is not None:
        if out_shape[0] <= y_min or out_shape[1] <= x_min:
            raise ValueError("ValueError")
        y_max = out_shape[0]
        x_max = out_shape[1]

    if chunks is None:
        rechunksize = "auto"
        chunks = int(np.max([y_max - y_min, x_max - x_min]))
    else:
        rechunksize = chunks
    _chunks = _get_chunks(y_max=y_max, x_max=x_max, y_min=y_min, x_min=x_min, chunksize=chunks)

    _dtype = _get_uint_dtype(shapes[index_name].max())
    if client is not None:
        # if working with a client, we scatter dask shapes, to reduce the size of the dask graph
        shapes = client.scatter(shapes)

    def _process_chunk(tile_bounds, _shapes):
        bbox = box(miny=tile_bounds[0], maxy=tile_bounds[1], minx=tile_bounds[2], maxx=tile_bounds[3])
        gpd_bbox = gpd.GeoDataFrame({"geometry": [bbox]}, crs=_shapes.crs)
        output_shape = (tile_bounds[1] - tile_bounds[0], tile_bounds[3] - tile_bounds[2])  # y,x
        polygons = _shapes.clip(gpd_bbox)

        if polygons.empty:
            output_shape = output_shape
            return np.zeros(shape=output_shape)

        # fix polygons after a .clip is done (i.e. convert GeometryCollection to a Polygon and remove all non Polygon items)
        polygons.geometry = polygons.geometry.map(
            lambda cell: _ensure_polygon_multipolygon(shapely.validation.make_valid(cell))
        )
        polygons = polygons[~polygons.geometry.isna()]
        if polygons.empty:
            output_shape = output_shape
            return np.zeros(shape=output_shape)

        y_min_chunk, y_max_chunk, x_min_chunk, x_max_chunk = tile_bounds
        min_coordinate = [int(y_min_chunk), int(x_min_chunk)]
        max_coordinate = [int(y_max_chunk), int(x_max_chunk)]
        axes = ("y", "x")
        plot_width = x_max_chunk - x_min_chunk
        plot_height = y_max_chunk - y_min_chunk

        y_range = [min_coordinate[axes.index("y")], max_coordinate[axes.index("y")]]
        x_range = [min_coordinate[axes.index("x")], max_coordinate[axes.index("x")]]

        cnv = ds.Canvas(plot_height=plot_height, plot_width=plot_width, x_range=x_range, y_range=y_range)

        agg = cnv.polygons(polygons, "geometry", ds.first(index_name))

        agg = agg.fillna(0)
        arr = agg.data.astype(_dtype)

        return arr

    blocks = []
    for _chunks_inner in _chunks:
        blocks_inner = []
        for _tile_bounds in _chunks_inner:
            output_shape = (_tile_bounds[1] - _tile_bounds[0], _tile_bounds[3] - _tile_bounds[2])
            mask = delayed(_process_chunk)(_tile_bounds, shapes)
            blocks_inner.append(
                da.from_delayed(
                    mask,
                    shape=output_shape,
                    dtype=_dtype,
                )
            )
        blocks.append(blocks_inner)

    arr = da.block(blocks)
    # we choose to pad with zeros, not add a translation,
    # if shapes layer has a (large) offset (y_min!=0 or x_min!=0),
    # it is therefore better, performance wise, to translate the shapes to the origin,
    # and add the offset as a translation to the shapes layer, so the labels layer does not need to be padded
    arr = da.pad(arr, pad_width=((y_min, 0), (x_min, 0)), mode="constant", constant_values=0)
    # rechunk to avoid irregular chunksize after padding
    arr = arr.rechunk(rechunksize)

    sdata = add_labels_layer(
        sdata,
        arr=arr,
        output_layer=output_layer,
        transformations=get_transformation(sdata[shapes_layer], get_all=True),
        scale_factors=scale_factors,
        overwrite=overwrite,
    )

    return sdata


def _get_chunks(y_max: int, x_max: int, y_min: int, x_min: int, chunksize: int):
    assert isinstance(chunksize, int), "Please only provide integer values."
    y_max = int(y_max)
    x_max = int(x_max)
    y_min = int(y_min)
    x_min = int(x_min)

    # Calculate the total range along x and y
    total_x_range = x_max - x_min
    total_y_range = y_max - y_min

    # Calculate the number of chunks along x and y axes
    num_chunks_x = math.ceil(total_x_range / chunksize)
    num_chunks_y = math.ceil(total_y_range / chunksize)

    # Generate the boundaries of each chunk
    chunks = []

    for j in range(num_chunks_y):
        y0 = y_min + j * chunksize
        y1 = min(y0 + chunksize, y_max)
        chunks_inner = []
        for i in range(num_chunks_x):
            x0 = x_min + i * chunksize
            x1 = min(x0 + chunksize, x_max)
            chunks_inner.append([y0, y1, x0, x1])
        chunks.append(chunks_inner)

    return chunks


def _ensure_polygon_multipolygon(cell: Polygon | MultiPolygon | GeometryCollection) -> Polygon | None:
    """
    Ensures that the provided cell becomes a Polygon or MultiPolygon.

    Helper function, because datashader can not work with GeometryCollection.

    Parameters
    ----------
    cell
        A shapely Polygon, MultiPolygon or GeometryCollection.

    Returns
    -------
        The shape as a Polygon or MultiPolygon
    """
    cell = shapely.make_valid(cell)

    if isinstance(cell, (Polygon, MultiPolygon)):
        return cell

    if isinstance(cell, GeometryCollection):
        # We only keep the geometries of type Polygon
        geoms = [geom for geom in cell.geoms if isinstance(geom, Polygon)]

        if not geoms:
            print(f"Removing cell of type {type(cell)} as it contains no Polygon geometry")
            return None

        return MultiPolygon(geoms)  # max(geoms, key=lambda polygon: polygon.area)

    return None
