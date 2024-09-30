import math

import dask
import dask.array as da
import dask_geopandas as dgpd
import geopandas as gpd
import numpy as np
from affine import Affine
from rasterio.features import rasterize
from shapely.geometry import box
from spatialdata import SpatialData
from spatialdata.models.models import ScaleFactors_t
from spatialdata.transformations import get_transformation

from sparrow.image._image import add_labels_layer
from sparrow.utils.utils import _get_uint_dtype


def add_labels_layer_from_shapes_layer(
    sdata: SpatialData,
    shapes_layer: str,
    output_layer: str,
    out_shape: tuple[int, int] | None = None,  # output shape in y, x.
    chunks: int | None = None,
    chunksize_shapes: int = 100000,
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
        output shape of the resulting labels layer `(y,x)`. Will be automatically calculated if set to None.
        If `out_shape` is not `None`, with  `x_min, y_min, x_max, y_max = sdata[shapes_layer].geometry.total_bounds`,
        and `out_shape[1]<x_max` or `out_shape[0]<y_max`, then shapes with coordinates outside `out_shapes` will
        not be in resulting `output_layer`.
    chunks
        If provided, creation of the labels layer will be done in a chunked manner, with data divided into chunks for efficient computation.
    chunksize_shapes
        Passed to `chunksize` parameter of `geopandas.from_geopandas`, when loading `shapes_layer` in a Dask dataframe.
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

    _, _, x_max, y_max = sdata[shapes_layer].geometry.total_bounds

    assert (
        x_max > 0 and y_max > 0
    ), f"The maximum of the bounding box of the shapes layer {shapes_layer} is negative. This is not allowed."
    index = sdata[shapes_layer].index.values.astype(int)

    if out_shape is not None:
        y_max = out_shape[0]
        x_max = out_shape[1]

    if chunks is None:
        chunks = int(np.max([y_max, x_max]))
    _chunks = _get_chunks(y_max=y_max, x_max=x_max, chunksize=chunks)

    dask_shapes = dgpd.from_geopandas(sdata.shapes[shapes_layer], chunksize=chunksize_shapes)
    _dtype = _get_uint_dtype(index.max())

    @dask.delayed
    def _process_chunk(tile_bounds, polygons):
        output_shape = (tile_bounds[1] - tile_bounds[0], tile_bounds[3] - tile_bounds[2])
        if polygons.empty:
            output_shape = output_shape
            return np.zeros(shape=output_shape)

        transform = Affine.translation(xoff=tile_bounds[2], yoff=tile_bounds[0])

        # TODO. Test datashader to do this. probably faster.
        masks = rasterize(
            zip(
                polygons.geometry,
                polygons.index.astype(int),  # take index of the polygons
            ),
            out_shape=output_shape,  # y,x
            dtype=_dtype,
            transform=transform,
            fill=0,
        )

        return masks

    blocks = []
    for _chunks_inner in _chunks:
        blocks_inner = []
        for _tile_bounds in _chunks_inner:
            bbox = box(miny=_tile_bounds[0], maxy=_tile_bounds[1], minx=_tile_bounds[2], maxx=_tile_bounds[3])
            gpd_bbox = gpd.GeoDataFrame({"geometry": [bbox]}, crs=dask_shapes.crs)
            output_shape = (_tile_bounds[1] - _tile_bounds[0], _tile_bounds[3] - _tile_bounds[2])
            # take a subset of the polygons
            polygons = dask_shapes.clip(gpd_bbox)
            mask = _process_chunk(_tile_bounds, polygons)
            blocks_inner.append(
                da.from_delayed(
                    mask,
                    shape=output_shape,
                    dtype=_dtype,
                )
            )
        blocks.append(blocks_inner)

    sdata = add_labels_layer(
        sdata,
        arr=da.block(blocks),
        output_layer=output_layer,
        transformations=get_transformation(sdata[shapes_layer], get_all=True),
        scale_factors=scale_factors,
        overwrite=overwrite,
    )

    return sdata


def _get_chunks(y_max: int, x_max: int, chunksize: int):
    y_max = int(y_max)
    x_max = int(x_max)

    y_min = 0
    x_min = 0

    # Calculate the total range along x and y
    total_x_range = x_max
    total_y_range = y_max

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
