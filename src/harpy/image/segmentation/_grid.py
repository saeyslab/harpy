from __future__ import annotations

import geopandas as gpd
import numpy as np
from dask.distributed import Client
from shapely import box
from shapely.geometry import Polygon
from spatialdata import SpatialData
from spatialdata.models._utils import MappingToCoordinateSystem_t
from spatialdata.models.models import ScaleFactors_t

from harpy.image._rasterize import rasterize
from harpy.shape._shape import add_shapes_layer


def add_grid_labels_layer(
    sdata,
    shape: tuple[int, int],  # shape of the resulting labels layer, shape y, x
    size: int,  # radius of the hexagon, or size length of the square.
    output_shapes_layer: str,  # shapes layer corresponding to the labels layer
    output_labels_layer: str,
    grid_type: str = "hexagon",  # can be either "hexagon" or "square".
    offset: tuple[int, int] = (0, 0),  # we recommend setting a non-zero offset via a translation.
    chunks: int | None = None,
    client: Client | None = None,
    transformations: MappingToCoordinateSystem_t | None = None,
    scale_factors: ScaleFactors_t | None = None,
    overwrite: bool = False,
) -> SpatialData:
    """
    Adds a grid-based labels layer to the SpatialData object using either a hexagonal or square grid.

    The function creates a corresponding shapes layer based on the specified grid type and parameters.

    Parameters
    ----------
    sdata
        The SpatialData object to which the new grid-based labels layer and shapes layer will be added.
    shape
        The (y, x) shape of the resulting labels layer. This defines the grid's size in terms of height (y) and width (x).
    size
        The size of the grid cells. For a hexagonal grid, this is the radius of the hexagons; for a square grid, this is the side length of the squares.
    output_shapes_layer
        The name of the shapes layer that corresponds to the generated grid. This layer will contain the polygons representing the grid's shapes.
    output_labels_layer
        The name of the labels layer that corresponds to the generated grid. This layer will contain the labels generated from the shapes.
    grid_type
        The type of grid to create. Can be either `"hexagon"` for a hexagonal grid or `"square"` for a square grid. The default is `"hexagon"`.
    offset
        An optional translation offset applied to the grid. This is a tuple `(y_offset, x_offset)` that can shift the grid. Default is `(0, 0)`,
        but it is recommended to use a zero offset, and specify the offset via passing a `spatialdata.transformations.Translation` to `transformations`.
    chunks
        Specifies the chunk size for Dask arrays when calculating the labels layer.
    client
        A Dask `Client` instance, which will be passed to 'harpy.im.rasterize' (function which rasterizes the generated `output_shapes_layer`) if specified.
        Refer to the 'harpy.im.rasterize' docstring for further details.
    transformations
        Transformations that will be added to the resulting `output_shapes_layer` and `output_labels_layer`.
    scale_factors
        Scale factors to apply for multiscale. Only applies to `output_labels_layer`.
    overwrite
        If True, overwrites the `output_shapes_layer` and `output_labels_layer` if it already exists in `sdata`.

    Returns
    -------
    The updated SpatialData object with the newly added grid-based shapes and labels layers.

    Raises
    ------
    ValueError
        If an unsupported grid type is specified. The valid options are `"hexagon"` or `"square"`.

    Notes
    -----
    The function first generates a grid of shapes (either hexagons or squares) based on the specified grid type and parameters. These shapes are added as
    a new shapes layer in `sdata`. Then, a corresponding labels layer is generated from the shapes layer. The labels layer has the same spatial
    dimensions as specified in `shape`.
    """
    grid_type_supported = ["hexagon", "square"]
    if grid_type not in grid_type_supported:
        raise ValueError(f"Invalid shape type: '{grid_type}'. Please choose from the list: {grid_type_supported}.")
    if grid_type == "hexagon":
        polygons = _create_hexagon_shapes(shape, hex_size=size, offset=offset)
    if grid_type == "square":
        polygons = _create_square_shapes(shape, square_size=size, offset=offset)

    sdata = add_shapes_layer(
        sdata=sdata,
        input=polygons,
        output_layer=output_shapes_layer,
        transformations=transformations,
        overwrite=overwrite,
    )
    sdata = rasterize(
        sdata=sdata,
        shapes_layer=output_shapes_layer,
        output_layer=output_labels_layer,
        out_shape=tuple(a + b for a, b in zip(shape, offset, strict=True)),
        chunks=chunks,
        client=client,
        scale_factors=scale_factors,
        overwrite=overwrite,
    )
    return sdata


def _create_square_shapes(
    shape: tuple[int, int],  # shape, in y, x
    square_size: int = 10,  # size of the square (side length)
    offset: tuple[int, int] = (0, 0),
) -> gpd.GeoDataFrame:
    assert len(shape) == len(offset) == 2, "currently we only support creating 2D square grid."

    def _create_square(cx, cy, a):
        """Creates a square centered at (cx, cy) with side length 'a'."""
        half_size = a / 2
        points = [
            (cx - half_size, cy - half_size),
            (cx + half_size, cy - half_size),
            (cx + half_size, cy + half_size),
            (cx - half_size, cy + half_size),
            (cx - half_size, cy - half_size),
        ]
        return Polygon(points)

    min_x, min_y, max_x, max_y = offset[1], offset[0], shape[1] + offset[1], shape[0] + offset[0]

    squares = []
    square_height = square_size  # y
    square_width = square_size  # x

    vertical_spacing = square_height  # y-spacing between square centers
    horizontal_spacing = square_width  # x-spacing between square centers

    # Calculate the boundaries for placing the square centers within the grid
    min_x_center = min_x + square_width / 2
    min_y_center = min_y + square_height / 2
    max_x_center = max_x - square_width / 2
    max_y_center = max_y - square_height / 2

    y = min_y_center
    while y <= max_y_center:
        x = min_x_center
        while x <= max_x_center:
            square = _create_square(x, y, square_size)
            squares.append(square)
            x += horizontal_spacing
        y += vertical_spacing

    polygons = gpd.GeoDataFrame(geometry=squares)
    polygons.index = polygons.index + 1  # index ==0 is reserved for background

    return polygons


def _create_square_shapes_vectorize(
    shape: tuple[int, int],
    square_size: int = 10,
    offset: tuple[int, int] = (0, 0),
) -> gpd.GeoDataFrame:
    assert len(shape) == len(offset) == 2, "Only 2D square grids are supported."
    # slightly faster than create_square_shapes

    min_x, min_y = offset[1], offset[0]
    max_x, max_y = shape[1] + offset[1], shape[0] + offset[0]

    eps = 1e-05  # to include max_x - square / 2 if square_size fits perfectly in shape
    x_centers = np.arange(min_x + square_size / 2, max_x - square_size / 2 + eps, square_size)
    y_centers = np.arange(min_y + square_size / 2, max_y - square_size / 2 + eps, square_size)
    x_grid, y_grid = np.meshgrid(x_centers, y_centers)

    half_size = square_size / 2
    minx = x_grid - half_size
    maxx = x_grid + half_size
    miny = y_grid - half_size
    maxy = y_grid + half_size

    squares = box(minx, miny, maxx, maxy)

    polygons = gpd.GeoDataFrame(geometry=squares.ravel())
    polygons.index += 1  # index ==0 is reserved for background
    return polygons


def _create_hexagon_shapes(
    shape: tuple[int, int],  # shape, in y, x
    hex_size: int = 10,  # size of the hexagon (distance from center to vertex)
    offset: tuple[int, int] = (0, 0),
) -> gpd.GeoDataFrame:
    assert len(shape) == len(offset) == 2, "currently we only support creating 2D hexagonal grid."

    def _create_hexagon(cx, cy, a):
        """Creates a regular hexagon centered at (cx, cy) with size 'a'."""
        angles = np.linspace(0, 2 * np.pi, 7)
        points = [(cx + a * np.sin(angle), cy + a * np.cos(angle)) for angle in angles]
        return Polygon(points)

    min_x, min_y, max_x, max_y = offset[1], offset[0], shape[1] + offset[1], shape[0] + offset[0]

    hexagons = []
    hex_height = 2 * hex_size  # y
    hex_width = np.sqrt(3) * hex_size  # x

    vertical_spacing = (
        3 / 2 * hex_size
    )  # y-spacing between hex centers, due hex needing to fit in each other (via offset in x every other row of hexagons) not equal to hex_height
    horizontal_spacing = hex_width  # x-spacing between hex centers, equal to hex_width

    # we only want full hexagons's, so set max and min of centers so they fit in given shape
    min_x_center = min_x + hex_width / 2
    min_y_center = min_y + hex_height / 2
    max_x_center = max_x - hex_width / 2
    max_y_center = max_y - hex_height / 2

    row = 0
    y = min_y_center
    while y <= max_y_center:
        x_offset = (row % 2) * (horizontal_spacing / 2)
        x = min_x_center + x_offset
        while x <= max_x_center:
            hexagon = _create_hexagon(x, y, hex_size)
            hexagons.append(hexagon)
            x += horizontal_spacing
        y += vertical_spacing
        row += 1

    polygons = gpd.GeoDataFrame(geometry=hexagons)
    polygons.index = polygons.index + 1  # index ==0 is reserved for background

    return polygons
