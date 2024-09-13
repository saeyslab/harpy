import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
from spatialdata import SpatialData
from spatialdata.models._utils import MappingToCoordinateSystem_t
from spatialdata.models.models import ScaleFactors_t

from sparrow.image._shapes_to_labels import add_labels_layer_from_shapes_layer
from sparrow.shape._shape import add_shapes_layer


def add_grid_labels_layer(
    sdata,
    shape: tuple[int, int],
    hex_size: int,
    output_shapes_layer: str,
    output_labels_layer: str,
    offset: tuple[int, int] = (0, 0),  # we recommend setting a non-zero offset via a translation
    chunks: str | tuple[int, ...] | int | None = None,
    transformations: MappingToCoordinateSystem_t | None = None,
    scale_factors: ScaleFactors_t | None = None,
    overwrite: bool = True,
) -> SpatialData:
    # add option for 'grid' or 'hexagonal'.
    polygons = _create_hexagon_shapes(shape, hex_size=hex_size, offset=offset)
    # polygons = _create_square_shapes(shape, square_size=hex_size, offset=offset)

    sdata = add_shapes_layer(
        sdata=sdata,
        input=polygons,
        output_layer=output_shapes_layer,
        transformations=transformations,
        overwrite=overwrite,
    )
    sdata = add_labels_layer_from_shapes_layer(
        sdata=sdata,
        shapes_layer=output_shapes_layer,
        output_layer=output_labels_layer,
        out_shape=tuple(a + b for a, b in zip(shape, offset)),
        chunks=chunks,
        scale_factors=scale_factors,
        overwrite=overwrite,
    )
    return sdata


# TODO: fix code duplication
def _create_square_shapes(
    shape: tuple[int, int],  # shape, in y, x
    square_size: int = 10,  # size of the square (side length)
    offset: tuple[int, int] = (0, 0),
) -> gpd.GeoDataFrame:
    assert len(shape) == len(offset) == 2, "currently we only support creating 2D square grid."

    def create_square(cx, cy, a):
        """Creates a square centered at (cx, cy) with side length 'a'."""
        half_size = a / 2
        points = [
            (cx - half_size, cy - half_size),  # bottom-left
            (cx + half_size, cy - half_size),  # bottom-right
            (cx + half_size, cy + half_size),  # top-right
            (cx - half_size, cy + half_size),  # top-left
            (cx - half_size, cy - half_size),  # back to bottom-left to close the square
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
            square = create_square(x, y, square_size)
            squares.append(square)
            x += horizontal_spacing
        y += vertical_spacing

    polygons = gpd.GeoDataFrame(geometry=squares)
    polygons.index = polygons.index + 1  # index ==0 is reserved for background

    return polygons


def _create_hexagon_shapes(
    shape: tuple[int, int],  # shape, in y, x
    hex_size: int = 10,  # size of the hexagon (distance from center to vertex)
    offset: tuple[int, int] = (0, 0),
) -> gpd.GeoDataFrame:
    assert len(shape) == len(offset) == 2, "currently we only support creating 2D hexagonal grid."

    def create_hexagon(cx, cy, a):
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
            hexagon = create_hexagon(x, y, hex_size)
            hexagons.append(hexagon)
            x += horizontal_spacing
        y += vertical_spacing
        row += 1

    polygons = gpd.GeoDataFrame(geometry=hexagons)
    polygons.index = polygons.index + 1  # index ==0 is reserved for background

    return polygons
