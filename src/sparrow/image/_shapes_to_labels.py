import dask.array as da
import numpy as np
from rasterio.features import rasterize
from spatialdata import SpatialData
from spatialdata.models.models import ScaleFactors_t
from spatialdata.transformations import get_transformation

from sparrow.image._image import add_labels_layer


def add_labels_layer_from_shapes_layer(
    sdata: SpatialData,
    shapes_layer: str,
    output_layer: str,
    chunks: str | tuple[int, int] | int | None = None,
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
    chunks
        If provided, the resulting dask array that contains the masks will be rechunked according to the specified chunk size.
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

    x_min, y_min, x_max, y_max = sdata[shapes_layer].geometry.total_bounds

    assert (
        x_max > 0 and y_max > 0
    ), f"The maximum of the bounding box of the shapes layer {shapes_layer} is negative. This is not allowed."
    index = sdata[shapes_layer].index.values.astype(int)

    if y_min > 0:
        y_shape = int((y_max - y_min) + y_min)
    else:
        y_shape = int(y_max)
    if x_min > 0:
        x_shape = int((x_max - x_min) + x_min)
    else:
        x_shape = int(x_max)

    out_shape = [y_shape, x_shape]

    _dtype = _get_dtype(index.max())

    masks = rasterize(
        zip(
            sdata[shapes_layer].geometry,
            index,
        ),
        out_shape=out_shape,
        dtype=_dtype,
        fill=0,
    )

    if chunks is None:
        chunks = "auto"
    masks = da.from_array(masks, chunks=chunks)

    sdata = add_labels_layer(
        sdata,
        arr=masks,
        output_layer=output_layer,
        chunks=chunks,
        transformations=get_transformation(sdata[shapes_layer], get_all=True),
        scale_factors=scale_factors,
        overwrite=overwrite,
    )

    return sdata


def _get_dtype(value: int) -> str:
    max_uint64 = np.iinfo(np.uint64).max
    max_uint32 = np.iinfo(np.uint32).max
    max_uint16 = np.iinfo(np.uint16).max

    if max_uint16 >= value:
        dtype = "uint16"
    elif max_uint32 >= value:
        dtype = "uint32"
    elif max_uint64 >= value:
        dtype = "uint64"
    else:
        raise ValueError(f"Maximum cell number is {value}. Values higher than {max_uint64} are not supported.")
    return dtype
