import dask.array as da
from affine import Affine
from rasterio.features import rasterize
from spatialdata import SpatialData
from spatialdata.models.models import ScaleFactors_t
from spatialdata.transformations import Translation

from sparrow.image._image import _add_label_layer


def add_label_layer_from_shapes_layer(
    sdata: SpatialData,
    shapes_layer: str,
    output_layer: str,
    chunks: str | tuple[int, int] | int | None = None,
    scale_factors: ScaleFactors_t | None = None,
    overwrite: bool = False,
) -> SpatialData:
    """
    Given a shapes layer in a SpatialData object, corresponding masks are created, and added as a labels layer to the SpatialData object.

    Parameters
    ----------
    sdata : SpatialData
        The SpatialData object.
    shapes_layer : str
        The shapes layer to be converted to a labels layer.
    output_layer: str, optional
        Name of the resulting labels layer that will be added to `sdata`.
    chunks : Optional[str | int | tuple[int, ...]], default=None
        If provided, the resulting dask array that contains the masks will be rechunked according to the specified chunk size.
    scale_factors : Optional[ScaleFactors_t], optional
        Scale factors to apply for multiscale.
    overwrite : bool, default=False
        If True, overwrites the `output_layer` if it already exists in `sdata`.

    Returns
    -------
        An updated SpatialData object with the added labels layer.

    Raises
    ------
    ValueError
        If the provided `shapes_layer` contains 3D polygons.
    """
    # only 2D polygons are suported.
    has_z = sdata.shapes[shapes_layer]["geometry"].apply(lambda geom: geom.has_z)
    if any(has_z):
        raise ValueError(
            "Allocating transcripts from a shapes layer is not supported "
            "for shapes layers containing 3D polygons. "
            "Please consider setting 'allocate_from_shapes_layer' to False, "
            "and passing the labels_layer corresponding to the shapes_layer."
        )

    x_min, y_min, x_max, y_max = sdata[shapes_layer].geometry.total_bounds

    transform = Affine.translation(x_min, y_min)

    masks = rasterize(
        zip(
            sdata[shapes_layer].geometry,
            sdata[shapes_layer].index.values.astype(float),
        ),
        out_shape=[int(y_max - y_min), int(x_max - x_min)],
        dtype="uint32",
        fill=0,
        transform=transform,
    )

    if chunks is None:
        chunks = "auto"
    masks = da.from_array(masks, chunks=chunks)

    sdata = _add_label_layer(
        sdata,
        arr=masks,
        output_layer=output_layer,
        chunks=chunks,
        transformation=Translation([x_min, y_min], axes=("x", "y")),
        scale_factors=scale_factors,
        overwrite=overwrite,
    )

    return sdata
