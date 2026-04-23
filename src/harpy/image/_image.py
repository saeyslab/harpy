from __future__ import annotations

import dask.array as da
import numpy as np
import xarray as xr
from dask.array import Array
from loguru import logger as log
from spatialdata import SpatialData
from spatialdata.models._utils import MappingToCoordinateSystem_t
from spatialdata.models.models import ScaleFactors_t
from spatialdata.transformations import get_transformation
from spatialdata.transformations.transformations import Affine, Identity, Sequence, Translation
from xarray import DataArray, DataTree

from harpy.image._manager import ImageLayerManager, LabelLayerManager
from harpy.utils._transformations import _get_translation_values


def _substract_translation_crd(
    spatial_image: DataArray,
    crd=tuple[int, int, int, int],
    to_coordinate_system: str = "global",
) -> tuple[int, int, int, int] | None:
    tx, ty = _get_translation(spatial_image, to_coordinate_system=to_coordinate_system)

    _crd = crd
    crd = [
        int(max(0, crd[0] - tx)),
        max(0, int(min(spatial_image.sizes["x"], crd[1] - tx))),
        int(max(0, crd[2] - ty)),
        max(0, int(min(spatial_image.sizes["y"], crd[3] - ty))),
    ]

    if crd[1] - crd[0] <= 0 or crd[3] - crd[2] <= 0:
        log.warning(
            f"Crop param {_crd} after correction for possible translation on "
            f"DataArray object '{spatial_image.name}' is "
            f"'{crd}. Falling back to setting crd to 'None'."
        )
        crd = None

    return crd


def _get_boundary(spatial_image: DataArray, to_coordinate_system: str = "global") -> tuple[int, int, int, int]:
    tx, ty = _get_translation(spatial_image, to_coordinate_system=to_coordinate_system)
    width = spatial_image.sizes["x"]
    height = spatial_image.sizes["y"]
    return (int(tx), int(tx + width), int(ty), int(ty + height))


def _get_translation(spatial_image: DataArray, to_coordinate_system: str = "global") -> tuple[float, float]:
    transformations = get_transformation(spatial_image, get_all=True)
    if to_coordinate_system not in [*transformations]:
        raise ValueError(
            f"Coordinate system '{to_coordinate_system}' does not appear to be a coordinate system of the spatial element. "
            f"Please choose a coordinate system from this list: {[*transformations]}."
        )
    translation = transformations[to_coordinate_system]

    if not isinstance(translation, Sequence | Translation | Identity | Affine):
        raise ValueError(
            f"Currently only transformations of type Translation are supported for coordinate system '{to_coordinate_system}', "
            f"while transformation associated with {spatial_image} in coordinate system '{to_coordinate_system}' is of type {type(translation)}."
        )

    return _get_translation_values(translation)


def _precondition(
    sdata, image_name: str, labels_name: str, to_coordinate_system: str = "global"
) -> tuple[DataArray, DataArray]:
    """Helper function that gets highest resolution `image_name` and `labels_name`, and checks that `image_name` and `labels_name` are co-registered."""
    if image_name not in sdata.images:
        raise ValueError(
            f"image layer with name '{image_name}' not found in sdata.images. Please choose from: {[*sdata.images]}."
        )
    if labels_name not in sdata.labels:
        raise ValueError(
            f"labels layer with name '{labels_name}' not found in sdata.labels. Please choose from: {[*sdata.labels]}."
        )
    se_image = _get_spatial_element(sdata, element_name=image_name)
    se_labels = _get_spatial_element(sdata, element_name=labels_name)

    if se_image.data.shape[1:] != se_labels.data.shape:
        raise ValueError(
            "Only arrays with same spatial shape are currently supported, "
            f"but image layer with name {image_name} has shape {se_image.data.shape}, "
            f"while labels layer with name {labels_name} has shape {se_labels.data.shape}."
        )

    t1x, t1y = _get_translation(se_image, to_coordinate_system=to_coordinate_system)
    t2x, t2y = _get_translation(se_labels, to_coordinate_system=to_coordinate_system)

    if (t1x, t1y) != (t2x, t2y):
        raise ValueError(
            f"image layer with name {image_name} should "
            f"be registered to labels layer with name {labels_name} in coordinate system {to_coordinate_system}."
        )
    return se_image, se_labels


def _apply_transform(se: DataArray, to_coordinate_system: str = "global") -> tuple[DataArray, np.ndarray, np.ndarray]:
    """
    Apply the translation (if any) of the given DataArray to its x- and y-coordinates array.

    The new DataArray is returned, as well as the original coordinates array.
    This function is used because some plotting functions ignore the DataArray transformation
    matrix, but do use the coordinates arrays for absolute positioning of the image in the plot.
    After plotting the coordinates can be restored with _unapply_transform().
    """
    # Get current coords
    x_orig_coords = se.x.data
    y_orig_coords = se.y.data

    # Translate
    tx, ty = _get_translation(se, to_coordinate_system=to_coordinate_system)
    x_coords = xr.DataArray(tx + np.arange(se.sizes["x"], dtype="float64"), dims="x")
    y_coords = xr.DataArray(ty + np.arange(se.sizes["y"], dtype="float64"), dims="y")
    se = se.assign_coords({"x": x_coords, "y": y_coords})
    # QUESTION: should we set the resulting DataArray's transformation matrix to the
    # identity matrix too, for consistency? If so we have to keep track of it too for restoring later.

    return se, x_orig_coords, y_orig_coords


def _unapply_transform(se: DataArray, x_coords: np.ndarray, y_coords: np.ndarray) -> DataArray:
    """Restore the coordinates which were temporarily modified via _apply_transform()."""
    se = se.assign_coords({"y": y_coords, "x": x_coords})
    return se


def get_dataarray(sdata: SpatialData, element_name: str, scale: str | None = None) -> DataArray:
    """
    Retrieve the highest-resolution :class:`xarray.DataArray` from an element in ``sdata.images`` or ``sdata.labels``.

    If ``sdata.images[element_name]`` or ``sdata.labels[element_name]`` is a
    :class:`xarray.DataTree`, this function returns the requested pyramid
    :class:`xarray.DataArray`. If ``scale`` is ``None``, the base scale
    (``"scale0"``) is returned. If the element is already a :class:`xarray.DataArray`,
    ``scale`` is ignored.

    Parameters
    ----------
    sdata : SpatialData
        The SpatialData object containing image and/or labels elements.
    element_name : str
        The name of the element to retrieve.
    scale : str | None
        Pyramid level to retrieve when the element is stored as a multiscale
        :class:`xarray.DataTree`. If ``None``, ``"scale0"`` is used.

    Returns
    -------
    The resolved :class:`xarray.DataArray` corresponding to the requested element.

    Raises
    ------
    KeyError
        If the element does not exist in ``sdata.images`` or ``sdata.labels``.
    ValueError
        If the element exists but is stored in an unsupported format.
    """
    if element_name in sdata.images:
        si = sdata.images[element_name]
    elif element_name in sdata.labels:
        si = sdata.labels[element_name]
    else:
        raise KeyError(f"'{element_name}' not found in 'sdata.images' or 'sdata.labels'")

    if isinstance(si, DataArray):
        return si
    if isinstance(si, DataTree):
        scale_key = "scale0" if scale is None else scale
        if scale_key not in si:
            raise ValueError(
                f"Scale '{scale_key}' not found in element '{element_name}'. Available scales are: {[*si]}."
            )
        name = si[scale_key].__iter__().__next__()
        return si[scale_key][name]
    raise ValueError(f"Not implemented for element '{element_name}' of type {type(si)}.")


def _get_spatial_element(sdata: SpatialData, element_name: str) -> DataArray:
    if element_name in sdata.images:
        si = sdata.images[element_name]
    elif element_name in sdata.labels:
        si = sdata.labels[element_name]
    else:
        raise KeyError(f"'{element_name}' not found in 'sdata.images' or 'sdata.labels'")
    if isinstance(si, DataArray):
        return si
    elif isinstance(si, DataTree):
        # get the name of the unscaled image
        scale_0 = si.__iter__().__next__()
        name = si[scale_0].__iter__().__next__()
        return si[scale_0][name]
    else:
        raise ValueError(f"Not implemented for element '{element_name}' of type {type(si)}.")


def _fix_dimensions(
    array: np.ndarray | da.Array,
    dims: list[str] = None,
    target_dims: list[str] = None,
) -> np.ndarray | da.Array:
    if target_dims is None:
        target_dims = ["c", "z", "y", "x"]
    if dims is None:
        dims = ["c", "z", "y", "x"]
    dims = list(dims)
    target_dims = list(target_dims)

    dims_set = set(dims)
    if len(dims) != len(dims_set):
        raise ValueError(f"dims list {dims} contains duplicates")

    target_dims_set = set(target_dims)

    extra_dims = dims_set - target_dims_set

    if extra_dims:
        raise ValueError(f"The dimension(s) {extra_dims} are not present in the target dimensions.")

    # check if the array already has the correct number of dimensions, if not add missing dimensions
    if len(array.shape) != len(dims):
        raise ValueError(f"Dimension of array {array.shape} is not equal to dimension of provided dims {dims}")
    if len(array.shape) > 4:
        raise ValueError(f"Arrays with dimension larger than 4 are not supported, shape of array is {array.shape}")

    for dim in target_dims:
        if dim not in dims:
            array = array[None, ...]
            dims.insert(0, dim)

    # create a mapping from input dims to target dims
    dim_mapping = {dim: i for i, dim in enumerate(dims)}
    target_order = [dim_mapping[dim] for dim in target_dims]

    # transpose the array to the target dimension order
    array = array.transpose(*target_order)

    return array


def add_image(
    sdata: SpatialData,
    arr: Array,
    output_image_name: str,
    dims: tuple[str, ...] | None = None,
    chunks: str | tuple[int, ...] | int | None = None,
    transformations: MappingToCoordinateSystem_t | None = None,
    scale_factors: ScaleFactors_t | None = None,
    c_coords: list[str] | None = None,
    overwrite: bool = False,
) -> SpatialData:
    """
    Add an image element to a SpatialData object.

    This function allows you to add an image element to `sdata`.
    If `sdata` is backed by a zarr store, the resulting image element will be backed to the zarr store, otherwise `arr` will be persisted in memory.
    All layers of the Dask graph associated with `arr` will therefore be materialized upon calling `add_image`.

    Parameters
    ----------
    sdata
        The SpatialData object to which the new image element will be added.
    arr
        The array containing the image data to be added.
    output_image_name
        The name of the output image element where the image data will be stored.
    dims
        A tuple specifying the dimensions of the image data (e.g., ("c", "z", "y", "x")). If None, defaults will be inferred.
    chunks
        Specification for chunking the data.
    transformations
        Transformations that will be added to resulting `output_image_name`.
    scale_factors
        Scale factors to apply for multiscale data. If specified `output_image_name` will be multiscale.
    c_coords
        Names of the channels. If None, channel names will be named sequentially as 0,1,...
    overwrite
        If True, overwrites `output_image_name` if it already exists in `sdata`.

    Returns
    -------
    The `sdata` object with the image element added.
    """
    manager = ImageLayerManager()
    sdata = manager.add_layer(
        sdata,
        arr=arr,
        element_name=output_image_name,
        dims=dims,
        chunks=chunks,
        transformations=transformations,
        scale_factors=scale_factors,
        c_coords=c_coords,
        overwrite=overwrite,
    )

    return sdata


def add_labels(
    sdata: SpatialData,
    arr: Array,
    output_labels_name: str,
    dims: tuple[str, ...] | None = None,
    chunks: str | tuple[int, ...] | int | None = None,
    transformations: MappingToCoordinateSystem_t | None = None,
    scale_factors: ScaleFactors_t | None = None,
    overwrite: bool = False,
) -> SpatialData:
    """
    Add a labels element to a SpatialData object.

    This function allows you to add a labels element to `sdata`.
    If `sdata` is backed by a zarr store, the resulting labels element will be backed to the zarr store, otherwise `arr` will be persisted in memory.
    All layers of the Dask graph associated with `arr` will therefore be materialized upon calling `add_labels`.

    Parameters
    ----------
    sdata
        The SpatialData object to which the new labels element will be added.
    arr
        The array containing the labels data to be added. Should be of type int.
    output_labels_name
        The name of the output labels element where the labels data will be stored.
    dims
        A tuple specifying the dimensions of the labels data (e.g., (""z", "y", "x")). If None, defaults will be inferred.
    chunks
        Specification for chunking the data.
    transformations
        Transformations that will be added to resulting `output_labels_name`.
    scale_factors
        Scale factors to apply for multiscale data. If specified `output_labels_name` will be multiscale
    overwrite
        If True, overwrites the `output_labels_name` if it already exists in `sdata`.

    Returns
    -------
    The `sdata` object with the labels element added.
    """
    manager = LabelLayerManager()
    sdata = manager.add_layer(
        sdata,
        arr=arr,
        element_name=output_labels_name,
        dims=dims,
        chunks=chunks,
        transformations=transformations,
        scale_factors=scale_factors,
        overwrite=overwrite,
    )

    return sdata
