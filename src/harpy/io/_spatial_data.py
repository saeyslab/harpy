from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

import dask.array as da
import numpy as np
import spatialdata
from dask.array import from_zarr
from dask_image import imread
from spatialdata import SpatialData
from spatialdata.models.models import ScaleFactors_t
from spatialdata.transformations import Identity, Translation

from harpy.image._image import _fix_dimensions, add_image_layer
from harpy.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def create_sdata(
    input: str | Path | np.ndarray | da.Array | list[str] | list[Path] | list[np.ndarray] | list[da.Array],
    output_path: str | Path | None = None,
    img_layer: str = "raw_image",
    chunks: str | tuple[int, int, int, int] | int | None = None,
    dims: list[str] | None = None,
    crd: tuple[int, int, int, int] | None = None,
    to_coordinate_system: str = "global",
    scale_factors: ScaleFactors_t | None = None,
    c_coords: int | str | Iterable[int | str] | None = None,
    z_projection: bool = True,
) -> SpatialData:
    """
    Convert input images or arrays into a SpatialData object with the image added as an image layer with name `img_layer`.

    This function allows you to ingest various input formats of images or data arrays,
    convert them into a unified SpatialData format and write them out to a specified
    path (zarr store) if needed. It provides flexibility in how you define the source data as well
    as certain processing options like chunking.

    The input parameter can be formatted in four ways:

    - 1. Path to a single image, either grayscale or multiple channels.

    Examples
    --------
      input='DAPI_z3.tif' -> single channel

      input='DAPI_Poly_z3.tif' -> multi (DAPI, Poly) channel

    - 2. Pattern representing a collection of z-stacks
      (if this is the case, a z-projection is performed which selects the maximum intensity value across the z-dimension).

    Examples
    --------
      input='DAPI_z*.tif' -> z-projection performed

      input='DAPI_Poly_z*.tif' -> z-projection performed

    - 3. List of filename patterns (where each list item corresponds to a different channel)

    Examples
    --------
      input=[ 'DAPI_z3.tif', 'Poly_z3.tif' ] -> multi (DAPI, Poly) channel

      input=[ 'DAPI_z*.tif', 'Poly_z*.tif' ] -> multi (DAPI, Poly) channel, z projection performed

    - 4. Single numpy or dask array.
      The dims parameter should specify its dimension, e.g. ['y','x'] for a 2D array or
      [ 'c', 'y', 'x', 'z' ] for a 4D array with channels. If a z-dimension >1 is present, a z-projection is performed.

    Parameters
    ----------
    input
        The filename pattern, path or list of filename patterns to the images that
        should be loaded. In case of a list, each list item should represent a different
        channel, and each image corresponding to a filename pattern should represent.
        a different z-stack.
        Input can also be a numpy array. In that case the dims parameter should be specified.
    output_path
        If specified, the resulting SpatialData object will be written to this path as a zarr.
    img_layer
        The name of the image layer to be created in the SpatialData object.
    chunks
        If specified, the underlying dask array will be rechunked to this size.
        If Tuple, desired chunksize along c,z,y,x should be specified, e.g. (1,1,1024,1024).
    dims
        The dimensions of the input data if it's a numpy array. E.g., ['y','x'] or ['c','y','x','z'].
        If input is a str, Path or List[str], List[Path], this parameter is ignored.
    crd
        The coordinates for a region of interest in the format (xmin, xmax, ymin, ymax).
        If specified, this region is cropped from the image, and added as image layer to the
        SpatialData object.
    to_coordinate_system
        Coordinate system to which `img_layer` will be added.
    scale_factors
        Scale factors to apply for multiscale.
    c_coords
        Names of the channels in the input image. If None, channel names will be named sequentially as 0,1,...
    z_projection
        Whether to perform a z projection (maximum intensity projection along the z-axis).

    Returns
    -------
    The constructed SpatialData object containing image layer with name `img_layer` and dimension (c,(z),y,x)

    Notes
    -----
    If `crd` is specified and some of its values are None, the function infers the missing
    values based on the input image's shape.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        # tmp_dir is used to write results for each channel out to separate zarr store,
        # these results are then combined and written to sdata
        dask_array = _load_image_to_dask(
            input=input,
            chunks=chunks,
            dims=dims,
            crd=crd,
            z_projection=z_projection,
            output_dir=tmp_dir,
        )

        if c_coords is not None:
            c_coords = (
                list(c_coords) if isinstance(c_coords, Iterable) and not isinstance(c_coords, str) else [c_coords]
            )

            assert dask_array.shape[0] == len(c_coords), (
                "Length of c_coords should match number of channels provided, "
                f"while provided c_coords is '{c_coords}' with len '{len( c_coords ) }', "
                f"and number of channels read from input is '{dask_array.shape[0]}'"
            )

            if len(set(c_coords)) != len(c_coords):
                raise ValueError("Each value in c_coords must be unique.")

        sdata = spatialdata.SpatialData()

        # make sure sdata is backed.
        if output_path is not None:
            sdata.write(output_path)

        if crd is not None:
            # replace None in crd with x-y bounds of dask_array
            crd = _fix_crd(crd, dask_array.shape[-2:])
            tx = crd[0]
            ty = crd[2]

            transformation = Translation([tx, ty], axes=("x", "y"))

        else:
            transformation = Identity()

        sdata = add_image_layer(
            sdata,
            arr=dask_array,
            output_layer=img_layer,
            chunks=chunks,
            transformations={to_coordinate_system: transformation},
            scale_factors=scale_factors,
            c_coords=c_coords,
            overwrite=False,
        )

    return sdata


def _load_image_to_dask(
    input: str | Path | np.ndarray | da.Array | list[str] | list[Path] | list[np.ndarray] | list[da.Array],
    chunks: str | tuple[int, int, int, int] | int | None = None,
    dims: list[str] | None = None,
    crd: list[int] | None = None,
    z_projection: bool = True,
    output_dir: Path | str | None = None,
) -> da.Array:
    """
    Load images into a dask array.

    This function facilitates the loading of one or more images into a 3 or 4D dask array depending whether
    z-projection is set to True or False..
    These images are designated by a provided filename pattern or numpy array. The resulting dask
    array will have three dimensions structured as follows: c, (z) y, and x.

    Parameters
    ----------
    input : Union[str, Path, List[Union[str, Path]]]
        The filename pattern, path or list of filename patterns to the images that
        should be loaded. In case of a list, each list item should represent a different
        channel, and each image corresponding to a filename pattern should represent.
        a different z-stack.
    chunks : Optional[int], default=None
        If specified, the underlying dask array will be rechunked to this size.
    dims : Optional[List[str]], default=None
        The dimensions of the input data if it's a numpy array. E.g., ['y','x'] or ['c','y','x','z'].
        If input is a str of Path, this parameter is ignored.
    crd : tuple of int, optional
        The coordinates for a region of interest in the format (xmin, xmax, ymin, ymax).
    z_projection
        Whether to perform a z projection (maximum intensity projection).
    output_dir: if input is a str, Path, or List[str], List[Path], the intermediate results for each channel
        will be written to zarr store in output_dir to prevent unnecessary memory usage.

    Returns
    -------
    dask.array.Array
        The resulting 3D dask array with dimensions ordered as: (c, (z), y, x).
    """
    if isinstance(input, list):
        # if filename pattern is a list, create (c, (z), y, x) for each filename pattern
        dask_arrays = [
            _load_image_to_dask(f, chunks, dims, z_projection=z_projection, output_dir=output_dir) for f in input
        ]

        dask_array = da.concatenate(dask_arrays, axis=0)

        if z_projection:
            # add z- dimension if we did a projection, we want (c,z,y,x)
            dask_array = dask_array[:, None, :, :]

    elif isinstance(input, np.ndarray | da.Array):
        # make sure we have (c,z,y,x)
        array = _fix_dimensions(input, dims=dims)
        if isinstance(array, np.ndarray):
            dask_array = da.from_array(array)
        else:
            dask_array = array

    elif isinstance(input, str | Path):
        if dims is not None:
            log.warning(f"dims parameter is equal to {dims}, but will be ignored when reading in images from a file")
        # make sure we have (c,z,y,x)
        dask_array = imread.imread(input)
        if len(dask_array.shape) == 4:
            # dask_image puts channel dim last
            dims = ["z", "y", "x", "c"]
            dask_array = _fix_dimensions(dask_array, dims=dims)
        elif len(dask_array.shape) == 3:
            # dask_image does not add channel dim for grayscale images
            dims = ["z", "y", "x"]
            dask_array = _fix_dimensions(dask_array, dims=dims)
        elif len(dask_array.shape) == 2:
            dims = ["y", "x"]
            dask_array = _fix_dimensions(dask_array, dims=dims)
        else:
            raise ValueError(f"Image has shape { dask_array.shape }, while (y, x) is required.")
    else:
        raise ValueError(f"input of type {type(input)} not supported.")

    if crd:
        # c,z,y,x
        # get slice of it:
        dask_array = dask_array[:, :, crd[2] : crd[3], crd[0] : crd[1]]

    if chunks:
        dask_array = dask_array.rechunk(chunks)

    if z_projection:
        # perform z-projection
        if dask_array.shape[1] > 1:
            dask_array = da.max(dask_array, axis=1)
        # if z-dimension is 1, then squeeze it.
        else:
            dask_array = dask_array.squeeze(1)

    if isinstance(input, str | Path) and output_dir is not None:
        name = os.path.splitext(os.path.basename(input))[0]
        output_zarr = os.path.join(output_dir, f"{name}.zarr")
        dask_array.to_zarr(output_zarr)
        dask_array = from_zarr(output_zarr)

    return dask_array


def _fix_crd(
    crd: tuple[int, int, int, int],
    shape_y_x=tuple[int, int],
) -> tuple[int, int, int, int]:
    x1, x2, y1, y2 = crd

    max_y, max_x = shape_y_x

    # Update the coordinates based on the conditions
    x1 = 0 if x1 is None else x1
    y1 = 0 if y1 is None else y1
    x2 = max_x if x2 is None else x2
    y2 = max_y if y2 is None else y2

    return (x1, x2, y1, y2)
