from __future__ import annotations

import importlib
import os
import re
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Literal, Mapping

import dask.array as da
import dask.dataframe as dd
from spatialdata import SpatialData, read_zarr
from spatialdata.models import Image2DModel, Image3DModel
from spatialdata.models._utils import get_axes_names
from spatialdata.transformations import get_transformation, set_transformation
from spatialdata_io import merscope as sdata_merscope
from spatialdata_io._constants._constants import MerscopeKeys

from sparrow.image._image import _get_spatial_element
from sparrow.io._transcripts import read_transcripts
from sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def merscope(
    path: str | Path | list[str] | list[Path],
    to_coordinate_system: str | list[str] = "global",
    z_layers: int | list[int] | None = 3,
    backend: Literal["dask_image", "rioxarray"] | None = None,
    transcripts: bool = True,
    mosaic_images: bool = True,
    do_3D: bool = False,
    z_projection: bool = False,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
    output: str | Path | None = None,
) -> SpatialData:
    """
    Read *MERSCOPE* data from Vizgen.

    Wrapper around `spatialdata_io.merscope`, but with support to read in the data as 3D `(z,y,x)` or perform a z-projection,
    and with support for reading multiple samples into one spatialdata object
    This function reads the following files:

        - ``{ms.TRANSCRIPTS_FILE!r}``: Transcript file.
        - `mosaic_**_z*.tif` images inside the ``{ms.IMAGES_DIR!r}`` directory.

    Parameters
    ----------
    path
        Path to the region/root directory containing the *Merscope* files (e.g., `detected_transcripts.csv`).
        This can either be a single path or a list of paths, where each path corresponds to a different experiment/roi.
    to_coordinate_system
        The coordinate system to which the elements will be added for each item in path.
        If provided as a list, its length should be equal to the number of paths specified in `path`.
    z_layers
        Indices of the z-layers to consider. Either one `int` index, or a list of `int` indices. If `None`, then no image is loaded.
        By default, only the middle layer is considered (that is, layer 3).
    backend
        Either `"dask_image"` or `"rioxarray"` (the latter uses less RAM, but requires `rioxarray` to be installed). By default, uses `"rioxarray"` if and only if the `rioxarray` library is installed.
    transcripts
        Whether to read transcripts.
    mosaic_images
        Whether to read the mosaic images.
    do_3D
        Read the mosaic images and the transcripts as 3D.
    z_projection
        Perform a z projection (maximum intensity along the z-stacks) on z-stacks of mosaic images.
    imread_kwargs
        Keyword arguments to pass to the image reader.
    image_models_kwargs
        Keyword arguments to pass to the image models.
    output
        The path where the resulting `SpatialData` object will be backed. If `None`, it will not be backed to a zarr store.

    Raises
    ------
    AssertionError
        Raised when the number of elements in `path` and `to_coordinate_system` are not the same.
    AssertionError
        If elements in `to_coordinate_system` are not unique.
    ValueError
        If both `do_3D` as `z_projection` are set to `True`.

    Returns
    -------
    A SpatialData object.
    """
    if mosaic_images:
        if backend is None:
            if not importlib.util.find_spec("rioxarray"):
                log.info(
                    "'backend' was set to None and 'rioxarray' library is not installed, "
                    "we will fall back to using 'dask_image' library to read in the images, "
                    "which will result in high RAM usage. Please consider installing the "
                    "'rioxarray' library."
                )
        elif backend == "dask_image":
            log.info(
                "'backend' was set to 'dask_image'. Please consider installing the "
                "'rioxarray' library and setting 'backend' to 'rioxarray' to reduce "
                "RAM usage when reading the images."
            )

        elif backend == "rioxarray":
            if not importlib.util.find_spec("rioxarray"):
                raise ValueError("'backend' was set to 'rioxarray' but 'rioxarray' is not installed.")

    def _fix_name(item: str | Iterable[str]):
        return list(item) if isinstance(item, Iterable) and not isinstance(item, str) else [item]

    z_layers = _fix_name(z_layers)
    path = _fix_name(path)
    to_coordinate_system = _fix_name(to_coordinate_system)
    assert len(path) == len(
        to_coordinate_system
    ), "If parameters 'path' and/or 'to_coordinate_system' are specified as a list, their length should be equal."
    assert len(to_coordinate_system) == len(
        set(to_coordinate_system)
    ), "All elements specified via 'to_coordinate_system' should be unique."

    for _path, _to_coordinate_system in zip(path, to_coordinate_system):
        sdata = sdata_merscope(
            path=_path,
            z_layers=z_layers,
            region_name=None,
            slide_name=None,
            backend=backend,
            transcripts=False,  # we have our own reader for transcripts
            cells_boundaries=False,
            cells_table=False,
            vpt_outputs=None,
            mosaic_images=mosaic_images,
            imread_kwargs=imread_kwargs,
            image_models_kwargs=image_models_kwargs,
        )

        if mosaic_images:
            if do_3D or z_projection:
                first_image_name = [*sdata.images][0]
                root_image_name = _get_root_image_name(first_image_name)

                c_coords = _get_spatial_element(sdata, first_image_name).c.data

                dims = get_axes_names(_get_spatial_element(sdata, first_image_name))

                arr = da.stack(
                    [_get_spatial_element(sdata, layer=f"{root_image_name}{z_layer}").data for z_layer in z_layers],
                    axis=1,
                )

                if do_3D and z_projection:
                    raise ValueError(
                        "The options 'do_3D' and 'z_projection' cannot both be enabled at the same time. Please choose one."
                    )

                if do_3D:
                    if "dims" in image_models_kwargs:
                        del image_models_kwargs["dims"]
                    if "chunks" in image_models_kwargs:
                        del image_models_kwargs["chunks"]  # already chunked in sdata_merscope step.
                    dims = list(dims)
                    dims.insert(1, "z")
                    dims_3D = tuple(dims)
                    sdata[root_image_name] = Image3DModel.parse(
                        arr, dims=dims_3D, c_coords=c_coords, **image_models_kwargs
                    )
                elif z_projection:
                    arr = da.max(arr, axis=1)
                    sdata[root_image_name] = Image2DModel.parse(
                        arr, dims=dims, c_coords=c_coords, **image_models_kwargs
                    )

                # delete the indivual z stacks.
                for z_layer in z_layers:
                    del sdata[f"{root_image_name}{z_layer}"]

        layers = [*sdata.images] + [*sdata.labels]

        for _layer in layers:
            # rename coordinate system "global" to _to_coordinate_system
            transformation = {_to_coordinate_system: get_transformation(sdata[_layer], to_coordinate_system="global")}
            set_transformation(sdata[_layer], transformation=transformation, set_all=True)
            sdata[f"{_layer}_{_to_coordinate_system}"] = sdata[_layer]
            del sdata[_layer]

    if output is not None:
        sdata.write(output)
        sdata = read_zarr(output)

    if transcripts:
        for _path, _to_coordinate_system in zip(path, to_coordinate_system):
            # read the table to get the metadata
            table = dd.read_csv(os.path.join(_path, MerscopeKeys.TRANSCRIPTS_FILE), header=0)

            column_x_name = MerscopeKeys.GLOBAL_X
            column_y_name = MerscopeKeys.GLOBAL_Y
            column_z_name = MerscopeKeys.GLOBAL_Z
            column_gene_name = "gene"  # TODO get this from MerscopeKeys, see merged PR on spatialdata-io

            column_x = table.columns.get_loc(column_x_name)
            column_y = table.columns.get_loc(column_y_name)
            column_z = table.columns.get_loc(column_z_name)
            column_gene = table.columns.get_loc(column_gene_name)

            sdata = read_transcripts(
                sdata,
                path_count_matrix=os.path.join(_path, MerscopeKeys.TRANSCRIPTS_FILE),
                transform_matrix=os.path.join(_path, MerscopeKeys.IMAGES_DIR, MerscopeKeys.TRANSFORMATION_FILE),
                column_x=column_x,
                column_y=column_y,
                column_z=column_z if do_3D else None,
                column_gene=column_gene,
                header=0,
                output_layer=f"transcripts_{_to_coordinate_system}",
                to_coordinate_system=_to_coordinate_system,
                overwrite=False,
            )

    return sdata


def _get_root_image_name(name: str) -> str:
    # Regular expression to extract the name
    match = re.match(r"^(.*?)[0-9]+$", name)

    # If a match is found, extract the first capturing group
    if match:
        name_no_trailing_number = match.group(1)
        return name_no_trailing_number
    return None
