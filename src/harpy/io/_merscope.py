from __future__ import annotations

import importlib
import os
import re
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import anndata as ad
import dask.array as da
import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
from spatialdata import SpatialData, read_zarr
from spatialdata.models import Image2DModel, Image3DModel
from spatialdata.models._utils import get_axes_names
from spatialdata.transformations import Affine, Identity, set_transformation
from spatialdata_io import merscope as sdata_merscope
from spatialdata_io._constants._constants import MerscopeKeys

from harpy.image._image import _get_spatial_element
from harpy.image._rasterize import rasterize
from harpy.io._transcripts import read_transcripts
from harpy.shape import add_shapes_layer
from harpy.table._table import add_table_layer
from harpy.utils._keys import _CELL_INDEX, _INSTANCE_KEY, _REGION_KEY, _SPATIAL
from harpy.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

# the column in MerscopeKeys.BOUNDARIES_FILE holding cell ID. This should be a unique ID.
MerscopeKeys_CELL_ID = "ID"


def merscope(
    path: str | Path | list[str] | list[Path],
    to_coordinate_system: str | list[str] = "global",
    z_layers: int | list[int] | None = 3,
    backend: Literal["dask_image", "rioxarray"] | None = None,
    transcripts: bool = True,
    cell_boundaries: bool = True,
    rasterize_cell_boundaries: bool = True,
    table: bool = True,
    mosaic_images: bool = True,
    do_3D: bool = False,
    z_projection: bool = False,
    imread_kwargs: Mapping[str, Any] = MappingProxyType({}),
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
    filter_gene_names: str | list[str] = None,
    instance_key: str = _INSTANCE_KEY,
    region_key: str = _REGION_KEY,
    spatial_key: str = _SPATIAL,
    cell_index_name: str = _CELL_INDEX,
    output: str | Path | None = None,
) -> SpatialData:
    """
    Read *MERSCOPE* data from Vizgen.

    A wrapper around :func:`spatialdata_io.merscope` that adds some additional capabilities:
    (i) loading data in full 3D (z, y, x) or applying a z-projection,
    (ii) rasterizing cell boundaries,
    (iii) adding a micron-based coordinate system, and
    (iv) loading multiple samples into a single SpatialData object.

    The micron coordinate system is added as '{to_coordinate_system}_micron' and is available to all spatial elements within the resulting SpatialData object.

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
        Either `"dask_image"` or `"rioxarray"` (the latter uses less RAM, but requires :mod:`rioxarray` to be installed). By default, uses `"rioxarray"` if and only if the :mod:`rioxarray` library is installed.
    transcripts
        Whether to read transcripts.
    cell_boundaries
        Whether to read cell boundaries (polygons).
    rasterize_cell_boundaries
        Whether to rasterize the cell boundaries (i.e. create a labels layer from polygons). We use :func:`harpy.im.rasterize` to rasterize the cell boundaries.
        Ignored if `cell_boundaries` is `False`, or if `mosaic_images` is `False`.
    table
        Whether to read in the :class:`~anndata.AnnData` table. The table will be annotated by a labels layer.
        If `table` is set to `True` then `cell_boundaries`, `rasterize_cell_boundaries` and `mosaic_images` must also be set to `True`.
    mosaic_images
        Whether to read the mosaic images.
    do_3D
        Read the mosaic images and the transcripts as 3D.
    z_projection
        Perform a z projection (maximum intensity along the z-stacks) on z-stacks of mosaic images. Ignored if `mosaic_images` is `False`.
    imread_kwargs
        Keyword arguments to pass to the image reader. Ignored if `mosaic_images` is `False`.
    image_models_kwargs
        Keyword arguments to pass to the image models. Ignored if `mosaic_images` is `False`.
    filter_gene_names
        Gene names that need to be filtered out (via `str.contains`) from the resulting points layer (transcripts), mostly control genes that were added, and which you don't want to use.
        Filtering is case insensitive. Also see :func:`harpy.read_transcripts`. Ignored if `transcripts` is `False`.
    instance_key
        Instance key. The name of the column in :class:`~anndata.AnnData` table `.obs` that will hold the instance ids.
        Ignored if `table` is `False`.
    region_key
        Region key. The name of the column in  :class:`~anndata.AnnData` table `.obs` that will hold the name of the elements that annotate the table.
        Ignored if `table` is `False`.
    spatial_key
        The key in the :class:`~anndata.AnnData` table `.obsm` that will hold the `x` and `y` center of the instances.
        Ignored if `table` is `False`.
    cell_index_name
        The name of the index of the resulting :class:`~anndata.AnnData` table. Ignored if `table` is `False`.
    output
        The path where the resulting `SpatialData` object will be backed. If `None`, it will not be backed to a zarr store.

    Raises
    ------
    AssertionError
        Raised when the number of elements in `path` and `to_coordinate_system` are not the same.
    AssertionError
        If elements in `to_coordinate_system` are not unique.
    ValueError
        If both `do_3D` and `z_projection` are set to `True`.
    ValueError
        If `table` is `True`, and `rasterize_cell_boundaries` or `mosaic_images` is not `True`.
    ValueError
        If `table` is `True` and `cell_boundaries` is `False`.

    Returns
    -------
    A SpatialData object.

    See Also
    --------
    harpy.io.read_transcripts : read transcripts.
    harpy.im.rasterize: rasterize cell boundaries.
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

    if table:
        if not rasterize_cell_boundaries or not mosaic_images:
            raise ValueError(
                "Please set both 'rasterize_cell_boundaries' and 'mosaic_images' to 'True' when setting 'table' to 'True'."
            )
        if not cell_boundaries:
            raise ValueError("Please set 'cell_boundaries' to 'True' when 'table' is 'True'.")

    def _fix_name(item: str | Iterable[str]):
        return list(item) if isinstance(item, Iterable) and not isinstance(item, str) else [item]

    if z_layers is None:
        log.info("Parameter 'z_layers' not specified, defaults to '3'.")
        z_layers = 3

    z_layers = _fix_name(z_layers)
    path = _fix_name(path)
    to_coordinate_system = _fix_name(to_coordinate_system)
    assert len(path) == len(to_coordinate_system), (
        "If parameters 'path' and/or 'to_coordinate_system' are specified as a list, their length should be equal."
    )
    assert len(to_coordinate_system) == len(set(to_coordinate_system)), (
        "All elements specified via 'to_coordinate_system' should be unique."
    )

    for _path, _to_coordinate_system in zip(path, to_coordinate_system, strict=True):
        # merscope provides an affine transformation between micron and pixels (MerscopeKeys.TRANSFORMATION_FILE)
        # so we inverse this, to get transformation from pixels_to_micron to get a micron coordinate system.
        pixels_to_micron = Affine(
            np.genfromtxt(os.path.join(_path, MerscopeKeys.IMAGES_DIR, MerscopeKeys.TRANSFORMATION_FILE)),
            input_axes=("x", "y"),
            output_axes=("x", "y"),
        ).inverse()

        _path = Path(_path).absolute()
        vizgen_region = _path.name
        slide_name = _path.parent.name
        dataset_id = f"{slide_name}_{vizgen_region}"
        sdata = sdata_merscope(
            path=_path,
            z_layers=z_layers,
            region_name=None,
            slide_name=None,
            backend=backend,
            transcripts=False,  # we have our own reader for transcripts
            cells_boundaries=False,  # we have our own reader for cell boundaries (+ rasterization)
            cells_table=False,
            vpt_outputs=None,  # we do not use vpt
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

        layers = [*sdata.images]
        log.info(f"Adding micron coordinate system for mosaic images: '{_to_coordinate_system}_micron'.")
        transformations = {_to_coordinate_system: Identity(), f"{_to_coordinate_system}_micron": pixels_to_micron}
        for _layer in layers:
            # rename coordinate system "global" to _to_coordinate_system
            # _to_coordinate_system is the pixel coordinate system -> intrinsic coordinate system
            # _to_coordinate_system_micron is the micron coordinate system
            set_transformation(sdata[_layer], transformation=transformations, set_all=True)
            sdata[f"{_layer}_{_to_coordinate_system}"] = sdata[_layer]
            del sdata[_layer]

    if output is not None:
        sdata.write(output)
        sdata = read_zarr(output)

    if cell_boundaries:
        for _path, _to_coordinate_system in zip(path, to_coordinate_system, strict=True):
            gdf = gpd.read_parquet(os.path.join(_path, MerscopeKeys.BOUNDARIES_FILE))
            # currently all z-stacks of gdf are the same (merscope does not provide us with real 3D segmentation).
            # therefore we take the middle z_layer in z_layers instead of iterating over z_layers
            # if merscope would provide us with real 3D, we should iterate over z_layers, and construct a 3D labels layer if do_3D is True.
            z_layers_to_iterate = z_layers if len(z_layers) == 1 else [z_layers[len(z_layers) // 2]]
            for _z_layer in z_layers_to_iterate:
                sdata, _output_shapes_layer = _add_shapes(
                    sdata,
                    gdf=gdf,
                    path=_path,
                    z_layer=_z_layer,
                    to_coordinate_system=_to_coordinate_system,
                    to_micron_coordinate_system=f"{to_coordinate_system}_micron",
                    dataset_id=dataset_id,
                    instance_key=instance_key,
                )
                # if not mosaic_images, we cannot guess output shape, so we do not rasterize
                if not mosaic_images and rasterize_cell_boundaries:
                    log.warning(
                        "'cell_boundaries' was set to 'True', while 'mosaic_images' was set to 'False', "
                        "cell boundaries will therefore not be rasterized."
                    )
                if mosaic_images and rasterize_cell_boundaries:
                    _mosaic_layer = [
                        *sdata.filter_by_coordinate_system(coordinate_system=_to_coordinate_system).images
                    ][0]
                    se = _get_spatial_element(sdata, layer=_mosaic_layer)
                    out_shape = se.data.shape[-2:]
                    _chunks = se.data.chunksize[-1]
                    if "scale_factors" in image_models_kwargs:
                        scale_factors = image_models_kwargs["scale_factors"]
                    else:
                        scale_factors = [2, 2, 2, 2]
                    _output_labels_layer = f"{dataset_id}_z{_z_layer}_{_to_coordinate_system}_labels"
                    log.info(f"Saving cell masks for z stack '{_z_layer}' as '{_output_labels_layer}'.")
                    log.info(
                        f"Adding micron coordinate system for cell masks: '{to_coordinate_system}_micron'."
                    )  # coordinate system of shapes is copied to labels via hp.im.rasterize
                    sdata = rasterize(
                        sdata,
                        shapes_layer=_output_shapes_layer,
                        output_layer=_output_labels_layer,
                        out_shape=out_shape,
                        chunks=_chunks,
                        scale_factors=scale_factors,
                        overwrite=False,
                    )
            assert len(z_layers_to_iterate) == 1, "Currently masks provided by merscope are not 3D."
            if table:
                count_path = os.path.join(_path, MerscopeKeys.COUNTS_FILE)
                obs_path = os.path.join(_path, MerscopeKeys.CELL_METADATA_FILE)

                data = pd.read_csv(count_path, index_col=0, dtype={MerscopeKeys.COUNTS_CELL_KEY: str})
                obs = pd.read_csv(obs_path, index_col=0, dtype={MerscopeKeys.METADATA_CELL_KEY: str})

                is_gene = ~data.columns.str.lower().str.contains("blank")  # exclude blank gene from adata.X
                adata = ad.AnnData(data.loc[:, is_gene], dtype=data.values.dtype, obs=obs)
                adata.obs.reset_index(inplace=True)

                shapes = sdata.shapes[_output_shapes_layer]
                # the index of shapes is instance_key
                shapes.reset_index(inplace=True)

                shapes[MerscopeKeys.METADATA_CELL_KEY] = shapes[MerscopeKeys.METADATA_CELL_KEY].astype(str)

                # Sanity checks before merging shapes and adata.obs. Note, we do not care that some cells in shapes would not be found in adata.obs

                # 1) check that all cells in adata.obs could be matched
                missing = set(adata.obs[MerscopeKeys.METADATA_CELL_KEY]) - set(shapes[MerscopeKeys.METADATA_CELL_KEY])
                if missing:
                    raise ValueError(
                        f"{len(missing)} {MerscopeKeys.METADATA_CELL_KEY} values in .obs not found in cell boundaries (e.g. {list(missing)[:5]})"
                    )

                # 2) check that there are no duplicates in shapes. If there would be duplicates in shapes, then rows in adata.obs would be multiplied, we want to catch this early
                duplicates = shapes[shapes.duplicated(subset=MerscopeKeys.METADATA_CELL_KEY, keep=False)]
                if not duplicates.empty:
                    raise ValueError(
                        f"Duplicate {MerscopeKeys.METADATA_CELL_KEY} values found in polygons (e.g. {duplicates[MerscopeKeys.METADATA_CELL_KEY].unique()[:5]})"
                    )
                # now merge
                adata.obs = pd.merge(
                    adata.obs,
                    shapes[
                        [MerscopeKeys.METADATA_CELL_KEY, MerscopeKeys.Z_INDEX, instance_key]
                    ],  # we merge, because we want the instance_key in adata.obs, so we can annotate by labels layer
                    how="inner",
                    on=[MerscopeKeys.METADATA_CELL_KEY],
                )

                # set adata.obs.index in same way as we set it in hp.tb.allocate
                _uuid_value = str(uuid.uuid4())[:8]
                adata.obs.index = adata.obs.index.map(
                    lambda x, layer=_output_labels_layer, uid=_uuid_value: f"{str(x)}_{layer}_{uid}"
                )
                adata.obs.index.name = cell_index_name

                adata.obs[region_key] = pd.Series(
                    _output_labels_layer, index=adata.obs_names, dtype="category"
                )  # we annotate with the labels layer

                data.index = adata.obs.index
                data.index.name = cell_index_name
                adata.obsm["blank"] = data.loc[
                    :, ~is_gene
                ]  #  genes with 'blank' in the name are excluded from adata.X, so we add them to .obsm
                adata.obsm[spatial_key] = adata.obs[[MerscopeKeys.CELL_X, MerscopeKeys.CELL_Y]].values

                sdata = add_table_layer(
                    sdata,
                    adata=adata,
                    output_layer=f"{dataset_id}_{_to_coordinate_system}_table",
                    region=[_output_labels_layer],
                    instance_key=instance_key,
                    region_key=region_key,
                    overwrite=False,
                )

    if transcripts:
        for _path, _to_coordinate_system in zip(path, to_coordinate_system, strict=True):
            # read the table to get the metadata
            table = dd.read_csv(os.path.join(_path, MerscopeKeys.TRANSCRIPTS_FILE), header=0)

            column_x_name = MerscopeKeys.GLOBAL_X
            column_y_name = MerscopeKeys.GLOBAL_Y
            column_z_name = MerscopeKeys.GLOBAL_Z
            column_gene_name = MerscopeKeys.GENE_KEY

            column_x = table.columns.get_loc(column_x_name)
            column_y = table.columns.get_loc(column_y_name)
            column_z = table.columns.get_loc(column_z_name)
            column_gene = table.columns.get_loc(column_gene_name)

            output_layer = f"{dataset_id}_{_to_coordinate_system}_points"
            log.info(f"Saving transcripts as {output_layer}")
            sdata = read_transcripts(
                sdata,
                path_count_matrix=os.path.join(_path, MerscopeKeys.TRANSCRIPTS_FILE),
                transform_matrix=os.path.join(_path, MerscopeKeys.IMAGES_DIR, MerscopeKeys.TRANSFORMATION_FILE),
                column_x=column_x,
                column_y=column_y,
                column_z=column_z if do_3D else None,
                column_gene=column_gene,
                header=0,
                output_layer=output_layer,
                to_coordinate_system=_to_coordinate_system,
                to_micron_coordinate_system=f"{to_coordinate_system}_micron",
                filter_gene_names=filter_gene_names,
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


def _add_shapes(
    sdata: SpatialData,
    gdf: gpd.GeoDataFrame,
    path: str | Path,
    z_layer: int,
    to_coordinate_system: str,
    to_micron_coordinate_system: str,
    dataset_id: str,
    instance_key: str,
) -> tuple[SpatialData, str]:
    from shapely import MultiPolygon
    from shapely.affinity import affine_transform

    # NOTE: currently, the gdf.geometry of merscope output is the same for all z-stacks.
    # for future compatibility, we add seperate mask for each z stack.
    gdf = gdf[gdf[MerscopeKeys.Z_INDEX] == z_layer].rename(columns={MerscopeKeys_CELL_ID: instance_key})

    gdf = gdf.rename_geometry("geometry")
    gdf = gdf[gdf.geometry.is_valid]
    gdf.geometry = gdf.geometry.map(lambda x: MultiPolygon(x.geoms))
    gdf[instance_key] = gdf[instance_key].astype(int)
    if not gdf[instance_key].is_unique:
        raise ValueError(
            f"Column '{MerscopeKeys_CELL_ID}' of '{MerscopeKeys.BOUNDARIES_FILE}' of dataset with id '{dataset_id}' contains duplicates. "
            "Please report this bug."
        )
    gdf.set_index(instance_key, inplace=True)

    # transformation from mircon to pixels
    # apply this to the cell boundaries
    affine_matrix = np.genfromtxt(os.path.join(path, MerscopeKeys.IMAGES_DIR, MerscopeKeys.TRANSFORMATION_FILE))

    # and save the inverse
    pixels_to_micron = Affine(
        affine_matrix,
        input_axes=("x", "y"),
        output_axes=("x", "y"),
    ).inverse()

    # transform to pixel space
    params = [
        affine_matrix[0, 0],  # a
        affine_matrix[0, 1],  # b
        affine_matrix[1, 0],  # d
        affine_matrix[1, 1],  # e
        affine_matrix[0, 2],  # xoff
        affine_matrix[1, 2],  # yoff
    ]

    gdf.geometry = gdf.geometry.apply(lambda geom: affine_transform(geom, params))

    output_layer = f"{dataset_id}_z{z_layer}_{to_coordinate_system}_shapes"

    log.info(f"Saving cell boundaries for z stack '{z_layer}' as '{output_layer}'.")
    log.info(f"Adding micron coordinate system for cell boundaries: '{to_micron_coordinate_system}'.")
    sdata = add_shapes_layer(
        sdata,
        input=gdf,
        output_layer=output_layer,
        transformations={
            to_coordinate_system: Identity(),
            f"{to_micron_coordinate_system}": pixels_to_micron,
        },
        overwrite=False,
    )

    return sdata, output_layer
