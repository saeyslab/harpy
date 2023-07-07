""" This file holds all the general functions that are used to build up the pipeline and the notebooks. The functions are odered by their occurence in the pipeline from top to bottom."""

# %load_ext autoreload
# %autoreload 2
import os
import warnings
from collections import namedtuple
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2

os.environ["USE_PYGEOS"] = "0"
import itertools

import dask
import dask.array as da
import dask.dataframe as dd
import dask_image.ndfilters
import geopandas
import matplotlib
import matplotlib.colors as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import scanpy as sc
import scipy as sp
import seaborn as sns
import shapely
import spatialdata
import squidpy as sq
import torch
import xarray as xr
import zarr
from affine import Affine
from aicsimageio import AICSImage
from anndata import AnnData
from basicpy import BaSiC
from cellpose import models
from dask import compute, delayed
from dask.array import Array
from dask.dataframe.core import DataFrame as DaskDataFrame
from dask_image import imread
from longsgis import voronoiDiagram4plg
from multiscale_spatial_image import to_multiscale
from numcodecs import Blosc
from pandas import DataFrame as PandasDataFrame
from rasterio import features
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from scipy.ndimage.filters import gaussian_filter
from shapely.affinity import translate
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.geometry.polygon import LinearRing
from spatial_image import to_spatial_image
from spatialdata import SpatialData


def create_sdata(
    filename_pattern: Union[str, Path, List[Union[str, Path]]],
    output_path: str | Path,
    layer_name="raw_image",
    chunks: Optional[int] = None,
    crd=None
):
    dask_array = load_image_to_dask(filename_pattern=filename_pattern, chunks=chunks)

    # make sure we have ( c, z, y, x )
    if len(dask_array.shape) == 4:
        # put channel dimension first (dask image puts channel dim last)
        dask_array = dask_array.transpose(3, 0, 1, 2)
    elif len(dask_array.shape) == 3:
        dask_array = dask_array[None, :, :, :]
    else:
        raise ValueError(
            f"Image has dimension { dask_array.shape }, while (c, z, y, x) is required."
        )

    if chunks:
        dask_array = dask_array.rechunk(chunks)

    # perform z-projection
    if dask_array.shape[1] > 1:
        dask_array = da.max(dask_array, axis=1)

    # if z-dimension is 1, then squeeze it.
    else:
        dask_array = dask_array.squeeze(1)
    if crd:
        dask_array=dask_array[:,crd[0]:crd[1],crd[2]:crd[3]]
    sdata = spatialdata.SpatialData()

    spatial_image = spatialdata.models.Image2DModel.parse(
        dask_array, dims=("c", "y", "x")
    )

    y_coords = xr.DataArray(np.arange(dask_array.shape[1], dtype="float64"), dims="y")
    x_coords = xr.DataArray(np.arange(dask_array.shape[2], dtype="float64"), dims="x")

    # If bug in spatialdata is fixed where coordinates are lost when saving to zarr, coordinates should be set here.
    # spatial_image=spatial_image.assign_coords({ 'y': y_coords, 'x': x_coords} )
    sdata.add_image(name=layer_name, image=spatial_image)

    sdata.write(output_path)

    # writing to zarr loses coordinates
    sdata.images[layer_name] = sdata[layer_name].assign_coords(
        {"y": y_coords, "x": x_coords}
    )

    return sdata


def load_image_to_dask(
    filename_pattern: Union[str, Path, List[Union[str, Path]]],
    chunks: Optional[int] = None,
) -> da.Array:
    """
    Load images into a dask array.

    This function facilitates the loading of one or more images into a 3D dask array.
    These images are designated by a provided filename pattern. The resulting dask
    array will have three dimensions structured as follows: channel (c), y, and x.

    The filename pattern parameter can be formatted in three ways:
    - A path to a single image, either grayscale or multiple channels.
        Examples:
        DAPI_z3.tif -> single channel
        DAPI_Poly_z3.tif -> multi (DAPI, Poly) channel
    - A pattern representing a collection of z-stacks (if this is the case, a z-projection
    is performed which selects the maximum intensity value across the z-dimension). 
        Examples:
        DAPI_z*.tif -> z-projection performed
        DAPI_Poly_z*.tif -> z-projection performed
    - A list of filename patterns (where each list item corresponds to a different channel)
        Examples
        [ DAPI_z3.tif, Poly_z3.tif ] -> multi (DAPI, Poly) channel
        [ DAPI_z*.tif, Poly_z*.tif ] -> multi (DAPI, Poly) channel, z projection performed

    Parameters
    ----------
    filename_pattern : Union[str, Path, List[Union[str, Path]]]
        The filename pattern, path or list of filename patterns to the images that
        should be loaded. In case of a list, each list item should represent a different
        channel, and each image corresponding to a filename pattern should represent
        a different z-stack.
    chunks : int, optional
        Chunk size for rechunking the resulting dask array. If not provided (None),
        the array will not be rechunked.

    Returns
    -------
    dask.array.Array
        The resulting 3D dask array with dimensions ordered as: (c, y, x).

    Raises
    ------
    ValueError
        If an image is not 3D (z,y,x), a ValueError is raised. z-dimension can be 1.
    """

    if isinstance(filename_pattern, list):
        # if filename pattern is a list, create (c, y, x) for each filename pattern
        dask_arrays = [load_image_to_dask(f, chunks) for f in filename_pattern]
        dask_array = da.concatenate(dask_arrays, axis=0)
        # add z- dimension, we want (c,z,y,x)
        dask_array = dask_array[:, None, :, :]
    else:
        dask_array = imread.imread(filename_pattern)
        # put channel dimension first (dask image puts channel dim last)
        if len(dask_array.shape) == 4:
            dask_array = dask_array.transpose(3, 0, 1, 2)
        # make sure we have ( c, z, y, x )
        # dask image does not add channel dim for images with channel dim==1, so add it here
        elif len(dask_array.shape) == 3:
            dask_array = dask_array[None, :, :, :]
        elif len(dask_array.shape) < 3:
            raise ValueError(
                f"Image has dimension { dask_array.shape }, while (z, y, x) is required."
            )

    if chunks:
        dask_array = dask_array.rechunk(chunks)

    # perform z-projection
    if dask_array.shape[1] > 1:
        dask_array = da.max(dask_array, axis=1)
        print(dask_array)
    # if z-dimension is 1, then squeeze it.
    else:
        dask_array = dask_array.squeeze(1)

    return dask_array


def tilingCorrection(
    sdata: SpatialData,
    tile_size: int = 2144,
    crop_param: Optional[Tuple[int, int, int]] = None,
    output_layer: str = "tiling_correction",
) -> Tuple[SpatialData, List[np.ndarray]]:
    """Returns the corrected image and the flatfield array

    This function corrects for the tiling effect that occurs in some image data for example the resolve dataset.
    The illumination within the tiles is adjusted, afterwards the tiles are connected as a whole image by inpainting the lines between the tiles.
    """

    layer = [*sdata.images][-1]

    result_list=[]
    flatfields=[]

    for channel in sdata[ layer ].c.data:

        ic = sq.im.ImageContainer(sdata[layer].isel(c=channel), layer=layer)

        x_coords = ic.data.x.data
        y_coords = ic.data.y.data

        # Create the tiles
        tiles = ic.generate_equal_crops(size=tile_size, as_array=layer)
        tiles = np.array([tile + 1 if ~np.any(tile) else tile for tile in tiles])
        black = np.array([1 if ~np.any(tile - 1) else 0 for tile in tiles])  # determine if

        # create the masks for inpainting
        i_mask = (
            np.block(
                [
                    list(tiles[i : i + (ic.shape[1] // tile_size)])
                    for i in range(0, len(tiles), ic.shape[1] // tile_size)
                ]
            ).astype(np.uint16)
            == 0
        )
        if tiles.shape[0] < 5:
            print(
                "There aren't enough tiles to perform tiling correction (less than 5). This step will be skipped."
            )
            tiles_corrected = tiles
            flatfields.append( None )
        else:
            basic = BaSiC(smoothness_flatfield=1)
            basic.fit(tiles)
            flatfields.append( basic.flatfield )
            tiles_corrected = basic.transform(tiles)

        tiles_corrected = np.array(
            [
                tiles[number] if black[number] == 1 else tile
                for number, tile in enumerate(tiles_corrected)
            ]
        )

        # Stitch the tiles back together
        i_new = np.block(
            [
                list(tiles_corrected[i : i + (ic.shape[1] // tile_size)])
                for i in range(0, len(tiles_corrected), ic.shape[1] // tile_size)
            ]
        ).astype(np.uint16)

        ic = sq.im.ImageContainer(i_new, layer=layer)
        ic.add_img(
            i_mask.astype(np.uint8),
            layer="mask_black_lines",
        )

        ic[layer] = ic[layer].assign_coords({"y": y_coords, "x": x_coords})

        if crop_param:
            ic = ic.crop_corner(y=crop_param[1], x=crop_param[0], size=crop_param[2])

            x_coords = ic.data.x.data
            y_coords = ic.data.y.data

        # Perform inpainting
        ic.apply(
            {"0": cv2.inpaint},
            layer=layer,
            drop=False,
            channel=0,
            new_layer=output_layer,
            copy=False,
            # chunks=10,
            fn_kwargs={
                "inpaintMask": ic.data.mask_black_lines.squeeze().to_numpy(),
                "inpaintRadius": 55,
                "flags": cv2.INPAINT_NS,
            },
        )

        # result for each channel
        result_list.append( ic[ output_layer ].data )

    # make one dask array of shape (c,y,x)
    result = da.concatenate(result_list, axis=-1).transpose(3, 0, 1, 2).squeeze(-1)

    spatial_image = spatialdata.models.Image2DModel.parse(result, dims=("c", "y", "x"))

    # if bug is fixed in spatialdata, you should set coordinates here, not after adding image to sdata
    spatial_image = spatial_image.assign_coords({"y": y_coords, "x": x_coords})

    # during adding of image it is written to zarr store
    sdata.add_image(name=output_layer, image=spatial_image)

    # add coordinates, due to bug these are lost when writing to zarr store
    sdata.images[output_layer] = sdata[output_layer].assign_coords(
        {"y": y_coords, "x": x_coords}
    )

    return sdata, flatfields


def tilingCorrectionPlot(
    img: np.ndarray, flatfield: np.ndarray, img_orig: np.ndarray, output: str = None
) -> None:
    """Creates the plots based on the correction overlay and the original and corrected images."""

    # disable interactive mode
    # if output:
    #    plt.ioff()

    # Tile correction overlay
    if flatfield is not None:
        fig1, ax1 = plt.subplots(1, 1, figsize=(20, 10))
        ax1.imshow(flatfield[0], cmap="gray")# TODO: fix for multiple ones 
        ax1.set_title("Correction performed per tile")

        # Save the plot to ouput
        if output:
            plt.close(fig1)
            fig1.savefig(output + "0.png")

    # Original and corrected image
    fig2, ax2 = plt.subplots(1, 2, figsize=(20, 10))
    ax2[0].imshow(img, cmap="gray")
    ax2[0].set_title("Corrected image")
    ax2[1].imshow(img_orig, cmap="gray")
    ax2[1].set_title("Original image")
    ax2[0].spines["top"].set_visible(False)
    ax2[0].spines["right"].set_visible(False)
    ax2[0].spines["bottom"].set_visible(False)
    ax2[0].spines["left"].set_visible(False)
    ax2[1].spines["top"].set_visible(False)
    ax2[1].spines["right"].set_visible(False)
    ax2[1].spines["bottom"].set_visible(False)
    ax2[1].spines["left"].set_visible(False)

    # Save the plot to ouput
    if output:
        plt.close(fig2)
        fig2.savefig(output + "1.png")


def tophat_filtering(
    sdata: SpatialData,
    output_layer="tophat_filtered",
    size_tophat: int = 85,
) -> SpatialData:
    # this is function to do tophat filtering using dask

    # take the last image as layer to do next step in pipeline
    layer = [*sdata.images][-1]

    # TODO size_tophat maybe different for different channels, probably fix this with size_tophat as a list.
    # Initialize list to store results
    result_list = []

    for channel in sdata[layer].c.data:
        image_array = sdata[layer].isel(c=channel).data

        # Apply the minimum filter
        minimum_t = dask_image.ndfilters.minimum_filter(image_array, size_tophat)

        # Apply the maximum filter
        max_of_min_t = dask_image.ndfilters.maximum_filter(minimum_t, size_tophat)

        # Subtract max_of_min_t from image_array and store in result_list
        result_list.append(image_array - max_of_min_t)

    result = da.stack(result_list, axis=0)

    spatial_image = spatialdata.models.Image2DModel.parse(result, dims=("c", "y", "x"))

    y_coords = sdata[layer].y.data
    x_coords = sdata[layer].x.data

    # if bug is fixed in spatialdata, you should set coordinates here, not after adding image to sdata
    spatial_image = spatial_image.assign_coords({"y": y_coords, "x": x_coords})

    # during adding of image it is written to zarr store
    sdata.add_image(name=output_layer, image=spatial_image)

    # add coordinates, due to bug these are lost when writing to zarr store
    sdata.images[output_layer] = sdata[output_layer].assign_coords(
        {"y": y_coords, "x": x_coords}
    )

    return sdata


def clahe_processing(
    sdata: SpatialData,
    output_layer: str = "clahe",
    contrast_clip: int = 3.5,
    chunksize_clahe: int = 10000,
) -> SpatialData:
    # TODO take whole image as chunksize + overlap tuning

    layer = [*sdata.images][-1]

    x_coords = sdata[layer].x.data
    y_coords = sdata[layer].y.data

    # convert to imagecontainer, because apply not yet implemented in sdata
    ic = sq.im.ImageContainer(sdata[layer], layer=layer)

    result_list = []

    for channel in sdata[layer].c.data:
        clahe = cv2.createCLAHE(clipLimit=contrast_clip, tileGridSize=(8, 8))

        ic_clahe = ic.apply(
            {"0": clahe.apply},
            layer=layer,
            new_layer=output_layer,
            drop=True,
            channel=channel,
            copy=True,
            chunks=chunksize_clahe,
            lazy=True,
            depth=3000,
            boundary='reflect'
        )

        # squeeze channel dim and z-dimension
        result_list.append(ic_clahe["clahe"].data.squeeze(axis=(2, 3)))

    result = da.stack(result_list, axis=0)

    spatial_image = spatialdata.models.Image2DModel.parse(result, dims=("c", "y", "x"))

    # if bug is fixed in spatialdata, you should set coordinates here, not after adding image to sdata
    spatial_image = spatial_image.assign_coords({"y": y_coords, "x": x_coords})

    # during adding of image it is written to zarr store
    sdata.add_image(name=output_layer, image=spatial_image)

    # add coordinates, due to bug these are lost when writing to zarr store
    sdata.images[output_layer] = sdata[output_layer].assign_coords(
        {"y": y_coords, "x": x_coords}
    )

    return sdata


def cellpose(
    img,
    min_size=80,
    cellprob_threshold=-4,
    flow_threshold=0.85,
    diameter=100,
    model_type="cyto",
    channels=[1, 0],
    device="cpu",
):

    gpu = torch.cuda.is_available()
    model = models.Cellpose(gpu=gpu, model_type=model_type, device=torch.device(device))
    masks, _, _, _ = model.eval(
        img,
        diameter=diameter,
        channels=channels,
        min_size=min_size,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )
    return masks


def segmentation_cellpose(
    sdata: SpatialData,
    layer: Optional[str] = None,
    output_layer: str = "segmentation_mask",
    crop_param: Optional[Tuple[int, int, int]] = None,
    device: str = "cpu",
    min_size: int = 80,
    flow_threshold: float = 0.6,
    diameter: int = 55,
    cellprob_threshold: int = 0,
    model_type: str = "nuclei",
    channels=[0, 0],
    chunks="auto",
    lazy=False,
) -> SpatialData:
    if layer is None:
        layer = [*sdata.images][-1]

    ic = sq.im.ImageContainer(sdata[layer], layer=layer)

    if crop_param:
        ic = ic.crop_corner(y=crop_param[1], x=crop_param[0], size=crop_param[2])
        # rechunk if you take crop, in order to be able to save as spatialdata object.
        # TODO check if this still necessary
        # for layer in ic.data.data_vars:
        #    chunksize = ic[layer].data.chunksize[0]
        #    ic[layer] = ic[layer].chunk(chunksize)

    sq.im.segment(
        img=ic,
        layer=layer,
        method=cellpose,
        channel=None,
        chunks=chunks,
        lazy=lazy,
        min_size=min_size,
        layer_added=output_layer,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        diameter=diameter,
        model_type=model_type,
        channels=channels,
        device=device,
    )

    imageContainerToSData(
        ic=ic, sdata=sdata, layers_im=[], layers_labels=[output_layer]
    )

    polygons = mask_to_polygons_layer_dask(mask=sdata[output_layer].data)
    polygons = polygons.dissolve(by="cells")

    x_coords = ic.data.x.data
    y_coords = ic.data.y.data

    x_translation = x_coords[0]
    y_translation = y_coords[0]

    polygons["geometry"] = polygons["geometry"].apply(
        lambda geom: translate(geom, xoff=x_translation, yoff=y_translation)
    )

    shapes_name = f'{output_layer}_boundaries'

    sdata.add_shapes(
        name=shapes_name,
        shapes=spatialdata.models.ShapesModel.parse(polygons),
    )

    return sdata


def imageContainerToSData(
    ic,
    sdata=None,
    layers_im=["corrected", "raw_image"],
    layers_labels=["segmentation_mask"],
):
    if sdata == None:
        sdata = spatialdata.SpatialData()

    # coordinates are not copied correctly from imagecontainer to spatialdata object, so do it 'by hand'
    x_coords = ic.data.x.data
    y_coords = ic.data.y.data

    temp = ic.data.rename_dims({"channels": "c"})

    for i in layers_im:
        spatial_image = spatialdata.models.Image2DModel.parse(
            temp[i].squeeze(dim="z").transpose("c", "y", "x")
        )
        spatial_image = spatial_image.assign_coords({"y": y_coords, "x": x_coords})
        sdata.add_image(name=i, image=spatial_image)
        # Due to bug in spatialdata, coordinates are lost after saving to .zarr, which happens automatically if sdata is backed by zarr store,
        # therefore assign coordinates again
        sdata.images[i] = sdata[i].assign_coords({"y": y_coords, "x": x_coords})

    for j in layers_labels:
        spatial_label = spatialdata.models.Labels2DModel.parse(
            temp[j].squeeze().transpose("y", "x")
        )
        spatial_label = spatial_label.assign_coords({"y": y_coords, "x": x_coords})
        sdata.add_labels(name=j, labels=spatial_label)
        # Due to bug in spatialdata, coordinates are lost after saving to .zarr, which happens automatically if sdata is backed by zarr store,
        # therefore assign coordinates again
        sdata.labels[j] = sdata[j].assign_coords({"y": y_coords, "x": x_coords})

    return sdata


def segmentationPlot(
    sdata,
    crd=None,
    layer: Optional[str] = None,
    channel: Optional[int] = None,
    shapes_layer="segmentation_mask_boundaries",
    output: str = None,
) -> None:
    if layer is None:
        layer = [*sdata.images][-1]

    si = sdata.images[layer]

    image_boundary = [si.x.data[0], si.x.data[-1] + 1, si.y.data[0], si.y.data[-1] + 1]

    if crd is not None:
        _crd = crd
        crd = overlapping_region_2D(crd, image_boundary)
        if crd is None:
            warnings.warn(
                (
                    f"Provided crd '{_crd}' and image_boundary '{image_boundary}' do not have any overlap. "
                    f"Please provide a crd that has some overlap with the image. "
                    f"Setting crd to image_boundary '{image_boundary}'."
                )
            )
            crd = image_boundary
    # if crd is None, set crd equal to image_boundary
    else:
        crd = image_boundary

    channels = [channel] if channel is not None else si.c.data

    for ch in channels:    

        fig, ax = plt.subplots(1, 2, figsize=(20, 20))
        si.isel(c=ch).squeeze().sel(
            x=slice(crd[0], crd[1]), y=slice(crd[2], crd[3])
        ).plot.imshow(cmap="gray", robust=True, ax=ax[0], add_colorbar=False)
        si.isel(c=ch).squeeze().sel(
            x=slice(crd[0], crd[1]), y=slice(crd[2], crd[3])
        ).plot.imshow(cmap="gray", robust=True, ax=ax[1], add_colorbar=False)
        sdata[shapes_layer].cx[crd[0] : crd[1], crd[2] : crd[3]].plot(
            ax=ax[1],
            edgecolor="white",
            linewidth=1,
            alpha=0.5,
            legend=True,
            aspect=1,
        )
        for i in range(len(ax)):
            ax[i].axes.set_aspect("equal")
            ax[i].set_xlim(crd[0], crd[1])
            ax[i].set_ylim(crd[2], crd[3])
            ax[i].invert_yaxis()
            ax[i].set_title("")
            # ax.axes.xaxis.set_visible(False)
            # ax.axes.yaxis.set_visible(False)
            ax[i].spines["top"].set_visible(False)
            ax[i].spines["right"].set_visible(False)
            ax[i].spines["bottom"].set_visible(False)
            ax[i].spines["left"].set_visible(False)

        # Save the plot to ouput
        if output:
            plt.close(fig)
            fig.savefig( f"{output}_{ch}")


def mask_to_polygons_layer(mask: np.ndarray) -> geopandas.GeoDataFrame:
    """Returns the polygons as GeoDataFrame

    This function converts the mask to polygons.
    https://rocreguant.com/convert-a-mask-into-a-polygon-for-images-using-shapely-and-rasterio/1786/
    """

    all_polygons = []
    all_values = []

    # Extract the polygons from the mask
    for shape, value in features.shapes(
        mask.astype(np.int32),
        mask=(mask > 0),
        transform=rasterio.Affine(1.0, 0, 0, 0, 1.0, 0),
    ):
        all_polygons.append(shapely.geometry.shape(shape))
        all_values.append(int(value))

    return geopandas.GeoDataFrame(dict(geometry=all_polygons), index=all_values)


def mask_to_polygons_layer_dask(mask: da.Array) -> geopandas.GeoDataFrame:
    """Returns the polygons as GeoDataFrame

    This function converts the mask to polygons.
    https://rocreguant.com/convert-a-mask-into-a-polygon-for-images-using-shapely-and-rasterio/1786/
    """

    # Define a function to extract polygons and values from each chunk
    @delayed
    def extract_polygons(mask_chunk: np.ndarray, chunk_coords: tuple) -> tuple:
        all_polygons = []
        all_values = []

        # Compute the boolean mask before passing it to the features.shapes() function
        bool_mask = mask_chunk > 0

        # Get chunk's top-left corner coordinates
        x_offset, y_offset = chunk_coords

        for shape, value in features.shapes(
            mask_chunk.astype(np.int32),
            mask=bool_mask,
            transform=rasterio.Affine(1.0, 0, y_offset, 0, 1.0, x_offset),
        ):
            all_polygons.append(shapely.geometry.shape(shape))
            all_values.append(int(value))

        return all_polygons, all_values

    # Map the extract_polygons function to each chunk
    # Create a list of delayed objects

    chunk_coords = list(
        itertools.product(
            *[range(0, s, cs) for s, cs in zip(mask.shape, mask.chunksize)]
        )
    )

    delayed_results = [
        extract_polygons(chunk, coord)
        for chunk, coord in zip(mask.to_delayed().flatten(), chunk_coords)
    ]
    # Compute the results
    results = dask.compute(*delayed_results, scheduler="threads")

    # Combine the results into a single list of polygons and values
    all_polygons = []
    all_values = []
    for polygons, values in results:
        all_polygons.extend(polygons)
        all_values.extend(values)

    # Create a GeoDataFrame from the extracted polygons and values
    return geopandas.GeoDataFrame({"geometry": all_polygons, "cells": all_values})


def color(_) -> matplotlib.colors.Colormap:
    """Select random color from set1 colors."""
    return plt.get_cmap("Set1")(np.random.choice(np.arange(0, 18)))


def border_color(r: bool) -> matplotlib.colors.Colormap:
    """Select border color from tab10 colors or preset color (1, 1, 1, 1) otherwise."""
    return plt.get_cmap("tab10")(3) if r else (1, 1, 1, 1)


def linewidth(r: bool) -> float:
    """Select linewidth 1 if true else 0.5."""
    return 1 if r else 0.5


def delete_overlap(voronoi, polygons):
    I1, I2 = voronoi.sindex.query_bulk(voronoi["geometry"], predicate="overlaps")
    voronoi2 = voronoi.copy()

    for cell1, cell2 in zip(I1, I2):
        # if cell1!=cell2:
        voronoi.geometry.iloc[cell1] = voronoi.iloc[cell1].geometry.intersection(
            voronoi2.iloc[cell1].geometry.difference(voronoi2.iloc[cell2].geometry)
        )
        voronoi.geometry.iloc[cell2] = voronoi.iloc[cell2].geometry.intersection(
            voronoi2.iloc[cell2].geometry.difference(voronoi2.iloc[cell1].geometry)
        )
    voronoi["geometry"] = voronoi.geometry.union(polygons.geometry)
    polygons = polygons.buffer(distance=0)
    voronoi = voronoi.buffer(distance=0)
    return voronoi


def create_voronoi_boundaries(
    sdata: SpatialData, radius: int = 0, shapes_layer: str = "segmentation_mask_boundaries"
):
    if radius < 0:
        raise ValueError(
            f"radius should be >0, provided value for radius is '{radius}'"
        )

    sdata[shapes_layer].index = sdata[shapes_layer].index.astype("str")

    expanded_layer_name = "expanded_cells" + str(radius)
    # sdata[shape_layer].index = list(map(str, sdata[shape_layer].index))

    si = sdata[[*sdata.images][0]]
    boundary = Polygon(
        [
            (si.x.data[0], si.y.data[0]),
            (si.x.data[-1] + 200, si.y.data[0]),
            (si.x.data[-1] + 200, si.y.data[-1] + 200),
            (si.x.data[0], si.y.data[-1] + 200),
        ]
    )

    if expanded_layer_name in [*sdata.shapes]:
        del sdata.shapes[expanded_layer_name]
    sdata[expanded_layer_name] = sdata[shapes_layer].copy()
    sdata[expanded_layer_name]["geometry"] = sdata[shapes_layer].simplify(2)

    vd = voronoiDiagram4plg(sdata[expanded_layer_name], boundary)
    voronoi = geopandas.sjoin(
        vd, sdata[expanded_layer_name], predicate="contains", how="left"
    )
    voronoi.index = voronoi.index_right
    voronoi = voronoi[~voronoi.index.duplicated(keep="first")]
    voronoi = delete_overlap(voronoi, sdata[expanded_layer_name])

    buffered = sdata[expanded_layer_name].buffer(distance=radius)
    intersected = voronoi.sort_index().intersection(buffered.sort_index())

    sdata[expanded_layer_name].geometry = intersected

    return sdata


def read_transcripts(
    sdata: SpatialData,
    path_count_matrix: Union[str, Path],
    path_transform_matrix: Optional[Union[str, Path]] = None,
    debug: bool = False,
    column_x: int = 0,
    column_y: int = 1,
    column_gene: int = 3,
    column_midcount: Optional[int] = None,
    delimiter: str = ",",
    header: Optional[int] = None,
) -> SpatialData:
    # Read the CSV file using Dask
    ddf = dd.read_csv(path_count_matrix, delimiter=delimiter, header=header)

    # Read the transformation matrix
    if path_transform_matrix is None:
        print("No transform matrix given, will use identity matrix.")
        transform_matrix = np.identity(3)
    else:
        transform_matrix = np.loadtxt(path_transform_matrix)

    print(transform_matrix)

    if debug:
        n = 100000
        fraction = n / len(ddf)
        ddf = ddf.sample(frac=fraction)

    # Function to repeat rows based on MIDCount value
    def repeat_rows(df):
        repeat_df = df.reindex(
            df.index.repeat(df.iloc[:, column_midcount])
        ).reset_index(drop=True)
        return repeat_df

    # Apply the row repeat function if column_midcount is not None (e.g. for Stereoseq)
    if column_midcount is not None:
        ddf = ddf.map_partitions(repeat_rows, meta=ddf)

    def transform_coordinates(df):
        micron_coordinates = df.iloc[:, [column_x, column_y]].values
        micron_coordinates = np.column_stack(
            (micron_coordinates, np.ones(len(micron_coordinates)))
        )
        pixel_coordinates = np.dot(micron_coordinates, transform_matrix.T)[:, :2]
        result_df = df.iloc[:, [column_gene]].copy()
        result_df["pixel_x"] = pixel_coordinates[:, 0]
        result_df["pixel_y"] = pixel_coordinates[:, 1]
        return result_df

    # Apply the transformation to the Dask DataFrame
    transformed_ddf = ddf.map_partitions(transform_coordinates)

    # Rename the columns
    transformed_ddf.columns = ["gene", "pixel_x", "pixel_y"]

    # Reorder
    transformed_ddf = transformed_ddf[["pixel_x", "pixel_y", "gene"]]

    sdata = _add_transcripts_to_sdata(sdata, transformed_ddf)

    return sdata


def _add_transcripts_to_sdata(sdata: SpatialData, transformed_ddf: DaskDataFrame):
    # TODO below fix to remove transcripts does not work, points not allowed to be deleted on disk.
    #if sdata.points:
    #    for points_layer in [*sdata.points]:
    #        del sdata.points[points_layer]

    sdata.add_points(
        name="transcripts",
        points=spatialdata.models.PointsModel.parse(
            transformed_ddf, coordinates={"x": "pixel_x", "y": "pixel_y"}
        ),
    )
    return sdata


def read_resolve_transcripts(
    sdata: SpatialData,
    path_count_matrix: str | Path,
) -> SpatialData:
    args = (sdata, path_count_matrix)
    kwargs = {
        "column_x": 0,
        "column_y": 1,
        "column_gene": 3,
        "delimiter": "\t",
        "header": None,
    }

    sdata = read_transcripts(*args, **kwargs)
    return sdata


def read_vizgen_transcripts(
    sdata: SpatialData,
    path_count_matrix: str | Path,
    path_transform_matrix: str | Path,
) -> SpatialData:
    args = (sdata, path_count_matrix, path_transform_matrix)
    kwargs = {
        "column_x": 2,
        "column_y": 3,
        "column_gene": 8,
        "delimiter": ",",
        "header": 0,
    }

    sdata = read_transcripts(*args, **kwargs)
    return sdata


def read_stereoseq_transcripts(
    sdata: SpatialData,
    path_count_matrix: str | Path,
) -> SpatialData:
    args = (sdata, path_count_matrix)
    kwargs = {
        "column_x": 1,
        "column_y": 2,
        "column_gene": 0,
        "column_midcount": 3,
        "delimiter": ",",
        "header": 0,
    }

    sdata = read_transcripts(*args, **kwargs)
    return sdata


def allocation(
    sdata: SpatialData,
    shapes_layer: str = "segmentation_mask_boundaries",
) -> Tuple[SpatialData, DaskDataFrame]:
    """Returns the AnnData object with transcript and polygon data."""

    sdata[shapes_layer].index = sdata[shapes_layer].index.astype("str")

    # need to do this transformation,
    # because the polygons have same offset coords.x0 and coords.y0 as in segmentation_mask
    Coords = namedtuple("Coords", ["x0", "y0"])
    s_mask = sdata["segmentation_mask"]
    coords = Coords(s_mask.x.data[0], s_mask.y.data[0])

    transform = Affine.translation(coords.x0, coords.y0)

    # Creating masks from polygons. TODO decide if you want to do this, even if voronoi is not calculated...
    # This is computationaly not heavy, but could take some ram,
    # because it creates image-size array of masks in memory
    #I guess not if no voronoi was created. 
    print("creating masks from polygons")
    masks = rasterio.features.rasterize(
        zip(
            sdata[shapes_layer].geometry, sdata[shapes_layer].index.values.astype(float)
        ),
        out_shape=[s_mask.shape[0], s_mask.shape[1]],
        dtype="uint32",
        fill=0,
        transform=transform,
    )

    print(f"Created masks with shape {masks.shape}")
    ddf = sdata["transcripts"]

    print("Calculate cell counts")

    # Define a function to process each partition using its index
    def process_partition(index, masks, coords):
        partition = ddf.get_partition(index).compute()

        filtered_partition = partition[
            (coords.y0 < partition["y"])
            & (partition["y"] < masks.shape[0] + coords.y0)
            & (coords.x0 < partition["x"])
            & (partition["x"] < masks.shape[1] + coords.x0)
        ]

        filtered_partition["cells"] = masks[
            filtered_partition["y"].values.astype(int) - int(coords.y0),
            filtered_partition["x"].values.astype(int) - int(coords.x0),
        ]

        return filtered_partition

    # Get the number of partitions in the Dask DataFrame
    num_partitions = ddf.npartitions

    # Process each partition using its index
    processed_partitions = [
        delayed(process_partition)(i, masks, coords) for i in range(num_partitions)
    ]

    # Combine the processed partitions into a single DataFrame
    combined_partitions = dd.from_delayed(processed_partitions)

    coordinates = combined_partitions.groupby("cells").mean().iloc[:, [0, 1]]
    cell_counts = combined_partitions.groupby(["cells", "gene"]).size()

    coordinates, cell_counts = compute(coordinates, cell_counts, scheduler="threads")

    cell_counts = cell_counts.unstack(fill_value=0)

    # make sure coordinates are sorted in same order as cell_counts
    index_order = cell_counts.index.argsort()
    coordinates = coordinates.loc[cell_counts.index[index_order]]

    print("Create anndata object")

    # Create the anndata object
    adata = AnnData(cell_counts[cell_counts.index != 0])
    coordinates.index = coordinates.index.map(str)
    adata.obsm["spatial"] = coordinates[coordinates.index != "0"].values

    adata.obs["region"] = 1
    adata.obs["instance"] = 1

    if sdata.table:
        del sdata.table

    sdata.table = spatialdata.models.TableModel.parse(
        adata, region_key="region", region=1, instance_key="instance"
    )

    for i in [*sdata.shapes]:
        if 'filtered' not in i:
            print(i)
            sdata[i].index = list(map(str, sdata[i].index))
            sdata.add_shapes(
                name='filtered_segmentation_'+i,
                shapes=spatialdata.models.ShapesModel.parse(
                    sdata[i][~np.isin(sdata[i].index.values.astype(int), sdata.table.obs.index.values.astype(int))]
                ),
                overwrite=True,
            )
            sdata.add_shapes(
                name=i,
                shapes=spatialdata.models.ShapesModel.parse(
                    sdata[i][np.isin(sdata[i].index.values.astype(int), sdata.table.obs.index.values.astype(int))]
                ),
                overwrite=True,
            )

    return sdata


def sanity_plot_transcripts_matrix(
    xarray: Union[np.ndarray, xr.DataArray],
    in_df: Optional[Union[PandasDataFrame, DaskDataFrame]] = None,
    polygons: Optional[geopandas.GeoDataFrame] = None,
    plot_cell_number: bool = False,
    n: Optional[int] = None,
    name_x: str = "x",
    name_y: str = "y",
    name_gene_column: str = "gene",
    gene: Optional[str] = None,
    crd: Optional[Tuple[int, int, int, int]] = None,
    cmap: str = "gray",
    output: Union[Path, str] = None,
):
    # in_df can be dask dataframe or pandas dataframe

    # plot for sanity check

    def extract_boundaries_from_geometry_collection(geometry):
        if isinstance(geometry, Polygon):
            return [geometry.boundary]
        elif isinstance(geometry, MultiPolygon):
            return [polygon.boundary for polygon in geometry.geoms]
        elif isinstance(geometry, GeometryCollection):
            boundaries = []
            for geom in geometry:
                boundaries.extend(extract_boundaries_from_geometry_collection(geom))
            return boundaries
        else:
            return []

    fig, ax = plt.subplots(figsize=(10, 10))

    if isinstance(xarray, np.ndarray):
        xarray = xr.DataArray(
            xarray,
            dims=("y", "x"),
            coords={"y": np.arange(xarray.shape[0]), "x": np.arange(xarray.shape[1])},
        )

    if crd is None:
        crd = [
            xarray.x.data[0],
            xarray.x.data[-1] + 1,
            xarray.y.data[0],
            xarray.y.data[-1] + 1,
        ]

    xarray.squeeze().sel(x=slice(crd[0], crd[1]), y=slice(crd[2], crd[3])).plot.imshow(
        cmap=cmap, robust=True, ax=ax, add_colorbar=False
    )

    # update so that a sample is taken from the dataframe (otherwise plotting takes too long), i.e. take n points max

    if in_df is not None:
        # query first and then slicing gene is faster than vice versa
        in_df = in_df.query(
            f"{crd[0]} <= {name_x} <= {crd[1]} and {crd[2]} <= {name_y} <= {crd[3]}"
        )

        if gene:
            in_df = in_df[in_df[name_gene_column] == gene]

        # we do not sample a fraction of the transcripts if a specific gene is given
        else:
            size = len(in_df)

            print(f"size before sampling is {size}")

            if n is not None and size > n:
                fraction = n / size
                in_df = in_df.sample(frac=fraction)

        if isinstance(in_df, DaskDataFrame):
            in_df = in_df.compute()

        print(f"Plotting {in_df.shape[0]} transcripts.")

        if gene:
            alpha = 0.5
        else:
            alpha = 0.2

        ax.scatter(in_df[name_x], in_df[name_y], color="r", s=8, alpha=alpha)

    if polygons is not None:
        print("Selecting boundaries")

        polygons_selected = polygons.cx[crd[0] : crd[1], crd[2] : crd[3]]

        polygons_selected["boundaries"] = polygons_selected["geometry"].apply(
            extract_boundaries_from_geometry_collection
        )
        exploded_boundaries = polygons_selected.explode("boundaries")
        exploded_boundaries["geometry"] = exploded_boundaries["boundaries"]
        exploded_boundaries = exploded_boundaries.drop(columns=["boundaries"])

        print("Plotting boundaries")

        # Plot the polygon boundaries
        exploded_boundaries.plot(
            ax=ax,
            aspect=1,
        )

        print("End plotting boundaries")

        # Plot the values inside the polygons
        if plot_cell_number:
            for _, row in polygons_selected.iterrows():
                centroid = row.geometry.centroid
                value = row.name
                ax.annotate(
                    value,
                    (centroid.x, centroid.y),
                    color="green",
                    fontsize=20,
                    ha="center",
                    va="center",
                )

    ax.set_xlim(crd[0], crd[1])
    ax.set_ylim(crd[2], crd[3])

    ax.axis("on")

    if gene:
        ax.set_title(f"Transcripts and cell boundaries for gene: {gene}.")

    if output:
        plt.savefig(output)
    else:
        plt.show()
    plt.close()


def control_transcripts(df, scaling_factor=100):
    """This function plots the transcript density of the tissue. You can use it to compare different regions in your tissue on transcript density."""
    Try = df.groupby(["x", "y"]).count()["gene"]
    Image = np.array(Try.unstack(fill_value=0))
    Image = Image / np.max(Image)
    blurred = gaussian_filter(scaling_factor * Image, sigma=7)
    return blurred


def plot_control_transcripts(blurred, sdata, layer: Optional[str] = None, crd=None):
    if layer is None:
        layer = [*sdata.images][-1]
    if crd:
        fig, ax = plt.subplots(1, 2, figsize=(20, 20))

        ax[0].imshow(blurred.T[crd[0] : crd[1], crd[2] : crd[3]], cmap="magma", vmax=5)
        sdata[layer].squeeze().sel(
            x=slice(crd[0], crd[1]), y=slice(crd[2], crd[3])
        ).plot.imshow(cmap="gray", robust=True, ax=ax[1], add_colorbar=False)
        ax[0].set_title("Transcript density")
        ax[1].set_title("Corrected image")
    fig, ax = plt.subplots(1, 2, figsize=(20, 20))

    ax[0].imshow(blurred.T, cmap="magma", vmax=5)
    sdata[layer].squeeze().plot.imshow(
        cmap="gray", robust=True, ax=ax[1], add_colorbar=False
    )
    ax[1].axes.set_aspect("equal")
    ax[1].invert_yaxis()

    ax[0].set_title("Transcript density")
    ax[1].set_title("Corrected image")


def analyse_genes_left_out(sdata, df):
    """This function"""
    filtered = pd.DataFrame(
        sdata.table.X.sum(axis=0)
        / df.groupby("gene").count()["x"][sdata.table.var.index]
    )
    filtered = filtered.rename(columns={"x": "proportion_kept"})
    filtered["raw_counts"] = df.groupby("gene").count()["x"][sdata.table.var.index]
    filtered["log_raw_counts"] = np.log(filtered["raw_counts"])

    sns.scatterplot(data=filtered, y="proportion_kept", x="log_raw_counts")

    plt.axvline(filtered["log_raw_counts"].median(), color="green", linestyle="dashed")
    plt.axhline(filtered["proportion_kept"].median(), color="red", linestyle="dashed")
    plt.xlim(
        left=-0.5, right=filtered["log_raw_counts"].quantile(0.99)
    )  # set y-axis limit from 0 to the 95th percentile of y
    # show the plot
    plt.show()
    r, p = sp.stats.pearsonr(filtered["log_raw_counts"], filtered["proportion_kept"])
    sns.regplot(x="log_raw_counts", y="proportion_kept", data=filtered)
    ax = plt.gca()
    ax.text(0.7, 0.9, "r={:.2f}, p={:.2g}".format(r, p), transform=ax.transAxes)

    plt.axvline(filtered["log_raw_counts"].median(), color="green", linestyle="dashed")
    plt.axhline(filtered["proportion_kept"].median(), color="red", linestyle="dashed")
    plt.show()
    print("The ten genes with the highest proportion of transcripts filtered out")
    print(filtered.sort_values(by="proportion_kept")[0:10].iloc[:, 0:2])
    return filtered


def plot_shapes(
    sdata,
    column: str = None,
    cmap: str = "magma",
    img_layer: Optional[str] = None,
    channel: Optional[int]=None,
    shapes_layer: str = "segmentation_mask_boundaries",
    alpha: float = 0.5,
    crd=None,
    output: str = None,
    vmin=None,
    vmax=None,
    plot_filtered=False,
    figsize=(20, 20),
) -> None:
    """
    This function plots an sdata object, with the cells on top. On default it plots the image layer that was added last.
    The default color is blue if no color is given as input. 
    Column: determines based on which column the cells need to be colored. Can be an obs column or a var column. 
    img_layer: the image layer that needs to be plotted, the last on on default
    shapes_layer: which shapes to plot, the default is nucleus_boundaries, but when performing an expansion it can be another layer. 
    alpha: the alpha-parameter of matplotlib: transperancy of the cells 
    crd: the crop that needs to be plotted, if none is given, the whole region is plotted, list of four coordinates
    output: whether you want to save it as an output or not, default is none and then plot is shown.
    vmin/vmax: adapting the color scale for continous data: give the percentile for which to color min and max. 
    ax: whne wanting to add the plot to another plot
    plot_filtered: whether or not to plot the cells that were filtered out during previous steps, this is a control function.
    """
    if img_layer is None:
        img_layer = [*sdata.images][-1]

    si = sdata.images[img_layer]

    image_boundary = [si.x.data[0], si.x.data[-1] + 1, si.y.data[0], si.y.data[-1] + 1]

    if crd is not None:
        _crd = crd
        crd = overlapping_region_2D(crd, image_boundary)
        if crd is None:
            warnings.warn(
                (
                    f"Provided crd '{_crd}' and image_boundary '{image_boundary}' do not have any overlap. "
                    f"Please provide a crd that has some overlap with the image. "
                    f"Setting crd to image_boundary '{image_boundary}'."
                )
            )
            crd = image_boundary
    # if crd is None, set crd equal to image_boundary
    else:
        crd = image_boundary
    size_im=(crd[1]-crd[0])*(crd[3]-crd[2])
    if column is not None:
        if column + "_colors" in sdata.table.uns:
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                "new_map",
                sdata.table.uns[column + "_colors"],
                N=len(sdata.table.uns[column + "_colors"]),
            )
        if column in sdata.table.obs.columns:
            column = sdata.table[
                sdata[shapes_layer].cx[crd[0] : crd[1], crd[2] : crd[3]].index, :
            ].obs[column]
        elif column in sdata.table.var.index:
            column = sdata.table[
                sdata[shapes_layer].cx[crd[0] : crd[1], crd[2] : crd[3]].index, :
            ].X[:, np.where(sdata.table.var.index == column)[0][0]]
        else:
            print(
                "The column defined in the function isnt a column in obs, nor is it a gene name, the plot is made without taking into account this value."
            )
            column = None
            cmap = None
    else:
        cmap = None
    if vmin != None:
        vmin = np.percentile(column, vmin)
    if vmax != None:
        vmax = np.percentile(column, vmax)
    

    channels = [channel] if channel is not None else si.c.data

    for ch in channels:

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        sdata[img_layer].isel(c=ch).squeeze().sel(
            x=slice(crd[0], crd[1]), y=slice(crd[2], crd[3])
        ).plot.imshow(cmap="gray", robust=True, ax=ax, add_colorbar=False)

        sdata[shapes_layer].cx[crd[0] : crd[1], crd[2] : crd[3]].plot(
            ax=ax,
            edgecolor="white",
            column=column,
            linewidth=1 if size_im< 5000*10000 else 0,
            alpha=alpha,
            legend=True,
            aspect=1,
            cmap=cmap,
            vmax=vmax,  # np.percentile(column,vmax),
            vmin=vmin,  # np.percentile(column,vmin)
        )
        if plot_filtered:
            for i in [*sdata.shapes]:
                if 'filtered' in i:
                    sdata[i].cx[crd[0] : crd[1], crd[2] : crd[3]].plot(
                    ax=ax,
                    edgecolor="red",
                    linewidth=1,
                    alpha=alpha,
                    legend=True,
                    aspect=1,
                    cmap='gray',
                    )
        ax.axes.set_aspect("equal")
        ax.set_xlim(crd[0], crd[1])
        ax.set_ylim(crd[2], crd[3])
        ax.invert_yaxis()
        ax.set_title("")
        # ax.axes.xaxis.set_visible(False)
        # ax.axes.yaxis.set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        # Save the plot to ouput
        if output:
            fig.savefig( f"{output}_{ch}")
        else:
            plt.show()
        plt.close()


def preprocessAdata(
    sdata,
    nuc_size_norm: bool = True,
    n_comps: int = 50,
    min_counts=10,
    min_cells=5,
    shapes_layer=None,
):
    """Returns the new and original AnnData objects

    This function calculates the QC metrics.
    All cells with less than 10 genes and all genes with less than 5 cells are removed.
    Normalization is performed based on the size of the nucleus in nuc_size_norm."""
    # calculate the max amount of pc's possible
    if min(sdata.table.shape) < n_comps:
        n_comps = min(sdata.table.shape)
        print(
            "amount of pc's was set to " + str(min(sdata.table.shape)),
            " because of the dimensionality of the data.",
        )
    # Calculate QC Metrics

    sc.pp.calculate_qc_metrics(sdata.table, inplace=True, percent_top=[2, 5])

    # Filter cells and genes
    sc.pp.filter_cells(sdata.table, min_counts=min_counts)
    sc.pp.filter_genes(sdata.table, min_cells=min_cells)

    # Normalize nucleus size
    if shapes_layer is None:
        shapes_layer = [*sdata.shapes][-1]
    sdata.table.obs["shapeSize"] = sdata[shapes_layer].area

    sdata.table.layers["raw_counts"] = sdata.table.X

    if nuc_size_norm:
        sdata.table.X = (sdata.table.X.T * 100 / sdata.table.obs.shapeSize.values).T
        sc.pp.log1p(sdata.table)
        # need to do .copy() here to set .raw value, because .scale still overwrites this .raw, which is unexpected behaviour
        sdata.table.raw = sdata.table.copy()
        sc.pp.scale(sdata.table, max_value=10)

    else:
        sc.pp.normalize_total(sdata.table)
        sc.pp.log1p(sdata.table)
        sdata.table.raw = sdata.table.copy()

    sc.tl.pca(sdata.table, svd_solver="arpack", n_comps=n_comps)
    # Is this the best way o doing it? Every time you subset your data, the polygons should be subsetted too!
    for i in [*sdata.shapes]:
        if 'filtered' not in i:
            sdata[i].index = sdata[i].index.astype("str")
   
            sdata.add_shapes(
                name='filtered_low_counts_'+i,
                shapes=spatialdata.models.ShapesModel.parse(
                    sdata[i][~np.isin(sdata[i].index.values.astype(int), sdata.table.obs.index.values.astype(int))]
                ),
                overwrite=True,
            )
            sdata.add_shapes(
                name=i,
                shapes=spatialdata.models.ShapesModel.parse(
                    sdata[i][np.isin(sdata[i].index.values.astype(int), sdata.table.obs.index.values.astype(int))]
                ),
                overwrite=True,
            )
           
    # need to update sdata.table via .parse, otherwise it will not be backed by zarr store
    _back_sdata_table_to_zarr( sdata )


    return sdata

def _back_sdata_table_to_zarr(sdata: SpatialData):
    adata=sdata.table.copy() 
    del sdata.table
    sdata.table = spatialdata.models.TableModel.parse( adata )

def preprocesAdataPlot(sdata: SpatialData, output: str = None) -> None:
    """This function plots the size of the nucleus related to the counts."""

    sc.pl.pca(
        sdata.table,
        color="total_counts",
        show=False,
        title="PC plot colored by total counts",
    )
    if output:
        plt.savefig(output + "_total_counts_pca.png")
        plt.close()
    else:
        plt.show()
    plt.close()
    sc.pl.pca(
        sdata.table,
        color="shapeSize",
        show=False,
        title="PC plot colored by object size",
    )
    if output:
        plt.savefig(output + "_shapeSize_pca.png")
        plt.close()
    else:
        plt.show()
    plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(15, 4))
    sns.histplot(sdata.table.obs["total_counts"], kde=False, ax=axs[0])
    sns.histplot(sdata.table.obs["n_genes_by_counts"], kde=False, bins=55, ax=axs[1])
    if output:
        plt.savefig(output + "_histogram.png")
    else:
        plt.show()
    plt.close()

    fig, ax = plt.subplots()
    plt.scatter(sdata.table.obs["shapeSize"], sdata.table.obs["total_counts"])
    ax.set_title("shapeSize vs Transcripts Count")
    ax.set_xlabel("shapeSize")
    ax.set_ylabel("Total Counts")
    if output:
        plt.savefig(output + "_size_count.png")
    else:
        plt.show()
    plt.close()


def filter_on_size(sdata: SpatialData, min_size: int = 100, max_size: int = 100000):
    """Returns a tuple with the AnnData object and the number of filtered cells.

    All cells outside of the min and max size range are removed.
    If the distance between the location of the transcript and the center of the polygon is large, the cell is deleted.
    """

    start = sdata.table.shape[0]

    # Filter cells based on size and distance
    table = sdata.table[sdata.table.obs["shapeSize"] < max_size, :]
    table = table[table.obs["shapeSize"] > min_size, :]
    del sdata.table
    ## TODO: Look for a better way of doing this!
    sdata.table = spatialdata.models.TableModel.parse(table)

    for i in [*sdata.shapes]:
        if 'filtered'  not in i:
            sdata[i].index = sdata[i].index.astype("str")
            sdata.add_shapes(
                name='filtered_size_'+i,
                shapes=spatialdata.models.ShapesModel.parse(
                    sdata[i][~np.isin(sdata[i].index.values.astype(int), sdata.table.obs.index.values.astype(int))]
                ),
                overwrite=True,
            )
            sdata.add_shapes(
                name=i,
                shapes=spatialdata.models.ShapesModel.parse(
                    sdata[i][np.isin(sdata[i].index.values.astype(int), sdata.table.obs.index.values.astype(int))]
                ),
                overwrite=True,
            )
    filtered = start - table.shape[0]
    print(str(filtered) + " cells were filtered out based on size.")

    return sdata


##TODO:rewrite this function
def extract(ic: sq.im.ImageContainer, adata: AnnData) -> AnnData:
    """This function performs segmenation feature extraction and adds cell area and mean intensity to the annData object under obsm segmentation_features."""
    sq.im.calculate_image_features(
        adata,
        ic,
        layer="raw",
        features="segmentation",
        key_added="segmentation_features",
        features_kwargs={
            "segmentation": {
                "label_layer": "segment_cellpose",
                "props": ["label", "area", "mean_intensity"],
                "channels": [0],
            }
        },
    )

    return adata


def clustering(
    sdata: SpatialData, pcs: int, neighbors: int, cluster_resolution: float = 0.8
) -> SpatialData:
    """Returns the AnnData object.

    Performs neighborhood analysis, Leiden clustering and UMAP.
    Provides option to save the plots to output.
    """

    # Neighborhood analysis
    sc.pp.neighbors(sdata.table, n_neighbors=neighbors, n_pcs=pcs, random_state=100)
    sc.tl.umap(sdata.table, random_state=100)

    # Leiden clustering
    sc.tl.leiden(sdata.table, resolution=cluster_resolution, random_state=100)
    sc.tl.rank_genes_groups(sdata.table, "leiden", method="wilcoxon")

    _back_sdata_table_to_zarr( sdata=sdata )

    return sdata


def clustering_plot(sdata: SpatialData, output: str = None) -> None:
    """This function plots the clusters and genes ranking"""

    # Leiden clustering
    sc.pl.umap(sdata.table, color=["leiden"], show=not output)

    # Save the plot to ouput
    if output:
        plt.savefig(output + "_umap.png", bbox_inches="tight")
        plt.close()
        sc.pl.rank_genes_groups(sdata.table, n_genes=8, sharey=False, show=False)
        plt.savefig(output + "_rank_genes_groups.png", bbox_inches="tight")
        plt.close()

    # Display plot
    else:
        sc.pl.rank_genes_groups(sdata.table, n_genes=8, sharey=False)


def scoreGenes(
    sdata: SpatialData,
    path_marker_genes: str,
    delimiter=",",
    row_norm: bool = False,
    repl_columns: Dict[str, str] = None,
    del_celltypes: List[str] = None,
    input_dict=False,
) -> Tuple[dict, pd.DataFrame]:
    """Returns genes dict and the scores per cluster

    Load the marker genes from csv file in path_marker_genes.
    If the marker gene list is a one hot endoded matrix, leave the input as is.
    If the marker gene list is a Dictionary, with the first column the name of the celltype and the other columns the marker genes beloning to this celltype,
    input
    repl_columns holds the column names that should be replaced the in the marker genes.
    del_genes holds the marker genes that should be deleted from the marker genes and genes dict.
    """

    # Load marker genes from csv
    if input_dict:
        df_markers = pd.read_csv(
            path_marker_genes, header=None, index_col=0, delimiter=delimiter
        )
        df_markers = df_markers.T
        genes_dict = df_markers.to_dict("list")
        for i in genes_dict:
            genes_dict[i] = [x for x in genes_dict[i] if str(x) != "nan"]
    # Replace column names in marker genes
    else:
        df_markers = pd.read_csv(path_marker_genes, index_col=0, delimiter=delimiter)
        if repl_columns:
            for column, replace in repl_columns.items():
                df_markers.columns = df_markers.columns.str.replace(column, replace)

        # Create genes dict with all marker genes for every celltype
        genes_dict = {}
        for i in df_markers:
            genes = []
            for row, value in enumerate(df_markers[i]):
                if value > 0:
                    genes.append(df_markers.index[row])
            genes_dict[i] = genes

    assert (
        "unknown_celltype" not in genes_dict.keys()
    ), "Cell type 'unknown_celltype' is reserved for cells that could not be assigned a specific cell type"

    # Score all cells for all celltypes
    for key, value in genes_dict.items():
        try:
            sc.tl.score_genes(sdata.table, value, score_name=key)
        except ValueError:
            warnings.warn(
                f"Markergenes {value} not present in region, celltype {key} not found"
            )

    # Delete genes from marker genes and genes dict
    if del_celltypes:
        for gene in del_celltypes:
            if gene in df_markers.columns:
                del df_markers[gene]
            if gene in genes_dict.keys():
                del genes_dict[gene]

    sdata, scoresper_cluster = _annotate_celltype(
        sdata=sdata,
        celltypes=df_markers.columns,
        row_norm=row_norm,
        celltype_column="annotation",
    )

    # add 'unknown_celltype' to the list of celltypes if it is detected.
    if "unknown_celltype" in sdata.table.obs["annotation"].cat.categories:
        genes_dict["unknown_celltype"] = []

    _back_sdata_table_to_zarr(sdata)

    return genes_dict, scoresper_cluster


def scoreGenesPlot(
    sdata: SpatialData,
    scoresper_cluster: pd.DataFrame,
    img_layer: Optional[str] = None,
    shapes_layer: str = "segmentation_mask_boundaries",
    crd=None,
    filter_index: Optional[int] = None,
    output: str = None,
) -> None:
    """This function plots the cleanliness and the leiden score next to the annotation."""
    if img_layer is None:
        img_layer = [*sdata.images][-1]
    si = sdata.images[img_layer]

    if crd is None:
        crd = [si.x.data[0], si.x.data[-1] + 1, si.y.data[0], si.y.data[-1] + 1]

    # Custom colormap:
    colors = np.concatenate(
        (plt.get_cmap("tab20c")(np.arange(20)), plt.get_cmap("tab20b")(np.arange(20)))
    )
    colors = [mpl.rgb2hex(colors[j * 4 + i]) for i in range(4) for j in range(10)]

    # Plot cleanliness and leiden next to annotation
    sc.pl.umap(sdata.table, color=["Cleanliness", "annotation"], show=False)

    if output:
        plt.savefig(output + "_Cleanliness_annotation", bbox_inches="tight")
    else:
        plt.show()
    plt.close()

    sc.pl.umap(sdata.table, color=["leiden", "annotation"], show=False)

    if output:
        plt.savefig(output + "_leiden_annotation", bbox_inches="tight")
    else:
        plt.show()
    plt.close()

    # Plot annotation and cleanliness columns of sdata.table (AnnData) object
    sdata.table.uns["annotation_colors"] = colors
    plot_shapes(
        sdata=sdata,
        column="annotation",
        crd=crd,
        img_layer=img_layer,
        shapes_layer=shapes_layer,
        output=output + "_annotation" if output else None,
    )

    # Plot heatmap of celltypes and filtered celltypes based on filter index
    sc.pl.heatmap(
        sdata.table,
        var_names=scoresper_cluster.columns.values,
        groupby="leiden",
        show=False,
    )

    if output:
        plt.savefig(output + "_leiden_heatmap", bbox_inches="tight")
    else:
        plt.show()
    plt.close()

    if filter_index:
        sc.pl.heatmap(
            sdata.table[
                sdata.table.obs.leiden.isin(
                    [
                        str(index)
                        for index in range(filter_index, len(sdata.table.obs.leiden))
                    ]
                )
            ],
            var_names=scoresper_cluster.columns.values,
            groupby="leiden",
            show=False,
        )

        if output:
            plt.savefig(
                output + f"_leiden_heatmap_filtered_{filter_index}",
                bbox_inches="tight",
            )
        else:
            plt.show()
        plt.close()


def correct_marker_genes(
    sdata: SpatialData,
    celltype_correction_dict: Dict[str, Tuple[float, float]],
):
    """Returns the new AnnData object.

    Corrects marker genes that are higher expessed by dividing them.
    The genes has as keys the genes that should be corrected and as values the threshold and the divider.
    """

    # Correct for all the genes
    for celltype, values in celltype_correction_dict.items():
        if celltype not in sdata.table.obs.columns:
            print(
                f"Cell type '{celltype}' not in obs of AnnData object. Skipping. Please first calculate gene expression for this cell type."
            )
            continue
        sdata.table.obs[celltype] = np.where(
            sdata.table.obs[celltype] < values[0],
            sdata.table.obs[celltype] / values[1],
            sdata.table.obs[celltype],
        )

    _back_sdata_table_to_zarr(sdata=sdata)

    return sdata


def annotate_maxscore(types: str, indexes: dict, sdata):
    """Returns the AnnData object.

    Adds types to the Anndata maxscore category.
    """
    sdata.table.obs.annotation = sdata.table.obs.annotation.cat.add_categories([types])
    for i, val in enumerate(sdata.table.obs.annotation):
        if val in indexes[types]:
            sdata.table.obs.annotation[i] = types
    return sdata


def remove_celltypes(types: str, indexes: dict, sdata):
    """Returns the AnnData object."""
    for index in indexes[types]:
        if index in sdata.table.obs.annotation.cat.categories:
            sdata.table.obs.annotation = (
                sdata.table.obs.annotation.cat.remove_categories(index)
            )
    return sdata


def _annotate_celltype(
    sdata: SpatialData,
    celltypes: List[str],
    row_norm: bool = False,
    celltype_column: str = "annotation",
) -> Tuple[SpatialData, PandasDataFrame]:
    scoresper_cluster = sdata.table.obs[
        [col for col in sdata.table.obs if col in celltypes]
    ]

    # Row normalization for visualisation purposes
    if row_norm:
        row_norm = scoresper_cluster.sub(
            scoresper_cluster.mean(axis=1).values, axis="rows"
        ).div(scoresper_cluster.std(axis=1).values, axis="rows")
        sdata.table.obs[scoresper_cluster.columns.values] = row_norm
        temp = pd.DataFrame(np.sort(row_norm)[:, -2:])
    else:
        temp = pd.DataFrame(np.sort(scoresper_cluster)[:, -2:])

    scores = (temp[1] - temp[0]) / ((temp[1] + temp[0]) / 2)
    sdata.table.obs["Cleanliness"] = scores.values

    def assign_cell_type(row):
        # Identify the cell type with the max score
        max_score_type = row.idxmax()
        # If max score is <= 0, assign 'unknown_celltype'
        if row[max_score_type] <= 0:
            return "unknown_celltype"
        else:
            return max_score_type

    # Assign 'unknown_celltype' cell_type if no cell type could be found that has larger expression than random sample
    # as calculated by sc.tl.score_genes function of scanpy.
    sdata.table.obs[celltype_column] = scoresper_cluster.apply(assign_cell_type, axis=1)
    sdata.table.obs[celltype_column] = sdata.table.obs[celltype_column].astype(
        "category"
    )
    # Set the Cleanliness score for unknown_celltype equal to 0 (i.e. not clean)
    sdata.table.obs.loc[
        sdata.table.obs[celltype_column] == "unknown_celltype", "Cleanliness"
    ] = 0

    return sdata, scoresper_cluster


def clustercleanliness(
    sdata: SpatialData,
    genes: List[str],
    gene_indexes: Dict[str, int] = None,
    colors: List[str] = None,
) -> Tuple[SpatialData, Optional[dict]]:
    """Returns a tuple with the AnnData object and the color dict."""

    celltypes = np.array(sorted(genes), dtype=str)
    color_dict = None

    # recalculate annotation, because we possibly did correction on celltype score for certain cells via correct_marker_genes function
    sdata, _ = _annotate_celltype(
        sdata=sdata,
        celltypes=celltypes,
        row_norm=False,
        celltype_column="annotation",
    )

    # Create custom colormap for clusters
    if not colors:
        color = np.concatenate(
            (
                plt.get_cmap("tab20c")(np.arange(20)),
                plt.get_cmap("tab20b")(np.arange(20)),
            )
        )
        colors = [mpl.rgb2hex(color[j * 4 + i]) for i in range(4) for j in range(10)]

    sdata.table.uns["annotation_colors"] = colors

    if gene_indexes:
        sdata.table.obs["annotationSave"] = sdata.table.obs.annotation
        gene_celltypes = {}

        for key, value in gene_indexes.items():
            gene_celltypes[key] = celltypes[value]

        for gene, indexes in gene_indexes.items():
            sdata = annotate_maxscore(gene, gene_celltypes, sdata)

        for gene, indexes in gene_indexes.items():
            sdata = remove_celltypes(gene, gene_celltypes, sdata)

        celltypes_f = np.delete(celltypes, list(chain(*gene_indexes.values())))  # type: ignore
        celltypes_f = np.append(celltypes_f, list(gene_indexes.keys()))
        color_dict = dict(zip(celltypes_f, sdata.table.uns["annotation_colors"]))

    else:
        color_dict = dict(zip(celltypes, sdata.table.uns["annotation_colors"]))

    for i, name in enumerate(color_dict.keys()):
        color_dict[name] = colors[i]
    sdata.table.uns["annotation_colors"] = list(
        map(color_dict.get, sdata.table.obs.annotation.cat.categories.values)
    )

    _back_sdata_table_to_zarr(sdata)

    return sdata, color_dict


def clustercleanlinessPlot(
    sdata: SpatialData,
    shapes_layer: str= 'segmentation_mask_boundaries',
    crd: List[int] = None,
    color_dict: dict = None,
    celltype_column: str = "annotation",
    output: str = None,
) -> None:
    """This function plots the clustercleanliness as barplots, the images with colored celltypes and the clusters."""

    # Create the barplot
    stacked = (
        sdata.table.obs.groupby(["leiden", celltype_column], as_index=False)
        .size()
        .pivot("leiden", celltype_column)
        .fillna(0)
    )
    stacked_norm = stacked.div(stacked.sum(axis=1), axis=0)
    stacked_norm.columns = list(sdata.table.obs.annotation.cat.categories)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Use custom colormap
    if color_dict:
        stacked_norm.plot(kind="bar", stacked=True, ax=fig.gca(), color=color_dict)
    else:
        stacked_norm.plot(kind="bar", stacked=True, ax=fig.gca())
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_yaxis().set_ticks([])
    plt.xlabel("Clusters")
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize="large")

    # Save the barplot to ouput
    if output:
        fig.savefig(output + "_barplot.png", bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)

    # Plot images with colored celltypes
    plot_shapes(
        sdata=sdata,
        column=celltype_column,
        alpha=0.8,
        shapes_layer=shapes_layer,
        output=output + f"_{celltype_column}" if output else None,
    )

    plot_shapes(
        sdata=sdata,
        column=celltype_column,
        crd=crd,
        alpha=0.8,
        shapes_layer=shapes_layer,
        output=output + f"_{celltype_column}_crop" if output else None,
    )

    # Plot clusters
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    sc.pl.umap(
        sdata.table,
        color=[celltype_column],
        ax=ax,
        show=not output,
        size=300000 / sdata.table.shape[0],
    )
    ax.axis("off")

    if output:
        fig.savefig(output + f"_{celltype_column}_umap.png", bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def enrichment(sdata, celltype_column: str = "annotation", seed: int = 0):
    """Returns the AnnData object.

    Performs some adaptations to save the data.
    Calculate the nhood enrichment"
    """

    # Adaptations for saving
    sdata.table.raw.var.index.names = ["genes"]
    sdata.table.var.index.names = ["genes"]
    # TODO: not used since napari spatialdata
    # adata.obsm["spatial"] = adata.obsm["spatial"].rename({0: "X", 1: "Y"}, axis=1)

    # Calculate nhood enrichment
    sq.gr.spatial_neighbors(sdata.table, coord_type="generic")
    sq.gr.nhood_enrichment(sdata.table, cluster_key=celltype_column, seed=seed)
    _back_sdata_table_to_zarr(sdata=sdata)
    return sdata


def enrichment_plot(
    sdata, celltype_column: str = "annotation", output: str = None
) -> None:
    """This function plots the nhood enrichment between different celltypes."""

    # remove 'nan' values from "adata.uns['annotation_nhood_enrichment']['zscore']"
    tmp = sdata.table.uns[f"{celltype_column}_nhood_enrichment"]["zscore"]
    sdata.table.uns[f"{celltype_column}_nhood_enrichment"]["zscore"] = np.nan_to_num(
        tmp
    )
    _back_sdata_table_to_zarr(sdata=sdata)
    sq.pl.nhood_enrichment(sdata.table, cluster_key=celltype_column, method="ward")

    # Save the plot to ouput
    if output:
        plt.savefig(output, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def micron_to_pixels(df, offset_x=45_000, offset_y=45_000, pixelSize=None):
    if pixelSize:
        df[x] /= pixelSize
        df[y] /= pixelSize
    if offset_x:
        df["x"] -= offset_x
    if offset_y:
        df["y"] -= offset_y

    return df


def read_in_stereoSeq(
    path_genes,
    xcol="x",
    ycol="y",
    genecol="geneID",
    countcol="MIDCount",
    skiprows=0,
    offset=None,
):
    """This function read in Stereoseq input data to a dask datafrmae with predefined column names.
    As we are working with BGI data, a column with counts is added."""
    in_df = dd.read_csv(path_genes, delimiter="\t", skiprows=skiprows)
    in_df = in_df.rename(
        columns={xcol: "x", ycol: "y", genecol: "gene", countcol: "counts"}
    )
    if offset:
        in_df = micron_to_pixels(in_df, offset_x=offset[0], offset_y=offset[1])
    in_df = in_df.loc[:, ["x", "y", "gene", "counts"]]

    in_df = in_df.dropna()
    return in_df


def read_in_RESOLVE(
    path_coordinates,
    sdata=None,
    xcol=0,
    ycol=1,
    genecol=3,
    filterGenes=None,
    offset=None,
):
    """The output of this function gives all locations of interesting transcripts in pixel coordinates matching the input image. Dask Dataframe contains columns x,y, and gene"""

    if sdata == None:
        sdata = spatialdata.SpatialData()
    in_df = dd.read_csv(path_coordinates, delimiter="\t", header=None)
    in_df = in_df.rename(columns={xcol: "x", ycol: "y", genecol: "gene"})
    if offset:
        in_df = micron_to_pixels(in_df, offset_x=offset[0], offset_y=offset[1])
    in_df = in_df.loc[:, ["x", "y", "gene"]]
    in_df = in_df.dropna()

    if filterGenes:
        for i in filter_genes:
            in_df = in_df[in_df["gene"].str.contains(i) == False]

    if sdata.points:
        for points_layer in [*sdata.points]:
            del sdata.points[points_layer]

    sdata.add_points(
        name="transcripts",
        points=spatialdata.models.PointsModel.parse(
            in_df, coordinates={"x": "x", "y": "y"}
        ),
    )

    return sdata


def read_in_Vizgen(
    path_genes,
    xcol="global_x",
    ycol="global_y",
    genecol="gene",
    skiprows=None,
    offset=None,
    bbox=None,
    pixelSize=None,
    filterGenes=None,
):
    """This function read in Vizgen input data to a dask datafrmae with predefined column names."""

    in_df = dd.read_csv(path_genes, skiprows=skiprows)
    in_df = in_df.loc[:, [xcol, ycol, genecol]]

    in_df = in_df.rename(columns={xcol: "x", ycol: "y", genecol: "gene"})

    if bbox:
        in_df["x"] -= bbox[0]
        in_df["y"] -= bbox[1]
    if pixelSize:
        in_df["x"] /= pixelSize
        in_df["y"] /= pixelSize
    if offset:
        in_df["x"] -= offset[0]
        in_df["y"] -= offset[1]

    in_df = in_df.dropna()

    if filterGenes:
        for i in filterGenes:
            in_df = in_df[in_df["gene"].str.contains(i) == False]
    return in_df


def write_to_zarr(filename: Path, output_name="raw_image", chunks: int = 1024):
    assert filename.exists()
    img = AICSImage(filename)
    img

    arr = img.dask_data[0, :, 0, :, :]
    arr

    # output_name=os.path.splitext( os.path.basename( filename) )[0]

    channels_names = ["DAPI"]

    image = to_spatial_image(
        arr, dims=["c", "y", "x"], name=output_name, c_coords=channels_names
    )

    # chunk_size can be 1 for channels
    max_chunk_size = chunks
    chunks = {
        "x": max_chunk_size,
        "y": max_chunk_size,
        "c": 1,
    }

    multiscale = to_multiscale(
        image,
        scale_factors=[2, 4, 8, 16],
        chunks=chunks,
    )

    zarr_path = os.path.join(os.path.dirname(filename), f"{output_name}.zarr")

    # For OME-NGFF, large datasets, use dimension_separator='/'
    # Write OME-NGFF images part
    store = zarr.storage.DirectoryStore(zarr_path, dimension_separator="/")

    # Compression options
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
    multiscale.to_zarr(store, encoding={output_name: {"compressor": compressor}})
    # consolidate_metadata is optional, but recommended and improves read latency
    zarr.consolidate_metadata(store)


def read_in_zarr_from_path(
    path,
    name=None,
    chunk: Optional[int] = None,
    crop_param: Tuple[int, int, int] = None,
) -> sq.im.ImageContainer:
    xarray_ds = xr.open_zarr(path)
    if chunk:
        # rechunking
        xarray_ds = xarray_ds.chunk(chunk)
    ic = sq.im.ImageContainer(xarray_ds[name], layer=name)
    if crop_param:
        ic = ic.crop_corner(y=crop_param[1], x=crop_param[0], size=crop_param[2])
    return ic


def plot_image_container(
    sdata: Union[SpatialData, sq.im.ImageContainer],
    output_path: Optional[str|Path] =None,
    crd:Optional[ List[int] ]=None,
    layer:str="image",
    channel: Optional[int] = None,
    aspect:str="equal",
    figsize:Tuple[int]=(10, 10),
):
    
    if not isinstance(sdata, (SpatialData, sq.im.ImageContainer)):
        raise ValueError("Only SpatialData and ImageContainer objects are supported.")

    channel_key = 'c' if isinstance(sdata, SpatialData) else 'channels'
    channels = [channel] if channel is not None else sdata[ layer ][ channel_key ].data
    
    for ch in channels:
        if isinstance(sdata, SpatialData):
            dataset = sdata[layer].isel(c=ch)
        elif isinstance(sdata, sq.im.ImageContainer):
            dataset = sdata[layer].isel(channels=ch)
        
        if crd is None:
            crd = [
                sdata[layer].x.data[0],
                sdata[layer].x.data[-1] + 1,
                sdata[layer].y.data[0],
                sdata[layer].y.data[-1] + 1,
            ]

        _, ax = plt.subplots(figsize=figsize)
        cmap = "gray"
        dataset.squeeze().sel(x=slice(crd[0], crd[1]), y=slice(crd[2], crd[3])).plot.imshow(
            cmap=cmap, robust=True, ax=ax, add_colorbar=False
        )

        ax.set_aspect(aspect)
        ax.invert_yaxis()

        if output_path:
            plt.savefig( f"{output_path}_{ch}")
        else:
            plt.show()
        plt.close()


def overlapping_region_2D(
    A: List[int | float], B: List[int | float]
) -> Optional[List[int | float]]:
    overlap_x = not (A[1] < B[0] or B[1] < A[0])
    overlap_y = not (A[3] < B[2] or B[3] < A[2])

    if overlap_x and overlap_y:
        # Calculate overlapping region
        x_min = max(A[0], B[0])
        x_max = min(A[1], B[1])
        y_min = max(A[2], B[2])
        y_max = min(A[3], B[3])

        # Return as a list: [x_min, x_max, y_min, y_max]
        return [x_min, x_max, y_min, y_max]
    else:
        return None
