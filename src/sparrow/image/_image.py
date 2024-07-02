from __future__ import annotations

from dask.array import Array
from spatialdata import SpatialData
from spatialdata.models.models import ScaleFactors_t
from spatialdata.transformations.transformations import BaseTransformation

from sparrow.image._manager import ImageLayerManager, LabelLayerManager


def _add_image_layer(
    sdata: SpatialData,
    arr: Array,
    output_layer: str,
    dims: tuple[str, ...] | None = None,
    chunks: str | tuple[int, int, int] | int | None = None,
    transformation: BaseTransformation | dict[str, BaseTransformation] = None,
    scale_factors: ScaleFactors_t | None = None,
    c_coords: list[str] | None = None,
    overwrite: bool = False,
):
    manager = ImageLayerManager()
    sdata = manager.add_layer(
        sdata,
        arr=arr,
        output_layer=output_layer,
        dims=dims,
        chunks=chunks,
        transformation=transformation,
        scale_factors=scale_factors,
        c_coords=c_coords,
        overwrite=overwrite,
    )

    return sdata


def _add_label_layer(
    sdata: SpatialData,
    arr: Array,
    output_layer: str,
    dims: tuple[str, ...] | None = None,
    chunks: str | tuple[int, int] | int | None = None,
    transformation: BaseTransformation | dict[str, BaseTransformation] = None,
    scale_factors: ScaleFactors_t | None = None,
    overwrite: bool = False,
):
    manager = LabelLayerManager()
    sdata = manager.add_layer(
        sdata,
        arr=arr,
        output_layer=output_layer,
        dims=dims,
        chunks=chunks,
        transformation=transformation,
        scale_factors=scale_factors,
        overwrite=overwrite,
    )

    return sdata
