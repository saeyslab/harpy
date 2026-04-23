from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import spatialdata
from dask.array import Array
from loguru import logger as log
from spatialdata import SpatialData, read_zarr
from spatialdata.models._utils import MappingToCoordinateSystem_t
from spatialdata.models.models import ScaleFactors_t
from xarray import DataArray, DataTree

from harpy.utils._io import _incremental_io_on_disk, _write_element_with_cleanup


class ElementManager(ABC):
    def add_element(
        self,
        sdata: SpatialData,
        arr: Array,
        element_name: str,
        dims: tuple[str, ...] | None = None,
        chunks: str | tuple[int, ...] | int | None = None,
        transformations: MappingToCoordinateSystem_t | None = None,
        scale_factors: ScaleFactors_t | None = None,
        overwrite: bool = False,
        **kwargs: Any,  # kwargs passed to create_spatial_element
    ) -> SpatialData:
        chunks = chunks or arr.chunksize
        if dims is None:
            dims = self.get_dims(arr)

        if not sdata.is_backed():
            # if sdata is not backed, we do a persist to prevent recomputation
            arr = arr.persist()

        spatial_element = self.create_spatial_element(
            arr,
            dims=dims,
            scale_factors=scale_factors,
            chunks=chunks,
            transformations=transformations,
            **kwargs,
        )

        log.info(f"Writing results to element '{element_name}'")

        sdata = self.add_to_sdata(
            sdata,
            element_name=element_name,
            spatial_element=spatial_element,
            overwrite=overwrite,
        )

        return sdata

    @abstractmethod
    def create_spatial_element(
        self,
        arr: Array,
        dims: tuple[int, ...],
        scale_factors: ScaleFactors_t | None = None,
        chunks: str | tuple[int, ...] | int | None = None,
        **kwargs: Any,
    ) -> DataArray | DataTree:
        pass

    @abstractmethod
    def get_dims(self) -> tuple[str, ...]:
        pass

    @abstractmethod
    def add_to_sdata(
        self,
        sdata: SpatialData,
        element_name: str,
        spatial_element: DataArray | DataTree,
        overwrite: bool = False,
    ) -> SpatialData:
        pass

    @abstractmethod
    def retrieve_data_from_sdata(self, sdata: SpatialData, name: str) -> SpatialData:
        pass


class ImageElementManager(ElementManager):
    def create_spatial_element(
        self,
        arr: Array,
        dims: tuple[str, ...],
        scale_factors: ScaleFactors_t | None = None,
        chunks: str | tuple[int, int, int] | int | None = None,
        transformations: MappingToCoordinateSystem_t | None = None,
        c_coords: list[str] | None = None,
    ) -> DataArray | DataTree:
        if len(dims) == 3:
            return spatialdata.models.Image2DModel.parse(
                arr,
                dims=dims,
                scale_factors=scale_factors,
                chunks=chunks,
                c_coords=c_coords,
                transformations=transformations,
            )
        elif len(dims) == 4:
            return spatialdata.models.Image3DModel.parse(
                arr,
                dims=dims,
                scale_factors=scale_factors,
                chunks=chunks,
                c_coords=c_coords,
                transformations=transformations,
            )
        else:
            raise ValueError(
                f"Provided dims is {dims}, which is not supported, "
                "please provide dims parameter that only contains c, (z), y, and x."
            )

    def get_dims(self, arr) -> tuple[str, ...]:
        if len(arr.shape) == 3:
            return ("c", "y", "x")
        elif len(arr.shape) == 4:
            return ("c", "z", "y", "x")
        else:
            raise ValueError("Only 2D and 3D images (c, (z), y, x) are currently supported.")

    def add_to_sdata(
        self,
        sdata: SpatialData,
        element_name: str,
        spatial_element: DataArray | DataTree,
        overwrite: bool = False,
    ) -> SpatialData:
        # given a spatial_element with some graph defined on it.
        if element_name in [*sdata.images]:
            if sdata.is_backed():
                if overwrite:
                    sdata = _incremental_io_on_disk(sdata, element_name=element_name, element=spatial_element, element_type="images")
                else:
                    raise ValueError(
                        f"Attempting to overwrite 'sdata.images[\"{element_name}\"]', but overwrite is set to False. Set overwrite to True to overwrite the .zarr store."
                    )
            else:
                sdata[element_name] = spatial_element

        else:
            sdata[element_name] = spatial_element
            if sdata.is_backed():
                # to make sdata point to the materialized element while keeping object identity.
                _write_element_with_cleanup(sdata, element_name)
                del sdata[element_name]
                sdata_temp = read_zarr(sdata.path, selection=["images"])
                sdata[element_name] = sdata_temp[element_name]
                del sdata_temp

        return sdata

    def retrieve_data_from_sdata(self, sdata: SpatialData, name: str) -> Array:
        return sdata.images[name].data


class LabelsElementManager(ElementManager):
    def create_spatial_element(
        self,
        arr: Array,
        dims: tuple[str, str],
        scale_factors: ScaleFactors_t | None = None,
        chunks: str | tuple[int, int] | int | None = None,
        transformations: MappingToCoordinateSystem_t | None = None,
    ) -> DataArray | DataTree:
        if len(dims) == 2:
            return spatialdata.models.Labels2DModel.parse(
                arr,
                dims=dims,
                scale_factors=scale_factors,
                chunks=chunks,
                transformations=transformations,
            )
        elif len(dims) == 3:
            return spatialdata.models.Labels3DModel.parse(
                arr, dims=dims, scale_factors=scale_factors, chunks=chunks, transformations=transformations
            )
        else:
            raise ValueError(
                f"Provided dims is {dims}, which is not supported, "
                "please provide dims parameter that only contains (z), y and x."
            )

    def get_dims(self, arr) -> tuple[str, str]:
        if len(arr.shape) == 2:
            return ("y", "x")
        elif len(arr.shape) == 3:
            return ("z", "y", "x")
        else:
            raise ValueError("Only 2D and 3D labels elements ( (z), y, x) are currently supported.")

    def add_to_sdata(
        self,
        sdata: SpatialData,
        element_name: str,
        spatial_element: DataArray | DataTree,
        overwrite: bool = False,
    ) -> SpatialData:
        # given a spatial_element with some graph defined on it.
        if element_name in [*sdata.labels]:
            if sdata.is_backed():
                if overwrite:
                    sdata = _incremental_io_on_disk(sdata, element_name=element_name, element=spatial_element, element_type="labels")
                else:
                    raise ValueError(
                        f"Attempting to overwrite 'sdata.labels[\"{element_name}\"]', but overwrite is set to False. Set overwrite to True to overwrite the .zarr store."
                    )
            else:
                sdata[element_name] = spatial_element
        else:
            sdata[element_name] = spatial_element
            if sdata.is_backed():
                _write_element_with_cleanup(sdata, element_name)
                del sdata[element_name]
                sdata_temp = read_zarr(sdata.path, selection=["labels"])
                sdata[element_name] = sdata_temp[element_name]
                del sdata_temp

        return sdata

    def retrieve_data_from_sdata(self, sdata: SpatialData, name: str) -> Array:
        return sdata.labels[name].data
