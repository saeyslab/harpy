from __future__ import annotations

import importlib
from functools import singledispatchmethod
from typing import Any

import dask
import geopandas
import numpy as np
import pandas as pd
import shapely
import skimage
import spatialdata
from dask.array import Array
from geopandas import GeoDataFrame
from shapely import MultiPolygon, Polygon
from skimage.measure._regionprops import RegionProperties
from spatialdata import SpatialData, read_zarr
from spatialdata.models._utils import MappingToCoordinateSystem_t
from spatialdata.transformations import get_transformation

from harpy.utils._io import _incremental_io_on_disk
from harpy.utils._keys import _INSTANCE_KEY, _REGION_KEY
from harpy.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class ShapesLayerManager:
    def add_shapes(
        self,
        sdata: SpatialData,
        input: Array | GeoDataFrame,
        output_layer: str,
        transformations: MappingToCoordinateSystem_t = None,
        overwrite: bool = False,
    ) -> SpatialData:
        polygons = self.get_polygons_from_input(input)

        if polygons.empty:
            log.warning(
                f"GeoDataFrame contains no polygons. Skipping addition of shapes layer '{output_layer}' to sdata."
            )
            return sdata

        polygons = spatialdata.models.ShapesModel.parse(polygons, transformations=transformations)

        sdata = self.add_to_sdata(
            sdata,
            output_layer=output_layer,
            spatial_element=polygons,
            overwrite=overwrite,
        )

        return sdata

    def filter_shapes(
        self,
        sdata: SpatialData,
        table_layer: str,
        labels_layer: str,
        prefix_filtered_shapes_layer: str,
    ) -> SpatialData:
        mask = sdata.tables[table_layer].obs[_REGION_KEY].isin([labels_layer])
        indexes_to_keep = sdata.tables[table_layer].obs[mask][_INSTANCE_KEY].values.astype(int)
        coordinate_systems_labels_layer = {*get_transformation(sdata.labels[labels_layer], get_all=True)}

        if len(indexes_to_keep) == 0:
            log.warning(
                "Length of the 'indexes_to_keep' parameter is 0. "
                "This would remove all shapes from sdata`. Skipping filtering step."
            )
            return sdata

        for _shapes_layer in [*sdata.shapes]:
            polygons = self.retrieve_data_from_sdata(sdata, name=_shapes_layer)
            polygons = self.get_polygons_from_input(polygons)
            # only filter shapes that are in same coordinate system as the labels layer
            if not set(coordinate_systems_labels_layer).intersection({*get_transformation(polygons, get_all=True)}):
                continue

            current_indexes_shapes_layer = polygons.index.values.astype(int)

            bool_to_keep = np.isin(current_indexes_shapes_layer, indexes_to_keep)

            if sum(bool_to_keep) == 0:
                # no overlap, this means data.shapes[_shapes_layer] is
                # a polygons layer containing polygons filtered out in a previous step
                continue

            output_filtered_shapes_layer = f"{prefix_filtered_shapes_layer}_{_shapes_layer}"

            if sum(~bool_to_keep) == 0:
                # this is case where there are no polygons filtered out, so no
                # output_filtered_shapes_layer should be created
                log.warning(
                    f"No polygons filtered out for shapes layer '{_shapes_layer}'. As a result, "
                    f"shapes layer '{output_filtered_shapes_layer}' will not be created. This is "
                    f"expected if 'indexes_to_keep' matches '{_shapes_layer}' indexes."
                )

                continue

            filtered_polygons = self.retrieve_data_from_sdata(sdata, name=_shapes_layer)[~bool_to_keep]

            log.info(
                f"Filtering {len(set(filtered_polygons.index))} cells from shapes layer '{_shapes_layer}'. "
                f"Adding new shapes layer '{output_filtered_shapes_layer}' containing these filtered out polygons."
            )

            # if this assert would break in future spatialdata, then pass transformations of polygons to .parse
            assert get_transformation(filtered_polygons, get_all=True) == get_transformation(polygons, get_all=True)
            sdata = self.add_to_sdata(
                sdata,
                output_layer=output_filtered_shapes_layer,
                spatial_element=spatialdata.models.ShapesModel.parse(filtered_polygons),
                overwrite=True,
            )

            updated_polygons = self.retrieve_data_from_sdata(sdata, name=_shapes_layer)[bool_to_keep]

            assert get_transformation(updated_polygons, get_all=True) == get_transformation(polygons, get_all=True)
            sdata = self.add_to_sdata(
                sdata,
                output_layer=_shapes_layer,
                spatial_element=spatialdata.models.ShapesModel.parse(updated_polygons),
                overwrite=True,
            )

        return sdata

    @singledispatchmethod
    def get_polygons_from_input(self, input: Any) -> GeoDataFrame:
        raise ValueError("Unsupported input type.")

    @get_polygons_from_input.register(Array)
    def _get_polygons_from_array(self, input: Array) -> GeoDataFrame:
        assert np.issubdtype(input.dtype, np.integer), "Only integer arrays are supported."
        dimension = self.get_dims(input)
        if importlib.util.find_spec("rasterio") is not None:
            _mask_to_polygons = _mask_to_polygons_rasterio
        else:
            log.info(
                "'rasterio' library is not installed. Falling back to 'skimage' for mask vectorization. "
                "For better performance and more precise vectorization, consider installing 'rasterio'."
            )
            _mask_to_polygons = _mask_to_polygons_skimage
        if dimension == 3:
            all_polygons = []
            for z_slice in range(input.shape[0]):
                polygons = _mask_to_polygons(input[z_slice], z_slice=z_slice)
                all_polygons.append(polygons)
            polygons = geopandas.GeoDataFrame(pd.concat(all_polygons, ignore_index=False))
            return polygons
        elif dimension == 2:
            return _mask_to_polygons(input)

    @get_polygons_from_input.register(GeoDataFrame)
    def _get_polygons_from_geodf(self, input: GeoDataFrame) -> GeoDataFrame:
        self.get_dims(input)
        return input

    @singledispatchmethod
    def get_dims(self, input: Any):
        raise ValueError("Unsupported input type.")

    @get_dims.register(Array)
    def _get_dims_array(self, input):
        assert len(input.shape) in [
            2,
            3,
        ], "Only 2D (y,x) and 3D (z,y,x) labels layers are supported."

        return len(input.shape)

    @get_dims.register(GeoDataFrame)
    def _get_dims_gdf(self, input):
        has_z = input["geometry"].apply(lambda geom: geom.has_z)
        if all(has_z):
            return 3
        elif not any(has_z):
            return 2
        else:
            raise ValueError("All geometries should either be 2D or 3D. Mixed dimensions found.")

    def add_to_sdata(
        self,
        sdata: SpatialData,
        output_layer: str,
        spatial_element: GeoDataFrame,
        overwrite: bool = False,
    ) -> SpatialData:
        # given a spatial_element
        if output_layer in [*sdata.shapes]:
            if sdata.is_backed():
                if overwrite:
                    sdata = _incremental_io_on_disk(
                        sdata, output_layer=output_layer, element=spatial_element, element_type="shapes"
                    )
                else:
                    raise ValueError(
                        f"Attempting to overwrite 'sdata.shapes[\"{output_layer}\"]', but overwrite is set to False. Set overwrite to True to overwrite the .zarr store."
                    )
            else:
                sdata[output_layer] = spatial_element
        else:
            sdata[output_layer] = spatial_element
            if sdata.is_backed():
                sdata.write_element(output_layer)
                del sdata[output_layer]
                sdata_temp = read_zarr(sdata.path, selection=["shapes"])
                sdata[output_layer] = sdata_temp[output_layer]
                del sdata_temp

        return sdata

    def retrieve_data_from_sdata(self, sdata: SpatialData, name: str) -> GeoDataFrame:
        return sdata.shapes[name]

    def remove_from_sdata(self, sdata: SpatialData, name: str) -> SpatialData:
        element_type = sdata._element_type_from_element_name(name)
        del getattr(sdata, element_type)[name]
        return sdata


def _mask_to_polygons_rasterio(mask: Array, z_slice: int = None) -> GeoDataFrame:
    """
    Convert a cell segmentation mask to polygons and return them as a GeoDataFrame using `rasterio`.

    This function computes the polygonal outlines of the cells present in the
    given segmentation mask. The polygons are calculated in parallel using Dask
    delayed computations.
    For optimal performance it is recommended to configure `Dask` so it uses "processes" instead of "threads".
    E.g. via:
    >>> import dask
    >>> dask.config.set(scheduler='processes')

    Parameters
    ----------
    mask : dask.array.core.Array
        A Dask array representing the segmentation mask. Non-zero pixels belong
        to a cell; pixels with the same intensity value belong to the same cell.
        Zero pixels represent background (no cell).
    z_slice: int or None, optional.
        The z slice that is being processed.

    Returns
    -------
    A GeoDataFrame containing polygons extracted from the input mask. Each polygon
    is associated with a cell ID.

    Notes
    -----
    The mask is processed in chunks to facilitate parallel computation. Polygons that
    are actually pieces of the same cell are combined back together to form coherent
    cells. This is necessary due to image chunking during parallel processing.

    Examples
    --------
    >>> import numpy as np
    >>> import dask.array as da
    >>> from harpy.shape._manager import _mask_to_polygons_rasterio
    >>> mask = da.from_array(np.array([[0, 3], [5, 5]]), chunks=(1, 1))
    >>> gdf = _mask_to_polygons_rasterio(mask)
    >>> gdf
                                                    geometry
    cells
    3      POLYGON ((1.00000 0.00000, 1.00000 1.00000, 2....
    5      POLYGON ((0.00000 2.00000, 1.00000 2.00000, 2....
    >>> gdf.geometry[3]
    <POLYGON ((1 0, 1 1, 2 1, 2 0, 1 0))>
    >>> gdf.geometry[5]
    <POLYGON ((0 2, 1 2, 2 2, 2 1, 1 1, 0 1, 0 2))>
    """
    from rasterio import Affine
    from rasterio.features import shapes

    assert mask.ndim == 2, "Only 2D dask arrays are supported."
    assert np.issubdtype(mask.dtype, np.integer), "Only integer arrays are supported."

    def _get_dtype(value: int) -> np.dtype:
        max_int16 = np.iinfo(np.int16).max
        max_int32 = np.iinfo(np.int32).max

        if max_int16 >= value:
            dtype = np.int16
        elif max_int32 >= value:
            dtype = np.int32
        else:
            raise ValueError(
                f"Maximum cell number is {value}. Values higher than {max_int32} are not supported. Consider relabeling the cells."
            )
        return dtype

    # Define a function to extract polygons and values from each chunk
    def _vectorize_chunk(mask_chunk: np.ndarray, y_offset: int, x_offset: int) -> GeoDataFrame:
        all_polygons = []
        all_values = []

        # Compute the boolean mask before passing it to the features.shapes() function
        bool_mask = mask_chunk > 0

        max_value = mask_chunk.max()
        dtype = _get_dtype(max_value)

        for shape, value in shapes(
            mask_chunk.astype(dtype),
            mask=bool_mask,
            transform=Affine(1.0, 0, x_offset, 0, 1.0, y_offset),
        ):
            if z_slice is not None:
                coordinates = shape["coordinates"]
                shape["coordinates"] = [[(*item, z_slice) for item in coord] for coord in coordinates]
            all_polygons.append(shapely.geometry.shape(shape))
            all_values.append(int(value))

        return geopandas.GeoDataFrame({"geometry": all_polygons, _INSTANCE_KEY: all_values})

    # Create a list of delayed objects
    chunk_sizes = mask.chunks

    tasks = [
        dask.delayed(_vectorize_chunk)(chunk, sum(chunk_sizes[0][:iy]), sum(chunk_sizes[1][:ix]))
        for iy, row in enumerate(mask.to_delayed())
        for ix, chunk in enumerate(row)
    ]
    results = dask.compute(*tasks)

    gdf = pd.concat(results)

    gdf[_INSTANCE_KEY] = gdf[_INSTANCE_KEY].astype(mask.dtype)

    log.info(
        "Finished vectorizing. Dissolving shapes at the border of the chunks. "
        "This can take a couple minutes if input mask contains a lot of chunks."
    )
    gdf = gdf.dissolve(by=_INSTANCE_KEY)

    log.info("Dissolve is done.")

    return gdf


def _mask_to_polygons_skimage(mask: Array, z_slice=None) -> GeoDataFrame:
    """
    Convert a cell segmentation mask to polygons and return them as a GeoDataFrame using `skimage`.

    This function computes the polygonal outlines of the cells present in the
    given segmentation mask. The polygons are calculated in parallel using Dask
    delayed computations.
    For optimal performance it is recommended to configure `Dask` so it uses "processes" instead of "threads".
    E.g. via:
    >>> import dask
    >>> dask.config.set(scheduler='processes')

    Parameters
    ----------
    mask : dask.array.core.Array
        A Dask array representing the segmentation mask. Non-zero pixels belong
        to a cell; pixels with the same intensity value belong to the same cell.
        Zero pixels represent background (no cell).
    z_slice: int or None, optional.
        The z slice that is being processed.

    Returns
    -------
    A GeoDataFrame containing polygons extracted from the input mask. Each polygon
    is associated with a cell ID.
    """

    # taken from spatialdata
    # https://github.com/scverse/spatialdata/blob/27bb4a7579d8ff7cc8f6dd9b782226cb984ceb20/src/spatialdata/_core/operations/vectorize.py#L181
    def _dissolve_on_overlaps(label: int, group: GeoDataFrame) -> GeoDataFrame:
        if len(group) == 1:
            return (label, group.geometry.iloc[0])
        if len(np.unique(group["chunk-location"])) == 1:
            return (label, MultiPolygon(list(group.geometry)))
        return (label, group.dissolve().geometry.iloc[0])

    def _vectorize_chunk(chunk: np.ndarray, yoff: int, xoff: int) -> GeoDataFrame:  # type: ignore[type-arg]
        gdf = _vectorize_mask(chunk)
        gdf["chunk-location"] = f"({yoff}, {xoff})"
        gdf.geometry = gdf.translate(xoff, yoff)
        return gdf

    def _region_props_to_polygons(region_props: RegionProperties) -> list[Polygon]:
        mask = np.pad(region_props.image, 1)
        contours = skimage.measure.find_contours(mask, 0.5)

        # shapes with <= 3 vertices, i.e. lines, can't be converted into a polygon
        if z_slice is None:
            polygons = [Polygon(contour[:, [1, 0]]) for contour in contours if contour.shape[0] >= 4]
        else:
            polygons = [
                Polygon(np.hstack((contour[:, [1, 0]], np.full((contour.shape[0], 1), z_slice))))
                for contour in contours
                if contour.shape[0] >= 4
            ]

        yoff, xoff, *_ = region_props.bbox
        return [shapely.affinity.translate(poly, xoff, yoff) for poly in polygons]

    def _vectorize_mask(
        mask: np.ndarray,  # type: ignore[type-arg]
    ) -> GeoDataFrame:
        if mask.max() == 0:
            return GeoDataFrame(geometry=[])

        regions = skimage.measure.regionprops(mask)

        polygons_list = [_region_props_to_polygons(region) for region in regions]
        geoms = [poly for polygons in polygons_list for poly in polygons]
        labels = [region.label for i, region in enumerate(regions) for _ in range(len(polygons_list[i]))]

        return GeoDataFrame({"label": labels}, geometry=geoms)

    chunk_sizes = mask.chunks

    tasks = [
        dask.delayed(_vectorize_chunk)(chunk, sum(chunk_sizes[0][:iy]), sum(chunk_sizes[1][:ix]))
        for iy, row in enumerate(mask.to_delayed())
        for ix, chunk in enumerate(row)
    ]
    results = dask.compute(*tasks)

    gdf = pd.concat(results)
    gdf = GeoDataFrame([_dissolve_on_overlaps(*item) for item in gdf.groupby("label")], columns=["label", "geometry"])
    gdf.index = gdf["label"].astype(mask.dtype)
    gdf.index.name = _INSTANCE_KEY
    gdf.drop("label", axis=1, inplace=True)

    return gdf
