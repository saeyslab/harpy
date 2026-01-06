from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import dask
import dask.array as da
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from geopandas import GeoDataFrame
from numpy.typing import NDArray
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from shapely.affinity import translate
from shapely.geometry import LineString, MultiLineString
from spatialdata import SpatialData
from spatialdata.models import get_axes_names
from spatialdata.transformations import get_transformation
from xarray import DataArray, DataTree

from harpy.utils._transformations import _get_translation_values


def _linestring_to_arrays(geometries):
    arrays = []
    for geometry in geometries:
        if isinstance(geometry, LineString):
            arrays.extend(list(geometry.coords))
        elif isinstance(geometry, MultiLineString):
            for item in geometry.geoms:
                arrays.extend(list(item.coords))
    return np.array(arrays)


# https://github.com/scverse/napari-spatialdata/blob/main/src/napari_spatialdata/_viewer.py#L105
def _get_polygons_in_napari_format(df: GeoDataFrame) -> list:
    polygons = []
    # affine = _get_transform(sdata.shapes[key], selected_cs)

    # when mulitpolygons are present, we select the largest ones
    if "MultiPolygon" in np.unique(df.geometry.type):
        # logger.info("Multipolygons are present in the data. Only the largest polygon per cell is retained.")
        df = df.explode(index_parts=False)
        df["area"] = df.area
        df = df.sort_values(by="area", ascending=False)  # sort by area
        df = df[~df.index.duplicated(keep="first")]  # only keep the largest area
        df.index = df.index.astype(int)  # convert index to integer
        df = df.sort_index()
        df.index = df.index.astype(str)

    if len(df) < 100:
        for i in range(0, len(df)):
            polygons.append(list(df.geometry.iloc[i].exterior.coords))
    else:
        for i in range(
            0, len(df)
        ):  # This can be removed once napari is sped up in the plotting. It changes the shapes only very slightly
            polygons.append(list(df.geometry.iloc[i].exterior.simplify(tolerance=2).coords))
    # this will only work for polygons and not for multipolygons
    # switch x,y positions of polygon indices, napari wants (y,x)
    polygons = _swap_coordinates(polygons)

    return polygons


def _translate_polygons(polygons: GeoDataFrame, to_coordinate_system: str = "global") -> GeoDataFrame:
    # get the transformation defined on "global"
    transformations = get_transformation(polygons, get_all=True)
    if to_coordinate_system not in [*transformations]:
        raise ValueError(
            f"'Coordinate system {to_coordinate_system}' does not appear to be a coordinate system of the spatial element. "
            f"Please choose a coordinate system from this list: {[*transformations]}."
        )
    transformation = transformations[to_coordinate_system]
    x_translation, y_translation = _get_translation_values(transformation)
    if x_translation != 0 or y_translation != 0:
        polygons["geometry"] = polygons["geometry"].apply(
            lambda geom: translate(geom, xoff=x_translation, yoff=y_translation)
        )

    return polygons


def _swap_coordinates(data: list[Any]) -> list[Any]:
    return [[(y, x) for x, y in sublist] for sublist in data]


def _get_raster_multiscale(element: DataTree) -> list[DataArray]:
    if not isinstance(element, DataTree):
        raise TypeError(f"Unsupported type for images or labels: {type(element)}")

    axes = get_axes_names(element)

    if "c" in axes:
        assert axes.index("c") == 0

    # sanity check
    scale_0 = element.__iter__().__next__()
    v = element[scale_0].values()
    assert len(v) == 1

    list_of_xdata = []
    for k in element:
        v = element[k].values()
        assert len(v) == 1
        xdata = v.__iter__().__next__()
        list_of_xdata.append(xdata)

    return list_of_xdata


def color(_) -> matplotlib.colors.Colormap:
    """Select random color from set1 colors."""
    return plt.get_cmap("Set1")(np.random.choice(np.arange(0, 18)))


def border_color(r: bool) -> matplotlib.colors.Colormap:
    """Select border color from tab10 colors or preset color (1, 1, 1, 1) otherwise."""
    return plt.get_cmap("tab10")(3) if r else (1, 1, 1, 1)


def linewidth(r: bool) -> float:
    """Select linewidth 1 if true else 0.5."""
    return 1 if r else 0.5


def _export_config(cfg: DictConfig, output_yaml: str | Path):
    yaml_config = OmegaConf.to_yaml(cfg)
    output_dir = os.path.dirname(output_yaml)
    os.makedirs(output_dir, exist_ok=True)
    with open(output_yaml, "w") as f:
        f.write(yaml_config)


def _get_uint_dtype(value: int) -> str:
    max_uint64 = np.iinfo(np.uint64).max
    max_uint32 = np.iinfo(np.uint32).max
    max_uint16 = np.iinfo(np.uint16).max
    max_uint8 = np.iinfo(np.uint8).max
    if max_uint8 >= value:
        dtype = "uint8"
    elif max_uint16 >= value:
        dtype = "uint16"
    elif max_uint32 >= value:
        dtype = "uint32"
    elif max_uint64 >= value:
        dtype = "uint64"
    else:
        raise ValueError(f"Maximum number is {value}. Values higher than {max_uint64} are not supported.")
    return dtype


def _self_contained_warning_message(sdata: SpatialData, layer: str) -> str | None:
    elements = sdata.elements_are_self_contained()
    if not elements[layer]:
        warning_message = (
            f"Element '{layer}' is Dask-backed, but the SpatialData object is not self-contained.\n"
            "To resolve this, ensure that you assign the result of operations:\n"
            "    sdata = harpy.{operation}(sdata, ...)\n"
            "Alternatively, manually reload from the Zarr store:\n"
            f"    spatialdata.read_zarr('{sdata.path}')\n"
            "For more details, see the discussion at:\n"
            "    https://github.com/saeyslab/harpy/issues/90"
        )

        return warning_message
    else:
        return None


def _dummy_embedding(array: NDArray, embedding_dimension: int, seed: int = 42) -> NDArray:
    rng = np.random.default_rng(seed)
    random_array = rng.random((array.shape[0], embedding_dimension), dtype=np.float32)
    return random_array


def _make_list(item: str | Iterable[str]) -> list[str]:
    if isinstance(item, str) or not isinstance(item, Iterable):
        return [item]
    return list(item)


def _affine_transform(coords: NDArray, transform_matrix: NDArray) -> NDArray:
    """
    Apply a 2D affine transformation to a set of coordinates.

    The input coordinates are interpreted as 2D points of shape ``(N, 2)``.
    They are augmented with a homogeneous dimension, multiplied by the
    provided 3×3 affine transformation matrix, and then converted back to
    standard 2D coordinates.

    Parameters
    ----------
    coords
        Array of shape ``(N, 2)`` containing the 2D coordinates to be
        transformed.
    transform_matrix
        Affine transformation matrix of shape ``(3, 3)``. It is applied
        in homogeneous coordinates as ``coords @ transform_matrix.T``.

    Raises
    ------
    AssertionError
        If `transform_matrix` does not have shape ``(3, 3)``.
    AssertionError
        If `coords` is not a 2D array of shape ``(N, 2)``.

    Returns
    -------
    Array of shape ``(N, 2)`` containing the transformed coordinates.
    """
    assert transform_matrix.shape == (3, 3)
    assert coords.ndim == 2
    assert coords.shape[1] == 2
    coords = np.hstack([coords, np.ones((coords.shape[0], 1))])
    coords = coords @ transform_matrix.T
    coords = coords[:, :2]
    return coords


def _dummy_statistic_image(array: NDArray, value: int) -> NDArray:
    np.random.seed(42)
    assert array.ndim == 2
    # shape of array=(c, number of pixels corresponding to non zero mask for instance i)
    C = array.shape[0]
    _statistic_dimension = 3
    # return dummy statistic of shape C, statistic_dimension
    return np.random.rand(C, _statistic_dimension) + value


def _dummy_statistic_mask(array: NDArray, value: int) -> NDArray:
    np.random.seed(42)
    # array should be of dtype int
    assert np.issubdtype(array.dtype, np.integer)
    # array is of shape = z,y,x, with y and x the size of the instance window.
    assert array.ndim == 3
    _statistic_dimension = 5
    result = np.random.rand(_statistic_dimension) + value
    # return array containing float of shape (statistic_dimension,)
    return result[None, ...]


def _get_xp(arr=None, run_on_gpu=True):
    """Returns (xp, is_cupy). If arr is provided, choose based on arr type (so you don't accidentally mix)."""
    if not run_on_gpu:
        return np, False
    try:
        import cupy as cp
    except ImportError:
        return np, False

    if arr is None:
        return cp, True

    # If arr is already a CuPy array, stick with CuPy
    if isinstance(arr, cp.ndarray):
        return cp, True

    return np, False


def _to_cupy_dask_array(arr: da.Array) -> da.Array:
    import cupy as cp

    x_cu = arr.map_blocks(
        cp.asarray,
        dtype=arr.dtype,
        meta=cp.empty((0,) * arr.ndim, dtype=arr.dtype),
    )
    return x_cu


def _to_numpy(x) -> NDArray:
    """Return a NumPy array (no-op for NumPy; explicit copy to host for CuPy)."""
    try:
        import cupy as cp

        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
    except ImportError:
        pass
    return np.asarray(x)


def _da_unique(arr: da.Array, run_on_gpu: bool = True) -> NDArray:  # FIXME fix type
    xp, _ = _get_xp(getattr(arr, "_meta", None), run_on_gpu=run_on_gpu)

    # build delayed blocks
    blocks = arr.to_delayed().ravel()

    @dask.delayed
    def unique_block(b):
        return xp.unique(b)

    uniques = [unique_block(b) for b in blocks]

    uniques_np = dask.compute(*uniques)
    return xp.unique(xp.concatenate(uniques_np))
