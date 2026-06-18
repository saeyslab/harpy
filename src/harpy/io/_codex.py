from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import dask.array as da
import tifffile
from spatialdata import SpatialData, read_zarr
from spatialdata.models import Image2DModel
from spatialdata.transformations import Identity, Scale

_CHANNEL_NAME_FIELDS = (
    "Biomarker",
    "Name",
)
_DEFAULT_PYRAMID_MAX_SIZE = 2048
_DEFAULT_SCALE_FACTOR = 2


def codex(
    path: str | Path,
    image_name: str = "image",
    to_coordinate_system: str = "global",
    to_micron_coordinate_system: str | None = None,
    channel_names: Sequence[str] | None = None,
    series: int | None = None,
    level: int = 0,
    chunks: str | tuple[int, ...] | int | None = None,
    image_models_kwargs: Mapping[str, Any] = MappingProxyType({}),
    output: str | Path | None = None,
    overwrite: bool = False,
) -> SpatialData:
    """
    Read a PerkinElmer/Akoya CODEX QPTIFF image as a :class:`spatialdata.SpatialData` object.

    The pixel data is loaded lazily through :func:`tifffile.imread` with ``return_as="zarr"``.
    QPTIFF metadata is read from TIFF tags and PerkinElmer XML descriptions; channel coordinates
    are taken from the XML ``Biomarker`` field when available, falling back to ``Name``.

    Parameters
    ----------
    path
        Path to the QPTIFF file.
    image_name
        Name of the image element in the returned SpatialData object.
    to_coordinate_system
        Pixel coordinate system assigned to the image element.
    to_micron_coordinate_system
        Physical coordinate system assigned through a scale transform in micrometers. Defaults to
        ``f"{to_coordinate_system}_micron"`` when physical pixel size metadata is available.
    channel_names
        Optional explicit channel names. If provided, the length must match the number of channels.
    series
        TIFF series index to read. If ``None``, the first non-RGB ``CYX``-like series is selected.
    level
        Pyramid level to read. Defaults to full resolution.
    chunks
        Optional Dask rechunking applied before creating the SpatialData image element.
    image_models_kwargs
        Additional keyword arguments passed to :meth:`spatialdata.models.Image2DModel.parse`. If ``scale_factors``
        is omitted, a size-based default is used. Pass ``{"scale_factors": None}`` to opt out.
    output
        Optional path where the resulting SpatialData object should be written.
    overwrite
        Whether to overwrite ``output`` when writing a backed SpatialData object.

    Returns
    -------
    A SpatialData object containing one image element.

    Notes
    -----
    The returned unbacked object points lazily at the QPTIFF file. The underlying ``ZarrTiffStore``
    is suitable for local Dask execution but is not pickleable; write to ``output`` first when a
    durable Zarr-backed SpatialData object or distributed execution is needed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"QPTIFF file does not exist: {path}")

    metadata = _read_codex_metadata(path, series=series, level=level, channel_names=channel_names)
    data = _read_codex_level(path, series=metadata["series"], level=level)

    image_models_kwargs = dict(image_models_kwargs)
    chunks = chunks if chunks is not None else image_models_kwargs.pop("chunks", None)
    if chunks is not None:
        data = data.rechunk(chunks)

    if "scale_factors" not in image_models_kwargs:
        image_models_kwargs["scale_factors"] = _default_scale_factors(data.shape[-2:])

    transformations = {to_coordinate_system: Identity()}
    pixel_size = metadata["physical_pixel_size"]
    if pixel_size["x"] is not None and pixel_size["y"] is not None:
        micron_coordinate_system = to_micron_coordinate_system or f"{to_coordinate_system}_micron"
        transformations[micron_coordinate_system] = Scale(
            axes=("x", "y"),
            scale=[pixel_size["x"], pixel_size["y"]],
        )

    se = Image2DModel.parse(
        data,
        dims=["c", "y", "x"],
        c_coords=metadata["channel_names"],
        transformations=transformations,
        **image_models_kwargs,
    )

    sdata = SpatialData()
    sdata[_clean_element_name(image_name)] = se

    if output is not None:
        sdata.write(output, overwrite=overwrite)
        sdata = read_zarr(sdata.path)

    return sdata


def _read_codex_level(path: Path, *, series: int, level: int) -> da.Array:
    store = tifffile.imread(path, series=series, level=level, return_as="zarr")
    try:
        array = da.from_zarr(store)
    finally:
        store.close()

    if array.ndim != 3:
        raise ValueError(f"Expected a 3D CYX QPTIFF level, found shape {array.shape}.")
    return array


def _read_codex_metadata(
    path: Path,
    *,
    series: int | None,
    level: int,
    channel_names: Sequence[str] | None,
) -> dict[str, Any]:
    with tifffile.TiffFile(path) as tif:
        series_index = _select_codex_series(tif, series=series)
        tiff_series = tif.series[series_index]
        if level < 0 or level >= len(tiff_series.levels):
            raise ValueError(
                f"Level {level} not found in QPTIFF series {series_index}. "
                f"Available levels are 0..{len(tiff_series.levels) - 1}."
            )

        axes = tiff_series.axes.lower()
        if len(tiff_series.shape) != 3 or axes[-2:] != "yx" or "s" in axes:
            raise ValueError(
                f"Expected a non-RGB 3D channel-by-YX QPTIFF series, found axes={tiff_series.axes!r} "
                f"and shape={tiff_series.shape}."
            )

        channel_count = tiff_series.shape[0]
        pages = list(tiff_series.pages)
        page_metadata = [
            _parse_channel_name_fields(pages[i].description if i < len(pages) else None)
            for i in range(channel_count)
        ]

        if channel_names is None:
            parsed_channel_names = [
                fields.get("Biomarker") or fields.get("Name") or f"channel_{i}"
                for i, fields in enumerate(page_metadata)
            ]
        else:
            parsed_channel_names = list(channel_names)
            if len(parsed_channel_names) != channel_count:
                raise ValueError(
                    f"'channel_names' has length {len(parsed_channel_names)}, but the selected QPTIFF series "
                    f"contains {channel_count} channels."
                )

        base_page = pages[0]
        pixel_size_x, pixel_size_y = _physical_pixel_size_from_page(base_page)

        return {
            "series": series_index,
            "axes": tiff_series.axes,
            "channel_names": _make_unique_channel_names(parsed_channel_names),
            "physical_pixel_size": {
                "x": pixel_size_x,
                "y": pixel_size_y,
            },
        }


def _select_codex_series(tif: tifffile.TiffFile, *, series: int | None) -> int:
    if series is not None:
        if series < 0 or series >= len(tif.series):
            raise ValueError(f"Series {series} not found. Available series are 0..{len(tif.series) - 1}.")
        return series

    fallback: int | None = None
    for i, tiff_series in enumerate(tif.series):
        axes = tiff_series.axes.lower()
        if len(tiff_series.shape) != 3 or axes[-2:] != "yx" or "s" in axes:
            continue
        if axes == "cyx":
            return i
        fallback = i if fallback is None else fallback

    if fallback is not None:
        return fallback

    raise ValueError("Could not find a non-RGB 3D channel-by-YX series in the QPTIFF file.")


def _parse_channel_name_fields(description: str | None) -> dict[str, str]:
    if not description:
        return {}
    try:
        root = ET.fromstring(description)
    except ET.ParseError:
        return {}

    fields = {}
    for field in _CHANNEL_NAME_FIELDS:
        value = root.findtext(field)
        if value is not None:
            value = value.strip()
        if value:
            fields[field] = value
    return fields


def _physical_pixel_size_from_page(page: tifffile.TiffPage) -> tuple[float | None, float | None]:
    tags = page.tags
    x_resolution = _resolution_value(tags.get("XResolution"))
    y_resolution = _resolution_value(tags.get("YResolution"))
    resolution_unit = _resolution_unit(tags.get("ResolutionUnit"))

    if x_resolution is None or y_resolution is None or resolution_unit is None:
        return None, None

    unit_size_um = {"inch": 25_400.0, "centimeter": 10_000.0}.get(resolution_unit)
    if unit_size_um is None:
        return None, None

    return unit_size_um / x_resolution, unit_size_um / y_resolution


def _resolution_value(tag: tifffile.TiffTag | None) -> float | None:
    if tag is None:
        return None
    value = tag.value
    if isinstance(value, tuple | list):
        numerator, denominator = value
        if denominator == 0:
            return None
        return float(numerator) / float(denominator)
    value = float(value)
    return value if value != 0 else None


def _resolution_unit(tag: tifffile.TiffTag | None) -> str | None:
    if tag is None:
        return None
    value = tag.value
    try:
        value = int(value)
    except TypeError:
        value = int(value.value)

    return {1: "none", 2: "inch", 3: "centimeter"}.get(value)


def _make_unique_channel_names(channel_names: Sequence[str]) -> list[str]:
    counts: dict[str, int] = {}
    unique_names = []
    for i, channel_name in enumerate(channel_names):
        base = str(channel_name).strip() or f"channel_{i}"
        count = counts.get(base, 0)
        unique_names.append(base if count == 0 else f"{base}_{count + 1}")
        counts[base] = count + 1
    return unique_names


def _default_scale_factors(shape: Sequence[int]) -> list[int] | None:
    max_size = max(int(size) for size in shape)
    scale_factors = []
    while max_size > _DEFAULT_PYRAMID_MAX_SIZE:
        scale_factors.append(_DEFAULT_SCALE_FACTOR)
        max_size = (max_size + _DEFAULT_SCALE_FACTOR - 1) // _DEFAULT_SCALE_FACTOR
    return scale_factors or None


def _clean_element_name(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    return name or "image"
