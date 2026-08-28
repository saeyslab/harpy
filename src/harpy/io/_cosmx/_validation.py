from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dask.dataframe import DataFrame as DaskDataFrame
from spatialdata import SpatialData, read_zarr

from harpy._metadata import (
    _FEATURE_PANELS_METADATA_KEY,
    _HARPY_METADATA_KEY,
    _IMAGES_METADATA_KEY,
    _LABELS_METADATA_KEY,
    _METADATA_VERSION,
    _METADATA_VERSION_KEY,
    _POINTS_METADATA_KEY,
    _PROVENANCE_METADATA_KEY,
)
from harpy.image._image import get_dataarray
from harpy.io._cosmx._constants import _COMPARTMENT_CATEGORIES, _INSTANCE_ID_FORMULA
from harpy.io._cosmx._models import _INSTANCE_ID_DTYPE, _MOSAIC_MODES, _validate_identifier
from harpy.io._cosmx._transcripts import _feature_panel_name

_ELEMENT_REGISTRIES = (_IMAGES_METADATA_KEY, _LABELS_METADATA_KEY, _POINTS_METADATA_KEY)


@dataclass(frozen=True)
class _FeaturePanelContract:
    feature_column: str
    class_column: str
    categories: tuple[str, ...]
    target_classes: dict[str, str]


def validate_cosmx_store(
    output: str | Path,
    *,
    check_point_contents: bool = False,
) -> None:
    """Validate a sample-aware CosMx SpatialData Zarr store without modifying it.

    Structural validation checks the Harpy metadata version and provenance,
    registered element records, label-family metadata, feature-panel records
    and referenced points schemas. It reads array and dataframe metadata but
    does not compute raster or transcript partitions.

    The optional content check projects only the feature and feature-class
    columns from each panel-associated points element, computes their distinct
    observed pairs out of core, and verifies those pairs against the referenced
    authoritative panel. Panel targets with zero detections remain valid.

    Parameters
    ----------
    output
        Existing SpatialData Zarr store created by the sample-aware CosMx
        reader.
    check_point_contents
        Whether to scan the two panel-declared categorical columns in referenced
        points elements. By default only their lazy schemas are checked.

    Raises
    ------
    ValueError
        If the output cannot be opened as a backed SpatialData store or any
        reader-owned metadata, registered element, points schema, or requested
        feature-panel content check violates the CosMx store contract.

    Notes
    -----
    This function never writes attributes or elements and does not repair or
    migrate invalid stores. Unregistered downstream-analysis elements are
    permitted, but they do not establish CosMx sample identity.
    """
    if not isinstance(check_point_contents, bool):
        raise ValueError(f"CosMx check_point_contents must be a bool, found {check_point_contents!r}.")
    try:
        output_path = Path(output).expanduser()
    except TypeError as error:
        raise ValueError(f"CosMx output must be a path, found {output!r}.") from error
    try:
        sdata = read_zarr(output_path)
    except Exception as error:
        raise ValueError(f"Could not read CosMx SpatialData Zarr store: {output_path}") from error
    _validate_cosmx_sdata(sdata, check_point_contents=check_point_contents)


def _validate_cosmx_sdata(
    sdata: SpatialData,
    *,
    check_point_contents: bool = False,
) -> frozenset[str]:
    """Validate an opened store and return its element-declared sample IDs."""
    if not isinstance(check_point_contents, bool):
        raise ValueError(f"CosMx check_point_contents must be a bool, found {check_point_contents!r}.")
    if not isinstance(sdata, SpatialData) or not sdata.is_backed():
        raise ValueError("CosMx store validation requires a backed SpatialData object.")

    harpy_metadata = _require_mapping(
        sdata.attrs.get(_HARPY_METADATA_KEY),
        path=_HARPY_METADATA_KEY,
    )
    version = harpy_metadata.get(_METADATA_VERSION_KEY)
    if isinstance(version, bool) or not isinstance(version, int) or version != _METADATA_VERSION:
        raise ValueError(
            f"CosMx metadata {_HARPY_METADATA_KEY}.{_METADATA_VERSION_KEY} must equal "
            f"{_METADATA_VERSION}, found {version!r}."
        )
    provenance = _require_mapping(
        harpy_metadata.get(_PROVENANCE_METADATA_KEY),
        path=f"{_HARPY_METADATA_KEY}.{_PROVENANCE_METADATA_KEY}",
    )
    if provenance.get("reader") != "cosmx":
        raise ValueError(
            f"CosMx metadata harpy.provenance.reader must equal 'cosmx', found {provenance.get('reader')!r}."
        )
    _require_nonempty_string(
        provenance.get("reader_version"),
        path="harpy.provenance.reader_version",
    )

    registries = {
        registry: _optional_mapping(
            harpy_metadata.get(registry),
            path=f"{_HARPY_METADATA_KEY}.{registry}",
        )
        for registry in (*_ELEMENT_REGISTRIES, _FEATURE_PANELS_METADATA_KEY)
    }
    if not any(registries[registry] for registry in _ELEMENT_REGISTRIES):
        raise ValueError("CosMx metadata must register at least one image, labels, or points element.")

    panels = _validate_feature_panels(registries[_FEATURE_PANELS_METADATA_KEY])
    sample_ids: set[str] = set()
    sample_ids.update(_validate_image_records(sdata, registries[_IMAGES_METADATA_KEY]))
    sample_ids.update(_validate_labels_records(sdata, registries[_LABELS_METADATA_KEY]))
    sample_ids.update(
        _validate_points_records(
            sdata,
            registries[_POINTS_METADATA_KEY],
            panels=panels,
            check_point_contents=check_point_contents,
        )
    )
    return frozenset(sample_ids)


def _validate_image_records(sdata: SpatialData, registry: Mapping[str, object]) -> set[str]:
    sample_ids: set[str] = set()
    for element_name, value in registry.items():
        path = _element_path(_IMAGES_METADATA_KEY, element_name)
        _require_registered_element(sdata.images, element_name, element_type="image", path=path)
        record = _require_mapping(value, path=path)
        sample_ids.add(_validate_common_element_record(record, path=path))

        channels = record.get("channels")
        if not isinstance(channels, list) or not channels:
            raise ValueError(f"CosMx metadata {path}.channels must be a non-empty list.")
        channel_ids: list[str] = []
        source_planes: list[int] = []
        output_coordinates: list[str] = []
        for index, channel_value in enumerate(channels):
            channel_path = f"{path}.channels[{index}]"
            channel = _require_mapping(channel_value, path=channel_path)
            channel_ids.append(_require_nonempty_string(channel.get("channel_id"), path=f"{channel_path}.channel_id"))
            _require_nonempty_string(channel.get("name"), path=f"{channel_path}.name")
            source_planes.append(
                _require_nonnegative_integer(channel.get("source_plane"), path=f"{channel_path}.source_plane")
            )
            output_coordinates.append(
                _require_nonempty_string(
                    channel.get("output_coordinate"),
                    path=f"{channel_path}.output_coordinate",
                )
            )
        _require_unique(channel_ids, path=f"{path}.channels.channel_id")
        _require_unique(source_planes, path=f"{path}.channels.source_plane")
        _require_unique(output_coordinates, path=f"{path}.channels.output_coordinate")

        array = get_dataarray(sdata, element_name)
        if "c" not in array.dims or "c" not in array.coords:
            raise ValueError(f"CosMx image element {element_name!r} must have a channel coordinate.")
        actual_coordinates = array.coords["c"].values.tolist()
        if actual_coordinates != output_coordinates:
            raise ValueError(
                f"CosMx metadata {path}.channels output coordinates {output_coordinates} "
                f"do not match image coordinates {actual_coordinates}."
            )
    return sample_ids


def _validate_labels_records(sdata: SpatialData, registry: Mapping[str, object]) -> set[str]:
    sample_ids: set[str] = set()
    for element_name, value in registry.items():
        path = _element_path(_LABELS_METADATA_KEY, element_name)
        _require_registered_element(sdata.labels, element_name, element_type="labels", path=path)
        record = _require_mapping(value, path=path)
        sample_ids.add(_validate_common_element_record(record, path=path))

        has_instance_encoding = "instance_id_encoding" in record
        has_categories = "categories" in record
        if has_instance_encoding == has_categories:
            raise ValueError(
                f"CosMx metadata {path} must contain exactly one of 'instance_id_encoding' or 'categories'."
            )

        array = get_dataarray(sdata, element_name)
        if has_instance_encoding:
            encoding = _require_mapping(record["instance_id_encoding"], path=f"{path}.instance_id_encoding")
            background = encoding.get("background")
            if isinstance(background, bool) or not isinstance(background, int) or background != 0:
                raise ValueError(f"CosMx metadata {path}.instance_id_encoding.background must equal 0.")
            base = encoding.get("base")
            if isinstance(base, bool) or not isinstance(base, int) or base < 2 or base & (base - 1) != 0:
                raise ValueError(
                    f"CosMx metadata {path}.instance_id_encoding.base must be a positive power of two, found {base!r}."
                )
            if encoding.get("formula") != _INSTANCE_ID_FORMULA:
                raise ValueError(
                    f"CosMx metadata {path}.instance_id_encoding.formula must equal {_INSTANCE_ID_FORMULA!r}."
                )
            if np.dtype(array.dtype) != _INSTANCE_ID_DTYPE:
                raise ValueError(
                    f"CosMx instance-label element {element_name!r} must have dtype "
                    f"{_INSTANCE_ID_DTYPE.name}, found {array.dtype}."
                )
        else:
            categories = _normalize_compartment_categories(record["categories"], path=f"{path}.categories")
            if categories != _COMPARTMENT_CATEGORIES:
                raise ValueError(
                    f"CosMx metadata {path}.categories must equal {_COMPARTMENT_CATEGORIES}, found {categories}."
                )
            if np.dtype(array.dtype).kind != "u":
                raise ValueError(
                    f"CosMx compartment-label element {element_name!r} must use an unsigned integer dtype, "
                    f"found {array.dtype}."
                )
    return sample_ids


def _validate_points_records(
    sdata: SpatialData,
    registry: Mapping[str, object],
    *,
    panels: Mapping[str, _FeaturePanelContract],
    check_point_contents: bool,
) -> set[str]:
    sample_ids: set[str] = set()
    for element_name, value in registry.items():
        path = _element_path(_POINTS_METADATA_KEY, element_name)
        _require_registered_element(sdata.points, element_name, element_type="points", path=path)
        record = _require_mapping(value, path=path)
        sample_ids.add(_validate_common_element_record(record, path=path))

        panel_name = record.get("feature_panel")
        if panel_name is None:
            continue
        panel_name = _require_nonempty_string(panel_name, path=f"{path}.feature_panel")
        if panel_name not in panels:
            raise ValueError(f"CosMx metadata {path}.feature_panel references missing panel {panel_name!r}.")
        panel = panels[panel_name]
        points = sdata.points[element_name]
        for column in (panel.feature_column, panel.class_column):
            if column not in points.columns:
                raise ValueError(f"CosMx points element {element_name!r} is missing panel-declared column {column!r}.")
            dtype = points.dtypes[column]
            if not isinstance(dtype, pd.CategoricalDtype):
                raise ValueError(
                    f"CosMx points element {element_name!r} column {column!r} must be categorical, found {dtype}."
                )
        if check_point_contents:
            _validate_points_panel_contents(points, element_name=element_name, panel=panel)
    return sample_ids


def _validate_feature_panels(registry: Mapping[str, object]) -> dict[str, _FeaturePanelContract]:
    panels: dict[str, _FeaturePanelContract] = {}
    for panel_name, value in registry.items():
        path = _element_path(_FEATURE_PANELS_METADATA_KEY, panel_name)
        panel_name = _require_nonempty_string(panel_name, path=f"{path} key")
        record = _require_mapping(value, path=path)
        feature_column = _require_nonempty_string(record.get("feature_column"), path=f"{path}.feature_column")
        class_column = _require_nonempty_string(record.get("class_column"), path=f"{path}.class_column")
        if feature_column == class_column:
            raise ValueError(f"CosMx metadata {path} feature and class columns must be different.")
        categories = _require_sorted_string_list(record.get("categories"), path=f"{path}.categories")
        targets_by_class = _require_mapping(record.get("targets_by_class"), path=f"{path}.targets_by_class")
        if set(targets_by_class) != set(categories):
            raise ValueError(
                f"CosMx metadata {path}.targets_by_class keys must equal categories {list(categories)}, "
                f"found {list(targets_by_class)}."
            )

        target_classes: dict[str, str] = {}
        canonical_targets: dict[str, list[str]] = {}
        for category in categories:
            targets = _require_sorted_string_list(
                targets_by_class[category],
                path=f"{path}.targets_by_class[{category!r}]",
            )
            canonical_targets[category] = list(targets)
            for target in targets:
                previous = target_classes.setdefault(target, category)
                if previous != category:
                    raise ValueError(
                        f"CosMx metadata {path} target {target!r} belongs to both {previous!r} and {category!r}."
                    )

        canonical = {
            "feature_column": feature_column,
            "class_column": class_column,
            "categories": list(categories),
            "targets_by_class": canonical_targets,
        }
        expected_name = _feature_panel_name(canonical)
        if panel_name != expected_name:
            raise ValueError(
                f"CosMx feature-panel key {panel_name!r} does not match canonical contents; expected {expected_name!r}."
            )
        panels[panel_name] = _FeaturePanelContract(
            feature_column=feature_column,
            class_column=class_column,
            categories=categories,
            target_classes=target_classes,
        )
    return panels


def _validate_points_panel_contents(
    points: DaskDataFrame,
    *,
    element_name: str,
    panel: _FeaturePanelContract,
) -> None:
    columns = [panel.feature_column, panel.class_column]
    try:
        observed = points[columns].drop_duplicates().compute()
    except Exception as error:
        raise ValueError(
            f"Could not validate feature-panel contents for CosMx points element {element_name!r}."
        ) from error

    observed_classes: dict[str, str] = {}
    for feature, feature_class in observed.itertuples(index=False, name=None):
        if pd.isna(feature) or pd.isna(feature_class):
            raise ValueError(f"CosMx points element {element_name!r} contains a null panel target or feature class.")
        if not isinstance(feature, str) or not isinstance(feature_class, str):
            raise ValueError(
                f"CosMx points element {element_name!r} panel targets and feature classes must be strings, "
                f"found {(feature, feature_class)!r}."
            )
        previous = observed_classes.setdefault(feature, feature_class)
        if previous != feature_class:
            raise ValueError(
                f"CosMx points element {element_name!r} target {feature!r} has multiple observed "
                f"feature classes: {previous!r} and {feature_class!r}."
            )
        expected_class = panel.target_classes.get(feature)
        if expected_class is None:
            raise ValueError(
                f"CosMx points element {element_name!r} contains target {feature!r} absent from its feature panel."
            )
        if feature_class != expected_class:
            raise ValueError(
                f"CosMx points element {element_name!r} target {feature!r} has feature class "
                f"{feature_class!r}; expected {expected_class!r}."
            )


def _validate_common_element_record(record: Mapping[str, object], *, path: str) -> str:
    sample_id = record.get("sample_id")
    try:
        _validate_identifier(sample_id, name="sample identifier")
    except ValueError as error:
        raise ValueError(f"Invalid CosMx metadata {path}.sample_id: {sample_id!r}.") from error
    assert isinstance(sample_id, str)

    fovs = record.get("fovs")
    if (
        not isinstance(fovs, list)
        or not fovs
        or any(isinstance(fov, bool) or not isinstance(fov, int) or fov < 1 for fov in fovs)
        or fovs != sorted(set(fovs))
    ):
        raise ValueError(f"CosMx metadata {path}.fovs must be a non-empty sorted list of unique positive integers.")

    mosaic = _require_mapping(record.get("mosaic"), path=f"{path}.mosaic")
    mode = mosaic.get("mode")
    if mode not in _MOSAIC_MODES:
        raise ValueError(f"CosMx metadata {path}.mosaic.mode must be one of {_MOSAIC_MODES}, found {mode!r}.")
    if "adjacency_tolerance_px" not in mosaic:
        raise ValueError(f"CosMx metadata {path}.mosaic.adjacency_tolerance_px is required.")
    tolerance = mosaic.get("adjacency_tolerance_px")
    if mode == "single":
        if tolerance is not None:
            raise ValueError(f"CosMx metadata {path}.mosaic.adjacency_tolerance_px must be None in single mode.")
    else:
        _require_nonnegative_integer(tolerance, path=f"{path}.mosaic.adjacency_tolerance_px")

    origin = _require_mapping(record.get("source_origin_px"), path=f"{path}.source_origin_px")
    for axis in ("x", "y"):
        value = origin.get(axis)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"CosMx metadata {path}.source_origin_px.{axis} must be an integer, found {value!r}.")

    orientation = _require_mapping(record.get("orientation"), path=f"{path}.orientation")
    for axis in ("flip_x", "flip_y"):
        if not isinstance(orientation.get(axis), bool):
            raise ValueError(f"CosMx metadata {path}.orientation.{axis} must be a bool.")

    pixel_size = record.get("pixel_size_um")
    if (
        isinstance(pixel_size, bool)
        or not isinstance(pixel_size, int | float)
        or not math.isfinite(pixel_size)
        or pixel_size <= 0
    ):
        raise ValueError(f"CosMx metadata {path}.pixel_size_um must be finite and positive, found {pixel_size!r}.")

    if "acquisition_timestamp" in record:
        _require_nonempty_string(record["acquisition_timestamp"], path=f"{path}.acquisition_timestamp")
    return sample_id


def _normalize_compartment_categories(value: object, *, path: str) -> dict[int, str]:
    categories = _require_mapping(value, path=path)
    result: dict[int, str] = {}
    for key, name in categories.items():
        if isinstance(key, bool):
            raise ValueError(f"CosMx metadata {path} has invalid category key {key!r}.")
        if isinstance(key, int):
            normalized = key
        elif isinstance(key, str) and key in {str(category) for category in _COMPARTMENT_CATEGORIES}:
            normalized = int(key)
        else:
            raise ValueError(f"CosMx metadata {path} has invalid category key {key!r}.")
        if normalized in result:
            raise ValueError(f"CosMx metadata {path} contains duplicate category {normalized}.")
        result[normalized] = _require_nonempty_string(name, path=f"{path}[{key!r}]")
    return result


def _require_registered_element(
    collection: Mapping[str, object],
    element_name: object,
    *,
    element_type: str,
    path: str,
) -> None:
    if not isinstance(element_name, str) or not element_name:
        raise ValueError(f"CosMx metadata {path} must use a non-empty string element name.")
    if element_name not in collection:
        raise ValueError(f"CosMx metadata {path} references missing {element_type} element {element_name!r}.")


def _element_path(registry: str, element_name: object) -> str:
    return f"{_HARPY_METADATA_KEY}.{registry}[{element_name!r}]"


def _require_mapping(value: object, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"CosMx metadata {path} must be a mapping, found {type(value).__name__}.")
    return value


def _optional_mapping(value: object, *, path: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    return _require_mapping(value, path=path)


def _require_nonempty_string(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"CosMx metadata {path} must be a non-empty trimmed string, found {value!r}.")
    return value


def _require_nonnegative_integer(value: object, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"CosMx metadata {path} must be a nonnegative integer, found {value!r}.")
    return value


def _require_sorted_string_list(value: object, *, path: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"CosMx metadata {path} must be a non-empty list of strings.")
    result = tuple(_require_nonempty_string(item, path=f"{path} item") for item in value)
    if result != tuple(sorted(result)) or len(set(result)) != len(result):
        raise ValueError(f"CosMx metadata {path} must be sorted and unique, found {value!r}.")
    return result


def _require_unique(values: list[object], *, path: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"CosMx metadata {path} values must be unique, found {values!r}.")
