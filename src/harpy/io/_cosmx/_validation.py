from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dask.dataframe import DataFrame as DaskDataFrame
from spatialdata import SpatialData, read_zarr
from spatialdata.transformations import Identity, get_transformation
from xarray import DataArray

from harpy._feature_panels import _FeaturePanelContract
from harpy.image._image import get_dataarray
from harpy.io._cosmx._constants import (
    _COMPARTMENT_CATEGORIES,
    _FOV_BOUNDS_FIELDS,
    _FOV_BOUNDS_KEY,
    _INSTANCE_ID_FORMULA,
)
from harpy.io._cosmx._models import _INSTANCE_ID_DTYPE, _MOSAIC_MODES, _validate_identifier
from harpy.metadata._registry import (
    _COORDINATE_SYSTEMS_METADATA_KEY,
    _FEATURE_PANELS_METADATA_KEY,
    _HARPY_METADATA_KEY,
    _IMAGES_METADATA_KEY,
    _LABELS_METADATA_KEY,
    _METADATA_VERSION,
    _METADATA_VERSION_KEY,
    _MICRONS_PER_UNIT_KEY,
    _MOSAIC_KEY,
    _POINTS_METADATA_KEY,
    _PROVENANCE_METADATA_KEY,
    _SAMPLE_ID_KEY,
)

_ELEMENT_REGISTRIES = (_IMAGES_METADATA_KEY, _LABELS_METADATA_KEY, _POINTS_METADATA_KEY)
# Fields of a coordinate-system record that describe the run of its sample, the same in each coordinate
# system of that sample.
_RUN_FIELDS = (_MICRONS_PER_UNIT_KEY, "orientation", "mosaic_grouping", "acquisition_timestamp")


def validate_cosmx_store(
    output: str | Path,
    *,
    check_point_contents: bool = True,
) -> None:
    """Validate a sample-aware CosMx SpatialData Zarr store without modifying it.

    Structural validation checks the Harpy metadata version and provenance;
    the record of each mosaic's coordinate system, and the agreement of the
    records of one sample; that each recorded image and labels element lies
    unscaled in exactly one recorded coordinate system, within whose FOV
    bounds it extends; label-family metadata; feature-panel records; and
    referenced points schemas. It reads array and dataframe metadata but does
    not compute raster or transcript partitions.

    By default, the content check projects only the feature and feature-class
    columns from each panel-associated points element and validates each
    partition independently against the referenced authoritative panel. It
    computes at most one small diagnostic per partition and does not require a
    global shuffle. Panel targets with zero detections remain valid. Set
    ``check_point_contents=False`` for a metadata- and schema-only check.

    Parameters
    ----------
    output
        Existing SpatialData Zarr store created by the sample-aware CosMx
        reader.
    check_point_contents
        Whether to scan the two panel-declared categorical columns in referenced
        points elements. Defaults to ``True``; set to ``False`` to check only
        their lazy schemas.

    Raises
    ------
    ValueError
        If the output cannot be opened as a backed SpatialData store or any
        reader-owned metadata, registered element, points schema, or requested
        feature-panel content check violates the CosMx store contract.

    Notes
    -----
    This function never writes attributes or elements and does not repair or
    migrate invalid stores. Elements without a record, such as downstream
    segmentations, are permitted and not checked. A points record needs no
    coordinate system, so the record that :func:`harpy.pt.add_feature_panel`
    writes for another points element is valid.
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
    check_point_contents: bool = True,
) -> frozenset[str]:
    """Validate an opened store and return the sample IDs of its coordinate-system records."""
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
        for registry in (_COORDINATE_SYSTEMS_METADATA_KEY, *_ELEMENT_REGISTRIES, _FEATURE_PANELS_METADATA_KEY)
    }
    coordinate_systems = registries[_COORDINATE_SYSTEMS_METADATA_KEY]
    if not coordinate_systems:
        raise ValueError(
            f"CosMx metadata {_HARPY_METADATA_KEY}.{_COORDINATE_SYSTEMS_METADATA_KEY} must record at least one "
            "coordinate system. Stores written by older Harpy versions lack it; run the CosMx reader again to "
            "rewrite the store."
        )

    sample_ids = _validate_coordinate_system_records(sdata, coordinate_systems)
    panels = _validate_feature_panels(registries[_FEATURE_PANELS_METADATA_KEY])
    _validate_image_records(sdata, registries[_IMAGES_METADATA_KEY], coordinate_systems=coordinate_systems)
    _validate_labels_records(sdata, registries[_LABELS_METADATA_KEY], coordinate_systems=coordinate_systems)
    _validate_points_records(
        sdata,
        registries[_POINTS_METADATA_KEY],
        panels=panels,
        check_point_contents=check_point_contents,
    )
    return frozenset(sample_ids)


def _validate_coordinate_system_records(sdata: SpatialData, records: Mapping[str, object]) -> set[str]:
    """Validate the record of each mosaic's coordinate system and their agreement; return the sample IDs.

    Each record must name a coordinate system of ``sdata``. A mosaic has one
    coordinate system, so ``(sample_id, mosaic)`` is unique across the records.
    The facts of a run, ``_RUN_FIELDS``, are the same in each coordinate system
    of its sample; the first record of a sample is the reference for the others.

    Examples
    --------
    Two records of sample ``s1`` with different pixel sizes are each valid on
    their own, but disagree on a fact of the run::

        coordinate_systems["s1_global_1"]  {"sample_id": "s1", "mosaic": 1, "microns_per_unit": 0.12, ...}
        coordinate_systems["s1_global_2"]  {"sample_id": "s1", "mosaic": 2, "microns_per_unit": 0.18, ...}
            -> ValueError: ...coordinate_systems['s1_global_2'].microns_per_unit differs from
               ...coordinate_systems['s1_global_1'].microns_per_unit ...
    """
    existing_coordinate_systems = set(sdata.coordinate_systems)
    coordinate_system_of_mosaic: dict[tuple[str, int], str] = {}
    first_of_sample: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for name, value in records.items():
        path = _coordinate_system_path(name)
        record = _require_mapping(value, path=path)
        if name not in existing_coordinate_systems:
            raise ValueError(
                f"CosMx metadata {path} records coordinate system {name!r}, to which no element of sdata has a "
                "transformation. Restore the elements of that mosaic, or run the CosMx reader again to rewrite the "
                "store."
            )
        sample_id, mosaic = _validate_coordinate_system_record(record, path=path)
        other = coordinate_system_of_mosaic.setdefault((sample_id, mosaic), name)
        if other != name:
            raise ValueError(
                f"CosMx metadata {path} and {_coordinate_system_path(other)} both record mosaic {mosaic} of sample "
                f"{sample_id!r}, which has one coordinate system. Run the CosMx reader again to rewrite the store."
            )
        first_name, first_record = first_of_sample.setdefault(sample_id, (name, record))
        for field in _RUN_FIELDS:
            if record.get(field) != first_record.get(field):
                raise ValueError(
                    f"CosMx metadata {path}.{field} differs from {_coordinate_system_path(first_name)}.{field}, "
                    f"although both are coordinate systems of sample {sample_id!r}. Run the CosMx reader again to "
                    "rewrite the store."
                )
    return set(first_of_sample)


def _validate_image_records(
    sdata: SpatialData,
    registry: Mapping[str, object],
    *,
    coordinate_systems: Mapping[str, Mapping[str, Any]],
) -> None:
    for element_name, value in registry.items():
        path = _element_path(_IMAGES_METADATA_KEY, element_name)
        _require_registered_element(sdata.images, element_name, element_type="image", path=path)
        record = _require_mapping(value, path=path)
        array = get_dataarray(sdata, element_name)
        _validate_raster_placement(
            sdata, element_name, raster_shape=_raster_shape(array), coordinate_systems=coordinate_systems
        )

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

        if "c" not in array.dims or "c" not in array.coords:
            raise ValueError(f"CosMx image element {element_name!r} must have a channel coordinate.")
        actual_coordinates = array.coords["c"].values.tolist()
        if actual_coordinates != output_coordinates:
            raise ValueError(
                f"CosMx metadata {path}.channels output coordinates {output_coordinates} "
                f"do not match image coordinates {actual_coordinates}."
            )


def _validate_labels_records(
    sdata: SpatialData,
    registry: Mapping[str, object],
    *,
    coordinate_systems: Mapping[str, Mapping[str, Any]],
) -> None:
    for element_name, value in registry.items():
        path = _element_path(_LABELS_METADATA_KEY, element_name)
        _require_registered_element(sdata.labels, element_name, element_type="labels", path=path)
        record = _require_mapping(value, path=path)
        array = get_dataarray(sdata, element_name)
        _validate_raster_placement(
            sdata, element_name, raster_shape=_raster_shape(array), coordinate_systems=coordinate_systems
        )

        has_instance_encoding = "instance_id_encoding" in record
        has_categories = "categories" in record
        if has_instance_encoding == has_categories:
            raise ValueError(
                f"CosMx metadata {path} must contain exactly one of 'instance_id_encoding' or 'categories'."
            )

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


def _validate_raster_placement(
    sdata: SpatialData,
    element_name: str,
    *,
    raster_shape: tuple[int, int],
    coordinate_systems: Mapping[str, Mapping[str, Any]],
) -> None:
    """Require a recorded image or labels element to lie unscaled in one recorded coordinate system.

    :func:`harpy.io.cosmx` writes each raster with an ``Identity`` to its
    mosaic's coordinate system, and no transformation to another recorded one,
    so the ``fov_bounds`` of that coordinate system are in the raster's
    intrinsic frame and must lie within its scale-0 ``(y, x)`` shape. Call only after
    the coordinate-system records have been validated.
    """
    transformations = get_transformation(sdata[element_name], get_all=True)
    recorded = sorted(set(transformations) & set(coordinate_systems))
    if len(recorded) != 1:
        raise ValueError(
            f"CosMx element {element_name!r} must have a transformation to exactly one coordinate system in "
            f"{_HARPY_METADATA_KEY}.{_COORDINATE_SYSTEMS_METADATA_KEY}, found {recorded or 'none'}. Run the "
            "CosMx reader again to rewrite the store."
        )
    coordinate_system = recorded[0]
    transformation = transformations[coordinate_system]
    if not isinstance(transformation, Identity):
        raise ValueError(
            f"CosMx element {element_name!r} must lie unscaled in coordinate system {coordinate_system!r}, with an "
            f"Identity transformation, found {transformation}. Run the CosMx reader again to rewrite the store."
        )
    bounds_path = f"{_coordinate_system_path(coordinate_system)}.{_FOV_BOUNDS_KEY}"
    for fov, rectangle in coordinate_systems[coordinate_system][_FOV_BOUNDS_KEY].items():
        if rectangle["y1"] > raster_shape[0] or rectangle["x1"] > raster_shape[1]:
            raise ValueError(
                f"CosMx metadata {bounds_path}[{fov!r}] must lie within the (y, x) shape {raster_shape} of element "
                f"{element_name!r}, found {dict(rectangle)}."
            )


def _validate_points_records(
    sdata: SpatialData,
    registry: Mapping[str, object],
    *,
    panels: Mapping[str, _FeaturePanelContract],
    check_point_contents: bool,
) -> None:
    """Validate points records: the feature-panel reference, if any, and the panel-declared columns.

    A points record needs no coordinate system: it may belong to any points
    element of the store, such as one registered with
    :func:`harpy.pt.add_feature_panel`. The columns are therefore checked
    against Harpy's general panel contract, as :func:`harpy.pt.validate_points`
    does: both exist, and the feature-class column is categorical. The feature
    column is not checked: :func:`harpy.io.cosmx` writes it as a categorical,
    but other writers need not. Their placement is not checked either, unlike
    that of images and labels: points need not be written by
    :func:`harpy.io.cosmx`, and have no shape to hold the FOV bounds against.
    """
    for element_name, value in registry.items():
        path = _element_path(_POINTS_METADATA_KEY, element_name)
        _require_registered_element(sdata.points, element_name, element_type="points", path=path)
        record = _require_mapping(value, path=path)

        panel_name = record.get("feature_panel")
        if panel_name is None:
            continue
        panel_name = _require_nonempty_string(panel_name, path=f"{path}.feature_panel")
        if panel_name not in panels:
            raise ValueError(f"CosMx metadata {path}.feature_panel references missing panel {panel_name!r}.")
        panel = panels[panel_name]
        points = sdata.points[element_name]
        for column in (panel.feature_key, panel.feature_class_key):
            if column not in points.columns:
                raise ValueError(f"CosMx points element {element_name!r} is missing panel-declared column {column!r}.")
        dtype = points.dtypes[panel.feature_class_key]
        if not isinstance(dtype, pd.CategoricalDtype):
            raise ValueError(
                f"CosMx points element {element_name!r} column {panel.feature_class_key!r} must be categorical, "
                f"found {dtype}."
            )
        if check_point_contents:
            _validate_points_panel_contents(points, element_name=element_name, panel=panel)


def _validate_feature_panels(registry: Mapping[str, object]) -> dict[str, _FeaturePanelContract]:
    panels: dict[str, _FeaturePanelContract] = {}
    for panel_name, value in registry.items():
        path = _element_path(_FEATURE_PANELS_METADATA_KEY, panel_name)
        panel_name = _require_nonempty_string(panel_name, path=f"{path} key")
        panel_record = _require_mapping(value, path=path)
        feature_key = _require_nonempty_string(panel_record.get("feature_key"), path=f"{path}.feature_key")
        feature_class_key = _require_nonempty_string(
            panel_record.get("feature_class_key"), path=f"{path}.feature_class_key"
        )
        classes = _require_sorted_string_list(panel_record.get("classes"), path=f"{path}.classes")
        features_by_class = _require_mapping(panel_record.get("features_by_class"), path=f"{path}.features_by_class")
        if set(features_by_class) != set(classes):
            raise ValueError(
                f"CosMx metadata {path}.features_by_class keys must equal classes {list(classes)}, "
                f"found {list(features_by_class)}."
            )

        features_by_class_items: list[tuple[str, tuple[str, ...]]] = []
        for feature_class in classes:
            features = _require_sorted_string_list(
                features_by_class[feature_class],
                path=f"{path}.features_by_class[{feature_class!r}]",
            )
            features_by_class_items.append((feature_class, features))
        try:
            panel = _FeaturePanelContract(
                feature_key=feature_key,
                feature_class_key=feature_class_key,
                classes=classes,
                features_by_class_items=tuple(features_by_class_items),
            )
        except ValueError as error:
            raise ValueError(f"CosMx metadata {path}: {error}") from error
        expected_name = panel.storage_key
        if panel_name != expected_name:
            raise ValueError(
                f"CosMx feature-panel key {panel_name!r} does not match canonical contents; expected {expected_name!r}."
            )
        panels[panel_name] = panel
    return panels


def _validate_points_panel_contents(
    points: DaskDataFrame,
    *,
    element_name: str,
    panel: _FeaturePanelContract,
) -> None:
    columns = [panel.feature_key, panel.feature_class_key]
    try:
        errors = (
            points[columns]
            .map_partitions(
                _points_panel_partition_error,
                feature_key=panel.feature_key,
                feature_class_key=panel.feature_class_key,
                class_by_feature=panel.class_by_feature,
                meta=pd.Series(name="error", dtype="object"),
            )
            .dropna()
            .compute()
        )
    except Exception as error:
        raise ValueError(
            f"Could not validate feature-panel contents for CosMx points element {element_name!r}."
        ) from error

    if not errors.empty:
        raise ValueError(f"CosMx points element {element_name!r} {errors.iloc[0]}")


def _points_panel_partition_error(
    partition: pd.DataFrame,
    *,
    feature_key: str,
    feature_class_key: str,
    class_by_feature: Mapping[str, str],
) -> pd.Series:
    """Validate one points partition against its authoritative feature panel.

    The feature panel maps every declared feature to exactly one feature class.
    This check therefore requires every observed feature to occur in
    ``class_by_feature`` and its observed class to equal the corresponding
    mapped class. Null features or classes are invalid. Panel features without
    detections in this partition remain valid.

    Only the first violation in a partition is returned. Consequently, the Dask
    computation collects at most one small diagnostic per partition and does
    not need a global shuffle or materialize transcript rows in the client.

    Parameters
    ----------
    partition
        In-memory pandas partition containing the two panel-declared columns.
    feature_key
        Name of the column containing observed feature identifiers.
    feature_class_key
        Name of the column containing observed feature classes.
    class_by_feature
        Authoritative mapping from every panel feature to its feature class.

    Returns
    -------
    pandas.Series
        A one-element series containing the first validation error, or ``None``
        when the partition satisfies the feature-panel contract.

    Examples
    --------
    Given a panel that assigns ``GeneA`` to ``Endogenous``, the observed pairs
    have the following outcomes::

        GeneA, Endogenous        -> valid
        GeneA, Negative          -> rejected: class disagrees with the panel
        GeneA, UnknownClass      -> rejected: class disagrees with the panel
        UnknownGene, Endogenous  -> rejected: feature is absent from the panel
    """
    error: str | None = None
    if not partition.empty:
        features = partition[feature_key]
        feature_classes = partition[feature_class_key]

        null = features.isna() | feature_classes.isna()
        if null.any():
            error = "contains a null panel feature or feature class."
        else:
            known = features.isin(class_by_feature)
            if not known.all():
                position = int(np.flatnonzero(~known.to_numpy())[0])
                feature = features.iloc[position]
                feature_class = feature_classes.iloc[position]
                if not isinstance(feature, str) or not isinstance(feature_class, str):
                    error = f"panel features and feature classes must be strings, found {(feature, feature_class)!r}."
                else:
                    error = f"contains feature {feature!r} absent from its feature panel."
            else:
                expected_classes = features.astype(object).map(class_by_feature)
                mismatch = feature_classes.astype(object) != expected_classes
                if mismatch.any():
                    position = int(np.flatnonzero(mismatch.to_numpy())[0])
                    feature = features.iloc[position]
                    feature_class = feature_classes.iloc[position]
                    if not isinstance(feature, str) or not isinstance(feature_class, str):
                        error = (
                            f"panel features and feature classes must be strings, found {(feature, feature_class)!r}."
                        )
                    else:
                        error = (
                            f"feature {feature!r} has feature class {feature_class!r}; "
                            f"expected {class_by_feature[feature]!r}."
                        )

    return pd.Series([error], name="error", dtype="object")


def _validate_coordinate_system_record(record: Mapping[str, object], *, path: str) -> tuple[str, int]:
    """Validate the fields of one mosaic's coordinate-system record and return its ``(sample_id, mosaic)``."""
    sample_id = record.get(_SAMPLE_ID_KEY)
    try:
        _validate_identifier(sample_id, name="sample identifier")
    except ValueError as error:
        raise ValueError(f"Invalid CosMx metadata {path}.{_SAMPLE_ID_KEY}: {sample_id!r}.") from error
    assert isinstance(sample_id, str)

    mosaic = record.get(_MOSAIC_KEY)
    if isinstance(mosaic, bool) or not isinstance(mosaic, int) or mosaic < 1:
        raise ValueError(f"CosMx metadata {path}.{_MOSAIC_KEY} must be a positive integer, found {mosaic!r}.")

    microns_per_unit = record.get(_MICRONS_PER_UNIT_KEY)
    if (
        isinstance(microns_per_unit, bool)
        or not isinstance(microns_per_unit, int | float)
        or not math.isfinite(microns_per_unit)
        or microns_per_unit <= 0
    ):
        raise ValueError(
            f"CosMx metadata {path}.{_MICRONS_PER_UNIT_KEY} must be finite and positive, found {microns_per_unit!r}."
        )

    fovs = record.get("fovs")
    if (
        not isinstance(fovs, list)
        or not fovs
        or any(isinstance(fov, bool) or not isinstance(fov, int) or fov < 1 for fov in fovs)
        or fovs != sorted(set(fovs))
    ):
        raise ValueError(f"CosMx metadata {path}.fovs must be a non-empty sorted list of unique positive integers.")

    _validate_fov_bounds(record, fovs=fovs, path=path)

    origin = _require_mapping(record.get("source_origin_px"), path=f"{path}.source_origin_px")
    for axis in ("x", "y"):
        value = origin.get(axis)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"CosMx metadata {path}.source_origin_px.{axis} must be an integer, found {value!r}.")

    orientation = _require_mapping(record.get("orientation"), path=f"{path}.orientation")
    for axis in ("flip_x", "flip_y"):
        if not isinstance(orientation.get(axis), bool):
            raise ValueError(f"CosMx metadata {path}.orientation.{axis} must be a bool.")

    grouping = _require_mapping(record.get("mosaic_grouping"), path=f"{path}.mosaic_grouping")
    mode = grouping.get("mode")
    if mode not in _MOSAIC_MODES:
        raise ValueError(f"CosMx metadata {path}.mosaic_grouping.mode must be one of {_MOSAIC_MODES}, found {mode!r}.")
    if "adjacency_tolerance_px" not in grouping:
        raise ValueError(f"CosMx metadata {path}.mosaic_grouping.adjacency_tolerance_px is required.")
    tolerance = grouping.get("adjacency_tolerance_px")
    if mode == "single":
        if tolerance is not None:
            raise ValueError(
                f"CosMx metadata {path}.mosaic_grouping.adjacency_tolerance_px must be None in single mode."
            )
    else:
        _require_nonnegative_integer(tolerance, path=f"{path}.mosaic_grouping.adjacency_tolerance_px")

    if "acquisition_timestamp" in record:
        _require_nonempty_string(record["acquisition_timestamp"], path=f"{path}.acquisition_timestamp")
    return sample_id, mosaic


def _validate_fov_bounds(record: Mapping[str, object], *, fovs: list[int], path: str) -> None:
    """Validate the half-open, non-overlapping FOV rectangles of a record with valid ``fovs``."""
    path = f"{path}.{_FOV_BOUNDS_KEY}"
    if _FOV_BOUNDS_KEY not in record:
        raise ValueError(f"CosMx metadata {path} is required; run the CosMx reader again to rewrite the store.")
    bounds = _require_mapping(record[_FOV_BOUNDS_KEY], path=path)
    keys = [str(fov) for fov in fovs]
    if set(bounds) != set(keys):
        raise ValueError(f"CosMx metadata {path} keys must equal fovs as strings, {keys}, found {list(bounds)}.")

    rectangles: list[tuple[str, int, int, int, int]] = []
    for key in keys:
        rectangle_path = f"{path}[{key!r}]"
        rectangle = _require_mapping(bounds[key], path=rectangle_path)
        if set(rectangle) != set(_FOV_BOUNDS_FIELDS):
            raise ValueError(
                f"CosMx metadata {rectangle_path} must contain exactly the fields {list(_FOV_BOUNDS_FIELDS)}, "
                f"found {list(rectangle)}."
            )
        y0, x0, y1, x1 = (
            _require_nonnegative_integer(rectangle[field], path=f"{rectangle_path}.{field}")
            for field in _FOV_BOUNDS_FIELDS
        )
        if y0 >= y1 or x0 >= x1:
            raise ValueError(f"CosMx metadata {rectangle_path} must have y0 < y1 and x0 < x1, found {dict(rectangle)}.")
        rectangles.append((key, y0, x0, y1, x1))

    for index, (key, y0, x0, y1, x1) in enumerate(rectangles):
        for other_key, other_y0, other_x0, other_y1, other_x1 in rectangles[index + 1 :]:
            if max(y0, other_y0) < min(y1, other_y1) and max(x0, other_x0) < min(x1, other_x1):
                raise ValueError(f"CosMx metadata {path} rectangles of FOVs {key} and {other_key} overlap.")


def _raster_shape(array: DataArray) -> tuple[int, int]:
    return array.sizes["y"], array.sizes["x"]


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


def _coordinate_system_path(coordinate_system: object) -> str:
    return _element_path(_COORDINATE_SYSTEMS_METADATA_KEY, coordinate_system)


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
