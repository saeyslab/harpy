from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import zarr
from anndata import AnnData
from anndata.io import write_elem
from loguru import logger as log
from spatialdata import SpatialData
from spatialdata.models import TableModel

from harpy.image._image import _get_translation, _precondition, get_dataarray
from harpy.table._regionprops import _calculate_regionprop_features
from harpy.table._table import ProcessTable, add_table_layer
from harpy.utils._aggregate import RasterAggregator, _get_mask_area
from harpy.utils._keys import _CELL_INDEX, _INSTANCE_KEY, _REGION_KEY
from harpy.utils.utils import _da_unique, _make_list

# clean up code

_HARPY_FEATURE_MATRICES_KEY = "harpy_feature_matrices"

_INTENSITY_FEATURES = ("sum", "mean", "var", "min", "max", "kurtosis", "skew")
_MORPHOLOGY_FEATURES = (
    "area",
    "eccentricity",
    "major_axis_length",
    "minor_axis_length",
    "perimeter",
    "convex_area",
    "equivalent_diameter",
    "major_minor_axis_ratio",
    "perim_square_over_area",
    "major_axis_equiv_diam_ratio",
    "convex_hull_resid",
    "centroid_dif",
)
_UNSUPPORTED_3D_MORPHOLOGY_FEATURES = ("eccentricity", "perimeter", "perim_square_over_area")


@dataclass(frozen=True)
class _FeaturePair:
    labels_layer: str
    img_layer: str | None
    coordinate_system: str


def feature_matrix(
    sdata: SpatialData,
    labels_layer: str | list[str],
    img_layer: str | list[str] | None,
    *,
    table_layer: str | None = None,
    output_layer: str | None = None,
    feature_key: str,
    features: tuple[str, ...] | list[str],
    channels: int | str | list[int] | list[str] | None = None,
    overwrite_output_layer: bool = False,
    overwrite_feature_key: bool = False,
    to_coordinate_system: str | list[str] = "global",
    region_key: str = _REGION_KEY,
    instance_key: str = _INSTANCE_KEY,
    chunks: str | int | tuple[int, ...] | None = None,
    run_on_gpu: bool = False,
) -> SpatialData:
    """
    Compute object features from labels and optional image data, and write them into ``adata.obsm``.

    The result is stored in ``adata.obsm[feature_key]`` and accompanied by metadata in
    ``adata.uns["harpy_feature_matrices"][feature_key]``.
    """
    requested_features = _normalize_requested_features(features)
    intensity_features = [feature for feature in requested_features if feature in _INTENSITY_FEATURES]
    morphology_features = [feature for feature in requested_features if feature in _MORPHOLOGY_FEATURES]

    if chunks is not None:
        log.warning(
            "Parameter 'chunks' rechunks arrays during feature extraction. "
            "When possible, prefer rechunking on disk ahead of time for better performance."
        )

    pair_specs = _normalize_feature_pairs(
        labels_layer=labels_layer,
        img_layer=img_layer,
        to_coordinate_system=to_coordinate_system,
        needs_image=bool(intensity_features),
    )
    selected_labels = [pair.labels_layer for pair in pair_specs]

    if table_layer is None:
        if output_layer is None:
            raise ValueError("Parameter 'output_layer' is required when 'table_layer' is None.")
        if overwrite_feature_key:
            raise ValueError(
                "Parameter 'overwrite_feature_key' can only be used when updating an existing table, "
                "which requires setting 'table_layer' to a table name."
            )
        if output_layer in sdata.tables and not overwrite_output_layer:
            raise ValueError(
                f"Table layer '{output_layer}' already exists in 'sdata.tables'. "
                "Set 'overwrite_output_layer=True' to replace it."
            )
        sdata = _create_empty_feature_table(
            sdata,
            labels_layers=selected_labels,
            output_layer=output_layer,
            region_key=region_key,
            instance_key=instance_key,
            overwrite=overwrite_output_layer,
        )
        target_table_layer = output_layer
    else:
        if output_layer is not None:
            raise ValueError(
                "Parameter 'output_layer' can only be used when 'table_layer' is None, "
                "because that is the mode where 'feature_matrix' creates a new table."
            )
        if overwrite_output_layer:
            raise ValueError(
                "Parameter 'overwrite_output_layer' can only be used when creating a new table, "
                "which requires setting 'table_layer=None'."
            )
        target_table_layer = table_layer
        # Validate that the target table exists, annotates the requested labels layers,
        # and uses unique instance ids within each selected region.
        ProcessTable(sdata, table_layer=target_table_layer, labels_layer=selected_labels)
        adata = sdata.tables[target_table_layer]
        region_key = adata.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY]
        instance_key = adata.uns[TableModel.ATTRS_KEY][TableModel.INSTANCE_KEY]

    adata = sdata.tables[target_table_layer]

    if feature_key in adata.obsm and not overwrite_feature_key:
        raise ValueError(
            f"Feature matrix '{feature_key}' already exists in 'sdata.tables[{target_table_layer!r}].obsm'. "
            "Set 'overwrite_feature_key=True' to replace it."
        )

    pair_frames: list[pd.DataFrame] = []
    columns: list[str] = []
    seen_columns: set[str] = set()
    for pair in pair_specs:
        pair_frame, pair_columns = _compute_pair_feature_frame(
            sdata,
            pair=pair,
            intensity_features=intensity_features,
            morphology_features=morphology_features,
            channels=channels,
            instance_key=instance_key,
            region_key=region_key,
            chunks=chunks,
            run_on_gpu=run_on_gpu,
        )
        pair_frames.append(pair_frame)
        for column in pair_columns:
            if column not in seen_columns:
                seen_columns.add(column)
                columns.append(column)

    computed_features = pd.concat(pair_frames, ignore_index=True, sort=False)
    computed_features = computed_features.reindex(columns=[region_key, instance_key, *columns])

    if computed_features.duplicated(subset=[region_key, instance_key]).any():
        duplicates = computed_features.loc[
            computed_features.duplicated(subset=[region_key, instance_key], keep=False),
            [region_key, instance_key],
        ].head()
        raise ValueError(
            "Calculated feature rows contain duplicate '(region_key, instance_key)' pairs, which would make the "
            f"alignment ambiguous. Examples: {duplicates.to_dict(orient='records')}"
        )

    selected_mask = adata.obs[region_key].isin(selected_labels).to_numpy()
    selected_keys = adata.obs.loc[selected_mask, [region_key, instance_key]]
    aligned = computed_features.set_index([region_key, instance_key]).reindex(pd.MultiIndex.from_frame(selected_keys))
    aligned_values = aligned.loc[:, columns].to_numpy(dtype=np.float64)

    shape = (adata.n_obs, len(columns))
    matrix = np.full(shape, np.nan, dtype=np.float64)
    if feature_key in adata.obsm and overwrite_feature_key:
        schema_matches = False
        existing = np.asarray(adata.obsm[feature_key])
        if existing.ndim == 2 and existing.shape[0] == adata.n_obs and existing.shape[1] == len(columns):
            feature_matrices = adata.uns.get(_HARPY_FEATURE_MATRICES_KEY, {})
            if isinstance(feature_matrices, dict) and feature_key in feature_matrices:
                existing_metadata = feature_matrices[feature_key]
                existing_columns = existing_metadata.get("columns") if isinstance(existing_metadata, dict) else None
                if existing_columns is not None:
                    schema_matches = [str(column) for column in list(existing_columns)] == [
                        str(column) for column in columns
                    ]

        if schema_matches:
            existing = np.asarray(adata.obsm[feature_key], dtype=np.float64)
            if existing.shape == shape:
                matrix = existing.copy()
    matrix[selected_mask] = aligned_values

    source_labels = [pair.labels_layer for pair in pair_specs]
    source_images = [pair.img_layer for pair in pair_specs]
    coordinate_systems = [pair.coordinate_system for pair in pair_specs]
    metadata = {
        "columns": list(columns),
        "schema_version": 1,
        "backend": "numpy",
        "dtype": str(matrix.dtype),
        "source_label": source_labels[0] if len(source_labels) == 1 else source_labels,
        "source_image": source_images[0] if len(source_images) == 1 else source_images,
        "coordinate_system": coordinate_systems[0] if len(coordinate_systems) == 1 else coordinate_systems,
        "features": list(requested_features),
    }

    adata.obsm[feature_key] = matrix
    existing_metadata = adata.uns.get(_HARPY_FEATURE_MATRICES_KEY, {})
    if not isinstance(existing_metadata, dict):
        existing_metadata = dict(existing_metadata)
    existing_metadata[feature_key] = metadata
    adata.uns[_HARPY_FEATURE_MATRICES_KEY] = existing_metadata

    if sdata.is_backed() and sdata.path is not None:
        root = zarr.open_group(sdata.path, mode="r+", use_consolidated=False)
        table_group = root["tables"][target_table_layer]
        write_elem(table_group["obsm"], feature_key, matrix)

        uns_group = table_group["uns"]
        if _HARPY_FEATURE_MATRICES_KEY not in uns_group:
            write_elem(uns_group, _HARPY_FEATURE_MATRICES_KEY, {})
        write_elem(uns_group[_HARPY_FEATURE_MATRICES_KEY], feature_key, metadata)
        zarr.consolidate_metadata(sdata.path)

    return sdata


def _normalize_requested_features(features: tuple[str, ...] | list[str]) -> list[str]:
    requested = _make_list(features)
    if not requested:
        raise ValueError("Parameter 'features' must contain at least one feature name.")

    supported = [*_INTENSITY_FEATURES, *_MORPHOLOGY_FEATURES]
    unsupported = [feature for feature in requested if feature not in supported]
    if unsupported:
        raise ValueError(f"Unsupported feature(s): {unsupported}. Please choose features from {supported}.")

    normalized: list[str] = []
    seen: set[str] = set()
    for feature in requested:
        if feature in seen:
            continue
        seen.add(feature)
        normalized.append(feature)

    return normalized


def _normalize_feature_pairs(
    labels_layer: str | list[str],
    img_layer: str | list[str] | None,
    to_coordinate_system: str | list[str],
    needs_image: bool,
) -> list[_FeaturePair]:
    labels_layers = _make_list(labels_layer)
    if not labels_layers:
        raise ValueError("Parameter 'labels_layer' must contain at least one labels layer.")
    if len(set(labels_layers)) != len(labels_layers):
        raise ValueError("Duplicate labels layers are not supported in a single 'feature_matrix' call.")

    coordinate_systems = _broadcast_parameter(
        to_coordinate_system,
        target_length=len(labels_layers),
        parameter_name="to_coordinate_system",
    )

    if needs_image:
        if img_layer is None:
            raise ValueError("An 'img_layer' is required when requesting intensity-derived features.")
        img_layers = _broadcast_parameter(
            img_layer,
            target_length=len(labels_layers),
            parameter_name="img_layer",
        )
    else:
        if img_layer is not None:
            log.warning("Only morphology features were requested, so the provided 'img_layer' input will be ignored.")
        img_layers = [None] * len(labels_layers)

    return [
        _FeaturePair(labels_layer=labels, img_layer=image, coordinate_system=coordinate_system)
        for labels, image, coordinate_system in zip(labels_layers, img_layers, coordinate_systems, strict=True)
    ]


def _broadcast_parameter(
    value: str | list[str] | None,
    target_length: int,
    parameter_name: str,
) -> list[str | None]:
    values = _make_list(value)
    if len(values) == target_length:
        return values
    if len(values) == 1:
        return values * target_length
    raise ValueError(
        f"Parameter '{parameter_name}' must either have length 1 or match the number of requested labels layers "
        f"({target_length}), but received length {len(values)}."
    )


def _create_empty_feature_table(
    sdata: SpatialData,
    labels_layers: Sequence[str],
    output_layer: str,
    region_key: str,
    instance_key: str,
    overwrite: bool,
) -> SpatialData:
    obs_frames: list[pd.DataFrame] = []
    uuid_value = str(uuid.uuid4())[:8]

    for labels in labels_layers:
        data = get_dataarray(sdata, layer=labels).data
        instance_ids = np.asarray(_da_unique(data, run_on_gpu=False))
        instance_ids = instance_ids[instance_ids != 0].astype(int, copy=False)

        obs = pd.DataFrame(
            {
                instance_key: instance_ids,
                region_key: labels,
            }
        )
        obs.index = pd.Index(
            [f"{instance_id}_{labels}_{uuid_value}" for instance_id in instance_ids],
            name=_CELL_INDEX,
        )
        obs_frames.append(obs)

    if obs_frames:
        table_obs = pd.concat(obs_frames, axis=0)
    else:
        # Defensive fallback: the public path should always provide at least one labels layer,
        # but keep this helper able to construct an empty .obs table if it is called with none.
        table_obs = pd.DataFrame(columns=[instance_key, region_key])
        table_obs.index = pd.Index([], name=_CELL_INDEX)

    table_obs[region_key] = pd.Categorical(table_obs[region_key], categories=list(labels_layers))
    adata = AnnData(obs=table_obs)

    return add_table_layer(
        sdata,
        adata=adata,
        output_layer=output_layer,
        region=list(labels_layers),
        instance_key=instance_key,
        region_key=region_key,
        overwrite=overwrite,
    )


def _compute_pair_feature_frame(
    sdata: SpatialData,
    pair: _FeaturePair,
    intensity_features: Sequence[str],
    morphology_features: Sequence[str],
    channels: int | str | list[int] | list[str] | None,
    instance_key: str,
    region_key: str,
    chunks: str | int | tuple[int, ...] | None,
    run_on_gpu: bool,
) -> tuple[pd.DataFrame, list[str]]:
    labels = get_dataarray(sdata, layer=pair.labels_layer)
    _ = _get_translation(labels, to_coordinate_system=pair.coordinate_system)

    feature_frames: list[pd.DataFrame] = []
    ordered_columns: list[str] = []

    if intensity_features:
        assert pair.img_layer is not None, "Intensity feature computation requires an image layer."
        image, labels = _precondition(
            sdata,
            img_layer=pair.img_layer,
            labels_layer=pair.labels_layer,
            to_coordinate_system=pair.coordinate_system,
        )
        channel_names, channel_indices = _resolve_channels(image, channels)
        image_array, labels_array = _prepare_raster_arrays(image.data, labels.data, chunks=chunks)
        ordered_columns.extend(_ordered_intensity_columns(intensity_features, channel_names))
    else:
        labels_array = labels.data

    mask_for_instances = labels_array if labels_array.ndim == 3 else labels_array[None, ...]
    # Keep shared instance ids on CPU; downstream intensity/area helpers move them
    # to the appropriate backend internally when needed.
    instance_ids = np.asarray(_da_unique(mask_for_instances, run_on_gpu=False))
    instance_ids = instance_ids[instance_ids != 0].astype(int, copy=False)

    if intensity_features:
        intensity_frame = _compute_intensity_feature_frame(
            image_array=image_array[channel_indices],
            labels_array=labels_array,
            intensity_features=intensity_features,
            channel_names=channel_names,
            instance_key=instance_key,
            instance_ids=instance_ids,
            run_on_gpu=run_on_gpu,
        )
        feature_frames.append(intensity_frame)

    if morphology_features:
        morphology_frame = _compute_morphology_feature_frame(
            labels_array=labels_array,
            morphology_features=morphology_features,
            instance_key=instance_key,
            instance_ids=instance_ids,
            run_on_gpu=run_on_gpu,
        )
        feature_frames.append(morphology_frame)
        ordered_columns.extend(morphology_features)

    pair_frame = feature_frames[0]
    for frame in feature_frames[1:]:
        pair_frame = pair_frame.merge(frame, how="outer", on=instance_key)

    pair_frame[region_key] = pair.labels_layer
    pair_frame = pair_frame.reindex(columns=[region_key, instance_key, *ordered_columns])

    return pair_frame, ordered_columns


def _resolve_channels(
    image,
    channels: int | str | list[int] | list[str] | None,
) -> tuple[list[str], list[int]]:
    available_channels = list(image.c.data)
    if channels is None:
        indices = list(range(len(available_channels)))
    else:
        requested = _make_list(channels)
        indices = []
        seen: set[int] = set()
        string_to_index = {str(name): index for index, name in enumerate(available_channels)}
        for channel in requested:
            if isinstance(channel, (int, np.integer)) and not isinstance(channel, bool):
                if channel < 0 or channel >= len(available_channels):
                    raise ValueError(
                        f"Channel index '{channel}' is out of range for image layer '{image.name}'. "
                        f"Available indices are 0 through {len(available_channels) - 1}."
                    )
                index = int(channel)
            else:
                channel_key = str(channel)
                if channel_key not in string_to_index:
                    raise ValueError(
                        f"Channel '{channel}' was not found in image layer '{image.name}'. "
                        f"Available channels are {[str(name) for name in available_channels]}."
                    )
                index = string_to_index[channel_key]

            if index in seen:
                continue
            seen.add(index)
            indices.append(index)

    if not indices:
        raise ValueError("At least one channel must be selected when requesting intensity-derived features.")

    channel_names = [_format_channel_name(available_channels[index], index=index) for index in indices]
    return channel_names, indices


def _prepare_raster_arrays(
    image_array,
    labels_array,
    chunks: str | int | tuple[int, ...] | None,
):
    is_2d = image_array.ndim == 3
    if is_2d:
        prepared_image = image_array[:, None, ...]
        prepared_labels = labels_array[None, ...]
    else:
        prepared_image = image_array
        prepared_labels = labels_array

    if prepared_image.ndim != 4 or prepared_labels.ndim != 3:
        raise ValueError(
            "Only 2D and 3D raster data are supported. "
            f"Received image dimensions {image_array.ndim} and label dimensions {labels_array.ndim}."
        )

    image_chunks = None
    labels_chunks = None
    if chunks is not None:
        if isinstance(chunks, tuple):
            expected_length = 2 if is_2d else 3
            if len(chunks) != expected_length:
                raise ValueError(
                    f"Parameter 'chunks' should have length {expected_length} for the provided data, "
                    f"but received {len(chunks)}."
                )
            if is_2d:
                image_chunks = (prepared_image.chunksize[0], 1, chunks[0], chunks[1])
                labels_chunks = (1, chunks[0], chunks[1])
            else:
                image_chunks = (prepared_image.chunksize[0], chunks[0], chunks[1], chunks[2])
                labels_chunks = tuple(chunks)
        else:
            image_chunks = chunks
            labels_chunks = chunks

    if image_chunks is not None:
        prepared_image = prepared_image.rechunk(image_chunks)
    if labels_chunks is not None:
        prepared_labels = prepared_labels.rechunk(labels_chunks)

    return prepared_image, prepared_labels


def _compute_intensity_feature_frame(
    image_array,
    labels_array,
    intensity_features: Sequence[str],
    channel_names: Sequence[str],
    instance_key: str,
    instance_ids: np.ndarray,
    run_on_gpu: bool,
) -> pd.DataFrame:
    result = pd.DataFrame({instance_key: instance_ids})

    aggregator = RasterAggregator(
        mask_dask_array=labels_array,
        image_dask_array=image_array,
        instance_key=instance_key,
        run_on_gpu=run_on_gpu,
    )

    aggregated_features = [feature for feature in intensity_features if feature not in {"max", "min"}]
    renamed_frames: dict[str, pd.DataFrame] = {}
    if aggregated_features:
        stats_funcs = tuple(aggregated_features)
        stats_frames = aggregator.aggregate_stats(stats_funcs=stats_funcs, index=instance_ids)
        for feature, frame in zip(stats_funcs, stats_frames, strict=True):
            renamed_frames[feature] = _rename_intensity_columns(frame, feature, channel_names, instance_key)

    if "max" in intensity_features:
        frame = aggregator.aggregate_max(index=instance_ids)
        renamed_frames["max"] = _rename_intensity_columns(
            frame,
            "max",
            channel_names,
            instance_key,
        )
    if "min" in intensity_features:
        frame = aggregator.aggregate_min(index=instance_ids)
        renamed_frames["min"] = _rename_intensity_columns(
            frame,
            "min",
            channel_names,
            instance_key,
        )

    for feature in intensity_features:
        result = result.merge(renamed_frames[feature], how="outer", on=instance_key)

    # sanity checks
    assert result[instance_key].is_unique, (
        f"Expected '{instance_key}' to remain unique after merging intensity features."
    )
    assert set(result[instance_key].to_numpy()) == set(instance_ids.tolist()), (
        f"Expected merged intensity result to contain exactly the provided '{instance_key}' values."
    )

    return result


def _rename_intensity_columns(
    frame: pd.DataFrame,
    prefix: str,
    channel_names: Sequence[str],
    instance_key: str,
) -> pd.DataFrame:
    rename_map = {index: f"{prefix}__{channel_name}" for index, channel_name in enumerate(channel_names)}
    renamed = frame.rename(columns=rename_map)
    assert instance_key in renamed.columns, f"Expected aggregated intensity frame to contain '{instance_key}'."
    renamed[instance_key] = renamed[instance_key].astype(int, copy=False)
    return renamed


def _ordered_intensity_columns(intensity_features: Sequence[str], channel_names: Sequence[str]) -> list[str]:
    return [f"{feature}__{channel_name}" for feature in intensity_features for channel_name in channel_names]


def _compute_morphology_feature_frame(
    labels_array,
    morphology_features: Sequence[str],
    instance_key: str,
    instance_ids: np.ndarray,
    run_on_gpu: bool,
) -> pd.DataFrame:
    if labels_array.ndim == 3:
        unsupported = [feature for feature in morphology_features if feature in _UNSUPPORTED_3D_MORPHOLOGY_FEATURES]
        if unsupported:
            raise ValueError(f"Morphology feature(s) {unsupported} are not supported for 3D labels data.")

    result_frames: list[pd.DataFrame] = []

    if "area" in morphology_features:
        mask_for_area = labels_array if labels_array.ndim == 3 else labels_array[None, ...]
        # Only the area fast path honors run_on_gpu; skimage regionprops remains CPU-only.
        area_frame = _get_mask_area(
            mask_for_area,
            index=instance_ids,
            instance_key=instance_key,
            instance_size_key="area",
            run_on_gpu=run_on_gpu,
        )
        area_frame[instance_key] = area_frame[instance_key].astype(int, copy=False)
        result_frames.append(area_frame.loc[:, [instance_key, "area"]])

    other_morphology_features = [feature for feature in morphology_features if feature != "area"]
    if other_morphology_features:
        masks = labels_array.compute()
        frame = _calculate_regionprop_features(
            masks=masks,
            properties=tuple(other_morphology_features),
            instance_key=instance_key,
        )
        other_frame = frame.loc[:, [instance_key, *other_morphology_features]].copy()
        other_frame[instance_key] = other_frame[instance_key].astype(int, copy=False)
        result_frames.append(other_frame)

    result = result_frames[0]
    for frame in result_frames[1:]:
        result = result.merge(frame, how="outer", on=instance_key)

    return result.loc[:, [instance_key, *morphology_features]]


def _format_channel_name(channel_name: object, index: int) -> str:
    if channel_name is None:
        return f"channel_{index}"
    return str(channel_name)
