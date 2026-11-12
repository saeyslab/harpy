from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData
from loguru import logger as log
from spatialdata import SpatialData
from spatialdata.models import TableModel

from harpy._storage._anndata import _matrix_format, _MissingAnnDataElement, _read_anndata_element
from harpy.image._image import _get_translation, _precondition, get_dataarray
from harpy.table._metadata import (
    _ADD_FEATURE_MATRIX_SOURCE_KIND,
    _COORDINATE_SYSTEM_FIELD,
    _FEATURE_COLUMNS_FIELD,
    _FEATURE_MATRICES_KEY,
    _FEATURE_MATRIX_SCHEMA_VERSION,
    _FEATURES_FIELD,
    _IMAGE_ELEMENT_FIELD,
    _REGIONS_FIELD,
    _SCHEMA_VERSION_FIELD,
    _SOURCE_CHANNELS_FIELD,
    _SOURCE_KIND_FIELD,
)
from harpy.table._regionprops import _calculate_regionprop_features
from harpy.table._validation import _RESERVED_OBSM_KEYS
from harpy.table.canonical_centers._models import _format_id_preview
from harpy.table.io._add_table import add_table
from harpy.table.io._components import _check_in_memory_versus_storage_axes, _component_update_destination
from harpy.table.io._components_by_region import add_table_components_by_region
from harpy.table.io._write_validation import _validated_observation_pairs
from harpy.utils._keys import _CELL_INDEX, _INSTANCE_KEY, _REGION_KEY
from harpy.utils._moments import MomentAggregator
from harpy.utils.utils import _make_list

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
# Features that one MomentAggregator pass over the labels (and the image) gives, with the
# statistic behind each: the intensity features per channel, and the area as the pixel count.
# The other morphology features come from regionprops, with the labels in memory.
_AGGREGATED_FEATURES = {**{feature: feature for feature in _INTENSITY_FEATURES}, "area": "count"}


@dataclass(frozen=True)
class _FeaturePair:
    labels_name: str
    image_name: str | None
    coordinate_system: str


def add_feature_matrix(
    sdata: SpatialData,
    labels_name: str | list[str],
    image_name: str | list[str] | None,
    *,
    table_name: str | None = None,
    output_table_name: str | None = None,
    feature_key: str,
    features: tuple[str, ...] | list[str],
    channels: int | str | list[int] | list[str] | None = None,
    overwrite_output_table: bool = False,
    overwrite_feature_key: bool = False,
    to_coordinate_system: str | list[str] = "global",
    region_key: str = _REGION_KEY,
    instance_key: str = _INSTANCE_KEY,
    feature_matrices_key: str = _FEATURE_MATRICES_KEY,
    chunks: str | int | tuple[int, ...] | None = None,
) -> SpatialData:
    """
    Compute per-instance feature matrices from labels and optional image data.

    This function computes requested object-level features from one or more
    labels elements and writes the resulting numeric matrix into
    `.obsm[feature_key]` of a target table. Companion metadata describing
    the matrix schema and inputs is stored in
    `.uns[feature_matrices_key][feature_key]`, including explicit
    `source_channels` for intensity-derived features.

    Features are aligned onto table rows by `(region_key, instance_key)`, not
    by row order. This makes the resulting matrix immediately reusable in
    downstream workflows that expect feature matrices in `.obsm`.

    The function supports two modes:

    - If `table_name is None`, a new annotated table is prepared and published
      after feature calculation succeeds. `output_table_name` is required.
    - If `table_name` is provided, the existing table is updated in place by
      writing or replacing `obsm[feature_key]` for the selected labels element or
      elements.

    Supported intensity features are `"sum"`, `"mean"`, `"var"`, `"min"`,
    `"max"`, `"kurtosis"`, and `"skew"`. Supported morphology features are
    `"area"`, `"eccentricity"`, `"major_axis_length"`,
    `"minor_axis_length"`, `"perimeter"`, `"convex_area"`,
    `"equivalent_diameter"`, `"major_minor_axis_ratio"`,
    `"perim_square_over_area"`, `"major_axis_equiv_diam_ratio"`,
    `"convex_hull_resid"`, and `"centroid_dif"`.

    Parameters
    ----------
    sdata
        The input SpatialData object.
    labels_name
        Labels element or elements from which object features are computed. When a
        list is provided, one feature matrix block is computed per labels element
        and aligned onto the target table using `region_key` and
        `instance_key`.
    image_name
        Image element or elements used for intensity-derived features. This is
        required if any requested feature is intensity-derived. If a list is
        provided, it must either have length 1 or match `labels_name`. If only
        morphology features are requested, a provided `image_name` is ignored.
    table_name
        Existing table element in `sdata.tables` to update. If `None`, a new
        annotated table is created and written to `output_table_name`.
    output_table_name
        Name of the output table element to create when `table_name is None`.
        This parameter is not allowed when updating an existing table.
    feature_key
        Key used to store the computed feature matrix in `adata.obsm`. Not
        `"spatial_canonical"` or `"spatial"`, which tools read as coordinates.
    features
        Requested feature names. Duplicate names are ignored while preserving
        order.
    channels
        Channel selection for intensity-derived features. Channels can be given
        by index, by name, or as a list of indices or names. If `None`, all
        channels of the image element are used.
    overwrite_output_table
        Allow replacing an output table that exists in the store of backed
        SpatialData, when creating a new table. It only concerns the store: a
        table attached but never saved is replaced without it. Ignored for
        unbacked SpatialData.
    overwrite_feature_key
        Allow updates to a feature matrix or its metadata that exist in the
        store of backed SpatialData. It only concerns the store: an entry
        present only in memory, never saved, is replaced without it. Ignored
        for unbacked SpatialData.
        In both modes, feature columns must retain their names and order;
        measurements and source descriptions for unselected regions are preserved.
    to_coordinate_system
        Coordinate system or systems used when pairing image and labels elements.
        If a list is provided, it must either have length 1 or match
        `labels_name`.
    region_key
        Column name in `adata.obs` identifying the source labels element. This is
        used when creating a new table and for aligning computed rows onto a
        target table.
    instance_key
        Column name in `adata.obs` identifying the instance id. This is used
        when creating a new table and for aligning computed rows onto a target
        table.
    feature_matrices_key
        Key in `adata.uns` under which metadata for computed feature matrices is
        stored.
    chunks
        Optional chunk specification used to rechunk image and labels arrays
        during feature extraction. Rechunking on disk ahead of time is often
        more efficient.

    Returns
    -------
    The updated SpatialData object.

    Notes
    -----
    New feature matrices use NaN for observations outside the selected regions.
    Existing matrices require a compatible feature schema and dense numeric
    storage; incompatible schemas are rejected rather than clearing other regions.

    In both storage modes, existing tables require valid SpatialData annotation
    with unique region/instance pairs. Selected labels elements must exist in
    ``sdata.labels`` and have observations in the target table. Every
    observation of a selected labels element must have an instance id that
    occurs in that element; otherwise a ``ValueError`` names the rows, rather
    than giving them NaN features. Instances of the labels without an
    observation in the table are left out. Features of instances that do occur
    are stored as calculated, NaN included.

    If ``sdata`` is backed by a Zarr store, existing-table updates publish only
    the feature matrix and its metadata, without reading or rewriting ``.X``.
    Unselected rows retain their stored values, not unsaved local replacements.
    The affected matrix is merged and rewritten in chunks, then attached lazily.
    Matrix and metadata publication, attachment and finalization share rollback;
    unrelated components and local annotations are preserved. The attached
    observation identities must match the stored table in value and order.

    Feature calculation and alignment complete before updating an existing table.
    When unbacked, the calculated features are merged with the retained
    measurements (or NaN fills for a new entry) as
    :func:`harpy.table.io.add_table_components_by_region` describes: the merge is
    computed when the attached feature matrix is in memory, and attached lazily
    when it is a Dask array or backed by Zarr. A later update would build on that
    lazy merge, so when updating repeatedly, compute the matrix between updates if
    it fits in memory, for example
    ``table.obsm[feature_key] = table.obsm[feature_key].compute()``: later updates
    then merge in memory. Nothing is written to disk.
    New-table creation still prepares a complete table before attachment.

    External references to replaced components are not refreshed. Use the entries
    in ``sdata.tables[table_name]`` after a successful update. The ``chunks``
    parameter controls raster feature extraction, not table-writing chunks.

    See Also
    --------
    harpy.tb.aggregate_image
        Aggregate intensity-derived features into a table.
    harpy.tb.add_regionprops
        Add morphology features to table observations.
    harpy.table.io.add_table_components_by_region
        Update regional measurements and their metadata in an attached table.

    Examples
    --------
    .. code-block:: python

        import harpy as hp

        sdata = hp.datasets.xenium_human_ovarian_cancer(
            subset=True,
            processed=False,
        )

        sdata = hp.tb.add_feature_matrix(
            sdata,
            labels_name="cell_labels_global",
            image_name="morphology_focus_global",
            table_name=None,
            output_table_name="table_cell_features",
            feature_key="cell_features",
            features=["mean", "area"],
            overwrite_output_table=True,
        )

        sdata["table_cell_features"].obsm["cell_features"].shape
    """
    if feature_key in _RESERVED_OBSM_KEYS:
        raise ValueError(
            f"Parameter 'feature_key' must not be a reserved .obsm key; reserved: {sorted(_RESERVED_OBSM_KEYS)}."
        )
    requested_features = _normalize_requested_features(features)
    intensity_features = [feature for feature in requested_features if feature in _INTENSITY_FEATURES]
    morphology_features = [feature for feature in requested_features if feature in _MORPHOLOGY_FEATURES]

    if chunks is not None:
        log.warning(
            "Parameter 'chunks' rechunks arrays during feature extraction. "
            "When possible, prefer rechunking on disk ahead of time for better performance."
        )

    pair_specs = _normalize_feature_pairs(
        labels_name=labels_name,
        image_name=image_name,
        to_coordinate_system=to_coordinate_system,
        needs_image=bool(intensity_features),
    )
    labels_names = [pair.labels_name for pair in pair_specs]

    if table_name is None:
        if output_table_name is None:
            raise ValueError("Parameter 'output_table_name' is required when 'table_name' is None.")
        if overwrite_feature_key:
            raise ValueError(
                "Parameter 'overwrite_feature_key' can only be used when updating an existing table, "
                "which requires setting 'table_name' to a table name."
            )
        # overwrite_output_table only concerns the store: a table attached but
        # never saved is replaced without it, as for unbacked SpatialData.
        if (
            sdata.path is not None
            and not overwrite_output_table
            and (Path(sdata.path) / "tables" / output_table_name).exists()
        ):
            raise ValueError(
                f"Table element '{output_table_name}' already exists in 'sdata.tables'. "
                "Set 'overwrite_output_table=True' to replace it."
            )
        # The rows of the new table are the instances found while calculating the features, so
        # the table is built after the calculation and the labels are not read for it separately.
        adata = None
        target_table_name = output_table_name
        existing_matrix = None
        existing_metadata = None
    else:
        if output_table_name is not None:
            raise ValueError(
                "Parameter 'output_table_name' can only be used when 'table_name' is None, "
                "because that is the mode where 'add_feature_matrix' creates a new table."
            )
        if overwrite_output_table:
            raise ValueError(
                "Parameter 'overwrite_output_table' can only be used when creating a new table, "
                "which requires setting 'table_name=None'."
            )
        target_table_name = table_name
        if target_table_name not in sdata.tables:
            raise ValueError(f"Table element {target_table_name!r} does not exist in sdata.tables.")
        adata = sdata.tables[target_table_name]
        existing_matrix, existing_metadata = _existing_feature_matrix(
            sdata,
            table_name=table_name,
            feature_key=feature_key,
            feature_matrices_key=feature_matrices_key,
        )
        region_key = adata.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY]
        instance_key = adata.uns[TableModel.ATTRS_KEY][TableModel.INSTANCE_KEY]
        # Identity validation above checks the table, not the requested sources.
        # Unused region categories do not make a labels element selectable.
        observed_regions = set(adata.obs[region_key].unique())
        for labels_element in labels_names:
            if labels_element not in sdata.labels:
                raise ValueError(f"Labels element {labels_element!r} does not exist in sdata.labels.")
            if labels_element not in observed_regions:
                raise ValueError(
                    f"Labels element {labels_element!r} has no observations in table {target_table_name!r}."
                )

    # overwrite_feature_key only concerns the store. When backed, existing_matrix
    # and existing_metadata come from the store; an entry present only in memory
    # is replaced without permission, as for unbacked SpatialData.
    if (
        sdata.path is not None
        and not overwrite_feature_key
        and (existing_matrix is not None or existing_metadata is not None)
    ):
        raise ValueError(
            f"Feature matrix '{feature_key}' already exists in 'sdata.tables[{target_table_name!r}].obsm'. "
            "Set 'overwrite_feature_key=True' to replace it."
        )

    pair_frames: list[pd.DataFrame] = []
    columns: list[str] = []
    source_channels: list[str] | None = None
    seen_columns: set[str] = set()
    for pair in pair_specs:
        pair_frame, pair_columns, pair_channels = _compute_pair_feature_frame(
            sdata,
            pair=pair,
            intensity_features=intensity_features,
            morphology_features=morphology_features,
            channels=channels,
            instance_key=instance_key,
            region_key=region_key,
            chunks=chunks,
        )
        if table_name is None and pair_frame.empty:
            # A new table declares every labels element as a region, and a region needs rows.
            raise ValueError(
                f"Labels element {pair.labels_name!r} contains no instances, only background (label 0), so the new "
                f"table {output_table_name!r} would have no rows for it. Remove it from 'labels_name'."
            )
        pair_frames.append(pair_frame)
        if pair_channels is not None:
            # Harpy does not support heterogeneous channels across samples, so every
            # intensity pair must resolve to the same channels. Enforcing this keeps
            # the feature matrix well-defined and the metadata a single channel list.
            if source_channels is None:
                source_channels = pair_channels
            elif source_channels != pair_channels:
                raise ValueError(
                    "All labels/image pairs must resolve to the same channels for intensity-derived features, "
                    f"but received {source_channels} and {pair_channels}. "
                    "Harpy does not support different channels for different samples."
                )
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

    if adata is None:
        adata = _new_feature_table(
            computed_features.loc[:, [region_key, instance_key]],
            labels_names=labels_names,
            region_key=region_key,
            instance_key=instance_key,
        )

    selected_mask = adata.obs[region_key].isin(labels_names).to_numpy()
    selected_keys = adata.obs.loc[selected_mask, [region_key, instance_key]]
    # Values go to table rows by (region, instance) key, not by row order.
    computed_by_key = computed_features.set_index([region_key, instance_key])
    table_keys = pd.MultiIndex.from_frame(selected_keys)
    _check_table_rows_in_labels(
        selected_keys,
        found=computed_by_key.index.get_indexer(table_keys) != -1,
        table_name=target_table_name,
        labels_names=labels_names,
        region_key=region_key,
        instance_key=instance_key,
    )
    aligned = computed_by_key.reindex(table_keys)
    aligned_values = aligned.loc[:, columns].to_numpy(dtype=np.float64)

    if existing_matrix is not None:
        # Check feature-matrix compatibility in both storage modes.
        # The regional adapter independently validates updates before installation.
        if _matrix_format(existing_matrix, label=f"Feature matrix {feature_key!r}") != "dense":
            raise ValueError(
                "Calculated features require an existing dense feature matrix; no format conversion occurs."
            )
        if existing_matrix.shape != (adata.n_obs, len(columns)):
            raise ValueError(f"Feature matrix {feature_key!r} must preserve its shape and feature columns.")
        if not np.can_cast(aligned_values.dtype, existing_matrix.dtype, casting="safe"):
            raise ValueError(f"Calculated features cannot be safely cast to stored dtype {existing_matrix.dtype}.")

    # The metadata of each labels region goes under "regions", keyed by the labels element,
    # so that regional updates and hp.tb.concat merge it by region.
    metadata = {
        _FEATURE_COLUMNS_FIELD: list(columns),
        _SCHEMA_VERSION_FIELD: _FEATURE_MATRIX_SCHEMA_VERSION,
        _SOURCE_KIND_FIELD: _ADD_FEATURE_MATRIX_SOURCE_KIND,
        _SOURCE_CHANNELS_FIELD: source_channels,
        _FEATURES_FIELD: list(requested_features),
        _REGIONS_FIELD: {
            pair.labels_name: {_IMAGE_ELEMENT_FIELD: pair.image_name, _COORDINATE_SYSTEM_FIELD: pair.coordinate_system}
            for pair in pair_specs
        },
    }

    if existing_matrix is not None:
        metadata = _merge_feature_matrix_metadata(existing_metadata, metadata)

    if table_name is None:
        # Build the complete result before publishing anything: calculation or
        # validation failures must not create an empty table or replace an old one.
        adata.obsm[feature_key] = aligned_values
        adata.uns[feature_matrices_key] = {feature_key: metadata}
        return add_table(
            sdata,
            adata=adata,
            output_table_name=target_table_name,
            region=labels_names,
            region_key=region_key,
            instance_key=instance_key,
            overwrite=overwrite_output_table,
        )

    matrix_path = ("obsm", feature_key)
    return add_table_components_by_region(
        sdata,
        table_name=target_table_name,
        components={matrix_path: aligned_values, ("uns", feature_matrices_key, feature_key): metadata},
        obs_identity=selected_keys,
        fill_values={matrix_path: np.nan},
        overwrite=overwrite_feature_key,
    )


def _existing_feature_matrix(
    sdata: SpatialData,
    *,
    table_name: str,
    feature_key: str,
    feature_matrices_key: str,
):
    """Read existing feature-matrix information after validating table identities.

    Checks in both storage modes
    ----------------------------
    Validate the in-memory table's SpatialData annotation against ``adata.obs``:
    identity columns must be valid, region/instance pairs non-null and unique,
    and declared regions must match observed regions. Also require
    ``adata.uns[feature_matrices_key]`` to be a mapping when present.

    Unbacked SpatialData
    -------------------
    No storage comparison is needed. Return the existing in-memory feature
    matrix and its metadata.

    Backed SpatialData
    ------------------
    Perform these additional checks:

    - Validate stored SpatialData annotation against stored observation identities.
    - Compare in-memory and stored region/instance column names: they must match.
    - Compare all in-memory and stored region/instance pairs: values and row order
      must match, including observations outside the selected regions.
    - Reject stored DataFrame-valued feature matrices before decoding their values.

    Return the stored matrix as a read-only handle and read its feature metadata
    eagerly. Stored entries, rather than unsaved local replacements, are used
    to prepare the update. Matrix values are not loaded and no Dask graph is built.

    Notes
    -----
    Missing matrix or metadata entries are returned as ``None`` independently.
    The caller checks matrix compatibility and feature-metadata compatibility.

    Run shared destination and identity checks here before expensive feature
    calculation. ``add_table_components_by_region()`` checks them again when
    applying the prepared update. For backed updates, both require the full
    in-memory observation identities and order to match storage before attaching
    the resulting matrix by position, including rows outside the updated regions.
    """
    adata, group = _component_update_destination(sdata, table_name=table_name)
    in_memory_attrs = adata.uns.get(TableModel.ATTRS_KEY)
    if not isinstance(in_memory_attrs, Mapping):
        raise ValueError(f"Table {table_name!r} must have SpatialData annotation.")
    # Reject invalid identities before feature calculation, using the same
    # validators as the regional adapter rather than a separate comparison.
    if group is None:
        _validated_observation_pairs(adata.obs, in_memory_attrs, label="In-memory observation")
    else:
        _check_in_memory_versus_storage_axes(
            table=adata, group=group, indices={"obs": adata.obs.index}, stored_new_raw_var=None
        )
    local_metadata = adata.uns.get(feature_matrices_key, {})
    if not isinstance(local_metadata, Mapping):
        raise ValueError(f"adata.uns[{feature_matrices_key!r}] must be a metadata mapping.")
    if group is None:
        return adata.obsm.get(feature_key), local_metadata.get(feature_key)

    matrix_path = ("obsm", feature_key)
    metadata_path = ("uns", feature_matrices_key, feature_key)
    # DataFrame decoding is eager even in backed mode. Keep this guard to avoid
    # loading its payload merely to prepare metadata; the writer also rejects it.
    if "/".join(matrix_path) in group and group["/".join(matrix_path)].attrs.get("encoding-type") == "dataframe":
        raise TypeError("Feature updates do not support DataFrame-valued obsm entries.")
    try:
        matrix = _read_anndata_element(group, matrix_path, mode="backed")
    except _MissingAnnDataElement:
        matrix = None  # No existing feature matrix to merge with.
    try:
        metadata = _read_anndata_element(group, metadata_path, mode="eager")
    except _MissingAnnDataElement:
        metadata = None
    return matrix, metadata


def _merge_feature_matrix_metadata(previous: object, current: dict) -> dict:
    """Merge the metadata of an update into the existing one, keeping the regions the update does not compute.

    The feature columns are not relabeled.
    """
    if (
        not isinstance(previous, Mapping)
        or previous.get(_SCHEMA_VERSION_FIELD) != _FEATURE_MATRIX_SCHEMA_VERSION
        or previous.get(_SOURCE_KIND_FIELD) != _ADD_FEATURE_MATRIX_SOURCE_KIND
        or not isinstance(previous.get(_REGIONS_FIELD), Mapping)
        or not np.array_equal(previous.get(_FEATURE_COLUMNS_FIELD), current[_FEATURE_COLUMNS_FIELD])
        or not np.array_equal(previous.get(_SOURCE_CHANNELS_FIELD), current[_SOURCE_CHANNELS_FIELD])
    ):
        raise ValueError(
            "Existing feature metadata must have a compatible schema, as add_feature_matrix writes it with schema "
            f"version {_FEATURE_MATRIX_SCHEMA_VERSION}, and identical feature columns and source channels in order. "
            "To replace the matrix, remove it and its metadata with hp.tb.io.remove_table_components, then compute "
            "it again for every region."
        )
    return {
        **deepcopy(dict(previous)),
        **current,
        _REGIONS_FIELD: {**deepcopy(dict(previous[_REGIONS_FIELD])), **current[_REGIONS_FIELD]},
    }


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
    labels_name: str | list[str],
    image_name: str | list[str] | None,
    to_coordinate_system: str | list[str],
    needs_image: bool,
) -> list[_FeaturePair]:
    labels_names = _make_list(labels_name)
    if not labels_names:
        raise ValueError("Parameter 'labels_name' must contain at least one labels element.")
    if len(set(labels_names)) != len(labels_names):
        raise ValueError("Duplicate labels elements are not supported in a single 'add_feature_matrix' call.")

    coordinate_systems = _broadcast_parameter(
        to_coordinate_system,
        target_length=len(labels_names),
        parameter_name="to_coordinate_system",
    )

    if needs_image:
        if image_name is None:
            raise ValueError("An 'image_name' is required when requesting intensity-derived features.")
        image_names = _broadcast_parameter(
            image_name,
            target_length=len(labels_names),
            parameter_name="image_name",
        )
    else:
        if image_name is not None:
            log.warning("Only morphology features were requested, so the provided 'image_name' input will be ignored.")
        image_names = [None] * len(labels_names)

    return [
        _FeaturePair(labels_name=labels, image_name=image, coordinate_system=coordinate_system)
        for labels, image, coordinate_system in zip(labels_names, image_names, coordinate_systems, strict=True)
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
        f"Parameter '{parameter_name}' must either have length 1 or match the number of requested labels elements "
        f"({target_length}), but received length {len(values)}."
    )


def _check_table_rows_in_labels(
    table_keys: pd.DataFrame,
    *,
    found: np.ndarray,
    table_name: str,
    labels_names: Sequence[str],
    region_key: str,
    instance_key: str,
) -> None:
    """Raise if a table row of a selected labels element has an instance id that does not occur in that element.

    ``found`` tells, per row of ``table_keys``, whether features were calculated for its
    ``(region, instance)`` pair. Without this check such a row would silently get NaN features.

    It checks only that every row gets calculated features, not their values: a feature of an
    instance that does occur can still be NaN, for example ``major_minor_axis_ratio`` of an
    instance one pixel wide, or an intensity feature of an image with NaN pixels. Those values
    are stored as calculated.
    """
    if found.all():
        return
    missing = table_keys.loc[~found]
    problems = []
    for labels_name in labels_names:
        instance_ids = missing.loc[missing[region_key] == labels_name, instance_key].to_numpy()
        if instance_ids.size:
            problems.append(
                f"{instance_ids.size} rows of table {table_name!r} have instance IDs that do not occur in labels "
                f"element {labels_name!r} (e.g. {_format_id_preview(instance_ids)})"
            )
    raise ValueError(
        "; ".join(problems) + ". The labels and the table are out of sync: remove these rows from the table, or "
        "calculate the features from the labels element the table was made from."
    )


def _new_feature_table(
    identities: pd.DataFrame,
    labels_names: Sequence[str],
    region_key: str,
    instance_key: str,
) -> AnnData:
    """Build the observations of a new table from the calculated identities, without attaching or publishing it.

    ``identities`` holds the ``(region_key, instance_key)`` pair of every calculated row: the
    instances of each labels element, background excluded, in the order of ``labels_names``
    and with ascending instance ids within an element.
    """
    uuid_value = str(uuid.uuid4())[:8]
    instance_ids = identities[instance_key].to_numpy(dtype=int)
    regions = identities[region_key].to_numpy()
    table_obs = pd.DataFrame(
        {
            instance_key: instance_ids,
            region_key: pd.Categorical(regions, categories=list(labels_names)),
        }
    )
    table_obs.index = pd.Index(
        [f"{instance_id}_{region}_{uuid_value}" for instance_id, region in zip(instance_ids, regions, strict=True)],
        name=_CELL_INDEX,
    )

    return TableModel.parse(
        AnnData(obs=table_obs),
        region=list(labels_names),
        instance_key=instance_key,
        region_key=region_key,
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
) -> tuple[pd.DataFrame, list[str], list[str] | None]:
    labels = get_dataarray(sdata, element_name=pair.labels_name)
    _ = _get_translation(labels, to_coordinate_system=pair.coordinate_system)
    source_labels_ndim = labels.data.ndim
    if source_labels_ndim == 3:
        unsupported = [feature for feature in morphology_features if feature in _UNSUPPORTED_3D_MORPHOLOGY_FEATURES]
        if unsupported:
            raise ValueError(f"Morphology feature(s) {unsupported} are not supported for 3D labels data.")

    feature_frames: list[pd.DataFrame] = []
    ordered_columns: list[str] = []
    source_channels: list[str] | None = None

    if intensity_features:
        assert pair.image_name is not None, "Intensity feature computation requires an image element."
        image, labels = _precondition(
            sdata,
            image_name=pair.image_name,
            labels_name=pair.labels_name,
            to_coordinate_system=pair.coordinate_system,
        )
        channel_names, channel_indices = _resolve_channels(image, channels)
        source_channels = list(channel_names)
        image_array, labels_array = _prepare_raster_arrays(image.data, labels.data, chunks=chunks)
        ordered_columns.extend(_ordered_intensity_columns(intensity_features, channel_names))
    else:
        labels_array = labels.data
    ordered_columns.extend(morphology_features)

    # The features split by how they are computed: one MomentAggregator pass gives the intensity
    # features and the area, regionprops the other morphology features. Every frame holds every
    # instance of the labels, background excluded: the instance ids come from the calculation
    # itself, not from a separate pass over the labels.
    aggregated_features = [
        feature for feature in (*intensity_features, *morphology_features) if feature in _AGGREGATED_FEATURES
    ]
    regionprops_features = [feature for feature in morphology_features if feature not in _AGGREGATED_FEATURES]

    if aggregated_features:
        feature_frames.append(
            _compute_aggregated_feature_frame(
                labels_array=labels_array if labels_array.ndim == 3 else labels_array[None, ...],
                image_array=image_array[channel_indices] if intensity_features else None,
                features=aggregated_features,
                channel_names=channel_names if intensity_features else [],
                instance_key=instance_key,
            )
        )

    if regionprops_features:
        # 2D intensity extraction adds a singleton z-axis for the aggregator.
        # Remove it again before calling skimage regionprops so 2D-only features
        # such as eccentricity still work.
        regionprops_labels_array = (
            labels_array[0] if source_labels_ndim == 2 and labels_array.ndim == 3 else labels_array
        )
        feature_frames.append(
            _compute_regionprops_feature_frame(
                labels_array=regionprops_labels_array,
                features=regionprops_features,
                instance_key=instance_key,
            )
        )

    pair_frame = feature_frames[0]
    for frame in feature_frames[1:]:
        pair_frame = pair_frame.merge(frame, how="outer", on=instance_key)
    # a new table takes its rows in this order: ascending instance ids within a labels element
    pair_frame = pair_frame.sort_values(instance_key, ignore_index=True)

    pair_frame[region_key] = pair.labels_name
    pair_frame = pair_frame.reindex(columns=[region_key, instance_key, *ordered_columns])

    return pair_frame, ordered_columns, source_channels


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
                        f"Channel index '{channel}' is out of range for image element '{image.name}'. "
                        f"Available indices are 0 through {len(available_channels) - 1}."
                    )
                index = int(channel)
            else:
                channel_key = str(channel)
                if channel_key not in string_to_index:
                    raise ValueError(
                        f"Channel '{channel}' was not found in image element '{image.name}'. "
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

    # MomentAggregator requires the labels and the image to share spatial (z, y, x) chunks.
    # The labels and image already share spatial shape (validated by _precondition), so align
    # the (cheaper) single-band labels onto the image's spatial chunks when they differ.
    # Compare the chunks, not the chunk sizes: a crop off the chunk grid has the same chunk
    # size with other chunk boundaries.
    if prepared_labels.chunks != prepared_image.chunks[1:]:
        log.info(
            "Image and labels have different chunks; rechunking labels to match "
            f"the image's spatial chunks {prepared_image.chunksize[1:]} for aggregation."
        )
        prepared_labels = prepared_labels.rechunk(prepared_image.chunks[1:])

    return prepared_image, prepared_labels


def _compute_aggregated_feature_frame(
    labels_array,
    image_array,
    features: Sequence[str],
    channel_names: Sequence[str],
    instance_key: str,
) -> pd.DataFrame:
    """Features of every instance that one pass of :class:`MomentAggregator` gives, see ``_AGGREGATED_FEATURES``.

    Intensity features get a column per channel, ``area`` a single column. ``image_array`` is
    `None` when only the area is requested: the pass then reads only the labels. Returns a row
    per instance, background excluded, in ascending order.
    """
    aggregator = MomentAggregator(
        mask_dask_array=labels_array,
        image_dask_array=image_array,
        instance_key=instance_key,
    )
    stats_funcs = tuple(_AGGREGATED_FEATURES[feature] for feature in features)
    # without an index: a row for every label of the mask, in ascending order
    stats_frames = [
        _without_background(frame, instance_key) for frame in aggregator.aggregate_stats(stats_funcs=stats_funcs)
    ]

    result = stats_frames[0].loc[:, [instance_key]]
    for feature, frame in zip(features, stats_frames, strict=True):
        if feature == "area":
            renamed = frame.rename(columns={0: "area"})
        else:
            renamed = _rename_intensity_columns(frame, feature, channel_names, instance_key)
        result = result.merge(renamed, how="outer", on=instance_key)

    # sanity check
    assert result[instance_key].is_unique, (
        f"Expected '{instance_key}' to remain unique after merging aggregated features."
    )

    return result


def _without_background(frame: pd.DataFrame, instance_key: str) -> pd.DataFrame:
    """Drop the background row (label 0) and store the instance ids as ``int``."""
    frame = frame.loc[frame[instance_key] != 0].reset_index(drop=True)
    frame[instance_key] = frame[instance_key].astype(int, copy=False)
    return frame


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


def _compute_regionprops_feature_frame(
    labels_array,
    features: Sequence[str],
    instance_key: str,
) -> pd.DataFrame:
    """Morphology features of every instance from :func:`skimage.measure.regionprops`, with the labels in memory.

    Returns a row per instance, background excluded, in ascending order.
    """
    frame = _calculate_regionprop_features(
        masks=labels_array.compute(),
        properties=tuple(features),
        instance_key=instance_key,
    )
    result = frame.loc[:, [instance_key, *features]].copy()
    result[instance_key] = result[instance_key].astype(int, copy=False)
    return result


def _format_channel_name(channel_name: object, index: int) -> str:
    if channel_name is None:
        return f"channel_{index}"
    return str(channel_name)
