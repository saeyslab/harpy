"""Register authoritative panels for existing points without reader-specific metadata."""

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import pandas as pd
from loguru import logger as log
from spatialdata import SpatialData

from harpy._feature_panels import (
    _feature_membership_partition_errors,
    _feature_panel_partition_errors,
    _FeaturePanelContract,
    _make_feature_panel,
    _parse_feature_panel_registry,
    _validate_feature_panel_collision,
)
from harpy._metadata import (
    _FEATURE_PANELS_METADATA_KEY,
    _POINTS_METADATA_KEY,
    _metadata_registry,
    _validate_metadata_destination,
)
from harpy._storage._spatialdata import _replace_element_on_disk


def add_feature_panel(
    sdata: SpatialData,
    points_name: str,
    *,
    feature_key: str,
    feature_class_key: str,
    features_by_class: Mapping[str, Sequence[str]],
) -> SpatialData:
    """Register a complete assay panel for one existing points element.

    Prepare points for ``hp.qc.summarize_points`` and feature-class-aware
    ``hp.tb.aggregate_points``. No images, labels, tables, sample identifiers,
    or reader-specific metadata are required. This function does not aggregate
    points or select an expression class.

    Parameters
    ----------
    sdata
        SpatialData containing the points. Update unbacked objects in memory;
        persist changes for local filesystem-backed objects. In a backed object,
        the selected points must already be saved to the store. Save pending
        point edits before using the metadata-only registration path.
    points_name
        Name of the existing points element to associate with the panel.
    feature_key
        Existing points column containing feature identifiers, such as genes.
    feature_class_key
        Name of the class column in ``sdata.points[points_name]`` to validate
        or create. Existing values must agree with the panel; conflicting
        assignments are never silently replaced. If absent, the column is added
        to that points element with values derived from the panel. Normalization
        uses a categorical dtype with all panel classes in sorted order,
        including undetected classes.
    features_by_class
        Complete assay panel mapping class names to sequences of feature names.
        Each feature belongs to exactly one class. Every observed feature must
        occur in the panel, but panel features need not have detected points:
        they remain available for zero-detection summaries. Harpy cannot infer
        missing assay features from the observations. For an unclassified assay,
        explicitly supply a single class, such as ``{"All": feature_names}``.
        Class and feature names are preserved exactly and sorted during panel
        construction, so their input order does not affect panel identity.
        Existing panel records must already be sorted; unsorted stored metadata
        is rejected, not silently normalized.

    Returns
    -------
    SpatialData
        The updated input object. Point rows, other columns, coordinates,
        transformations, unrelated metadata and other elements are preserved.

    Notes
    -----
    The versioned root metadata associates points with a shared panel::

        sdata.attrs["harpy"]["points"][points_name]["feature_panel"] = panel_name
        sdata.attrs["harpy"]["feature_panels"][panel_name] = {
            "feature_key": feature_key,
            "feature_class_key": feature_class_key,
            "classes": [...],
            "features_by_class": {...},
        }

    Identical panels share a content-derived key. Re-registering an identical
    association is safe; conflicting associations or panel contents raise an
    error. No reader provenance is added, and existing tables are not migrated.

    Validate source features and existing class values partition-wise before
    publishing changes. If the class column already has the required categorical
    dtype and categories, write metadata only: validation reads source columns,
    but no points files are rewritten. Otherwise, derive or normalize the class
    column lazily and rewrite only the selected points element, partition-wise.
    Parquet stores the complete dataframe, so this is not a single-column write.
    Dask may report unknown categories after reopening Parquet, and empty
    partitions may have no category values. The panel metadata retains the
    complete class list for downstream consumers in either case.

    During replacement, retain the original points until the metadata commit and
    context finalization succeed. On an exception, restore affected in-memory
    state and attempt to restore root metadata on disk; the shared replacement
    context handles points rollback. This does not provide crash recovery or
    concurrent-writer isolation.

    Examples
    --------
    >>> sdata = hp.pt.add_feature_panel(
    ...     sdata, "transcripts", feature_key="gene", feature_class_key="feature_class",
    ...     features_by_class={"Endogenous": ["EPCAM", "VIM"], "Negative": ["Negative1"]},
    ... )
    >>> summary = hp.qc.summarize_points(sdata, "transcripts")
    """
    if points_name not in sdata.points:
        raise ValueError(f"Points element {points_name!r} does not exist.")
    backed = sdata.is_backed()
    if backed:
        if sdata.path is None or "://" in str(sdata.path):
            raise ValueError(
                "Feature-panel registration requires a local filesystem-backed store or an unbacked object."
            )
        if not (Path(sdata.path) / "points" / points_name).is_dir():
            raise ValueError(
                f"Points element {points_name!r} is not saved in the backing store. "
                f"Call sdata.write_element({points_name!r}) before registering its feature panel."
            )
    panel = _make_feature_panel(
        feature_key=feature_key, feature_class_key=feature_class_key, features_by_class=features_by_class
    )
    panel_record = panel.to_dict()
    panel_name = panel.storage_key
    points = sdata.points[points_name]
    if feature_key not in points.columns:
        raise ValueError(f"Points element {points_name!r} has no feature column {feature_key!r}.")

    _validate_metadata_destination(sdata, _POINTS_METADATA_KEY, _FEATURE_PANELS_METADATA_KEY)
    previous_attrs = deepcopy(sdata.attrs)
    attrs = deepcopy(previous_attrs)
    panel_records = _metadata_registry(attrs, _FEATURE_PANELS_METADATA_KEY)
    existing_panels = _parse_feature_panel_registry(panel_records)
    _validate_feature_panel_collision(panel, existing_panels=existing_panels)
    records = _metadata_registry(attrs, _POINTS_METADATA_KEY)
    record = records.setdefault(points_name, {})
    if not isinstance(record, dict):
        raise ValueError(f"Harpy metadata harpy.points.{points_name} must be a mapping.")
    previous_panel = record.get("feature_panel")
    if previous_panel is not None and previous_panel != panel_name:
        raise ValueError(
            f"Points element {points_name!r} already references a different feature panel {previous_panel!r}. "
            "Panel replacement is not supported."
        )
    # The checks above reject conflicting panel contents or an existing reference
    # to a different panel. This only sets an unset reference or repeats the same one.
    record["feature_panel"] = panel_name
    panel_records.setdefault(panel_name, panel_record)

    columns = [feature_key, feature_class_key] if feature_class_key in points.columns else [feature_key]
    # Only one error/compatibility record per source partition reaches the
    # driver. Checking actual partition dtypes also handles unknown Dask categories.
    status = (
        points[columns]
        .map_partitions(
            _panel_registration_partition_status,
            panel=panel,
            meta=pd.DataFrame({"error": pd.Series(dtype="object"), "compatible": pd.Series(dtype=bool)}),
        )
        .compute()
    )
    errors = status["error"].dropna()
    if not errors.empty:
        raise ValueError(f"Points element {points_name!r}: {errors.iloc[0]}")
    # After value validation, leave points unchanged if every partition already
    # has a categorical class column with exactly panel.classes in that order
    # and ordered=False. Then only the panel metadata and reference need updating.
    rewrite_points = not status["compatible"].all()
    replacement = points
    if rewrite_points:
        log.info(
            f"Preparing replacement for points element {points_name!r}: "
            f"derive or normalize feature-class column {feature_class_key!r} to match the panel."
        )
        meta = points._meta.copy()
        meta[feature_class_key] = pd.Series(index=meta.index, dtype=pd.CategoricalDtype(panel.classes))
        replacement = points.map_partitions(_normalize_feature_classes, panel=panel, meta=meta)
        replacement.attrs.update(deepcopy(dict(points.attrs)))

    # Metadata-only registration and unbacked updates need no disk publisher.
    context = (
        _replace_element_on_disk(sdata, points_name, replacement, element_type="points")
        if backed and rewrite_points
        else nullcontext()
    )
    metadata_attempted = False
    try:
        with context:
            if rewrite_points and not backed:
                sdata.points[points_name] = replacement
            metadata_attempted = True
            sdata.attrs = attrs
            if backed:
                sdata.write_attrs()
                if not rewrite_points:
                    sdata.write_consolidated_metadata()
    except BaseException:
        # Catch context-exit failures too. The publisher restores points on
        # disk; this operation owns its root-metadata snapshot and memory state.
        sdata.points[points_name] = points
        sdata.attrs = previous_attrs
        if backed and metadata_attempted:
            try:
                sdata.write_attrs()
                sdata.write_consolidated_metadata()
            except Exception as error:  # noqa: BLE001
                log.warning(f"Could not restore feature-panel metadata for points {points_name!r}: {error}")
        raise
    return sdata


def _panel_registration_partition_status(partition: pd.DataFrame, *, panel: _FeaturePanelContract) -> pd.DataFrame:
    """Validate source values and report whether this partition needs normalization.

    Even a categorical partition with the right categories must pass feature
    membership and class-assignment checks. Return one compact record, including
    for empty partitions, without collecting the original point rows.
    """
    class_key = panel.feature_class_key
    if class_key in partition.columns:
        errors = _feature_panel_partition_errors(
            partition,
            feature_key=panel.feature_key,
            feature_class_key=class_key,
            class_by_feature=panel.class_by_feature,
        )
        dtype = partition[class_key].dtype
    else:
        errors = _feature_membership_partition_errors(
            partition, feature_key=panel.feature_key, class_by_feature=panel.class_by_feature
        )
        dtype = None
    compatible = (
        isinstance(dtype, pd.CategoricalDtype) and tuple(dtype.categories) == panel.classes and not dtype.ordered
    )
    return pd.DataFrame({"error": [None if errors.empty else errors.iloc[0]], "compatible": [compatible]})


def _normalize_feature_classes(partition: pd.DataFrame, *, panel: _FeaturePanelContract) -> pd.DataFrame:
    """Derive or normalize only the class column, preserving all other point data."""
    result = partition.copy()
    key = panel.feature_class_key
    values = result[key] if key in result else result[panel.feature_key].astype(object).map(panel.class_by_feature)
    if isinstance(values.dtype, pd.CategoricalDtype):
        # astype() can leave unordered categories in their old order because
        # pandas considers those dtypes equal; explicitly recode their order.
        result[key] = values.cat.set_categories(panel.classes, ordered=False)
    else:
        result[key] = values.astype(pd.CategoricalDtype(panel.classes))
    return result
