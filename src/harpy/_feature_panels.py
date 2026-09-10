"""Shared feature-panel serialization, identity and read-only validation."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from dask.dataframe import DataFrame as DaskDataFrame
from spatialdata import SpatialData

from harpy._metadata import (
    _FEATURE_PANELS_METADATA_KEY,
    _HARPY_METADATA_KEY,
    _METADATA_VERSION,
    _METADATA_VERSION_KEY,
    _POINTS_METADATA_KEY,
)


@dataclass(frozen=True)
class _FeaturePanelContract:
    """Normalized immutable description of a points feature panel.

    The contract describes source metadata shared by compatible points
    elements. It does not select which feature class contributes to an
    expression matrix and contains no observed point counts.

    Construction validates immutable structure, class alignment and unique
    feature membership. It does not sort fields or inspect any points or store.

    Attributes
    ----------
    feature_key
        Points column containing feature identifiers, such as gene names.
    feature_class_key
        Categorical points column assigning each feature to a panel class.
    classes
        Ordered feature classes declared by the panel.
    features_by_class_items
        Immutable ordered ``(feature class, features)`` representation of the
        panel's class-to-features mapping.
    """

    feature_key: str
    feature_class_key: str
    classes: tuple[str, ...]
    features_by_class_items: tuple[tuple[str, tuple[str, ...]], ...]

    def __post_init__(self) -> None:
        """Enforce panel invariants without changing the supplied ordering."""
        _require_nonempty_string(self.feature_key, path="feature_key")
        _require_nonempty_string(self.feature_class_key, path="feature_class_key")
        if self.feature_key == self.feature_class_key:
            raise ValueError("Feature panels must use different feature and feature-class keys.")
        if not isinstance(self.classes, tuple) or not self.classes:
            raise ValueError("Feature-panel classes must be a non-empty tuple of strings.")
        for feature_class in self.classes:
            _require_nonempty_string(feature_class, path="classes item")
        if len(set(self.classes)) != len(self.classes):
            raise ValueError("Feature-panel classes must contain unique values.")
        if not isinstance(self.features_by_class_items, tuple):
            raise ValueError("Feature-panel features_by_class_items must be a tuple.")
        for item in self.features_by_class_items:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError("Feature-panel items must be (class, features) tuples.")
            _require_nonempty_string(item[0], path="features_by_class_items class")
        if tuple(item[0] for item in self.features_by_class_items) != self.classes:
            raise ValueError("Feature-panel items must contain exactly the declared classes, in the same order.")

        seen_features: dict[str, str] = {}
        for feature_class, features in self.features_by_class_items:
            if not isinstance(features, tuple) or not features:
                raise ValueError(f"Feature-panel features for {feature_class!r} must be a non-empty tuple.")
            for feature in features:
                _require_nonempty_string(feature, path=f"features_by_class[{feature_class!r}] item")
            if len(set(features)) != len(features):
                raise ValueError(f"Feature-panel features for {feature_class!r} must contain unique values.")
            for feature in features:
                previous = seen_features.setdefault(feature, feature_class)
                if previous != feature_class:
                    raise ValueError(f"Feature {feature!r} belongs to both {previous!r} and {feature_class!r}.")

    @property
    def features_by_class(self) -> dict[str, tuple[str, ...]]:
        return dict(self.features_by_class_items)

    @property
    def class_by_feature(self) -> dict[str, str]:
        return {
            feature: feature_class for feature_class, features in self.features_by_class_items for feature in features
        }

    @property
    def storage_key(self) -> str:
        """Derive an order-independent feature-panel registry key from its contents.

        Sort a fresh record's classes and features for hashing, without changing
        this panel's ordering. The key is also independent of sample IDs and
        points element names, including for panels read from existing metadata.

        The first 16 hexadecimal characters of the SHA-256 digest identify
        contents for deduplication, not security. Callers must still compare
        normalized records before reusing a key and reject conflicting contents.
        """
        canonical = json.dumps(
            self._canonical_record(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return f"feature_panel_{hashlib.sha256(canonical).hexdigest()[:16]}"

    def to_dict(self) -> dict[str, object]:
        """Return a fresh storage record, not a dictionary of internal dataclass fields.

        Convert the immutable class/feature pairs to ``features_by_class`` and
        lists, preserving class and feature order. This performs no validation,
        hashing or disk writes. Order-independent identity is handled separately
        by ``storage_key``.
        """
        return {
            "feature_key": self.feature_key,
            "feature_class_key": self.feature_class_key,
            "classes": list(self.classes),
            "features_by_class": {name: list(features) for name, features in self.features_by_class_items},
        }

    def _canonical_record(self) -> dict[str, object]:
        """Normalize a fresh record for hashing and content comparison only."""
        record = self.to_dict()
        record["classes"] = sorted(self.classes)
        record["features_by_class"] = {
            feature_class: sorted(features) for feature_class, features in self.features_by_class_items
        }
        return record


def _resolve_points_feature_panel(
    sdata: SpatialData, points_name: str
) -> tuple[str, _FeaturePanelContract, Mapping[str, object]]:
    """Resolve a points element's authoritative panel and source metadata, without writes."""
    missing_panel = (
        f"Points element {points_name!r} has no feature-panel mapping. Register its complete assay panel with "
        "hp.pt.add_feature_panel(sdata, points_name=..., feature_key=..., feature_class_key=..., "
        "features_by_class=...)."
    )
    if sdata.attrs.get(_HARPY_METADATA_KEY) is None:
        raise ValueError(missing_panel)
    root = _require_mapping(sdata.attrs[_HARPY_METADATA_KEY], path=_HARPY_METADATA_KEY)
    version = root.get(_METADATA_VERSION_KEY)
    if isinstance(version, bool) or not isinstance(version, int) or version != _METADATA_VERSION:
        raise ValueError(f"Harpy metadata version must equal {_METADATA_VERSION}, found {version!r}.")
    points_records = _require_mapping(root.get(_POINTS_METADATA_KEY, {}), path="harpy.points")
    record = _require_mapping(points_records.get(points_name, {}), path=f"harpy.points.{points_name}")
    if record.get("feature_panel") is None:
        raise ValueError(missing_panel)
    panel_name = _require_nonempty_string(record.get("feature_panel"), path=f"harpy.points.{points_name}.feature_panel")
    panels = _require_mapping(root.get(_FEATURE_PANELS_METADATA_KEY, {}), path="harpy.feature_panels")
    if panel_name not in panels:
        raise ValueError(missing_panel)
    panel_record = _require_mapping(panels.get(panel_name), path=f"harpy.feature_panels.{panel_name}")
    return panel_name, _parse_feature_panel(panel_record, panel_name=panel_name), record


def _make_feature_panel(
    *,
    feature_key: str,
    feature_class_key: str,
    features_by_class: Mapping[str, Sequence[str]],
) -> _FeaturePanelContract:
    """Validate supplied panel fields and construct a canonical in-memory panel.

    Classes and their features are sorted as the default storage ordering for
    new panels. Identity hashing is independently order-insensitive. Names are
    kept exactly as supplied, and duplicates are rejected rather than discarded.
    Input containers are converted here; the dataclass validates panel invariants on construction.
    No storage record is built or parsed here.
    """
    if not isinstance(features_by_class, Mapping):
        raise ValueError("features_by_class must be a mapping from classes to sequences of feature names.")
    items = []
    for feature_class, features in features_by_class.items():
        if isinstance(features, (str, bytes)) or not isinstance(features, Sequence):
            raise ValueError(f"features_by_class[{feature_class!r}] must be a non-string sequence of feature names.")
        items.append((feature_class, tuple(features)))
    try:
        canonical_items = tuple((feature_class, tuple(sorted(features))) for feature_class, features in sorted(items))
    except TypeError as error:
        # Mixed types may fail sorting before the dataclass can check them.
        raise ValueError("Feature-panel classes and feature identifiers must be strings.") from error
    return _FeaturePanelContract(
        feature_key=feature_key,
        feature_class_key=feature_class_key,
        classes=tuple(feature_class for feature_class, _ in canonical_items),
        features_by_class_items=canonical_items,
    )


def _validate_feature_panel_collision(
    panels: Mapping[str, object],
    panel_name: str,
    panel: _FeaturePanelContract,
) -> _FeaturePanelContract:
    """Check key reuse by normalized contents and retain the stored panel's order.

    Return the supplied panel when its key is absent. Otherwise parse and compare
    the stored record, ignoring only class and feature ordering, not differences
    in other fields. Matching hashes alone are insufficient. Return the existing
    panel on a match so callers can align categorical columns with its stored
    ordering. Neither the registry nor its records are modified here.
    """
    if panel_name not in panels:
        return panel
    try:
        existing_record = _require_mapping(panels[panel_name], path=f"harpy.feature_panels.{panel_name}")
        existing_panel = _parse_feature_panel(existing_record, panel_name=panel_name)
    except ValueError as error:
        raise ValueError(f"Harpy feature-panel hash collision for {panel_name!r}: {error}") from error
    # Preserve extension fields in the comparison: only ordering is ignored.
    existing_canonical = dict(existing_record)
    existing_canonical.update(existing_panel._canonical_record())
    if existing_canonical != panel._canonical_record():
        raise ValueError(f"Harpy feature-panel hash collision for {panel_name!r}.")
    return existing_panel


def _parse_feature_panel(panel_record: Mapping[str, object], *, panel_name: str) -> _FeaturePanelContract:
    """Validate a stored panel record and restore its declared ordering.

    Storage-specific checks cover list encoding and mapping coverage. The
    dataclass validates panel invariants on construction. Stored ordering is
    retained because downstream consumers may rely on it.
    """
    path = f"{_HARPY_METADATA_KEY}.{_FEATURE_PANELS_METADATA_KEY}.{panel_name}"
    feature_key = _require_nonempty_string(panel_record.get("feature_key"), path=f"{path}.feature_key")
    feature_class_key = _require_nonempty_string(
        panel_record.get("feature_class_key"),
        path=f"{path}.feature_class_key",
    )
    classes_value = panel_record.get("classes")
    if not isinstance(classes_value, list):
        raise ValueError(f"Harpy metadata {path}.classes must be a list of strings.")
    classes = tuple(_require_nonempty_string(value, path=f"{path}.classes item") for value in classes_value)

    grouped = _require_mapping(panel_record.get("features_by_class"), path=f"{path}.features_by_class")
    if set(grouped) != set(classes):
        raise ValueError(f"Harpy metadata {path}.features_by_class must contain exactly the declared classes.")
    for feature_class in classes:
        values = grouped[feature_class]
        if not isinstance(values, list):
            raise ValueError(f"Harpy metadata {path}.features_by_class[{feature_class!r}] must be a list of strings.")
    try:
        return _FeaturePanelContract(
            feature_key=feature_key,
            feature_class_key=feature_class_key,
            classes=classes,
            features_by_class_items=tuple((feature_class, tuple(grouped[feature_class])) for feature_class in classes),
        )
    except ValueError as error:
        raise ValueError(f"Harpy metadata {path}: {error}") from error


def _validate_feature_class_dtype(
    points: DaskDataFrame,
    *,
    points_name: str,
    panel: _FeaturePanelContract,
) -> None:
    dtype = points.dtypes[panel.feature_class_key]
    if not isinstance(dtype, pd.CategoricalDtype):
        raise ValueError(
            f"Points element {points_name!r} feature-class column {panel.feature_class_key!r} "
            f"must be categorical, found {dtype}."
        )
    column = points[panel.feature_class_key]
    if column.cat.known:
        categories = tuple(column.cat.categories.tolist())
        if categories != panel.classes:
            raise ValueError(
                f"Points element {points_name!r} feature-class categories {list(categories)!r} "
                f"do not match panel classes {list(panel.classes)!r}."
            )


def _feature_membership_partition_errors(
    partition: pd.DataFrame,
    *,
    feature_key: str,
    class_by_feature: Mapping[str, str],
) -> pd.Series:
    """Check that one points partition contains only declared, non-null features.

    No class column is inspected. Use this check before deriving a missing
    class column from ``class_by_feature``.

    Parameters
    ----------
    partition
        One in-memory pandas partition from the source Dask points element.
    feature_key
        Name of the points column containing feature identifiers.
    class_by_feature
        The panel's feature-to-class mapping, e.g.
        ``{"EPCAM": "Endogenous", "Negative1": "Negative"}``. This is the
        relevant part of the panel, not its complete metadata record. Keys
        define the allowed features; this helper does not inspect the values.

    Returns
    -------
    pandas.Series
        An empty series when all features are valid, or a one-element series
        containing the first feature-validation error found in the partition.

    Examples
    --------
    No class column is needed. ``EPCAM`` occurs in ``class_by_feature``, but
    ``UnknownGene`` does not, so the partition produces an error:

    .. code-block:: python

        partition = pd.DataFrame({"gene": ["EPCAM", "UnknownGene"]})
        errors = _feature_membership_partition_errors(
            partition,
            feature_key="gene",
            class_by_feature={"EPCAM": "Endogenous"},
        )
        errors.iloc[0]
        # "feature 'UnknownGene' is absent from the panel."
    """
    features = partition[feature_key]
    if features.isna().any():
        return pd.Series(["feature values must not be null."], name="error", dtype="object")
    missing = ~features.isin(class_by_feature)
    if missing.any():
        feature = features.loc[missing].iloc[0]
        return pd.Series([f"feature {feature!r} is absent from the panel."], name="error", dtype="object")
    return pd.Series(name="error", dtype="object")


def _feature_panel_partition_errors(
    partition: pd.DataFrame,
    *,
    feature_key: str,
    feature_class_key: str,
    class_by_feature: Mapping[str, str],
) -> pd.Series:
    """Validate one points partition against its feature-panel assignments.

    Each point must contain a non-null feature declared in ``class_by_feature``.
    Its observed class must also be non-null and equal the value assigned by
    that mapping. Feature membership is checked by
    ``_feature_membership_partition_errors()`` within the same partition task;
    this does not introduce another Dask scan of the points.

    Parameters
    ----------
    partition
        One in-memory pandas partition from the source Dask points element.
    feature_key
        Name of the column containing feature identifiers, such as genes.
    feature_class_key
        Name of the existing points column containing observed feature classes.
        To validate features before creating a missing class column, use
        ``_feature_membership_partition_errors()`` instead.
    class_by_feature
        The panel's feature-to-class mapping, e.g.
        ``{"EPCAM": "Endogenous", "Negative1": "Negative"}``. This is the
        relevant part of the panel, not its complete metadata record. Keys
        define the allowed features and values define their expected classes.

    Returns
    -------
    pandas.Series
        An empty series when the partition is valid, or a one-element series
        containing the first validation error found in the partition.

    Examples
    --------
    ``EPCAM`` is declared endogenous, so observing it as negative produces a
    compact error that the caller can collect alongside the Dask reductions:

    .. code-block:: python

        partition = pd.DataFrame({"gene": ["EPCAM"], "code_class": ["Negative"]})
        errors = _feature_panel_partition_errors(
            partition,
            feature_key="gene",
            feature_class_key="code_class",
            class_by_feature={"EPCAM": "Endogenous"},
        )
        errors.iloc[0]
        # "feature 'EPCAM' has class 'Negative'; expected 'Endogenous'."
    """
    errors = _feature_membership_partition_errors(partition, feature_key=feature_key, class_by_feature=class_by_feature)
    if not errors.empty:
        return errors
    features = partition[feature_key]
    feature_classes = partition[feature_class_key]
    if feature_classes.isna().any():
        return pd.Series(["feature-class values must not be null."], name="error", dtype="object")
    expected = features.astype(object).map(class_by_feature)
    mismatched = feature_classes.astype(object) != expected
    if mismatched.any():
        position = int(np.flatnonzero(mismatched.to_numpy())[0])
        return pd.Series(
            [
                f"feature {features.iloc[position]!r} has class {feature_classes.iloc[position]!r}; "
                f"expected {expected.iloc[position]!r}."
            ],
            name="error",
            dtype="object",
        )
    return pd.Series(name="error", dtype="object")


def _require_mapping(value: object, *, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Harpy metadata {path} must be a mapping.")
    return value


def _require_nonempty_string(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Harpy metadata {path} must be a non-empty string.")
    return value
