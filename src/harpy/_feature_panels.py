"""Shared, read-only feature-panel contracts for point consumers."""

from collections.abc import Mapping
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

    @property
    def features_by_class(self) -> dict[str, tuple[str, ...]]:
        return dict(self.features_by_class_items)

    @property
    def class_by_feature(self) -> dict[str, str]:
        return {
            feature: feature_class for feature_class, features in self.features_by_class_items for feature in features
        }


def _resolve_points_feature_panel(
    sdata: SpatialData, points_name: str
) -> tuple[str, _FeaturePanelContract, Mapping[str, object]]:
    """Resolve a points element's authoritative panel and source metadata, without writes."""
    root = _require_mapping(sdata.attrs.get(_HARPY_METADATA_KEY), path=_HARPY_METADATA_KEY)
    version = root.get(_METADATA_VERSION_KEY)
    if isinstance(version, bool) or not isinstance(version, int) or version != _METADATA_VERSION:
        raise ValueError(f"Harpy metadata version must equal {_METADATA_VERSION}, found {version!r}.")
    points_records = _require_mapping(root.get(_POINTS_METADATA_KEY), path="harpy.points")
    record = _require_mapping(points_records.get(points_name), path=f"harpy.points.{points_name}")
    panel_name = _require_nonempty_string(record.get("feature_panel"), path=f"harpy.points.{points_name}.feature_panel")
    panels = _require_mapping(root.get(_FEATURE_PANELS_METADATA_KEY), path="harpy.feature_panels")
    panel_record = _require_mapping(panels.get(panel_name), path=f"harpy.feature_panels.{panel_name}")
    return panel_name, _parse_feature_panel(panel_record, panel_name=panel_name), record


def _parse_feature_panel(record: Mapping[str, object], *, panel_name: str) -> _FeaturePanelContract:
    path = f"{_HARPY_METADATA_KEY}.{_FEATURE_PANELS_METADATA_KEY}.{panel_name}"
    feature_key = _require_nonempty_string(record.get("feature_key"), path=f"{path}.feature_key")
    feature_class_key = _require_nonempty_string(
        record.get("feature_class_key"),
        path=f"{path}.feature_class_key",
    )
    if feature_key == feature_class_key:
        raise ValueError(f"Feature panel {panel_name!r} must use different feature and feature-class keys.")

    classes_value = record.get("classes")
    if not isinstance(classes_value, list) or not classes_value:
        raise ValueError(f"Harpy metadata {path}.classes must be a non-empty list of strings.")
    classes = tuple(_require_nonempty_string(value, path=f"{path}.classes item") for value in classes_value)
    if len(set(classes)) != len(classes):
        raise ValueError(f"Harpy metadata {path}.classes must contain unique values.")

    grouped = _require_mapping(record.get("features_by_class"), path=f"{path}.features_by_class")
    if set(grouped) != set(classes):
        raise ValueError(f"Harpy metadata {path}.features_by_class must contain exactly the declared classes.")
    features_by_class_items: list[tuple[str, tuple[str, ...]]] = []
    seen_features: dict[str, str] = {}
    for feature_class in classes:
        values = grouped[feature_class]
        if not isinstance(values, list) or not values:
            raise ValueError(f"Harpy metadata {path}.features_by_class[{feature_class!r}] must be non-empty.")
        features = tuple(
            _require_nonempty_string(value, path=f"{path}.features_by_class[{feature_class!r}] item")
            for value in values
        )
        if len(set(features)) != len(features):
            raise ValueError(f"Harpy metadata {path}.features_by_class[{feature_class!r}] must contain unique values.")
        for feature in features:
            previous = seen_features.setdefault(feature, feature_class)
            if previous != feature_class:
                raise ValueError(
                    f"Feature {feature!r} belongs to both {previous!r} and {feature_class!r} in panel {panel_name!r}."
                )
        features_by_class_items.append((feature_class, features))
    return _FeaturePanelContract(
        feature_key=feature_key,
        feature_class_key=feature_class_key,
        classes=classes,
        features_by_class_items=tuple(features_by_class_items),
    )


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


def _feature_panel_partition_errors(
    partition: pd.DataFrame,
    *,
    feature_key: str,
    feature_class_key: str,
    class_by_feature: Mapping[str, str],
) -> pd.Series:
    """Validate one points partition against its feature-panel assignments.

    Each point must contain a non-null feature and feature class. Its feature
    must occur in ``class_by_feature``, and its observed feature class must
    equal the class assigned by that mapping. Validation is partition-wise so
    the complete points element does not need to be materialized.

    Parameters
    ----------
    partition
        One in-memory pandas partition from the source Dask points element.
    feature_key
        Name of the column containing feature identifiers, such as genes.
    feature_class_key
        Name of the column containing feature classes.
    class_by_feature
        Expected feature class for every feature declared by the panel.

    Returns
    -------
    pandas.Series
        An empty series when the partition is valid, or a one-element series
        containing the first validation error found in the partition.

    Examples
    --------
    ``EPCAM`` is declared endogenous, so observing it as negative produces a
    compact error that the caller can collect alongside the Dask reductions:

    >>> partition = pd.DataFrame({"gene": ["EPCAM"], "code_class": ["Negative"]})
    >>> errors = _feature_panel_partition_errors(
    ...     partition,
    ...     feature_key="gene",
    ...     feature_class_key="code_class",
    ...     class_by_feature={"EPCAM": "Endogenous"},
    ... )
    >>> errors.iloc[0]
    "feature 'EPCAM' has class 'Negative'; expected 'Endogenous'."
    """
    features = partition[feature_key]
    feature_classes = partition[feature_class_key]
    invalid = features.isna() | feature_classes.isna()
    if invalid.any():
        return pd.Series(["feature and feature-class values must not be null."], name="error", dtype="object")
    expected = features.astype(object).map(class_by_feature)
    missing = expected.isna()
    if missing.any():
        feature = features.loc[missing].iloc[0]
        return pd.Series([f"feature {feature!r} is absent from the panel."], name="error", dtype="object")
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
