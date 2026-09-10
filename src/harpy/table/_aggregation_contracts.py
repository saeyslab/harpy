"""Shared feature-class aggregation contracts, independent of table I/O."""

from __future__ import annotations

import re
from dataclasses import dataclass

from harpy._feature_panels import _FeaturePanelContract


@dataclass(frozen=True)
class _FeatureClassAggregationContract:
    """Class-aware aggregation configuration for one compatible feature panel.

    The contract selects one panel class for ``adata.X``. Every remaining
    class is treated as an auxiliary class. Output count-column names,
    auxiliary classes, and per-class feature counts are derived from the panel,
    so they cannot disagree with its metadata. The contract contains no spatial
    assignment results or observed point counts.

    Attributes
    ----------
    panel
        Normalized panel shared by every points element in the aggregation.
    expression_class
        Panel class whose features are retained in ``adata.X``.
    """

    panel: _FeaturePanelContract
    expression_class: str

    def __post_init__(self) -> None:
        if not isinstance(self.expression_class, str) or not self.expression_class:
            raise ValueError(
                f"Parameter 'expression_class' must be a non-empty string, found {self.expression_class!r}."
            )
        if self.expression_class not in self.panel.classes:
            raise ValueError(
                f"Expression class {self.expression_class!r} is not present in panel classes "
                f"{list(self.panel.classes)!r}."
            )
        if not self.auxiliary_classes:
            raise ValueError(
                "Class-aware aggregation requires at least one non-expression feature class. "
                "Use expression_class=None for a panel containing only the expression class."
            )
        generated = [column for _, column in self.count_columns]
        if len(set(generated)) != len(generated):
            raise ValueError(f"Feature classes produce colliding count-column names: {generated!r}.")

    @property
    def count_columns(self) -> tuple[tuple[str, str], ...]:
        return tuple((feature_class, f"n_{_snake_case(feature_class)}_points") for feature_class in self.panel.classes)

    @property
    def auxiliary_classes(self) -> tuple[str, ...]:
        return tuple(feature_class for feature_class in self.panel.classes if feature_class != self.expression_class)

    @property
    def expression_feature_axis(self) -> tuple[str, ...]:
        """Return the ordered features defining the columns of ``adata.X``.

        The axis contains every feature assigned to ``expression_class`` in
        the authoritative panel's order, including features with no observed
        points. The same ordered values become ``adata.var_names``.
        """
        return self.panel.features_by_class[self.expression_class]

    @property
    def auxiliary_feature_axis(self) -> tuple[str, ...]:
        """Return the ordered features defining the auxiliary matrix columns.

        Features outside ``expression_class`` are concatenated first in the
        panel's class order and then in each class's feature order, including
        features with no observed points. The resulting axis describes
        ``adata.obsm["auxiliary_feature_counts"]`` and is recorded as that
        matrix's ``feature_columns`` metadata.
        """
        features_by_class = self.panel.features_by_class
        return tuple(
            feature for feature_class in self.auxiliary_classes for feature in features_by_class[feature_class]
        )

    @property
    def auxiliary_class_slices(self) -> tuple[tuple[str, slice], ...]:
        """Map each non-expression class to its columns on the auxiliary axis.

        Each slice selects that class's contiguous feature block from
        :attr:`auxiliary_feature_axis`. Because this axis also defines the
        columns of ``adata.obsm["auxiliary_feature_counts"]``, the same slice
        selects the class's columns from the auxiliary matrix for calculating
        per-instance class totals.
        """
        result: list[tuple[str, slice]] = []
        start = 0
        features_by_class = self.panel.features_by_class
        for feature_class in self.auxiliary_classes:
            stop = start + len(features_by_class[feature_class])
            result.append((feature_class, slice(start, stop)))
            start = stop
        return tuple(result)

    @property
    def auxiliary_class_feature_counts(self) -> tuple[tuple[str, int], ...]:
        """Return the panel-defined feature count for each non-expression class.

        Each value is the number of features assigned to that auxiliary class
        in the authoritative panel, including panel features for which the
        selected points elements contain zero detections. The values are
        recorded with the aggregation metadata so downstream QC can normalize
        per-class point counts without inferring panel size from observed data.
        """
        features_by_class = self.panel.features_by_class
        return tuple((feature_class, len(features_by_class[feature_class])) for feature_class in self.auxiliary_classes)


def _snake_case(value: str) -> str:
    value = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", value)
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
    value = re.sub(r"[^\w]+", "_", value, flags=re.UNICODE)
    value = re.sub(r"_+", "_", value).strip("_").casefold()
    if not value:
        raise ValueError("Feature-class names must produce a non-empty snake-case output name.")
    return value
