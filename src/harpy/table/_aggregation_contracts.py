"""Shared feature-class aggregation contracts, independent of table I/O."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from harpy._feature_panels import _FeaturePanelContract
from harpy.utils._names import _snake_case

# The dtype of the point-count matrices that aggregation writes: adata.X and the class matrices in
# .obsm. Not uint32: numba 0.67 and 0.68 type the sum of a uint8/16/32 array as int64, and then fail
# to compile an in-place add of that sum to an unsigned array, which scanpy's calculate_qc_metrics
# does on CSR input (https://github.com/numba/numba/issues/10939). int32 also leaves Harpy free of a
# numba pin, which would hold numpy back as well.
_COUNT_DTYPE = np.dtype(np.int32)


@dataclass(frozen=True)
class _FeatureClassAggregationContract:
    """Class-aware aggregation configuration for one compatible feature panel.

    The contract selects one panel class for ``adata.X``. Every remaining
    class gets its own count matrix in ``adata.obsm``, keyed by
    ``_snake_case(class)``. The ``.obsm`` keys and the feature axes are
    computed from ``panel``, the feature panel of the aggregated points,
    rather than stored, so they always follow its ``classes`` and
    ``features_by_class``.

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
        if not self.non_expression_classes:
            raise ValueError(
                "Class-aware aggregation requires at least one non-expression feature class. "
                "Use expression_class=None for a panel containing only the expression class."
            )
        keys = [key for _, key in self.non_expression_class_matrix_keys]
        if len(set(keys)) != len(keys):
            raise ValueError(f"Feature classes produce colliding .obsm keys: {keys!r}.")

    @property
    def non_expression_classes(self) -> tuple[str, ...]:
        return tuple(feature_class for feature_class in self.panel.classes if feature_class != self.expression_class)

    @property
    def non_expression_class_matrix_keys(self) -> tuple[tuple[str, str], ...]:
        """Return each non-expression class with the ``.obsm`` key of its count matrix."""
        return tuple((feature_class, _snake_case(feature_class)) for feature_class in self.non_expression_classes)

    @property
    def expression_feature_axis(self) -> tuple[str, ...]:
        """Return the ordered features defining the columns of ``adata.X``.

        The axis contains every feature assigned to ``expression_class`` in
        the authoritative panel's order, including features with no observed
        points. The same ordered values become ``adata.var_names``.
        """
        return self.panel.features_by_class[self.expression_class]

    @property
    def non_expression_class_matrix_axes(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Return the ``.obsm`` key and feature axis of each non-expression class's count matrix.

        Each axis contains every feature of that class in the panel's order,
        including features with no observed points, and becomes the matrix's
        ``feature_columns``.
        """
        features_by_class = self.panel.features_by_class
        return tuple(
            (key, features_by_class[feature_class]) for feature_class, key in self.non_expression_class_matrix_keys
        )
