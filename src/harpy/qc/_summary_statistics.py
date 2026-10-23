"""Statistics shared by the QC summaries, so that their conventions cannot drift apart."""

from __future__ import annotations

import numpy as np

from harpy.qc._summary_schema import _statistic_name


def _points_statistics(values: np.ndarray, *, unit: str, measure: str = "points") -> dict[str, float]:
    """Return the mean, median, sample SD and 95th percentile of ``values``, named per ``measure`` and ``unit``.

    The keys follow ``_statistic_name`` of ``harpy.qc._summary_schema``: ``mean_{measure}_per_{unit}``,
    ``median_…``, ``std_…`` and ``p95_…``, for example ``mean_points_per_bin``
    or ``mean_detected_features_per_instance``. Every value counts, zeros
    included. The SD uses ``ddof=1`` and is NaN for fewer than two values; the
    percentile uses linear interpolation; all four are NaN for no values.
    """
    n_values = len(values)
    return {
        _statistic_name("mean", unit=unit, measure=measure): float(values.mean()) if n_values else np.nan,
        _statistic_name("median", unit=unit, measure=measure): float(np.median(values)) if n_values else np.nan,
        _statistic_name("std", unit=unit, measure=measure): float(values.std(ddof=1)) if n_values > 1 else np.nan,
        _statistic_name("p95", unit=unit, measure=measure): float(np.percentile(values, 95)) if n_values else np.nan,
    }
