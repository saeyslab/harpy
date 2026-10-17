"""Statistics shared by the QC summaries, so that their conventions cannot drift apart."""

from __future__ import annotations

import numpy as np


def _points_statistics(values: np.ndarray, *, unit: str, measure: str = "points") -> dict[str, float]:
    """Return the mean, median, sample SD and 95th percentile of ``values``, named per ``measure`` and ``unit``.

    The keys are ``mean_{measure}_per_{unit}``, ``median_{measure}_per_{unit}``,
    ``std_{measure}_per_{unit}`` and ``p95_{measure}_per_{unit}``, for example
    ``mean_points_per_bin`` or ``mean_detected_features_per_instance``. Every
    value counts, zeros included. The SD uses ``ddof=1`` and is NaN for fewer
    than two values; the percentile uses linear interpolation; all four are NaN
    for no values.
    """
    n_values = len(values)
    return {
        f"mean_{measure}_per_{unit}": float(values.mean()) if n_values else np.nan,
        f"median_{measure}_per_{unit}": float(np.median(values)) if n_values else np.nan,
        f"std_{measure}_per_{unit}": float(values.std(ddof=1)) if n_values > 1 else np.nan,
        f"p95_{measure}_per_{unit}": float(np.percentile(values, 95)) if n_values else np.nan,
    }
