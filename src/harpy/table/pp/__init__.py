"""Preprocessing of tables in the style of ``scanpy.pp``: normalisations that scanpy does not provide, and QC metrics."""

from harpy.table.pp._normalize import normalize_by_quantile, normalize_by_size
from harpy.table.pp._qc_metrics import calculate_qc_metrics

__all__ = ["normalize_by_size", "normalize_by_quantile", "calculate_qc_metrics"]
