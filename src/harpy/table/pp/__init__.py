"""Normalisations of table matrices that scanpy does not provide, in the style of ``scanpy.pp``."""

from harpy.table.pp._normalize import normalize_by_quantile, normalize_by_size

__all__ = ["normalize_by_size", "normalize_by_quantile"]
