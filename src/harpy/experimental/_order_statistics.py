"""Per-instance order statistics, computed with one vectorized sort per tile and channel.

All pixel values of the instances owned by a tile are sorted by ``(instance, value)`` in a
single sort. For dtypes of at most 32 bits, the dense instance index and an
order-preserving ``uint32`` encoding of the value are packed into one ``uint64`` key, so
the grouped sort is a plain :func:`numpy.sort` (several times faster than
:func:`numpy.lexsort`). Once the values are sorted, every quantile of every instance is a
gather at a computed position, without a Python loop over instances.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

SUPPORTED_STATS = ("median", "quantiles", "iqr", "mad")

_LOW_32_BITS = np.uint64(0xFFFFFFFF)
_SIGN_BIT_32 = np.uint32(0x80000000)
_ALL_BITS_32 = np.uint32(0xFFFFFFFF)


def _packable(dtype: np.dtype) -> bool:
    return dtype.kind in "biuf" and dtype.itemsize <= 4


def _to_sortable_uint32(values: NDArray) -> NDArray[np.uint32]:
    """Encode values of a dtype with at most 32 bits as ``uint32`` keys with the same order; NaN sorts last."""
    kind = values.dtype.kind
    if kind in "bu":
        return values.astype(np.uint32, copy=False)
    if kind == "i":
        return values.astype(np.int32, copy=False).view(np.uint32) ^ _SIGN_BIT_32
    values = values.astype(np.float32, copy=False)
    bits = values.view(np.uint32)
    keys = np.where((bits & _SIGN_BIT_32) != 0, ~bits, bits | _SIGN_BIT_32)
    keys[np.isnan(values)] = _ALL_BITS_32
    return keys


def _from_sortable_uint32(keys: NDArray[np.uint32], dtype: np.dtype) -> NDArray:
    """Inverse of :func:`_to_sortable_uint32`."""
    kind = dtype.kind
    if kind in "bu":
        return keys.astype(dtype)
    if kind == "i":
        return (keys ^ _SIGN_BIT_32).view(np.int32).astype(dtype)
    bits = np.where((keys & _SIGN_BIT_32) != 0, keys & ~_SIGN_BIT_32, ~keys)
    return bits.view(np.float32).astype(dtype)


def sort_within_segments(segments: NDArray[np.integer], values: NDArray) -> NDArray:
    """
    Sort `values` by ``(segments, values)``.

    `segments` holds a dense segment index (``0 <= segment < 2**32``) per value. NaN values
    sort last within their segment. Exact for every dtype; dtypes of at most 32 bits take the
    fast packed-key path, wider dtypes fall back to :func:`numpy.lexsort`.
    """
    if values.size == 0:
        return values.copy()
    if _packable(values.dtype):
        keys = segments.astype(np.uint64) << np.uint64(32)
        keys |= _to_sortable_uint32(values).astype(np.uint64)
        keys.sort()
        return _from_sortable_uint32((keys & _LOW_32_BITS).astype(np.uint32), values.dtype)
    return values[np.lexsort((values, segments))]


def _lerp(a: NDArray, b: NDArray, t: NDArray) -> NDArray:
    # same formula as numpy's quantile interpolation, so results match np.quantile(method="linear")
    diff = b - a
    out = a + diff * t
    upper = t >= 0.5
    out[upper] = (b - diff * (1 - t))[upper]
    return out


def segment_quantiles(
    sorted_values: NDArray,
    offsets: NDArray[np.intp],
    n_valid: NDArray[np.integer],
    q: Sequence[float] | NDArray,
) -> NDArray[np.float64]:
    """
    Quantiles (numpy's default ``"linear"`` method) of every segment of `sorted_values`.

    Segment ``k`` starts at ``offsets[k]``; its first ``n_valid[k]`` values are used (NaN values
    are sorted after them). Segments without valid values give NaN. Returns shape ``(K, len(q))``.
    """
    q = np.asarray(q, dtype=np.float64)
    out = np.full((n_valid.size, q.size), np.nan, dtype=np.float64)
    has_values = n_valid > 0
    if not has_values.any():
        return out
    n = n_valid[has_values][:, None]
    position = (n - 1) * q[None, :]
    below = np.floor(position).astype(np.intp)
    above = np.minimum(below + 1, n - 1)
    base = offsets[has_values][:, None]
    a = sorted_values[base + below].astype(np.float64)
    b = sorted_values[base + above].astype(np.float64)
    out[has_values] = _lerp(a, b, position - below)
    return out


def segment_mad(
    sorted_values: NDArray,
    offsets: NDArray[np.intp],
    counts: NDArray[np.integer],
    n_valid: NDArray[np.integer],
    median: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Median absolute deviation from the median (unscaled) of every segment.

    Integer data of at most 16 bits is exact. Other dtypes sort the deviations as
    ``float32``, which is exact up to float32 rounding (the precision of the output).
    """
    segments = np.repeat(np.arange(counts.size), counts)
    if sorted_values.dtype.kind in "biu" and sorted_values.dtype.itemsize <= 2:
        # 2 * |x - median| is an integer below 2**18, so uint32 keys are exact
        deviation = np.abs(2.0 * sorted_values - 2.0 * median[segments]).astype(np.uint32)
        scale = 0.5
    else:
        deviation = np.abs(sorted_values.astype(np.float64) - median[segments]).astype(np.float32)
        scale = 1.0
    deviation = sort_within_segments(segments, deviation)
    return segment_quantiles(deviation, offsets, n_valid, [0.5])[:, 0] * scale


@dataclass(frozen=True)
class StatSpec:
    """Which statistics to compute, and how each output feature is derived from the quantile table."""

    names: tuple[str, ...]
    quantiles: tuple[float, ...]
    # per output feature: ("quantile", (i,)) | ("iqr", (i_q25, i_q75)) | ("mad", (i_median,))
    sources: tuple[tuple[str, tuple[int, ...]], ...]


def build_stat_spec(stats_funcs: str | Sequence[str], q: float | Sequence[float] | None = None) -> StatSpec:
    """Validate the requested statistics and collect every quantile they need."""
    if isinstance(stats_funcs, str):
        stats_funcs = (stats_funcs,)
    stats_funcs = tuple(stats_funcs)
    invalid = [s for s in stats_funcs if s not in SUPPORTED_STATS]
    if invalid:
        raise ValueError(f"Unsupported statistic(s) {invalid} in 'stats_funcs'. Choose from {list(SUPPORTED_STATS)}.")
    if not stats_funcs:
        raise ValueError(f"'stats_funcs' is empty. Choose one or more of {list(SUPPORTED_STATS)}.")
    if "quantiles" in stats_funcs:
        if q is None:
            raise ValueError(
                "'stats_funcs' contains 'quantiles', but 'q' is None. Pass the quantiles, e.g. 'q=(0.1, 0.9)'."
            )
        q = tuple(float(v) for v in np.atleast_1d(q))
        out_of_range = [v for v in q if not 0.0 <= v <= 1.0]
        if out_of_range:
            raise ValueError(f"Quantiles {out_of_range} in 'q' are outside [0, 1].")
    else:
        q = ()

    quantiles: list[float] = []

    def index_of(value: float) -> int:
        if value not in quantiles:
            quantiles.append(value)
        return quantiles.index(value)

    names: list[str] = []
    sources: list[tuple[str, tuple[int, ...]]] = []
    for stat in stats_funcs:
        if stat == "median":
            names.append("median")
            sources.append(("quantile", (index_of(0.5),)))
        elif stat == "quantiles":
            for value in q:
                names.append(f"q{value:g}")
                sources.append(("quantile", (index_of(value),)))
        elif stat == "iqr":
            names.append("iqr")
            sources.append(("iqr", (index_of(0.25), index_of(0.75))))
        elif stat == "mad":
            names.append("mad")
            sources.append(("mad", (index_of(0.5),)))
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate statistics requested: {names}. Request each statistic once.")
    return StatSpec(names=tuple(names), quantiles=tuple(quantiles), sources=tuple(sources))


def compute_segment_stats(
    sorted_values: NDArray,
    offsets: NDArray[np.intp],
    counts: NDArray[np.integer],
    n_valid: NDArray[np.integer],
    spec: StatSpec,
) -> NDArray[np.float64]:
    """Evaluate `spec` on every segment of `sorted_values`. Returns shape ``(K, len(spec.names))``."""
    table = segment_quantiles(sorted_values, offsets, n_valid, spec.quantiles)
    out = np.empty((counts.size, len(spec.names)), dtype=np.float64)
    for j, (kind, idx) in enumerate(spec.sources):
        if kind == "quantile":
            out[:, j] = table[:, idx[0]]
        elif kind == "iqr":
            out[:, j] = table[:, idx[1]] - table[:, idx[0]]
        elif kind == "mad":
            out[:, j] = segment_mad(sorted_values, offsets, counts, n_valid, table[:, idx[0]])
    return out
