"""Names derived from user-supplied names, and selections of recorded names, shared across Harpy's packages."""

from __future__ import annotations

import re
from collections.abc import Sequence


def _names(value: object) -> tuple[str, ...]:
    """Return a recorded name or sequence of names as a tuple of strings."""
    if isinstance(value, str):
        return (value,)
    return tuple(str(name) for name in value)


def _selected_names(
    requested: str | Sequence[str] | None, *, recorded: tuple[str, ...], parameter: str, kind: str
) -> tuple[str, ...]:
    """Return the requested names in their recorded order, or every recorded name for ``None``.

    ``requested`` is one name or a sequence of distinct names, each among
    ``recorded``; ``parameter`` and ``kind`` word the errors, for example
    ``"region"`` and ``"regions"``.
    """
    if requested is None:
        return recorded
    if not isinstance(requested, (str, Sequence)):
        raise ValueError(f"{parameter} must contain one or more exact names of {kind}.")
    names = (requested,) if isinstance(requested, str) else tuple(requested)
    if not names or any(not isinstance(name, str) for name in names):
        raise ValueError(f"{parameter} must contain one or more exact names of {kind}.")
    if len(set(names)) != len(names):
        raise ValueError(f"{parameter} must not contain duplicate names.")
    unknown = set(names) - set(recorded)
    if unknown:
        raise ValueError(f"Unknown {kind}: {sorted(unknown)}; choose from {list(recorded)}.")
    return tuple(name for name in recorded if name in names)


def _snake_case(value: str) -> str:
    """Return ``value`` in snake case, for example ``"SystemControl"`` -> ``"system_control"``.

    Used for the ``.obsm`` key of a class matrix of :func:`harpy.tb.aggregate_points`
    and for the class labels of :func:`harpy.qc.summarize_table`, so that both
    follow one rule.
    """
    name = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", value)
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    name = re.sub(r"[^\w]+", "_", name, flags=re.UNICODE)
    name = re.sub(r"_+", "_", name).strip("_").casefold()
    if not name:
        raise ValueError(f"Name {value!r} has no snake-case form: it must contain a letter or digit.")
    return name
