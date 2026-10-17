"""Output names derived from user-supplied names, shared across Harpy's packages."""

from __future__ import annotations

import re


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
