"""Harpy's deprecations: a ``FutureWarning`` with one message template, attributed to the caller."""

from __future__ import annotations

import functools
import os
import sys
import warnings
from collections.abc import Callable, Mapping

import lazy_loader


def _skipped_prefixes() -> tuple[str, ...]:
    """Return the paths whose frames a deprecation warning skips: Harpy's code, without its tests, and lazy_loader.

    The warning then names the first frame outside them: the caller of the
    deprecated function, at any call depth. The tests live inside the package,
    so ``harpy/_tests`` is not skipped, and a test is named as the caller.
    """
    harpy_directory = os.path.dirname(os.path.dirname(__file__))
    prefixes = []
    for entry in os.scandir(harpy_directory):
        if entry.name != "_tests":
            # The separator keeps the prefix of harpy/qc from matching harpy/qc_other.py.
            prefixes.append(entry.path + os.sep if entry.is_dir() else entry.path)
    prefixes.append(os.path.dirname(lazy_loader.__file__) + os.sep)
    return tuple(prefixes)


_SKIPPED_PREFIXES = _skipped_prefixes()


def _warn_deprecated(deprecated: str, *, use: str) -> None:
    """Warn with a ``FutureWarning`` that ``deprecated`` is deprecated, and say what to use instead.

    ``deprecated`` names the function, method or parameter, and ``use``
    completes "Use …"::

        _warn_deprecated("harpy.qc.obs_scatter", use="harpy.qc.table_scatter, which takes the same arguments")
        _warn_deprecated("The labels_name parameter of harpy.qc.table_histogram", use="region")

    give

    .. code-block:: text

        harpy.qc.obs_scatter is deprecated and will be removed in a future release. Use harpy.qc.table_scatter,
        which takes the same arguments.
        The labels_name parameter of harpy.qc.table_histogram is deprecated and will be removed in a future
        release. Use region.

    The warning names the caller of the deprecated function, outside Harpy.
    """
    warnings.warn(
        f"{deprecated} is deprecated and will be removed in a future release. Use {use}.",
        FutureWarning,
        skip_file_prefixes=_SKIPPED_PREFIXES,
    )


def _deprecated_function[**P, R](function: Callable[P, R], deprecated: str, *, use: str) -> Callable[P, R]:
    """Return a public ``function`` that warns that ``deprecated`` is deprecated; Harpy calls ``function`` itself.

    For a deprecated function that Harpy also calls internally, so that Harpy
    never triggers the warning::

        def _plot(sdata, ax, ...):
            ...

        plot = _deprecated_function(_plot, "harpy.pl.plot", use="harpy.pl.plot_sdata")

    The public function keeps the docstring and signature of ``function``, and
    takes the last part of ``deprecated`` as its name.
    """

    @functools.wraps(function)
    def deprecated_function(*args: P.args, **kwargs: P.kwargs) -> R:
        _warn_deprecated(deprecated, use=use)
        return function(*args, **kwargs)

    deprecated_function.__name__ = deprecated_function.__qualname__ = deprecated.rsplit(".", 1)[-1]
    return deprecated_function


def _renamed_functions(module_name: str, *, namespace: str, renamed: Mapping[str, str]) -> Callable[[str], object]:
    """Return a module ``__getattr__`` that resolves each old name of a renamed function to the function.

    ``renamed`` maps each old name to the new one, both public in
    ``namespace``, for example ``{"metric_histogram": "table_histogram"}`` in
    ``"harpy.qc"``. Each access warns, whether through the package or the
    module; Python's default filter shows the warning once per calling location.
    """
    module = sys.modules[module_name]

    def __getattr__(name: str) -> object:
        if name in renamed:
            new_name = renamed[name]
            _warn_deprecated(f"{namespace}.{name}", use=f"{namespace}.{new_name}, which takes the same arguments")
            return getattr(module, new_name)
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")

    return __getattr__
