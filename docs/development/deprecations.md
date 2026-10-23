# Deprecating functions and parameters

Every deprecation in Harpy warns in the same way, through the helpers of
`harpy/utils/_deprecation.py`, so that users meet one kind of warning with one
kind of message.

## The warning

A `FutureWarning`, raised with `warnings.warn`. Python shows it by default; it
can be filtered or turned into an error, and tests check it with
`pytest.warns`. Not a `DeprecationWarning`, which Python hides outside
`__main__`, nor a loguru message, which is text in a log only.

The helpers attribute the warning to the caller of the deprecated function: the
first frame outside Harpy's package and lazy_loader, through
`skip_file_prefixes`, at any call depth. Do not pass a `stacklevel`.

Messages and docstrings name no version. The release in which a deprecation
starts cannot be derived automatically, and the installed version would
misstate it; the removal is announced for "a future release".

## The message

```text
<what> is deprecated and will be removed in a future release. Use <instead>.
```

For example:

```text
harpy.tb.add_table is deprecated and will be removed in a future release. Use harpy.tb.io.add_table, which takes the same arguments.
The celltype_column parameter of harpy.tb.score_genes is deprecated and will be removed in a future release. Use key_added.
```

After "Use" comes the replacement or, for a function without one, what to do
instead, as for the scanpy wrappers of `harpy.tb`: scanpy or squidpy on the
table directly, then `harpy.tb.io.add_table_updates`.

## How to deprecate

- **A function:** call `_warn_deprecated("harpy.<namespace>.<name>", use=...)`
  first in its body.
- **A function that Harpy also calls:** move its body to a private function,
  which Harpy's callers call, and define the public one with
  `_deprecated_function`, so that Harpy itself never triggers the warning:

  ```python
  def _plot(sdata, ax, ...):
      ...

  plot = _deprecated_function(_plot, "harpy.pl.plot", use="harpy.pl.plot_sdata")
  ```

- **A renamed function:** in the module of the new function, define its
  `__getattr__` with `_renamed_functions`, and keep the old name in the
  package's `__init__.pyi`, imported from that module:

  ```python
  __getattr__ = _renamed_functions(__name__, namespace="harpy.qc", renamed={"metric_histogram": "table_histogram"})
  ```

- **A parameter:** call
  `_warn_deprecated("The <name> parameter of harpy.<namespace>.<function>", use=...)`
  when it is passed. If the parameter it replaces is passed too, raise a
  `ValueError`.

## The docstring

Sphinx's `.. deprecated::` requires a version, so a deprecated function's
docstring carries, after its summary, a warning with the wording of the
message:

```rst
.. warning::
   Deprecated, and will be removed in a future release. Use
   :func:`harpy.tb.io.add_table`, which takes the same arguments.
```

A deprecated parameter ends its description with "Deprecated, and will be
removed in a future release; use `key_added`."

## Tests

Check the warning with `pytest.warns(FutureWarning, match=...)`. Where the
attribution matters, as for a renamed function, check that the recorded
warning's `filename` is the test file.
