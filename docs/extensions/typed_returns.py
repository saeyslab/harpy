# code from https://github.com/scverse/spatialdata/blob/main/docs/extensions/typed_returns.py
# with some minor adjustment
import re

from sphinx.application import Sphinx
from sphinx.ext.napoleon import NumpyDocstring


def _process_return(lines):
    for line in lines:
        m = re.fullmatch(r"(?P<param>\w+)\s+:\s+(?P<type>[\w.]+)", line)
        if m:
            # Once this is in scanpydoc, we can use the fancy hover stuff
            yield f"-{m['param']} (:class:`~{m['type']}`)"
        else:
            yield line


def _parse_returns_section(self, section):
    lines = self._dedent(self._consume_to_next_section())
    # sphinx_autodoc_typehints puts a ":" line under the Returns header for
    # napoleon's own parser, which this one replaces.
    if lines and lines[0] == ":":
        lines = self._dedent(lines[1:])
    # A type with its description indented below it: keep the description, as the
    # return annotation shows the type. Kept as is, the indented description would
    # follow the type's line inside the field, which docutils rejects.
    if (
        len(lines) > 1
        and lines[0]
        and lines[1][:1].isspace()
        and all(not line or line[:1].isspace() for line in lines[1:])
    ):
        lines = self._dedent(lines[1:])
    lines_raw = list(_process_return(lines))
    lines = self._format_block(":returns: ", lines_raw)
    if lines and lines[-1]:
        lines.append("")
    return lines


def setup(app: Sphinx):
    """Set app."""
    NumpyDocstring._parse_returns_section = _parse_returns_section
