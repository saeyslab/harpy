import inspect
import warnings

import pytest

from harpy.utils._deprecation import _deprecated_function, _warn_deprecated


def test_message_follows_the_template_and_names_the_caller():
    with pytest.warns(FutureWarning) as record:
        _warn_deprecated("The old parameter of harpy.tb.function", use="new")
    assert [str(warning.message) for warning in record] == [
        "The old parameter of harpy.tb.function is deprecated and will be removed in a future release. Use new."
    ]
    # The caller outside Harpy's code is this test, as the tests are not skipped.
    assert [warning.filename for warning in record] == [__file__]


def _implementation(value: int, *, scale: int = 2) -> int:
    """Multiply ``value`` by ``scale``."""
    return value * scale


def test_deprecated_function_warns_for_its_callers_and_leaves_the_private_function_silent():
    public = _deprecated_function(_implementation, "harpy.utils.old_function", use="harpy.utils.new_function")
    with pytest.warns(FutureWarning, match=r"harpy\.utils\.old_function is deprecated") as record:
        assert public(3, scale=4) == 12
    assert [warning.filename for warning in record] == [__file__]
    assert public.__name__ == "old_function"
    assert public.__doc__ == _implementation.__doc__
    assert inspect.signature(public) == inspect.signature(_implementation)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _implementation(3) == 6
