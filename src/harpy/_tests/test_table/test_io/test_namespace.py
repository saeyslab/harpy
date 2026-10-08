"""harpy.tb.io holds the table I/O functions; harpy.tb.add_table remains, deprecated."""

import pytest

import harpy as hp
import harpy.table._deprecated_add_table as deprecated_add_table
from harpy.utils._keys import _INSTANCE_KEY, _REGION_KEY

_TABLE_IO = (
    "read_table",
    "read_table_components",
    "write_table",
    "write_table_updates",
    "write_table_components",
    "write_table_components_by_region",
    "delete_table_components",
    "add_table",
    "add_table_updates",
    "add_table_components",
    "add_table_components_by_region",
    "remove_table_components",
)


def test_the_table_io_functions_live_in_harpy_tb_io():
    assert sorted(hp.tb.io.__all__) == sorted(_TABLE_IO)
    assert all(callable(getattr(hp.tb.io, name)) for name in _TABLE_IO)
    # Only the released add_table stays on harpy.tb, deprecated; the others moved without aliases.
    assert [name for name in _TABLE_IO if hasattr(hp.tb, name)] == ["add_table"]


def test_the_deprecated_add_table_warns_and_calls_harpy_tb_io_add_table(monkeypatch):
    calls = []

    def record(*args, **kwargs):
        calls.append((args, kwargs))
        return "sdata"

    monkeypatch.setattr(deprecated_add_table, "_add_table", record)
    with pytest.warns(FutureWarning, match=r"Use harpy\.tb\.io\.add_table"):
        result = hp.tb.add_table("sdata", "adata", "out", ["cells"], overwrite=True)
    assert result == "sdata"
    assert calls == [
        (
            ("sdata", "adata", "out", ["cells"]),
            {"instance_key": _INSTANCE_KEY, "region_key": _REGION_KEY, "overwrite": True},
        )
    ]
