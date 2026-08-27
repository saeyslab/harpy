from pathlib import Path

import pytest

from harpy.io._macsima import _validate_unique_channel_names


def test_validate_unique_channel_names() -> None:
    _validate_unique_channel_names(
        ["01_S_CD3_003_CD3", "02_S_CD8_003_CD8"],
        [Path("roi/C-001_CD3.tif"), Path("roi/C-002_CD8.tif")],
    )


def test_validate_unique_channel_names_raises_with_source_paths() -> None:
    duplicate_name = "01_S_CD3_003_CD3"
    first_path = Path("roi/C-001_CD3.tif")
    duplicate_path = Path("roi/copy/C-001_CD3.tif")

    with pytest.raises(ValueError) as exc_info:
        _validate_unique_channel_names(
            [duplicate_name, "02_S_CD8_003_CD8", duplicate_name],
            [first_path, Path("roi/C-002_CD8.tif"), duplicate_path],
        )

    message = str(exc_info.value)
    assert "Duplicate composite MACSima channel names" in message
    assert duplicate_name in message
    assert str(first_path) in message
    assert str(duplicate_path) in message
    assert "No channels were discarded" in message
