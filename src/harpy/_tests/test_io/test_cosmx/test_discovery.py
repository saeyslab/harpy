from __future__ import annotations

import builtins
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import tifffile

from harpy.io._cosmx import _discovery
from harpy.io._cosmx._discovery import _discover_cosmx, _is_decoded_cosmx, _resolve_decoded_cosmx_root

_CHANNEL_ORDER = "BGYRU"
_TILE_SIZE = 8


def test_discover_compact_manifest_without_opening_deferred_data(
    decoded_cosmx_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_open = Path.open
    original_builtin_open = builtins.open

    def is_forbidden(path: object) -> bool:
        name = str(path)
        return name.endswith("target_call_coord.csv") or "Cell_Stats" in name

    def guarded_open(path: Path, *args: object, **kwargs: object):
        if is_forbidden(path):
            raise AssertionError(f"Discovery opened deferred content: {path}")
        return original_open(path, *args, **kwargs)

    def guarded_builtin_open(path: object, *args: object, **kwargs: object):
        if is_forbidden(path):
            raise AssertionError(f"Discovery opened deferred content: {path}")
        return original_builtin_open(path, *args, **kwargs)

    def forbid_pixel_read(*args: object, **kwargs: object) -> None:
        raise AssertionError("Discovery read raster pixels.")

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(builtins, "open", guarded_builtin_open)
    monkeypatch.setattr(tifffile.TiffFile, "asarray", forbid_pixel_read)
    monkeypatch.setattr(tifffile.TiffPage, "asarray", forbid_pixel_read)
    manifest = _discover_cosmx(decoded_cosmx_path)

    assert _is_decoded_cosmx(decoded_cosmx_path)
    assert manifest.root == decoded_cosmx_path.resolve()
    assert manifest.fov_ids == (1, 2, 3, 4)
    assert manifest.available_fovs("morphology") == (1, 2, 3)
    assert manifest.available_fovs("instance_labels") == (1, 2, 3, 4)
    assert manifest.available_fovs("compartment_labels") == (1, 2, 3, 4)
    assert manifest.available_fovs("transcripts") == (1, 2, 3, 4)
    assert tuple(position.fov for position in manifest.positions) == (1, 2, 3)
    assert (manifest.positions_by_fov[2].x_px, manifest.positions_by_fov[2].y_px) == (8, 0)
    assert not hasattr(manifest.fovs[0], "cell_stats")
    assert not hasattr(manifest.fovs[0], "cell_boundaries")
    assert tuple(channel.channel_id for channel in manifest.run.channels) == tuple(_CHANNEL_ORDER)
    assert tuple(channel.name for channel in manifest.run.channels) == ("Histone", "G", "rRNA", "GFAP", "DNA")
    assert manifest.run.pixel_size_um == pytest.approx(1.0)
    assert manifest.run.tile_shape == (_TILE_SIZE, _TILE_SIZE)
    assert manifest.run.instance_labels_dtype == "uint16"
    assert manifest.run.compartment_labels_dtype == "uint8"
    assert manifest.feature_panel is not None
    assert manifest.feature_panel.feature_column == "gene"
    assert manifest.feature_panel.class_column == "code_class"
    assert manifest.feature_panel.categories == ("Endogenous", "Negative", "SystemControl")
    assert manifest.feature_panel.targets_by_class["Negative"] == ("Negative01",)


def test_resolve_decoded_cosmx_root(decoded_cosmx_path: Path) -> None:
    assert _resolve_decoded_cosmx_root(decoded_cosmx_path.parents[2]) == decoded_cosmx_path.resolve()


def test_discovery_rejects_duplicate_morphology(decoded_cosmx_path: Path) -> None:
    morphology_dir = decoded_cosmx_path / "CellStatsDir" / "Morphology2D"
    source = morphology_dir / "20240101_120000_S2_C001_P01_N01_F00001.TIF"
    duplicate = morphology_dir / "20240102_120000_S2_C001_P01_N01_F00001.TIF"
    shutil.copyfile(source, duplicate)

    with pytest.raises(ValueError, match="Duplicate morphology files for FOV 1"):
        _discover_cosmx(decoded_cosmx_path)


def test_discovery_reads_one_plex_once(
    decoded_cosmx_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Path] = []
    original = _discovery._read_feature_panel

    def record(path: Path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(_discovery, "_read_feature_panel", record)

    _discover_cosmx(decoded_cosmx_path)

    assert calls == [decoded_cosmx_path / "plex-analysis.txt"]


def test_discovery_allows_missing_plex(decoded_cosmx_path: Path) -> None:
    (decoded_cosmx_path / "plex-analysis.txt").unlink()

    manifest = _discover_cosmx(decoded_cosmx_path)

    assert manifest.feature_panel is None


def test_discovery_rejects_multiple_plex_files(decoded_cosmx_path: Path) -> None:
    shutil.copyfile(decoded_cosmx_path / "plex-analysis.txt", decoded_cosmx_path / "plex-second.txt")

    with pytest.raises(ValueError, match="multiple CosMx plex files"):
        _discover_cosmx(decoded_cosmx_path)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("DisplayName,CodeClass\nGeneA,Endogenous\nGeneA,Endogenous\n", "occurs more than once"),
        ("DisplayName,CodeClass\nGeneA,\n", "empty or untrimmed CodeClass"),
        ("DisplayName,ProbeID\nGeneA,NA\n", "missing required columns.*CodeClass"),
    ],
)
def test_discovery_rejects_invalid_plex(
    decoded_cosmx_path: Path,
    content: str,
    message: str,
) -> None:
    (decoded_cosmx_path / "plex-analysis.txt").write_text(content)

    with pytest.raises(ValueError, match=message):
        _discover_cosmx(decoded_cosmx_path)


def test_discovery_allows_varying_non_stitching_metadata(decoded_cosmx_path: Path) -> None:
    morphology = decoded_cosmx_path / "CellStatsDir" / "Morphology2D" / "20240101_120000_S2_C001_P01_N01_F00002.TIF"
    _rewrite_morphology_metadata(morphology, OrigTimeStamp="another-acquisition", Slot=99)

    manifest = _discover_cosmx(decoded_cosmx_path)

    assert manifest.available_fovs("morphology") == (1, 2, 3)


def test_discovery_rejects_incompatible_pixel_size(decoded_cosmx_path: Path) -> None:
    morphology = decoded_cosmx_path / "CellStatsDir" / "Morphology2D" / "20240101_120000_S2_C001_P01_N01_F00002.TIF"
    _rewrite_morphology_metadata(morphology, ImPixelSize_nm=500.0)

    with pytest.raises(ValueError, match="Contradictory morphology metadata for ImPixelSize_nm"):
        _discover_cosmx(decoded_cosmx_path)


def test_discovery_rejects_fov_directory_file_mismatch(decoded_cosmx_path: Path) -> None:
    fov_dir = decoded_cosmx_path / "CellStatsDir" / "FOV00001"
    labels = np.zeros((_TILE_SIZE, _TILE_SIZE), dtype=np.uint16)
    tifffile.imwrite(fov_dir / "CellLabels_F00002.tif", labels, metadata=None)

    with pytest.raises(ValueError, match="FOV directory/file mismatch"):
        _discover_cosmx(decoded_cosmx_path)


def test_discovery_rejects_signed_compartment_labels(decoded_cosmx_path: Path) -> None:
    labels_path = decoded_cosmx_path / "CellStatsDir" / "FOV00001" / "CompartmentLabels_F00001.tif"
    tifffile.imwrite(labels_path, np.zeros((_TILE_SIZE, _TILE_SIZE), dtype=np.int8), metadata=None)

    with pytest.raises(ValueError, match="CosMx compartment labels must use an unsigned integer dtype"):
        _discover_cosmx(decoded_cosmx_path)


def _rewrite_morphology_metadata(path: Path, **updates: object) -> None:
    with tifffile.TiffFile(path) as tif:
        metadata = json.loads(tif.pages[0].description)
        data = tif.asarray()
    metadata.update(updates)
    tifffile.imwrite(path, data, description=json.dumps(metadata), metadata=None, photometric="minisblack")
