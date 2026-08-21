from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

CHANNEL_ORDER = "BGYRU"
TILE_SIZE = 8


@pytest.fixture
def decoded_cosmx_path(tmp_path: Path) -> Path:
    root = tmp_path / "transfer" / "DecodedFiles" / "sample" / "20240101_100000_S2"
    morphology_dir = root / "CellStatsDir" / "Morphology2D"
    morphology_dir.mkdir(parents=True)
    analysis_dir = root / "AnalysisResults" / "analysis"

    positions = {
        1: (10.000, 20.000),
        2: (10.008, 20.000),
        3: (10.040, 20.040),
    }
    for fov, (x_mm, y_mm) in positions.items():
        write_morphology(morphology_dir, fov=fov, x_mm=x_mm, y_mm=y_mm)

    for fov in range(1, 5):
        fov_dir = root / "CellStatsDir" / f"FOV{fov:05d}"
        fov_dir.mkdir(parents=True)
        labels = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint16)
        labels[1:3, 1:3] = 1
        tifffile.imwrite(fov_dir / f"CellLabels_F{fov:05d}.tif", labels, metadata=None)
        tifffile.imwrite(
            fov_dir / f"CompartmentLabels_F{fov:05d}.tif",
            (labels > 0).astype(np.uint8),
            metadata=None,
        )

        # Deliberately malformed deferred inputs: Slice 1 must ignore them.
        (fov_dir / f"CellRegionLabels_F{fov:05d}.tif").write_bytes(b"not-a-tiff")
        (fov_dir / f"CellBoundaries_F{fov:05d}.csv").write_bytes(b"not,csv\n")
        (fov_dir / f"Run_test_Cell_Stats_F{fov:05d}.csv").write_bytes(b"not,csv\n")
        (fov_dir / f"RegStats_F{fov:05d}.csv").write_bytes(b"not,csv\n")

        transcript_dir = analysis_dir / f"FOV{fov:05d}"
        transcript_dir.mkdir(parents=True)
        transcript_path = (
            transcript_dir
            / f"Run_83f88b0c-fc17-418d-8cb4-4675e4f6c12a_FOV{fov:05d}__complete_code_cell_target_call_coord.csv"
        )
        transcript_path.write_bytes(b"discovery-must-not-read-this")

    (root / "plex-analysis.txt").write_bytes(b"ignored")
    return root


def write_morphology(
    directory: Path,
    *,
    fov: int,
    x_mm: float,
    y_mm: float,
    acquisition_timestamp: str = "20240101_120000",
) -> None:
    metadata = {
        "NFov": 4,
        "ChannelOrder": CHANNEL_ORDER,
        "X_mm": x_mm,
        "Y_mm": y_mm,
        "ImPixelSize_nm": 1000.0,
        "ImRows": TILE_SIZE,
        "ImCols": TILE_SIZE,
        "OrigTimeStamp": "20240101_100000_S2",
        "Slot": 2,
        "RunLabel": "test",
        "RunNumber": "run-id",
        "Fov": fov,
        "MorphologyKit": {
            "MorphologyReagents": [
                _reagent("B", "Histone", "Nuclei"),
                _reagent("Y", "rRNA", "Membrane"),
                _reagent("R", "GFAP", "Astrocytes"),
                _reagent("U", "DNA", "Nuclei"),
            ]
        },
    }
    data = np.zeros((len(CHANNEL_ORDER), TILE_SIZE, TILE_SIZE), dtype=np.uint16)
    tifffile.imwrite(
        directory / f"{acquisition_timestamp}_S2_C001_P01_N01_F{fov:05d}.TIF",
        data,
        description=json.dumps(metadata),
        metadata=None,
        photometric="minisblack",
    )


def _reagent(channel_id: str, target: str, biological_class: str) -> dict:
    return {
        "Fluorophore": {"ChannelId": channel_id, "Name": channel_id},
        "BiologicalTarget": target,
        "BiologicalClass": biological_class,
    }
