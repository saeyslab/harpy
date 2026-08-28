from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
from spatialdata import SpatialData, read_zarr
from spatialdata.transformations import get_transformation

from harpy import __version__
from harpy.image._image import get_dataarray
from harpy.io import CosmxSample, cosmx
from harpy.io._cosmx import _reader


@pytest.fixture
def ingestible_cosmx_path(decoded_cosmx_path: Path) -> Path:
    """Populate positioned synthetic FOVs with minimal valid transcripts."""
    preview = _reader._preview_cosmx(_reader._discover_cosmx(decoded_cosmx_path))
    for fov in preview.included_fovs:
        path = preview.manifest.fovs_by_id[fov].transcripts
        assert path is not None
        pd.DataFrame(
            {
                "V1": [fov],
                "CellComp": ["Nuclear"],
                "codeclass": ["Endogenous"],
                "target": [f"Gene{fov}"],
                "x": [1.0],
                "y": [2.0],
                "z": [0],
            }
        ).to_csv(path, index=False)
    return decoded_cosmx_path


def test_cosmx_reads_one_sample_with_sample_scoped_names_and_metadata(
    ingestible_cosmx_path: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "cosmx.zarr"

    sdata = cosmx(
        {"sample_a": CosmxSample(path=ingestible_cosmx_path)},
        output,
        image_chunks=(1, 4, 4),
        labels_chunks=(4, 4),
        transcript_blocksize=64,
    )

    assert Path(sdata.path) == output
    assert set(sdata.images) == {"sample_a_morphology_image_mosaic_1", "sample_a_morphology_image_mosaic_2"}
    assert set(sdata.labels) == {
        "sample_a_instance_labels_mosaic_1",
        "sample_a_instance_labels_mosaic_2",
        "sample_a_compartment_labels_mosaic_1",
        "sample_a_compartment_labels_mosaic_2",
    }
    assert set(sdata.points) == {"sample_a_transcripts_mosaic_1", "sample_a_transcripts_mosaic_2"}
    assert get_dataarray(sdata, "sample_a_instance_labels_mosaic_1").dtype == np.dtype(np.uint32)
    assert sum(len(points.compute()) for points in sdata.points.values()) == 3

    expected = {"sample_a_global_1", "sample_a_global_1_micron"}
    assert set(get_transformation(sdata.images["sample_a_morphology_image_mosaic_1"], get_all=True)) == expected
    assert set(get_transformation(sdata.labels["sample_a_instance_labels_mosaic_1"], get_all=True)) == expected
    assert set(get_transformation(sdata.points["sample_a_transcripts_mosaic_1"], get_all=True)) == expected

    metadata = sdata.attrs["harpy"]
    assert metadata["provenance"] == {"reader": "cosmx", "reader_version": __version__}
    image_metadata = metadata["images"]["sample_a_morphology_image_mosaic_1"]
    assert image_metadata["sample_id"] == "sample_a"
    assert image_metadata["fovs"] == [1, 2]
    assert image_metadata["mosaic"] == {"mode": "spatial_groups", "adjacency_tolerance_px": 1}
    panel_names = set(metadata["feature_panels"])
    assert len(panel_names) == 1
    panel_name = panel_names.pop()
    assert panel_name.startswith("feature_panel_")
    assert {record["feature_panel"] for record in metadata["points"].values()} == {panel_name}
    assert not _generated_siblings(output)


def test_cosmx_writes_two_samples_with_independent_configuration(
    ingestible_cosmx_path: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "multi.zarr"

    sdata = cosmx(
        {
            "sample_a": CosmxSample(
                path=ingestible_cosmx_path,
                fovs=[1, 2],
                channels=["U"],
                adjacency_tolerance_px=0,
                coordinate_system="pixels",
                flip_x=False,
            ),
            "sample_b": CosmxSample(
                path=ingestible_cosmx_path,
                fovs=[3],
                channels=["B"],
                mosaic_mode="single",
                adjacency_tolerance_px=99,
                flip_y=True,
            ),
        },
        output,
        instance_labels=False,
        compartment_labels=False,
        points=False,
        image_chunks=(1, 4, 4),
    )

    assert set(sdata.images) == {
        "sample_a_morphology_image_mosaic_1",
        "sample_b_morphology_image_mosaic_1",
    }
    assert get_dataarray(sdata, "sample_a_morphology_image_mosaic_1").coords["c"].values.tolist() == ["DNA"]
    assert get_dataarray(sdata, "sample_b_morphology_image_mosaic_1").coords["c"].values.tolist() == ["Histone"]
    assert set(get_transformation(sdata.images["sample_a_morphology_image_mosaic_1"], get_all=True)) == {
        "sample_a_pixels_1",
        "sample_a_pixels_1_micron",
    }
    assert set(get_transformation(sdata.images["sample_b_morphology_image_mosaic_1"], get_all=True)) == {
        "sample_b_global_1",
        "sample_b_global_1_micron",
    }
    metadata = sdata.attrs["harpy"]["images"]
    assert metadata["sample_a_morphology_image_mosaic_1"]["orientation"] == {"flip_x": False, "flip_y": False}
    assert metadata["sample_a_morphology_image_mosaic_1"]["mosaic"] == {
        "mode": "spatial_groups",
        "adjacency_tolerance_px": 0,
    }
    assert metadata["sample_b_morphology_image_mosaic_1"]["orientation"] == {"flip_x": True, "flip_y": True}
    assert metadata["sample_b_morphology_image_mosaic_1"]["mosaic"] == {
        "mode": "single",
        "adjacency_tolerance_px": None,
    }


def test_cosmx_deduplicates_identical_feature_panels_across_samples(
    ingestible_cosmx_path: Path,
    tmp_path: Path,
) -> None:
    sdata = cosmx(
        {
            "sample_a": CosmxSample(path=ingestible_cosmx_path, fovs=[1]),
            "sample_b": CosmxSample(path=ingestible_cosmx_path, fovs=[2]),
        },
        tmp_path / "panels.zarr",
        images=False,
        instance_labels=False,
        compartment_labels=False,
        points=True,
        transcript_blocksize=64,
    )

    panels = sdata.attrs["harpy"]["feature_panels"]
    references = {record["feature_panel"] for record in sdata.attrs["harpy"]["points"].values()}
    assert len(panels) == 1
    assert references == set(panels)


def test_cosmx_keeps_different_feature_panels_separate(
    ingestible_cosmx_path: Path,
    tmp_path: Path,
) -> None:
    second_path = tmp_path / "second_sample"
    shutil.copytree(ingestible_cosmx_path, second_path)
    plex = second_path / "plex-analysis.txt"
    plex.write_text(plex.read_text().replace("SystemControl1,SystemControl", "SystemControl2,SystemControl"))

    sdata = cosmx(
        {
            "sample_a": CosmxSample(path=ingestible_cosmx_path, fovs=[1]),
            "sample_b": CosmxSample(path=second_path, fovs=[1]),
        },
        tmp_path / "different_panels.zarr",
        images=False,
        instance_labels=False,
        compartment_labels=False,
        points=True,
        transcript_blocksize=64,
    )

    panels = sdata.attrs["harpy"]["feature_panels"]
    references = {record["feature_panel"] for record in sdata.attrs["harpy"]["points"].values()}
    assert len(panels) == 2
    assert references == set(panels)


def test_cosmx_intersects_fovs_only_across_enabled_modalities(
    decoded_cosmx_path: Path,
    tmp_path: Path,
) -> None:
    (decoded_cosmx_path / "CellStatsDir" / "FOV00002" / "CellLabels_F00002.tif").unlink()

    morphology_only = cosmx(
        {"sample": CosmxSample(path=decoded_cosmx_path)},
        tmp_path / "morphology.zarr",
        instance_labels=False,
        compartment_labels=False,
        points=False,
        image_chunks=(1, 4, 4),
    )
    represented = {fov for record in morphology_only.attrs["harpy"]["images"].values() for fov in record["fovs"]}
    assert represented == {1, 2, 3}

    with_instance_labels = cosmx(
        {"sample": CosmxSample(path=decoded_cosmx_path)},
        tmp_path / "labels.zarr",
        compartment_labels=False,
        points=False,
        image_chunks=(1, 4, 4),
        labels_chunks=(4, 4),
    )
    represented = {fov for record in with_instance_labels.attrs["harpy"]["images"].values() for fov in record["fovs"]}
    assert represented == {1, 3}


@pytest.mark.parametrize(
    ("inconsistency", "error_message"),
    [
        ("channel_order", "Contradictory morphology metadata for ChannelOrder"),
        ("dtype", "Contradictory morphology raster metadata"),
    ],
)
def test_cosmx_transcript_only_ignores_morphology_image_inconsistencies(
    ingestible_cosmx_path: Path,
    tmp_path: Path,
    inconsistency: str,
    error_message: str,
) -> None:
    morphology = next((ingestible_cosmx_path / "CellStatsDir" / "Morphology2D").glob("*F00002.TIF"))
    with tifffile.TiffFile(morphology) as tif:
        metadata = json.loads(tif.pages[0].description)
        data = tif.asarray()
    if inconsistency == "channel_order":
        metadata["ChannelOrder"] = "UGYRB"
    else:
        data = data.astype(np.uint8)
    tifffile.imwrite(
        morphology,
        data,
        description=json.dumps(metadata),
        metadata=None,
        photometric="minisblack",
    )

    with pytest.raises(ValueError, match=error_message):
        _reader._discover_cosmx(ingestible_cosmx_path)

    sdata = cosmx(
        {"sample": CosmxSample(path=ingestible_cosmx_path)},
        tmp_path / f"transcripts_{inconsistency}.zarr",
        images=False,
        instance_labels=False,
        compartment_labels=False,
        points=True,
        transcript_blocksize=64,
    )

    assert set(sdata.points) == {
        "sample_transcripts_mosaic_1",
        "sample_transcripts_mosaic_2",
    }


def test_cosmx_rejects_configuration_errors_before_staging(decoded_cosmx_path: Path, tmp_path: Path) -> None:
    output = tmp_path / "invalid.zarr"

    with pytest.raises(ValueError, match="non-empty mapping"):
        cosmx({}, output)
    with pytest.raises(ValueError, match="sample identifier must match"):
        cosmx({"invalid-id": CosmxSample(path=decoded_cosmx_path)}, output, points=False)
    with pytest.raises(ValueError, match="planned by both"):
        cosmx(
            {
                "Sample": CosmxSample(path=decoded_cosmx_path),
                "sample": CosmxSample(path=decoded_cosmx_path),
            },
            output,
            instance_labels=False,
            compartment_labels=False,
            points=False,
        )
    with pytest.raises(ValueError, match="at least one enabled modality"):
        cosmx(
            {"sample": CosmxSample(path=decoded_cosmx_path)},
            output,
            images=False,
            instance_labels=False,
            compartment_labels=False,
            points=False,
        )
    with pytest.raises(ValueError, match="planned by both"):
        cosmx(
            {"sample": CosmxSample(path=decoded_cosmx_path)},
            output,
            output_image_name="shared",
            output_instance_labels_name="shared",
            compartment_labels=False,
            points=False,
        )

    assert not output.exists()
    assert not _generated_siblings(output)


def test_cosmx_failure_while_writing_later_sample_preserves_existing_store(
    decoded_cosmx_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "existing.zarr"
    cosmx(
        {"original": CosmxSample(path=decoded_cosmx_path)},
        output,
        instance_labels=False,
        compartment_labels=False,
        points=False,
        image_chunks=(1, 4, 4),
    )
    before = _store_snapshot(output)
    original = _reader._add_morphology_images
    calls = 0

    def fail_second(*args: object, **kwargs: object) -> SpatialData:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("planned second-sample failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(_reader, "_add_morphology_images", fail_second)
    with pytest.raises(RuntimeError, match="planned second-sample failure"):
        cosmx(
            {
                "sample_a": CosmxSample(path=decoded_cosmx_path),
                "sample_b": CosmxSample(path=decoded_cosmx_path),
            },
            output,
            instance_labels=False,
            compartment_labels=False,
            points=False,
            image_chunks=(1, 4, 4),
            overwrite=True,
        )

    assert _store_snapshot(output) == before
    assert not _generated_siblings(output)


def test_cosmx_rejects_source_output_alias_and_unrecognized_overwrite(
    decoded_cosmx_path: Path,
    tmp_path: Path,
) -> None:
    nested_output = decoded_cosmx_path / "nested-output.zarr"
    with pytest.raises(ValueError, match="must not equal or contain one another"):
        cosmx({"sample": CosmxSample(path=decoded_cosmx_path)}, nested_output, points=False)

    output = tmp_path / "unrecognized.zarr"
    SpatialData().write(output)
    with pytest.raises(ValueError, match="not created by the CosMx reader"):
        cosmx({"sample": CosmxSample(path=decoded_cosmx_path)}, output, points=False, overwrite=True)
    assert read_zarr(output).attrs.get("harpy") is None


def test_cosmx_rejects_existing_output_before_source_discovery(
    decoded_cosmx_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_discovery(*args: object, **kwargs: object) -> None:
        raise AssertionError("Existing-output preflight reached source discovery.")

    monkeypatch.setattr(_reader, "_discover_cosmx", fail_discovery)

    existing = tmp_path / "existing.zarr"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        cosmx({"sample": CosmxSample(path=decoded_cosmx_path)}, existing, points=False)

    unrecognized = tmp_path / "unrecognized.zarr"
    SpatialData().write(unrecognized)
    with pytest.raises(ValueError, match="not created by the CosMx reader"):
        cosmx(
            {"sample": CosmxSample(path=decoded_cosmx_path)},
            unrecognized,
            points=False,
            overwrite=True,
        )


def _generated_siblings(output: Path) -> list[Path]:
    return list(output.parent.glob(f".{output.name}.cosmx-*"))


def _store_snapshot(path: Path) -> dict[Path, bytes]:
    return {file.relative_to(path): file.read_bytes() for file in path.rglob("*") if file.is_file()}
