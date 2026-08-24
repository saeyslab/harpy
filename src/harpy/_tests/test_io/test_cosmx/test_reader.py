from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from spatialdata import SpatialData, read_zarr
from spatialdata.transformations import get_transformation

from harpy.image._image import get_dataarray
from harpy.io import cosmx
from harpy.io._cosmx import _reader


@pytest.fixture
def ingestible_cosmx_path(decoded_cosmx_path: Path) -> Path:
    """Populate the selected synthetic FOVs with minimal valid transcripts."""
    preview = _reader._preview_cosmx(_reader._discover_cosmx(decoded_cosmx_path))
    for fov in preview.included_fovs:
        path = preview.manifest.fovs_by_id[fov].transcripts
        assert path is not None
        pd.DataFrame(
            {
                "V1": [fov],
                "CellComp": ["Nuclear"],
                "codeclass": ["Call"],
                "target": [f"Gene{fov}"],
                "x": [1.0],
                "y": [2.0],
                "z": [0],
            }
        ).to_csv(path, index=False)
    return decoded_cosmx_path


def test_cosmx_reads_all_modalities_into_matching_mosaics(
    ingestible_cosmx_path: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "cosmx.zarr"

    sdata = cosmx(
        ingestible_cosmx_path,
        output,
        image_chunks=(1, 4, 4),
        labels_chunks=(4, 4),
        transcript_blocksize=64,
    )

    assert Path(sdata.path) == output
    assert set(sdata.images) == {"morphology_image_mosaic_1", "morphology_image_mosaic_2"}
    assert set(sdata.labels) == {
        "instance_labels_mosaic_1",
        "instance_labels_mosaic_2",
        "compartment_labels_mosaic_1",
        "compartment_labels_mosaic_2",
    }
    assert set(sdata.points) == {"transcripts_mosaic_1", "transcripts_mosaic_2"}
    assert get_dataarray(sdata, "instance_labels_mosaic_1").dtype == np.dtype(np.uint32)
    assert get_dataarray(sdata, "compartment_labels_mosaic_1").dtype == np.dtype(np.uint8)
    assert sum(len(points.compute()) for points in sdata.points.values()) == 3

    for mosaic in (1, 2):
        expected = {f"global_{mosaic}", f"global_{mosaic}_micron"}
        assert set(get_transformation(sdata.images[f"morphology_image_mosaic_{mosaic}"], get_all=True)) == expected
        assert set(get_transformation(sdata.labels[f"instance_labels_mosaic_{mosaic}"], get_all=True)) == expected
        assert set(get_transformation(sdata.points[f"transcripts_mosaic_{mosaic}"], get_all=True)) == expected

    assert sdata.attrs["cosmx"]["selection"] == {
        "source_root": str(ingestible_cosmx_path.resolve()),
        "requested_fovs": [1, 2, 3, 4],
        "included_fovs": [1, 2, 3],
        "excluded_fovs": [4],
        "unpositioned_fovs": [4],
        "mosaic_mode": "spatial_groups",
        "adjacency_tolerance_px": 1,
    }
    assert not _generated_siblings(output)


def test_cosmx_applies_fov_and_channel_selection(
    ingestible_cosmx_path: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "subset.zarr"

    sdata = cosmx(
        ingestible_cosmx_path,
        output,
        fovs=(2, 1, 2),
        channels=("U",),
        adjacency_tolerance_px=0,
        morphology=True,
        instance_labels=False,
        compartment_labels=False,
        transcripts=False,
        image_chunks=(1, 4, 4),
    )

    assert set(sdata.images) == {"morphology_image_mosaic_1"}
    image = get_dataarray(sdata, "morphology_image_mosaic_1")
    assert image.coords["c"].values.tolist() == ["DNA"]
    assert sdata.attrs["cosmx"]["selection"]["requested_fovs"] == [1, 2]
    assert sdata.attrs["cosmx"]["selection"]["included_fovs"] == [1, 2]
    assert sdata.attrs["cosmx"]["selection"]["excluded_fovs"] == [3, 4]
    assert sdata.attrs["cosmx"]["selection"]["mosaic_mode"] == "spatial_groups"
    assert sdata.attrs["cosmx"]["selection"]["adjacency_tolerance_px"] == 0


@pytest.mark.parametrize(
    "helper_name",
    [
        pytest.param("_add_morphology_images", id="morphology"),
        pytest.param("_add_instance_labels", id="instance-labels-after-morphology"),
        pytest.param("_add_compartment_labels", id="compartment-labels-after-instance-labels"),
        pytest.param("_add_transcript_points", id="transcripts-after-rasters"),
    ],
)
def test_cosmx_stage_failure_leaves_new_output_absent(
    ingestible_cosmx_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    helper_name: str,
) -> None:
    output = tmp_path / "failed.zarr"

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("planned stage failure")

    monkeypatch.setattr(_reader, helper_name, fail)
    with pytest.raises(RuntimeError, match="planned stage failure"):
        _reader.cosmx(
            ingestible_cosmx_path,
            output,
            image_chunks=(1, 4, 4),
            labels_chunks=(4, 4),
            transcript_blocksize=64,
        )

    assert not output.exists()
    assert not _generated_siblings(output)


def test_cosmx_failed_overwrite_preserves_existing_store(
    ingestible_cosmx_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "existing.zarr"
    cosmx(
        ingestible_cosmx_path,
        output,
        morphology=True,
        instance_labels=False,
        compartment_labels=False,
        transcripts=False,
        image_chunks=(1, 4, 4),
    )
    before = _store_snapshot(output)

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("planned replacement failure")

    monkeypatch.setattr(_reader, "_add_instance_labels", fail)
    with pytest.raises(RuntimeError, match="planned replacement failure"):
        _reader.cosmx(
            ingestible_cosmx_path,
            output,
            morphology=False,
            instance_labels=True,
            compartment_labels=False,
            transcripts=False,
            labels_chunks=(4, 4),
            overwrite=True,
        )

    assert _store_snapshot(output) == before
    assert set(read_zarr(output).images) == {"morphology_image_mosaic_1", "morphology_image_mosaic_2"}
    assert not _generated_siblings(output)


def test_cosmx_successful_overwrite_replaces_complete_store(
    ingestible_cosmx_path: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "replace.zarr"
    cosmx(
        ingestible_cosmx_path,
        output,
        morphology=True,
        instance_labels=False,
        compartment_labels=False,
        transcripts=False,
        image_chunks=(1, 4, 4),
    )

    sdata = cosmx(
        ingestible_cosmx_path,
        output,
        morphology=False,
        instance_labels=True,
        compartment_labels=False,
        transcripts=False,
        labels_chunks=(4, 4),
        overwrite=True,
    )

    assert not sdata.images
    assert set(sdata.labels) == {"instance_labels_mosaic_1", "instance_labels_mosaic_2"}
    assert not _generated_siblings(output)


def test_cosmx_rejects_public_orchestration_errors_before_staging(
    decoded_cosmx_path: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "invalid.zarr"

    with pytest.raises(ValueError, match="at least one enabled modality"):
        cosmx(
            decoded_cosmx_path,
            output,
            morphology=False,
            instance_labels=False,
            compartment_labels=False,
            transcripts=False,
        )
    with pytest.raises(ValueError, match="no common positioned FOVs"):
        cosmx(decoded_cosmx_path, output, fovs=(4,), morphology=True)
    with pytest.raises(ValueError, match="planned by both"):
        cosmx(
            decoded_cosmx_path,
            output,
            output_image_name="shared",
            output_instance_labels_name="shared",
            compartment_labels=False,
            transcripts=False,
        )

    assert not output.exists()
    assert not _generated_siblings(output)


def test_cosmx_rejects_source_output_alias_and_existing_output_without_overwrite(
    decoded_cosmx_path: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="must not equal or contain one another"):
        cosmx(decoded_cosmx_path, decoded_cosmx_path, morphology=True)

    output = tmp_path / "existing.zarr"
    SpatialData().write(output)
    with pytest.raises(FileExistsError, match="already exists"):
        cosmx(decoded_cosmx_path, output, morphology=True)


def test_cosmx_refuses_to_overwrite_unrecognized_spatialdata(
    decoded_cosmx_path: Path,
    tmp_path: Path,
) -> None:
    output = tmp_path / "unrecognized.zarr"
    SpatialData().write(output)

    with pytest.raises(ValueError, match="not created by the CosMx reader"):
        cosmx(decoded_cosmx_path, output, morphology=True, overwrite=True)

    assert read_zarr(output).attrs.get("cosmx") is None


def _generated_siblings(output: Path) -> list[Path]:
    return list(output.parent.glob(f".{output.name}.cosmx-*"))


def _store_snapshot(path: Path) -> dict[Path, bytes]:
    return {file.relative_to(path): file.read_bytes() for file in path.rglob("*") if file.is_file()}
