from __future__ import annotations

import shutil
from copy import deepcopy
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import pytest
from spatialdata import SpatialData, read_zarr
from spatialdata.transformations import Identity, get_transformation

from harpy import __version__
from harpy.image._image import add_image, get_dataarray
from harpy.io import CosmxSample, add_cosmx_samples, cosmx, validate_cosmx_store
from harpy.io._cosmx import _reader


def test_add_cosmx_samples_preserves_existing_sample_and_adds_independent_sample(
    decoded_cosmx_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "incremental.zarr"
    original = cosmx(
        {"sample_a": CosmxSample(path=decoded_cosmx_path, fovs=[1])},
        output,
        instance_labels=False,
        compartment_labels=False,
        points=False,
        image_chunks=(1, 4, 4),
    )
    original_name = "sample_a_morphology_image_mosaic_1"
    original_data = get_dataarray(original, original_name).compute().values.copy()
    original_metadata = deepcopy(original.attrs["harpy"]["images"][original_name])

    def fail_staging(*args: object, **kwargs: object) -> None:
        raise AssertionError("Incremental addition attempted whole-store publication.")

    monkeypatch.setattr(_reader, "_publish_staging_store", fail_staging)

    result = add_cosmx_samples(
        output,
        {"sample_b": CosmxSample(path=decoded_cosmx_path, fovs=[2], coordinate_system="pixels")},
        instance_labels=False,
        compartment_labels=False,
        points=False,
        image_chunks=(1, 4, 4),
    )

    assert Path(result.path) == output
    assert set(result.images) == {original_name, "sample_b_morphology_image_mosaic_1"}
    np.testing.assert_array_equal(get_dataarray(result, original_name).compute(), original_data)
    assert result.attrs["harpy"]["images"][original_name] == original_metadata
    assert set(get_transformation(result.images["sample_b_morphology_image_mosaic_1"], get_all=True)) == {
        "sample_b_pixels_1",
        "sample_b_pixels_1_micron",
    }
    assert result.attrs["harpy"]["provenance"] == {"reader": "cosmx", "reader_version": __version__}
    validate_cosmx_store(output)


def test_add_cosmx_samples_rejects_existing_sample_before_source_discovery(
    decoded_cosmx_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "duplicate.zarr"
    cosmx(
        {"sample": CosmxSample(path=decoded_cosmx_path, fovs=[1])},
        output,
        instance_labels=False,
        compartment_labels=False,
        points=False,
        image_chunks=(1, 4, 4),
    )

    def fail_discovery(*args: object, **kwargs: object) -> None:
        raise AssertionError("Duplicate sample preflight reached source discovery.")

    monkeypatch.setattr(_reader, "_discover_cosmx", fail_discovery)
    with pytest.raises(ValueError, match="sample identifiers already exist"):
        add_cosmx_samples(
            output,
            {"sample": CosmxSample(path=decoded_cosmx_path, fovs=[2])},
            instance_labels=False,
            compartment_labels=False,
            points=False,
        )


def test_add_cosmx_samples_checks_existing_point_contents_before_source_discovery(
    decoded_cosmx_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "point-content-preflight.zarr"
    cosmx(
        {"sample_a": CosmxSample(path=decoded_cosmx_path, fovs=[1])},
        output,
        instance_labels=False,
        compartment_labels=False,
        points=False,
        image_chunks=(1, 4, 4),
    )
    original_validate = _reader._validate_cosmx_sdata
    requested_content_checks = []

    def record_validation(sdata: SpatialData, *, check_point_contents: bool = False) -> frozenset[str]:
        requested_content_checks.append(check_point_contents)
        return original_validate(sdata, check_point_contents=False)

    def fail_discovery(*args: object, **kwargs: object) -> None:
        raise AssertionError("Incremental point-content preflight reached source discovery.")

    monkeypatch.setattr(_reader, "_validate_cosmx_sdata", record_validation)
    monkeypatch.setattr(_reader, "_discover_cosmx", fail_discovery)
    with pytest.raises(AssertionError, match="reached source discovery"):
        add_cosmx_samples(
            output,
            {"sample_b": CosmxSample(path=decoded_cosmx_path, fovs=[2])},
            instance_labels=False,
            compartment_labels=False,
            points=False,
        )

    assert requested_content_checks == [True]


@pytest.mark.parametrize("collision", ["element", "coordinate_system"])
def test_add_cosmx_samples_rejects_existing_namespace_collisions(
    decoded_cosmx_path: Path,
    tmp_path: Path,
    collision: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / f"{collision}.zarr"
    cosmx(
        {"sample_a": CosmxSample(path=decoded_cosmx_path, fovs=[1])},
        output,
        instance_labels=False,
        compartment_labels=False,
        points=False,
        image_chunks=(1, 4, 4),
    )
    sdata = read_zarr(output)
    unrelated_name = "sample_b_morphology_image_mosaic_1" if collision == "element" else "unrelated_image"
    coordinate_system = "unrelated" if collision == "element" else "sample_b_global_1"
    add_image(
        sdata,
        arr=da.zeros((1, 2, 2), chunks=(1, 2, 2), dtype=np.uint8),
        output_image_name=unrelated_name,
        dims=("c", "y", "x"),
        transformations={coordinate_system: Identity()},
    )

    def fail_writing(*args: object, **kwargs: object) -> None:
        raise AssertionError("Namespace preflight reached payload writing.")

    monkeypatch.setattr(_reader, "_write_cosmx_sample", fail_writing)

    with pytest.raises(ValueError, match="output element|output coordinate system"):
        add_cosmx_samples(
            output,
            {"sample_b": CosmxSample(path=decoded_cosmx_path, fovs=[1])},
            instance_labels=False,
            compartment_labels=False,
            points=False,
            image_chunks=(1, 4, 4),
        )

    reopened = read_zarr(output)
    assert set(reopened.images) == {"sample_a_morphology_image_mosaic_1", unrelated_name}
    assert "sample_b" not in {record["sample_id"] for record in reopened.attrs["harpy"]["images"].values()}


def test_add_cosmx_samples_reuses_identical_panels_and_separates_different_panels(
    decoded_cosmx_path: Path,
    tmp_path: Path,
) -> None:
    source = _write_valid_transcripts(decoded_cosmx_path)
    different_source = tmp_path / "different_source"
    shutil.copytree(source, different_source)
    plex = different_source / "plex-analysis.txt"
    plex.write_text(plex.read_text().replace("SystemControl1,SystemControl", "SystemControl2,SystemControl"))
    output = tmp_path / "panels.zarr"
    cosmx(
        {"sample_a": CosmxSample(path=source, fovs=[1])},
        output,
        images=False,
        instance_labels=False,
        compartment_labels=False,
        points_blocksize=64,
    )

    add_cosmx_samples(
        output,
        {"sample_b": CosmxSample(path=source, fovs=[2])},
        images=False,
        instance_labels=False,
        compartment_labels=False,
        points_blocksize=64,
    )
    after_reuse = read_zarr(output)
    assert len(after_reuse.attrs["harpy"]["feature_panels"]) == 1

    result = add_cosmx_samples(
        output,
        {"sample_c": CosmxSample(path=different_source, fovs=[3])},
        images=False,
        instance_labels=False,
        compartment_labels=False,
        points_blocksize=64,
    )
    panel_names = set(result.attrs["harpy"]["feature_panels"])
    assert len(panel_names) == 2
    assert {record["feature_panel"] for record in result.attrs["harpy"]["points"].values()} == panel_names


def test_add_cosmx_samples_later_element_failure_retains_preceding_commit_and_rejects_retry(
    decoded_cosmx_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "partial_sample.zarr"
    cosmx(
        {"sample_a": CosmxSample(path=decoded_cosmx_path, fovs=[1])},
        output,
        instance_labels=False,
        compartment_labels=False,
        points=False,
        image_chunks=(1, 4, 4),
    )

    def fail_labels(*args: object, **kwargs: object) -> None:
        raise RuntimeError("planned labels failure")

    monkeypatch.setattr(_reader, "__version__", "incremental-test-version")
    monkeypatch.setattr(_reader, "_add_instance_labels", fail_labels)
    with pytest.raises(RuntimeError, match="planned labels failure"):
        add_cosmx_samples(
            output,
            {"sample_b": CosmxSample(path=decoded_cosmx_path, fovs=[2])},
            compartment_labels=False,
            points=False,
            image_chunks=(1, 4, 4),
            labels_chunks=(4, 4),
        )

    partial = read_zarr(output)
    assert "sample_b_morphology_image_mosaic_1" in partial.images
    assert partial.attrs["harpy"]["images"]["sample_b_morphology_image_mosaic_1"]["sample_id"] == "sample_b"
    assert not any(name.startswith("sample_b_") for name in partial.labels)
    assert partial.attrs["harpy"]["provenance"]["reader_version"] == "incremental-test-version"
    validate_cosmx_store(output)

    with pytest.raises(ValueError, match="sample identifiers already exist"):
        add_cosmx_samples(
            output,
            {"sample_b": CosmxSample(path=decoded_cosmx_path, fovs=[2])},
            instance_labels=False,
            compartment_labels=False,
            points=False,
        )


def test_add_cosmx_samples_metadata_failure_removes_only_new_element_and_restores_attributes(
    decoded_cosmx_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "metadata_failure.zarr"
    cosmx(
        {"sample_a": CosmxSample(path=decoded_cosmx_path, fovs=[1])},
        output,
        instance_labels=False,
        compartment_labels=False,
        points=False,
        image_chunks=(1, 4, 4),
    )
    before = read_zarr(output)
    before_attrs = deepcopy(before.attrs)
    original_write_attrs = SpatialData.write_attrs
    failed = False

    def fail_first_incremental_metadata_write(self: SpatialData, *args: object, **kwargs: object) -> object:
        nonlocal failed
        if not failed and Path(self.path) == output:
            failed = True
            raise RuntimeError("planned metadata failure")
        return original_write_attrs(self, *args, **kwargs)

    monkeypatch.setattr(SpatialData, "write_attrs", fail_first_incremental_metadata_write)
    with pytest.raises(RuntimeError, match="planned metadata failure"):
        add_cosmx_samples(
            output,
            {"sample_b": CosmxSample(path=decoded_cosmx_path, fovs=[2])},
            instance_labels=False,
            compartment_labels=False,
            points=False,
            image_chunks=(1, 4, 4),
        )

    restored = read_zarr(output)
    assert set(restored.images) == {"sample_a_morphology_image_mosaic_1"}
    assert restored.attrs == before_attrs
    validate_cosmx_store(output)


def test_add_cosmx_samples_commits_new_panel_with_its_first_points_element(
    decoded_cosmx_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_valid_transcripts(decoded_cosmx_path)
    different_source = tmp_path / "different_panel_source"
    shutil.copytree(source, different_source)
    plex = different_source / "plex-analysis.txt"
    plex.write_text(plex.read_text().replace("SystemControl1,SystemControl", "SystemControl2,SystemControl"))
    output = tmp_path / "panel_metadata_failure.zarr"
    cosmx(
        {"sample_a": CosmxSample(path=source, fovs=[1])},
        output,
        images=False,
        instance_labels=False,
        compartment_labels=False,
        points_blocksize=64,
    )
    before = read_zarr(output)
    before_attrs = deepcopy(before.attrs)
    original_write_attrs = SpatialData.write_attrs
    failed = False

    def fail_first_incremental_metadata_write(self: SpatialData, *args: object, **kwargs: object) -> object:
        nonlocal failed
        if not failed and Path(self.path) == output:
            failed = True
            raise RuntimeError("planned panel metadata failure")
        return original_write_attrs(self, *args, **kwargs)

    monkeypatch.setattr(SpatialData, "write_attrs", fail_first_incremental_metadata_write)
    with pytest.raises(RuntimeError, match="planned panel metadata failure"):
        add_cosmx_samples(
            output,
            {"sample_b": CosmxSample(path=different_source, fovs=[2])},
            images=False,
            instance_labels=False,
            compartment_labels=False,
            points_blocksize=64,
        )

    restored = read_zarr(output)
    assert set(restored.points) == {"sample_a_transcripts_mosaic_1"}
    assert restored.attrs == before_attrs
    validate_cosmx_store(output, check_point_contents=True)


def test_add_cosmx_samples_commits_each_mosaic_element_independently(
    decoded_cosmx_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "second_mosaic_failure.zarr"
    cosmx(
        {"sample_a": CosmxSample(path=decoded_cosmx_path, fovs=[2])},
        output,
        instance_labels=False,
        compartment_labels=False,
        points=False,
        image_chunks=(1, 4, 4),
    )
    original_write_attrs = SpatialData.write_attrs
    incremental_writes = 0

    def fail_second_incremental_metadata_write(self: SpatialData, *args: object, **kwargs: object) -> object:
        nonlocal incremental_writes
        if Path(self.path) == output:
            incremental_writes += 1
            if incremental_writes == 2:
                raise RuntimeError("planned second mosaic metadata failure")
        return original_write_attrs(self, *args, **kwargs)

    monkeypatch.setattr(SpatialData, "write_attrs", fail_second_incremental_metadata_write)
    with pytest.raises(RuntimeError, match="planned second mosaic metadata failure"):
        add_cosmx_samples(
            output,
            {"sample_b": CosmxSample(path=decoded_cosmx_path, fovs=[1, 3])},
            instance_labels=False,
            compartment_labels=False,
            points=False,
            image_chunks=(1, 4, 4),
        )

    partial = read_zarr(output)
    assert "sample_b_morphology_image_mosaic_1" in partial.images
    assert "sample_b_morphology_image_mosaic_2" not in partial.images
    assert "sample_b_morphology_image_mosaic_1" in partial.attrs["harpy"]["images"]
    assert "sample_b_morphology_image_mosaic_2" not in partial.attrs["harpy"]["images"]
    validate_cosmx_store(output)


def test_add_cosmx_samples_failure_in_later_requested_sample_keeps_earlier_sample(
    decoded_cosmx_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "later_sample_failure.zarr"
    cosmx(
        {"sample_a": CosmxSample(path=decoded_cosmx_path, fovs=[1])},
        output,
        instance_labels=False,
        compartment_labels=False,
        points=False,
        image_chunks=(1, 4, 4),
    )
    original = _reader._add_morphology_images

    def fail_sample_c(*args: object, **kwargs: object) -> SpatialData:
        if kwargs["sample_id"] == "sample_c":
            raise RuntimeError("planned sample_c failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(_reader, "_add_morphology_images", fail_sample_c)
    with pytest.raises(RuntimeError, match="planned sample_c failure"):
        add_cosmx_samples(
            output,
            {
                "sample_b": CosmxSample(path=decoded_cosmx_path, fovs=[2]),
                "sample_c": CosmxSample(path=decoded_cosmx_path, fovs=[3]),
            },
            instance_labels=False,
            compartment_labels=False,
            points=False,
            image_chunks=(1, 4, 4),
        )

    partial = read_zarr(output)
    assert "sample_b_morphology_image_mosaic_1" in partial.images
    assert "sample_c_morphology_image_mosaic_1" not in partial.images
    validate_cosmx_store(output)


def _write_valid_transcripts(root: Path) -> Path:
    preview = _reader._preview_cosmx(_reader._discover_cosmx(root))
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
    return root
