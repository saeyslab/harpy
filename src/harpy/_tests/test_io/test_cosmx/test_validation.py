from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import dask.dataframe as dd
import pandas as pd
import pytest
from dask.dataframe import DataFrame as DaskDataFrame
from spatialdata import SpatialData, read_zarr
from spatialdata.transformations import Identity

from harpy import __version__
from harpy.io import CosmxSample, cosmx, validate_cosmx_store
from harpy.io._cosmx._transcripts import _feature_panel_name
from harpy.io._cosmx._validation import _validate_cosmx_sdata
from harpy.points._points import add_points

_PANEL_METADATA = {
    "feature_column": "gene",
    "class_column": "code_class",
    "categories": ["Endogenous", "Negative"],
    "targets_by_class": {
        "Endogenous": ["GeneA", "GeneB"],
        "Negative": ["Negative01"],
    },
}


@pytest.fixture
def reader_store(decoded_cosmx_path: Path, tmp_path: Path) -> Path:
    output = tmp_path / "reader.zarr"
    cosmx(
        {"sample": CosmxSample(path=decoded_cosmx_path)},
        output,
        transcripts=False,
        image_chunks=(1, 4, 4),
        labels_chunks=(4, 4),
    )
    return output


def test_validate_cosmx_store_accepts_reader_output_and_unregistered_elements(reader_store: Path) -> None:
    before = read_zarr(reader_store)
    before_attrs = deepcopy(before.attrs)
    before_elements = {(element_type, name) for element_type, name, _ in before.gen_elements()}

    validate_cosmx_store(reader_store)

    after = read_zarr(reader_store)
    assert after.attrs == before_attrs
    assert {(element_type, name) for element_type, name, _ in after.gen_elements()} == before_elements

    # Removing reader metadata turns an existing element into an unrelated,
    # unregistered downstream element; other registered CosMx elements remain.
    attrs = deepcopy(after.attrs)
    image_name = next(iter(attrs["harpy"]["images"]))
    del attrs["harpy"]["images"][image_name]
    after.attrs = attrs
    after.write_attrs()

    validate_cosmx_store(reader_store)


@pytest.mark.parametrize("discriminator", ["both", "neither"])
def test_validate_cosmx_store_rejects_ambiguous_label_family(reader_store: Path, discriminator: str) -> None:
    sdata = read_zarr(reader_store)
    attrs = deepcopy(sdata.attrs)
    record = next(iter(attrs["harpy"]["labels"].values()))
    if discriminator == "both":
        record["categories"] = {"0": "background", "1": "nuclear", "2": "membrane", "3": "cytoplasmic"}
    else:
        record.pop("instance_id_encoding")
    sdata.attrs = attrs
    sdata.write_attrs()

    with pytest.raises(ValueError, match="exactly one of 'instance_id_encoding' or 'categories'"):
        validate_cosmx_store(reader_store)


def test_validate_cosmx_store_structural_check_is_lazy_and_non_mutating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = _write_points_store(tmp_path)

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("Structural CosMx validation computed or wrote store data.")

    monkeypatch.setattr(DaskDataFrame, "compute", fail)
    monkeypatch.setattr(SpatialData, "write_attrs", fail)
    monkeypatch.setattr(SpatialData, "write_element", fail)
    monkeypatch.setattr(SpatialData, "delete_element_from_disk", fail)

    validate_cosmx_store(output)


def test_validate_cosmx_store_accepts_compatible_metadata_extensions(tmp_path: Path) -> None:
    output = _write_points_store(tmp_path)
    sdata = read_zarr(output)
    attrs = deepcopy(sdata.attrs)
    point_record = next(iter(attrs["harpy"]["points"].values()))
    panel_name = point_record["feature_panel"]
    point_record["extension"] = {"consumer": "example"}
    attrs["harpy"]["feature_panels"][panel_name]["extension"] = True
    sdata.attrs = attrs
    sdata.write_attrs()

    validate_cosmx_store(output)


def test_validate_cosmx_store_accepts_panel_less_points_with_string_class(tmp_path: Path) -> None:
    output = _write_points_store(tmp_path, with_panel=False)

    validate_cosmx_store(output)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("version", "metadata_version"),
        ("reader", "provenance.reader"),
        ("empty", "at least one"),
        ("missing_element", "references missing points element"),
        ("invalid_sample", "sample_id"),
        ("missing_panel", "references missing panel"),
        ("panel_hash", "does not match canonical contents"),
    ],
)
def test_validate_cosmx_store_rejects_invalid_structure(tmp_path: Path, mutation: str, message: str) -> None:
    output = _write_points_store(tmp_path)
    sdata = read_zarr(output)
    attrs = deepcopy(sdata.attrs)
    harpy_metadata = attrs["harpy"]
    points_metadata = harpy_metadata["points"]
    point_name, point_record = next(iter(points_metadata.items()))
    panel_name = point_record["feature_panel"]

    if mutation == "version":
        harpy_metadata["metadata_version"] = 999
    elif mutation == "reader":
        harpy_metadata["provenance"]["reader"] = "another-reader"
    elif mutation == "empty":
        harpy_metadata["points"] = {}
    elif mutation == "missing_element":
        points_metadata["missing_points"] = points_metadata.pop(point_name)
    elif mutation == "invalid_sample":
        point_record["sample_id"] = "invalid-sample"
    elif mutation == "missing_panel":
        point_record["feature_panel"] = "feature_panel_missing"
    else:
        harpy_metadata["feature_panels"][panel_name]["targets_by_class"]["Endogenous"].append("GeneZ")

    sdata.attrs = attrs
    sdata.write_attrs()

    with pytest.raises(ValueError, match=message):
        validate_cosmx_store(output)


def test_validate_cosmx_store_deep_check_accepts_detected_panel_subset(tmp_path: Path) -> None:
    output = _write_points_store(tmp_path, rows=[("GeneA", "Endogenous")])

    validate_cosmx_store(output, check_point_contents=True)


def test_validate_cosmx_store_deep_check_validates_each_points_partition(tmp_path: Path) -> None:
    output = _write_points_store(
        tmp_path,
        rows=[("GeneA", "Endogenous"), ("GeneB", "Endogenous"), ("Unknown", "Endogenous")],
        npartitions=3,
    )

    with pytest.raises(ValueError, match="target 'Unknown' absent from its feature panel"):
        validate_cosmx_store(output, check_point_contents=True)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([("Unknown", "Endogenous")], "absent from its feature panel"),
        ([("GeneA", "Negative")], "expected 'Endogenous'"),
        ([("GeneA", "Endogenous"), ("GeneA", "Negative")], "expected 'Endogenous'"),
        ([(None, "Endogenous")], "null panel target or feature class"),
    ],
)
def test_validate_cosmx_store_deep_check_rejects_inconsistent_points(
    tmp_path: Path,
    rows: list[tuple[str | None, str | None]],
    message: str,
) -> None:
    output = _write_points_store(tmp_path, rows=rows)

    with pytest.raises(ValueError, match=message):
        validate_cosmx_store(output, check_point_contents=True)


def test_internal_cosmx_store_validator_requires_backed_spatialdata() -> None:
    with pytest.raises(ValueError, match="requires a backed SpatialData"):
        _validate_cosmx_sdata(SpatialData())


def test_validate_cosmx_store_rejects_unreadable_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Could not read CosMx SpatialData Zarr store"):
        validate_cosmx_store(tmp_path / "missing.zarr")


def _write_points_store(
    tmp_path: Path,
    *,
    rows: list[tuple[str | None, str | None]] | None = None,
    with_panel: bool = True,
    npartitions: int = 1,
) -> Path:
    rows = [("GeneA", "Endogenous")] if rows is None else rows
    panel_targets = {target for targets in _PANEL_METADATA["targets_by_class"].values() for target in targets}
    observed_targets = {target for target, _ in rows if target is not None}
    observed_classes = {feature_class for _, feature_class in rows if feature_class is not None}
    frame = pd.DataFrame(
        {
            "x": [float(index) for index in range(len(rows))],
            "y": [float(index) for index in range(len(rows))],
            "gene": pd.Categorical(
                [target for target, _ in rows],
                categories=sorted(panel_targets | observed_targets),
            ),
            "code_class": (
                pd.Categorical(
                    [feature_class for _, feature_class in rows],
                    categories=sorted(set(_PANEL_METADATA["categories"]) | observed_classes),
                )
                if with_panel
                else pd.Series([feature_class for _, feature_class in rows], dtype="string[pyarrow]")
            ),
        }
    )
    output = tmp_path / "points.zarr"
    SpatialData().write(output)
    sdata = read_zarr(output)
    sdata = add_points(
        sdata,
        ddf=dd.from_pandas(frame, npartitions=npartitions),
        output_points_name="sample_transcripts_mosaic_1",
        coordinates={"x": "x", "y": "y"},
        transformations={"sample_global_1": Identity()},
        overwrite=False,
    )

    panel_metadata: dict[str, Any] = deepcopy(_PANEL_METADATA)
    panel_name = _feature_panel_name(panel_metadata)
    point_metadata = {
        "sample_id": "sample",
        "fovs": [1],
        "mosaic": {"mode": "spatial_groups", "adjacency_tolerance_px": 0},
        "source_origin_px": {"x": 0, "y": 0},
        "orientation": {"flip_x": True, "flip_y": False},
        "pixel_size_um": 1.0,
    }
    harpy_metadata = {
        "metadata_version": 1,
        "provenance": {"reader": "cosmx", "reader_version": __version__},
        "points": {"sample_transcripts_mosaic_1": point_metadata},
    }
    if with_panel:
        point_metadata["feature_panel"] = panel_name
        harpy_metadata["feature_panels"] = {panel_name: panel_metadata}
    sdata.attrs = {"harpy": harpy_metadata}
    sdata.write_attrs()
    return output
