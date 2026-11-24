from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import dask.dataframe as dd
import pandas as pd
import pytest
from dask.dataframe import DataFrame as DaskDataFrame
from spatialdata import SpatialData, read_zarr
from spatialdata.models import PointsModel
from spatialdata.transformations import Identity, Scale, set_transformation

from harpy import __version__
from harpy._feature_panels import _parse_feature_panel
from harpy.io import CosmxSample, cosmx, validate_cosmx_store
from harpy.io._cosmx._validation import _validate_cosmx_sdata
from harpy.points import add_feature_panel
from harpy.points._points import add_points

_PANEL_RECORD = {
    "feature_key": "gene",
    "feature_class_key": "code_class",
    "classes": ["Endogenous", "Negative"],
    "features_by_class": {
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
        points=False,
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


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", r"fov_bounds is required; run the CosMx reader again"),
        ("keys", "keys must equal fovs as strings"),
        ("missing_bound", "must contain exactly the fields"),
        ("non_integer", r"\.x1 must be a nonnegative integer"),
        ("empty", "must have y0 < y1 and x0 < x1"),
        ("overlap", "rectangles of FOVs 1 and 2 overlap"),
        ("outside", r"must lie within the \(y, x\) shape \(8, 16\) of element 'sample_morphology_image_mosaic_1'"),
        (
            "outside_labels",
            r"must lie within the \(y, x\) shape \(8, 16\) of element 'sample_instance_labels_mosaic_1'",
        ),
    ],
)
def test_validate_cosmx_store_rejects_invalid_fov_bounds(reader_store: Path, mutation: str, message: str) -> None:
    sdata = read_zarr(reader_store)
    attrs = deepcopy(sdata.attrs)
    record = attrs["harpy"]["coordinate_systems"]["sample_global_1"]
    bounds = record["fov_bounds"]
    assert bounds["2"] == {"y0": 0, "x0": 8, "y1": 8, "x1": 16}
    if mutation == "missing":
        del record["fov_bounds"]
    elif mutation == "keys":
        del bounds["2"]
    elif mutation == "missing_bound":
        del bounds["2"]["x1"]
    elif mutation == "non_integer":
        bounds["2"]["x1"] = 16.0
    elif mutation == "empty":
        bounds["2"]["x1"] = bounds["2"]["x0"]
    elif mutation == "overlap":
        bounds["2"]["x0"] = 7
    else:
        bounds["2"]["x1"] = 17
        if mutation == "outside_labels":
            # Without its record, the image is not checked, and the labels of the mosaic are.
            del attrs["harpy"]["images"]["sample_morphology_image_mosaic_1"]
    sdata.attrs = attrs
    sdata.write_attrs()

    with pytest.raises(ValueError, match=message):
        validate_cosmx_store(reader_store)


@pytest.mark.parametrize("number", [0, -1, True, "1", 1.0, None])
def test_validate_cosmx_store_rejects_invalid_mosaic_number(reader_store: Path, number: object) -> None:
    sdata = read_zarr(reader_store)
    attrs = deepcopy(sdata.attrs)
    attrs["harpy"]["coordinate_systems"]["sample_global_1"]["mosaic"] = number
    sdata.attrs = attrs
    sdata.write_attrs()

    with pytest.raises(
        ValueError,
        match=re.escape(
            f"harpy.coordinate_systems['sample_global_1'].mosaic must be a positive integer, found {number!r}."
        ),
    ):
        validate_cosmx_store(reader_store)


def test_validate_cosmx_store_requires_the_coordinate_system_registry(reader_store: Path) -> None:
    sdata = read_zarr(reader_store)
    attrs = deepcopy(sdata.attrs)
    del attrs["harpy"]["coordinate_systems"]
    sdata.attrs = attrs
    sdata.write_attrs()

    with pytest.raises(
        ValueError,
        match=r"harpy\.coordinate_systems must record at least one coordinate system\. .*run the CosMx reader again",
    ):
        validate_cosmx_store(reader_store)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            "unknown",
            "harpy.coordinate_systems['sample_global_9'] records coordinate system 'sample_global_9', to which no "
            "element of sdata has a transformation.",
        ),
        (
            "duplicate_mosaic",
            "harpy.coordinate_systems['sample_global_2'] and harpy.coordinate_systems['sample_global_1'] both "
            "record mosaic 1 of sample 'sample', which has one coordinate system.",
        ),
        (
            "run_fact",
            "harpy.coordinate_systems['sample_global_2'].microns_per_unit differs from "
            "harpy.coordinate_systems['sample_global_1'].microns_per_unit, although both are coordinate systems of "
            "sample 'sample'.",
        ),
    ],
)
def test_validate_cosmx_store_rejects_disagreeing_coordinate_system_records(
    reader_store: Path,
    mutation: str,
    message: str,
) -> None:
    sdata = read_zarr(reader_store)
    attrs = deepcopy(sdata.attrs)
    records = attrs["harpy"]["coordinate_systems"]
    assert list(records) == ["sample_global_1", "sample_global_2"]
    if mutation == "unknown":
        records["sample_global_9"] = deepcopy(records["sample_global_1"]) | {"mosaic": 9}
    elif mutation == "duplicate_mosaic":
        records["sample_global_2"]["mosaic"] = 1
    else:
        records["sample_global_2"]["microns_per_unit"] *= 2
    sdata.attrs = attrs
    sdata.write_attrs()

    with pytest.raises(ValueError, match=re.escape(message)):
        validate_cosmx_store(reader_store)


@pytest.mark.parametrize(
    ("transformations", "message"),
    [
        (
            {"sample_global_1": Scale([2.0, 2.0], axes=("x", "y"))},
            "CosMx element 'sample_morphology_image_mosaic_1' must lie unscaled in coordinate system "
            "'sample_global_1', with an Identity transformation",
        ),
        (
            {"sample_global_1": Identity(), "sample_global_2": Identity()},
            "CosMx element 'sample_morphology_image_mosaic_1' must have a transformation to exactly one coordinate "
            "system in harpy.coordinate_systems, found ['sample_global_1', 'sample_global_2']",
        ),
        (
            {"aligned": Identity()},
            "CosMx element 'sample_morphology_image_mosaic_1' must have a transformation to exactly one coordinate "
            "system in harpy.coordinate_systems, found none",
        ),
    ],
)
def test_validate_cosmx_store_rejects_misplaced_reader_rasters(
    reader_store: Path,
    transformations: dict[str, Any],
    message: str,
) -> None:
    sdata = read_zarr(reader_store)
    set_transformation(sdata["sample_morphology_image_mosaic_1"], transformations, set_all=True)
    sdata.write_transformations("sample_morphology_image_mosaic_1")

    with pytest.raises(ValueError, match=re.escape(message)):
        validate_cosmx_store(reader_store)


def test_validate_cosmx_store_accepts_points_registered_with_add_feature_panel(tmp_path: Path) -> None:
    output = _write_points_store(tmp_path)
    sdata = read_zarr(output)
    sdata.points["filtered_transcripts"] = PointsModel.parse(
        pd.DataFrame({"x": [1.0], "y": [2.0], "gene": ["GeneA"]}), transformations={"aligned": Identity()}
    )
    sdata.write_element("filtered_transcripts")
    add_feature_panel(
        sdata,
        "filtered_transcripts",
        feature_key="gene",
        feature_class_key="code_class",
        features_by_class=_PANEL_RECORD["features_by_class"],
    )
    assert read_zarr(output).attrs["harpy"]["points"]["filtered_transcripts"].keys() == {"feature_panel"}

    validate_cosmx_store(output)


def test_validate_cosmx_store_structural_opt_out_is_lazy_and_non_mutating(
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

    validate_cosmx_store(output, check_point_contents=False)


def test_validate_cosmx_store_accepts_compatible_metadata_extensions(tmp_path: Path) -> None:
    output = _write_points_store(tmp_path)
    sdata = read_zarr(output)
    attrs = deepcopy(sdata.attrs)
    point_record = next(iter(attrs["harpy"]["points"].values()))
    panel_name = point_record["feature_panel"]
    point_record["extension"] = {"consumer": "example"}
    attrs["harpy"]["feature_panels"][panel_name]["extension"] = True
    attrs["harpy"]["coordinate_systems"]["sample_global_1"]["extension"] = [1, 2]
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
        ("empty", "at least one coordinate system"),
        ("missing_element", "references missing points element"),
        ("invalid_sample", "sample_id"),
        ("missing_panel", "references missing panel"),
        ("panel_hash", "does not match canonical contents"),
        ("ambiguous_feature", "belongs to both"),
        ("same_columns", "different feature and feature-class keys"),
        ("legacy_feature_keys", "feature_panels.*feature_key"),
        ("legacy_grouping_keys", "feature_panels.*classes"),
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
        harpy_metadata["coordinate_systems"] = {}
    elif mutation == "missing_element":
        points_metadata["missing_points"] = points_metadata.pop(point_name)
    elif mutation == "invalid_sample":
        harpy_metadata["coordinate_systems"]["sample_global_1"]["sample_id"] = "invalid-sample"
    elif mutation == "missing_panel":
        point_record["feature_panel"] = "feature_panel_missing"
    elif mutation == "panel_hash":
        harpy_metadata["feature_panels"][panel_name]["features_by_class"]["Endogenous"].append("GeneZ")
    elif mutation == "ambiguous_feature":
        harpy_metadata["feature_panels"][panel_name]["features_by_class"]["Negative"].insert(0, "GeneA")
    elif mutation == "same_columns":
        harpy_metadata["feature_panels"][panel_name]["feature_class_key"] = "gene"
    elif mutation == "legacy_feature_keys":
        panel = harpy_metadata["feature_panels"][panel_name]
        panel["feature_column"] = panel.pop("feature_key")
        panel["class_column"] = panel.pop("feature_class_key")
    else:
        panel = harpy_metadata["feature_panels"][panel_name]
        panel["categories"] = panel.pop("classes")
        panel["targets_by_class"] = panel.pop("features_by_class")

    sdata.attrs = attrs
    sdata.write_attrs()

    with pytest.raises(ValueError, match=message):
        validate_cosmx_store(output)


def test_validate_cosmx_store_default_content_check_accepts_detected_panel_subset(tmp_path: Path) -> None:
    output = _write_points_store(tmp_path, rows=[("GeneA", "Endogenous")])

    validate_cosmx_store(output)


def test_validate_cosmx_store_default_content_check_validates_each_points_partition(tmp_path: Path) -> None:
    output = _write_points_store(
        tmp_path,
        rows=[("GeneA", "Endogenous"), ("GeneB", "Endogenous"), ("Unknown", "Endogenous")],
        npartitions=3,
    )

    with pytest.raises(ValueError, match="feature 'Unknown' absent from its feature panel"):
        validate_cosmx_store(output)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([("Unknown", "Endogenous")], "absent from its feature panel"),
        ([("GeneA", "Negative")], "expected 'Endogenous'"),
        ([("GeneA", "Endogenous"), ("GeneA", "Negative")], "expected 'Endogenous'"),
        ([(None, "Endogenous")], "null panel feature or feature class"),
    ],
)
def test_validate_cosmx_store_default_content_check_rejects_inconsistent_points(
    tmp_path: Path,
    rows: list[tuple[str | None, str | None]],
    message: str,
) -> None:
    output = _write_points_store(tmp_path, rows=rows)

    with pytest.raises(ValueError, match=message):
        validate_cosmx_store(output)


def test_validate_cosmx_store_structural_opt_out_skips_inconsistent_point_contents(tmp_path: Path) -> None:
    output = _write_points_store(tmp_path, rows=[("Unknown", "Endogenous")])

    validate_cosmx_store(output, check_point_contents=False)


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
    panel_features = {feature for features in _PANEL_RECORD["features_by_class"].values() for feature in features}
    observed_features = {feature for feature, _ in rows if feature is not None}
    observed_classes = {feature_class for _, feature_class in rows if feature_class is not None}
    frame = pd.DataFrame(
        {
            "x": [float(index) for index in range(len(rows))],
            "y": [float(index) for index in range(len(rows))],
            "gene": pd.Categorical(
                [feature for feature, _ in rows],
                categories=sorted(panel_features | observed_features),
            ),
            "code_class": (
                pd.Categorical(
                    [feature_class for _, feature_class in rows],
                    categories=sorted(set(_PANEL_RECORD["classes"]) | observed_classes),
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

    panel_record: dict[str, Any] = deepcopy(_PANEL_RECORD)
    panel_name = _parse_feature_panel(panel_record, panel_name="test_panel").storage_key
    coordinate_system_record = {
        "sample_id": "sample",
        "mosaic": 1,
        "microns_per_unit": 1.0,
        "fovs": [1],
        "fov_bounds": {"1": {"y0": 0, "x0": 0, "y1": 8, "x1": 8}},
        "source_origin_px": {"x": 0, "y": 0},
        "orientation": {"flip_x": True, "flip_y": False},
        "mosaic_grouping": {"mode": "spatial_groups", "adjacency_tolerance_px": 0},
    }
    harpy_metadata: dict[str, Any] = {
        "metadata_version": 1,
        "provenance": {"reader": "cosmx", "reader_version": __version__},
        "coordinate_systems": {"sample_global_1": coordinate_system_record},
    }
    if with_panel:
        harpy_metadata["points"] = {"sample_transcripts_mosaic_1": {"feature_panel": panel_name}}
        harpy_metadata["feature_panels"] = {panel_name: panel_record}
    sdata.attrs = {"harpy": harpy_metadata}
    sdata.write_attrs()
    return output
