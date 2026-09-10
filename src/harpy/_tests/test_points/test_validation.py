from collections import Counter
from copy import deepcopy

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from spatialdata import SpatialData, read_zarr
from spatialdata.models import PointsModel
from spatialdata.transformations import Identity

from harpy.points import _validation, validate_points

PANEL = {
    "feature_key": "target",
    "feature_class_key": "kind",
    "classes": ["Control", "Expression", "Unused"],
    "features_by_class": {
        "Control": ["CtrlA", "CtrlZero"],
        "Expression": ["GeneA", "GeneZero"],
        "Unused": ["NeverDetected"],
    },
}


def _frame():
    return pd.DataFrame(
        {
            "x": np.arange(6, dtype=float),
            "y": np.zeros(6),
            "target": ["GeneA", "CtrlA", "GeneA", "CtrlA", "CtrlA", "GeneA"],
            "kind": pd.Categorical(
                ["Expression", "Control", "Expression", "Control", "Control", "Expression"],
                categories=PANEL["classes"],
            ),
            "quality": np.ones(6),
        }
    )


def _sdata(frame=None):
    if frame is None:
        frame = _frame()
    points = frame if isinstance(frame, dd.DataFrame) else dd.from_pandas(frame, npartitions=3)
    return SpatialData(
        points={"calls": PointsModel.parse(points, transformations={"global": Identity()})},
        attrs={
            "external": {"note": "keep"},
            "harpy": {
                "metadata_version": 1,
                "points": {"calls": {"feature_panel": "assay"}},
                "feature_panels": {"assay": deepcopy(PANEL)},
            },
        },
    )


def _files(root):
    return {
        str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("backed", [False, True])
def test_validation_preserves_valid_points_metadata_and_store(tmp_path, backed):
    """The public validator is read-only, including when points are backed by a store."""
    sdata = _sdata()
    if backed:
        sdata.write(tmp_path / "points.zarr")
        sdata = read_zarr(sdata.path)
    original = sdata.points["calls"]
    expected = original.compute()
    attrs = deepcopy(sdata.attrs)
    points_attrs = deepcopy(dict(original.attrs))
    files = _files(sdata.path) if backed else None

    assert validate_points(sdata, "calls") is None

    assert sdata.points["calls"] is original
    assert sdata.attrs == attrs
    assert original.attrs == points_attrs
    pd.testing.assert_frame_equal(original.compute(), expected)
    if backed:
        assert _files(sdata.path) == files


def test_validation_accepts_empty_points_with_a_valid_schema():
    sdata = _sdata(_frame().iloc[:0])

    assert validate_points(sdata, "calls") is None


def test_validation_ignores_unrelated_panels_points_and_reader_metadata():
    sdata = _sdata()
    sdata.points["other"] = PointsModel.parse(
        _frame().assign(target="NotInPanel"), transformations={"global": Identity()}
    )
    root = sdata.attrs["harpy"]
    root["points"]["other"] = {"feature_panel": "missing"}
    root["points"]["calls"]["sample_id"] = None
    root["feature_panels"]["unrelated"] = {"invalid": "record"}
    root["images"] = "not part of points validation"
    attrs = deepcopy(sdata.attrs)

    assert validate_points(sdata, "calls") is None
    assert sdata.attrs == attrs


@pytest.mark.parametrize(
    "missing, match",
    [
        ("element", "does not exist"),
        ("panel", r"hp\.pt\.add_feature_panel.*features_by_class"),
        ("target", "panel column 'target'"),
        ("kind", "panel column 'kind'"),
    ],
)
def test_validation_rejects_missing_inputs_without_repair(missing, match):
    sdata = _sdata()
    if missing == "element":
        del sdata.points["calls"]
    elif missing == "panel":
        del sdata.attrs["harpy"]
    else:
        sdata.points["calls"] = sdata.points["calls"].drop(columns=missing)
    attrs = deepcopy(sdata.attrs)
    original_points = dict(sdata.points)

    with pytest.raises(ValueError, match=match):
        validate_points(sdata, "calls")

    assert sdata.attrs == attrs
    assert set(sdata.points) == set(original_points)
    for name, original in original_points.items():
        assert sdata.points[name] is original


def test_validation_propagates_a_later_partition_error_with_unknown_categories():
    """Execute content checks beyond the first partition and report the selected element."""
    frame = _frame()
    frame.loc[frame.index[-1], "kind"] = "Control"
    sdata = _sdata(frame)
    sdata.points["calls"]["kind"] = sdata.points["calls"]["kind"].cat.as_unknown()
    original = sdata.points["calls"]
    expected = original.compute()
    attrs = deepcopy(sdata.attrs)

    with pytest.raises(ValueError, match="Points element 'calls'.*expected 'Expression'"):
        validate_points(sdata, "calls")

    assert sdata.points["calls"] is original
    assert sdata.attrs == attrs
    pd.testing.assert_frame_equal(original.compute(), expected)


def test_validation_reads_each_partition_once_and_collects_only_errors(monkeypatch):
    """Protect the projected, single-scan validation path without materializing source rows."""
    reads = Counter()
    checked_rows = []

    def read_partition(ordinal):
        reads[ordinal] += 1
        return _frame().iloc[2 * ordinal : 2 * ordinal + 2]

    points = dd.from_delayed([dask.delayed(read_partition)(ordinal) for ordinal in range(3)], meta=_frame().iloc[:0])
    sdata = _sdata(points)
    # SpatialData checks index ordering while constructing the fixture. Count
    # only source-partition reads performed by the explicit validation below.
    reads.clear()
    validate_partition = _validation._feature_panel_partition_errors

    def check_partition(partition, **kwargs):
        assert list(partition.columns) == ["target", "kind"]
        checked_rows.append(len(partition))
        errors = validate_partition(partition, **kwargs)
        assert isinstance(errors, pd.Series) and len(errors) <= 1
        return errors

    def unexpected_dataframe_operation(*args, **kwargs):
        pytest.fail("Validation must not collect the source dataframe or shuffle points")

    monkeypatch.setattr(_validation, "_feature_panel_partition_errors", check_partition)
    monkeypatch.setattr(dd.DataFrame, "compute", unexpected_dataframe_operation)
    monkeypatch.setattr(dd.DataFrame, "shuffle", unexpected_dataframe_operation)
    with dask.config.set(scheduler="synchronous"):
        assert validate_points(sdata, "calls") is None

    assert reads == Counter({0: 1, 1: 1, 2: 1})
    assert checked_rows == [2, 2, 2]
