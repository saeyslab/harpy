from copy import deepcopy

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from spatialdata import SpatialData, read_zarr
from spatialdata.models import Image2DModel, Labels2DModel, PointsModel
from spatialdata.transformations import Translation, get_transformation

import harpy as hp
from harpy._feature_panels import _make_feature_panel
from harpy.points import _feature_panel

PANEL = {
    "SystemControl": ["System1"],
    "Negative": ["Negative2", "Negative1"],
    "Endogenous": ["ZeroGene", "GeneB", "GeneA"],
}


def _sdata(tmp_path, *, class_kind="missing", backed=False):
    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [0.0, 1.0, 0.0, 1.0],
            "gene": pd.Categorical(["GeneB", "Negative1", "GeneA", "GeneB"]),
            "quality": [8, 9, 10, 11],
        },
        index=pd.Index([4, 8, 12, 16], name="source_index"),
    )
    if class_kind != "missing":
        values = ["Endogenous", "Negative", "Endogenous", "Endogenous"]
        if class_kind == "string":
            frame["kind"] = pd.array(values, dtype="string")
        else:
            categories = list(PANEL) if class_kind == "wrong_order" else sorted(PANEL)
            frame["kind"] = pd.Categorical(values, categories=categories)
    points = PointsModel.parse(
        dd.from_pandas(frame, npartitions=2), transformations={"sample": Translation([11, 23], axes=("x", "y"))}
    )
    sdata = SpatialData(
        points={"calls": points, "unrelated": points.copy()},
        images={"image": Image2DModel.parse(np.ones((1, 2, 2), dtype=np.uint16), dims=("c", "y", "x"))},
        labels={"labels": Labels2DModel.parse(np.ones((2, 2), dtype=np.uint32), dims=("y", "x"))},
        tables={"table": AnnData(X=np.ones((1, 1)))},
        attrs={"external": {"project": "keep"}},
    )
    if backed:
        sdata.write(tmp_path / "points.zarr")
        return read_zarr(sdata.path)
    return sdata


def _register(sdata, **kwargs):
    return hp.pt.add_feature_panel(
        sdata, "calls", feature_key="gene", feature_class_key="kind", features_by_class=kwargs.get("panel", PANEL)
    )


def _files(path):
    return {
        str(file.relative_to(path)): (file.read_bytes(), file.stat().st_mtime_ns)
        for file in path.rglob("*")
        if file.is_file()
    }


@pytest.mark.parametrize("backed", [False, True])
@pytest.mark.parametrize("class_kind", ["missing", "string", "wrong_order"])
def test_registration_normalizes_only_classes_and_enables_panel_summaries(tmp_path, backed, class_kind):
    sdata = _sdata(tmp_path, class_kind=class_kind, backed=backed)
    original = sdata.points["calls"].compute()
    transformations = deepcopy(get_transformation(sdata.points["calls"], get_all=True))
    unrelated = {name: sdata[name] for name in ("image", "labels", "table", "unrelated")}
    untouched_paths = ("images/image", "labels/labels", "tables/table", "points/unrelated")
    files = {path: _files(sdata.path / path) for path in untouched_paths} if backed else {}

    assert _register(sdata) is sdata
    for name, element in unrelated.items():
        assert sdata[name] is element
    for path, snapshot in files.items():
        assert _files(sdata.path / path) == snapshot

    containers = (sdata, read_zarr(sdata.path)) if backed else (sdata,)
    for container in containers:
        actual = container.points["calls"].compute()
        pd.testing.assert_frame_equal(actual.drop(columns="kind"), original.drop(columns="kind", errors="ignore"))
        assert actual["kind"].tolist() == ["Endogenous", "Negative", "Endogenous", "Endogenous"]
        assert actual["kind"].cat.categories.tolist() == sorted(PANEL)
        assert get_transformation(container.points["calls"], get_all=True) == transformations
        assert container.attrs["external"] == {"project": "keep"}
        assert set(container.attrs["harpy"]) == {"metadata_version", "points", "feature_panels"}
        record = container.attrs["harpy"]["points"]["calls"]
        assert set(record) == {"feature_panel"}
        panel = container.attrs["harpy"]["feature_panels"][record["feature_panel"]]
        assert panel["classes"] == sorted(PANEL)
        assert panel["features_by_class"] == {key: sorted(value) for key, value in PANEL.items()}
        targets = hp.qc.summarize_points(container, "calls").per_target.set_index("feature")
        assert targets.loc["GeneB", "n_points"] == 2
        assert targets.loc[["ZeroGene", "Negative2", "System1"], "n_points"].tolist() == [0, 0, 0]
        assert targets.n_points.sum() == len(original)
    assert not list(tmp_path.glob(".points.zarr.harpy-*"))


@pytest.mark.parametrize("backed", [False, True])
@pytest.mark.parametrize("unknown_categories", [False, True])
def test_compatible_classes_register_metadata_without_writing_points(tmp_path, monkeypatch, backed, unknown_categories):
    """Compatible partition dtypes avoid rewrites, even when Dask's categories are unknown."""
    sdata = _sdata(tmp_path, class_kind="categorical", backed=backed)
    if unknown_categories:
        sdata.points["calls"]["kind"] = sdata.points["calls"]["kind"].cat.as_unknown()
    original = sdata.points["calls"]
    attrs = deepcopy(dict(original.attrs))
    sdata.attrs["harpy"] = {"metadata_version": 1, "points": {"calls": {"note": "keep"}}}
    if backed:
        sdata.write_attrs()
    files = _files(sdata.path / "points") if backed else None

    def unexpected_write(*args, **kwargs):
        pytest.fail("Compatible classes must not stage or rewrite points")

    monkeypatch.setattr(_feature_panel, "_replace_element_on_disk", unexpected_write)
    monkeypatch.setattr(SpatialData, "write", unexpected_write)
    _register(sdata)
    # Ordering differences must not change panel identity or trigger a rewrite.
    _register(sdata, panel={key: list(reversed(PANEL[key])) for key in reversed(PANEL)})
    assert sdata.points["calls"] is original
    assert original.attrs == attrs
    assert sdata.attrs["harpy"]["points"]["calls"]["note"] == "keep"
    assert len(sdata.attrs["harpy"]["feature_panels"]) == 1
    if backed:
        assert _files(sdata.path / "points") == files
        assert read_zarr(sdata.path).attrs == sdata.attrs
    assert not list(tmp_path.glob(".points.zarr.harpy-*"))


@pytest.mark.parametrize("unsorted_field", ["classes", "features"])
def test_registration_rejects_unsorted_stored_panels_without_changing_points(tmp_path, unsorted_field):
    """Do not silently repair panel metadata or rewrite points to match it."""
    sdata = _sdata(tmp_path)
    _register(sdata)
    panel_name = sdata.attrs["harpy"]["points"]["calls"]["feature_panel"]
    record = sdata.attrs["harpy"]["feature_panels"][panel_name]
    if unsorted_field == "classes":
        record["classes"].reverse()
    else:
        record["features_by_class"]["Endogenous"].reverse()
    sdata.write(tmp_path / "reuse.zarr")
    sdata = read_zarr(sdata.path)
    previous_attrs = deepcopy(sdata.attrs)
    points_files = _files(sdata.path / "points")
    previous_points = sdata.points["unrelated"]

    with pytest.raises(ValueError, match="must be sorted"):
        hp.pt.add_feature_panel(
            sdata, "unrelated", feature_key="gene", feature_class_key="kind", features_by_class=PANEL
        )

    assert sdata.points["unrelated"] is previous_points
    assert _files(sdata.path / "points") == points_files
    for container in (sdata, read_zarr(sdata.path)):
        assert container.attrs == previous_attrs
        with pytest.raises(ValueError, match="must be sorted"):
            hp.qc.summarize_points(container, "calls")


def test_missing_panel_guidance_is_read_only(tmp_path):
    sdata = _sdata(tmp_path)
    original = sdata.points["calls"]
    attrs = deepcopy(sdata.attrs)
    with pytest.raises(ValueError, match=r"hp\.pt\.add_feature_panel.*features_by_class"):
        hp.qc.summarize_points(sdata, "calls")
    assert sdata.points["calls"] is original
    assert sdata.attrs == attrs


def test_registration_supports_custom_keys_and_explicit_single_class(tmp_path):
    sdata = _sdata(tmp_path)
    original = sdata.points["calls"]
    points = original.rename(columns={"gene": "marker"})
    points.attrs.update(original.attrs)
    sdata.points["calls"] = points
    hp.pt.add_feature_panel(
        sdata,
        "calls",
        feature_key="marker",
        feature_class_key="classification",
        features_by_class={"All": ["GeneA", "GeneB", "Negative1", "Undetected"]},
    )
    result = hp.qc.summarize_points(sdata, "calls")
    assert result.per_class["feature_class"].tolist() == ["All"]
    assert result.per_class["n_points"].tolist() == [4]
    assert result.per_target.set_index("feature").loc["Undetected", "n_points"] == 0


@pytest.mark.parametrize(
    "mutation, match",
    [
        ("unknown_feature", "absent from the panel"),
        ("null_feature", "must not be null"),
        ("wrong_class", "expected"),
        ("null_class", "must not be null"),
        ("duplicate", "unique"),
        ("two_classes", "belongs to both"),
        ("string_features", "sequence of feature names"),
        ("empty_panel", "non-empty"),
        ("empty_class", "non-empty"),
        ("empty_feature", "non-empty"),
        ("conflicting_association", "different feature panel"),
    ],
)
def test_invalid_panel_or_source_assignments_fail_without_mutation(tmp_path, mutation, match):
    sdata = _sdata(tmp_path, class_kind="string")
    panel = deepcopy(PANEL)
    points = sdata.points["calls"]
    if mutation == "unknown_feature":
        points = points.assign(gene="NotInPanel")
    elif mutation == "null_feature":
        points = points.assign(gene=None)
    elif mutation == "wrong_class":
        points = points.assign(kind="Negative")
    elif mutation == "null_class":
        points = points.assign(kind=None)
    elif mutation == "duplicate":
        panel["Endogenous"].append("GeneA")
    elif mutation == "two_classes":
        panel["Negative"].append("GeneA")
    elif mutation == "string_features":
        panel["Endogenous"] = "GeneA"
    elif mutation == "empty_panel":
        panel = {}
    elif mutation == "empty_class":
        panel["Negative"] = []
    elif mutation == "empty_feature":
        panel["Negative"].append("")
    else:
        sdata.attrs["harpy"] = {"metadata_version": 1, "points": {"calls": {"feature_panel": "other"}}}
    points.attrs.update(sdata.points["calls"].attrs)
    sdata.points["calls"] = points
    attrs = deepcopy(sdata.attrs)
    with pytest.raises(ValueError, match=match):
        _register(sdata, panel=panel)
    assert sdata.points["calls"] is points
    assert sdata.attrs == attrs


@pytest.mark.parametrize("feature_key, class_key", [("gene", "gene"), ("missing", "kind"), ("gene", "")])
def test_registration_rejects_invalid_source_keys(tmp_path, feature_key, class_key):
    sdata = _sdata(tmp_path)
    with pytest.raises(ValueError):
        hp.pt.add_feature_panel(
            sdata, "calls", feature_key=feature_key, feature_class_key=class_key, features_by_class=PANEL
        )
    assert "harpy" not in sdata.attrs


@pytest.mark.parametrize("feature", [None, "Unknown"])
def test_missing_class_column_does_not_skip_feature_validation(tmp_path, feature):
    sdata = _sdata(tmp_path)
    points = sdata.points["calls"].assign(gene=feature)
    points.attrs.update(sdata.points["calls"].attrs)
    sdata.points["calls"] = points
    with pytest.raises(ValueError, match="must not be null|absent from the panel"):
        _register(sdata)
    assert sdata.points["calls"] is points
    assert "harpy" not in sdata.attrs


def test_compatible_dtype_does_not_skip_class_assignment_validation(tmp_path, monkeypatch):
    sdata = _sdata(tmp_path, class_kind="categorical", backed=True)
    points = sdata.points["calls"]
    points["kind"] = points["kind"].mask(points["gene"] == "GeneB", "Negative")

    def unexpected_write(*args, **kwargs):
        pytest.fail("Invalid source assignments must fail before writing metadata or points")

    monkeypatch.setattr(SpatialData, "write_attrs", unexpected_write)
    monkeypatch.setattr(_feature_panel, "_replace_element_on_disk", unexpected_write)
    with pytest.raises(ValueError, match="expected 'Endogenous'"):
        _register(sdata)
    assert "harpy" not in sdata.attrs


@pytest.mark.parametrize(
    "class_kind, failure",
    [
        ("missing", "staging"),
        ("missing", "metadata"),
        ("missing", "finalize"),
        ("categorical", "metadata"),
        ("categorical", "finalize"),
    ],
)
def test_registration_failure_restores_points_and_root_metadata(tmp_path, monkeypatch, class_kind, failure):
    """Keep panel references and point contents consistent across both persistence paths."""
    sdata = _sdata(tmp_path, class_kind=class_kind, backed=True)
    original = sdata.points["calls"]
    expected = original.compute()
    attrs = deepcopy(sdata.attrs)
    files = _files(sdata.path / "points" / "calls")
    if failure == "staging":
        writer = SpatialData.write

        def fail_staging(self, *args, **kwargs):
            writer(self, *args, **kwargs)
            raise RuntimeError("injected staging failure")

        monkeypatch.setattr(SpatialData, "write", fail_staging)
    else:
        method = "write_attrs" if failure == "metadata" else "write_consolidated_metadata"
        writer = getattr(SpatialData, method)
        failed = False

        def fail_commit(self, *args, **kwargs):
            nonlocal failed
            writer(self, *args, **kwargs)
            if self is sdata and "harpy" in self.attrs and not failed:
                failed = True
                raise RuntimeError("injected metadata/finalization failure")

        monkeypatch.setattr(SpatialData, method, fail_commit)

    with pytest.raises(RuntimeError, match="injected"):
        _register(sdata)
    assert sdata.points["calls"] is original
    reopened = read_zarr(sdata.path)
    for container in (sdata, reopened):
        pd.testing.assert_frame_equal(container.points["calls"].compute(), expected)
        assert container.attrs == attrs
    assert _files(sdata.path / "points" / "calls") == files
    assert not list(tmp_path.glob(".points.zarr.harpy-*"))


def test_registration_rewrite_does_not_collect_source_points(tmp_path, monkeypatch):
    sdata = _sdata(tmp_path, backed=True)
    compute = dd.DataFrame.compute
    normalize = _feature_panel._normalize_feature_classes
    partition_sizes = []

    def guard_compute(self, *args, **kwargs):
        assert not {"x", "y", "gene"} <= set(self.columns), "Do not collect the full points dataframe"
        return compute(self, *args, **kwargs)

    def record_normalization(partition, **kwargs):
        partition_sizes.append(len(partition))
        return normalize(partition, **kwargs)

    monkeypatch.setattr(dd.DataFrame, "compute", guard_compute)
    monkeypatch.setattr(_feature_panel, "_normalize_feature_classes", record_normalization)
    _register(sdata)
    assert partition_sizes and max(partition_sizes) <= 2


def test_identical_panels_are_shared_without_changing_other_points_records(tmp_path):
    sdata = _sdata(tmp_path)
    _register(sdata)
    first = deepcopy(sdata.attrs["harpy"]["points"]["calls"])
    hp.pt.add_feature_panel(sdata, "unrelated", feature_key="gene", feature_class_key="kind", features_by_class=PANEL)
    assert len(sdata.attrs["harpy"]["feature_panels"]) == 1
    assert sdata.attrs["harpy"]["points"]["calls"] == first
    assert sdata.attrs["harpy"]["points"]["unrelated"]["feature_panel"] == first["feature_panel"]


def test_conflicting_panel_contents_reject_hash_reuse(tmp_path):
    sdata = _sdata(tmp_path)
    _register(sdata)
    name = sdata.attrs["harpy"]["points"]["calls"]["feature_panel"]
    sdata.attrs["harpy"]["feature_panels"][name]["features_by_class"]["Endogenous"].append("Corrupt")
    sdata.attrs["harpy"]["feature_panels"][name]["features_by_class"]["Endogenous"].sort()
    attrs = deepcopy(sdata.attrs)
    with pytest.raises(ValueError, match="hash collision"):
        _register(sdata)
    assert sdata.attrs == attrs


def test_registration_rejects_unsaved_points_before_writing_root_metadata(tmp_path):
    sdata = _sdata(tmp_path, class_kind="categorical", backed=True)
    sdata.points["new_points"] = sdata.points["calls"].copy()
    attrs = deepcopy(sdata.attrs)
    with pytest.raises(ValueError, match=r"sdata.write_element\('new_points'\)"):
        hp.pt.add_feature_panel(
            sdata, "new_points", feature_key="gene", feature_class_key="kind", features_by_class=PANEL
        )
    assert sdata.attrs == read_zarr(sdata.path).attrs == attrs


def test_empty_points_can_be_registered_with_an_entirely_undetected_panel(tmp_path):
    sdata = _sdata(tmp_path)
    source = sdata.points["calls"]
    empty = source.map_partitions(lambda frame: frame.iloc[:0], meta=source._meta)
    empty.attrs.update(source.attrs)
    sdata.points["calls"] = empty
    sdata.write(tmp_path / "empty.zarr")
    sdata = read_zarr(sdata.path)
    _register(sdata)
    reopened = read_zarr(sdata.path)
    result = hp.qc.summarize_points(reopened, "calls")
    assert len(result.per_target) == sum(len(features) for features in PANEL.values())
    assert result.per_target.n_points.sum() == 0
    # Empty Parquet partitions need not preserve unused dictionary values;
    # the panel, not observed categorical values, defines the complete classes.
    assert isinstance(reopened.points["calls"].dtypes["kind"], pd.CategoricalDtype)
    assert result.per_class["feature_class"].tolist() == sorted(PANEL)


def test_panel_storage_key_preserves_the_existing_panel_identifier():
    """Pin a pre-existing CosMx-format identifier independently of its implementation."""
    panel = _make_feature_panel(
        feature_key="gene",
        feature_class_key="code_class",
        features_by_class={"Negative": ["Negative01"], "Endogenous": ["GeneB", "GeneA"]},
    )
    assert panel.storage_key == "feature_panel_f9cb6de853758cec"


def test_registered_external_panel_supports_class_aware_aggregation(tmp_path):
    sdata = _sdata(tmp_path)
    sdata.labels["labels"] = Labels2DModel.parse(
        np.ones((2, 4), dtype=np.uint32),
        dims=("y", "x"),
        transformations={"sample": Translation([11, 23], axes=("x", "y"))},
    )
    sdata.write(tmp_path / "aggregation.zarr")
    sdata = read_zarr(sdata.path)
    _register(sdata)
    hp.tb.aggregate_points(
        sdata,
        labels_name="labels",
        points_name="calls",
        output_table_name="aggregated",
        to_coordinate_system="sample",
        expression_class="Endogenous",
    )
    table = sdata.tables["aggregated"]
    assert table.var_names.tolist() == ["GeneA", "GeneB", "ZeroGene"]
    np.testing.assert_array_equal(table.X.to_memory().toarray(), [[1, 2, 0]])
    np.testing.assert_array_equal(table.obsm["auxiliary_feature_counts"].to_memory().toarray(), [[1, 0, 0]])
