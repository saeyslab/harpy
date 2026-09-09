from copy import deepcopy

import dask.array as da
import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import zarr
from anndata import AnnData
from geopandas.testing import assert_geodataframe_equal
from shapely.geometry import box
from spatialdata import SpatialData, read_zarr
from spatialdata._io.format import SpatialDataContainerFormatV01, SpatialDataContainerFormatV02
from spatialdata.models import Image2DModel, Labels2DModel, PointsModel, ShapesModel, TableModel
from spatialdata.transformations import Translation, get_transformation

from harpy._storage import _spatialdata
from harpy.image._image import add_image, add_labels, get_dataarray
from harpy.points._points import add_points
from harpy.shape._shape import add_shapes
from harpy.table._table import add_table


def _make_element(kind):
    transformations = {"sample": Translation([11, 23], axes=("x", "y"))}
    if kind in {"images", "labels"}:
        array = da.arange(64, dtype=np.uint16).reshape(8, 8).rechunk((4, 4))
        if kind == "images":
            return Image2DModel.parse(
                array[None], dims=("c", "y", "x"), c_coords=["DAPI"], transformations=transformations
            )
        return Labels2DModel.parse(array, dims=("y", "x"), transformations=transformations)
    if kind == "points":
        frame = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0], "gene": pd.Categorical(["A", "B"]), "quality": [8, 9]})
        return PointsModel.parse(dd.from_pandas(frame, npartitions=2), transformations=transformations)
    if kind == "shapes":
        return ShapesModel.parse(
            gpd.GeoDataFrame({"quality": [8, 9]}, geometry=[box(0, 0, 1, 1), box(1, 1, 2, 2)]),
            transformations=transformations,
        )
    return TableModel.parse(
        AnnData(
            X=np.array([[1, 2]], dtype=np.uint32),
            obs=pd.DataFrame(index=["cell"]),
            var=pd.DataFrame(index=["A", "B"]),
            uns={"note": "preserve"},
        )
    )


def _backed_sdata(tmp_path, kind="points", zarr_format=3):
    sdata = SpatialData(**{kind: {"element": _make_element(kind)}}, attrs={"keep": {"value": 1}})
    sdata["unrelated"] = _make_element("points")
    formats = SpatialDataContainerFormatV01() if zarr_format == 2 else SpatialDataContainerFormatV02()
    sdata.write(tmp_path / "sdata.zarr", sdata_formats=formats)
    return read_zarr(sdata.path)


def _assert_clean(tmp_path):
    assert not list(tmp_path.glob(".sdata.zarr.harpy-*"))


@pytest.mark.parametrize("kind", ["points", "images"])
def test_replacement_rejects_other_elements_depending_on_destination(tmp_path, monkeypatch, kind):
    sdata = _backed_sdata(tmp_path, kind=kind)
    original = sdata["element"]
    sdata["alias"] = original

    def unexpected_write(*args, **kwargs):
        pytest.fail("Shared-source conflict must fail before staging")

    monkeypatch.setattr(SpatialData, "write", unexpected_write)
    with pytest.raises(ValueError, match="other elements still read.*alias"):
        with _spatialdata._replace_element_on_disk(sdata, "element", original, element_type=kind):
            pass
    assert sdata["element"] is original
    assert sdata["alias"] is original
    _assert_clean(tmp_path)


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("kind", ["images", "labels", "points", "shapes", "tables"])
def test_existing_overwrite_callers_serialize_once_and_reopen_final_paths(tmp_path, monkeypatch, kind, zarr_format):
    """Exercise each public caller, including a lazy replacement reading its original.

    Single-scale rasters become multiscale, so reopening against stale
    consolidated metadata would also fail the round-trip assertions.
    """
    sdata = _backed_sdata(tmp_path, kind, zarr_format)
    unrelated = sdata["unrelated"]
    root_attrs = deepcopy(sdata.attrs)
    original = sdata["element"]
    transformations = None if kind == "tables" else get_transformation(original, get_all=True)
    writes = []
    writer = SpatialData._write_element

    def record_write(self, *args, **kwargs):
        writes.append(kwargs["zarr_container_path"])
        return writer(self, *args, **kwargs)

    monkeypatch.setattr(SpatialData, "_write_element", record_write)
    if kind in {"images", "labels"}:
        expected = get_dataarray(sdata, "element").values + 1
        array = get_dataarray(sdata, "element").data + 1
        if kind == "images":
            result = add_image(
                sdata,
                arr=array,
                output_image_name="element",
                c_coords=["DAPI"],
                scale_factors=[2],
                transformations=transformations,
                overwrite=True,
            )
        else:
            result = add_labels(
                sdata,
                arr=array,
                output_labels_name="element",
                scale_factors=[2],
                transformations=transformations,
                overwrite=True,
            )
    elif kind == "points":
        replacement = original.assign(quality=original.quality + 1)
        expected = replacement.compute()
        result = add_points(
            sdata,
            replacement,
            coordinates={"x": "x", "y": "y"},
            output_points_name="element",
            transformations=transformations,
            overwrite=True,
        )
    elif kind == "shapes":
        expected = original.assign(quality=original.quality + 1)
        result = add_shapes(sdata, input=expected, output_shapes_name="element", overwrite=True)
    else:
        expected = original.copy()
        expected.X += 1
        result = add_table(sdata, adata=expected, output_table_name="element", region=None, overwrite=True)

    assert result is sdata
    assert len(writes) == 1
    assert writes[0] != sdata.path
    assert sdata["unrelated"] is unrelated
    assert sdata.attrs == root_attrs
    _assert_clean(tmp_path)
    assert sdata.is_self_contained()
    reopened = read_zarr(sdata.path)
    for container in (sdata, reopened):
        actual = container["element"]
        if transformations is not None:
            assert get_transformation(actual, get_all=True) == transformations
        if kind in {"images", "labels"}:
            np.testing.assert_array_equal(get_dataarray(container, "element").values, expected)
            assert set(actual.children) == {"scale0", "scale1"}
            if kind == "images":
                assert get_dataarray(container, "element").c.values.tolist() == ["DAPI"]
        elif kind == "points":
            pd.testing.assert_frame_equal(actual.compute(), expected)
        elif kind == "shapes":
            assert_geodataframe_equal(actual, expected)
        else:
            np.testing.assert_array_equal(actual.X, expected.X)
            pd.testing.assert_frame_equal(actual.obs, expected.obs)
            pd.testing.assert_frame_equal(actual.var, expected.var)
            assert actual.uns == expected.uns
    group = zarr.open_group(str(sdata.path), mode="r")
    assert group.metadata.zarr_format == zarr_format
    assert group[kind]["element"].metadata.zarr_format == zarr_format


@pytest.mark.parametrize("failure", ["staging", "reopen", "attach", "consolidate"])
def test_replacement_failure_restores_disk_memory_and_consolidation(tmp_path, monkeypatch, failure):
    sdata = _backed_sdata(tmp_path)
    original = sdata["element"]
    expected = original.compute()
    replacement = original.assign(quality=original.quality + 1)
    replacement.attrs.update(original.attrs)
    if failure == "staging":
        writer = SpatialData.write

        def fail_write(self, *args, **kwargs):
            writer(self, *args, **kwargs)
            raise RuntimeError("injected staging failure")

        monkeypatch.setattr(SpatialData, "write", fail_write)
    elif failure == "reopen":

        def fail_read(*args, **kwargs):
            raise RuntimeError("injected reopen failure")

        monkeypatch.setattr(_spatialdata, "_read_zarr_with_annotating_table_warning_suppressed", fail_read)
    elif failure == "attach":
        setter = SpatialData.__setitem__

        def fail_attach(self, key, value):
            setter(self, key, value)
            if self is sdata:
                raise RuntimeError("injected attach failure")

        monkeypatch.setattr(SpatialData, "__setitem__", fail_attach)
    else:
        consolidate = SpatialData.write_consolidated_metadata
        calls = 0

        def fail_consolidation(self):
            nonlocal calls
            consolidate(self)
            calls += 1
            if self is sdata and calls == 2:
                raise RuntimeError("injected consolidate failure")

        monkeypatch.setattr(SpatialData, "write_consolidated_metadata", fail_consolidation)

    with pytest.raises(RuntimeError, match="injected"):
        with _spatialdata._replace_element_on_disk(sdata, "element", replacement, element_type="points"):
            pass
    assert sdata["element"] is original
    pd.testing.assert_frame_equal(original.compute(), expected)
    pd.testing.assert_frame_equal(read_zarr(sdata.path)["element"].compute(), expected)
    assert sdata.attrs == {"keep": {"value": 1}}
    _assert_clean(tmp_path)


@pytest.mark.parametrize("failure", [None, "metadata", "finalize"])
def test_replacement_context_keeps_metadata_commit_inside_rollback_window(tmp_path, monkeypatch, failure):
    sdata = _backed_sdata(tmp_path)
    original = sdata["element"]
    attrs = deepcopy(sdata.attrs)
    expected = original.compute()
    replacement = original.assign(quality=original.quality + 1)
    replacement.attrs.update(original.attrs)
    body_finished = False
    if failure == "finalize":
        consolidate = SpatialData.write_consolidated_metadata

        def fail_final_consolidation(self):
            consolidate(self)
            if self is sdata and body_finished and self.attrs["keep"]["value"] == 2:
                raise RuntimeError("injected final consolidation failure")

        monkeypatch.setattr(SpatialData, "write_consolidated_metadata", fail_final_consolidation)

    def update():
        nonlocal body_finished
        # The calling operation owns its metadata snapshot and restoration,
        # including failures raised when exiting (not only inside) the context.
        try:
            with _spatialdata._replace_element_on_disk(sdata, "element", replacement, element_type="points"):
                assert list(tmp_path.glob(".sdata.zarr.harpy-replace-backup-*"))
                sdata.attrs["keep"]["value"] = 2
                sdata.write_attrs()
                if failure == "metadata":
                    raise RuntimeError("injected metadata failure")
                body_finished = True
        except BaseException:
            sdata.attrs = attrs
            sdata.write_attrs()
            sdata.write_consolidated_metadata()
            raise

    if failure is not None:
        with pytest.raises(RuntimeError, match="injected"):
            update()
        assert sdata["element"] is original
    else:
        update()
        expected["quality"] += 1
    reopened = read_zarr(sdata.path)
    pd.testing.assert_frame_equal(reopened["element"].compute(), expected)
    assert reopened.attrs == sdata.attrs == {"keep": {"value": 1 if failure is not None else 2}}
    _assert_clean(tmp_path)


def test_replacing_annotating_table_keeps_relation_and_unrelated_slots(tmp_path, recwarn):
    table = TableModel.parse(
        AnnData(
            X=np.array([[1, 2]], dtype=np.uint32),
            obs=pd.DataFrame({"region": pd.Categorical(["labels"]), "cell_ID": [1]}, index=["cell"]),
            var=pd.DataFrame(index=["A", "B"]),
            obsm={"spatial": np.array([[1.0, 2.0]])},
            layers={"counts": np.array([[3, 4]], dtype=np.uint32)},
        ),
        region="labels",
        region_key="region",
        instance_key="cell_ID",
    )
    sdata = SpatialData(labels={"labels": _make_element("labels")}, tables={"table": table})
    sdata.write(tmp_path / "sdata.zarr")
    sdata = read_zarr(sdata.path)
    replacement = sdata["table"].copy()
    replacement.X += 1
    add_table(sdata, adata=replacement, output_table_name="table", region="labels", region_key="region", overwrite=True)
    reopened = read_zarr(sdata.path)["table"]
    np.testing.assert_array_equal(reopened.X, [[2, 3]])
    np.testing.assert_array_equal(reopened.layers["counts"], [[3, 4]])
    np.testing.assert_array_equal(reopened.obsm["spatial"], [[1.0, 2.0]])
    assert reopened.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY] == "labels"
    assert not any("The table is annotating" in str(w.message) for w in recwarn)
    _assert_clean(tmp_path)
