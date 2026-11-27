from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
import tifffile
from spatialdata import SpatialData, read_zarr
from spatialdata.models import PointsModel
from spatialdata.transformations import Identity, get_transformation

from harpy.io._cosmx import (
    _add_instance_labels,
    _add_transcript_points,
    _discover_cosmx,
    _preview_cosmx,
    _transcripts,
)
from harpy.io._cosmx._labels import _remap_instance_ids
from harpy.points import add_feature_panel
from harpy.table._allocation import _assign_points_to_labels
from harpy.transformations._transformations import _resolve_point_to_labels_transform


def test_add_transcript_points_splits_mosaics_and_roundtrips(
    decoded_cosmx_path: Path,
    tmp_path: Path,
) -> None:
    preview = _preview_cosmx(_discover_cosmx(decoded_cosmx_path))
    _write_transcript_csv(
        preview.manifest.fovs_by_id[1].transcripts,
        genes=("GeneB", "GeneA"),
        x=(1.0, 3.5),
        y=(2.0, 4.5),
        cell_comp=("Nuclear", None),
        invalid_ignored_fields=True,
    )
    _write_transcript_csv(
        preview.manifest.fovs_by_id[2].transcripts,
        genes=("GeneC",),
        x=(0.0,),
        y=(1.0,),
        invalid_ignored_fields=True,
    )
    _write_transcript_csv(
        preview.manifest.fovs_by_id[3].transcripts,
        genes=("GeneA",),
        x=(7.0,),
        y=(6.0,),
        cell_comp=("Cytoplasm",),
        include_ignored_fields=False,
    )
    # FOV 4 remains deliberately malformed and proves excluded sources are not read.
    sdata = _backed_sdata(tmp_path)

    _add_transcript_points(sdata, preview, sample_id="sample", blocksize=64)

    assert set(sdata.points) == {"transcripts_mosaic_1", "transcripts_mosaic_2"}
    points_1 = sdata.points["transcripts_mosaic_1"]
    points_2 = sdata.points["transcripts_mosaic_2"]
    assert set(points_1.columns) == {
        "transcript_id",
        "source_compartment",
        "code_class",
        "gene",
        "x",
        "y",
        "source_z",
        "quality",
    }
    assert "CellId" not in points_1.columns
    assert "fov" not in points_1.columns

    # AtoMx pixel corners become pixel centers, x - 0.5, and x is flipped, as 8 - 1 - (x - 0.5).
    values_1 = points_1.compute().reset_index(drop=True)
    np.testing.assert_allclose(sorted(values_1["x"]), [4.0, 6.5, 15.5], atol=1e-5)
    np.testing.assert_allclose(sorted(values_1["y"]), [0.5, 1.5, 4.0], atol=1e-5)
    assert values_1["source_compartment"].isna().sum() == 1
    assert sorted(values_1["quality"]) == [0.5, 0.5, 1.5]

    values_2 = points_2.compute()
    np.testing.assert_allclose(values_2[["x", "y"]], [[0.5, 5.5]], atol=1e-5)
    _assert_transformations(sdata, "transcripts_mosaic_1", mosaic=1)
    _assert_transformations(sdata, "transcripts_mosaic_2", mosaic=2)

    roundtripped = read_zarr(sdata.path)
    assert isinstance(roundtripped.points["transcripts_mosaic_1"], dd.DataFrame)
    roundtripped_values = roundtripped.points["transcripts_mosaic_1"].compute()
    assert isinstance(roundtripped_values["gene"].dtype, pd.CategoricalDtype)
    assert roundtripped_values["gene"].cat.categories.tolist() == ["GeneA", "GeneB", "GeneC"]
    assert roundtripped.attrs["harpy"]["coordinate_systems"]["global_1"] == {
        "sample_id": "sample",
        "mosaic": 1,
        "microns_per_unit": 1.0,
        "fovs": [1, 2],
        "fov_bounds": {
            "1": {"y0": 0, "x0": 0, "y1": 8, "x1": 8},
            "2": {"y0": 0, "x0": 8, "y1": 8, "x1": 16},
        },
        "source_origin_px": {"x": 0, "y": 0},
        "orientation": {"flip_x": True, "flip_y": False},
        "mosaic_grouping": {
            "mode": preview.mosaic_mode,
            "adjacency_tolerance_px": preview.adjacency_tolerance_px,
        },
        "acquisition_timestamp": "20240101_100000_S2",
    }
    metadata = roundtripped.attrs["harpy"]["points"]["transcripts_mosaic_1"]
    panel_name = metadata["feature_panel"]
    assert metadata == {"feature_panel": panel_name}
    assert roundtripped_values["code_class"].cat.categories.tolist() == [
        "Endogenous",
        "Negative",
        "SystemControl",
    ]
    assert roundtripped.attrs["harpy"]["feature_panels"] == {
        panel_name: {
            "feature_key": "gene",
            "feature_class_key": "code_class",
            "classes": ["Endogenous", "Negative", "SystemControl"],
            "features_by_class": {
                "Endogenous": ["Gene1", "Gene2", "Gene3", "GeneA", "GeneB", "GeneC"],
                "Negative": ["Negative01"],
                "SystemControl": ["SystemControl1"],
            },
        }
    }


@pytest.mark.parametrize(
    ("flip_x", "flip_y", "expected"),
    [
        (False, False, (14.0, 23.0)),
        (True, False, (15.0, 23.0)),
        (False, True, (14.0, 24.0)),
        (True, True, (15.0, 24.0)),
    ],
)
def test_transcript_coordinates_follow_raster_orientation(
    flip_x: bool,
    flip_y: bool,
    expected: tuple[float, float],
) -> None:
    frame = _transcript_partition(x=2.0, y=3.0)

    result = _transcripts._normalize_transcript_partition(
        frame,
        tile_origin=(20, 12),
        tile_shape=(8, 6),
        gene_categories=("GeneA",),
        flip_x=flip_x,
        flip_y=flip_y,
        path=Path("transcripts.csv"),
    )

    # The pixel on which the transcript lies, as labels lookups round to it.
    assert (np.rint(result.loc[0, "x"]), np.rint(result.loc[0, "y"])) == expected


@pytest.mark.parametrize(
    ("x", "y"),
    [
        (-0.01, 1.0),
        (6.01, 1.0),
        (1.0, -0.01),
        (1.0, 8.01),
    ],
)
def test_transcript_coordinates_must_be_inside_fov_bounds(x: float, y: float) -> None:
    frame = _transcript_partition(x=x, y=y)

    with pytest.raises(ValueError, match=r"outside FOV bounds 0 <= x <= 6 and 0 <= y <= 8, such as \(x, y\) = "):
        _transcripts._normalize_transcript_partition(
            frame,
            tile_origin=(20, 12),
            tile_shape=(8, 6),
            gene_categories=("GeneA",),
            flip_x=False,
            flip_y=False,
            path=Path("transcripts.csv"),
        )


@pytest.mark.parametrize("flip_y", [False, True])
@pytest.mark.parametrize("flip_x", [False, True])
@pytest.mark.parametrize(
    ("x", "y"),
    [
        # Whole numbers lie on a pixel boundary.
        (0.0, 0.0),
        (2.0, 3.0),
        (5.0, 7.0),
        # Halves, and the last values before a boundary.
        (2.5, 3.5),
        (2.99, 7.99),
        # AtoMx writes points on a FOV's far edge, x == width or y == height.
        (6.0, 1.25),
        (1.25, 8.0),
        (6.0, 8.0),
    ],
)
def test_transcript_coordinates_round_to_the_atomx_pixel(x: float, y: float, flip_x: bool, flip_y: bool) -> None:
    frame = _transcript_partition(x=x, y=y)

    # An odd and a large origin: the conversion must survive the float64 spacing at mosaic coordinates.
    result = _transcripts._normalize_transcript_partition(
        frame,
        tile_origin=(100_001, 13),
        tile_shape=(8, 6),
        gene_categories=("GeneA",),
        flip_x=flip_x,
        flip_y=flip_y,
        path=Path("transcripts.csv"),
    )

    assert np.rint(result.loc[0, "x"]) == 13 + _atomx_pixel(x, 6, flip=flip_x)
    assert np.rint(result.loc[0, "y"]) == 100_001 + _atomx_pixel(y, 8, flip=flip_y)


@pytest.mark.parametrize("flip_y", [False, True])
@pytest.mark.parametrize("flip_x", [False, True])
def test_transcripts_are_counted_in_the_cell_atomx_assigns_them_to(
    decoded_cosmx_path: Path,
    tmp_path: Path,
    flip_x: bool,
    flip_y: bool,
) -> None:
    # One cell per pixel, so that a transcript a pixel off lands in another cell.
    source_labels = np.arange(1, 8 * 8 + 1, dtype=np.uint16).reshape(8, 8)
    for fov in (1, 2):
        tifffile.imwrite(
            decoded_cosmx_path / "CellStatsDir" / f"FOV{fov:05d}" / f"CellLabels_F{fov:05d}.tif",
            source_labels,
            metadata=None,
        )
    preview = _preview_cosmx(_discover_cosmx(decoded_cosmx_path), fovs=(1, 2))
    # Whole numbers, halves, the origin and the far edges, in FOVs 1 and 2, adjacent 8 x 8 tiles of one mosaic.
    x = (0.0, 3.0, 2.5, 4.99, 8.0, 1.5, 8.0, 7.99)
    y = (0.0, 5.0, 6.5, 1.99, 3.25, 8.0, 8.0, 0.01)
    for fov in (1, 2):
        _write_transcript_csv(
            preview.manifest.fovs_by_id[fov].transcripts,
            genes=("GeneA",) * len(x),
            x=x,
            y=y,
            first_transcript_id=100 * fov,
        )
    sdata = _backed_sdata(tmp_path)

    _add_instance_labels(sdata, preview, sample_id="sample", chunks=(4, 4), flip_x=flip_x, flip_y=flip_y)
    _add_transcript_points(sdata, preview, sample_id="sample", blocksize=64, flip_x=flip_x, flip_y=flip_y)

    labels = sdata.labels["instance_labels_mosaic_1"]
    points = sdata.points["transcripts_mosaic_1"]
    assigned = _assign_points_to_labels(
        labels,
        points,
        _resolve_point_to_labels_transform(points, labels, to_coordinate_system="global_1"),
        value_key="transcript_id",
        instance_key="cell_ID",
    ).compute()
    # AtoMx assigns a transcript to the cell at its floor pixel, or the last pixel on the far edge.
    expected = {
        100 * fov + index: int(
            _remap_instance_ids(source_labels, fov=fov)[
                _atomx_pixel(y_, 8, flip=False), _atomx_pixel(x_, 8, flip=False)
            ]
        )
        for fov in (1, 2)
        for index, (x_, y_) in enumerate(zip(x, y, strict=True))
    }
    assert dict(zip(assigned["transcript_id"], assigned["cell_ID"].astype(int), strict=True)) == expected


def test_transcript_preflight_rejects_missing_retained_column_without_writing(
    decoded_cosmx_path: Path,
    tmp_path: Path,
) -> None:
    preview = _preview_cosmx(_discover_cosmx(decoded_cosmx_path), fovs=(1,))
    path = preview.manifest.fovs_by_id[1].transcripts
    assert path is not None
    path.write_text("V1,CellComp,codeclass,target,x,y\n1,Nuclear,Endogenous,GeneA,1,2\n")
    sdata = _backed_sdata(tmp_path)

    with pytest.raises(ValueError, match=r"missing required columns \['z'\]"):
        _add_transcript_points(sdata, preview, sample_id="sample")

    assert not sdata.points
    assert not read_zarr(sdata.path).points


@pytest.mark.parametrize("blocksize", [0, -1, True, "nonsense"])
def test_transcript_ingestion_rejects_invalid_blocksize(
    decoded_cosmx_path: Path,
    tmp_path: Path,
    blocksize: object,
) -> None:
    preview = _preview_cosmx(_discover_cosmx(decoded_cosmx_path), fovs=(1,))
    sdata = _backed_sdata(tmp_path)

    with pytest.raises(ValueError, match="blocksize"):
        _add_transcript_points(sdata, preview, sample_id="sample", blocksize=blocksize)  # type: ignore[arg-type]


def test_transcript_ingestion_requires_backed_spatialdata(decoded_cosmx_path: Path) -> None:
    preview = _preview_cosmx(_discover_cosmx(decoded_cosmx_path), fovs=(1,))

    with pytest.raises(ValueError, match="requires a backed SpatialData"):
        _add_transcript_points(SpatialData(), preview, sample_id="sample")


def test_transcript_materialization_rejects_nonfinite_coordinates(
    decoded_cosmx_path: Path,
    tmp_path: Path,
) -> None:
    preview = _preview_cosmx(_discover_cosmx(decoded_cosmx_path), fovs=(1,))
    _write_transcript_csv(
        preview.manifest.fovs_by_id[1].transcripts,
        genes=("GeneA",),
        x=(np.inf,),
        y=(1.0,),
    )
    sdata = _backed_sdata(tmp_path)

    with pytest.raises(ValueError, match="non-finite x or y"):
        _add_transcript_points(sdata, preview, sample_id="sample")

    assert "transcripts_mosaic_1" not in read_zarr(sdata.path).points


def test_transcript_ingestion_without_plex_keeps_raw_classes(
    decoded_cosmx_path: Path,
    tmp_path: Path,
) -> None:
    (decoded_cosmx_path / "plex-analysis.txt").unlink()
    preview = _preview_cosmx(_discover_cosmx(decoded_cosmx_path), fovs=(1,))
    _write_transcript_csv(
        preview.manifest.fovs_by_id[1].transcripts,
        genes=("GeneA",),
        x=(1.0,),
        y=(2.0,),
    )
    sdata = _backed_sdata(tmp_path)

    _add_transcript_points(sdata, preview, sample_id="sample")

    roundtripped = read_zarr(sdata.path)
    values = roundtripped.points["transcripts_mosaic_1"].compute()
    assert not isinstance(values["code_class"].dtype, pd.CategoricalDtype)
    # Without a panel, the points have no fields of their own, and so no record; their mosaic is recorded.
    assert "points" not in roundtripped.attrs["harpy"]
    assert "feature_panels" not in roundtripped.attrs["harpy"]
    assert roundtripped.attrs["harpy"]["coordinate_systems"]["global_1"]["sample_id"] == "sample"


def test_transcript_materialization_rejects_panel_disagreement(
    decoded_cosmx_path: Path,
    tmp_path: Path,
) -> None:
    preview = _preview_cosmx(_discover_cosmx(decoded_cosmx_path), fovs=(1,))
    _write_transcript_csv(
        preview.manifest.fovs_by_id[1].transcripts,
        genes=("GeneA",),
        code_classes=("Negative",),
        x=(1.0,),
        y=(2.0,),
    )
    sdata = _backed_sdata(tmp_path)

    with pytest.raises(ValueError, match="disagree with the feature panel"):
        _add_transcript_points(sdata, preview, sample_id="sample")

    assert "transcripts_mosaic_1" not in read_zarr(sdata.path).points


def test_transcript_preflight_rejects_malformed_stored_panel(
    decoded_cosmx_path: Path,
    tmp_path: Path,
) -> None:
    preview = _preview_cosmx(_discover_cosmx(decoded_cosmx_path), fovs=(1,))
    _write_transcript_csv(
        preview.manifest.fovs_by_id[1].transcripts,
        genes=("GeneA",),
        x=(1.0,),
        y=(2.0,),
    )
    sdata = _backed_sdata(tmp_path)
    panel = preview.manifest.feature_panel
    assert panel is not None
    panel_name = panel.storage_key
    sdata.attrs = {
        "harpy": {
            "metadata_version": 1,
            "feature_panels": {panel_name: {"feature_key": "incompatible"}},
        }
    }
    sdata.write_attrs()

    with pytest.raises(ValueError, match=r"harpy.feature_panels.*feature_class_key.*non-empty"):
        _add_transcript_points(sdata, preview, sample_id="sample")

    assert not read_zarr(sdata.path).points


def test_transcript_ingestion_rejects_unsorted_stored_panel(decoded_cosmx_path, tmp_path):
    """Invalid persisted ordering fails before points or panel metadata are written."""
    preview = _preview_cosmx(_discover_cosmx(decoded_cosmx_path), fovs=(1,))
    _write_transcript_csv(preview.manifest.fovs_by_id[1].transcripts, genes=("GeneA",), x=(1.0,), y=(2.0,))
    panel = preview.manifest.feature_panel
    assert panel is not None
    record = panel.to_dict()
    record["classes"].reverse()
    sdata = _backed_sdata(tmp_path)
    sdata.attrs["harpy"] = {"metadata_version": 1, "feature_panels": {panel.storage_key: deepcopy(record)}}
    sdata.write_attrs()
    previous_attrs = deepcopy(sdata.attrs)

    with pytest.raises(ValueError, match="must be sorted"):
        _add_transcript_points(sdata, preview, sample_id="sample")

    reopened = read_zarr(sdata.path)
    assert sdata.attrs == reopened.attrs == previous_attrs
    assert not reopened.points


def test_cosmx_and_existing_points_share_identical_panel_metadata(decoded_cosmx_path, tmp_path):
    """Generic registration reuses a reader's panel without changing its key or source records."""
    preview = _preview_cosmx(_discover_cosmx(decoded_cosmx_path), fovs=(1,))
    _write_transcript_csv(preview.manifest.fovs_by_id[1].transcripts, genes=("GeneA",), x=(1.0,), y=(2.0,))
    sdata = _add_transcript_points(_backed_sdata(tmp_path), preview, sample_id="sample")
    previous_attrs = deepcopy(sdata.attrs)
    panel_name = sdata.attrs["harpy"]["points"]["transcripts_mosaic_1"]["feature_panel"]
    panel_record = sdata.attrs["harpy"]["feature_panels"][panel_name]
    sdata.points["external"] = PointsModel.parse(
        pd.DataFrame({"x": [1.0], "y": [2.0], "gene": ["GeneA"]}), transformations={"global": Identity()}
    )
    sdata.write_element("external")
    add_feature_panel(
        sdata,
        "external",
        feature_key="gene",
        feature_class_key="code_class",
        features_by_class={
            key: list(reversed(values)) for key, values in reversed(panel_record["features_by_class"].items())
        },
    )
    reopened = read_zarr(sdata.path)
    assert reopened.attrs["harpy"]["feature_panels"] == previous_attrs["harpy"]["feature_panels"]
    assert (
        reopened.attrs["harpy"]["points"]["transcripts_mosaic_1"]
        == previous_attrs["harpy"]["points"]["transcripts_mosaic_1"]
    )
    assert reopened.attrs["harpy"]["points"]["external"]["feature_panel"] == panel_name
    assert reopened.points["external"].compute().code_class.tolist() == ["Endogenous"]


def _write_transcript_csv(
    path: Path | None,
    *,
    genes: tuple[str, ...],
    x: tuple[float, ...],
    y: tuple[float, ...],
    code_classes: tuple[str, ...] | None = None,
    cell_comp: tuple[str | None, ...] | None = None,
    invalid_ignored_fields: bool = False,
    include_ignored_fields: bool = True,
    first_transcript_id: int = 1,
) -> None:
    assert path is not None
    count = len(genes)
    values: dict[str, object] = {
        "V1": np.arange(first_transcript_id, first_transcript_id + count),
        "CellComp": cell_comp if cell_comp is not None else ("Nuclear",) * count,
        "codeclass": code_classes if code_classes is not None else ("Endogenous",) * count,
        "target": genes,
        "x": x,
        "y": y,
        "z": np.zeros(count, dtype=int),
        "quality": np.arange(count, dtype=float) + 0.5,
    }
    if include_ignored_fields:
        ignored = ("invalid",) * count if invalid_ignored_fields else np.arange(count)
        values["CellId"] = ignored
        values["fov"] = ignored
    pd.DataFrame(values).to_csv(path, index=False)


def _atomx_pixel(source: float, size: int, *, flip: bool) -> int:
    """Return the pixel AtoMx assigns a FOV-local coordinate to, flipped by index like the raster tile."""
    pixel = size - 1 if source == size else int(np.floor(source))
    return size - 1 - pixel if flip else pixel


def _transcript_partition(*, x: float, y: float) -> pd.DataFrame:
    """Create the smallest valid source partition for coordinate-focused tests."""
    return pd.DataFrame(
        {
            "V1": [1],
            "CellComp": ["Nuclear"],
            "codeclass": ["Endogenous"],
            "target": ["GeneA"],
            "x": [x],
            "y": [y],
            "z": [0],
        }
    )


def _backed_sdata(tmp_path: Path) -> SpatialData:
    path = tmp_path / "cosmx.zarr"
    SpatialData().write(path)
    return read_zarr(path)


def _assert_transformations(sdata: SpatialData, points_name: str, *, mosaic: int) -> None:
    # Only the mosaic's pixel coordinate system; no micron coordinate system.
    assert get_transformation(sdata.points[points_name], get_all=True) == {f"global_{mosaic}": Identity()}
