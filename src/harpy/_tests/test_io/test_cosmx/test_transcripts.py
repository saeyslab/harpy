from __future__ import annotations

from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from spatialdata import SpatialData, read_zarr
from spatialdata.transformations import Identity, Scale, get_transformation

from harpy.io._cosmx import _add_transcript_points, _discover_cosmx, _preview_cosmx, _transcripts


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

    _add_transcript_points(sdata, preview, blocksize=64)

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

    values_1 = points_1.compute().reset_index(drop=True)
    np.testing.assert_allclose(sorted(values_1["x"]), [3.5, 6.0, 15.0])
    np.testing.assert_allclose(sorted(values_1["y"]), [1.0, 2.0, 4.5])
    assert values_1["source_compartment"].isna().sum() == 1
    assert sorted(values_1["quality"]) == [0.5, 0.5, 1.5]

    values_2 = points_2.compute()
    np.testing.assert_allclose(values_2[["x", "y"]], [[0.0, 6.0]])
    _assert_transformations(sdata, "transcripts_mosaic_1", mosaic=1)
    _assert_transformations(sdata, "transcripts_mosaic_2", mosaic=2)

    roundtripped = read_zarr(sdata.path)
    assert isinstance(roundtripped.points["transcripts_mosaic_1"], dd.DataFrame)
    roundtripped_values = roundtripped.points["transcripts_mosaic_1"].compute()
    assert isinstance(roundtripped_values["gene"].dtype, pd.CategoricalDtype)
    assert roundtripped_values["gene"].cat.categories.tolist() == ["GeneA", "GeneB", "GeneC"]
    metadata = roundtripped.attrs["cosmx"]["transcripts"]["transcripts_mosaic_1"]
    assert metadata == {
        "fovs": [1, 2],
        "source_origin_px": {"x": 0, "y": 0},
        "orientation": {"flip_x": True, "flip_y": False},
        "pixel_size_um": 1.0,
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
        placement=(20, 12),
        tile_shape=(8, 6),
        gene_categories=("GeneA",),
        flip_x=flip_x,
        flip_y=flip_y,
        path=Path("transcripts.csv"),
    )

    assert (result.loc[0, "x"], result.loc[0, "y"]) == expected


@pytest.mark.parametrize(
    ("x", "y"),
    [
        (-0.01, 1.0),
        (6.0, 1.0),
        (1.0, -0.01),
        (1.0, 8.0),
    ],
)
def test_transcript_coordinates_must_be_inside_fov_bounds(x: float, y: float) -> None:
    frame = _transcript_partition(x=x, y=y)

    with pytest.raises(ValueError, match=r"outside FOV bounds 0 <= x < 6 and 0 <= y < 8"):
        _transcripts._normalize_transcript_partition(
            frame,
            placement=(20, 12),
            tile_shape=(8, 6),
            gene_categories=("GeneA",),
            flip_x=False,
            flip_y=False,
            path=Path("transcripts.csv"),
        )


def test_transcript_preflight_rejects_missing_retained_column_without_writing(
    decoded_cosmx_path: Path,
    tmp_path: Path,
) -> None:
    preview = _preview_cosmx(_discover_cosmx(decoded_cosmx_path), fovs=(1,))
    path = preview.manifest.fovs_by_id[1].transcripts
    assert path is not None
    path.write_text("V1,CellComp,codeclass,target,x,y\n1,Nuclear,Call,GeneA,1,2\n")
    sdata = _backed_sdata(tmp_path)

    with pytest.raises(ValueError, match=r"missing required columns \['z'\]"):
        _add_transcript_points(sdata, preview)

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
        _add_transcript_points(sdata, preview, blocksize=blocksize)  # type: ignore[arg-type]


def test_transcript_ingestion_requires_backed_spatialdata(decoded_cosmx_path: Path) -> None:
    preview = _preview_cosmx(_discover_cosmx(decoded_cosmx_path), fovs=(1,))

    with pytest.raises(ValueError, match="requires a backed SpatialData"):
        _add_transcript_points(SpatialData(), preview)


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
        _add_transcript_points(sdata, preview)

    assert "transcripts_mosaic_1" not in read_zarr(sdata.path).points


def _write_transcript_csv(
    path: Path | None,
    *,
    genes: tuple[str, ...],
    x: tuple[float, ...],
    y: tuple[float, ...],
    cell_comp: tuple[str | None, ...] | None = None,
    invalid_ignored_fields: bool = False,
    include_ignored_fields: bool = True,
) -> None:
    assert path is not None
    count = len(genes)
    values: dict[str, object] = {
        "V1": np.arange(1, count + 1),
        "CellComp": cell_comp if cell_comp is not None else ("Nuclear",) * count,
        "codeclass": ("Call",) * count,
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


def _transcript_partition(*, x: float, y: float) -> pd.DataFrame:
    """Create the smallest valid source partition for coordinate-focused tests."""
    return pd.DataFrame(
        {
            "V1": [1],
            "CellComp": ["Nuclear"],
            "codeclass": ["Call"],
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
    transformations = get_transformation(sdata.points[points_name], get_all=True)
    assert set(transformations) == {f"global_{mosaic}", f"global_{mosaic}_micron"}
    assert transformations[f"global_{mosaic}"] == Identity()
    expected = Scale([1.0, 1.0], axes=("x", "y")).to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
    actual = transformations[f"global_{mosaic}_micron"].to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
    np.testing.assert_array_equal(actual, expected)
