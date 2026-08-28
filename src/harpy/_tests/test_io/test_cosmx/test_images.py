from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import tifffile
from spatialdata import SpatialData, read_zarr
from spatialdata.models import Labels2DModel
from spatialdata.transformations import Identity, Scale, get_transformation
from xarray import DataArray

from harpy.image._image import get_dataarray
from harpy.io._cosmx import _add_morphology_images, _discover_cosmx, _images, _preview_cosmx

_TILE_SHAPE = (8, 8)


def test_add_morphology_images_stitches_groups_and_roundtrips(
    decoded_cosmx_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preview = _valued_preview(decoded_cosmx_path, fov_2_x_px=9)
    sdata = _backed_sdata(tmp_path)
    reads = _instrument_plane_reads(monkeypatch)

    result = _add_morphology_images(
        sdata,
        preview,
        sample_id="sample",
        channels=("U", "Histone"),
        chunks=(1, 4, 4),
    )

    assert result is sdata
    assert set(sdata.images) == {"morphology_image_mosaic_1", "morphology_image_mosaic_2"}
    assert len(reads) == 6
    assert sorted(plane for _, plane in reads) == [0, 0, 0, 4, 4, 4]

    mosaic_1 = get_dataarray(sdata, "morphology_image_mosaic_1")
    assert isinstance(sdata.images["morphology_image_mosaic_1"], DataArray)
    assert mosaic_1.dims == ("c", "y", "x")
    assert mosaic_1.coords["c"].values.tolist() == ["Histone", "DNA"]
    assert mosaic_1.dtype == np.dtype(np.uint16)
    values = mosaic_1.data.compute()
    np.testing.assert_array_equal(values[:, :, :8], _oriented_tile(1, planes=(0, 4)))
    np.testing.assert_array_equal(values[:, :, 8], 0)
    np.testing.assert_array_equal(values[:, :, 9:], _oriented_tile(2, planes=(0, 4)))

    mosaic_2 = get_dataarray(sdata, "morphology_image_mosaic_2")
    np.testing.assert_array_equal(mosaic_2.data.compute(), _oriented_tile(3, planes=(0, 4)))
    _assert_transformations(sdata, "morphology_image_mosaic_1", mosaic=1)
    _assert_transformations(sdata, "morphology_image_mosaic_2", mosaic=2)

    roundtripped = read_zarr(sdata.path)
    np.testing.assert_array_equal(
        get_dataarray(roundtripped, "morphology_image_mosaic_1").data.compute(),
        values,
    )
    metadata = roundtripped.attrs["harpy"]["images"]["morphology_image_mosaic_1"]
    assert metadata == {
        "fovs": [1, 2],
        "sample_id": "sample",
        "mosaic": {
            "mode": preview.mosaic_mode,
            "adjacency_tolerance_px": preview.adjacency_tolerance_px,
        },
        "source_origin_px": {"x": 0, "y": 0},
        "orientation": {"flip_x": True, "flip_y": False},
        "pixel_size_um": 1.0,
        "acquisition_timestamp": "20240101_100000_S2",
        "channels": [
            {
                "channel_id": "B",
                "name": "Histone",
                "source_plane": 0,
                "output_coordinate": "Histone",
            },
            {"channel_id": "U", "name": "DNA", "source_plane": 4, "output_coordinate": "DNA"},
        ],
    }


def test_morphology_mosaic_does_not_read_pixels_during_construction(
    decoded_cosmx_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preview = _valued_preview(decoded_cosmx_path, fov_2_x_px=9)
    mosaic = preview.mosaics[0]
    reads = _instrument_plane_reads(monkeypatch)
    selected = _images._select_channels(preview, ("B",))
    placements = _images._mosaic_placements(preview, mosaic)

    array = _images._morphology_mosaic(
        preview,
        mosaic,
        placements=placements,
        channels=selected,
        flip_x=True,
        flip_y=False,
        chunks=(1, 4, 4),
    )

    assert array.shape == (1, 8, 17)
    assert reads == []


def test_channel_selection_is_deterministic_and_strict(decoded_cosmx_path: Path) -> None:
    preview = _preview_cosmx(_discover_cosmx(decoded_cosmx_path))

    selected = _images._select_channels(preview, ("U", "Histone"))
    assert [(channel.channel_id, channel.plane) for channel in selected] == [("B", 0), ("U", 4)]

    with pytest.raises(ValueError, match="Unknown CosMx morphology channel"):
        _images._select_channels(preview, ("missing",))
    with pytest.raises(ValueError, match="selects a channel more than once"):
        _images._select_channels(preview, ("B", "Histone"))

    duplicated_name_channels = tuple(
        replace(channel, name="shared") if channel.channel_id in {"B", "G"} else channel
        for channel in preview.manifest.run.channels
    )
    run = replace(preview.manifest.run, channels=duplicated_name_channels)
    manifest = replace(preview.manifest, run=run)
    ambiguous_preview = replace(preview, manifest=manifest)
    with pytest.raises(ValueError, match="is ambiguous"):
        _images._select_channels(ambiguous_preview, ("shared",))

    disambiguated = _images._select_channels(ambiguous_preview, ("B", "G"))
    assert [channel.output_coordinate for channel in disambiguated] == ["shared [B]", "shared [G]"]


@pytest.mark.parametrize(
    ("flip_x", "flip_y", "expected_index"),
    [
        (False, False, (slice(None), slice(None))),
        (True, False, (slice(None), slice(None, None, -1))),
        (False, True, (slice(None, None, -1), slice(None))),
        (True, True, (slice(None, None, -1), slice(None, None, -1))),
    ],
)
def test_read_morphology_plane_supports_dataset_wide_axis_flips(
    decoded_cosmx_path: Path,
    flip_x: bool,
    flip_y: bool,
    expected_index: tuple[slice, slice],
) -> None:
    preview = _valued_preview(decoded_cosmx_path)
    source = preview.manifest.fovs_by_id[1].morphology
    assert source is not None

    result = _images._read_morphology_plane(
        source,
        0,
        _TILE_SHAPE,
        np.dtype(np.uint16).name,
        flip_x=flip_x,
        flip_y=flip_y,
    )

    np.testing.assert_array_equal(result, _tile(1, planes=(0,))[0][expected_index])


def test_add_morphology_images_forwards_explicit_orientation(
    decoded_cosmx_path: Path,
    tmp_path: Path,
) -> None:
    preview = _valued_preview(decoded_cosmx_path)
    sdata = _backed_sdata(tmp_path)

    _add_morphology_images(
        sdata,
        preview,
        sample_id="sample",
        channels=("B",),
        flip_x=False,
        flip_y=True,
        chunks=(1, 4, 4),
    )

    values = get_dataarray(sdata, "morphology_image_mosaic_1").data.compute()
    np.testing.assert_array_equal(values[:, :, :8], _tile(1, planes=(0,))[:, ::-1, :])
    metadata = sdata.attrs["harpy"]["images"]["morphology_image_mosaic_1"]
    assert metadata["orientation"] == {"flip_x": False, "flip_y": True}


@pytest.mark.parametrize("overwrite", [False, True])
def test_morphology_rejects_non_image_name_collision_before_pixel_reads(
    decoded_cosmx_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overwrite: bool,
) -> None:
    preview = _valued_preview(decoded_cosmx_path)
    labels = Labels2DModel.parse(np.zeros(_TILE_SHAPE, dtype=np.uint8), dims=("y", "x"))
    path = tmp_path / "cosmx.zarr"
    SpatialData(labels={"morphology_image_mosaic_1": labels}).write(path)
    sdata = read_zarr(path)
    reads = _instrument_plane_reads(monkeypatch)

    with pytest.raises(ValueError, match="output names already belong to non-image elements"):
        _add_morphology_images(
            sdata,
            preview,
            sample_id="sample",
            channels=("B",),
            chunks=(1, 4, 4),
            overwrite=overwrite,
        )

    assert not reads
    assert "morphology_image_mosaic_1" in read_zarr(path).labels


def test_multiscale_morphology_reads_each_source_plane_once(
    decoded_cosmx_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enforce one TIFF decode per selected FOV/channel even with multiscales."""
    preview = _valued_preview(decoded_cosmx_path)
    sdata = _backed_sdata(tmp_path)
    reads = _instrument_plane_reads(monkeypatch)

    _add_morphology_images(
        sdata,
        preview,
        sample_id="sample",
        channels=("B",),
        chunks=(1, 4, 4),
        scale_factors=[2],
    )

    expected_reads = {(preview.manifest.fovs_by_id[fov].morphology, 0) for fov in preview.included_fovs}
    assert set(reads) == expected_reads
    assert len(reads) == len(expected_reads)


def test_changed_source_metadata_fails_write_without_leaving_an_element(
    decoded_cosmx_path: Path,
    tmp_path: Path,
) -> None:
    preview = _valued_preview(decoded_cosmx_path)
    source = preview.manifest.fovs_by_id[1].morphology
    assert source is not None
    with tifffile.TiffFile(source) as tif:
        description = tif.pages[0].description
    tifffile.imwrite(
        source,
        np.zeros((5, *_TILE_SHAPE), dtype=np.uint8),
        description=description,
        metadata=None,
        photometric="minisblack",
    )
    sdata = _backed_sdata(tmp_path)

    with pytest.raises(ValueError, match="changed dtype after discovery"):
        _add_morphology_images(sdata, preview, sample_id="sample", channels=("B",), chunks=(1, 4, 4))

    assert "morphology_image_mosaic_1" not in sdata.images
    assert "morphology_image_mosaic_1" not in read_zarr(sdata.path).images


def _valued_preview(decoded_cosmx_path: Path, *, fov_2_x_px: int = 8):
    manifest = _discover_cosmx(decoded_cosmx_path)
    for files in manifest.fovs:
        if files.morphology is not None:
            _rewrite_morphology_values(files.morphology, fov=files.fov)
    if fov_2_x_px != 8:
        positions = tuple(
            replace(position, x_px=fov_2_x_px) if position.fov == 2 else position for position in manifest.positions
        )
        manifest = replace(manifest, positions=positions)
    return _preview_cosmx(manifest)


def _rewrite_morphology_values(path: Path, *, fov: int) -> None:
    with tifffile.TiffFile(path) as tif:
        description = tif.pages[0].description
    data = _tile(fov, planes=tuple(range(5)))
    tifffile.imwrite(path, data, description=description, metadata=None, photometric="minisblack")


def _tile(fov: int, *, planes: tuple[int, ...]) -> np.ndarray:
    pixels = np.arange(np.prod(_TILE_SHAPE), dtype=np.uint16).reshape(_TILE_SHAPE)
    return np.stack([fov * 1000 + plane * 100 + pixels for plane in planes]).astype(np.uint16)


def _oriented_tile(fov: int, *, planes: tuple[int, ...]) -> np.ndarray:
    return _tile(fov, planes=planes)[:, :, ::-1]


def _backed_sdata(tmp_path: Path) -> SpatialData:
    path = tmp_path / "cosmx.zarr"
    SpatialData().write(path)
    return read_zarr(path)


def _instrument_plane_reads(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Path, int]]:
    original = _images._read_morphology_plane
    reads: list[tuple[Path, int]] = []

    def instrumented(
        path: Path,
        plane: int,
        expected_shape: tuple[int, int],
        expected_dtype: str,
        *,
        flip_x: bool,
        flip_y: bool,
    ) -> np.ndarray:
        reads.append((path, plane))
        return original(
            path,
            plane,
            expected_shape,
            expected_dtype,
            flip_x=flip_x,
            flip_y=flip_y,
        )

    monkeypatch.setattr(_images, "_read_morphology_plane", instrumented)
    return reads


def _assert_transformations(sdata: SpatialData, image_name: str, *, mosaic: int) -> None:
    transformations = get_transformation(sdata.images[image_name], get_all=True)
    assert set(transformations) == {f"global_{mosaic}", f"global_{mosaic}_micron"}
    assert transformations[f"global_{mosaic}"] == Identity()
    expected = Scale([1.0, 1.0], axes=("x", "y")).to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
    actual = transformations[f"global_{mosaic}_micron"].to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
    np.testing.assert_array_equal(actual, expected)
