from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import tifffile
from spatialdata import SpatialData, read_zarr
from spatialdata.models import Image2DModel
from spatialdata.transformations import Identity, Scale, get_transformation

from harpy.image._image import get_dataarray
from harpy.io._cosmx import (
    _add_compartment_labels,
    _add_instance_labels,
    _discover_cosmx,
    _labels,
    _preview_cosmx,
)

_TILE_SHAPE = (8, 8)


def test_add_instance_labels_remaps_stitches_and_roundtrips(
    decoded_cosmx_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preview = _labelled_preview(decoded_cosmx_path, fov_2_x_px=9)
    sdata = _backed_sdata(tmp_path)
    reads = _instrument_label_reads(monkeypatch)

    result = _add_instance_labels(
        sdata,
        preview,
        sample_id="sample",
        chunks=(4, 4),
        scale_factors=[2],
    )

    assert result is sdata
    assert set(sdata.labels) == {"instance_labels_mosaic_1", "instance_labels_mosaic_2"}
    assert len(reads) == len(preview.included_fovs)
    assert {path for path, family in reads if family == "instance_labels"} == {
        preview.manifest.fovs_by_id[fov].instance_labels for fov in preview.included_fovs
    }

    mosaic_1 = get_dataarray(sdata, "instance_labels_mosaic_1")
    values = mosaic_1.data.compute()
    assert mosaic_1.dims == ("y", "x")
    assert mosaic_1.dtype == np.dtype(np.uint32)
    np.testing.assert_array_equal(values[:, :8], _expected_instance_tile(1))
    np.testing.assert_array_equal(values[:, 8], 0)
    np.testing.assert_array_equal(values[:, 9:], _expected_instance_tile(2))
    assert values[4, 7] == 7
    assert values[4, 9] == 65_543

    mosaic_2 = get_dataarray(sdata, "instance_labels_mosaic_2")
    np.testing.assert_array_equal(mosaic_2.data.compute(), _expected_instance_tile(3))
    _assert_transformations(sdata, "instance_labels_mosaic_1", mosaic=1)

    roundtripped = read_zarr(sdata.path)
    np.testing.assert_array_equal(
        get_dataarray(roundtripped, "instance_labels_mosaic_1").data.compute(),
        values,
    )
    assert roundtripped.attrs["harpy"]["labels"]["instance_labels_mosaic_1"] == {
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
        "instance_id_encoding": {
            "background": 0,
            "base": 65_536,
            "formula": "global_id = (fov - 1) * base + local_id",
        },
    }


def test_instance_id_remapping_is_stable_by_original_fov() -> None:
    values = np.array([[0, 1, 65_535]], dtype=np.uint16)

    remapped = _labels._remap_instance_ids(values, fov=2)

    np.testing.assert_array_equal(remapped, np.array([[0, 65_537, 131_071]], dtype=np.uint32))


def test_add_compartment_labels_preserves_semantic_values(
    decoded_cosmx_path: Path,
    tmp_path: Path,
) -> None:
    preview = _labelled_preview(decoded_cosmx_path)
    sdata = _backed_sdata(tmp_path)

    _add_compartment_labels(sdata, preview, sample_id="sample", chunks=(4, 4))

    values = get_dataarray(sdata, "compartment_labels_mosaic_1").data.compute()
    assert values.dtype == np.dtype(np.uint8)
    np.testing.assert_array_equal(values[:, :8], _compartment_tile(1)[:, ::-1])
    np.testing.assert_array_equal(values[:, 8:], _compartment_tile(2)[:, ::-1])

    roundtripped = read_zarr(sdata.path)
    categories = roundtripped.attrs["harpy"]["labels"]["compartment_labels_mosaic_1"]["categories"]
    assert {int(value): name for value, name in categories.items()} == {
        0: "background",
        1: "nuclear",
        2: "membrane",
        3: "cytoplasmic",
    }


def test_unsupported_compartment_value_fails_without_leaving_an_element(
    decoded_cosmx_path: Path,
    tmp_path: Path,
) -> None:
    preview = _labelled_preview(decoded_cosmx_path)
    source = preview.manifest.fovs_by_id[1].compartment_labels
    assert source is not None
    values = _compartment_tile(1)
    values[0, 0] = 4
    tifffile.imwrite(source, values, metadata=None)
    sdata = _backed_sdata(tmp_path)

    with pytest.raises(ValueError, match=r"unsupported category values \[4\]"):
        _add_compartment_labels(sdata, preview, sample_id="sample", chunks=(4, 4))

    assert "compartment_labels_mosaic_1" not in sdata.labels
    assert "compartment_labels_mosaic_1" not in read_zarr(sdata.path).labels


@pytest.mark.parametrize(
    ("flip_x", "flip_y", "expected_index"),
    [
        (False, False, (slice(None), slice(None))),
        (True, False, (slice(None), slice(None, None, -1))),
        (False, True, (slice(None, None, -1), slice(None))),
        (True, True, (slice(None, None, -1), slice(None, None, -1))),
    ],
)
def test_read_label_plane_supports_dataset_wide_axis_flips(
    decoded_cosmx_path: Path,
    flip_x: bool,
    flip_y: bool,
    expected_index: tuple[slice, slice],
) -> None:
    preview = _labelled_preview(decoded_cosmx_path)
    source = preview.manifest.fovs_by_id[1].compartment_labels
    assert source is not None

    result = _labels._read_label_plane(
        source,
        1,
        "compartment_labels",
        _TILE_SHAPE,
        np.dtype(np.uint8).name,
        flip_x=flip_x,
        flip_y=flip_y,
    )

    np.testing.assert_array_equal(result, _compartment_tile(1)[expected_index])


def test_changed_label_metadata_fails_write_without_leaving_an_element(
    decoded_cosmx_path: Path,
    tmp_path: Path,
) -> None:
    preview = _labelled_preview(decoded_cosmx_path)
    source = preview.manifest.fovs_by_id[1].instance_labels
    assert source is not None
    tifffile.imwrite(source, np.zeros(_TILE_SHAPE, dtype=np.uint8), metadata=None)
    sdata = _backed_sdata(tmp_path)

    with pytest.raises(ValueError, match="changed dtype after discovery"):
        _add_instance_labels(sdata, preview, sample_id="sample", chunks=(4, 4))

    assert "instance_labels_mosaic_1" not in sdata.labels
    assert "instance_labels_mosaic_1" not in read_zarr(sdata.path).labels


@pytest.mark.parametrize("overwrite", [False, True])
def test_labels_reject_non_label_name_collision_before_pixel_reads(
    decoded_cosmx_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overwrite: bool,
) -> None:
    preview = _labelled_preview(decoded_cosmx_path)
    image = Image2DModel.parse(np.zeros((1, *_TILE_SHAPE), dtype=np.uint8), dims=("c", "y", "x"))
    path = tmp_path / "cosmx.zarr"
    SpatialData(images={"instance_labels_mosaic_1": image}).write(path)
    sdata = read_zarr(path)
    reads = _instrument_label_reads(monkeypatch)

    with pytest.raises(ValueError, match="output names already belong to non-label elements"):
        _add_instance_labels(sdata, preview, sample_id="sample", chunks=(4, 4), overwrite=overwrite)

    assert not reads
    assert "instance_labels_mosaic_1" in read_zarr(path).images


def _labelled_preview(decoded_cosmx_path: Path, *, fov_2_x_px: int = 8):
    manifest = _discover_cosmx(decoded_cosmx_path)
    for files in manifest.fovs:
        if files.instance_labels is not None:
            tifffile.imwrite(files.instance_labels, _instance_tile(files.fov), metadata=None)
        if files.compartment_labels is not None:
            tifffile.imwrite(files.compartment_labels, _compartment_tile(files.fov), metadata=None)
    if fov_2_x_px != 8:
        positions = tuple(
            replace(position, x_px=fov_2_x_px) if position.fov == 2 else position for position in manifest.positions
        )
        manifest = replace(manifest, positions=positions)
    return _preview_cosmx(manifest)


def _instance_tile(fov: int) -> np.ndarray:
    values = np.zeros(_TILE_SHAPE, dtype=np.uint16)
    values[1:3, 1:3] = 1
    if fov == 1:
        values[4, 0] = 7
    elif fov == 2:
        values[4, -1] = 7
    return values


def _expected_instance_tile(fov: int) -> np.ndarray:
    values = _instance_tile(fov)[:, ::-1].astype(np.uint32)
    foreground = values != 0
    values[foreground] += np.uint32((fov - 1) * 65_536)
    return values


def _compartment_tile(fov: int) -> np.ndarray:
    values = np.zeros(_TILE_SHAPE, dtype=np.uint8)
    values[0, :4] = np.arange(4, dtype=np.uint8)
    values[1, fov % _TILE_SHAPE[1]] = 3
    return values


def _backed_sdata(tmp_path: Path) -> SpatialData:
    path = tmp_path / "cosmx.zarr"
    SpatialData().write(path)
    return read_zarr(path)


def _instrument_label_reads(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Path, str]]:
    original = _labels._read_label_plane
    reads: list[tuple[Path, str]] = []

    def instrumented(
        path: Path,
        fov: int,
        family: str,
        expected_shape: tuple[int, int],
        expected_dtype: str,
        *,
        flip_x: bool,
        flip_y: bool,
    ) -> np.ndarray:
        reads.append((path, family))
        return original(
            path,
            fov,
            family,
            expected_shape,
            expected_dtype,
            flip_x=flip_x,
            flip_y=flip_y,
        )

    monkeypatch.setattr(_labels, "_read_label_plane", instrumented)
    return reads


def _assert_transformations(sdata: SpatialData, labels_name: str, *, mosaic: int) -> None:
    transformations = get_transformation(sdata.labels[labels_name], get_all=True)
    assert set(transformations) == {f"global_{mosaic}", f"global_{mosaic}_micron"}
    assert transformations[f"global_{mosaic}"] == Identity()
    expected = Scale([1.0, 1.0], axes=("x", "y")).to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
    actual = transformations[f"global_{mosaic}_micron"].to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))
    np.testing.assert_array_equal(actual, expected)
