from pathlib import Path

import pytest
import zarr
from spatialdata import SpatialData, read_zarr
from spatialdata._io.format import SpatialDataContainerFormatV01
from spatialdata.datasets import blobs

from harpy.io import convert_to_zarr_2


def _read_root_zarr_format(path: str | Path) -> int:
    group = zarr.open_group(store=str(path), mode="r")
    return group.metadata.zarr_format


@pytest.fixture
def backed_sdata_v3(tmp_path):
    path = tmp_path / "sdata_v3.zarr"
    sdata = blobs(length=64, n_channels=2)
    sdata.write(path)
    return read_zarr(path)


def test_convert_to_zarr_2(backed_sdata_v3, tmp_path):
    output = tmp_path / "sdata_transcripts_v2.zarr"

    converted = convert_to_zarr_2(backed_sdata_v3, output=output, overwrite=False)

    assert converted.is_backed()
    assert Path(converted.path) == output
    assert _read_root_zarr_format(backed_sdata_v3.path) == 3
    assert _read_root_zarr_format(output) == 2
    assert set(converted.images) == set(backed_sdata_v3.images)
    assert set(converted.labels) == set(backed_sdata_v3.labels)
    assert set(converted.shapes) == set(backed_sdata_v3.shapes)
    assert set(converted.points) == set(backed_sdata_v3.points)
    assert set(converted.tables) == set(backed_sdata_v3.tables)


def test_convert_to_zarr_2_requires_backed(tmp_path):
    sdata = blobs(length=64, n_channels=2)

    with pytest.raises(ValueError, match="requires a backed SpatialData object"):
        convert_to_zarr_2(sdata, output=tmp_path / "out.zarr")


def test_convert_to_zarr_2_rejects_zarr_v2_input(tmp_path):
    path = tmp_path / "sdata_v2.zarr"
    SpatialData().write(path, sdata_formats=SpatialDataContainerFormatV01())
    sdata_v2 = read_zarr(path)

    with pytest.raises(ValueError, match="backed by Zarr v3"):
        convert_to_zarr_2(sdata_v2, output=tmp_path / "converted.zarr")


def test_convert_to_zarr_2_rejects_same_output_path(backed_sdata_v3):
    with pytest.raises(ValueError, match="output path must differ"):
        convert_to_zarr_2(backed_sdata_v3, output=backed_sdata_v3.path)
