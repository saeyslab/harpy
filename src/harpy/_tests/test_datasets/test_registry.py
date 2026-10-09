import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pooch
import pytest

from harpy.datasets.registry import BASE_URL, get_ome_registry, get_registry, get_spatialdata_registry

MEMBERS = ["sdata.zarr/zarr.json", "sdata.zarr/images/zarr.json", "sdata.zarr/images/image/c/0/0/0"]


@pytest.mark.parametrize("make_registry", [get_registry, get_spatialdata_registry, get_ome_registry])
def test_parallel_fetches_download_and_unpack_archive_once(make_registry, tmp_path, monkeypatch):
    """Concurrent fetches of one archive download it once, and each caller sees it fully unpacked."""
    monkeypatch.delenv("HARPY_POOCH_CACHE", raising=False)

    archive = tmp_path / "source" / "data.zip"
    archive.parent.mkdir()
    with zipfile.ZipFile(archive, "w") as zf:
        for member in MEMBERS:
            zf.writestr(member, member)

    registry = make_registry(path=tmp_path / "cache")
    registry.registry["data.zip"] = pooch.file_hash(str(archive))

    downloads = []

    def slow_download(url, output_file, pooch_instance):
        downloads.append(url)
        time.sleep(0.2)  # keep the first download in flight while the other fetches start
        Path(output_file).write_bytes(archive.read_bytes())

    n_fetches = 4
    barrier = threading.Barrier(n_fetches)

    def fetch(_):
        barrier.wait()
        return registry.fetch("data.zip", processor=pooch.Unzip(), downloader=slow_download)

    with ThreadPoolExecutor(max_workers=n_fetches) as executor:
        results = list(executor.map(fetch, range(n_fetches)))

    assert len(downloads) == 1
    unzip_dir = Path(registry.abspath) / "data.zip.unzip"
    for fnames in results:
        assert sorted(Path(f).relative_to(unzip_dir).as_posix() for f in fnames) == sorted(MEMBERS)


def test_registry_keeps_cache_location_and_base_url(tmp_path, monkeypatch):
    """The locked registry keeps what ``pooch.create`` resolved, including the ``HARPY_POOCH_CACHE`` override."""
    monkeypatch.setenv("HARPY_POOCH_CACHE", str(tmp_path / "env_cache"))

    registry = get_registry(path=tmp_path / "ignored")

    assert Path(registry.path) == tmp_path / "env_cache" / "0.0.1"
    assert registry.base_url == f"{BASE_URL}/"
