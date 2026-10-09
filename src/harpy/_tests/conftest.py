import os

import dask
import pytest
from spatialdata import read_zarr
from spatialdata.datasets import blobs

from harpy.datasets.cluster_blobs import cluster_blobs
from harpy.datasets.pixie_example import pixie_example
from harpy.datasets.proteomics import mibi_example
from harpy.datasets.registry import get_registry
from harpy.datasets.transcriptomics import (
    resolve_example,
    resolve_example_multiple_coordinate_systems,
    visium_hd_example_custom_binning,
)
from harpy.table._allocation_intensity import aggregate_image


def pytest_configure(config):
    """Limit Dask's threads in each pytest-xdist worker (``pytest -n auto``).

    Every worker process runs Dask's threaded scheduler, which by default starts
    one thread per CPU core; with one worker per core that oversubscribes the
    machine. Tests that set a scheduler or ``num_workers`` themselves override
    this locally. Without pytest-xdist nothing changes.
    """
    if os.environ.get("PYTEST_XDIST_WORKER") is not None:
        dask.config.set(num_workers=2)


@pytest.fixture
def sdata_multi_c(tmpdir):
    sdata = mibi_example()
    # backing store for specific unit test
    sdata.write(os.path.join(tmpdir, "sdata.zarr"))
    sdata = read_zarr(os.path.join(tmpdir, "sdata.zarr"))
    yield sdata


@pytest.fixture
def sdata_multi_c_no_backed():
    sdata = mibi_example()
    yield sdata


@pytest.fixture
def sdata_transcripts(tmpdir):
    sdata = resolve_example()
    # backing store for specific unit test
    sdata.write(os.path.join(tmpdir, "sdata_transcriptomics.zarr"))
    sdata = read_zarr(os.path.join(tmpdir, "sdata_transcriptomics.zarr"))
    yield sdata


@pytest.fixture
def sdata_transcripts_no_backed():
    sdata = resolve_example()
    yield sdata


@pytest.fixture
def sdata_transcripts_mul_coord(tmpdir):
    sdata = resolve_example_multiple_coordinate_systems()
    # backing store for specific unit test
    sdata.write(os.path.join(tmpdir, "sdata_transcriptomics.zarr"))
    sdata = read_zarr(os.path.join(tmpdir, "sdata_transcriptomics.zarr"))
    yield sdata


@pytest.fixture
def sdata_bin():
    sdata = visium_hd_example_custom_binning()
    yield sdata


@pytest.fixture
def sdata_blobs():
    sdata = cluster_blobs(
        shape=(512, 512), n_cell_types=10, n_cells=100, noise_level_channels=1.2, noise_level_nuclei=1.2, seed=10
    )
    yield sdata


@pytest.fixture
def sdata():
    yield blobs(length=1000, n_channels=3)


@pytest.fixture
def sdata_pixie():
    sdata = pixie_example()
    yield sdata


@pytest.fixture
def sdata_pixie_intensities():
    sdata = pixie_example()
    sdata = aggregate_image(
        sdata,
        image_name="raw_image_fov0",
        labels_name="label_whole_fov0",
        to_coordinate_system="fov0",
        mode="sum",
        output_table_name="table_intensities",
        overwrite=True,
    )
    sdata = aggregate_image(
        sdata,
        image_name="raw_image_fov1",
        labels_name="label_whole_fov1",
        to_coordinate_system="fov1",
        mode="sum",
        output_table_name="table_intensities",
        append=True,
        overwrite=True,
    )
    yield sdata


@pytest.fixture
def path_dataset_markers():
    registry = get_registry()
    return registry.fetch("transcriptomics/resolve/mouse/dummy_markers.csv")


@pytest.fixture
def path_transcripts():
    registry = get_registry()
    return registry.fetch("transcriptomics/resolve/mouse/20272_slide1_A1-1_results_4288_2144.txt")
