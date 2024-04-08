import os
from pathlib import Path

import pyrootutils
import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from spatialdata import read_zarr


@pytest.fixture(scope="function")
def cfg_pipeline_global() -> DictConfig:
    # Expecting pytest to be run from the root dir. config_path should be relative to this file
    # The data_dir needs to be overwritten to point to the test data

    root = str(pyrootutils.setup_root(os.getcwd(), dotenv=True, pythonpath=True))

    with initialize(version_base="1.2", config_path="../configs"):
        cfg = compose(
            config_name="pipeline",
            overrides=[
                f"paths.data_dir={root}/src/sparrow/_tests/test_data",
                "dataset.data_dir=${paths.data_dir}",
                "dataset.image=${dataset.data_dir}/20272_slide1_A1-1_DAPI_4288_2144.tiff",
                "dataset.coords=${dataset.data_dir}/20272_slide1_A1-1_results_4288_2144.txt",
                "dataset.markers=${dataset.data_dir}/dummy_markers.csv",
                "allocate.delimiter='\t'",
                "allocate.column_x=0",
                "allocate.column_y=1",
                "allocate.column_gene=3",
                "segmentation=cellpose",
            ],
            return_hydra_config=True,
        )
        HydraConfig().set_config(cfg)

    return cfg


# this is called by each test which uses `cfg_pipeline` arg
# each test generates its own temporary logging path
@pytest.fixture(scope="function")
def cfg_pipeline(cfg_pipeline_global, tmp_path) -> DictConfig:
    cfg = cfg_pipeline_global.copy()

    cfg.paths.output_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


@pytest.fixture
def sdata_multi_c(tmpdir):
    root = str(pyrootutils.setup_root(os.getcwd(), dotenv=True, pythonpath=True))
    path = f"{root}/src/sparrow/_tests/test_data/multi_channel_zarr"
    sdata_path = os.path.join(path, "sdata.zarr")
    sdata = read_zarr(sdata_path)
    # backing store for specific unit test
    sdata.write(os.path.join(tmpdir, "sdata.zarr"))
    sdata = read_zarr(os.path.join(tmpdir, "sdata.zarr"))
    yield sdata


@pytest.fixture
def sdata_transcripts(tmpdir):
    root = str(pyrootutils.setup_root(os.getcwd(), dotenv=True, pythonpath=True))
    path = f"{root}/src/sparrow/_tests/test_data/transcriptomics_zarr"
    sdata_path = os.path.join(path, "sdata_transcriptomics.zarr")
    sdata = read_zarr(sdata_path)
    # backing store for specific unit test
    sdata.write(os.path.join(tmpdir, "sdata_transcriptomics.zarr"))
    sdata = read_zarr(os.path.join(tmpdir, "sdata_transcriptomics.zarr"))
    yield sdata


@pytest.fixture
def path_dataset_markers(tmpdir):
    root = str(pyrootutils.setup_root(os.getcwd(), dotenv=True, pythonpath=True))
    path = f"{root}/src/sparrow/_tests/test_data/dummy_markers.csv"
    return Path(path)
