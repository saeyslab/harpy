"""This file tests the entire pipeline and should be used for development purposes."""

import importlib.util

import pytest
from hydra.core.hydra_config import HydraConfig

from harpy.single import main


@pytest.mark.skipif(
    not importlib.util.find_spec("cellpose") or not importlib.util.find_spec("basicpy"),
    reason="requires the cellpose and basicpy libraries",
)
def test_pipeline(cfg_pipeline):
    HydraConfig().set_config(cfg_pipeline)
    main(cfg_pipeline)
