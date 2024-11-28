"""File acts as a starting point for running the pipeline for single sample analysis. It can be run from any place with the command: 'harpy'."""

import logging

import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="configs", config_name="pipeline.yaml")
def main(cfg: DictConfig) -> None:
    """Main function for the single pipeline which checks the supplied paths first and then calls all seven steps from the pipeline functions."""
    from harpy.pipeline import HarpyPipeline

    pipeline = HarpyPipeline(cfg)

    _ = pipeline.run_pipeline()

    return


if __name__ == "__main__":
    main()
