"""This file tests the napari widgets and should be used for development purposes."""

import os

import pytest
from hydra.core.hydra_config import HydraConfig


@pytest.mark.skip
def test_harpy_widgets(
    make_napari_viewer,
    cfg_pipeline,
):
    """
    Integration test for harpy plugin in napari
    """
    from harpy import utils as utils
    from harpy.widgets import (
        allocate_widget,
        annotate_widget,
        clean_widget,
        load_widget,
        segment_widget,
    )

    HydraConfig().set_config(cfg_pipeline)

    viewer = make_napari_viewer()

    # Start load widget
    _load_widget = load_widget()

    worker = _load_widget(
        viewer,
        path_image=cfg_pipeline.dataset.image,
        output_dir=cfg_pipeline.paths.output_dir,
    )

    _run_event_loop_until_worker_finishes(worker)

    assert os.path.exists(os.path.join(cfg_pipeline.paths.output_dir, "sdata.zarr"))

    # Start clean widget
    _clean_widget = clean_widget()

    worker = _clean_widget(viewer, viewer.layers[utils.LOAD])

    _run_event_loop_until_worker_finishes(worker)

    assert os.path.exists(os.path.join(cfg_pipeline.paths.output_dir, "original.png"))
    assert os.path.exists(os.path.join(cfg_pipeline.paths.output_dir, "tiling_correction.png"))
    assert os.path.exists(os.path.join(cfg_pipeline.paths.output_dir, "min_max_filtered.png"))
    assert os.path.exists(os.path.join(cfg_pipeline.paths.output_dir, "clahe.png"))

    # Start segment widget
    _segment_widget = segment_widget()

    worker = _segment_widget(viewer, viewer.layers[utils.CLEAN])

    _run_event_loop_until_worker_finishes(worker)

    assert os.path.exists(os.path.join(cfg_pipeline.paths.output_dir, "plot_segmentation.png"))

    # Start allocate widget
    _allocate_widget = allocate_widget()

    worker = _allocate_widget(viewer, transcripts_file=cfg_pipeline.dataset.coords)

    _run_event_loop_until_worker_finishes(worker)

    assert os.path.exists(os.path.join(cfg_pipeline.paths.output_dir, "plot_transcript_density.png"))
    assert os.path.exists(os.path.join(cfg_pipeline.paths.output_dir, "plot_shapes_leiden.png"))

    if cfg_pipeline.dataset.markers is not None:
        # Start annotate widget
        _annotate_widget = annotate_widget()

        worker = _annotate_widget(viewer, markers_file=cfg_pipeline.dataset.markers)

        _run_event_loop_until_worker_finishes(worker)

        assert os.path.exists(os.path.join(cfg_pipeline.paths.output_dir, "plot_score_genes_leiden_annotation.png"))


@pytest.mark.skip
def test_load_widget(
    make_napari_viewer,
    cfg_pipeline,
):
    """Test if the load works."""
    from harpy import utils as utils
    from harpy.widgets import (
        load_widget,
    )

    HydraConfig().set_config(cfg_pipeline)

    viewer = make_napari_viewer()

    _load_widget = load_widget()

    worker = _load_widget(
        viewer,
        path_image=cfg_pipeline.dataset.image,
        output_dir=cfg_pipeline.paths.output_dir,
    )

    _run_event_loop_until_worker_finishes(worker)

    assert os.path.exists(os.path.join(cfg_pipeline.paths.output_dir, "sdata.zarr"))


def _run_event_loop_until_worker_finishes(worker):
    from PyQt5.QtCore import QEventLoop

    loop = QEventLoop()
    worker.finished.connect(loop.quit)
    loop.exec_()
