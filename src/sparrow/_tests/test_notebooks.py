import importlib
import os

import nbformat
import pyrootutils
import pytest
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor


def run_notebook(notebook_path, timeout=600):
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=timeout, kernel_name="python3")
    try:
        ep.preprocess(nb, {"metadata": {"path": os.path.dirname(notebook_path)}})
    except CellExecutionError as e:
        raise RuntimeError(f"Error executing the notebook '{notebook_path}': {e}") from e


@pytest.mark.skip
@pytest.mark.skipif(
    not importlib.util.find_spec("cellpose"),
    reason="requires the cellpose library",
)
@pytest.mark.parametrize(
    "notebook",
    [
        "Harpy_transcriptomics.ipynb",
    ],
)
def test_notebooks_harpy_transcriptomics(notebook):
    root = str(pyrootutils.setup_root(os.getcwd(), dotenv=True, pythonpath=True))

    run_notebook(os.path.join(root, "docs/tutorials/advanced", notebook))


@pytest.mark.skip
@pytest.mark.skipif(
    not importlib.util.find_spec("cellpose") or not importlib.util.find_spec("spatialdata_plot"),
    reason="requires the cellpose library",
)
@pytest.mark.parametrize(
    "notebook",
    [
        "Harpy_feature_calculation.ipynb",
    ],
)
def test_notebooks_harpy_feature_calculation(notebook):
    root = str(pyrootutils.setup_root(os.getcwd(), dotenv=True, pythonpath=True))

    run_notebook(os.path.join(root, "docs/tutorials/general", notebook))


@pytest.mark.skip
@pytest.mark.skipif(
    not importlib.util.find_spec("textalloc") or not importlib.util.find_spec("spatialdata_plot"),
    reason="requires the textalloc library",
)
@pytest.mark.parametrize(
    "notebook",
    [
        "Harpy_QC_IMC.ipynb",
    ],
)
def test_notebooks_harpy_qc_imc(notebook):
    root = str(pyrootutils.setup_root(os.getcwd(), dotenv=True, pythonpath=True))

    run_notebook(os.path.join(root, "docs/tutorials/general", notebook))


@pytest.mark.skip
@pytest.mark.skipif(
    not importlib.util.find_spec("textalloc")
    or not importlib.util.find_spec("joypy")
    or not importlib.util.find_spec("spatialdata_plot"),
    reason="requires the textalloc library",
)
@pytest.mark.parametrize(
    "notebook",
    [
        "Harpy_QC_in_silico.ipynb",
    ],
)
def test_notebooks_harpy_qc_in_silico(notebook):
    root = str(pyrootutils.setup_root(os.getcwd(), dotenv=True, pythonpath=True))

    run_notebook(os.path.join(root, "docs/tutorials/advanced", notebook))


@pytest.mark.skip
@pytest.mark.parametrize(
    "notebook",
    [
        "coordinate_systems.ipynb",
    ],
)
def test_notebooks_coordinate_systems(notebook):
    root = str(pyrootutils.setup_root(os.getcwd(), dotenv=True, pythonpath=True))

    run_notebook(os.path.join(root, "docs/tutorials/advanced", notebook))


@pytest.mark.skip
@pytest.mark.skipif(not importlib.util.find_spec("flowsom"), reason="requires the flowSOM library")
@pytest.mark.parametrize(
    "notebook",
    [
        "FlowSOM_for_pixel_and_cell_clustering.ipynb",
    ],
)
def test_notebooks_flowsom(notebook):
    root = str(pyrootutils.setup_root(os.getcwd(), dotenv=True, pythonpath=True))

    run_notebook(os.path.join(root, "docs/tutorials/general", notebook))


@pytest.mark.skip
@pytest.mark.skipif(
    not importlib.util.find_spec("cellpose") or not importlib.util.find_spec("basicpy"),
    reason="requires the cellpose and basicpy libraries",
)
@pytest.mark.parametrize(
    "notebook",
    [
        "SPArrOW_how_to_start.ipynb",
    ],
)
def test_notebook_sparrow_pipeline(notebook):
    root = str(pyrootutils.setup_root(os.getcwd(), dotenv=True, pythonpath=True))

    run_notebook(os.path.join(root, "docs/tutorials", notebook))
