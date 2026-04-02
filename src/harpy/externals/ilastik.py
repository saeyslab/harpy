from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
from loguru import logger as log
from spatialdata import SpatialData
from spatialdata.models import TableModel

from harpy.image._image import get_dataarray
from harpy.io._zarr import _get_backing_zarr_format
from harpy.table._table import ProcessTable, add_table_layer


def _resolve_ilastik_executable(path_to_ilastik_executable: str | Path) -> Path:
    path_to_ilastik_executable = Path(path_to_ilastik_executable)
    return (
        path_to_ilastik_executable / "ilastik"
        if path_to_ilastik_executable.is_dir()
        else path_to_ilastik_executable
    )


def _check_backed_zarr_2(sdata: SpatialData) -> Path:
    if not sdata.is_backed() or sdata.path is None:
        raise ValueError(
            "`run_object_classification()` requires a backed SpatialData object. "
            "Write `sdata` to disk first and, if needed, convert it with `harpy.io.convert_to_zarr_2()`."
        )

    if _get_backing_zarr_format(sdata) != 2:
        raise ValueError(
            "`run_object_classification()` requires `sdata` to be backed by a Zarr v2 store. "
            "Convert the backing store with `harpy.io.convert_to_zarr_2()`."
        )

    return Path(sdata.path).resolve()


def _get_layer_path(store_path: Path, element_type: str, layer: str) -> Path:
    layer_path = store_path / element_type / layer
    scale_zero_path = layer_path / "0"
    if not layer_path.exists():
        raise FileNotFoundError(layer_path)
    if not scale_zero_path.exists():
        raise FileNotFoundError(scale_zero_path)
    return layer_path


def _read_ilastik_predictions(prediction_path: str | Path) -> np.ndarray:
    with h5py.File(prediction_path, "r") as f:
        return np.asarray(f["exported_data"][...]).squeeze()


def _map_instance_ids_to_prediction_label(segmentation: np.ndarray, predictions: np.ndarray) -> dict[int, int]:
    segmentation = np.asarray(segmentation)
    predictions = np.asarray(predictions)

    if segmentation.shape != predictions.shape:
        raise ValueError(
            f"Segmentation shape {segmentation.shape} does not match ilastik prediction shape {predictions.shape}."
        )

    mask = segmentation > 0
    segmentation_values = segmentation[mask].astype(np.int64)
    prediction_values = predictions[mask].astype(np.int64)

    if segmentation_values.size == 0:
        return {}

    instance_ids, segmentation_dense = np.unique(segmentation_values, return_inverse=True)
    prediction_labels, prediction_dense = np.unique(prediction_values, return_inverse=True)

    n_instances = len(instance_ids)
    n_prediction_labels = len(prediction_labels)

    counts = np.bincount(
        segmentation_dense * n_prediction_labels + prediction_dense,
        minlength=n_instances * n_prediction_labels,
    ).reshape(n_instances, n_prediction_labels)

    winner_idx = counts.argmax(axis=1)
    winner_labels = prediction_labels[winner_idx]

    return {
        int(instance_id): int(prediction_label)
        for instance_id, prediction_label in zip(instance_ids, winner_labels, strict=True)
    }


def run_object_classification(
    sdata: SpatialData,
    img_layer: str,
    labels_layer: str,
    table_layer: str,
    output_layer: str,
    path_to_classifier: str | Path,
    path_to_ilastik_executable: str | Path,
    obs_key: str = "ilastik_label",
    export_source: Literal["Blockwise Object Predictions", "Object Predictions"] = "Object Predictions",
    overwrite: bool = False,
    output_dir: str | Path | None = None,
    runtime_dir: str | Path | None = None,
    lazyflow_threads: int = 8,
    lazyflow_total_ram_mb: int = 16000,
) -> SpatialData:
    """
    Run ilastik headless object classification and add predicted labels to a table layer.

    Parameters
    ----------
    sdata
        Backed SpatialData object stored as a Zarr v2 store.
    img_layer
        Image layer used as ilastik raw input.
    labels_layer
        Labels layer used as ilastik segmentation input and to map predictions back to table instances.
    table_layer
        Table layer from which the annotated cells are selected.
    output_layer
        Output table layer receiving the predicted ilastik labels in ``adata.obs[obs_key]``.
    path_to_classifier
        Path to the ilastik project ``.ilp`` file.
    path_to_ilastik_executable
        Path to the ilastik executable, or to the app directory containing the executable.
    obs_key
        Column name added to ``adata.obs`` with the predicted ilastik labels.
    export_source
        ilastik export source passed to ``--export_source``.
    overwrite
        Whether to overwrite ``output_layer`` if it already exists.
    output_dir
        Directory for ilastik exported files. If ``None``, a temporary directory is created.
    runtime_dir
        Directory for ilastik runtime logs. If ``None``, a temporary directory is created.
    lazyflow_threads
        Value for the ``LAZYFLOW_THREADS`` environment variable.
    lazyflow_total_ram_mb
        Value for the ``LAZYFLOW_TOTAL_RAM_MB`` environment variable.

    Returns
    -------
    The updated :class:`~spatialdata.SpatialData` object.
    """
    store_path = _check_backed_zarr_2(sdata)

    if img_layer not in sdata.images:
        raise ValueError(f"Image layer '{img_layer}' not found in 'sdata.images'.")
    if labels_layer not in sdata.labels:
        raise ValueError(f"Labels layer '{labels_layer}' not found in 'sdata.labels'.")

    process_table_instance = ProcessTable(sdata, labels_layer=labels_layer, table_layer=table_layer)
    adata = process_table_instance._get_adata()

    path_to_classifier = Path(path_to_classifier).expanduser().resolve()
    ilastik_executable = _resolve_ilastik_executable(path_to_ilastik_executable).expanduser().resolve()
    path_to_image = _get_layer_path(store_path, "images", img_layer)
    path_to_segmentation = _get_layer_path(store_path, "labels", labels_layer)

    for path in (ilastik_executable, path_to_classifier, path_to_image, path_to_segmentation):
        if not path.exists():
            raise FileNotFoundError(path)

    cleanup_output_dir = False
    cleanup_runtime_dir = False

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="harpy_ilastik_output_"))
        cleanup_output_dir = True
        log.info(f"Created temporary ilastik output directory at '{output_dir}'.")
    else:
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    if runtime_dir is None:
        runtime_dir = Path(tempfile.mkdtemp(prefix="harpy_ilastik_runtime_"))
        cleanup_runtime_dir = True
    else:
        runtime_dir = Path(runtime_dir).expanduser().resolve()
        runtime_dir.mkdir(parents=True, exist_ok=True)

    session_log_dir = runtime_dir / "Logs" / "ilastik"
    session_log_dir.mkdir(parents=True, exist_ok=True)

    prediction_path = output_dir / f"{img_layer}_{labels_layer}_object_predictions" / "exported_data.h5"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = runtime_dir / "ilastik.log"

    cmd = [
        str(ilastik_executable),
        "--headless",
        "--readonly",
        f"--project={path_to_classifier}",
        f"--export_source={export_source}",
        f"--raw_data={(path_to_image / '0').as_uri()}",
        f"--segmentation_image={(path_to_segmentation / '0').as_uri()}",
        f"--output_filename_format={prediction_path}",
        f"--logfile={log_path}",
    ]

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env.setdefault("LAZYFLOW_THREADS", str(lazyflow_threads))
    env.setdefault("LAZYFLOW_TOTAL_RAM_MB", str(lazyflow_total_ram_mb))

    try:
        subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "ilastik headless object classification failed.\n"
            f"Command: {shlex.join(cmd)}\n"
            f"STDOUT:\n{e.stdout}\n"
            f"STDERR:\n{e.stderr}\n"
            f"Ilastik log path: {log_path}"
        ) from e

    if not prediction_path.exists():
        raise FileNotFoundError(
            f"ilastik completed without creating the expected prediction file at '{prediction_path}'."
        )

    predictions = _read_ilastik_predictions(prediction_path)
    segmentation = np.asarray(get_dataarray(sdata, layer=labels_layer).data.compute()).squeeze()
    instance_to_prediction = _map_instance_ids_to_prediction_label(segmentation, predictions)

    adata.obs[obs_key] = (
        adata.obs[process_table_instance.instance_key].map(instance_to_prediction).fillna(0).astype("int64")
    ).astype("category")

    sdata = add_table_layer(
        sdata,
        adata=adata,
        output_layer=output_layer,
        region=sdata[table_layer].uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY],
        instance_key=sdata[table_layer].uns[TableModel.ATTRS_KEY][TableModel.INSTANCE_KEY],
        region_key=sdata[table_layer].uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY],
        overwrite=overwrite,
    )

    if cleanup_output_dir:
        shutil.rmtree(output_dir)

    if cleanup_runtime_dir:
        shutil.rmtree(runtime_dir)

    return sdata
