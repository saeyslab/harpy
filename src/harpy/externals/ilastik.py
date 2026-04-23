from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Literal

import dask
import dask.array as da
import h5py
import numpy as np
import pandas as pd
from anndata import AnnData
from loguru import logger as log
from spatialdata import SpatialData
from spatialdata.models import TableModel

from harpy.image._image import get_dataarray
from harpy.io._zarr import _get_backing_zarr_format
from harpy.table._table import ProcessTable, add_table
from harpy.utils._keys import _INSTANCE_KEY, _REGION_KEY

_VALID_EXPORT_SOURCES = ("Blockwise Object Predictions", "Object Predictions")


def _resolve_ilastik_executable(path_to_ilastik_executable: str | Path) -> Path:
    path_to_ilastik_executable = Path(path_to_ilastik_executable)
    return path_to_ilastik_executable / "ilastik" if path_to_ilastik_executable.is_dir() else path_to_ilastik_executable


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


def _validate_ilastik_inputs(sdata: SpatialData, image_name: str, labels_name: str) -> None:
    image = get_dataarray(sdata, element_name=image_name)
    labels = get_dataarray(sdata, element_name=labels_name)

    image_dims = tuple(str(dim) for dim in image.dims)
    labels_dims = tuple(str(dim) for dim in labels.dims)

    if image_dims not in (("y", "x"), ("c", "y", "x")):
        raise ValueError(
            f"Image element '{image_name}' must have dims ('y', 'x') or ('c', 'y', 'x') for ilastik, "
            f"but found {image_dims}."
        )

    if labels_dims != ("y", "x"):
        raise ValueError(
            f"Labels element '{labels_name}' must have dims ('y', 'x') for ilastik, but found {labels_dims}."
        )

    image_spatial_shape = tuple(int(image.sizes[dim]) for dim in ("y", "x"))
    labels_spatial_shape = tuple(int(labels.sizes[dim]) for dim in ("y", "x"))

    if image_spatial_shape != labels_spatial_shape:
        raise ValueError(
            f"Image element '{image_name}' and labels element '{labels_name}' must have the same spatial shape for ilastik, "
            f"but found {image_spatial_shape} and {labels_spatial_shape}."
        )


def _get_layer_path(store_path: Path, element_type: str, element_name: str) -> Path:
    element_path = store_path / element_type / element_name
    scale_zero_path = element_path / "0"
    if not element_path.exists():
        raise FileNotFoundError(element_path)
    if not scale_zero_path.exists():
        raise FileNotFoundError(scale_zero_path)
    return element_path


def _read_ilastik_predictions(prediction_path: str | Path) -> np.ndarray:
    with h5py.File(prediction_path, "r") as f:
        return np.asarray(f["exported_data"][...]).squeeze()


def _map_instance_ids_to_prediction_label_numpy(
    segmentation: np.ndarray,
    predictions: np.ndarray,
) -> dict[int, int]:
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


def _accumulate_instance_prediction_counts_chunk(
    segmentation_block: np.ndarray,
    predictions_block: np.ndarray,
    prediction_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Aggregate one chunk into a dense local `(n_instances_in_chunk, n_labels)` vote table.

    Foreground pixels are selected with `segmentation_block > 0`. Their
    instance ids are remapped to a dense local range with `np.unique(...,
    return_inverse=True)`. Prediction values are mapped to dense columns using
    the sorted global `prediction_labels` axis so the chunk can be reduced with
    a single `np.bincount`.

    The returned tuple contains the sorted local `instance_ids` and a dense
    count matrix where `counts[i, j]` is the number of pixels in that chunk
    belonging to local instance `i` and prediction-label column `j`.
    """
    segmentation_block = np.asarray(segmentation_block)
    predictions_block = np.asarray(predictions_block)
    prediction_labels = np.asarray(prediction_labels, dtype=np.int64)

    mask = segmentation_block > 0
    if not np.any(mask):
        return np.empty((0,), dtype=np.int64), np.empty((0, prediction_labels.size), dtype=np.uint64)

    segmentation_values = segmentation_block[mask].astype(np.int64, copy=False)
    prediction_values = predictions_block[mask].astype(np.int64, copy=False)
    prediction_columns = np.searchsorted(prediction_labels, prediction_values)
    prediction_columns_safe = np.where(prediction_columns >= prediction_labels.size, 0, prediction_columns)
    found_prediction_labels = prediction_labels[prediction_columns_safe] == prediction_values
    if not np.all(found_prediction_labels):
        raise ValueError("Encountered prediction labels that are not present in the global prediction-label index.")

    instance_ids, segmentation_dense = np.unique(segmentation_values, return_inverse=True)
    n_instances = instance_ids.size
    counts = np.bincount(
        segmentation_dense * prediction_labels.size + prediction_columns_safe,
        minlength=n_instances * prediction_labels.size,
    ).reshape(n_instances, prediction_labels.size)

    return instance_ids, counts.astype(np.uint64, copy=False)


def _map_instance_ids_to_prediction_label(
    segmentation: np.ndarray | da.Array,
    predictions: np.ndarray | da.Array,
    instance_ids: np.ndarray | None = None,
    prediction_labels: np.ndarray | None = None,
) -> dict[int, int]:
    """
    Map each non-zero instance id to the prediction label with the most pixels.

    For eager NumPy inputs this falls back to eager in-memory
    contingency-table implementation via np.bin_counts. For Dask-backed inputs, the arrays are
    processed chunk by chunk: each chunk is reduced to a local dense
    `(n_instances_in_chunk, n_prediction_labels)` vote table using
    `np.bincount`, then accumulated into a single global `(I, L)` matrix.

    When available, `instance_ids` and `prediction_labels` can be passed in as
    precomputed global axes to avoid extra global `unique` passes over the
    chunked arrays.
    """
    if not isinstance(segmentation, da.Array):
        segmentation = np.asarray(segmentation)
    if not isinstance(predictions, da.Array):
        predictions = np.asarray(predictions)

    if segmentation.shape != predictions.shape:
        raise ValueError(
            f"Segmentation shape {segmentation.shape} does not match ilastik prediction shape {predictions.shape}."
        )

    if not isinstance(segmentation, da.Array) and not isinstance(predictions, da.Array):
        return _map_instance_ids_to_prediction_label_numpy(segmentation=segmentation, predictions=predictions)

    if isinstance(segmentation, da.Array):
        segmentation_da = segmentation
    else:
        segmentation_da = da.from_array(
            segmentation,
            chunks=predictions.chunks if isinstance(predictions, da.Array) else segmentation.shape,
        )

    if isinstance(predictions, da.Array):
        predictions_da = predictions
    else:
        predictions_da = da.from_array(predictions, chunks=segmentation_da.chunks)

    if segmentation_da.chunks != predictions_da.chunks:
        predictions_da = predictions_da.rechunk(segmentation_da.chunks)

    if instance_ids is None:
        instance_ids_global = np.asarray(da.unique(segmentation_da).compute(), dtype=np.int64)
    else:
        instance_ids_global = np.asarray(instance_ids, dtype=np.int64)
    instance_ids_global = np.unique(instance_ids_global)
    instance_ids_global = instance_ids_global[instance_ids_global != 0]

    if instance_ids_global.size == 0:
        return {}

    if prediction_labels is None:
        prediction_labels_global = np.asarray(da.unique(predictions_da).compute(), dtype=np.int64)
    else:
        prediction_labels_global = np.asarray(prediction_labels, dtype=np.int64)
    prediction_labels_global = np.unique(prediction_labels_global)

    if prediction_labels_global.size == 0:
        return {}

    global_counts = np.zeros((instance_ids_global.size, prediction_labels_global.size), dtype=np.uint64)

    segmentation_blocks = segmentation_da.to_delayed().ravel()
    predictions_blocks = predictions_da.to_delayed().ravel()
    chunk_tasks = [
        dask.delayed(_accumulate_instance_prediction_counts_chunk)(
            segmentation_block=segmentation_block_delayed,
            predictions_block=predictions_block_delayed,
            prediction_labels=prediction_labels_global,
        )
        for segmentation_block_delayed, predictions_block_delayed in zip(
            segmentation_blocks,
            predictions_blocks,
            strict=True,
        )
    ]

    # Each delayed task consumes one segmentation/prediction chunk pair and returns:
    #   1. `local_instance_ids`: the sorted instance ids present in that chunk only
    #   2. `local_counts`: a dense vote matrix of shape
    #      `(n_instances_in_chunk, n_global_prediction_labels)`
    # So we materialize one small chunk-local contingency table per chunk rather
    # than a single full-image `(n_global_instances, n_global_prediction_labels)` table.
    for local_instance_ids, local_counts in dask.compute(*chunk_tasks):
        if local_instance_ids.size == 0:
            continue

        global_rows = np.searchsorted(instance_ids_global, local_instance_ids)
        global_rows_safe = np.where(global_rows >= instance_ids_global.size, 0, global_rows)
        found_instances = instance_ids_global[global_rows_safe] == local_instance_ids
        if not np.any(found_instances):
            continue

        global_counts[global_rows_safe[found_instances]] += local_counts[found_instances]

    valid_instances = global_counts.sum(axis=1) > 0
    if not np.any(valid_instances):
        return {}

    winner_idx = global_counts[valid_instances].argmax(axis=1)
    winner_labels = prediction_labels_global[winner_idx]
    winner_instance_ids = instance_ids_global[valid_instances]

    return {
        int(instance_id): int(prediction_label)
        for instance_id, prediction_label in zip(winner_instance_ids, winner_labels, strict=True)
    }


def _create_adata_from_labels(
    sdata: SpatialData,
    labels_name: str,
    instance_key: str,
    region_key: str,
) -> AnnData:
    instance_ids = da.unique(get_dataarray(sdata, element_name=labels_name).data)
    instance_ids = np.asarray(instance_ids.compute())
    instance_ids = instance_ids[instance_ids != 0]

    obs = pd.DataFrame(
        {
            instance_key: instance_ids.astype(np.int64),
            region_key: pd.Categorical([labels_name] * len(instance_ids)),
        }
    )
    _uuid_value = str(uuid.uuid4())[:8]
    obs.index = obs[instance_key].map(lambda x: f"{x}_{labels_name}_{_uuid_value}")
    obs.index.name = None

    return AnnData(obs=obs)


def _map_instance_predictions_to_obs(
    adata: AnnData,
    instance_key: str,
    instance_to_prediction: dict[int, int],
    unmapped_label: int = 0,
    raise_on_unmapped_instance: bool = False,
) -> pd.Series:
    mapped_predictions = adata.obs[instance_key].map(instance_to_prediction)

    if mapped_predictions.isna().any():
        unmapped_instance_ids = np.unique(
            adata.obs.loc[mapped_predictions.isna(), instance_key].to_numpy(dtype=np.int64, copy=False)
        )
        if raise_on_unmapped_instance:
            preview = ", ".join(str(instance_id) for instance_id in unmapped_instance_ids[:10])
            if unmapped_instance_ids.size > 10:
                preview = f"{preview}, ..."
            raise ValueError(
                "Found table instance ids without an ilastik prediction mapping: "
                f"[{preview}]. These ids are present in the table but not in the segmentation mask."
            )

        mapped_predictions = mapped_predictions.fillna(unmapped_label)

    return mapped_predictions.astype("int64").astype("category")


def run_object_classification(
    sdata: SpatialData,
    image_name: str,
    labels_name: str,
    table_name: str | None,
    output_table_name: str,
    path_to_classifier: str | Path,
    path_to_ilastik_executable: str | Path,
    obs_key: str = "ilastik_label",
    export_source: Literal["Blockwise Object Predictions", "Object Predictions"] = "Object Predictions",
    instance_key: str = _INSTANCE_KEY,
    region_key: str = _REGION_KEY,
    overwrite: bool = False,
    output_dir: str | Path | None = None,
    runtime_dir: str | Path | None = None,
    lazyflow_threads: int = 8,
    lazyflow_total_ram_mb: int = 16000,
    unmapped_label: int = 0,
    raise_on_unmapped_instance: bool = False,
) -> SpatialData:
    """
    Run ilastik headless object classification and add predicted labels to a table element.

    Parameters
    ----------
    sdata
        Backed SpatialData object stored as a Zarr v2 store.
    image_name
        Image element used as ilastik raw input.
    labels_name
        Labels element used as ilastik segmentation input and to map predictions back to table instances.
    table_name
        Table element from which the annotated cells are selected. If `None`, a new table is created
        from the non-zero instance ids in `labels_name`.
    output_table_name
        Output table element receiving the predicted ilastik labels in ``adata.obs[obs_key]``.
    path_to_classifier
        Path to the ilastik project ``.ilp`` file.
    path_to_ilastik_executable
        Path to the ilastik executable, or to the app directory containing the executable.
        Example:
        ``".../ilastik"``.
    obs_key
        Column name added to ``adata.obs`` with the predicted ilastik labels.
    export_source
        ilastik export source passed to ``--export_source``. This must match the export source
        configured in the ilastik GUI/project, i.e. choose either ``"Blockwise Object Predictions"``
        or ``"Object Predictions"`` consistently in both places.
    instance_key
        Name of the instance id column in ``adata.obs``. Only used if ``table_name`` is `None`.
    region_key
        Name of the region column in ``adata.obs``. Only used if ``table_name`` is `None`.
    overwrite
        Whether to overwrite ``output_table_name`` if it already exists.
    output_dir
        Directory for ilastik exported files. If ``None``, a temporary directory is created.
    runtime_dir
        Directory for ilastik runtime logs. If ``None``, a temporary directory is created.
    lazyflow_threads
        Value for the ``LAZYFLOW_THREADS`` environment variable.
    lazyflow_total_ram_mb
        Value for the ``LAZYFLOW_TOTAL_RAM_MB`` environment variable.
    unmapped_label
        Label assigned to table rows whose ``instance_key`` is not present in the segmentation-to-prediction mapping.
        This only applies when ``raise_on_unmapped_instance`` is ``False``.
    raise_on_unmapped_instance
        If ``True``, raise when a table instance id is not present in the segmentation mask instead of assigning
        ``unmapped_label``.

    Returns
    -------
    The updated :class:`~spatialdata.SpatialData` object.
    """
    store_path = _check_backed_zarr_2(sdata)

    if image_name not in sdata.images:
        raise ValueError(f"Image element '{image_name}' not found in 'sdata.images'.")
    if labels_name not in sdata.labels:
        raise ValueError(f"Labels element '{labels_name}' not found in 'sdata.labels'.")
    _validate_ilastik_inputs(sdata=sdata, image_name=image_name, labels_name=labels_name)
    if export_source not in _VALID_EXPORT_SOURCES:
        raise ValueError(f"Invalid 'export_source': {export_source!r}. Expected one of {list(_VALID_EXPORT_SOURCES)}.")

    if table_name is not None:
        adata = ProcessTable(sdata, labels_name=labels_name, table_name=table_name)._get_adata()
        region = adata.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY]
        instance_key = sdata[table_name].uns[TableModel.ATTRS_KEY][TableModel.INSTANCE_KEY]
        region_key = sdata[table_name].uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY]
    else:
        adata = _create_adata_from_labels(
            sdata=sdata,
            labels_name=labels_name,
            instance_key=instance_key,
            region_key=region_key,
        )
        region = [labels_name]

    path_to_classifier = Path(path_to_classifier).expanduser().resolve()
    ilastik_executable = _resolve_ilastik_executable(path_to_ilastik_executable).expanduser().resolve()
    path_to_image = _get_layer_path(store_path, "images", image_name)
    path_to_segmentation = _get_layer_path(store_path, "labels", labels_name)

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
        log.info(f"Created temporary ilastik runtime directory at '{runtime_dir}'.")
    else:
        runtime_dir = Path(runtime_dir).expanduser().resolve()
        runtime_dir.mkdir(parents=True, exist_ok=True)

    session_log_dir = runtime_dir / "Logs" / "ilastik"
    session_log_dir.mkdir(parents=True, exist_ok=True)

    prediction_path = output_dir / f"{image_name}_{labels_name}_object_predictions" / "exported_data.h5"
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

        log.info(f"Ilastik prediction finished. Prediction file expected at '{prediction_path}'.")

        if not prediction_path.exists():
            raise FileNotFoundError(
                f"ilastik completed without creating the expected prediction file at '{prediction_path}'."
            )

        log.info(f"Mapping instance ids from labels element '{labels_name}' to ilastik prediction labels.")
        segmentation = get_dataarray(sdata, element_name=labels_name).data.squeeze()
        predictions_np = _read_ilastik_predictions(prediction_path)
        # to speed things up for large matrices, we do the remap using Dask.
        predictions = da.from_array(predictions_np, chunks=segmentation.chunks)
        instance_to_prediction = _map_instance_ids_to_prediction_label(
            segmentation,
            predictions,
            instance_ids=adata.obs[instance_key].to_numpy(dtype=np.int64, copy=False),
            prediction_labels=np.asarray(da.unique(predictions).compute(), dtype=np.int64),
        )
        log.info(
            f"Finished mapping instance ids to ilastik prediction labels. Adding predictions to adata.obs['{obs_key}']."
        )

        adata.obs[obs_key] = _map_instance_predictions_to_obs(
            adata=adata,
            instance_key=instance_key,
            instance_to_prediction=instance_to_prediction,
            unmapped_label=unmapped_label,
            raise_on_unmapped_instance=raise_on_unmapped_instance,
        )

        log.info(
            f"Writing updated table element '{output_table_name}' with ilastik predictions in adata.obs['{obs_key}']."
        )
        return add_table(
            sdata,
            adata=adata,
            output_table_name=output_table_name,
            region=region,
            instance_key=instance_key,
            region_key=region_key,
            overwrite=overwrite,
        )
    finally:
        if cleanup_output_dir:
            shutil.rmtree(output_dir, ignore_errors=True)
        if cleanup_runtime_dir:
            shutil.rmtree(runtime_dir, ignore_errors=True)
