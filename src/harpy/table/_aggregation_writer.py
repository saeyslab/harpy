from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import dask
import dask.array as da
import numpy as np
import pandas as pd
import zarr
from anndata import AnnData
from anndata.io import sparse_dataset, write_elem
from scipy import sparse
from spatialdata import SpatialData
from spatialdata.models import TableModel

from harpy.table._aggregation_checkpoint import (
    _COUNT_COLUMN,
    _FEATURE_COLUMN,
    _INSTANCE_COLUMN,
    _PAIR_COLUMN,
    _AggregationCheckpoint,
    _CheckpointPartition,
)
from harpy.table._metadata import (
    _AGGREGATE_POINTS_SOURCE_KIND,
    _AUXILIARY_FEATURE_MATRIX_KEY,
    _AUXILIARY_POINTS_FRACTION_COLUMN,
    _FEATURE_CLASS_AGGREGATION_KEY,
    _FEATURE_CLASS_AGGREGATION_SCHEMA_VERSION,
    _FEATURE_MATRIX_SCHEMA_VERSION,
)
from harpy.utils._keys import _FEATURE_MATRICES_KEY


@dataclass(frozen=True)
class _AggregationDestination:
    """Validated local backing-store destination for one table write."""

    root: Path
    tables: Path
    output: Path
    zarr_format: int
    replace_existing: bool


@dataclass(frozen=True)
class _FeatureClassWriteContract:
    """Panel-defined axes and summaries needed by the component writer."""

    feature_key: str
    feature_class_key: str
    classes: tuple[str, ...]
    expression_class: str
    features_by_class_items: tuple[tuple[str, tuple[str, ...]], ...]
    count_columns: tuple[tuple[str, str], ...]

    @property
    def expression_feature_axis(self) -> tuple[str, ...]:
        return dict(self.features_by_class_items)[self.expression_class]

    @property
    def auxiliary_classes(self) -> tuple[str, ...]:
        return tuple(feature_class for feature_class in self.classes if feature_class != self.expression_class)

    @property
    def auxiliary_feature_axis(self) -> tuple[str, ...]:
        grouped = dict(self.features_by_class_items)
        return tuple(feature for feature_class in self.auxiliary_classes for feature in grouped[feature_class])

    @property
    def class_by_feature(self) -> dict[str, str]:
        return {
            feature: feature_class for feature_class, features in self.features_by_class_items for feature in features
        }

    @property
    def auxiliary_class_feature_counts(self) -> dict[str, int]:
        grouped = dict(self.features_by_class_items)
        return {feature_class: len(grouped[feature_class]) for feature_class in self.auxiliary_classes}


def _validate_aggregation_destination(
    sdata: SpatialData,
    *,
    output_table_name: str,
    overwrite: bool,
) -> _AggregationDestination:
    """Validate the backed local destination before constructing a Dask graph."""
    if not sdata.is_backed() or sdata.path is None:
        raise ValueError(
            "hp.tb.aggregate_points requires a SpatialData object backed by a writable Zarr store. "
            'Write it first with:\n\n    sdata.write("sdata.zarr")'
        )
    if (
        not isinstance(output_table_name, str)
        or not output_table_name
        or Path(output_table_name).name != output_table_name
        or output_table_name in {".", ".."}
    ):
        raise ValueError(
            f"Parameter 'output_table_name' must be a non-empty element name, found {output_table_name!r}."
        )

    root = Path(sdata.path)
    if "://" in str(sdata.path):
        raise ValueError("hp.tb.aggregate_points currently requires a local filesystem-backed SpatialData Zarr store.")
    root_group = zarr.open_group(store=str(root), mode="r+", use_consolidated=False)
    zarr_format = getattr(getattr(root_group, "metadata", None), "zarr_format", None)
    if zarr_format not in {2, 3}:
        raise ValueError(f"Could not determine the Zarr format of the backing store at {root!s}.")

    tables = root / "tables"
    output = tables / output_table_name
    memory_collision = output_table_name in sdata.tables
    disk_collision = output.exists()
    existing_element = sdata.get(output_table_name)
    if existing_element is not None and not memory_collision:
        raise ValueError(f"Element name {output_table_name!r} already belongs to a non-table SpatialData element.")
    if (memory_collision or disk_collision) and not overwrite:
        raise ValueError(f"Table element {output_table_name!r} already exists. Set 'overwrite=True' to replace it.")
    if memory_collision != disk_collision:
        raise ValueError(
            f"Table element {output_table_name!r} is inconsistent between the SpatialData object and its backing store."
        )
    return _AggregationDestination(
        root=root,
        tables=tables,
        output=output,
        zarr_format=zarr_format,
        replace_existing=memory_collision,
    )


def _create_aggregation_workspace(destination: _AggregationDestination) -> Path:
    """Create the hidden directory owned exclusively by one aggregation call."""
    # SpatialData stores without tables do not yet have this group. Create the
    # empty container only after all public request validation has completed.
    root = zarr.open_group(store=str(destination.root), mode="r+", use_consolidated=False)
    root.require_group("tables")
    workspace = destination.tables / f".harpy-aggregate-{uuid.uuid4()}"
    workspace.mkdir()
    return workspace


def _write_aggregation_table(
    sdata: SpatialData,
    *,
    destination: _AggregationDestination,
    workspace: Path,
    checkpoint: _AggregationCheckpoint,
    centers_by_pair: Mapping[int, pd.DataFrame],
    output_table_name: str,
    feature_key: str,
    region_key: str,
    instance_key: str,
    spatial_key: str,
    cell_index_name: str,
    class_contract: _FeatureClassWriteContract | None,
) -> SpatialData:
    """Construct, publish, and attach one table from a merged-count checkpoint."""
    expression_axis = checkpoint.observed_features if class_contract is None else class_contract.expression_feature_axis
    if not expression_axis:
        raise ValueError("Aggregation produced no expression features.")
    auxiliary_axis = () if class_contract is None else class_contract.auxiliary_feature_axis

    identities = tuple(identity for partition in checkpoint.partitions for identity in partition.identities)
    obs = _aggregation_obs(
        checkpoint,
        identities=identities,
        cell_index_name=cell_index_name,
        region_key=region_key,
        instance_key=instance_key,
        class_contract=class_contract,
    )
    centers = _aligned_centers(
        checkpoint,
        identities=identities,
        centers_by_pair=centers_by_pair,
    )
    var = pd.DataFrame(index=pd.Index(expression_axis, name=feature_key))
    uns = _aggregation_uns(
        checkpoint,
        class_contract=class_contract,
        auxiliary_axis=auxiliary_axis,
    )

    # Phase A already owns ``workspace/merged_counts``. Open the workspace in
    # append mode so initializing the AnnData group cannot erase that durable
    # checkpoint before the delayed CSR readers consume it.
    staging_root = zarr.open_group(store=str(workspace), mode="a", zarr_format=destination.zarr_format)
    table = TableModel.parse(
        AnnData(X=None, obs=obs, var=var, uns=uns, obsm={spatial_key: centers}),
        region_key=region_key,
        region=[pair.labels_name for pair in checkpoint.pairs],
        instance_key=instance_key,
    )
    write_elem(staging_root, "table", table)
    staging_group = staging_root["table"]

    expression = _checkpoint_sparse_array(
        checkpoint,
        feature_axis=expression_axis,
    )
    write_elem(staging_group, "X", expression)
    if class_contract is not None:
        auxiliary = _checkpoint_sparse_array(
            checkpoint,
            feature_axis=auxiliary_axis,
        )
        write_elem(staging_group["obsm"], _AUXILIARY_FEATURE_MATRIX_KEY, auxiliary)

    _validate_staged_table(
        staging_group,
        n_obs=len(identities),
        n_vars=len(expression_axis),
        n_auxiliary=len(auxiliary_axis) if class_contract is not None else None,
    )
    _set_spatialdata_table_attrs(
        staging_group,
        regions=[pair.labels_name for pair in checkpoint.pairs],
        region_key=region_key,
        instance_key=instance_key,
    )
    return _publish_aggregation_table(
        sdata,
        destination=destination,
        workspace=workspace,
        output_table_name=output_table_name,
        obs=table.obs,
        var=table.var,
        uns=table.uns,
        centers=centers,
        spatial_key=spatial_key,
        has_auxiliary=class_contract is not None,
    )


def _checkpoint_sparse_array(
    checkpoint: _AggregationCheckpoint,
    *,
    feature_axis: tuple[str, ...],
) -> da.Array:
    """Expose checkpoint partitions as full-width delayed CSR row blocks."""
    feature_axis_hash = _feature_axis_hash(feature_axis)
    blocks = [
        da.from_delayed(
            dask.delayed(_checkpoint_partition_to_csr)(
                partition,
                feature_axis=feature_axis,
                feature_axis_hash=feature_axis_hash,
            ),
            shape=(len(partition.identities), len(feature_axis)),
            dtype=np.uint32,
            meta=sparse.csr_matrix((0, 0), dtype=np.uint32),
        )
        for partition in checkpoint.partitions
    ]
    return da.concatenate(blocks, axis=0)


def _checkpoint_partition_to_csr(
    partition: _CheckpointPartition,
    *,
    feature_axis: tuple[str, ...],
    feature_axis_hash: str,
) -> sparse.csr_matrix:
    """Convert one merged-count Parquet part to a verified full-width CSR block."""
    if _feature_axis_hash(feature_axis) != feature_axis_hash:
        raise ValueError("Checkpoint CSR conversion received an inconsistent feature axis.")
    frame = pd.read_parquet(partition.path, columns=[_PAIR_COLUMN, _INSTANCE_COLUMN, _FEATURE_COLUMN, _COUNT_COLUMN])
    row_by_identity = {identity: row for row, identity in enumerate(partition.identities)}
    column_by_feature = {feature: column for column, feature in enumerate(feature_axis)}

    rows = np.fromiter(
        (
            row_by_identity[(int(pair), int(instance))]
            for pair, instance in frame[[_PAIR_COLUMN, _INSTANCE_COLUMN]].itertuples(index=False, name=None)
        ),
        dtype=np.int64,
        count=len(frame),
    )
    columns = frame[_FEATURE_COLUMN].map(column_by_feature)
    selected = columns.notna().to_numpy()
    values = frame[_COUNT_COLUMN].to_numpy(dtype=np.uint64, copy=False)
    if np.any(values[selected] > np.iinfo(np.uint32).max):
        raise ValueError("Merged assigned-point counts exceed the uint32 output range.")
    matrix = sparse.coo_matrix(
        (
            values[selected].astype(np.uint32, copy=False),
            (rows[selected], columns.loc[selected].to_numpy(dtype=np.int64, copy=False)),
        ),
        shape=(len(partition.identities), len(feature_axis)),
        dtype=np.uint32,
    ).tocsr()
    matrix.indices = matrix.indices.astype(np.int64, copy=False)
    matrix.indptr = matrix.indptr.astype(np.int64, copy=False)
    return matrix


def _feature_axis_hash(feature_axis: tuple[str, ...]) -> str:
    """Return a stable digest binding every CSR row block to one column axis."""
    payload = json.dumps(feature_axis, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _aggregation_obs(
    checkpoint: _AggregationCheckpoint,
    *,
    identities: tuple[tuple[int, int], ...],
    cell_index_name: str,
    region_key: str,
    instance_key: str,
    class_contract: _FeatureClassWriteContract | None,
) -> pd.DataFrame:
    token = str(uuid.uuid4())[:8]
    labels_names = [checkpoint.pairs[pair].labels_name for pair, _ in identities]
    instance_ids = np.asarray([instance for _, instance in identities], dtype=np.uint64)
    obs_names = [
        f"{instance}_{labels_name}_{token}" for labels_name, instance in zip(labels_names, instance_ids, strict=True)
    ]
    obs = pd.DataFrame(index=pd.Index(obs_names, name=cell_index_name))
    obs[instance_key] = instance_ids
    obs[region_key] = pd.Categorical(labels_names, categories=[pair.labels_name for pair in checkpoint.pairs])

    if class_contract is not None:
        summaries = dask.compute(
            *(
                dask.delayed(_checkpoint_partition_class_counts)(partition, contract=class_contract)
                for partition in checkpoint.partitions
            )
        )
        class_counts = pd.concat(summaries).reindex(pd.MultiIndex.from_tuples(identities))
        if class_counts.isna().any(axis=None):
            raise ValueError("Checkpoint class summaries do not cover the complete output-row manifest.")
        for feature_class, column_name in class_contract.count_columns:
            values = class_counts[feature_class].to_numpy(dtype=np.uint64, copy=False)
            if np.any(values > np.iinfo(np.uint32).max):
                raise ValueError("Per-instance assigned-point counts exceed the uint32 output range.")
            obs[column_name] = values.astype(np.uint32, copy=False)
        total = class_counts.loc[:, list(class_contract.classes)].sum(axis=1).to_numpy(dtype=np.uint64)
        auxiliary = class_counts.loc[:, list(class_contract.auxiliary_classes)].sum(axis=1).to_numpy(dtype=np.uint64)
        if np.any(total == 0):
            raise RuntimeError("Retained aggregation rows must contain at least one assigned point.")
        obs[_AUXILIARY_POINTS_FRACTION_COLUMN] = auxiliary / total
    return obs


def _checkpoint_partition_class_counts(
    partition: _CheckpointPartition,
    *,
    contract: _FeatureClassWriteContract,
) -> pd.DataFrame:
    frame = pd.read_parquet(partition.path, columns=[_PAIR_COLUMN, _INSTANCE_COLUMN, _FEATURE_COLUMN, _COUNT_COLUMN])
    feature_classes = frame[_FEATURE_COLUMN].map(contract.class_by_feature)
    if feature_classes.isna().any():
        raise ValueError("Checkpoint contains a feature absent from its feature-panel contract.")
    values = frame.assign(feature_class=feature_classes).pivot_table(
        index=[_PAIR_COLUMN, _INSTANCE_COLUMN],
        columns="feature_class",
        values=_COUNT_COLUMN,
        aggfunc="sum",
        fill_value=0,
        observed=True,
    )
    values = values.reindex(
        index=pd.MultiIndex.from_tuples(partition.identities), columns=contract.classes, fill_value=0
    )
    values.index.names = [_PAIR_COLUMN, _INSTANCE_COLUMN]
    return values.astype(np.uint64, copy=False)


def _aligned_centers(
    checkpoint: _AggregationCheckpoint,
    *,
    identities: tuple[tuple[int, int], ...],
    centers_by_pair: Mapping[int, pd.DataFrame],
) -> np.ndarray:
    coordinate_columns = checkpoint.pairs[0].coordinate_columns
    if any(pair.coordinate_columns != coordinate_columns for pair in checkpoint.pairs[1:]):
        raise ValueError("All aggregation pairs must use the same coordinate dimensions.")
    indexed_centers: list[pd.DataFrame] = []
    for pair in checkpoint.pairs:
        try:
            centers = centers_by_pair[pair.ordinal].loc[:, list(coordinate_columns)].copy()
        except KeyError as exc:
            raise ValueError(f"Centers are missing for aggregation pair {pair.ordinal}.") from exc
        if not centers.index.is_unique:
            raise ValueError(f"Labels element {pair.labels_name!r} produced duplicate center instance IDs.")
        centers.index = pd.MultiIndex.from_arrays(
            [np.full(len(centers), pair.ordinal, dtype=np.int64), centers.index.to_numpy()],
            names=[_PAIR_COLUMN, _INSTANCE_COLUMN],
        )
        indexed_centers.append(centers)

    requested = pd.MultiIndex.from_tuples(identities, names=[_PAIR_COLUMN, _INSTANCE_COLUMN])
    aligned = pd.concat(indexed_centers).reindex(requested)
    if aligned.isna().any(axis=None):
        missing = requested[aligned.isna().any(axis=1)][0]
        pair_ordinal, instance_id = missing
        raise ValueError(
            f"Label centers of mass are missing for retained instance {instance_id} in "
            f"{checkpoint.pairs[pair_ordinal].labels_name!r}."
        )
    result = aligned.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(result).all():
        raise ValueError("Label centers of mass must be finite for every retained instance.")
    return result


def _aggregation_uns(
    checkpoint: _AggregationCheckpoint,
    *,
    class_contract: _FeatureClassWriteContract | None,
    auxiliary_axis: tuple[str, ...],
) -> dict[str, object]:
    if class_contract is None:
        return {}
    return {
        _FEATURE_MATRICES_KEY: {
            _AUXILIARY_FEATURE_MATRIX_KEY: {
                "schema_version": _FEATURE_MATRIX_SCHEMA_VERSION,
                "source_kind": _AGGREGATE_POINTS_SOURCE_KIND,
                "feature_columns": list(auxiliary_axis),
            }
        },
        _FEATURE_CLASS_AGGREGATION_KEY: {
            "schema_version": _FEATURE_CLASS_AGGREGATION_SCHEMA_VERSION,
            "source_kind": _AGGREGATE_POINTS_SOURCE_KIND,
            "feature_key": class_contract.feature_key,
            "feature_class_key": class_contract.feature_class_key,
            "expression_class": class_contract.expression_class,
            "classes": list(class_contract.classes),
            "auxiliary_class_feature_counts": class_contract.auxiliary_class_feature_counts,
            "count_columns": dict(class_contract.count_columns),
            "auxiliary_points_fraction_column": _AUXILIARY_POINTS_FRACTION_COLUMN,
            "auxiliary_feature_matrix_key": _AUXILIARY_FEATURE_MATRIX_KEY,
            "regions": {
                pair.labels_name: {
                    "points_element": pair.points_name,
                    "coordinate_system": pair.coordinate_system,
                }
                for pair in checkpoint.pairs
            },
        },
    }


def _validate_staged_table(
    group: zarr.Group,
    *,
    n_obs: int,
    n_vars: int,
    n_auxiliary: int | None,
) -> None:
    expression = sparse_dataset(group["X"])
    if expression.shape != (n_obs, n_vars):
        raise ValueError(f"Staged expression matrix has shape {expression.shape}, expected {(n_obs, n_vars)}.")
    if n_auxiliary is not None:
        auxiliary = sparse_dataset(group["obsm"][_AUXILIARY_FEATURE_MATRIX_KEY])
        if auxiliary.shape != (n_obs, n_auxiliary):
            raise ValueError(f"Staged auxiliary matrix has shape {auxiliary.shape}, expected {(n_obs, n_auxiliary)}.")


def _set_spatialdata_table_attrs(
    group: zarr.Group,
    *,
    regions: list[str],
    region_key: str,
    instance_key: str,
) -> None:
    """Adopt an AnnData group using SpatialData's current table attributes."""
    group.attrs["spatialdata-encoding-type"] = "ngff:regions_table"
    group.attrs["region"] = regions
    group.attrs["region_key"] = region_key
    group.attrs["instance_key"] = instance_key
    group.attrs["version"] = "0.2"


def _publish_aggregation_table(
    sdata: SpatialData,
    *,
    destination: _AggregationDestination,
    workspace: Path,
    output_table_name: str,
    obs: pd.DataFrame,
    var: pd.DataFrame,
    uns: Mapping[str, object],
    centers: np.ndarray,
    spatial_key: str,
    has_auxiliary: bool,
) -> SpatialData:
    """Atomically publish a staged local table and attach backed sparse handles."""
    staging = workspace / "table"
    # Keep the rollback copy beside, rather than inside, the Zarr root. It is
    # on the same filesystem for atomic renames but cannot be discovered as a
    # table while consolidated metadata is rebuilt.
    backup = destination.root.parent / f".{destination.root.name}.harpy-aggregate-backup-{uuid.uuid4()}"
    published = False
    attached = False
    previous_table = sdata.tables.get(output_table_name)
    try:
        if destination.replace_existing:
            destination.output.rename(backup)
        try:
            staging.rename(destination.output)
            published = True
        except Exception:
            if backup.exists():
                backup.rename(destination.output)
            raise

        # No checkpoint input remains live after component writing. Remove the
        # hidden workspace before consolidating so it cannot be mistaken for a
        # SpatialData/Zarr child in consolidated metadata.
        shutil.rmtree(workspace)
        root = zarr.open_group(store=str(destination.root), mode="r+", use_consolidated=False)
        table_group = root["tables"][output_table_name]
        obsm: dict[str, object] = {spatial_key: centers}
        if has_auxiliary:
            obsm[_AUXILIARY_FEATURE_MATRIX_KEY] = sparse_dataset(table_group["obsm"][_AUXILIARY_FEATURE_MATRIX_KEY])
        backed_table = AnnData(
            X=sparse_dataset(table_group["X"]),
            obs=obs,
            var=var,
            uns=uns,
            obsm=obsm,
        )
        TableModel.validate(backed_table)
        sdata.tables[output_table_name] = backed_table
        attached = True
        sdata.write_consolidated_metadata()
    except Exception:
        if attached:
            if previous_table is None:
                del sdata.tables[output_table_name]
            else:
                sdata.tables[output_table_name] = previous_table
        if published and destination.output.exists():
            shutil.rmtree(destination.output)
        if backup.exists():
            backup.rename(destination.output)
        try:
            sdata.write_consolidated_metadata()
        except (OSError, RuntimeError, TypeError, ValueError):
            pass
        raise
    if backup.exists():
        shutil.rmtree(backup)
    return sdata


def _remove_aggregation_workspace(workspace: Path) -> None:
    """Remove only the hidden workspace owned by the current call."""
    if workspace.exists():
        shutil.rmtree(workspace)
