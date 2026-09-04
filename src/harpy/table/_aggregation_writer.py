from __future__ import annotations

import shutil
import uuid
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import dask
import dask.array as da
import numpy as np
import pandas as pd
import zarr
from anndata import AnnData
from anndata.io import sparse_dataset, write_elem
from loguru import logger as log
from scipy import sparse
from spatialdata import SpatialData
from spatialdata.models import TableModel

from harpy.table._aggregation_checkpoint import (
    _CHECKPOINT_INSTANCE_COLUMN,
    _COUNT_COLUMN,
    _FEATURE_COLUMN,
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
from harpy.table._zarr import _read_backed_table, _write_spatialdata_table_attrs
from harpy.table.canonical_centers import (
    CANONICAL_ALGORITHM_VERSION,
    CANONICAL_OBSM_KEY,
    SPATIAL_COORDINATES_KEY,
    CanonicalCacheUpdatePayload,
    CanonicalRegionMetadata,
    build_canonical_metadata,
    canonical_metadata_to_storage,
    validate_canonical_payload,
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
    """Create a temporary per-call workspace inside the store's tables group.

    The workspace holds the merged-count Parquet checkpoint and the staged
    AnnData Zarr group. Successful publication moves the staged table to its
    final element path; the remaining workspace is removed after either success
    or failure.
    """
    root = zarr.open_group(store=str(destination.root), mode="r+", use_consolidated=False)
    root.require_group("tables")
    workspace = destination.tables / f".harpy-aggregate-{uuid.uuid4().hex[:8]}"
    workspace.mkdir()
    return workspace


def _write_aggregation_table(
    sdata: SpatialData,
    *,
    destination: _AggregationDestination,
    workspace: Path,
    checkpoint: _AggregationCheckpoint,
    expression_axis: tuple[str, ...],
    canonical_payloads_by_pair: Mapping[int, CanonicalCacheUpdatePayload],
    output_table_name: str,
    feature_key: str,
    region_key: str,
    instance_key: str,
    table_index_name: str,
    class_contract: _FeatureClassWriteContract | None,
) -> SpatialData:
    """Construct, publish, and attach one table from a merged-count checkpoint.

    The caller supplies the resolved feature axis defining ``adata.X``: observed
    assigned-point features in ordinary mode or the panel-defined expression
    axis in class-aware mode. In class-aware mode, the concatenated
    non-expression axes define ``adata.obsm["auxiliary_feature_counts"]``. Both
    matrices select their counts from the same lossless checkpoint.
    """
    if not expression_axis:
        raise ValueError("Aggregation produced no expression features.")
    auxiliary_axis = () if class_contract is None else class_contract.auxiliary_feature_axis

    output_row_keys = checkpoint.output_row_keys
    # Class-aware construction currently scans each checkpoint Parquet part
    # three times: once for the ``.obs`` class summaries and once for each sparse
    # output (``.X`` and ``.obsm["auxiliary_feature_counts"]``). Separate passes
    # keep component writes memory-bounded, but a future single-pass block writer
    # could derive all three outputs from one checkpoint read.
    log.info(f"Constructing AnnData '.obs' for table '{output_table_name}'.")
    obs = _aggregation_obs(
        checkpoint,
        output_row_keys=output_row_keys,
        table_index_name=table_index_name,
        region_key=region_key,
        instance_key=instance_key,
        class_contract=class_contract,
    )
    log.info(f"Finished constructing AnnData '.obs' for table '{output_table_name}'.")
    canonical_centers, canonical_metadata = _canonical_table_payload(
        checkpoint,
        output_row_keys=output_row_keys,
        payloads_by_pair=canonical_payloads_by_pair,
        region_key=region_key,
        instance_key=instance_key,
    )
    var = pd.DataFrame(index=pd.Index(expression_axis, name=feature_key))
    uns = _aggregation_uns(
        checkpoint,
        class_contract=class_contract,
        auxiliary_axis=auxiliary_axis,
    )
    uns[SPATIAL_COORDINATES_KEY] = {CANONICAL_OBSM_KEY: canonical_metadata}

    # The workspace already contains ``merged_counts``. Open it in append mode
    # so initializing the AnnData group cannot erase that durable checkpoint
    # before the delayed CSR readers consume it.
    staging_root = zarr.open_group(store=str(workspace), mode="a", zarr_format=destination.zarr_format)
    table = TableModel.parse(
        AnnData(X=None, obs=obs, var=var, uns=uns, obsm={CANONICAL_OBSM_KEY: canonical_centers}),
        region_key=region_key,
        region=[pair.labels_name for pair in checkpoint.pairs],
        instance_key=instance_key,
    )
    staging_table_path = workspace / "table"
    log.info(
        f"Writing AnnData '.obs', '.var', '.uns' and '.obsm[{CANONICAL_OBSM_KEY}]' to staged table at "
        f"'{staging_table_path}'."
    )
    write_elem(staging_root, "table", table)
    log.info(
        f"Finished writing AnnData '.obs', '.var', '.uns' and '.obsm[{CANONICAL_OBSM_KEY}]' to staged table at "
        f"'{staging_table_path}'."
    )
    staging_group = staging_root["table"]

    expression = _checkpoint_sparse_array(
        checkpoint,
        feature_axis=expression_axis,
    )
    log.info(f"Writing AnnData '.X' to staged table at '{staging_table_path / 'X'}'.")
    # AnnData recognizes this as a Dask array with CSR chunks. Its sparse
    # writer computes and appends one row chunk at a time: this bounds memory,
    # but the ordered CSR append serializes the chunk writes.
    write_elem(staging_group, "X", expression)
    log.info(f"Finished writing AnnData '.X' to staged table at '{staging_table_path / 'X'}'.")
    if class_contract is not None:
        auxiliary = _checkpoint_sparse_array(
            checkpoint,
            feature_axis=auxiliary_axis,
        )
        auxiliary_path = staging_table_path / "obsm" / _AUXILIARY_FEATURE_MATRIX_KEY
        log.info(f"Writing AnnData '.obsm[{_AUXILIARY_FEATURE_MATRIX_KEY}]' to staged table at '{auxiliary_path}'.")
        write_elem(staging_group["obsm"], _AUXILIARY_FEATURE_MATRIX_KEY, auxiliary)
        log.info(
            f"Finished writing AnnData '.obsm[{_AUXILIARY_FEATURE_MATRIX_KEY}]' to staged table at '{auxiliary_path}'."
        )

    _validate_staged_table(
        staging_group,
        n_obs=len(output_row_keys),
        n_vars=len(expression_axis),
        n_auxiliary=len(auxiliary_axis) if class_contract is not None else None,
    )
    staged_table = _read_backed_table(staging_group)
    # Validate the reopened serialized payload before publication, so the Zarr
    # representation must satisfy the same canonical contract as the in-memory
    # construction rather than relying only on construction-time checks.
    validate_canonical_payload(
        sdata,
        staged_table,
        table_name=output_table_name,
        region_key=region_key,
        instance_key=instance_key,
        regions=[pair.labels_name for pair in checkpoint.pairs],
    )
    _write_spatialdata_table_attrs(
        staging_group,
        regions=[pair.labels_name for pair in checkpoint.pairs],
        region_key=region_key,
        instance_key=instance_key,
    )
    # The final in-memory table is reconstructed from the serialized group so
    # disk, rather than these construction objects, is the source of truth.
    del table, staged_table, obs, var, uns, canonical_centers, canonical_metadata
    return _install_aggregation_table(
        sdata,
        destination=destination,
        workspace=workspace,
        output_table_name=output_table_name,
    )


def _checkpoint_sparse_array(
    checkpoint: _AggregationCheckpoint,
    *,
    feature_axis: tuple[str, ...],
) -> da.Array:
    """Expose checkpoint partitions as full-width delayed CSR row blocks."""
    blocks = [
        da.from_delayed(
            dask.delayed(_checkpoint_partition_to_csr)(
                partition,
                feature_axis=feature_axis,
            ),
            shape=(len(partition.output_row_keys), len(feature_axis)),
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
) -> sparse.csr_matrix:
    """Convert one merged-count Parquet part to an axis-aligned CSR block.

    ``feature_axis`` is the authoritative matrix-column order. Each long-form
    checkpoint feature is looked up by name and placed at its corresponding
    column index, so checkpoint row order cannot affect matrix-column order. The
    caller also uses this same axis for ``adata.var_names`` or the auxiliary
    matrix's ``feature_columns`` metadata.
    """
    frame = pd.read_parquet(
        partition.path,
        columns=[_PAIR_COLUMN, _CHECKPOINT_INSTANCE_COLUMN, _FEATURE_COLUMN, _COUNT_COLUMN],
    )
    row_by_output_row_key = {output_row_key: row for row, output_row_key in enumerate(partition.output_row_keys)}
    column_by_feature = {feature: column for column, feature in enumerate(feature_axis)}

    rows = np.fromiter(
        (
            row_by_output_row_key[(int(pair), int(instance))]
            for pair, instance in frame[[_PAIR_COLUMN, _CHECKPOINT_INSTANCE_COLUMN]].itertuples(index=False, name=None)
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
        shape=(len(partition.output_row_keys), len(feature_axis)),
        dtype=np.uint32,
    ).tocsr()
    matrix.indices = matrix.indices.astype(np.int64, copy=False)
    matrix.indptr = matrix.indptr.astype(np.int64, copy=False)
    return matrix


def _aggregation_obs(
    checkpoint: _AggregationCheckpoint,
    *,
    output_row_keys: tuple[tuple[int, int], ...],
    table_index_name: str,
    region_key: str,
    instance_key: str,
    class_contract: _FeatureClassWriteContract | None,
) -> pd.DataFrame:
    token = str(uuid.uuid4())[:8]
    labels_names = [checkpoint.pairs[pair].labels_name for pair, _ in output_row_keys]
    instance_ids = np.asarray([instance for _, instance in output_row_keys], dtype=np.uint64)
    obs_names = [
        f"{instance}_{labels_name}_{token}" for labels_name, instance in zip(labels_names, instance_ids, strict=True)
    ]
    obs = pd.DataFrame(index=pd.Index(obs_names, name=table_index_name))
    obs[instance_key] = instance_ids
    obs[region_key] = pd.Categorical(labels_names, categories=[pair.labels_name for pair in checkpoint.pairs])

    if class_contract is not None:
        summaries = dask.compute(
            *(
                dask.delayed(_checkpoint_partition_class_counts)(partition, contract=class_contract)
                for partition in checkpoint.partitions
            )
        )
        class_counts = pd.concat(summaries).reindex(pd.MultiIndex.from_tuples(output_row_keys))
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
    """Derive per-instance feature-class totals for one checkpoint partition.

    The checkpoint stores ``(instance, feature, count)`` rows rather than a
    separate class reduction. This function maps each feature to its class
    through the validated panel contract, then sums the feature counts by
    ``(aggregation pair, instance, class)``. Its columns follow the complete
    panel class order, including zero totals.

    Parameters
    ----------
    partition
        Merged-count checkpoint partition whose instances are owned entirely
        by that partition.
    contract
        Validated feature panel and expression-class selection.

    Returns
    -------
    Dataframe indexed by ``(aggregation_pair, instance_id)`` with one unsigned
    count column for every panel class.
    """
    frame = pd.read_parquet(
        partition.path,
        columns=[_PAIR_COLUMN, _CHECKPOINT_INSTANCE_COLUMN, _FEATURE_COLUMN, _COUNT_COLUMN],
    )
    feature_classes = frame[_FEATURE_COLUMN].map(contract.class_by_feature)
    if feature_classes.isna().any():
        raise ValueError("Checkpoint contains a feature absent from its feature-panel contract.")
    values = frame.assign(feature_class=feature_classes).pivot_table(
        index=[_PAIR_COLUMN, _CHECKPOINT_INSTANCE_COLUMN],
        columns="feature_class",
        values=_COUNT_COLUMN,
        aggfunc="sum",
        fill_value=0,
        observed=True,
    )
    values = values.reindex(
        index=pd.MultiIndex.from_tuples(partition.output_row_keys), columns=contract.classes, fill_value=0
    )
    values.index.names = [_PAIR_COLUMN, _CHECKPOINT_INSTANCE_COLUMN]
    return values.astype(np.uint64, copy=False)


def _canonical_table_payload(
    checkpoint: _AggregationCheckpoint,
    *,
    output_row_keys: tuple[tuple[int, int], ...],
    payloads_by_pair: Mapping[int, CanonicalCacheUpdatePayload],
    region_key: str,
    instance_key: str,
) -> tuple[np.ndarray, dict[str, object]]:
    """Align region payloads to table rows and serialize their shared metadata."""
    expected_ordinals = {pair.ordinal for pair in checkpoint.pairs}
    if set(payloads_by_pair) != expected_ordinals:
        raise ValueError("Canonical center payloads must cover every aggregation pair exactly once.")

    centers = np.full((len(output_row_keys), 3), np.nan, dtype=np.float64)
    regions: dict[str, CanonicalRegionMetadata] = {}
    for pair in checkpoint.pairs:
        payload = payloads_by_pair[pair.ordinal]
        binding = payload.binding
        expected_positions = np.asarray(
            [row for row, (pair_ordinal, _) in enumerate(output_row_keys) if pair_ordinal == pair.ordinal],
            dtype=np.intp,
        )
        expected_ids = np.asarray([output_row_keys[row][1] for row in expected_positions], dtype=np.uint64)
        if (
            binding.labels_name != pair.labels_name
            or binding.region_key != region_key
            or binding.instance_key != instance_key
            or not np.array_equal(binding.row_positions, expected_positions)
            or not np.array_equal(binding.instance_ids, expected_ids)
        ):
            raise ValueError(f"Canonical center binding for aggregation pair {pair.ordinal} is not row-aligned.")
        centers[binding.row_positions] = payload.centers
        regions[pair.labels_name] = CanonicalRegionMetadata(
            source_signature=payload.source_signature,
            n_obs=binding.n_obs,
            instance_set_digest=binding.instance_set_digest,
            algorithm_version=CANONICAL_ALGORITHM_VERSION,
        )
    if not np.isfinite(centers).all():
        raise ValueError("Canonical centers must cover every retained aggregation row with finite coordinates.")

    metadata = build_canonical_metadata(
        region_key=region_key,
        instance_key=instance_key,
        regions=regions,
    )
    return centers, canonical_metadata_to_storage(metadata)


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


def _install_aggregation_table(
    sdata: SpatialData,
    *,
    destination: _AggregationDestination,
    workspace: Path,
    output_table_name: str,
) -> SpatialData:
    """Publish, reopen and attach one fully staged aggregation table.

    Filesystem publication is isolated in
    :func:`_publish_staged_aggregation_table`. The table is then reconstructed
    exclusively from its published Zarr group by :func:`_read_backed_table`,
    validated, and attached to the in-memory SpatialData object. Any failure in
    reading, validation, attachment, or consolidated-metadata writing propagates
    through the publication context and restores the previous on-disk table::

        _publish_staged_aggregation_table()
            |
            |-- preserve previous table
            |-- publish staged table
            `-- yield published Zarr group
                     |
                     v
               _read_backed_table()
                     |
                     v
               TableModel.validate()
                     |
                     v
               attach to sdata.tables
                     |
                     v
               write consolidated metadata
                     |
               +-----+-----+
               |           |
            success      failure
               |           |
        delete backup   restore table on disk
                        restore table in memory
                        re-raise the original error
    """
    previous_table = sdata.tables.get(output_table_name)
    attached = False
    try:
        with _publish_staged_aggregation_table(destination=destination, workspace=workspace) as table_group:
            backed_table = _read_backed_table(table_group)
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
        try:
            sdata.write_consolidated_metadata()
        except (OSError, RuntimeError, TypeError, ValueError):
            pass
        raise
    return sdata


@contextmanager
def _publish_staged_aggregation_table(
    *,
    destination: _AggregationDestination,
    workspace: Path,
) -> Generator[zarr.Group, None, None]:
    """Temporarily publish a staged table and roll it back on failure.

    The staged table is complete before publication starts. When replacing an
    existing table, it is first moved to ``backup``; when creating a table,
    ``backup`` remains absent. Publication then follows this flow::

        destination.output (previous table) --rename--> backup
                                                only when replacing

        workspace/table       --rename--> destination.output
                                           |
                                           v
                                  remove remaining workspace
                                           |
                                           v
                                  yield published Zarr group
                                           |
                                           v
                                  caller reads, validates,
                                  attaches and consolidates
                                           |
                                  did the body succeed?
                                     /           \
                                   yes            no
                                    |              |
                                    v              v
                             remove backup    remove destination
                             if present       if it was published
                                                   |
                                                   v
                                            restore backup
                                            if one existed
                                                   |
                                                   v
                                            re-raise exception

    Each rename is atomic because the source and destination are on the same
    filesystem. The complete two-rename replacement is rollback-safe, but it is
    not one indivisible filesystem operation. The context keeps the backup until
    its caller has read, validated and attached the new table and rebuilt
    consolidated metadata.
    """
    staging = workspace / "table"
    # Keep the rollback copy beside, rather than inside, the Zarr root. It is
    # on the same filesystem for atomic renames but cannot be discovered as a
    # table while consolidated metadata is rebuilt.
    backup = destination.root.parent / f".{destination.root.name}.harpy-aggregate-backup-{uuid.uuid4().hex[:8]}"
    published = False
    log.info(f"Publishing staged AnnData table from '{staging}' to '{destination.output}'.")
    try:
        if destination.replace_existing:
            destination.output.rename(backup)
        staging.rename(destination.output)
        published = True

        # No checkpoint input remains live after component writing. Remove the
        # hidden workspace before consolidating so it cannot be mistaken for a
        # SpatialData/Zarr child in consolidated metadata.
        _remove_aggregation_workspace(workspace)
        root = zarr.open_group(store=str(destination.root), mode="r+", use_consolidated=False)
        yield root["tables"][destination.output.name]
    except Exception:
        if published and destination.output.exists():
            shutil.rmtree(destination.output)
        if backup.exists():
            backup.rename(destination.output)
        raise
    if backup.exists():
        shutil.rmtree(backup)
    log.info(f"Finished publishing AnnData table to '{destination.output}'.")


def _remove_aggregation_workspace(workspace: Path) -> None:
    """Remove only the hidden workspace owned by the current call."""
    if workspace.exists():
        log.info(f"Removing temporary aggregation workspace at '{workspace}'.")
        shutil.rmtree(workspace)
        log.info(f"Finished removing temporary aggregation workspace at '{workspace}'.")
