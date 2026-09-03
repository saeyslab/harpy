from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyarrow as pa
from dask.dataframe import DataFrame as DaskDataFrame

_PAIR_COLUMN = "aggregation_pair"
# Configurable ``instance_key`` names are normalized to this stable checkpoint
# column. The label IDs themselves remain unchanged and are later written to
# ``adata.obs`` under the user-configured name.
_CHECKPOINT_INSTANCE_COLUMN = "instance_id"
_FEATURE_COLUMN = "feature"
_COUNT_COLUMN = "count"
_CHECKPOINT_COLUMNS = (_PAIR_COLUMN, _CHECKPOINT_INSTANCE_COLUMN, _FEATURE_COLUMN, _COUNT_COLUMN)


@dataclass(frozen=True)
class _CheckpointPair:
    """Describe one normalized labels/points pair represented in a checkpoint."""

    ordinal: int
    labels_name: str
    points_name: str
    coordinate_system: str
    coordinate_columns: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.ordinal < 0:
            raise ValueError(f"Checkpoint pair ordinal must be non-negative, found {self.ordinal}.")
        if any(
            not isinstance(value, str) or not value
            for value in (self.labels_name, self.points_name, self.coordinate_system)
        ):
            raise ValueError("Checkpoint pair element names and coordinate system must be non-empty strings.")
        if self.coordinate_columns not in {("x", "y"), ("x", "y", "z")}:
            raise ValueError(
                "Checkpoint pair coordinates must use ('x', 'y') or ('x', 'y', 'z'), "
                f"found {self.coordinate_columns!r}."
            )


@dataclass(frozen=True)
class _CheckpointPartition:
    """Describe one non-empty physical partition of merged feature counts."""

    ordinal: int
    path: Path
    identities: tuple[tuple[int, int], ...]
    row_count: int
    row_start: int = 0

    def __post_init__(self) -> None:
        if self.ordinal < 0:
            raise ValueError(f"Checkpoint partition ordinal must be non-negative, found {self.ordinal}.")
        if self.row_count < len(self.identities):
            raise ValueError("A checkpoint partition cannot contain fewer rows than represented instances.")
        if self.row_start < 0:
            raise ValueError(f"Checkpoint output row start must be non-negative, found {self.row_start}.")
        if not self.identities:
            raise ValueError("Checkpoint manifests describe only non-empty partitions.")
        if tuple(sorted(set(self.identities))) != self.identities:
            raise ValueError("Checkpoint partition identities must be sorted and unique.")

    @property
    def row_stop(self) -> int:
        """Return the exclusive end of this partition's output-row interval."""
        return self.row_start + len(self.identities)


@dataclass(frozen=True)
class _AggregationCheckpoint:
    """Validated durable boundary between point assignment and table writing."""

    path: Path
    pairs: tuple[_CheckpointPair, ...]
    partitions: tuple[_CheckpointPartition, ...]
    observed_features: tuple[str, ...] | None

    def __post_init__(self) -> None:
        if not self.pairs:
            raise ValueError("An aggregation checkpoint must describe at least one pair.")
        if tuple(pair.ordinal for pair in self.pairs) != tuple(range(len(self.pairs))):
            raise ValueError("Checkpoint pair ordinals must be consecutive and start at zero.")
        partition_ordinals = tuple(partition.ordinal for partition in self.partitions)
        if partition_ordinals != tuple(sorted(set(partition_ordinals))):
            raise ValueError("Checkpoint partition ordinals must be sorted and unique.")

        owner_by_identity: dict[tuple[int, int], int] = {}
        represented_pairs: set[int] = set()
        expected_row_start = 0
        for partition in self.partitions:
            if partition.row_start != expected_row_start:
                raise ValueError("Checkpoint output-row intervals must be consecutive and non-overlapping.")
            expected_row_start = partition.row_stop
            for identity in partition.identities:
                pair_ordinal, instance_id = identity
                if pair_ordinal not in range(len(self.pairs)):
                    raise ValueError(f"Checkpoint references unknown aggregation pair {pair_ordinal}.")
                if instance_id <= 0:
                    raise ValueError(f"Checkpoint instance IDs must be positive, found {instance_id}.")
                previous = owner_by_identity.setdefault(identity, partition.ordinal)
                if previous != partition.ordinal:
                    raise ValueError(
                        f"Checkpoint identity {identity!r} occurs in partitions {previous} and {partition.ordinal}."
                    )
                represented_pairs.add(pair_ordinal)

        missing_pairs = [
            (pair.labels_name, pair.points_name) for pair in self.pairs if pair.ordinal not in represented_pairs
        ]
        if missing_pairs:
            raise ValueError(
                "Aggregation produced no retained non-background instances for labels/points pair(s): "
                f"{missing_pairs!r}."
            )
        if self.observed_features is not None:
            if not self.observed_features:
                raise ValueError("Ordinary aggregation produced no observed features.")
            if self.observed_features != tuple(sorted(set(self.observed_features))):
                raise ValueError("Checkpoint observed features must be sorted and unique.")

    def instance_ids(self, pair_ordinal: int) -> np.ndarray:
        """Return sorted retained instance IDs for one aggregation pair."""
        return np.asarray(
            sorted(
                instance_id
                for partition in self.partitions
                for candidate_pair, instance_id in partition.identities
                if candidate_pair == pair_ordinal
            ),
            dtype=np.uint64,
        )


def _local_feature_counts(
    assigned_points: DaskDataFrame,
    *,
    pair_ordinal: int,
    instance_key: str,
    feature_key: str,
) -> DaskDataFrame:
    """Lazily reduce assigned points to partition-local feature counts.

    ``assigned_points`` contains one row per point assigned to a non-background
    labels instance. This function independently groups every pandas partition
    by ``(instance_key, feature_key)`` and replaces repeated point rows with one
    count row. For example, with Harpy's default ``instance_key="cell_ID"``,
    ``feature_key="gene"`` and ``pair_ordinal=0``::

        assigned points          canonical partition-local checkpoint counts
        cell_ID  gene            aggregation_pair  instance_id  feature  count
        42       EPCAM           0                 42           EPCAM    2
        42       EPCAM  ---->    0                 42           VIM      1
        42       VIM

    Here ``instance_id`` is the fixed transient checkpoint column named by
    :data:`_CHECKPOINT_INSTANCE_COLUMN`. It contains the values copied unchanged
    from ``cell_ID``; final table construction writes them to ``adata.obs`` under
    the configured ``instance_key``.

    The returned Dask dataframe remains lazy and uses the checkpoint's
    canonical ``(aggregation_pair, instance_id, feature, count)`` schema. Counts
    for the same instance and feature may still occur in different partitions;
    they are shuffled together and merged later by
    :func:`_stage_aggregation_checkpoint`.

    Parameters
    ----------
    assigned_points
        Lazy dataframe containing the requested feature column and the assigned
        nonzero label ID in ``instance_key``.
    pair_ordinal
        Zero-based identifier of the labels/points aggregation pair. It is
        written to every local count row so multiple pairs can share one
        checkpoint.
    instance_key
        Column containing the assigned labels instance ID.
    feature_key
        Column containing the point feature identifier, such as a gene.

    Returns
    -------
    Lazy partition-local counts using the canonical checkpoint schema. No
    global shuffle, computation or file write occurs in this function.
    """
    return assigned_points.map_partitions(
        _local_feature_count_partition,
        pair_ordinal=pair_ordinal,
        instance_key=instance_key,
        feature_key=feature_key,
        meta=_checkpoint_meta(),
    )


def _stage_aggregation_checkpoint(
    partial_counts: Sequence[DaskDataFrame],
    *,
    path: Path,
    pairs: tuple[_CheckpointPair, ...],
    validation_errors: Sequence[tuple[str, object]] = (),
    discover_features: bool,
) -> _AggregationCheckpoint:
    """Shuffle, merge and persist compact counts while executing assignment once.

    Every local count row is shuffled by ``(aggregation_pair, instance_id)``.
    Consequently, all feature rows for one retained instance arrive in one
    output partition and can be converted to one complete CSR row in Phase B.
    Dask preserves the compact-count partition count; Harpy does not inspect or
    rebalance partition byte sizes before writing the checkpoint.

    For example, partition-local reductions from two aggregation pairs can
    contain the following rows::

        partial_counts[0], partition A    partial_counts[0], partition B
        pair  instance  feature  count    pair  instance  feature  count
        0     42        EPCAM    2        0     42        EPCAM    3
        0     42        VIM      1        0     51        VIM      4

        partial_counts[1]
        pair  instance  feature  count
        1     42        EPCAM    5

    The shuffle hashes ``(pair, instance)`` to an output-partition number. All
    three rows keyed by ``(0, 42)`` therefore move into the same routed
    partition, irrespective of their input partitions::

        routed partition containing (0, 42)
        pair  instance  feature  count
        0     42        EPCAM    2
        0     42        VIM      1
        0     42        EPCAM    3

        one or more other routed partitions
        pair  instance  feature  count
        0     51        VIM      4
        1     42        EPCAM    5

    Different keys may share an output partition, but one key cannot be split
    across output partitions. :func:`_merge_count_partition` can consequently
    merge duplicate feature rows locally into the following canonical counts::

        pair  instance  feature  count
        0     42        EPCAM    5
        0     42        VIM      1
        0     51        VIM      4
        1     42        EPCAM    5

    Instance 42 in pair 0 remains distinct from instance 42 in pair 1. The
    example shows the logical merged rows; their physical Parquet partitioning
    is determined by the Dask shuffle.

    The deferred Parquet write, manifests, optional feature-axis tree reduction
    and feature-panel diagnostics are submitted through one ``dask.compute``
    call. Shared upstream assignment and shuffle tasks therefore execute once.
    """
    if not partial_counts:
        raise ValueError("At least one aggregation pair is required to construct a checkpoint.")
    if path.exists():
        raise ValueError(f"Checkpoint path already exists: {path}.")

    compact_counts = dd.concat(list(partial_counts), interleave_partitions=True)
    # Hash routing gives one output partition complete ownership of every
    # (aggregation pair, instance), so the following partition-local merge is
    # globally complete.
    routed = compact_counts.shuffle(on=[_PAIR_COLUMN, _CHECKPOINT_INSTANCE_COLUMN], ignore_index=True)
    merged = routed.map_partitions(_merge_count_partition, meta=_checkpoint_meta())

    def name_function(ordinal: int) -> str:
        return f"part-{ordinal:05d}.parquet"

    write_task = merged.to_parquet(
        path,
        compute=False,
        name_function=name_function,
        schema=_checkpoint_schema(),
        write_index=False,
        write_metadata_file=False,
    )
    delayed_partitions = merged.to_delayed()
    manifest_tasks = tuple(
        dask.delayed(_checkpoint_partition_manifest)(
            partition,
            ordinal=ordinal,
            path=path / name_function(ordinal),
        )
        for ordinal, partition in enumerate(delayed_partitions)
    )

    if discover_features:
        feature_tasks = [dask.delayed(_partition_features)(partition) for partition in delayed_partitions]
        feature_axis_task = _tree_union(feature_tasks)
    else:
        feature_axis_task = None

    error_tasks = tuple(
        dask.delayed(_first_partition_error)(*collection.to_delayed()) for _, collection in validation_errors
    )
    # Convert dataframe expressions to delayed objects before combining them.
    # This preserves their shared assignment/shuffle keys without asking Dask
    # to optimize mixed expression and HighLevelGraph collections.
    computed = dask.compute(write_task.to_delayed(), manifest_tasks, feature_axis_task, error_tasks)
    _, manifests, observed_features, computed_errors = computed

    for (points_name, _), error in zip(validation_errors, computed_errors, strict=True):
        if error is not None:
            raise ValueError(f"Points element {points_name!r} disagrees with its feature panel: {error}")

    nonempty_manifests: list[_CheckpointPartition] = []
    row_start = 0
    for manifest in manifests:
        if manifest is None:
            continue
        manifest = replace(manifest, row_start=row_start)
        nonempty_manifests.append(manifest)
        row_start = manifest.row_stop
    return _AggregationCheckpoint(
        path=path,
        pairs=pairs,
        partitions=tuple(nonempty_manifests),
        observed_features=None if observed_features is None else tuple(sorted(observed_features)),
    )


def _checkpoint_meta() -> pd.DataFrame:
    return pd.DataFrame(
        {
            _PAIR_COLUMN: pd.Series(dtype=np.int64),
            _CHECKPOINT_INSTANCE_COLUMN: pd.Series(dtype=np.uint64),
            _FEATURE_COLUMN: pd.Series(dtype="string"),
            _COUNT_COLUMN: pd.Series(dtype=np.uint64),
        }
    )


def _checkpoint_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field(_PAIR_COLUMN, pa.int64(), nullable=False),
            pa.field(_CHECKPOINT_INSTANCE_COLUMN, pa.uint64(), nullable=False),
            pa.field(_FEATURE_COLUMN, pa.string(), nullable=False),
            pa.field(_COUNT_COLUMN, pa.uint64(), nullable=False),
        ]
    )


def _local_feature_count_partition(
    partition: pd.DataFrame,
    *,
    pair_ordinal: int,
    instance_key: str,
    feature_key: str,
) -> pd.DataFrame:
    if partition.empty:
        return _checkpoint_meta()
    instance_values = partition[instance_key]
    feature_values = partition[feature_key]
    if instance_values.isna().any() or feature_values.isna().any():
        raise ValueError("Assigned points must not contain null instance or feature values.")
    if not pd.api.types.is_integer_dtype(instance_values.dtype):
        raise ValueError(f"Assigned instance IDs must be integral, found {instance_values.dtype}.")
    if (instance_values <= 0).any():
        raise ValueError("Assigned instance IDs must be positive; background zero must be absent.")

    # Construct the canonical checkpoint fields from the source Series instead
    # of renaming in place: a valid source feature key may equal one of these
    # private checkpoint column names.
    values = pd.DataFrame(
        {
            _CHECKPOINT_INSTANCE_COLUMN: instance_values.astype(np.uint64, copy=False),
            _FEATURE_COLUMN: feature_values.astype("string"),
        },
        index=partition.index,
    )
    counts = (
        values.groupby([_CHECKPOINT_INSTANCE_COLUMN, _FEATURE_COLUMN], observed=True, sort=False)
        .size()
        .rename(_COUNT_COLUMN)
        .reset_index()
    )
    counts.insert(0, _PAIR_COLUMN, np.int64(pair_ordinal))
    counts[_COUNT_COLUMN] = counts[_COUNT_COLUMN].astype(np.uint64, copy=False)
    return counts.loc[:, list(_CHECKPOINT_COLUMNS)]


def _merge_count_partition(partition: pd.DataFrame) -> pd.DataFrame:
    """Merge duplicate feature counts within one shuffled partition.

    :func:`_stage_aggregation_checkpoint` first shuffles by
    ``(aggregation_pair, instance_id)``, so every feature row for one instance
    is owned by exactly one output partition. A partition-local groupby on
    ``(aggregation_pair, instance_id, feature)`` is therefore sufficient to
    produce globally merged feature counts without another shuffle.
    """
    if partition.empty:
        return _checkpoint_meta()
    merged = (
        partition.groupby([_PAIR_COLUMN, _CHECKPOINT_INSTANCE_COLUMN, _FEATURE_COLUMN], observed=True, sort=False)[
            _COUNT_COLUMN
        ]
        .sum()
        .reset_index()
    )
    merged[_PAIR_COLUMN] = merged[_PAIR_COLUMN].astype(np.int64, copy=False)
    merged[_CHECKPOINT_INSTANCE_COLUMN] = merged[_CHECKPOINT_INSTANCE_COLUMN].astype(np.uint64, copy=False)
    merged[_FEATURE_COLUMN] = merged[_FEATURE_COLUMN].astype("string")
    merged[_COUNT_COLUMN] = merged[_COUNT_COLUMN].astype(np.uint64, copy=False)
    return merged.loc[:, list(_CHECKPOINT_COLUMNS)]


def _checkpoint_partition_manifest(
    partition: pd.DataFrame,
    *,
    ordinal: int,
    path: Path,
) -> _CheckpointPartition | None:
    """Describe the matrix-row ownership of one merged-count partition.

    This function validates one materialized Dask partition and returns a small
    in-memory descriptor; it does not write a separate manifest file. For
    example, a physical Parquet partition containing::

        pair  instance  feature  count
        0     42        EPCAM    5
        0     42        VIM      1
        0     51        VIM      4

    produces a descriptor equivalent to::

        _CheckpointPartition(
            ordinal=0,
            path=Path(".../part-00000.parquet"),
            identities=((0, 42), (0, 51)),
            row_count=3,
            row_start=0,
        )

    ``row_count`` is the number of long-form ``(instance, feature)`` rows in
    the Parquet partition, whereas ``len(identities)`` is the number of output
    matrix rows it owns. ``row_start`` initially uses its default of zero;
    :func:`_stage_aggregation_checkpoint` later assigns consecutive output-row
    intervals after discarding empty partitions.

    Parameters
    ----------
    partition
        One merged-count pandas partition. It must use the canonical checkpoint
        schema, contain no nulls or non-positive IDs/counts, and contain at most
        one row for each ``(aggregation_pair, instance_id, feature)`` key.
    ordinal
        Position of the physical partition in the merged Dask dataframe.
    path
        Expected path of the corresponding Parquet part written by the shared
        checkpoint computation.

    Returns
    -------
    A descriptor containing the partition path, sorted unique
    ``(aggregation_pair, instance_id)`` identities and long-form row count, or
    ``None`` when the partition is empty.
    """
    if partition.empty:
        return None
    if tuple(partition.columns) != _CHECKPOINT_COLUMNS:
        raise ValueError(
            f"Checkpoint partition columns must be {list(_CHECKPOINT_COLUMNS)!r}, found {list(partition.columns)!r}."
        )
    if partition[list(_CHECKPOINT_COLUMNS)].isna().any(axis=None):
        raise ValueError("Checkpoint partitions must not contain null values.")
    if (partition[_CHECKPOINT_INSTANCE_COLUMN] <= 0).any() or (partition[_COUNT_COLUMN] <= 0).any():
        raise ValueError("Checkpoint instance IDs and counts must be positive.")
    if partition.duplicated([_PAIR_COLUMN, _CHECKPOINT_INSTANCE_COLUMN, _FEATURE_COLUMN]).any():
        raise ValueError("Checkpoint partition contains duplicate aggregation-pair/instance/feature rows.")

    identities = tuple(
        sorted(
            {
                (int(pair_ordinal), int(instance_id))
                for pair_ordinal, instance_id in partition[[_PAIR_COLUMN, _CHECKPOINT_INSTANCE_COLUMN]].itertuples(
                    index=False, name=None
                )
            }
        )
    )
    return _CheckpointPartition(
        ordinal=ordinal,
        path=path,
        identities=identities,
        row_count=len(partition),
    )


def _partition_features(partition: pd.DataFrame) -> frozenset[str]:
    if partition.empty:
        return frozenset()
    return frozenset(partition[_FEATURE_COLUMN].astype(str).unique())


def _first_partition_error(*partitions: pd.Series) -> object | None:
    for partition in partitions:
        if len(partition):
            return partition.iloc[0]
    return None


def _tree_union(tasks: Sequence[object], *, fan_in: int = 8) -> object:
    if not tasks:
        return dask.delayed(frozenset)()
    level = list(tasks)
    while len(level) > 1:
        level = [
            dask.delayed(_union_feature_sets)(*level[start : start + fan_in]) for start in range(0, len(level), fan_in)
        ]
    return level[0]


def _union_feature_sets(*values: frozenset[str]) -> frozenset[str]:
    return frozenset().union(*values)
