from __future__ import annotations

import shutil
import uuid
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr
from anndata import AnnData
from loguru import logger as log
from spatialdata import SpatialData

from harpy.table._validation import _validate_table_without_canonical
from harpy.table._zarr import (
    _publish_staged_anndata_elements,
    _read_anndata_element,
    _StagedAnnDataElement,
    _write_anndata_element,
)
from harpy.table.canonical_centers import (
    CANONICAL_ALGORITHM_VERSION,
    CANONICAL_OBSM_KEY,
    SPATIAL_COORDINATES_KEY,
    CanonicalCacheUpdatePayload,
    CanonicalRegionMetadata,
    build_canonical_metadata,
    calculate_canonical_centers,
    canonical_metadata_to_storage,
    inspect_canonical_cache,
    validate_canonical_payload,
)

_CANONICAL_MATRIX_PATH = ("obsm", CANONICAL_OBSM_KEY)
_CANONICAL_METADATA_PATH = ("uns", SPATIAL_COORDINATES_KEY, CANONICAL_OBSM_KEY)


@dataclass(frozen=True)
class _CanonicalCentersDestination:
    """Validated local component paths for one existing backed table."""

    root: Path
    table: Path
    zarr_format: int

    @property
    def matrix(self) -> Path:
        return self.table.joinpath(*_CANONICAL_MATRIX_PATH)

    @property
    def registry(self) -> Path:
        return self.table.joinpath(*_CANONICAL_METADATA_PATH[:-1])

    @property
    def metadata(self) -> Path:
        return self.table.joinpath(*_CANONICAL_METADATA_PATH)


def add_canonical_centers(
    sdata: SpatialData,
    *,
    table_name: str,
    labels_name: str | Sequence[str] | None = None,
    overwrite: bool = False,
) -> SpatialData:
    """Add label-derived canonical centers to an existing SpatialData table.

    The table must be a regions table backed by a writable local Zarr store.
    Canonical centers are calculated only for the positive instance IDs already
    represented by the table and are stored in the intrinsic ``scale0`` frame
    of each source labels element. Dense ``float64`` coordinates in fixed
    ``(z, y, x)`` order are written to
    ``adata.obsm["spatial_canonical"]``. Their schema, source, calculation and
    table-row coverage metadata are written to
    ``adata.uns["spatial_coordinates"]["spatial_canonical"]``.

    A two-dimensional labels source uses ``z=0``; a three-dimensional source
    keeps its measured ``z`` center. For a multi-region table, rows remain in
    their original order and each region is bound to its own intrinsic labels
    frame.

    Only the two coordinated canonical components are staged and written. The
    table's expression matrix and all unrelated AnnData components remain
    untouched. Publication is rollback-safe across both disk components, the
    in-memory table refresh and consolidated-metadata writing.

    Parameters
    ----------
    sdata
        SpatialData object backed by a writable local Zarr store.
    table_name
        Existing regions-table element to update.
    labels_name
        Optional labels element or sequence of labels elements. ``None`` uses
        every region declared by the table annotation. An explicit value is an
        assertion and must name that complete set; it cannot select a partial
        table.
    overwrite
        Replace an existing complete, stale, malformed or asymmetric canonical
        payload. If ``False``, either existing canonical component is a
        collision.

    Returns
    -------
    The same SpatialData object with its selected table updated.

    Raises
    ------
    ValueError
        If the request, backing store, table annotation, labels binding,
        non-canonical Harpy metadata, existing-component state or calculated
        payload is invalid.
    """
    destination = _validate_canonical_centers_destination(
        sdata,
        table_name=table_name,
        overwrite=overwrite,
    )
    annotation = _validate_table_without_canonical(sdata, table_name)
    if annotation is None:
        raise ValueError(f"Table element {table_name!r} is not a SpatialData regions table.")
    region_key, instance_key, annotated_regions = annotation
    labels_names = _normalize_labels_names(
        sdata,
        table_name=table_name,
        labels_name=labels_name,
        region_key=region_key,
        annotated_regions=annotated_regions,
    )
    _validate_existing_canonical_components(
        sdata,
        table_name=table_name,
        destination=destination,
        overwrite=overwrite,
    )

    log.info(
        f"Calculating canonical label centers for existing table {table_name!r} from "
        f"{len(labels_names)} labels region(s)."
    )
    payloads = tuple(
        calculate_canonical_centers(
            sdata,
            inspect_canonical_cache(sdata, table_name=table_name, labels_name=current_labels_name),
        )
        for current_labels_name in labels_names
    )
    log.info(f"Finished calculating canonical label centers for existing table {table_name!r}.")

    table = sdata.tables[table_name]
    centers, metadata = _assemble_canonical_table_payload(
        table,
        table_name=table_name,
        region_key=region_key,
        instance_key=instance_key,
        labels_names=labels_names,
        payloads=payloads,
    )
    _validate_candidate_payload(
        sdata,
        table,
        table_name=table_name,
        region_key=region_key,
        instance_key=instance_key,
        labels_names=labels_names,
        centers=centers,
        metadata=metadata,
    )

    workspace = _stage_canonical_components(
        destination,
        centers=centers,
        metadata=metadata,
    )
    try:
        _validate_staged_canonical_components(
            sdata,
            table,
            workspace=workspace,
            table_name=table_name,
            region_key=region_key,
            instance_key=instance_key,
            labels_names=labels_names,
        )
        _install_canonical_components(
            sdata,
            table_name=table_name,
            destination=destination,
            workspace=workspace,
            region_key=region_key,
            instance_key=instance_key,
            labels_names=labels_names,
        )
    finally:
        _remove_generated_path(workspace)
    return sdata


def _validate_canonical_centers_destination(
    sdata: SpatialData,
    *,
    table_name: str,
    overwrite: bool,
) -> _CanonicalCentersDestination:
    """Validate the local backing store without inspecting canonical content."""
    if not isinstance(sdata, SpatialData):
        raise ValueError(f"Parameter 'sdata' must be a SpatialData object, found {type(sdata).__name__}.")
    if (
        not isinstance(table_name, str)
        or not table_name
        or Path(table_name).name != table_name
        or table_name in {".", ".."}
    ):
        raise ValueError(f"Parameter 'table_name' must be a non-empty element name, found {table_name!r}.")
    if not isinstance(overwrite, bool):
        raise ValueError(f"Parameter 'overwrite' must be a bool, found {type(overwrite).__name__}.")
    if not sdata.is_backed() or sdata.path is None:
        raise ValueError(
            "hp.tb.add_canonical_centers requires a SpatialData object backed by a writable Zarr store. "
            'Write it first with:\n\n    sdata.write("sdata.zarr")'
        )
    if "://" in str(sdata.path):
        raise ValueError("hp.tb.add_canonical_centers currently requires a local filesystem-backed Zarr store.")

    root = Path(sdata.path)
    root_group = zarr.open_group(store=str(root), mode="r+", use_consolidated=False)
    zarr_format = getattr(getattr(root_group, "metadata", None), "zarr_format", None)
    if zarr_format not in {2, 3}:
        raise ValueError(f"Could not determine the Zarr format of the backing store at {root!s}.")

    memory_exists = table_name in sdata.tables
    disk_exists = "tables" in root_group and table_name in root_group["tables"]
    if memory_exists != disk_exists:
        raise ValueError(
            f"Table element {table_name!r} is inconsistent between the SpatialData object and its backing store."
        )
    if not memory_exists:
        raise ValueError(f"Table element {table_name!r} is not present in 'sdata.tables'.")
    table_path = root / "tables" / table_name
    if not table_path.is_dir():
        raise ValueError(f"Backed table element {table_name!r} does not use a local directory Zarr layout.")
    return _CanonicalCentersDestination(root=root, table=table_path, zarr_format=zarr_format)


def _normalize_labels_names(
    sdata: SpatialData,
    *,
    table_name: str,
    labels_name: str | Sequence[str] | None,
    region_key: str,
    annotated_regions: tuple[str, ...],
) -> tuple[str, ...]:
    """Validate the asserted labels set and return annotation-defined order."""
    if labels_name is None:
        requested = annotated_regions
    elif isinstance(labels_name, str):
        requested = (labels_name,)
    elif isinstance(labels_name, Sequence):
        requested = tuple(labels_name)
    else:
        raise ValueError("Parameter 'labels_name' must be a string, a sequence of strings, or None.")

    if not requested or any(not isinstance(name, str) or not name for name in requested):
        raise ValueError("Parameter 'labels_name' must contain non-empty labels element names.")
    if len(set(requested)) != len(requested):
        raise ValueError(f"Parameter 'labels_name' contains duplicate elements: {list(requested)!r}.")
    if set(requested) != set(annotated_regions):
        missing = sorted(set(annotated_regions) - set(requested))
        additional = sorted(set(requested) - set(annotated_regions))
        raise ValueError(
            f"Parameter 'labels_name' must match every region annotated by table {table_name!r}; "
            f"missing={missing!r}, additional={additional!r}."
        )

    not_labels = sorted(name for name in annotated_regions if name not in sdata.labels)
    if not_labels:
        raise ValueError(f"Table {table_name!r} references non-label spatial elements: {not_labels!r}.")
    table = sdata.tables[table_name]
    empty_regions = [name for name in annotated_regions if not bool((table.obs[region_key] == name).any())]
    if empty_regions:
        raise ValueError(f"Table {table_name!r} declares labels regions without observations: {empty_regions!r}.")
    return annotated_regions


def _validate_existing_canonical_components(
    sdata: SpatialData,
    *,
    table_name: str,
    destination: _CanonicalCentersDestination,
    overwrite: bool,
) -> None:
    """Validate memory/disk component presence and enforce overwrite policy."""
    table = sdata.tables[table_name]
    memory_matrix = CANONICAL_OBSM_KEY in table.obsm
    memory_registry = table.uns.get(SPATIAL_COORDINATES_KEY)
    if SPATIAL_COORDINATES_KEY in table.uns and not isinstance(memory_registry, Mapping):
        raise ValueError(f"adata.uns[{SPATIAL_COORDINATES_KEY!r}] must be a mapping registry.")
    memory_metadata = isinstance(memory_registry, Mapping) and CANONICAL_OBSM_KEY in memory_registry

    root = zarr.open_group(store=str(destination.root), mode="r+", use_consolidated=False)
    table_group = root["tables"][table_name]
    disk_matrix = CANONICAL_OBSM_KEY in table_group["obsm"]
    uns_group = table_group["uns"]
    disk_registry_element = uns_group.get(SPATIAL_COORDINATES_KEY)
    if disk_registry_element is not None and not isinstance(disk_registry_element, zarr.Group):
        raise ValueError(f"On-disk AnnData uns[{SPATIAL_COORDINATES_KEY!r}] must be a mapping registry.")
    disk_metadata = isinstance(disk_registry_element, zarr.Group) and CANONICAL_OBSM_KEY in disk_registry_element
    if (memory_matrix, memory_metadata) != (disk_matrix, disk_metadata):
        raise ValueError(
            f"Canonical components for table {table_name!r} are inconsistent between memory and the backing store."
        )

    if not overwrite and (memory_matrix or memory_metadata):
        if memory_matrix and memory_metadata:
            raise ValueError(
                f"Table {table_name!r} already contains canonical centers. Set 'overwrite=True' to replace them."
            )
        raise ValueError(
            f"Table {table_name!r} contains an incomplete canonical-center payload. "
            "Set 'overwrite=True' to replace both coordinated components."
        )


def _assemble_canonical_table_payload(
    table: AnnData,
    *,
    table_name: str,
    region_key: str,
    instance_key: str,
    labels_names: tuple[str, ...],
    payloads: tuple[CanonicalCacheUpdatePayload, ...],
) -> tuple[np.ndarray, dict[str, object]]:
    """Align one calculated payload per labels region to the existing table rows."""
    by_labels = {payload.labels_name: payload for payload in payloads}
    if len(by_labels) != len(payloads) or set(by_labels) != set(labels_names):
        raise ValueError("Canonical center payloads must cover every annotated labels region exactly once.")

    centers = np.full((table.n_obs, 3), np.nan, dtype=np.float64)
    covered = np.zeros(table.n_obs, dtype=bool)
    regions: dict[str, CanonicalRegionMetadata] = {}
    for current_labels_name in labels_names:
        payload = by_labels[current_labels_name]
        binding = payload.binding
        if (
            binding.table_name != table_name
            or binding.region_key != region_key
            or binding.instance_key != instance_key
            or binding.labels_name != current_labels_name
        ):
            raise ValueError(f"Canonical center binding for labels element {current_labels_name!r} is incompatible.")
        if np.any(binding.row_positions >= table.n_obs) or covered[binding.row_positions].any():
            raise ValueError("Canonical center bindings overlap or exceed the table-row space.")
        centers[binding.row_positions] = payload.centers
        covered[binding.row_positions] = True
        regions[current_labels_name] = CanonicalRegionMetadata(
            source_signature=payload.source_signature,
            n_obs=binding.n_obs,
            instance_set_digest=binding.instance_set_digest,
            algorithm_version=CANONICAL_ALGORITHM_VERSION,
        )
    if not covered.all() or not np.isfinite(centers).all():
        raise ValueError("Canonical centers must cover every table row with finite coordinates.")

    metadata = build_canonical_metadata(
        region_key=region_key,
        instance_key=instance_key,
        regions=regions,
    )
    return centers, canonical_metadata_to_storage(metadata)


def _validate_candidate_payload(
    sdata: SpatialData,
    source_table: AnnData,
    *,
    table_name: str,
    region_key: str,
    instance_key: str,
    labels_names: tuple[str, ...],
    centers: object,
    metadata: object,
) -> None:
    """Validate a replacement payload without mutating the source table."""
    candidate = AnnData(
        X=None,
        obs=source_table.obs.copy(),
        obsm={CANONICAL_OBSM_KEY: centers},
        uns={SPATIAL_COORDINATES_KEY: {CANONICAL_OBSM_KEY: metadata}},
    )
    validate_canonical_payload(
        sdata,
        candidate,
        table_name=table_name,
        region_key=region_key,
        instance_key=instance_key,
        regions=labels_names,
    )


def _stage_canonical_components(
    destination: _CanonicalCentersDestination,
    *,
    centers: np.ndarray,
    metadata: Mapping[str, object],
) -> Path:
    """Write both components to a temporary store in their final hierarchy.

    The staging store mirrors the logical paths the components will eventually
    occupy in the existing AnnData table::

        staging/
        |-- obsm/spatial_canonical
        `-- uns/spatial_coordinates/spatial_canonical

    This makes staged validation use the same paths and AnnData encodings as
    the published representation. Publication later moves only these two
    encoded leaves into the existing table; it does not replace or reconstruct
    the table itself.
    """
    workspace = destination.root.parent / f".{destination.root.name}.harpy-canonical-{uuid.uuid4().hex[:8]}"
    try:
        staging = zarr.open_group(store=str(workspace), mode="w", zarr_format=destination.zarr_format)
        log.info(f"Writing staged canonical-center components to '{workspace}'.")
        _write_anndata_element(staging, _CANONICAL_MATRIX_PATH, centers, create_parents=True)
        _write_anndata_element(staging, _CANONICAL_METADATA_PATH, dict(metadata), create_parents=True)
        log.info(f"Finished writing staged canonical-center components to '{workspace}'.")
    except Exception:
        _remove_generated_path(workspace)
        raise
    return workspace


def _validate_staged_canonical_components(
    sdata: SpatialData,
    source_table: AnnData,
    *,
    workspace: Path,
    table_name: str,
    region_key: str,
    instance_key: str,
    labels_names: tuple[str, ...],
) -> None:
    """Reopen and validate the exact serialized replacement components."""
    staging = zarr.open_group(store=str(workspace), mode="r", use_consolidated=False)
    metadata = _read_anndata_element(staging, _CANONICAL_METADATA_PATH)
    # Dense centers reopen as a storage-backed Zarr array. Canonical validation
    # intentionally materializes this small ``(n_obs, 3)`` matrix temporarily.
    centers = _read_anndata_element(staging, _CANONICAL_MATRIX_PATH)
    _validate_candidate_payload(
        sdata,
        source_table,
        table_name=table_name,
        region_key=region_key,
        instance_key=instance_key,
        labels_names=labels_names,
        centers=centers,
        metadata=metadata,
    )


def _install_canonical_components(
    sdata: SpatialData,
    *,
    table_name: str,
    destination: _CanonicalCentersDestination,
    workspace: Path,
    region_key: str,
    instance_key: str,
    labels_names: tuple[str, ...],
) -> None:
    """Publish, attach and consolidate the staged canonical components."""
    table = sdata.tables[table_name]
    previous_matrix_exists = CANONICAL_OBSM_KEY in table.obsm
    previous_matrix = table.obsm.get(CANONICAL_OBSM_KEY)
    previous_registry_exists = SPATIAL_COORDINATES_KEY in table.uns
    previous_registry = table.uns.get(SPATIAL_COORDINATES_KEY)

    try:
        with _publish_staged_canonical_components(destination=destination, workspace=workspace) as table_group:
            matrix = _read_anndata_element(table_group, _CANONICAL_MATRIX_PATH)
            metadata = _read_anndata_element(table_group, _CANONICAL_METADATA_PATH)
            registry = dict(previous_registry) if isinstance(previous_registry, Mapping) else {}
            registry[CANONICAL_OBSM_KEY] = metadata
            table.obsm[CANONICAL_OBSM_KEY] = matrix
            table.uns[SPATIAL_COORDINATES_KEY] = registry
            validate_canonical_payload(
                sdata,
                table,
                table_name=table_name,
                region_key=region_key,
                instance_key=instance_key,
                regions=labels_names,
            )
            sdata.write_consolidated_metadata()
    except Exception:
        if previous_matrix_exists:
            table.obsm[CANONICAL_OBSM_KEY] = previous_matrix
        elif CANONICAL_OBSM_KEY in table.obsm:
            del table.obsm[CANONICAL_OBSM_KEY]
        if previous_registry_exists:
            table.uns[SPATIAL_COORDINATES_KEY] = previous_registry
        else:
            table.uns.pop(SPATIAL_COORDINATES_KEY, None)
        try:
            sdata.write_consolidated_metadata()
        except (OSError, RuntimeError, TypeError, ValueError):
            pass
        raise


@contextmanager
def _publish_staged_canonical_components(
    *,
    destination: _CanonicalCentersDestination,
    workspace: Path,
) -> Generator[zarr.Group, None, None]:
    """Publish two canonical components through the shared element transaction.

    Existing canonical components, including an asymmetric old payload, are
    preserved as rollback copies while the caller refreshes the in-memory
    table, validates it and writes consolidated metadata::

        existing components --rename--> same-filesystem backups
        staged components   --rename--> table component paths
                                      |
                                      v
                    yield the existing table Zarr group
                                      |
                         +------------+------------+
                         |                         |
                      success                    failure
                         |                         |
                 remove workspace       remove replacements,
                                        restore backups

    The filesystem transaction itself is implemented by
    :func:`_publish_staged_anndata_elements`; this wrapper additionally creates
    and, on failure, removes the nested ``uns["spatial_coordinates"]`` mapping
    when the table did not already contain it.
    """
    registry_created = not destination.registry.exists()
    try:
        if registry_created:
            root = zarr.open_group(store=str(destination.root), mode="r+", use_consolidated=False)
            _write_anndata_element(
                root["tables"][destination.table.name]["uns"],
                (SPATIAL_COORDINATES_KEY,),
                {},
            )

        with _publish_staged_anndata_elements(
            root=destination.root,
            workspace=workspace,
            elements=(
                _StagedAnnDataElement(
                    staged=workspace.joinpath(*_CANONICAL_MATRIX_PATH),
                    destination=destination.matrix,
                ),
                _StagedAnnDataElement(
                    staged=workspace.joinpath(*_CANONICAL_METADATA_PATH),
                    destination=destination.metadata,
                ),
            ),
            operation="canonical",
        ) as root:
            yield root["tables"][destination.table.name]
    except BaseException:
        if registry_created:
            _remove_generated_path(destination.registry)
        raise


def _remove_generated_path(path: Path) -> None:
    """Remove one explicitly generated workspace or component path."""
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
