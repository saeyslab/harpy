"""Low-level AnnData/Zarr helpers shared by Harpy table readers and writers."""

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
from anndata.io import read_elem, sparse_dataset, write_elem
from loguru import logger as log
from spatialdata.models import TableModel

# Harpy owns these literals as the compatibility boundary for tables assembled
# with the low-level AnnData writers, without depending on private SpatialData
# writer APIs.
_SPATIALDATA_TABLE_ENCODING_TYPE = "ngff:regions_table"
_SPATIALDATA_TABLE_FORMAT_VERSION = "0.2"


@dataclass(frozen=True)
class _StagedAnnDataElement:
    """Bind one encoded staging path to its permanent AnnData/Zarr path."""

    staged: Path
    destination: Path


def _read_backed_table(group: zarr.Group) -> AnnData:
    """Read a Harpy-written AnnData table from its published Zarr group.

    ``obs``, ``var`` and ``uns`` are decoded through AnnData's component
    reader because AnnData represents them as in-memory pandas and Python
    objects. Dense Zarr arrays and sparse CSR/CSC datasets in ``X`` and
    ``obsm`` remain storage-backed. This gives the returned SpatialData object
    lightweight matrix handles while making the serialized table the source of
    truth for every component.

    The direct reader intentionally covers the components written by Harpy's
    out-of-core table path: ``X``, ``obs``, ``var``, ``uns`` and ``obsm``. It
    is not a replacement for ``anndata.read_zarr()`` for arbitrary AnnData
    stores.

    Parameters
    ----------
    group
        Published AnnData Zarr group at
        ``sdata.zarr/tables/<table_name>``. The group must already be at this
        persistent element path because the returned ``zarr.Array`` and
        ``CSRDataset`` objects retain its location for subsequent reads.

    Returns
    -------
    AnnData
        Table with storage-backed array components.
    """
    uns = read_elem(group["uns"])
    if not isinstance(uns, Mapping):
        raise ValueError("AnnData Zarr component 'uns' must decode to a mapping.")
    uns = dict(uns)
    spatialdata_attrs = uns.get(TableModel.ATTRS_KEY)
    if isinstance(spatialdata_attrs, Mapping):
        spatialdata_attrs = dict(spatialdata_attrs)
        region = spatialdata_attrs.get(TableModel.REGION_KEY)
        if isinstance(region, np.ndarray):
            spatialdata_attrs[TableModel.REGION_KEY] = region.tolist()
        uns[TableModel.ATTRS_KEY] = spatialdata_attrs

    obsm_group = group["obsm"]
    obsm = {key: _read_backed_element(obsm_group[key]) for key in obsm_group.keys()}
    return AnnData(
        X=_read_backed_element(group["X"]),
        obs=read_elem(group["obs"]),
        var=read_elem(group["var"]),
        uns=uns,
        obsm=obsm,
    )


def _read_backed_element(element: zarr.Array | zarr.Group) -> object:
    """Decode one AnnData element while preserving array storage backing.

    Dense arrays remain ``zarr.Array`` objects and encoded CSR/CSC matrices
    become AnnData sparse-dataset handles. Structured encodings such as pandas
    dataframes and mappings are materialized through ``read_elem``. The policy
    depends on the stored encoding rather than its AnnData slot, so it also
    applies to array-valued ``layers``, ``obsm``, ``varm``, ``obsp`` and
    ``varp`` elements.
    """
    if isinstance(element, zarr.Array):
        return element
    if element.attrs.get("encoding-type") in {"csr_matrix", "csc_matrix"}:
        return sparse_dataset(element)
    return read_elem(element)


def _write_anndata_element(
    group: zarr.Group,
    path: tuple[str, ...],
    value: object,
    *,
    create_parents: bool = False,
) -> None:
    """Write one logical AnnData path through AnnData's encoding registry.

    Parent groups normally must already exist. ``create_parents=True`` creates
    missing parents as AnnData-encoded mappings, which is useful for building a
    partial hierarchy in an isolated staging store. Keeping path traversal here
    gives full table writers and partial component writers the same encoding
    boundary; safe replacement is provided separately by
    :func:`_publish_staged_anndata_elements`.
    """
    parent, key = _resolve_anndata_parent(group, path, create_parents=create_parents)
    write_elem(parent, key, value)


def _read_anndata_element(group: zarr.Group, path: tuple[str, ...]) -> object:
    """Read one logical AnnData path with encoding-aware storage backing."""
    parent, key = _resolve_anndata_parent(group, path, create_parents=False)
    if key not in parent:
        raise ValueError(f"AnnData element path {path!r} does not exist in Zarr group {group.name!r}.")
    return _read_backed_element(parent[key])


def _resolve_anndata_parent(
    group: zarr.Group,
    path: tuple[str, ...],
    *,
    create_parents: bool,
) -> tuple[zarr.Group, str]:
    """Resolve the existing parent group and leaf key of one AnnData path."""
    if not path or any(not isinstance(key, str) or not key for key in path):
        raise ValueError(f"AnnData element path must contain non-empty string keys, found {path!r}.")
    parent = group
    for key in path[:-1]:
        if key not in parent and create_parents:
            write_elem(parent, key, {})
        if key not in parent or not isinstance(parent[key], zarr.Group):
            raise ValueError(f"AnnData element parent path {path[:-1]!r} does not exist as a Zarr group.")
        parent = parent[key]
    return parent, path[-1]


@contextmanager
def _publish_staged_anndata_elements(
    *,
    root: Path,
    workspace: Path,
    elements: Sequence[_StagedAnnDataElement],
    operation: str,
) -> Generator[zarr.Group, None, None]:
    """Publish encoded AnnData elements and restore every destination on failure.

    Each staged element must already have been written with AnnData's encoding
    machinery. Existing destinations are moved to a same-filesystem backup,
    all staged elements are moved into place, and the reopened Zarr root is
    yielded to the caller. The backup remains available while the caller reads
    the published elements, updates its in-memory object, validates it and
    rebuilds consolidated metadata.

    If publication or the caller's context body fails, every newly published
    element is removed and every previous element is restored. The transaction
    is rollback-safe for raised exceptions, although publishing several paths
    is not one crash-atomic filesystem operation.

    Parameters
    ----------
    root
        Local SpatialData Zarr root containing the permanent destinations.
    workspace
        Harpy-owned workspace containing all staged paths. It is removed after
        the staged paths have been moved to their destinations.
    elements
        Non-overlapping staged-to-destination path bindings comprising one
        logical update.
    operation
        Short operation name used only in backup paths and log messages.
    """
    replacements = tuple(elements)
    _validate_staged_anndata_elements(
        root=root,
        workspace=workspace,
        elements=replacements,
        operation=operation,
    )
    backup = root.parent / f".{root.name}.harpy-{operation}-backup-{uuid.uuid4().hex[:8]}"
    backup.mkdir()
    backups: list[tuple[Path, Path]] = []
    published: list[Path] = []

    destinations = [str(replacement.destination) for replacement in replacements]
    log.info(f"Publishing {len(replacements)} staged AnnData element(s) for '{operation}' to {destinations!r}.")
    try:
        for ordinal, replacement in enumerate(replacements):
            if replacement.destination.exists():
                backup_path = backup / f"element-{ordinal}"
                replacement.destination.rename(backup_path)
                backups.append((replacement.destination, backup_path))

        for replacement in replacements:
            replacement.staged.rename(replacement.destination)
            published.append(replacement.destination)

        log.info(f"Removing AnnData staging workspace at '{workspace}'.")
        _remove_staging_path(workspace)
        log.info(f"Finished removing AnnData staging workspace at '{workspace}'.")
        yield zarr.open_group(store=str(root), mode="r+", use_consolidated=False)
    except BaseException:
        try:
            for destination in reversed(published):
                _remove_staging_path(destination)
            for destination, backup_path in reversed(backups):
                backup_path.rename(destination)
            _remove_staging_path(backup)
        except BaseException as rollback_error:  # pragma: no cover - catastrophic filesystem failure
            raise RuntimeError(
                f"AnnData element publication for {operation!r} failed, and rollback could not restore "
                "the previous Zarr state."
            ) from rollback_error
        raise
    else:
        try:
            _remove_staging_path(backup)
        except OSError as error:  # pragma: no cover - committed update with failed housekeeping
            log.warning(f"Could not remove completed AnnData backup at '{backup}': {error}")
        log.info(f"Finished publishing staged AnnData elements for '{operation}'.")


def _validate_staged_anndata_elements(
    *,
    root: Path,
    workspace: Path,
    elements: tuple[_StagedAnnDataElement, ...],
    operation: str,
) -> None:
    """Validate the structural path contract before mutating any destination."""
    if not operation or Path(operation).name != operation:
        raise ValueError(f"AnnData staging operation must be a non-empty path-safe name, found {operation!r}.")
    if not elements:
        raise ValueError("At least one staged AnnData element is required for publication.")
    if not root.is_dir():
        raise ValueError(f"AnnData publication root does not exist: {root!s}.")
    if not workspace.is_dir():
        raise ValueError(f"AnnData staging workspace does not exist: {workspace!s}.")

    staged_paths = tuple(element.staged for element in elements)
    destination_paths = tuple(element.destination for element in elements)
    if len(set(staged_paths)) != len(staged_paths) or len(set(destination_paths)) != len(destination_paths):
        raise ValueError("Staged AnnData source and destination paths must be unique.")
    if any(not path.exists() for path in staged_paths):
        missing = [str(path) for path in staged_paths if not path.exists()]
        raise ValueError(f"Staged AnnData elements do not exist: {missing!r}.")
    if any(not path.is_relative_to(workspace) for path in staged_paths):
        raise ValueError("Every staged AnnData element must live inside its declared workspace.")
    if any(not path.is_relative_to(root) for path in destination_paths):
        raise ValueError("Every AnnData destination must live inside its declared Zarr root.")
    if any(not path.parent.is_dir() for path in destination_paths):
        missing = [str(path.parent) for path in destination_paths if not path.parent.is_dir()]
        raise ValueError(f"AnnData destination parent groups do not exist: {missing!r}.")
    if _paths_overlap(staged_paths) or _paths_overlap(destination_paths):
        raise ValueError("Staged AnnData element paths cannot contain one another.")


def _paths_overlap(paths: tuple[Path, ...]) -> bool:
    """Return whether any path is an ancestor of another path."""
    return any(
        first in second.parents or second in first.parents
        for index, first in enumerate(paths)
        for second in paths[index + 1 :]
    )


def _remove_staging_path(path: Path) -> None:
    """Remove one explicitly owned staging, backup or published path."""
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _write_spatialdata_table_attrs(
    group: zarr.Group,
    *,
    regions: Sequence[str],
    region_key: str,
    instance_key: str,
) -> None:
    """Write SpatialData's disk-level regions-table contract.

    ``TableModel.parse()`` records the semantic table relationship in
    ``adata.uns["spatialdata_attrs"]``. A SpatialData Zarr store also requires
    attributes on the AnnData group itself so that its reader recognizes the
    group as a regions table. Harpy writes the AnnData components directly in
    its out-of-core table path, so this helper adds that second, on-disk
    representation without calling SpatialData's private writer APIs.

    Parameters
    ----------
    group
        AnnData Zarr group that will become a SpatialData table element.
    regions
        Labels elements annotated by the table.
    region_key
        Column in ``adata.obs`` that identifies the labels element.
    instance_key
        Column in ``adata.obs`` that identifies an instance within that labels
        element.
    """
    group.attrs["spatialdata-encoding-type"] = _SPATIALDATA_TABLE_ENCODING_TYPE
    group.attrs["region"] = list(regions)
    group.attrs["region_key"] = region_key
    group.attrs["instance_key"] = instance_key
    group.attrs["version"] = _SPATIALDATA_TABLE_FORMAT_VERSION
