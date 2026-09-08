"""Local filesystem publication shared by format-specific element writers.

Serialization, reopening and in-memory/metadata restoration belong to callers.
This module only moves already-written paths and retains rollback copies while
the caller installs the published payload. It does not implement crash recovery
or concurrent-writer transactions.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from loguru import logger as log


@dataclass(frozen=True)
class _StagedElement:
    """Bind one fully serialized staging path to its permanent element path."""

    staged: Path
    destination: Path


@contextmanager
def _publish_staged_elements(
    *,
    root: Path,
    workspace: Path,
    elements: Sequence[_StagedElement],
    operation: str,
) -> Generator[None, None, None]:
    """Publish one logical update, keeping backups until the caller succeeds.

    The payload can be a complete SpatialData element or several coordinated
    components, such as an AnnData matrix and its metadata. Writers must finish
    serialization before entering this context. Readers must reopen from the
    permanent destination inside it, never retain handles to staging paths::

        existing destinations --rename--> backups (when present)
        staged payloads       --rename--> permanent destinations
                                          |
                              yield to caller: reopen, attach,
                              validate, update metadata
                                          |
                                +---------+---------+
                                |                   |
                             success              failure
                                |                   |
                         remove backups     remove replacements,
                                            restore backups, raise

    Only filesystem paths are restored here. Callers must restore their own
    affected in-memory objects and metadata and refresh consolidated metadata
    after rollback. Multiple moves are not one crash-atomic transaction.
    Staging workspaces and element paths must not themselves be symbolic
    links; symlinks in ancestor directories are allowed.

    Parameters
    ----------
    root
        Local store containing the permanent destinations.
    workspace
        Directory owned exclusively by this operation, containing all staged
        payloads. Removed after publication, or on a publication/body failure.
        The writer remains responsible for cleanup if staging itself fails.
    elements
        Non-overlapping staged/destination bindings on the same filesystem.
    operation
        Path-safe operation label for backup names and logging.
    """
    replacements = tuple(elements)
    _validate_staged_elements(root=root, workspace=workspace, elements=replacements, operation=operation)
    backup = Path(tempfile.mkdtemp(prefix=f".{root.name}.harpy-{operation}-backup-", dir=root.parent))
    backups: list[tuple[Path, Path]] = []
    published: list[Path] = []
    destinations = [str(element.destination) for element in replacements]
    log.info(f"Publishing {len(replacements)} staged element(s) for '{operation}' to {destinations!r}.")
    try:
        for ordinal, replacement in enumerate(replacements):
            if replacement.destination.exists():
                backup_path = backup / f"element-{ordinal}"
                replacement.destination.rename(backup_path)
                backups.append((replacement.destination, backup_path))
        for replacement in replacements:
            replacement.staged.rename(replacement.destination)
            published.append(replacement.destination)
        log.info(f"Removing staging workspace at '{workspace}'.")
        _remove_owned_path(workspace)
        log.info(f"Finished removing staging workspace at '{workspace}'.")
        yield
    except BaseException:
        try:
            for destination in reversed(published):
                _remove_owned_path(destination)
            for destination, backup_path in reversed(backups):
                backup_path.rename(destination)
        except BaseException as rollback_error:  # pragma: no cover - filesystem failure
            raise RuntimeError(
                f"Element publication for {operation!r} failed and rollback could not restore the previous "
                f"store state. Remaining backup data are at '{backup}'."
            ) from rollback_error
        _cleanup_owned_path(backup)
        _cleanup_owned_path(workspace)
        raise
    else:
        _cleanup_owned_path(backup)
        log.info(f"Finished publishing staged elements for '{operation}'.")


def _validate_staged_elements(
    *, root: Path, workspace: Path, elements: tuple[_StagedElement, ...], operation: str
) -> None:
    """Check path ownership and same-filesystem moves before changing destinations."""
    if not operation or Path(operation).name != operation or operation in {".", ".."}:
        raise ValueError(f"Publication operation must be a non-empty path-safe name, found {operation!r}.")
    if not elements:
        raise ValueError("At least one staged element is required for publication.")
    # Check the explicit paths before resolve() follows links, including broken
    # ones. Ancestor aliases (for example macOS /tmp) remain supported.
    managed_paths = (
        workspace,
        *(element.staged for element in elements),
        *(element.destination for element in elements),
    )
    symlinks = [str(path) for path in managed_paths if path.is_symlink()]
    if symlinks:
        raise ValueError(f"Symbolic links are not supported for staging workspaces or element paths: {symlinks!r}.")
    if not root.is_dir() or not workspace.is_dir():
        raise ValueError("Publication root and staging workspace must be existing directories.")

    # Resolve aliases before checking ownership: cleanup must never reach the
    # store root, an ancestor of it, or a destination through a staging alias.
    root_path, workspace_path = root.resolve(), workspace.resolve()
    if root_path.is_relative_to(workspace_path):
        raise ValueError("The staging workspace cannot contain the publication root.")
    staged_paths = tuple(element.staged.resolve() for element in elements)
    destination_paths = tuple(element.destination.resolve() for element in elements)
    if len(set(staged_paths)) != len(staged_paths) or len(set(destination_paths)) != len(destination_paths):
        raise ValueError("Staged source and destination paths must be unique.")
    if any(not path.exists() for path in staged_paths):
        raise ValueError("Every staged element must exist before publication.")
    if any(path == workspace_path or not path.is_relative_to(workspace_path) for path in staged_paths):
        raise ValueError("Every staged element must live inside its declared workspace.")
    if any(path == root_path or not path.is_relative_to(root_path) for path in destination_paths):
        raise ValueError("Every destination must live inside its declared store root.")
    if any(path.is_relative_to(workspace_path) or workspace_path.is_relative_to(path) for path in destination_paths):
        raise ValueError("Destinations must not overlap the staging workspace.")
    if _paths_overlap(staged_paths) or _paths_overlap(destination_paths):
        raise ValueError("Staged element paths cannot contain one another.")
    if any(not path.parent.is_dir() for path in destination_paths):
        raise ValueError("Every destination parent directory must already exist.")
    device = root.parent.stat().st_dev
    if any(path.stat().st_dev != device for path in (*staged_paths, *(p.parent for p in destination_paths))):
        raise ValueError("Element publication requires same-filesystem staging, destinations and backups.")


def _paths_overlap(paths: tuple[Path, ...]) -> bool:
    return any(
        first in second.parents or second in first.parents
        for index, first in enumerate(paths)
        for second in paths[index + 1 :]
    )


def _remove_owned_path(path: Path) -> None:
    """Remove an owned path, refusing symlinks without changing the link or target."""
    if path.is_symlink():
        raise ValueError(f"Refusing to remove symbolic link '{path}'.")
    elif path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _cleanup_owned_path(path: Path) -> None:
    """Report housekeeping failures without masking the operation's outcome."""
    try:
        if path.exists() or path.is_symlink():
            log.info(f"Removing temporary storage path '{path}'.")
            _remove_owned_path(path)
            log.info(f"Finished removing temporary storage path '{path}'.")
    except (OSError, ValueError) as error:
        log.warning(f"Could not remove temporary storage path '{path}': {error}")
