"""Whole-element SpatialData I/O using shared filesystem publication.

SpatialData owns serialization and reopening. Replacement keeps the old
element available during staging, then uses ``_publication`` to move paths
and retain rollback copies while the reopened element is attached. This
adapter restores the in-memory element after failure; callers must restore
any associated metadata they modify themselves.
"""

from __future__ import annotations

import tempfile
import warnings
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import zarr
from anndata import AnnData
from dask.dataframe import DataFrame
from geopandas import GeoDataFrame
from loguru import logger as log
from spatialdata import SpatialData, get_dask_backing_files, read_zarr
from spatialdata._io.format import SpatialDataContainerFormatV01, SpatialDataContainerFormatV02
from xarray import DataArray, DataTree

from harpy._storage._publication import _cleanup_owned_path, _publish_staged_paths, _StagedPath


def _write_element_with_cleanup(sdata: SpatialData, element_name: str) -> None:
    """Write an already-attached element and remove partial state if the write fails."""
    try:
        sdata.write_element(element_name)
    except Exception as e:
        log.warning(
            f"Writing element '{element_name}' failed with error: {e}. Attempting best-effort cleanup before re-raising."
        )
        if sdata.get(element_name) is not None:
            del sdata[element_name]
        try:
            sdata.delete_element_from_disk(element_name)
        except Exception as e:  # noqa: BLE001
            log.warning(f"Best-effort cleanup failed for element '{element_name}': {e}")
        raise


def _read_zarr_with_annotating_table_warning_suppressed(
    path: str | Path,
    selection: list[str],
) -> SpatialData:
    """Read a partial SpatialData selection without surfacing expected table-target warnings."""
    with _suppress_missing_table_region_warning():
        return read_zarr(path, selection=selection)


@contextmanager
def _suppress_missing_table_region_warning() -> Generator[None, None, None]:
    """Suppress missing targets only while constructing an intentional partial container."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="The table is annotating",
            module="spatialdata._core.spatialdata",
        )
        yield


@contextmanager
def _replace_element_on_disk(
    sdata: SpatialData,
    element_name: str,
    element: DataArray | DataTree | DataFrame | GeoDataFrame | AnnData,
    element_type: Literal["images", "labels", "shapes", "tables", "points"] = "images",
) -> Generator[SpatialData, None, None]:
    """Replace a backed element through staging, retaining its backup until the context succeeds.

    Flow (using the backing store's Zarr format)::

        element (data + metadata)
            | SpatialData.write
            v
        temporary workspace/store.zarr (original unchanged and readable)
            | _publish_staged_paths
            v
        old destination -> backup; staged element -> permanent destination
        remove workspace
            |
            v
        refresh consolidated metadata; reopen and attach replacement
            |
            v
        yield sdata -> caller's with-body (pass or related updates)
            |
            v
        refresh consolidated metadata -> success: remove backup

    Use ``pass`` when no additional caller-side updates are needed::

        with _replace_element_on_disk(
            sdata, element_name=element_name, element=element, element_type=element_type
        ):
            pass

    Staging failure leaves the original untouched. Failures during publication,
    reopening, the caller's body or finalization trigger attempted disk rollback,
    restoration of the previous in-memory element and best-effort metadata
    consolidation. Let exceptions propagate out of the context; rollback is no
    longer available after successful exit.

    Root attributes and other caller-owned state are not automatically restored.
    Snapshot them before updating them inside the body, and catch failures around
    the entire ``with`` statement, including entry and exit. On failure, restore
    that state in memory and on disk and refresh consolidated metadata.

    Only local, same-filesystem stores are supported; there is no crash recovery
    or concurrent-writer isolation. Other attached elements must not depend
    lazily on the destination being replaced.
    """
    if element_type not in {"images", "labels", "shapes", "tables", "points"}:
        raise ValueError(f"Unsupported SpatialData element type: {element_type!r}.")
    if not sdata.is_backed() or sdata.path is None or "://" in str(sdata.path):
        raise ValueError("Element replacement requires a local filesystem-backed SpatialData Zarr store.")
    if not element_name or Path(element_name).name != element_name or element_name in {".", ".."}:
        raise ValueError(f"Invalid element name for replacement: {element_name!r}.")

    root = Path(sdata.path)
    destination = root / element_type / element_name
    collection = getattr(sdata, element_type)
    previous_element = collection.get(element_name)
    if previous_element is None or not destination.is_dir():
        raise ValueError(f"Element {element_name!r} must already exist in memory and in its backing store.")
    # Preserve SpatialData's deletion safeguard for other attached lazy
    # elements. Only the replacement itself may depend on the old destination:
    # its graph is fully written before any paths move.
    destination_path = destination.resolve()
    dependents = [
        name
        for _, name, other in sdata.gen_elements()
        if name != element_name
        and any(Path(path).resolve().is_relative_to(destination_path) for path in get_dask_backing_files(other))
    ]
    if dependents:
        raise ValueError(
            f"Cannot replace {element_name!r}: other elements still read from its backing path: {dependents!r}. "
            "Write those elements to independent backing paths first."
        )
    # Refuse to replace an arbitrary directory where a Zarr element should be.
    zarr.open_group(store=str(destination), mode="r", use_consolidated=False)
    root_group = zarr.open_group(store=str(root), mode="r+", use_consolidated=False)
    zarr_format = root_group.metadata.zarr_format
    if zarr_format not in {2, 3}:
        raise ValueError(f"Unsupported backing Zarr format: {zarr_format!r}.")
    container_format = SpatialDataContainerFormatV01() if zarr_format == 2 else SpatialDataContainerFormatV02()

    # A table-only staging container intentionally omits its annotation targets;
    # they remain untouched in the original SpatialData store.
    with _suppress_missing_table_region_warning():
        staging_sdata = SpatialData(**{element_type: {element_name: element}})
    workspace = Path(tempfile.mkdtemp(prefix=f".{root.name}.harpy-replace-", dir=root.parent))
    staged_store = workspace / "store.zarr"
    try:
        log.info(f"Writing replacement {element_type}/{element_name} to staging at '{staged_store}'.")
        staging_sdata.write(staged_store, sdata_formats=container_format, consolidate_metadata=False)
        log.info(f"Finished writing staged replacement {element_type}/{element_name}.")
        try:
            with _publish_staged_paths(
                root=root,
                workspace=workspace,
                paths=(_StagedPath(staged=staged_store / element_type / element_name, destination=destination),),
                operation="replace",
            ):
                # SpatialData's reader can use consolidated metadata. Refresh
                # it after moving paths, before reopening the new payload.
                sdata.write_consolidated_metadata()
                reopened = _read_zarr_with_annotating_table_warning_suppressed(root, selection=[element_type])
                sdata[element_name] = reopened[element_name]
                yield sdata
                # The caller may have written associated metadata while the
                # backup was retained; include those changes before committing.
                sdata.write_consolidated_metadata()
        except BaseException:
            collection[element_name] = previous_element
            try:
                sdata.write_consolidated_metadata()
            except Exception as error:  # noqa: BLE001
                log.warning(f"Could not refresh consolidated metadata after replacement rollback: {error}")
            raise
    finally:
        _cleanup_owned_path(workspace)
