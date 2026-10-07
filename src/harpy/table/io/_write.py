"""Scoped table writes using AnnData serialization and shared path publication."""

from __future__ import annotations

import tempfile
import warnings
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from os import PathLike
from pathlib import Path

import pandas as pd
import zarr
from anndata import AnnData, Raw
from loguru import logger as log
from spatialdata.models import TableModel

from harpy._storage._anndata import (
    _read_anndata_element,
    _read_anndata_table,
    _write_anndata_element,
    _write_spatialdata_table_attrs,
)
from harpy._storage._publication import (
    _cleanup_owned_path,
    _DeletedPath,
    _publish_staged_paths,
    _remove_owned_path,
    _StagedPath,
)
from harpy._storage._spatialdata import _open_spatialdata_group
from harpy.table.io._read import (
    ComponentPath,
    _check_component_path_overlap,
    _open_table_group,
    _validate_component_paths,
    _validate_path_segment,
)
from harpy.table.io._write_validation import (
    AxisNames,
    _check_component_deletion_destination,
    _check_component_write_destination,
    _check_write_destination,
    _prepare_raw_creation,
    _storage_axis_indices,
    _validate_complete_table,
    _validate_component_values_against_storage,
    _validate_deletion_paths,
    _validate_table_identities,
)


def write_table(
    store: str | PathLike[str],
    *,
    table_name: str,
    adata: AnnData,
    overwrite: bool = False,
) -> None:
    """Write one complete table, staging it before replacing any existing data.

    Parameters
    ----------
    store
        Local path to an existing SpatialData Zarr root. Its Zarr format is preserved.
    table_name
        Destination table name, without path separators.
    adata
        Complete AnnData, including any layers, metadata and raw data to retain.
        Replacement is not a merge. Matrices may be in memory, lazy or storage-backed;
        lazy replacements may read the destination being overwritten.
    overwrite
        Allow replacement of an existing table. Otherwise only creation is allowed.

    Notes
    -----
    The input is not modified; its matrices retain their original representations.

    Zarr-backed matrices are internally wrapped in lazy Dask arrays and evaluated
    in chunks during writing, without preliminary whole-matrix computation or
    densification. Serialization finishes in staging before existing data is moved.

    Matrices are stored in a layout suited to lazy reading. Matrices are the
    values of ``X``, of each entry in ``layers``, ``obsm``, ``varm``, ``obsp`` and
    ``varp``, and of ``raw.X`` and each entry in ``raw.varm``. They can be dense
    (NumPy arrays, or Dask arrays with NumPy blocks) or sparse (SciPy CSR or CSC
    matrices, or Dask arrays with such blocks). Dense matrices are stored in
    chunks of whole rows of about 4 MiB, and sparse matrices in chunks of at most
    4 MiB, whatever chunks or blocks they arrive in and whatever Dask's
    ``array.chunk-size``. Everything else is stored as AnnData stores it by
    default: ``obs`` and ``var``, dataframes in ``obsm``, everything in ``uns``,
    and arrays of strings. Sharding follows AnnData's ``auto_shard_zarr_v3``
    setting.

    Validation checks table structure and SpatialData annotation, not scientific
    metadata.

    Returns None after publication and metadata finalization. This path-based
    writer does not update live SpatialData objects or existing references;
    reopen affected data after writing. Handled failures attempt rollback.
    Crash recovery and concurrent-access isolation are not provided.

    See Also
    --------
    harpy.table.read_table : Reopen the written table.
    harpy.table.write_table_components : Replace only selected components.

    Examples
    --------
    .. code-block:: python

        adata = hp.tb.read_table("sdata.zarr", table_name="counts")
        adata.X = adata.X * 2
        hp.tb.write_table("sdata.zarr", table_name="counts", adata=adata, overwrite=True)
        adata = hp.tb.read_table("sdata.zarr", table_name="counts")
    """
    if not isinstance(adata, AnnData):
        raise TypeError("adata must be an AnnData.")
    with _write_table_operation(store, table_name=table_name, adata=adata, overwrite=overwrite):
        pass


def write_table_components(
    store: str | PathLike[str],
    *,
    table_name: str,
    components: Mapping[ComponentPath, object],
    delete: Sequence[ComponentPath] = (),
    obs_identity: pd.DataFrame | AxisNames | None = None,
    var_names: AxisNames | None = None,
    raw_var_names: AxisNames | None = None,
    overwrite: bool = False,
) -> None:
    """Write selected components of an existing table as one rollback-protected update.

    Parameters
    ----------
    store
        Local path to an existing SpatialData Zarr root. Its Zarr format is preserved.
    table_name
        Name of the existing table.
    components
        Nonempty mapping of non-overlapping logical tuple paths to replacement
        values. Supports X, whole obs/var dataframes, individual layers/obsm/varm/
        obsp/varp entries, whole or nested uns, and X/var/varm entries in raw data.
        To create raw, supply raw.X and either raw_var_names or a raw.var dataframe;
        raw.varm entries may accompany them. Mapping roots other than uns,
        dataframe columns, matrix slices and encoding internals cannot be written.
        Mappings replace their contents. Unrequested components remain unchanged.
        None is an encoded value where supported, not deletion.
    delete
        Optional component paths to remove in the same operation. Supports the
        targets of :func:`harpy.table.delete_table_components`. Replacement and
        deletion paths must be unique and non-overlapping, even when absent.
        Missing deletion targets are logged at INFO and skipped.
    obs_identity
        Ordered observation identities for X, layers, obsm, obsp or raw.X.
        For annotated tables, a two-column dataframe using the stored region and
        instance keys, for example ``adata.obs[[region_key, instance_key]]``;
        its index is ignored, so observation names are not checked through it.
        For unannotated tables, the ordered observation names
        (``adata.obs_names``, equivalent to ``adata.obs.index``).
        A supplied obs dataframe provides this context instead; as a replacement,
        its index must also equal the stored obs index (see Notes).
        Identities must be unique and match the stored observations in value and order.
    var_names
        Ordered feature names for X, layers, varm or varp. A supplied var dataframe
        provides these instead. Names must be unique and match the stored axis.
    raw_var_names
        Equivalent feature identities for raw.X and raw.varm, using raw's independent
        feature axis. A supplied raw.var dataframe provides these instead. When
        creating raw from names alone, raw.var has this index and no annotation columns.
    overwrite
        Allow replacement of existing requested components; otherwise only new
        entries are allowed. A raw stored as None, which AnnData writes for every
        table without raw, counts as absent. Paths in ``delete`` are removed
        whatever ``overwrite``: naming them is the permission.

    Notes
    -----
    Existing axes, their order and SpatialData linkage cannot change; use
    :func:`harpy.table.write_table` for those changes. No automatic reordering
    occurs. If both a dataframe and explicit identities are supplied, both are
    checked. Dataframe replacements must also preserve their stored index.
    Callers prepare consistent scientific data and metadata.

    Missing raw data can be created without rewriting the table. Its observations
    match the existing table, while the supplied features define a new raw axis.
    Creation requires a non-None raw.X; raw.var or raw.varm alone is insufficient.

    Inputs are not modified; matrices retain their original representations.

    Zarr-backed matrices are internally wrapped in lazy Dask arrays and evaluated
    in chunks during writing, without preliminary whole-matrix computation or
    densification. Stored layouts follow :func:`harpy.table.write_table`, whose
    Notes list which elements are stored as matrices.

    An obs-only update does not read or rewrite X.
    Related matrix and metadata updates should be submitted in the same call.
    A lazy replacement may need to read a component scheduled for deletion in
    the same request. We therefore finish serializing all replacements into
    staging before moving any deletion targets from their original locations.
    Replacements and deletions share one rollback operation. Separate API calls
    commit independently.
    Staging, completion, reopening and recovery follow :func:`harpy.table.write_table`.

    See Also
    --------
    harpy.table.read_table_components : Read only selected components.
    harpy.table.write_table : Write a complete table.
    harpy.table.delete_table_components : Remove components without replacements.

    Examples
    --------
    .. code-block:: python

        hp.tb.write_table_components(
            "sdata.zarr", table_name="counts",
            components={("obsm", "embedding"): embedding, ("uns", "embedding"): metadata},
            obs_identity=adata.obs[[region_key, instance_key]], overwrite=True,
        )

        hp.tb.write_table_components(
            "sdata.zarr", table_name="counts",
            components={("raw", "X"): raw_counts},
            obs_identity=adata.obs[[region_key, instance_key]],
            raw_var_names=gene_names, overwrite=True,
        )
    """
    if not isinstance(components, Mapping):
        raise TypeError("components must be a mapping from tuple paths to values.")
    if not components:
        raise ValueError("components must not be empty; use delete_table_components() for deletion only.")
    with _write_table_operation(
        store,
        table_name=table_name,
        components=components,
        delete=delete,
        obs_identity=obs_identity,
        var_names=var_names,
        raw_var_names=raw_var_names,
        overwrite=overwrite,
    ):
        pass


def delete_table_components(
    store: str | PathLike[str], *, table_name: str, components: Sequence[ComponentPath]
) -> None:
    """Remove optional table components as one rollback-protected update.

    Parameters
    ----------
    store
        Local path to an existing SpatialData Zarr root. Its Zarr format is preserved.
    table_name
        Name of the existing table.
    components
        Nonempty sequence of unique, non-overlapping logical tuple paths.
        Allowed targets are:

        1. Entire X or raw: ``("X",)`` or ``("raw",)``.
        2. Individual entries in layers, obsm, varm, obsp, varp and raw.varm,
           such as ``("obsm", "embedding")`` or ``("raw", "varm", "loadings")``.
        3. Individual or nested uns records, such as ``("uns", "analysis", "method")``,
           except ``uns["spatialdata_attrs"]`` and its descendants.

        Required axes (obs, var, raw.var), raw.X alone and whole mapping
        containers (such as obsm or uns) cannot be deleted.

    Notes
    -----
    Naming a target authorizes its removal; no overwrite or identity arguments
    are needed. Missing targets are logged at INFO and skipped. If all are
    absent, no staging or metadata writes occur. Invalid paths, malformed parents
    and I/O errors still raise. A stored None is a present value, not a missing path.

    Deletion does not read component payloads. Unrequested components and parent
    mappings remain intact. Removing X preserves the table's shape, obs and var;
    reopening returns X=None. Removing raw returns raw=None. Callers must explicitly
    identify related scientific records to remove; no cascading deletion occurs.

    Returns None after publication and metadata finalization. Live AnnData and
    SpatialData objects are not updated; reopen affected data after deletion.
    Handled failures attempt to restore removed data and store-root metadata.
    Crash recovery and concurrent-access isolation are not provided.

    See Also
    --------
    harpy.table.write_table_components : Combine replacements and deletions.

    Examples
    --------
    .. code-block:: python

        hp.tb.delete_table_components(
            "sdata.zarr", table_name="counts",
            components=[("obsm", "cell_features"), ("uns", "feature_matrices", "cell_features")],
        )
    """
    if not components:
        raise ValueError("components must not be empty.")
    with _write_table_operation(store, table_name=table_name, components={}, delete=components):
        pass


@contextmanager
def _write_table_operation(
    store: str | PathLike[str],
    *,
    table_name: str,
    adata: AnnData | None = None,
    components: Mapping[ComponentPath, object] | None = None,
    delete: Sequence[ComponentPath] = (),
    obs_identity: pd.DataFrame | AxisNames | None = None,
    var_names: AxisNames | None = None,
    raw_var_names: AxisNames | None = None,
    overwrite: bool = False,
) -> Generator[zarr.Group, None, None]:
    """Stage, validate and publish one table update; commit after the caller succeeds.

    Supply either adata or components. Component updates may include deletions;
    only deletion-only callers may supply an empty components mapping.
    The yielded read-only group uses permanent paths. Adapters can reopen/attach
    there while backups remain; they own any
    in-memory rollback. Public path-based writers use an empty with-body because
    they do not attach data. Final consolidation is part of the rollback window.

    When supplied, obs_identity, var_names and raw_var_names must agree with
    adata's corresponding axes, or with the expected axes for component writes.
    They are checked even when axis dataframes are also supplied; no reordering
    occurs. Complete-table replacement may change the destination's old axes.
    """
    if (adata is None) == (components is None):
        raise ValueError("Supply either adata or components, not both.")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean.")
    deletion_paths = _validate_deletion_paths(delete)
    if adata is not None and deletion_paths:
        raise ValueError("Component deletion cannot accompany complete-table replacement.")
    if components is not None and not components and not deletion_paths:
        raise ValueError("Supply at least one replacement or deletion.")
    _validate_path_segment(table_name)
    source_root = _open_spatialdata_group(store)
    root = Path(store)
    table_path = root / "tables" / table_name
    for kind in ("images", "labels", "points", "shapes"):
        if f"{kind}/{table_name}" in source_root:
            raise ValueError(f"Table name {table_name!r} collides with a {kind} element.")
    if "tables" in source_root and not isinstance(source_root["tables"], zarr.Group):
        raise ValueError("The tables container must be a Zarr group.")

    if adata is not None:
        _check_write_destination(table_path, root=root, overwrite=overwrite)
        _validate_complete_table(adata)
        _validate_table_identities(adata, obs_identity=obs_identity, var_names=var_names, raw_var_names=raw_var_names)
    else:
        assert components is not None
        paths = _validate_component_paths(tuple(components), to_write=True) if components else ()
        # Check the original request before filtering absent deletions: a path
        # cannot be both written and deleted, even if it does not exist yet.
        _check_component_path_overlap((*paths, *deletion_paths))
        if table_path.is_symlink() or not table_path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Unsafe table-write destination: {table_path}.")
        source_table = _open_table_group(store, table_name=table_name)
        present_deletions = []
        for path in deletion_paths:
            if _check_component_deletion_destination(source_table, path, table_path=table_path, root=root):
                present_deletions.append(path)
            else:
                log.info(f"Table {table_name!r}: component {path!r} is already absent; skipping deletion.")
        new_raw_var = _prepare_raw_creation(source_table, components, raw_var_names=raw_var_names)
        expected_new_raw_var_index = None if new_raw_var is None else new_raw_var.index
        # This flag means creating the raw container, not merely writing raw components.
        # It is False when raw already exists, even if its components will be updated,
        # or when no raw components were requested.
        create_raw = new_raw_var is not None
        if create_raw:
            # If create_raw is True, raw components were requested, but raw is
            # absent or stored as None.
            # Unlike an obsm mapping, raw defines its own feature axis via var.
            # Below we assemble X, var and optional varm into one encoded Raw
            # container, so publication targets the whole raw path, not its
            # children. Rollback can then restore the original absence or None.

            # Zarr may not recognize an existing file/directory as a child.
            # Requests target raw's components, so even overwrite=True must not
            # replace an unrecognized raw parent with the new container.
            if "raw" not in source_table and (table_path / "raw").exists():
                raise ValueError("Cannot create raw over an existing unrecognized raw path.")
            # Check path safety only: raw is absent here, or an encoded None that
            # _prepare_raw_creation() validated and that counts as absent.
            _check_write_destination(table_path / "raw", root=root, overwrite=True)
            components = dict(components)
            components[("raw", "var")] = new_raw_var
            paths = tuple(components)
        for path in paths:
            if create_raw and path[0] == "raw":
                continue
            _check_component_write_destination(
                source_table, path, table_path=table_path, root=root, overwrite=overwrite
            )
        # Prepare the expected axes once, including any new raw feature index.
        # Both validation rounds use these indices, not axes from staged values.
        expected_axis_indices = _storage_axis_indices(
            source_table,
            components,
            obs_identity=obs_identity,
            var_names=var_names,
            raw_var_names=raw_var_names,
            expected_new_raw_var_index=expected_new_raw_var_index,
        )
        # 1) Validate caller-supplied shapes, identities and linkage before staging.
        # Step 2 below repeats these checks on the serialized output.
        _validate_component_values_against_storage(
            source_table,
            components_to_validate=components,
            expected_axis_indices=expected_axis_indices,
            obs_identity=obs_identity,
            var_names=var_names,
            raw_var_names=raw_var_names,
        )
        if not paths and not present_deletions:
            # A validated all-missing deletion request must not create a
            # workspace or rewrite the store's consolidated metadata.
            yield source_table
            return
    workspace = Path(tempfile.mkdtemp(prefix=f".{root.name}.harpy-table-staging-", dir=root.parent))
    try:
        staged_root = zarr.open_group(str(workspace), mode="w", zarr_format=source_root.metadata.zarr_format)
        replacements: list[_StagedPath | _DeletedPath] = []
        if adata is not None:
            _write_anndata_element(staged_root, ("table",), adata, logical_path=(), create_parents=False)
            staged_table = _read_anndata_table(staged_root["table"], mode="lazy")
            _validate_complete_table(staged_table)
            _validate_table_identities(
                staged_table, obs_identity=obs_identity, var_names=var_names, raw_var_names=raw_var_names
            )
            annotation = staged_table.uns.get(TableModel.ATTRS_KEY)
            regions = None if annotation is None else annotation[TableModel.REGION_KEY]
            if isinstance(regions, str):
                regions = [regions]
            _write_spatialdata_table_attrs(
                staged_root["table"],
                regions=regions,
                region_key=None if annotation is None else annotation[TableModel.REGION_KEY_KEY],
                instance_key=None if annotation is None else annotation[TableModel.INSTANCE_KEY],
            )
            replacements.append(_StagedPath(workspace / "table", table_path))
        else:
            assert components is not None
            if create_raw:
                matrix = components[("raw", "X")]
                # This Raw is only a serialization payload: the shape-only parent
                # supplies n_obs without loading the actual table. AnnData writes
                # only X, var and varm, not this parent's placeholder obs names.
                # _validate_component_values_against_storage() already checked row identities
                # against the stored table's real observations.
                raw = Raw(
                    AnnData(shape=(matrix.shape[0], 0)),
                    X=matrix,
                    var=new_raw_var,
                    varm={path[2]: value for path, value in components.items() if path[:2] == ("raw", "varm")},
                )
                _write_anndata_element(staged_root, ("raw",), raw, logical_path=("raw",), create_parents=False)
                replacements.append(_StagedPath(workspace / "raw", table_path / "raw"))
            # Read back the new raw components from their shared container;
            # other replacements are serialized and published individually.
            staged_values = {}
            for ordinal, path in enumerate(paths):
                if create_raw and path[0] == "raw":
                    staged_path = path
                else:
                    staged_name = f"component-{ordinal}"
                    staged_path = (staged_name,)
                    # The temporary name hides the slot, so pass the logical path.
                    _write_anndata_element(
                        staged_root, staged_path, components[path], logical_path=path, create_parents=False
                    )
                    replacements.append(_StagedPath(workspace / staged_name, table_path.joinpath(*path)))
                # Temporary names hide the original slot, so explicitly preserve
                # uns's eager-reading policy. Unlike uns, obs/var/raw.var are
                # recognized as dataframes and decoded eagerly in any mode.
                mode = "eager" if path[0] == "uns" else "backed"
                staged_values[path] = _read_anndata_element(staged_root, staged_path, mode=mode)
            # 2) Repeat step 1 on the reopened staged components, before publication.
            # This checks the serialized output rather than the caller-supplied inputs.
            # Backed matrix handles allow shape checks without scanning matrix values.
            _validate_component_values_against_storage(
                source_table,
                components_to_validate=staged_values,
                expected_axis_indices=expected_axis_indices,
                obs_identity=obs_identity,
                var_names=var_names,
                raw_var_names=raw_var_names,
            )
            # Register deletion targets alongside staged replacements;
            # _publish_table_paths() processes these instructions below.
            replacements.extend(_DeletedPath(table_path.joinpath(*path)) for path in present_deletions)

        # All replacement serialization has finished, including any reads from
        # deletion targets (e.g. X derived from a layer being deleted). Publication
        # can now move existing targets into backups and install replacements,
        # keeping removals and replacements within the same rollback window.
        with _publish_table_paths(
            root=root, table_name=table_name, workspace=workspace, replacements=replacements
        ) as published:
            yield published
    finally:
        _cleanup_owned_path(workspace)


@contextmanager
def _publish_table_paths(
    *,
    root: Path,
    table_name: str,
    workspace: Path,
    replacements: Sequence[_StagedPath | _DeletedPath],
) -> Generator[zarr.Group, None, None]:
    """Publish validated table payloads and finalize metadata after caller installation.

    Writing and publication are separate steps: callers first write and validate
    a table or its components in staging. This helper then moves those existing
    paths to their final destinations without serializing the data again.

    Callers may prepare staged data differently, but share this flow::

        Caller:      write and validate staged data
        This helper: publish paths and yield the reopened table
        Caller:      finish its with-body (e.g. attach the table in memory)
        This helper: consolidate metadata and finalize publication

    Parameters
    ----------
    root
        Existing local SpatialData Zarr store. No permanent paths may have been
        changed yet: its root metadata is saved before creating missing parents.
    table_name
        Table to reopen read-only from permanent paths and yield to the caller.
    workspace
        Owned staging directory, cleaned on success or failure. Callers also
        clean up failures during preparation, before entering this operation.
    replacements
        Pairs of fully serialized staged paths and permanent destinations for
        this table or its components, optionally accompanied by explicit deletion
        destinations without staged payloads. Callers must validate the AnnData paths
        and payloads, overwrite permission, identities and domain constraints.
        This helper checks filesystem safety; it does not determine whether
        arbitrary paths represent valid AnnData components.

    Notes
    -----
    Installation and any final domain validation run in the caller's with-body
    while backups remain available. Consolidation runs only after that body
    succeeds. Failure restores payloads, newly created parents and saved root
    metadata; the caller restores affected in-memory references. The shared
    publisher's crash-recovery and concurrent-access limitations still apply.
    """
    created_parents: list[Path] = []
    root_metadata_before = {}
    metadata_attempted = False
    try:
        _validate_path_segment(table_name)
        source_root = _open_spatialdata_group(root)
        table_path = root / "tables" / table_name
        destinations = tuple(replacement.destination for replacement in replacements)
        for destination in destinations:
            if (
                destination.is_symlink()
                or not destination.is_relative_to(table_path)
                or not destination.resolve().is_relative_to(root.resolve())
            ):
                raise ValueError(f"Unsafe table-write destination: {destination}.")
        # The zarr.consolidate_metadata(str(root)) call below rewrites the SpatialData
        # Zarr store's root metadata, which is not included in the paths backed up by
        # _publish_staged_paths(). Keep its exact prior bytes so even a failed finalization
        # can be undone without opening unrelated tables or depending on consolidation
        # succeeding again.
        metadata_files = (
            (".zgroup", ".zattrs", ".zmetadata") if source_root.metadata.zarr_format == 2 else ("zarr.json",)
        )
        for filename in metadata_files:
            metadata_path = root / filename
            if metadata_path.is_symlink():
                raise ValueError(f"Refusing to update symbolic-link metadata path: {metadata_path}.")
            root_metadata_before[metadata_path] = metadata_path.read_bytes() if metadata_path.exists() else None
        writable_root = zarr.open_group(str(root), mode="r+", use_consolidated=False)
        replacement_destinations = tuple(item.destination for item in replacements if isinstance(item, _StagedPath))
        _create_destination_parents(writable_root, root, replacement_destinations, created_parents)
        with _publish_staged_paths(root=root, workspace=workspace, paths=replacements, operation="table"):
            # Read the published paths directly (use_consolidated=False), because
            # consolidated metadata may still describe the previous table/components.
            published_root = zarr.open_group(str(root), mode="r", use_consolidated=False)
            # Let the caller of _publish_table_paths() finish its work
            # (i.e. we yield before consolidating metadata),
            # such as attaching the reopened table to sdata in memory, while backups
            # remain available. Consolidate only after the caller's with-body succeeds,
            # so failed installation can roll back without first rewriting the store's
            # root metadata.
            yield published_root["tables"][table_name]
            metadata_attempted = True
            # SpatialData also stores non-Zarr payloads (for example Parquet).
            # Ignore those routine discovery warnings, not I/O failures.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="Object at .* is not recognized", category=zarr.errors.ZarrUserWarning
                )
                warnings.filterwarnings(
                    "ignore",
                    message="Consolidated metadata is currently not part",
                    category=zarr.errors.ZarrUserWarning,
                )
                zarr.consolidate_metadata(str(root))
    except BaseException as error:
        # _publish_staged_paths() attempts rollback of the table/component paths.
        # Newly created parents and root metadata are outside its scope, so restore
        # them here. This also covers setup failures before entering the publisher.
        restoration_errors = []
        for parent in reversed(created_parents):
            try:
                _remove_owned_path(parent)
            except BaseException as restore_error:  # noqa: BLE001 - finish other restoration attempts, then report
                restoration_errors.append(f"{parent}: {restore_error}")
        if metadata_attempted:
            for metadata_path, previous_bytes in root_metadata_before.items():
                try:
                    if previous_bytes is None:
                        metadata_path.unlink(missing_ok=True)
                    else:
                        metadata_path.write_bytes(previous_bytes)
                except BaseException as restore_error:  # noqa: BLE001 - preserve all restoration failures
                    restoration_errors.append(f"{metadata_path}: {restore_error}")
        if restoration_errors:
            raise RuntimeError(
                f"Table write failed; restoration also failed: {'; '.join(restoration_errors)}"
            ) from error
        raise
    finally:
        _cleanup_owned_path(workspace)


def _create_destination_parents(
    group: zarr.Group, root: Path, destinations: tuple[Path, ...], created: list[Path]
) -> None:
    """Create only missing destination parents and record ownership before writing."""
    for destination in destinations:
        parent = group
        parent_path = root
        parts = destination.relative_to(root).parts
        for index, key in enumerate(parts[:-1]):
            parent_path = parent_path / key
            if key not in parent:
                if parent_path.exists() or parent_path.is_symlink():
                    raise ValueError(f"Cannot create a Zarr parent over an existing unrecognized path: {parent_path}.")
                created.append(parent_path)
                if parent_path == root / "tables":
                    parent.create_group(key)
                else:
                    # Destinations are tables/<table name>/<logical path>.
                    _write_anndata_element(parent, (key,), {}, logical_path=parts[2 : index + 1], create_parents=False)
            parent = parent[key]
