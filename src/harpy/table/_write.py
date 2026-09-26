"""Scoped table writes using AnnData serialization and shared path publication."""

from __future__ import annotations

import tempfile
import warnings
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from os import PathLike
from pathlib import Path

import pandas as pd
import zarr
from anndata import AnnData, Raw
from spatialdata.models import TableModel

from harpy._storage._anndata import (
    _read_anndata_element,
    _read_anndata_table,
    _write_anndata_element,
    _write_spatialdata_table_attrs,
)
from harpy._storage._publication import _cleanup_owned_path, _publish_staged_paths, _remove_owned_path, _StagedPath
from harpy._storage._spatialdata import _open_spatialdata_group
from harpy.table._io import ComponentPath, _open_table_group, _validate_component_paths, _validate_path_segment
from harpy.table._write_validation import (
    AxisNames,
    _check_component_destination,
    _prepare_raw_creation,
    _validate_complete_table,
    _validate_component_values,
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
        Mappings replace their contents;
        omitted components remain unchanged. None encodes absence where supported,
        not deletion.
    obs_identity
        Ordered observation identities for X, layers, obsm, obsp or raw.X.
        For annotated tables, a two-column dataframe using the stored region and
        instance keys, for example ``adata.obs[[region_key, instance_key]]``;
        its index is ignored. For unannotated tables, the ordered observation names
        (``adata.obs_names``, equivalent to ``adata.obs.index``).
        A supplied obs dataframe provides this context instead. Identities must
        be unique and match the stored observations in value and order.
    var_names
        Ordered feature names for X, layers, varm or varp. A supplied var dataframe
        provides these instead. Names must be unique and match the stored axis.
    raw_var_names
        Equivalent feature identities for raw.X and raw.varm, using raw's independent
        feature axis. A supplied raw.var dataframe provides these instead. When
        creating raw from names alone, raw.var has this index and no annotation columns.
    overwrite
        Allow replacement of existing requested components; otherwise only new
        entries are allowed. Creating raw over a stored None also requires True.

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
    densification.

    An obs-only update does not read or rewrite X.
    Related matrix and metadata updates should be submitted in the same call.
    Staging, completion, reopening and recovery follow :func:`harpy.table.write_table`.

    See Also
    --------
    harpy.table.read_table_components : Read only selected components.
    harpy.table.write_table : Write a complete table.

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
    with _write_table_operation(
        store,
        table_name=table_name,
        components=components,
        obs_identity=obs_identity,
        var_names=var_names,
        raw_var_names=raw_var_names,
        overwrite=overwrite,
    ):
        pass


@contextmanager
def _write_table_operation(
    store: str | PathLike[str],
    *,
    table_name: str,
    adata: AnnData | None = None,
    components: Mapping[ComponentPath, object] | None = None,
    obs_identity: pd.DataFrame | AxisNames | None = None,
    var_names: AxisNames | None = None,
    raw_var_names: AxisNames | None = None,
    overwrite: bool = False,
) -> Generator[zarr.Group, None, None]:
    """Stage, validate and publish one table update; commit after the caller succeeds.

    Supply either adata or components. The yielded read-only group uses permanent
    paths. Adapters can reopen/attach there while backups remain; they own any
    in-memory rollback. Public path-based writers use an empty with-body because
    they do not attach data. Final consolidation is part of the rollback window.
    """
    if (adata is None) == (components is None):
        raise ValueError("Supply either adata or components, not both.")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean.")
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
        destinations = (table_path,)
        if table_path.exists() and not overwrite:
            raise FileExistsError(f"Table {table_name!r} already exists; use overwrite=True.")
        _validate_complete_table(adata)
    else:
        assert components is not None
        paths = _validate_component_paths(tuple(components), to_write=True)
        source_table = _open_table_group(store, table_name=table_name)
        new_raw_var = _prepare_raw_creation(source_table, components, raw_var_names=raw_var_names, overwrite=overwrite)
        # Use the same expected raw feature index before and after staging.
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
            components = dict(components)
            components[("raw", "var")] = new_raw_var
            paths = tuple(components)
            publication_paths = (("raw",), *(path for path in paths if path[0] != "raw"))
        else:
            # Existing raw containers keep unrequested components untouched.
            publication_paths = paths
        destinations = tuple(table_path.joinpath(*path) for path in publication_paths)
        for path in paths:
            if create_raw and path[0] == "raw":
                continue
            _check_component_destination(source_table, path, overwrite=overwrite)
        # 1) Validate caller-supplied shapes, identities and linkage before staging.
        # Step 2 below repeats these checks on the serialized output.
        _validate_component_values(
            source_table,
            components_to_validate=components,
            obs_identity=obs_identity,
            var_names=var_names,
            raw_var_names=raw_var_names,
            expected_new_raw_var_index=expected_new_raw_var_index,
        )
    for destination in destinations:
        if destination.is_symlink() or not destination.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Unsafe table-write destination: {destination}.")
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Destination {destination} already exists; use overwrite=True.")

    workspace = Path(tempfile.mkdtemp(prefix=f".{root.name}.harpy-table-staging-", dir=root.parent))
    created_parents: list[Path] = []
    # The zarr.consolidate_metadata(str(root)) call below rewrites the SpatialData
    # Zarr store's root metadata, which is not included in the paths backed up by
    # _publish_staged_paths(). Keep its exact prior bytes so even a failed finalization
    # can be undone without opening unrelated tables or depending on consolidation
    # succeeding again.
    metadata_files = (".zgroup", ".zattrs", ".zmetadata") if source_root.metadata.zarr_format == 2 else ("zarr.json",)
    root_metadata_before = {}
    metadata_attempted = False
    try:
        staged_root = zarr.open_group(str(workspace), mode="w", zarr_format=source_root.metadata.zarr_format)
        replacements: list[_StagedPath] = []
        if adata is not None:
            _write_anndata_element(staged_root, ("table",), adata, create_parents=False)
            staged_table = _read_anndata_table(staged_root["table"], mode="lazy")
            _validate_complete_table(staged_table)
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
                # _validate_component_values() already checked row identities
                # against the stored table's real observations.
                raw = Raw(
                    AnnData(shape=(matrix.shape[0], 0)),
                    X=matrix,
                    var=new_raw_var,
                    varm={path[2]: value for path, value in components.items() if path[:2] == ("raw", "varm")},
                )
                _write_anndata_element(staged_root, ("raw",), raw, create_parents=False)
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
                    _write_anndata_element(staged_root, staged_path, components[path], create_parents=False)
                    replacements.append(_StagedPath(workspace / staged_name, table_path.joinpath(*path)))
                # Temporary names hide the original slot, so explicitly preserve
                # uns's eager-reading policy. Unlike uns, obs/var/raw.var are
                # recognized as dataframes and decoded eagerly in any mode.
                mode = "eager" if path[0] == "uns" else "backed"
                staged_values[path] = _read_anndata_element(staged_root, staged_path, mode=mode)
            # 2) Repeat step 1 on the reopened staged components, before publication.
            # This checks the serialized output rather than the caller-supplied inputs.
            # Backed matrix handles allow shape checks without scanning matrix values.
            _validate_component_values(
                source_table,
                components_to_validate=staged_values,
                obs_identity=obs_identity,
                var_names=var_names,
                raw_var_names=raw_var_names,
                expected_new_raw_var_index=expected_new_raw_var_index,
            )

        for filename in metadata_files:
            metadata_path = root / filename
            if metadata_path.is_symlink():
                raise ValueError(f"Refusing to update symbolic-link metadata path: {metadata_path}.")
            root_metadata_before[metadata_path] = metadata_path.read_bytes() if metadata_path.exists() else None
        writable_root = zarr.open_group(str(root), mode="r+", use_consolidated=False)
        _create_destination_parents(writable_root, root, destinations, created_parents)
        with _publish_staged_paths(root=root, workspace=workspace, paths=replacements, operation="table"):
            # Read the published paths directly (use_consolidated=False), because
            # consolidated metadata may still describe the previous table/components.
            published_root = zarr.open_group(str(root), mode="r", use_consolidated=False)
            # Let the caller of _write_table_operation() finish its work
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
        for key in destination.relative_to(root).parts[:-1]:
            parent_path = parent_path / key
            if key not in parent:
                if parent_path.exists() or parent_path.is_symlink():
                    raise ValueError(f"Cannot create a Zarr parent over an existing unrecognized path: {parent_path}.")
                created.append(parent_path)
                if parent_path == root / "tables":
                    parent.create_group(key)
                else:
                    _write_anndata_element(parent, (key,), {}, create_parents=False)
            parent = parent[key]
