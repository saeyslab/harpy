"""Selective updates of live AnnData components, with shared storage publication."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import copy

import numpy as np
import pandas as pd
import zarr
from anndata import AnnData
from anndata._core.raw import Raw
from loguru import logger as log
from spatialdata import SpatialData
from spatialdata.models import TableModel

from harpy._storage._anndata import _MissingAnnDataElement, _read_anndata_element
from harpy.table.io._read import (
    ComponentPath,
    _check_component_path_overlap,
    _open_table_group,
    _validate_component_paths,
    _validate_path_segment,
)
from harpy.table.io._write import _write_table_operation
from harpy.table.io._write_validation import (
    AxisNames,
    _annotation_columns,
    _component_axes,
    _match_identity,
    _new_raw_var,
    _prepare_raw_creation,
    _read_observation_identity,
    _read_spatialdata_attrs,
    _storage_axis_index,
    _validate_component_values_against_axes,
    _validate_deletion_paths,
    _validated_observation_pairs,
)

_ABSENT = object()


def add_table_components(
    sdata: SpatialData,
    *,
    table_name: str,
    components: Mapping[ComponentPath, object],
    delete: Sequence[ComponentPath] = (),
    obs_identity: pd.DataFrame | AxisNames | None = None,
    var_names: AxisNames | None = None,
    raw_var_names: AxisNames | None = None,
    overwrite: bool = False,
) -> SpatialData:
    """Update selected table components in SpatialData and, when backed, its store.

    Keep the existing AnnData object and update only the requested components,
    preserving unrelated local edits and matrix references.

    For example, replace only an embedding in backed SpatialData while keeping
    an annotation column that has not yet been saved::

        Requested update: obsm["embedding"]

        Component                   In memory after update     On disk after update
        obsm["embedding"]           Reopened replacement       Replacement written
        obs["manual_annotation"]    Unsaved column retained    Not written
        X                           Same matrix reference      Unchanged

    Parameters
    ----------
    sdata
        SpatialData to update. Its path determines whether changes are also
        persisted; matrix representations and AnnData.isbacked do not select
        the storage mode.
    table_name
        Existing table in sdata.tables, also required in the store when backed.
        The attached AnnData must not be a view or an HDF5-backed object.
    components
        Nonempty mapping from logical tuple paths to replacement values, using
        the same scopes as :func:`harpy.table.io.write_table_components`: X,
        obs/var dataframes, individual matrix-mapping entries, whole or nested
        uns, and raw.X/raw.var/raw.varm entries. For example,
        ``{("obsm", "embedding"): embedding, ("uns", "embedding"): metadata}``.
        Mappings replace their contents; omitted paths remain unchanged.
        None is a replacement value where supported, not a deletion instruction.
    delete
        Optional explicit deletion paths, using the scopes of
        :func:`harpy.table.io.delete_table_components`. All replacement and deletion
        paths must be unique and non-overlapping. Missing targets in both
        memory and storage are logged and skipped.
    obs_identity
        Ordered identities for observation-aligned replacements. For annotated
        tables, use ``adata.obs[[region_key, instance_key]]``; its index is
        ignored, so observation names are not checked through it. For
        unannotated tables, use observation names. Supplying ``("obs",)``
        provides this context instead, and its index must also equal the stored
        obs index. Both are checked if supplied.
    var_names
        Ordered feature names for main-feature-aligned replacements. A supplied
        ``("var",)`` dataframe provides this context instead.
    raw_var_names
        Ordered names on raw's independent feature axis. A supplied
        ``("raw", "var")`` provides these instead. Creating missing raw requires
        a non-None raw.X and either these names or a raw.var dataframe.
    overwrite
        Allow replacing targets that exist in the store. Ignored for unbacked
        SpatialData, matching :func:`harpy.table.io.add_table`. It only concerns
        the store: a target present only in memory, never saved, is replaced
        without it, as for unbacked SpatialData. Explicit deletions do not
        require overwrite permission. Validation still applies in both modes.

    Returns
    -------
    spatialdata.SpatialData
        The supplied sdata, with only requested components updated in its
        existing table object.

    Notes
    -----
    Axes and SpatialData linkage cannot change. Supply replacement identities
    even though sdata is provided; matrices are never reordered automatically.
    Backed updates additionally require the relevant in-memory axes to match storage.

    Unbacked updates perform no serialization or matrix computation and retain
    matrix representations. Supplied matrix data may be shared; this is not a
    deep-copy API. Backed updates use the shared staged writer, then reopen only
    affected entries, with matrices lazy and annotations in memory. Backed
    writes store matrices as :func:`harpy.table.io.write_table` does; its Notes
    list which elements are stored as matrices.

    Installation finishes before disk publication commits. Handled failures
    restore affected live references as well as stored data. No crash recovery
    or concurrent-access isolation is provided. External references to replaced
    values are not refreshed; callers own dirty/stale tracking.

    See Also
    --------
    harpy.table.io.write_table_components : Update storage without modifying live objects.
    harpy.table.io.remove_table_components : Remove selected live and stored components.
    """
    if not isinstance(components, Mapping):
        raise TypeError("components must be a mapping from tuple paths to values.")
    if not components:
        raise ValueError("components must not be empty; use remove_table_components() for deletion only.")
    return _update_table_components(
        sdata,
        table_name=table_name,
        components=components,
        delete=delete,
        obs_identity=obs_identity,
        var_names=var_names,
        raw_var_names=raw_var_names,
        overwrite=overwrite,
    )


def remove_table_components(sdata: SpatialData, *, table_name: str, components: Sequence[ComponentPath]) -> SpatialData:
    """Remove selected components from a live table and, when backed, its store.

    Parameters
    ----------
    sdata
        SpatialData to update. With no path, only memory is changed.
    table_name
        Existing attached table, also required on disk for backed SpatialData.
        AnnData views and HDF5-backed destination objects are not supported.
    components
        Nonempty sequence of unique, non-overlapping logical tuple paths.
        Supports X, entire raw, individual layers/obsm/varm/obsp/varp/raw.varm
        entries, and individual or nested uns records. Required axes, whole
        mapping containers and SpatialData annotation cannot be deleted.
        For example, ``[("obsm", "embedding"), ("uns", "embedding")]``.

    Returns
    -------
    spatialdata.SpatialData
        The supplied sdata, with requested components removed.

    Notes
    -----
    Explicit paths authorize deletion; no identities or overwrite flag are
    required. Presence is resolved independently in memory and storage. Targets
    absent from both are logged and skipped. Removing a memory-only entry does
    not create staging or rewrite store metadata. Invalid paths and malformed
    parents still raise. Unrelated entries and empty parent mappings remain.

    Related measurements and scientific metadata must be listed explicitly;
    deletions do not cascade. To combine replacements and removals in one
    rollback operation, use :func:`harpy.table.io.add_table_components` with delete.
    Its installation, recovery and external-reference rules also apply here.
    """
    if not components:
        raise ValueError("components must not be empty.")
    return _update_table_components(sdata, table_name=table_name, components={}, delete=components)


def _update_table_components(
    sdata: SpatialData,
    *,
    table_name: str,
    components: Mapping[ComponentPath, object],
    delete: Sequence[ComponentPath],
    obs_identity: pd.DataFrame | AxisNames | None = None,
    var_names: AxisNames | None = None,
    raw_var_names: AxisNames | None = None,
    overwrite: bool = False,
    reopen_also: Sequence[ComponentPath] = (),
) -> SpatialData:
    """Validate both destinations, then install inside the writer's rollback window.

    For backed SpatialData, the written components are reopened lazily from the
    store and installed in the attached table, so that their slots refer to what
    is stored. ``reopen_also`` names components that are not written, but must be
    reinstalled as well, because the attached value differs from the stored one.
    ``add_table_updates`` uses it after ``x_to``: the processed ``X`` is written
    to a layer, while the stored ``X`` remains the counts::

        Component         Store after the write   Attached, without reopen_also   Attached, with reopen_also=(("X",),)
        X                 counts                  processed values                counts, reopened
        layers["log1p"]   processed values        reopened                        reopened

    Without ``reopen_also``, the attached ``X`` would not match the store, and the
    next call would find it changed and write it to the layer again. A path that
    the store lacks, such as ``X`` of a table stored without one, is installed as
    absent. These paths also join the rollback snapshot, so a failure after
    installation restores their previous objects as well.
    """
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean.")
    # overwrite only concerns the store, where the writer checks it: a component
    # present only in memory is replaced without it, as for unbacked SpatialData.
    paths = _validate_component_paths(tuple(components), to_write=True) if components else ()
    deletions = _validate_deletion_paths(delete)
    _check_component_path_overlap((*paths, *deletions, *reopen_also))
    table, group = _component_update_destination(sdata, table_name=table_name)
    if reopen_also and group is None:
        raise ValueError("Reopening components requires backed SpatialData.")

    # Check live parents even for missing deletion targets. A scalar where a
    # mapping is expected is malformed, not a missing component.
    for path in (*paths, *deletions):
        value = _memory_component(table, path)
        if group is None and path in deletions and value is _ABSENT:
            log.info(f"Table {table_name!r}: component {path!r} is already absent; skipping deletion.")

    new_raw_var = None
    if table.raw is None and any(path[0] == "raw" for path in paths):
        new_raw_var = _new_raw_var(components, raw_var_names=raw_var_names)
    indices = _memory_axis_indices(table, components, obs_identity, var_names, raw_var_names, new_raw_var=new_raw_var)
    if group is not None:
        # Compare the attached table with the store first: the in-memory
        # validation below reads the annotation of the attached table, and would
        # reject the region/instance identities of an annotated store less clearly
        # if the attached annotation were missing.
        stored_new_raw_var = _prepare_raw_creation(group, components, raw_var_names=raw_var_names)
        _check_in_memory_versus_storage_axes(table, group, indices, stored_new_raw_var=stored_new_raw_var)
    _validate_component_values_against_memory(
        table,
        components,
        expected_axis_indices=indices,
        obs_identity=obs_identity,
        var_names=var_names,
        raw_var_names=raw_var_names,
    )

    # Retain original slot objects. Prepared mappings/raw containers are shallow
    # copies, so neither successful installation nor rollback mutates old entries.
    slots = {path[0] for path in (*paths, *deletions, *reopen_also)}
    previous = {slot: getattr(table, f"_{slot}") for slot in slots}
    try:
        if group is None:
            updates = _prepare_memory_updates(table, components, deletions, new_raw_var=new_raw_var)
            _install_memory_updates(table, updates)
        else:
            with _write_table_operation(
                sdata.path,
                table_name=table_name,
                components=components,
                delete=deletions,
                obs_identity=obs_identity,
                var_names=var_names,
                raw_var_names=raw_var_names,
                overwrite=overwrite,
            ) as published:
                # Serialization is complete. Reopen requested components from permanent
                # paths and attach them inside _write_table_operation()'s rollback window.
                # Its nested _publish_staged_paths() context retains disk backups until
                # this with-body and metadata consolidation succeed. On failure, that
                # context restores disk components; the except block below restores
                # the previous in-memory slot objects.
                reopened = {path: _read_anndata_element(published, path, mode="lazy") for path in paths}
                for path in reopen_also:
                    try:
                        reopened[path] = _read_anndata_element(published, path, mode="lazy")
                    except _MissingAnnDataElement:
                        reopened[path] = _ABSENT
                updates = _prepare_memory_updates(table, reopened, deletions, new_raw_var=new_raw_var)
                _install_memory_updates(table, updates)
    except BaseException:
        # The shared writer restores storage. Restore exact live slot references
        # without invoking setters that might repeat the failed installation.
        for slot, value in previous.items():
            setattr(table, f"_{slot}", value)
        raise
    return sdata


def _component_update_destination(sdata: SpatialData, *, table_name: str) -> tuple[AnnData, zarr.Group | None]:
    """Require an attached, non-view table and, when backed, its existing stored counterpart.

    Return the attached table and a read-only Zarr group, or None for the group
    when sdata has no path. Neither loads nor creates a complete table.
    """
    if not isinstance(sdata, SpatialData):
        raise TypeError("sdata must be a SpatialData object.")
    _validate_path_segment(table_name)
    if table_name not in sdata.tables:
        raise ValueError(f"Table {table_name!r} is not attached to sdata; load and attach it explicitly.")
    table = sdata.tables[table_name]
    if table.is_view:
        raise ValueError("Component updates require a non-view destination table; prepare and attach it explicitly.")
    # HDF5-backed setters can write through to another file. They cannot satisfy
    # this adapter's memory-only / shared-Zarr-publication ownership contract.
    if table.isbacked:
        raise ValueError("HDF5-backed AnnData destinations are not supported; attach a detached table explicitly.")
    group = None
    if sdata.path is not None:
        try:
            group = _open_table_group(sdata.path, table_name=table_name)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"The store and table {table_name!r} must already exist; persist the table explicitly first."
            ) from error
    return table, group


def _memory_component(table: AnnData, path: ComponentPath) -> object:
    """Look up a component in the attached AnnData without computing matrix values.

    Parameters
    ----------
    table
        Attached AnnData to inspect, not its on-disk representation.
    path
        Previously validated logical path, such as ("obsm", "embedding") or
        ("uns", "analysis", "threshold").

    Returns
    -------
    object
        The component's value, or _ABSENT if a required key is missing.
        An existing mapping entry whose value is None, such as
        ``table.uns["threshold"] = None``, returns None rather than _ABSENT.
        In contrast, ``table.X is None`` and ``table.raw is None`` represent
        absent in-memory components. An absent raw also makes paths inside
        raw absent.

    Raises
    ------
    ValueError
        An existing parent is not a mapping. For example, if
        ``table.uns["analysis"] = 42``, looking up ("uns", "analysis", "threshold")
        raises instead of treating the target as merely missing.
    """
    value = getattr(table, path[0])
    if path[0] in {"X", "raw"} and value is None:
        return _ABSENT
    keys = path[1:]
    if path[0] == "raw" and keys:
        value = getattr(value, keys[0])
        keys = keys[1:]
    for key in keys:
        if not isinstance(value, Mapping):
            raise ValueError(f"In-memory component parent of {path!r} is not a mapping.")
        if key not in value:
            return _ABSENT
        value = value[key]
    return value


def _memory_axis_indices(
    table: AnnData,
    components: Mapping[ComponentPath, object],
    obs_identity: pd.DataFrame | AxisNames | None,
    var_names: AxisNames | None,
    raw_var_names: AxisNames | None,
    *,
    new_raw_var: pd.DataFrame | None,
) -> dict[str, pd.Index]:
    """Collect destination axis indices needed to validate a component update.

    Include axes used by the replacement components, plus any axis with an
    explicit identity argument. Explicit identities are validated even when no
    replacement component uses their axis; they do not supply the returned indices.

    Parameters
    ----------
    table
        Existing in-memory destination table. Supplies the expected axis indices,
        not the replacement obs/var/raw.var dataframes in ``components``.
    components
        Replacement components whose alignment determines the required axes.
        For example, obsm requires obs; X requires obs and var; uns requires none.
    obs_identity, var_names, raw_var_names
        Caller-supplied identity arguments. A non-None value additionally requires
        validation of obs, var or raw_var, respectively.
    new_raw_var
        Prepared raw.var dataframe when creating a raw container. Its index
        defines the new raw_var axis; otherwise use the destination's raw.var index.

    Returns
    -------
    dict[str, pandas.Index]
        Expected indices for only the axes requiring validation. Possible keys
        are ``"obs"``, ``"var"`` and ``"raw_var"``, with values from table.obs.index,
        table.var.index and table.raw.var.index (or new_raw_var.index), respectively.
        The obs value contains observation names, not region/instance pairs.

        Without explicit identity arguments::

            obsm update -> {"obs": table.obs.index}
            X update    -> {"obs": table.obs.index, "var": table.var.index}
            uns update  -> {}

        Thus, ``"obs" in indices`` means observation validation is needed, not
        that the caller supplied an obs dataframe or that the table has an obs slot.
    """
    required = {axis for path in components for axis in _component_axes(path)}
    for axis, identity in (("obs", obs_identity), ("var", var_names), ("raw_var", raw_var_names)):
        if identity is not None:
            required.add(axis)
    indices = {}
    for axis in required:
        if axis == "raw_var":
            if new_raw_var is not None:
                indices[axis] = new_raw_var.index
            elif table.raw is not None:
                indices[axis] = table.raw.var.index
            else:
                raise ValueError("The in-memory table has no raw feature axis.")
        else:
            indices[axis] = getattr(table, axis).index
    return indices


def _validate_component_values_against_memory(
    table: AnnData,
    components_to_validate: Mapping[ComponentPath, object],
    *,
    expected_axis_indices: Mapping[str, pd.Index],
    obs_identity: pd.DataFrame | AxisNames | None,
    var_names: AxisNames | None,
    raw_var_names: AxisNames | None,
) -> None:
    """Validate replacements against attached AnnData metadata, without storage reads.

    Prepare the required region/instance dataframe, then delegate validation to
    ``_validate_component_values_against_axes()``. "Memory" refers to the attached
    table's metadata; its matrices may remain lazy or storage-backed.

    Parameters
    ----------
    table
        Attached destination AnnData supplying the expected SpatialData annotation
        and region/instance columns, not a replacement table.
    components_to_validate
        Component replacements, using the same mapping format as ``components``
        in :func:`harpy.table.io.add_table_components`.
    expected_axis_indices
        Indices prepared by ``_memory_axis_indices()``. The caller retains them
        for the separate memory-versus-storage comparison in ``_check_in_memory_versus_storage_axes()``.
    obs_identity, var_names, raw_var_names
        Corresponding identity arguments from the public component-update APIs.
    """
    expected_spatialdata_attrs = table.uns.get(TableModel.ATTRS_KEY)
    expected_region_instance_identity = None
    if "obs" in expected_axis_indices and expected_spatialdata_attrs is not None:
        if not isinstance(expected_spatialdata_attrs, Mapping):
            raise ValueError("SpatialData annotation must be a mapping.")
        region_key, instance_key = _annotation_columns(expected_spatialdata_attrs)
        identity_columns = [region_key, instance_key]
        # Check the full frame before selecting: selection must not hide duplicate
        # column names or replace the missing-identity error with a pandas KeyError.
        if any(key not in table.obs for key in identity_columns):
            raise ValueError(
                f"Destination observation must contain the stored region and instance columns {identity_columns!r}."
            )
        if not table.obs.columns.is_unique:
            raise ValueError("Destination observation must not contain duplicate column names.")
        expected_region_instance_identity = table.obs[identity_columns]

    _validate_component_values_against_axes(
        components_to_validate,
        # Expected destination metadata:
        expected_axis_indices=expected_axis_indices,
        expected_region_instance_identity=expected_region_instance_identity,
        expected_spatialdata_attrs=expected_spatialdata_attrs,
        # Identity context supplied with the update:
        supplied_obs_identity=obs_identity,
        supplied_var_names=var_names,
        supplied_raw_var_names=raw_var_names,
    )


def _check_in_memory_versus_storage_axes(
    table: AnnData,
    group: zarr.Group,
    indices: Mapping[str, pd.Index],
    *,
    stored_new_raw_var: pd.DataFrame | None,
) -> None:
    """Check whole in-memory axes against storage before positional installation.

    "In-memory" refers to the supplied AnnData ``table``; "storage" refers to
    the destination table's AnnData Zarr ``group``, not the SpatialData store root.
    The comparison uses axis metadata, never matrix values; ``table`` may still
    contain lazy or storage-backed matrices.
    """
    for axis, index in indices.items():
        if axis == "obs":
            stored_attrs = _read_spatialdata_attrs(group)
            in_memory_attrs = table.uns.get(TableModel.ATTRS_KEY)
            if in_memory_attrs is None and stored_attrs is not None:
                raise ValueError(
                    "The attached table has no SpatialData annotation (uns['spatialdata_attrs']), while the stored "
                    "table has one; they must agree. Reopen the table, or restore its annotation."
                )
            if in_memory_attrs is not None and stored_attrs is None:
                raise ValueError(
                    "The attached table has a SpatialData annotation (uns['spatialdata_attrs']), while the stored "
                    "table has none; they must agree. Reopen the table, or write it whole to store the annotation."
                )
            if in_memory_attrs is not None and not isinstance(in_memory_attrs, Mapping):
                raise ValueError("SpatialData annotation must be a mapping.")
            if stored_attrs is not None:
                if _annotation_columns(in_memory_attrs) != _annotation_columns(stored_attrs):
                    raise ValueError("In-memory and stored region/instance keys must agree; reopen the table.")
                in_memory_pairs = _validated_observation_pairs(
                    table.obs, in_memory_attrs, label="In-memory observation"
                )
                stored_pairs = _validated_observation_pairs(
                    _read_observation_identity(group, stored_attrs), stored_attrs, label="Stored observation"
                )
                _match_identity(in_memory_pairs, stored_pairs, label="In-memory observation")
                continue
        if axis == "raw_var" and stored_new_raw_var is not None:
            stored_index = stored_new_raw_var.index
        else:
            frame_path = ("raw", "var") if axis == "raw_var" else (axis,)
            stored_index = _storage_axis_index(group, frame_path)
        _match_identity(index, stored_index, label=f"In-memory {axis}")


def _updated_mapping(mapping: Mapping, keys: ComponentPath, value: object) -> dict:
    """Copy only mappings along a changed path, keeping unrelated values by reference."""
    updated = dict(mapping)
    key = keys[0]
    if len(keys) == 1:
        if value is _ABSENT:
            updated.pop(key, None)
        else:
            updated[key] = value
    else:
        child = mapping.get(key, _ABSENT)
        if child is _ABSENT:
            if value is _ABSENT:
                return updated
            child = {}
        if not isinstance(child, Mapping):
            raise ValueError(f"In-memory component parent at {keys!r} is not a mapping.")
        updated[key] = _updated_mapping(child, keys[1:], value)
    return updated


def _prepare_memory_updates(
    table: AnnData,
    components: Mapping[ComponentPath, object],
    deletions: Sequence[ComponentPath],
    *,
    new_raw_var: pd.DataFrame | None,
) -> dict[str, object]:
    """Prepare replacement slot objects without mutating live or caller-owned data."""
    changes = dict(components)
    for path, value in changes.items():
        if isinstance(value, pd.DataFrame) and path[0] not in {"obs", "var", "uns"}:
            # AnnData's aligned-mapping setters normalize dataframe index names.
            # Isolate that metadata while retaining the supplied data columns.
            changes[path] = value.copy(deep=False)
            changes[path].index = value.index.copy()
    changes.update({path: _ABSENT for path in deletions if _memory_component(table, path) is not _ABSENT})
    updates = {}
    raw_changes = {path[1:]: value for path, value in changes.items() if path[0] == "raw"}
    if raw_changes:
        if () in raw_changes:
            updates["raw"] = None
        else:
            if table.raw is None:
                # Creation requires raw.X. Explicit X prevents Raw's fallback
                # from copying the parent table's unrelated expression matrix.
                assert new_raw_var is not None
                raw = Raw(table, X=raw_changes[("X",)], var=new_raw_var.copy(deep=True))
            else:
                raw = copy(table.raw)
            for path, value in raw_changes.items():
                if path == ("X",):
                    raw._X = value
                elif path[0] == "varm":
                    raw.varm = _updated_mapping(raw.varm, path[1:], value)
            if ("var",) in raw_changes:
                raw._var = raw_changes[("var",)].copy(deep=True)
            updates["raw"] = raw
    for path, value in changes.items():
        slot = path[0]
        if slot == "raw":
            continue
        if len(path) == 1:
            if slot in {"obs", "var"}:
                value = value.copy(deep=True)
            elif slot == "uns":
                value = dict(value)
            updates[slot] = None if value is _ABSENT else value
        else:
            current = updates.get(slot, getattr(table, slot))
            updates[slot] = _updated_mapping(current, path[1:], value)
    if ("uns",) in components or any(path[:2] == ("uns", TableModel.ATTRS_KEY) for path in components):
        attrs = updates["uns"].get(TableModel.ATTRS_KEY)
        if isinstance(attrs, Mapping) and isinstance(attrs.get(TableModel.REGION_KEY), np.ndarray):
            # Serialized lists reopen as arrays; SpatialData expects region lists.
            updates["uns"][TableModel.ATTRS_KEY] = dict(attrs)
            updates["uns"][TableModel.ATTRS_KEY][TableModel.REGION_KEY] = attrs[TableModel.REGION_KEY].tolist()
    return updates


def _install_memory_updates(table: AnnData, updates: Mapping[str, object]) -> None:
    """Install prepared slots; the caller owns rollback of their original references."""
    # Install aligned mappings before axis frames: their setters normalize all
    # dataframe index names, including those of unrequested entries. Keep the
    # original axes in place during that validation to avoid changing those entries.
    for slot, value in updates.items():
        if slot not in {"obs", "var", "raw"}:
            setattr(table, slot, value)
    for slot, value in updates.items():
        if slot in {"obs", "var", "raw"}:
            # Axes have already been validated. AnnData's dataframe setters also
            # reset indices of unrelated obsm/varm DataFrames; raw's setter rebuilds
            # Raw (and can copy X). Replace only the prepared slot instead.
            setattr(table, f"_{slot}", value)
