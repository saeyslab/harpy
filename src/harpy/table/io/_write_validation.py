"""Structural and ordered-identity checks for scoped table writes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from anndata import AnnData
from anndata.io import read_elem
from spatialdata.models import TableModel

from harpy._storage._anndata import _MATRIX_MAPPINGS, _MissingAnnDataElement, _read_anndata_element
from harpy.table.io._read import ComponentPath, _check_component_path_overlap, _validate_path_segment

type AxisNames = pd.Index | Sequence[str]


def _validate_deletion_paths(components: Sequence[ComponentPath]) -> tuple[ComponentPath, ...]:
    """Allow only optional entries, protecting axes, containers and SpatialData linkage.

    An empty sequence is allowed for replacement-only operations. Public
    deletion-only requests additionally require at least one target.
    """
    if isinstance(components, (str, bytes)) or not isinstance(components, Sequence):
        raise TypeError("Deletion components must be a sequence of tuple paths.")
    paths = []
    for path in components:
        if not isinstance(path, tuple):
            raise TypeError("Each component path must be a tuple of strings.")
        if not path:
            raise ValueError("Component paths must not be empty.")
        for segment in path:
            _validate_path_segment(segment)
        valid = (
            path in {("X",), ("raw",)}
            or (path[0] in _MATRIX_MAPPINGS and len(path) == 2)
            or (path[:2] == ("raw", "varm") and len(path) == 3)
            or (path[0] == "uns" and len(path) >= 2 and path[1] != TableModel.ATTRS_KEY)
        )
        if not valid:
            raise ValueError(f"Cannot delete required, protected or unsupported AnnData component {path!r}.")
        paths.append(path)
    _check_component_path_overlap(paths)
    return tuple(paths)


def _check_component_deletion_destination(
    group: zarr.Group, path: ComponentPath, *, table_path: Path, root: Path
) -> bool:
    """Return whether a safe logical target exists, without decoding its payload.

    Only a genuinely missing path is a no-op. Malformed mapping parents,
    unrecognized filesystem entries and unsafe destinations still raise.
    """
    parent = group
    for key_index, key in enumerate(path):
        logical_path = path[: key_index + 1]
        destination = table_path.joinpath(*logical_path)
        if destination.is_symlink() or not destination.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Unsafe table-deletion destination: {destination}.")
        try:
            element = parent[key]
        except KeyError:
            # Do not mistake an unrecognized file/directory for an absent
            # component. stat() also propagates permission and other I/O errors.
            try:
                destination.stat()
            except FileNotFoundError:
                return False
            raise ValueError(f"Unrecognized AnnData component path: {logical_path!r}.") from None
        if key_index == len(path) - 1:
            return True
        expected_encoding = "raw" if logical_path == ("raw",) else "dict"
        if (
            not isinstance(element, zarr.Group)
            or element.attrs.get("encoding-type") != expected_encoding
            or element.attrs.get("encoding-version") != "0.1.0"
        ):
            raise ValueError(f"AnnData component parent {logical_path!r} is not an encoded mapping.")
        parent = element
    return False  # Paths are nonempty after validation.


def _prepare_raw_creation(
    group: zarr.Group,
    components: Mapping[ComponentPath, object],
    *,
    raw_var_names: AxisNames | None,
    overwrite: bool,
) -> pd.DataFrame | None:
    """Prepare feature annotations when the request needs a new raw container.

    Existing raw containers retain their stored axis. Creation from an absent
    or encoded-null entry requires raw.X and explicit feature identities;
    replacing a stored null still requires overwrite permission.

    Returns
    -------
    pandas.DataFrame or None
        The feature dataframe for a new raw container, copied from the supplied
        raw.var or constructed from raw_var_names. None means no raw components
        were requested, or a valid raw container already exists and only its
        requested components need updating.
    """
    if not any(path[0] == "raw" for path in components):
        return None
    if "raw" in group:
        raw = group["raw"]
        encoding = (raw.attrs.get("encoding-type"), raw.attrs.get("encoding-version"))
        if isinstance(raw, zarr.Group) and encoding == ("raw", "0.1.0"):
            return None
        if not (isinstance(raw, zarr.Array) and encoding == ("null", "0.1.0") and raw.shape == ()):
            raise ValueError("Cannot create raw over a malformed or unsupported raw encoding.")
        if not overwrite:
            raise FileExistsError("The raw entry already exists as None; use overwrite=True.")
    return _new_raw_var(components, raw_var_names=raw_var_names)


def _new_raw_var(components: Mapping[ComponentPath, object], *, raw_var_names: AxisNames | None) -> pd.DataFrame:
    """Prepare a new raw feature axis identically for memory and storage updates."""
    if components.get(("raw", "X")) is None:
        raise ValueError("Creating raw requires a non-None ('raw', 'X') matrix.")
    if ("raw", "var") in components:
        frame = components[("raw", "var")]
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("Component ('raw', 'var') must be a pandas DataFrame.")
        names = _named_identity(frame.index, label="New raw feature")
        if raw_var_names is not None:
            _match_identity(_named_identity(raw_var_names, label="raw_var_names"), names, label="raw_var_names")
        return frame.copy(deep=True)
    if raw_var_names is None:
        raise ValueError("Creating raw requires raw_var_names or a ('raw', 'var') dataframe.")
    return pd.DataFrame(index=_named_identity(raw_var_names, label="raw_var_names"))


def _check_write_destination(destination: Path, *, root: Path, overwrite: bool) -> None:
    """Check filesystem safety and overwrite permission for a table or component.

    Filesystem existence matters even when Zarr does not recognize the path
    as an element. This check does not inspect AnnData encodings or payloads.
    """
    if destination.is_symlink() or not destination.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Unsafe table-write destination: {destination}.")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Destination {destination} already exists; use overwrite=True.")


def _check_component_write_destination(
    group: zarr.Group, path: ComponentPath, *, table_path: Path, root: Path, overwrite: bool
) -> None:
    """Check filesystem safety, overwrite permission and logical parent mappings.

    The destination is resolved relative to the permanent table_path, not
    staging. No payload is decoded and no directories are created.
    """
    _check_write_destination(table_path.joinpath(*path), root=root, overwrite=overwrite)
    parent = group
    for key_index, key in enumerate(path):
        if key not in parent:
            if path[: key_index + 1] == ("raw",):
                raise ValueError("Raw creation must be prepared before checking individual raw destinations.")
            return
        element = parent[key]
        if key_index == len(path) - 1:
            return
        expected_encoding = "raw" if path[: key_index + 1] == ("raw",) else "dict"
        if (
            not isinstance(element, zarr.Group)
            or element.attrs.get("encoding-type") != expected_encoding
            or element.attrs.get("encoding-version") != "0.1.0"
        ):
            raise ValueError(f"AnnData component parent {path[: key_index + 1]!r} is not an encoded mapping.")
        parent = element


def _storage_axis_index(group: zarr.Group, path: ComponentPath) -> pd.Index:
    """Read only a dataframe's index, not its other annotation columns."""
    frame = group["/".join(path)]
    if frame.attrs.get("encoding-type") != "dataframe" or frame.attrs.get("encoding-version") != "0.2.0":
        raise ValueError(f"Unsupported dataframe encoding at {path!r}.")
    return pd.Index(read_elem(frame[frame.attrs["_index"]]))


def _read_spatialdata_attrs(group: zarr.Group) -> dict | None:
    """Read the table's uns["spatialdata_attrs"] annotation, or return None if absent."""
    try:
        value = _read_anndata_element(group, ("uns", TableModel.ATTRS_KEY), mode="eager")
    except _MissingAnnDataElement:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("Stored SpatialData table annotation must be a mapping.")
    value = dict(value)
    if isinstance(value.get(TableModel.REGION_KEY), np.ndarray):
        value[TableModel.REGION_KEY] = value[TableModel.REGION_KEY].tolist()
    return value


def _same_metadata(left: object, right: object) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(_same_metadata(left[key], right[key]) for key in left)
    return bool(np.array_equal(left, right))


def _validate_spatialdata_attrs_unchanged(
    spatialdata_attrs: Mapping | None, components: Mapping[ComponentPath, object]
) -> None:
    """Ensure component writes preserve uns["spatialdata_attrs"], including its absence."""
    annotation_path = ("uns", TableModel.ATTRS_KEY)
    for path, value in components.items():
        if path == ("uns",):
            if not isinstance(value, Mapping):
                raise ValueError("uns must be a mapping.")
            stored = spatialdata_attrs
            present = stored is not None
            if (TableModel.ATTRS_KEY in value) != present or (
                present and not _same_metadata(stored, value[TableModel.ATTRS_KEY])
            ):
                raise ValueError("Component writes must preserve SpatialData annotation; use write_table().")
        elif path[:2] == annotation_path:
            stored = spatialdata_attrs
            try:
                if stored is None:
                    raise KeyError(path)
                for key in path[2:]:
                    if not isinstance(stored, Mapping):
                        raise ValueError(f"SpatialData annotation parent of {path!r} is not a mapping.")
                    stored = stored[key]
            except KeyError:
                raise ValueError("Component writes must not add SpatialData annotation; use write_table().") from None
            if not _same_metadata(stored, value):
                raise ValueError("Component writes must preserve SpatialData annotation; use write_table().")


def _named_identity(value: AxisNames, *, label: str) -> pd.Index:
    if isinstance(value, (str, bytes)) or not isinstance(value, (pd.Index, Sequence)):
        raise TypeError(f"{label} must be a pandas index or a sequence of strings.")
    if any(not isinstance(name, str) for name in value):
        raise ValueError(f"{label} must contain non-null string names.")
    index = pd.Index(value)
    if not index.is_unique:
        raise ValueError(f"{label} must not contain duplicate identities.")
    return index


def _match_identity(actual: pd.Index, expected: pd.Index, *, label: str) -> None:
    # Identity is defined by ordered values, not index names or equivalent
    # representations such as object-backed versus pandas string indices.
    if np.array_equal(actual.to_numpy(dtype=object), expected.to_numpy(dtype=object)):
        return
    if len(actual) == len(expected) and actual.isin(expected).all():
        raise ValueError(
            f"{label} identities match but differ in order. Reorder the supplied components and their identity "
            "information together before writing."
        )
    raise ValueError(f"{label} must match the expected identities exactly in value and order.")


def _annotation_columns(spatialdata_attrs: Mapping) -> tuple[str, str]:
    """Resolve the stored identity column names without constructing an AnnData."""
    region_key = spatialdata_attrs.get(TableModel.REGION_KEY_KEY)
    instance_key = spatialdata_attrs.get(TableModel.INSTANCE_KEY)
    if (
        not isinstance(region_key, str)
        or not region_key
        or not isinstance(instance_key, str)
        or not instance_key
        or region_key == instance_key
    ):
        raise ValueError("SpatialData annotation requires distinct, nonempty region_key and instance_key names.")
    return region_key, instance_key


def _observation_pairs(frame: pd.DataFrame, *, region_key: str, instance_key: str, label: str) -> pd.MultiIndex:
    """Check identity columns and return ordered pairs, for a full table or a subset.

    This checks column types and non-null, unique pairs, not declared regions.
    Callers separately validate the complete stored annotation and compare these
    pairs with the expected full or selected observation axis.
    """
    keys = [region_key, instance_key]
    if any(key not in frame for key in keys):
        raise ValueError(f"{label} must contain the stored region and instance columns {keys!r}.")
    if not frame.columns.is_unique:
        raise ValueError(f"{label} must not contain duplicate column names.")
    identity = frame[keys]
    if identity.isna().any().any() or identity.duplicated().any():
        raise ValueError(f"{label} region/instance pairs must be non-null and unique.")
    if not isinstance(identity[region_key].dtype, pd.CategoricalDtype):
        raise ValueError(f"{label} region column must be categorical.")
    # Preserve the integer/string instance types accepted by SpatialData,
    # without applying its whole-table region-set check to a regional subset.
    instances = identity[instance_key]
    dtype = instances.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        dtype = dtype.categories.dtype
    integer_types = (np.int16, np.int32, np.int64, np.uint16, np.uint32, np.uint64)
    if not (dtype in integer_types or isinstance(dtype, pd.StringDtype) or pd.api.types.is_string_dtype(instances)):
        raise TypeError(f"{label} instance column must contain supported integer or string identifiers.")
    return pd.MultiIndex.from_frame(identity.astype(object))


def _match_observation_identity(
    obs_identity: pd.DataFrame | AxisNames,
    expected: pd.MultiIndex,
    *,
    region_key: str,
    instance_key: str,
) -> None:
    """Match explicit region/instance pairs, ignoring the identity dataframe's index."""
    keys = [region_key, instance_key]
    if not isinstance(obs_identity, pd.DataFrame):
        raise TypeError("obs_identity must be a two-column region/instance dataframe for annotated tables.")
    if set(obs_identity.columns) != set(keys) or len(obs_identity.columns) != 2:
        raise ValueError(f"obs_identity must contain exactly the columns {keys!r}.")
    _match_identity(
        _observation_pairs(obs_identity, region_key=region_key, instance_key=instance_key, label="obs_identity"),
        expected,
        label="obs_identity",
    )


def _validated_observation_pairs(frame: pd.DataFrame, spatialdata_attrs: Mapping, *, label: str) -> pd.MultiIndex:
    """Validate the full table annotation and return ordered region/instance pairs.

    Declared regions must match observed values, not unused categorical levels.
    With annotation keys ``region`` and ``instance``, for example::

        region   instance
        A        1
        B        1
        A        2

        -> MultiIndex([("A", 1), ("B", 1), ("A", 2)])

    The pairs preserve dataframe row order; the dataframe's own index is
    neither used as identity nor modified.
    """
    region_key, instance_key = _annotation_columns(spatialdata_attrs)
    regions = spatialdata_attrs.get(TableModel.REGION_KEY)
    regions = [regions] if isinstance(regions, str) else regions
    if not isinstance(regions, (list, tuple, np.ndarray)) or any(
        not isinstance(region, str) or not region for region in regions
    ):
        raise ValueError("SpatialData annotation requires region names as a string or sequence of strings.")
    pairs = _observation_pairs(frame, region_key=region_key, instance_key=instance_key, label=label)
    # Use observed values, not categorical categories: unused categories do
    # not declare regions or make an otherwise absent region selectable.
    if set(regions) != set(frame[region_key].unique()):
        raise ValueError(f"{label} declared regions must match the regions present in its observations.")
    return pairs


def _read_observation_identity(group: zarr.Group, spatialdata_attrs: Mapping) -> pd.DataFrame:
    """Read only the observation index and its two spatial identity columns."""
    keys = _annotation_columns(spatialdata_attrs)
    if any(key not in group["obs"] for key in keys):
        raise ValueError("Stored observation is missing a region or instance column referenced by its annotation.")
    return pd.DataFrame({key: read_elem(group["obs"][key]) for key in keys}, index=_storage_axis_index(group, ("obs",)))


def _storage_axis_indices(
    group: zarr.Group,
    components: Mapping[ComponentPath, object],
    *,
    obs_identity: pd.DataFrame | AxisNames | None,
    var_names: AxisNames | None,
    raw_var_names: AxisNames | None,
    expected_new_raw_var_index: pd.Index | None = None,
) -> dict[str, pd.Index]:
    """Collect stored destination axis indices needed to validate a component update.

    Include axes used by the replacement components, plus any axis with an
    explicit identity argument. Explicit identities are validated even when no
    replacement component uses their axis; they do not supply the returned indices.

    Parameters
    ----------
    group
        Existing destination table's AnnData Zarr group, not the SpatialData
        root or staging group. Only required indices are read, never numerical matrices.
    components
        Replacement components whose alignment determines the required axes.
        For example, obsm requires obs; X requires obs and var; uns requires none.
    obs_identity, var_names, raw_var_names
        Caller-supplied identity arguments. A non-None value additionally requires
        validation of obs, var or raw_var, respectively.
    expected_new_raw_var_index
        Prepared feature index when creating a raw container. Raw is optional and
        has no stored feature index yet in this case, whereas the table's obs and
        var indices already exist. None uses the stored raw axis when needed.

    Returns
    -------
    dict[str, pandas.Index]
        Expected indices for only the axes requiring validation: ``"obs"``,
        ``"var"`` and/or ``"raw_var"``. Values come from the corresponding stored
        dataframes, except for a newly created raw axis supplied above. The obs
        value contains observation names, not region/instance pairs. Prepare once
        before staging and reuse when validating the reopened staged components.
    """
    required = {axis for path in components for axis in _component_axes(path)}
    explicit = {"obs": obs_identity, "var": var_names, "raw_var": raw_var_names}
    frame_paths = {"obs": ("obs",), "var": ("var",), "raw_var": ("raw", "var")}
    # Expected indices for alignment checks come from the stored table,
    # except when creating raw: use the index prepared before staging, not
    # the raw.var dataframe currently being validated.
    expected_axis_indices = {}
    for axis, frame_path in frame_paths.items():
        if axis not in required and explicit[axis] is None:
            continue
        if axis == "raw_var" and expected_new_raw_var_index is not None:
            expected_axis_indices[axis] = expected_new_raw_var_index
        else:
            try:
                expected_axis_indices[axis] = _storage_axis_index(group, frame_path)
            except KeyError:
                raise ValueError(f"The stored table has no {axis!r} axis.") from None
    return expected_axis_indices


def _validate_component_values_against_storage(
    group: zarr.Group,
    components_to_validate: Mapping[ComponentPath, object],
    *,
    expected_axis_indices: Mapping[str, pd.Index],
    obs_identity: pd.DataFrame | AxisNames | None,
    var_names: AxisNames | None,
    raw_var_names: AxisNames | None,
) -> None:
    """Read required spatial annotation and validate replacements against stored metadata.

    Used before staging and again on reopened serialized values. Combine the
    prepared axis indices with annotation from storage, then delegate validation
    to ``_validate_component_values_against_axes()``.

    Parameters
    ----------
    group
        Existing destination table's AnnData Zarr group, not the SpatialData
        root or staging group. Supplies any required observation identities and
        linkage metadata; numerical matrices are not read.
    components_to_validate
        Component replacements to validate: either caller-supplied values or their
        reopened staged representations. Uses the same mapping format as
        ``components`` in :func:`harpy.table.write_table_components`.
    expected_axis_indices
        Indices prepared by ``_storage_axis_indices()`` before staging, including
        the intended feature index when creating raw. Reuse the same indices when
        validating reopened staged components.
    obs_identity, var_names, raw_var_names
        Corresponding identity arguments from the public component-update APIs.
    """
    expected_spatialdata_attrs = (
        _read_spatialdata_attrs(group)
        if "obs" in expected_axis_indices or any(path[0] == "uns" for path in components_to_validate)
        else None
    )
    expected_region_instance_identity = (
        _read_observation_identity(group, expected_spatialdata_attrs)
        if "obs" in expected_axis_indices and expected_spatialdata_attrs is not None
        else None
    )
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


def _component_axes(path: ComponentPath) -> tuple[str, ...]:
    """Return the complete axes needed to align one replacement component."""
    slot = path[0]
    if slot == "uns":
        return ()
    if slot in {"obs", "obsm", "obsp"}:
        return ("obs",)
    if slot in {"var", "varm", "varp"}:
        return ("var",)
    if slot == "raw":
        return ("obs", "raw_var") if path[1] == "X" else ("raw_var",)
    return ("obs", "var")


def _validate_component_values_against_axes(
    components_to_validate: Mapping[ComponentPath, object],
    *,
    expected_axis_indices: Mapping[str, pd.Index],
    expected_region_instance_identity: pd.DataFrame | None,
    expected_spatialdata_attrs: Mapping | None,
    supplied_obs_identity: pd.DataFrame | AxisNames | None,
    supplied_var_names: AxisNames | None,
    supplied_raw_var_names: AxisNames | None,
) -> None:
    """Validate replacement values against metadata, without storage or matrix reads.

    The ``expected_*`` arguments describe the destination metadata; ``supplied_*``
    arguments describe the identity context submitted with the update.
    Identities must match in value and order; validation never reorders components.

    Parameters
    ----------
    components_to_validate
        Component replacements to validate: either caller-supplied values or their
        reopened staged representations. Uses the same mapping format as
        ``components`` in :func:`harpy.table.add_table_components` and
        :func:`harpy.table.write_table_components`.
    expected_axis_indices
        Expected destination indices for axes requiring validation, either because
        replacement components use them or explicit identity arguments were supplied.

        Keys identify axes:

        - ``"obs"``: the destination's obs.index.
        - ``"var"``: the destination's var.index.
        - ``"raw_var"``: the destination's raw.var.index, or the prepared feature
          index when creating raw.

        The caller reads existing axis indices from the stored table or obtains
        them from the attached in-memory AnnData, not from replacement dataframes.
        Used to validate dataframe indices, matrix dimensions and named identities.
        The ``"obs"`` value always contains observation index labels; annotated
        region/instance identities are supplied separately through
        ``expected_region_instance_identity``.
    expected_region_instance_identity
        Dataframe containing only the destination's region and instance columns,
        in observation order. Required for annotated observation validation;
        otherwise None. Unannotated observations use ``expected_axis_indices["obs"]``.
    expected_spatialdata_attrs
        Destination SpatialData annotation, including declared regions and the
        region/instance column names. Used for annotated observation checks and
        protecting linkage metadata. None for unannotated tables; may also be None
        when the update requires neither observation nor linkage-metadata checks.
    supplied_obs_identity
        Corresponds to ``obs_identity`` in the public component-update APIs.
    supplied_var_names
        Corresponds to ``var_names`` in the public component-update APIs.
    supplied_raw_var_names
        Corresponds to ``raw_var_names`` in the public component-update APIs.
    """
    if expected_spatialdata_attrs is not None and not isinstance(expected_spatialdata_attrs, Mapping):
        raise ValueError("SpatialData annotation must be a mapping.")
    axes_by_path = {path: _component_axes(path) for path in components_to_validate}
    frame_paths = {"obs": ("obs",), "var": ("var",), "raw_var": ("raw", "var")}
    # 1) Any supplied obs/var/raw.var dataframe must have an index matching
    # the expected axis in value and order, regardless of SpatialData annotation.
    # These may be the original inputs or reopened staged values.
    for axis in expected_axis_indices:
        frame_path = frame_paths[axis]
        # Get the obs, var or raw.var dataframe being checked in this round,
        # if any (not the dataframe already stored in the destination table).
        axis_frame_to_validate = components_to_validate.get(frame_path)
        if frame_path in components_to_validate:
            if not isinstance(axis_frame_to_validate, pd.DataFrame):
                raise TypeError(f"Component {frame_path!r} must be a pandas DataFrame.")
            # In-memory adapters install these frames without AnnData's axis
            # setters (which also modify unrelated aligned dataframes). Retain
            # their dataframe-metadata restrictions in the shared validator.
            if isinstance(axis_frame_to_validate.columns, pd.MultiIndex):
                raise ValueError(f"Component {frame_path!r} must not have MultiIndex columns.")
            if not isinstance(axis_frame_to_validate.index.name, (str, type(None))):
                raise ValueError(f"Component {frame_path!r} index name must be a string or None.")
            _match_identity(axis_frame_to_validate.index, expected_axis_indices[axis], label=f"{axis} dataframe index")

    # 2) Validate observation identities using the destination's annotation policy.
    if "obs" in expected_axis_indices:
        supplied_obs_frame = components_to_validate.get(("obs",))
        if expected_spatialdata_attrs is not None:
            assert expected_region_instance_identity is not None
            _validate_annotated_obs_update(
                expected_region_instance_identity=expected_region_instance_identity,
                expected_spatialdata_attrs=expected_spatialdata_attrs,
                supplied_obs_frame=supplied_obs_frame,
                supplied_obs_identity=supplied_obs_identity,
            )
        else:
            _validate_unannotated_obs_update(
                expected_obs_index=expected_axis_indices["obs"],
                supplied_obs_frame=supplied_obs_frame,
                supplied_obs_identity=supplied_obs_identity,
            )

    # 3) Feature axes always use ordered names, regardless of observation annotation.
    # Check both the supplied var/raw.var index and explicit names when both are given.
    for axis, supplied_names in (("var", supplied_var_names), ("raw_var", supplied_raw_var_names)):
        if axis not in expected_axis_indices:
            continue
        axis_frame_to_validate = components_to_validate.get(frame_paths[axis])
        expected = _named_identity(expected_axis_indices[axis], label=f"Stored {axis}")
        for value in (None if axis_frame_to_validate is None else axis_frame_to_validate.index, supplied_names):
            if value is not None:
                _match_identity(_named_identity(value, label=axis), expected, label=axis)
        if axis_frame_to_validate is None and supplied_names is None:
            parameter = "var_names" if axis == "var" else "raw_var_names"
            raise ValueError(
                f"Supply {parameter} or the corresponding dataframe when writing {axis}-aligned components."
            )

    _validate_spatialdata_attrs_unchanged(expected_spatialdata_attrs, components_to_validate)
    for path, axes in axes_by_path.items():
        value = components_to_validate[path]
        if not axes or path in frame_paths.values():
            continue
        if path in {("X",), ("raw", "X")} and value is None:
            continue
        shape = getattr(value, "shape", ())
        if not shape or any(not isinstance(size, Integral) or size < 0 for size in shape):
            raise ValueError(f"Component {path!r} requires a matrix with a known shape.")
        if path[0] in {"obsp", "varp"}:
            expected_shape = (len(expected_axis_indices[axes[0]]),) * 2
        elif len(axes) == 2:
            expected_shape = tuple(len(expected_axis_indices[axis]) for axis in axes)
        else:
            expected_shape = (len(expected_axis_indices[axes[0]]), *shape[1:])
        if tuple(shape) != expected_shape:
            raise ValueError(f"Component {path!r} has shape {shape}; expected {expected_shape}.")
        if isinstance(value, pd.DataFrame):
            # Dataframe-valued matrices (e.g. obsm/varm entries) carry row labels:
            # matching shape alone cannot detect reordered or different identities.
            # This is separate from the obs/var/raw.var checks above; those frames
            # are skipped in this loop.
            _match_identity(value.index, expected_axis_indices[axes[0]], label=f"Component {path!r} dataframe index")


def _validate_annotated_obs_update(
    *,
    expected_region_instance_identity: pd.DataFrame,
    expected_spatialdata_attrs: Mapping,
    supplied_obs_frame: pd.DataFrame | None,
    supplied_obs_identity: pd.DataFrame | AxisNames | None,
) -> None:
    """Match supplied obs and/or explicit identities by ordered region/instance pairs.

    The shared dataframe checks already validated the replacement obs index.
    The explicit ``supplied_obs_identity`` dataframe's index is ignored.
    """
    region_key, instance_key = _annotation_columns(expected_spatialdata_attrs)
    expected = _validated_observation_pairs(
        expected_region_instance_identity, expected_spatialdata_attrs, label="Destination observation"
    )
    # A replacement obs frame and explicit obs_identity are independent sources
    # of identity information: validate both when both are supplied.
    if supplied_obs_frame is not None:
        _match_identity(
            _observation_pairs(
                supplied_obs_frame, region_key=region_key, instance_key=instance_key, label="Supplied obs"
            ),
            expected,
            label="obs_identity",
        )
    if supplied_obs_identity is not None:
        _match_observation_identity(
            supplied_obs_identity,
            expected,
            region_key=region_key,
            instance_key=instance_key,
        )
    if supplied_obs_frame is None and supplied_obs_identity is None:
        raise ValueError("Supply obs_identity or the corresponding dataframe when writing obs-aligned components.")


def _validate_unannotated_obs_update(
    *,
    expected_obs_index: pd.Index,
    supplied_obs_frame: pd.DataFrame | None,
    supplied_obs_identity: pd.DataFrame | AxisNames | None,
) -> None:
    """Match supplied obs and/or explicit identities by ordered observation names."""
    # Without SpatialData annotation, observation names identify rows rather
    # than region/instance pairs. Validate both supplied sources when present.
    expected = _named_identity(expected_obs_index, label="Stored obs")
    for value in (None if supplied_obs_frame is None else supplied_obs_frame.index, supplied_obs_identity):
        if value is not None:
            _match_identity(_named_identity(value, label="obs"), expected, label="obs")
    if supplied_obs_frame is None and supplied_obs_identity is None:
        raise ValueError("Supply obs_identity or the corresponding dataframe when writing obs-aligned components.")


def _validate_complete_table(table: AnnData) -> None:
    """Validate the structural contract for a complete-table write.

    "Complete" refers to the write scope, rather than exhaustive validation
    of Harpy-specific metadata. Checks cover AnnData axis alignment,
    SpatialData annotation and, when annotated, unique, non-null region/instance
    identities, without reading matrix values.

    This helper requires only the AnnData object. It does not validate
    registered feature-matrix metadata, feature-panel references or canonical
    centers. Those checks belong to ``harpy.table._validation.validate_table()``
    and its helpers, which also have access to the surrounding SpatialData object.
    """
    # Accessing aligned mappings invokes AnnData's shape/index checks, including
    # after callers have edited obs/var. It does not read the matrices' values.
    for slot in ("layers", "obsm", "varm", "obsp", "varp"):
        dict(getattr(table, slot))
    if table.X is not None and table.X.shape != table.shape:
        raise ValueError("X shape must match obs and var.")
    if table.raw is not None:
        dict(table.raw.varm)
        if table.raw.X is not None and table.raw.X.shape != (table.n_obs, table.raw.n_vars):
            raise ValueError("raw.X shape must match obs and raw.var.")
    TableModel.validate(table)
    # TableModel.validate() does not check region/instance pair uniqueness.
    spatialdata_attrs = table.uns.get(TableModel.ATTRS_KEY)
    if spatialdata_attrs is not None:
        region_key, instance_key = _annotation_columns(spatialdata_attrs)
        _observation_pairs(table.obs, region_key=region_key, instance_key=instance_key, label="Table observation")


def _validate_table_identities(
    table: AnnData,
    *,
    obs_identity: pd.DataFrame | AxisNames | None = None,
    var_names: AxisNames | None = None,
    raw_var_names: AxisNames | None = None,
) -> None:
    """Check optional explicit identities against the table without reordering.

    Any supplied obs_identity, var_names or raw_var_names must match this
    table's corresponding axis in value and order, not the destination's old
    axis: a complete replacement may change those axes. Omitted arguments
    need no check because the table already carries its axis dataframes.
    """
    if obs_identity is not None:
        spatialdata_attrs = table.uns.get(TableModel.ATTRS_KEY)
        if spatialdata_attrs is not None:
            region_key, instance_key = _annotation_columns(spatialdata_attrs)
            expected = _observation_pairs(
                table.obs, region_key=region_key, instance_key=instance_key, label="Table observation"
            )
            _match_observation_identity(obs_identity, expected, region_key=region_key, instance_key=instance_key)
        else:
            _match_identity(
                _named_identity(obs_identity, label="obs_identity"),
                _named_identity(table.obs.index, label="Table obs"),
                label="obs_identity",
            )
    if var_names is not None:
        _match_identity(
            _named_identity(var_names, label="var_names"),
            _named_identity(table.var.index, label="Table var"),
            label="var_names",
        )
    if raw_var_names is not None:
        if table.raw is None:
            raise ValueError("raw_var_names was supplied, but the table has no raw feature axis.")
        _match_identity(
            _named_identity(raw_var_names, label="raw_var_names"),
            _named_identity(table.raw.var.index, label="Table raw.var"),
            label="raw_var_names",
        )
