"""Structural and ordered-identity checks for scoped table writes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral

import numpy as np
import pandas as pd
import zarr
from anndata import AnnData
from anndata.io import read_elem
from spatialdata.models import TableModel

from harpy._storage._anndata import _MissingAnnDataElement, _read_anndata_element
from harpy.table._io import ComponentPath

type AxisNames = pd.Index | Sequence[str]


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


def _check_component_destination(group: zarr.Group, path: ComponentPath, *, overwrite: bool) -> None:
    """Check traversal and collisions without decoding the destination value."""
    parent = group
    for key_index, key in enumerate(path):
        if key not in parent:
            if path[: key_index + 1] == ("raw",):
                raise ValueError("Raw creation must be prepared before checking individual raw destinations.")
            return
        element = parent[key]
        if key_index == len(path) - 1:
            if not overwrite:
                raise FileExistsError(f"Table component {path!r} already exists; use overwrite=True.")
            return
        expected_encoding = "raw" if path[: key_index + 1] == ("raw",) else "dict"
        if (
            not isinstance(element, zarr.Group)
            or element.attrs.get("encoding-type") != expected_encoding
            or element.attrs.get("encoding-version") != "0.1.0"
        ):
            raise ValueError(f"AnnData component parent {path[: key_index + 1]!r} is not an encoded mapping.")
        parent = element


def _axis_index(group: zarr.Group, path: ComponentPath) -> pd.Index:
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


def _check_annotation_updates(group: zarr.Group, components: Mapping[ComponentPath, object]) -> None:
    """Protect spatial linkage, including when replacing all of uns."""
    annotation_path = ("uns", TableModel.ATTRS_KEY)
    for path, value in components.items():
        if path == ("uns",):
            if not isinstance(value, Mapping):
                raise ValueError("uns must be a mapping.")
            stored = _read_spatialdata_attrs(group)
            present = stored is not None
            if (TableModel.ATTRS_KEY in value) != present or (
                present and not _same_metadata(stored, value[TableModel.ATTRS_KEY])
            ):
                raise ValueError("Component writes must preserve SpatialData annotation; use write_table().")
        elif path[:2] == annotation_path:
            try:
                stored = _read_anndata_element(group, path, mode="eager")
            except _MissingAnnDataElement:
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


def _observation_pairs(frame: pd.DataFrame, spatialdata_attrs: dict, *, label: str) -> pd.MultiIndex:
    keys = [spatialdata_attrs[TableModel.REGION_KEY_KEY], spatialdata_attrs[TableModel.INSTANCE_KEY]]
    if any(key not in frame for key in keys):
        raise ValueError(f"{label} must contain the stored region and instance columns {keys!r}.")
    identity = frame[keys]
    if identity.isna().any().any() or identity.duplicated().any():
        raise ValueError(f"{label} region/instance pairs must be non-null and unique.")
    # Check SpatialData's required column types without reading any expression data.
    TableModel.validate(AnnData(obs=identity, uns={TableModel.ATTRS_KEY: spatialdata_attrs}))
    return pd.MultiIndex.from_frame(identity.astype(object))


def _validate_component_values(
    group: zarr.Group,
    components: Mapping[ComponentPath, object],
    *,
    obs_identity: pd.DataFrame | AxisNames | None,
    var_names: AxisNames | None,
    raw_var_names: AxisNames | None,
    new_raw_var: pd.DataFrame | None = None,
) -> None:
    """Validate only affected axes and linkage, before staging and after serialization.

    ``obs_identity`` describes the observation identities and their order in
    the submitted matrices. ``var_names`` and ``raw_var_names`` describe the
    corresponding feature identities and order. These arguments are used for
    validation, never to reorder matrix rows or columns.

    Supplied obs/var/raw.var frames can provide those identities instead.
    Only required indices and spatial annotation columns are read from the
    original table, never its numerical matrices.
    During raw creation, new_raw_var defines the new feature axis; all other
    axes still come from the stored table.
    """
    axes_by_path = {}
    for path in components:
        slot = path[0]
        if slot == "uns":
            axes = ()
        elif slot in {"obs", "obsm", "obsp"}:
            axes = ("obs",)
        elif slot in {"var", "varm", "varp"}:
            axes = ("var",)
        elif slot == "raw":
            axes = ("obs", "raw_var") if path[1] == "X" else ("raw_var",)
        else:
            axes = ("obs", "var")
        axes_by_path[path] = axes

    required = {axis for axes in axes_by_path.values() for axis in axes}
    explicit = {"obs": obs_identity, "var": var_names, "raw_var": raw_var_names}
    frame_paths = {"obs": ("obs",), "var": ("var",), "raw_var": ("raw", "var")}
    indices = {}
    for axis, frame_path in frame_paths.items():
        if axis not in required and explicit[axis] is None:
            continue
        if axis == "raw_var" and new_raw_var is not None:
            indices[axis] = new_raw_var.index
        else:
            try:
                indices[axis] = _axis_index(group, frame_path)
            except KeyError:
                raise ValueError(f"The stored table has no {axis!r} axis.") from None
        supplied_axis_frame = components.get(frame_path)
        if frame_path in components:
            if not isinstance(supplied_axis_frame, pd.DataFrame):
                raise TypeError(f"Component {frame_path!r} must be a pandas DataFrame.")
            _match_identity(supplied_axis_frame.index, indices[axis], label=f"{axis} dataframe index")

        spatialdata_attrs = _read_spatialdata_attrs(group) if axis == "obs" else None
        # Keep the obs check explicit: this branch validates region/instance
        # identities only for annotated observations, never var/raw.var. Any
        # supplied_axis_frame here is therefore obs; unannotated obs uses the
        # index-based validation below.
        if axis == "obs" and spatialdata_attrs is not None:
            keys = [spatialdata_attrs[TableModel.REGION_KEY_KEY], spatialdata_attrs[TableModel.INSTANCE_KEY]]
            stored_obs_identity = pd.DataFrame({key: read_elem(group["obs"][key]) for key in keys}, index=indices[axis])
            expected = _observation_pairs(stored_obs_identity, spatialdata_attrs, label="Stored observation")
            if supplied_axis_frame is not None:
                _match_identity(
                    _observation_pairs(supplied_axis_frame, spatialdata_attrs, label="Supplied obs"),
                    expected,
                    label="obs_identity",
                )
            if obs_identity is not None:
                if not isinstance(obs_identity, pd.DataFrame):
                    raise TypeError("obs_identity must be a two-column region/instance dataframe for annotated tables.")
                if set(obs_identity.columns) != set(keys) or len(obs_identity.columns) != 2:
                    raise ValueError(f"obs_identity must contain exactly the columns {keys!r}.")
                _match_identity(
                    _observation_pairs(obs_identity, spatialdata_attrs, label="obs_identity"),
                    expected,
                    label="obs_identity",
                )
        else:
            # For var, raw.var and obs without SpatialData annotation, identities
            # are ordered index names, not region/instance pairs. Check the
            # supplied dataframe index and any explicit names against that axis.
            expected = _named_identity(indices[axis], label=f"Stored {axis}")
            for value in (None if supplied_axis_frame is None else supplied_axis_frame.index, explicit[axis]):
                if value is not None:
                    _match_identity(_named_identity(value, label=axis), expected, label=axis)
        if supplied_axis_frame is None and explicit[axis] is None:
            parameter = {"obs": "obs_identity", "var": "var_names", "raw_var": "raw_var_names"}[axis]
            raise ValueError(
                f"Supply {parameter} or the corresponding dataframe when writing {axis}-aligned components."
            )

    _check_annotation_updates(group, components)
    for path, axes in axes_by_path.items():
        value = components[path]
        if not axes or path in frame_paths.values():
            continue
        if path in {("X",), ("raw", "X")} and value is None:
            continue
        shape = getattr(value, "shape", ())
        if not shape or any(not isinstance(size, Integral) or size < 0 for size in shape):
            raise ValueError(f"Component {path!r} requires a matrix with a known shape.")
        if path[0] in {"obsp", "varp"}:
            expected_shape = (len(indices[axes[0]]),) * 2
        elif len(axes) == 2:
            expected_shape = tuple(len(indices[axis]) for axis in axes)
        else:
            expected_shape = (len(indices[axes[0]]), *shape[1:])
        if tuple(shape) != expected_shape:
            raise ValueError(f"Component {path!r} has shape {shape}; expected {expected_shape}.")
        if isinstance(value, pd.DataFrame):
            _match_identity(value.index, indices[axes[0]], label=f"Component {path!r} dataframe index")


def _validate_complete_table(table: AnnData) -> None:
    """Check AnnData alignment and SpatialData linkage without computing matrices."""
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
