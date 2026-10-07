"""Write back the components of a table that are new or changed against its store.

``write_table_updates`` classifies each component of an AnnData table as new,
changed or unchanged against the stored table, and writes the new and changed
ones in one ``write_table_components`` call. ``add_table_updates`` does the
same for a table attached to backed SpatialData, and reinstalls what it wrote.
``_component_changed`` decides, for one value whose path exists in the store,
whether it differs from the stored element.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import zarr
from anndata import AnnData
from anndata.abc import CSCDataset, CSRDataset
from loguru import logger as log
from scipy import sparse
from spatialdata import SpatialData
from spatialdata.models import TableModel

from harpy._storage._anndata import (
    _MATRIX_MAPPINGS,
    _decode_anndata_element,
    _element_identity,
    _lazy_read_source,
    _read_anndata_element,
)
from harpy.table.io._components import _component_update_destination, _update_table_components
from harpy.table.io._read import ComponentPath, _open_table_group, _validate_path_segment
from harpy.table.io._write import write_table_components
from harpy.table.io._write_validation import (
    _annotation_columns,
    _read_observation_identity,
    _read_spatialdata_attrs,
    _storage_axis_index,
)

type _InMemoryMatrix = np.ndarray | sparse.csr_matrix | sparse.csc_matrix | sparse.csr_array | sparse.csc_array

_AXIS_WAYS_OUT = (
    "To store a table with other cells or genes, use write_table(..., overwrite=True) to replace the stored "
    "table, or write_table under a new table_name to keep it. To update obsm matrices that cover whole regions, "
    "use write_table_components_by_region or add_table_components_by_region, with fill_values for the other "
    "observations."
)


def write_table_updates(
    store: str | PathLike[str],
    *,
    table_name: str,
    adata: AnnData,
    x_to: ComponentPath | None = None,
    overwrite: bool = False,
) -> None:
    """Write the components of a table that are new or changed against its store.

    Compares ``adata``, typically read with :func:`harpy.table.read_table` and
    then processed, for example by scanpy, with the stored table, and writes
    only its new and changed components, in one rollback-protected
    :func:`harpy.table.write_table_components` call. Components missing from
    ``adata`` are never deleted.

    Parameters
    ----------
    store
        Local path to an existing SpatialData Zarr root.
    table_name
        Name of the existing table to update.
    adata
        The table with its updates. Its axes must match the stored table:
        the same observations and features, in the same order. It is not
        modified.
    x_to
        Destination of a new or changed ``X``. ``("layers", key)`` writes it to
        that layer and keeps the stored ``X``, usually the counts;
        ``("X",)`` replaces the stored ``X``. Without ``x_to``, a changed ``X``
        raises: the stored ``X`` is never replaced implicitly. A new ``X``, for
        a stored table without one, is then written to ``X``.
    overwrite
        Allow replacing components that exist in storage, the ``x_to``
        destination included. A scanpy run typically changes ``obs``, ``var``
        and ``uns`` entries, which exist, so its write-back needs ``True``. New
        components need no permission. Whether a component is written at all is
        decided by comparison, not by ``overwrite``.

    Raises
    ------
    ValueError
        If the axes of ``adata`` differ from storage: its ``obs_names``, the
        region/instance pairs of an annotated table, its ``var_names``, or,
        when both have ``raw``, its ``raw.var_names``. If
        ``uns["spatialdata_attrs"]`` differs from the stored annotation, or
        would add one. If ``X`` changed without ``x_to``, ``x_to`` is neither
        ``("layers", key)`` nor ``("X",)``, or ``X`` and ``adata.layers[key]``
        would both be written to ``x_to``.
    FileExistsError
        Without ``overwrite=True``, if a component to write exists in storage.
        The message lists every such component.

    Notes
    -----
    A component is new if its path does not exist in storage, and written.
    Otherwise it is written only if it changed:

    - a lazy read of its own stored element (:func:`harpy.table.read_table`
      with ``mode="lazy"``) is unchanged, without reading values, as long as no
      operation was applied to it: copies and ``persist()`` keep it
      recognised, while any operation, a real rechunk included, makes it
      changed. A backed handle to its own stored element is unchanged too;
    - values in memory are compared with the stored values: ``obs``, ``var``
      and DataFrame-valued entries strictly, in index, columns, dtypes,
      categories and values; ``uns`` per top-level key; NumPy and SciPy
      CSR/CSC matrices by shape, dtype, format and number of stored values
      first, then block by block, stopping at the first difference;
    - any other value is changed: a derived Dask array, without computing it
      for a comparison, a backed handle to another element, or a value of
      another type, which the writer then accepts or rejects.

    The SpatialData annotation, ``uns["spatialdata_attrs"]``, is never
    written: an unchanged annotation is skipped, a missing one stays in
    storage, and a changed or added one raises.

    ``obs`` and ``var`` are written whole, and ``uns`` per top-level key, so
    columns and nested keys removed from them are removed from storage too.
    Missing components, top-level ``uns`` keys included, stay in storage;
    remove them with :func:`harpy.table.delete_table_components`.

    Not detected: a change that another writer made in storage since the
    read, to a component that ``adata`` holds in memory or that changed in
    ``adata``, is overwritten (last writer wins); and a persisted read that
    another writer made stale. Concurrent-access isolation is not provided.

    Returns None. ``adata`` is not modified: after ``x_to``, ``adata.X`` still
    holds the written values and ``adata.layers`` lacks the destination. Reopen
    the table to get one that matches the store; called again with the same
    ``adata``, the function writes ``X`` again.

    See Also
    --------
    harpy.table.read_table : Read the table, lazily, before processing it.
    harpy.table.add_table_updates : The same for a table attached to backed SpatialData.
    harpy.table.write_table_components : Write selected components explicitly.
    harpy.table.write_table : Write a complete table, also with other axes.

    Examples
    --------
    .. code-block:: python

        adata = hp.tb.read_table("sdata.zarr", table_name="counts", mode="lazy")
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.pca(adata)
        # overwrite=True for re-runs: their layer and X_pca replace those of an earlier run.
        hp.tb.write_table_updates(
            "sdata.zarr", table_name="counts", adata=adata, x_to=("layers", "log1p"), overwrite=True
        )
        adata = hp.tb.read_table("sdata.zarr", table_name="counts", mode="lazy")
    """
    if not isinstance(adata, AnnData):
        raise TypeError("adata must be an AnnData.")
    plan = _plan_table_updates(store, table_name=table_name, adata=adata, x_to=x_to, overwrite=overwrite)
    if plan is None:
        return
    write_table_components(
        store,
        table_name=table_name,
        components=plan.components,
        obs_identity=plan.obs_identity,
        var_names=plan.var_names,
        raw_var_names=plan.raw_var_names,
        overwrite=overwrite,
    )


def add_table_updates(
    sdata: SpatialData,
    *,
    table_name: str,
    x_to: ComponentPath | None = None,
    overwrite: bool = False,
) -> SpatialData:
    """Write the components of an attached table that are new or changed against its store, and reinstall them.

    The SpatialData counterpart of :func:`harpy.table.write_table_updates`: it
    compares ``sdata.tables[table_name]``, typically read lazily and processed
    in place, for example by scanpy, with the same table in the store of
    ``sdata``, and writes only its new and changed components, with the same
    rules for what changed, ``x_to`` and ``overwrite``. It then reinstalls the
    written components in the attached table, reopened lazily from the store,
    as :func:`harpy.table.add_table_components` does, so that the attached
    table matches the store and a second call writes nothing.

    Parameters
    ----------
    sdata
        SpatialData backed by a store, with ``table_name`` attached and in
        that store.
    table_name
        Name of the attached table to update.
    x_to
        Destination of a new or changed ``X``, as for
        :func:`harpy.table.write_table_updates`. When ``X`` is written to a
        layer, the attached ``X`` is reinstalled from the store too, usually
        the counts, or set to ``None`` if the store has no ``X``.
    overwrite
        Allow replacing components that exist in storage, as for
        :func:`harpy.table.write_table_updates`. A component that is attached
        but not stored is new, and needs no permission.

    Returns
    -------
    spatialdata.SpatialData
        The supplied ``sdata``. Its attached table object is kept; only the
        slots of written components, and ``X`` after ``x_to``, are replaced.

    Raises
    ------
    ValueError
        If ``sdata`` has no store: there is nothing to compare with. Also for
        the reasons that :func:`harpy.table.write_table_updates` raises.
    FileNotFoundError
        If the store, or the table in it, does not exist.
    FileExistsError
        Without ``overwrite=True``, if a component to write exists in storage.

    Notes
    -----
    The write uses the same staging and rollback as
    :func:`harpy.table.add_table_components`. If it fails, also after the
    attached table was updated, the store and the attached slots are
    restored. External references to replaced values are not refreshed.

    See Also
    --------
    harpy.table.write_table_updates : Write the updates of a table to a store, without SpatialData.
    harpy.table.add_table_components : Update selected components of an attached table explicitly.

    Examples
    --------
    .. code-block:: python

        sdata = hp.io.read_zarr("sdata.zarr", table_mode="lazy")
        adata = sdata.tables["counts"]
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.pca(adata)
        # overwrite=True for re-runs: their layer and X_pca replace those of an earlier run.
        hp.tb.add_table_updates(sdata, table_name="counts", x_to=("layers", "log1p"), overwrite=True)
    """
    if not isinstance(sdata, SpatialData):
        raise TypeError("sdata must be a SpatialData object.")
    if sdata.path is None:
        raise ValueError(
            f"add_table_updates compares the attached table {table_name!r} with its store, but sdata has none "
            "(sdata.path is None). Write the SpatialData itself with sdata.write(path); write the table into an "
            "existing store with hp.tb.write_table; or, if the table was read from a store, call "
            f"hp.tb.write_table_updates(store, table_name=..., adata=sdata.tables[{table_name!r}]) with that store."
        )
    try:
        table, _ = _component_update_destination(sdata, table_name=table_name)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Table {table_name!r} has no stored counterpart to compare with. Write the SpatialData with "
            "sdata.write(path), or the table with hp.tb.add_table, which writes and attaches it whole."
        ) from error
    plan = _plan_table_updates(sdata.path, table_name=table_name, adata=table, x_to=x_to, overwrite=overwrite)
    if plan is None:
        return sdata
    return _update_table_components(
        sdata,
        table_name=table_name,
        components=plan.components,
        delete=(),
        obs_identity=plan.obs_identity,
        var_names=plan.var_names,
        raw_var_names=plan.raw_var_names,
        overwrite=overwrite,
        # X written to a layer leaves the stored X as it is: reinstall it, so the
        # attached X matches the store and the next call does not write it again.
        reopen_also=(("X",),) if plan.x_destination not in (None, ("X",)) else (),
    )


@dataclass(frozen=True)
class _TableUpdates:
    """The components to write back, and the identities that ``write_table_components`` checks them with."""

    components: dict[ComponentPath, object]
    obs_identity: pd.DataFrame | pd.Index
    var_names: pd.Index
    raw_var_names: pd.Index | None
    # Where X is written, or None if X is not written.
    x_destination: ComponentPath | None


def _plan_table_updates(
    store: str | PathLike[str], *, table_name: str, adata: AnnData, x_to: ComponentPath | None, overwrite: bool
) -> _TableUpdates | None:
    """Check ``adata`` against the stored table and return what to write, or None if nothing changed.

    Shared by ``write_table_updates`` and ``add_table_updates``: the structural
    checks, the classification of every component, the routing of ``X`` and the
    ``overwrite`` check, which lists every existing destination. Nothing is
    written here.
    """
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean.")
    _validate_x_to(x_to)
    group = _open_table_group(store, table_name=table_name)
    spatialdata_attrs = _read_spatialdata_attrs(group)
    # The structural checks come first: they are cheap, and must fail before
    # any stored values are read for a comparison.
    _check_annotation_unchanged(adata, spatialdata_attrs, table_name=table_name)
    _check_axes_unchanged(group, adata, spatialdata_attrs, table_name=table_name)
    updates, x_destination = _updates_to_write(group, adata, x_to=x_to)
    if not updates:
        log.info(f"Table {table_name!r}: no new or changed components; nothing written.")
        return None
    existing = [path for path in updates if _exists_on_disk(group, path)]
    if existing and not overwrite:
        raise FileExistsError(
            f"Table {table_name!r}: writing the new and changed components would replace "
            f"{', '.join(map(repr, existing))}, which exist in storage; use overwrite=True. "
            "Unchanged components are not written, whatever overwrite."
        )
    if spatialdata_attrs is None:
        obs_identity = adata.obs_names
    else:
        obs_identity = adata.obs[list(_annotation_columns(spatialdata_attrs))]
    log.info(f"Table {table_name!r}: writing new and changed components {list(updates)!r}.")
    return _TableUpdates(
        components=updates,
        obs_identity=obs_identity,
        var_names=adata.var_names,
        # Raw identities are only needed when raw components are written:
        # otherwise the writer would read and check the stored raw axis for
        # nothing, and adata.raw may be None.
        raw_var_names=adata.raw.var_names if any(path[0] == "raw" for path in updates) else None,
        x_destination=x_destination,
    )


def _validate_x_to(x_to: object) -> None:
    """Accept ``None``, ``("X",)`` or ``("layers", key)`` as the destination of ``X``."""
    if x_to is None or (isinstance(x_to, tuple) and x_to == ("X",)):
        return
    if isinstance(x_to, tuple) and len(x_to) == 2 and x_to[0] == "layers":
        _validate_path_segment(x_to[1])
        return
    raise ValueError(f"x_to must be ('layers', key) or ('X',), not {x_to!r}.")


def _check_annotation_unchanged(adata: AnnData, spatialdata_attrs: Mapping | None, *, table_name: str) -> None:
    """Raise if ``adata`` changes or adds the stored SpatialData annotation; a missing one is left alone."""
    if TableModel.ATTRS_KEY not in adata.uns:
        return
    if spatialdata_attrs is None:
        change = "would add SpatialData annotation to the unannotated stored table"
    elif not _same_uns_value(adata.uns[TableModel.ATTRS_KEY], spatialdata_attrs):
        change = "differs from the stored SpatialData annotation of table"
    else:
        return
    raise ValueError(
        f"adata.uns[{TableModel.ATTRS_KEY!r}] {change} {table_name!r}. write_table_updates does not change a "
        "table's SpatialData linkage; use write_table(..., overwrite=True)."
    )


def _check_axes_unchanged(
    group: zarr.Group, adata: AnnData, spatialdata_attrs: Mapping | None, *, table_name: str
) -> None:
    """Raise if the axes of ``adata`` differ from the stored table, naming the ways out.

    Observations are checked by their names and, in an annotated table, also by
    their region/instance pairs, which identify them for the component writer.
    The writer ignores the index of the pairs it is given, but a written ``obs``
    must keep its stored index; checking both here raises once, with the ways
    out, before any value is compared or written.
    """
    # The stored region and instance columns, indexed by the stored obs_names.
    stored_identity = None if spatialdata_attrs is None else _read_observation_identity(group, spatialdata_attrs)
    stored_obs_names = _storage_axis_index(group, ("obs",)) if stored_identity is None else stored_identity.index
    differences = []
    if not _same_values(adata.obs_names, stored_obs_names):
        differences.append("obs_names")
    if stored_identity is not None:
        keys = list(stored_identity.columns)
        if any(key not in adata.obs for key in keys) or not _same_values(adata.obs[keys], stored_identity):
            differences.append(f"region/instance pairs in obs{keys!r}")
    if not _same_values(adata.var_names, _storage_axis_index(group, ("var",))):
        differences.append("var_names")
    if differences:
        raise ValueError(
            f"The {' and '.join(differences)} of adata differ from table {table_name!r} in storage, for example "
            f"after filtering cells or genes; write_table_updates keeps the stored axes. {_AXIS_WAYS_OUT}"
        )
    if (
        adata.raw is not None
        and _has_stored_raw(group)
        and not _same_values(adata.raw.var_names, _storage_axis_index(group, ("raw", "var")))
    ):
        raise ValueError(
            f"The raw.var_names of adata differ from the stored raw of table {table_name!r}; write_table_updates "
            "keeps raw's own gene axis. Use write_table(..., overwrite=True) to replace the stored table, or "
            "write_table under a new table_name to keep it."
        )


def _same_values(left: pd.Index | pd.DataFrame, right: pd.Index | pd.DataFrame) -> bool:
    """Compare ordered identities by value, whatever their dtypes, such as object or string, categorical or not."""
    return bool(np.array_equal(left.to_numpy(dtype=object), right.to_numpy(dtype=object)))


def _has_stored_raw(group: zarr.Group) -> bool:
    """Return whether the stored table holds raw data, rather than no raw or a raw stored as None."""
    return "raw" in group and group["raw"].attrs.get("encoding-type") == "raw"


def _updates_to_write(
    group: zarr.Group, adata: AnnData, *, x_to: ComponentPath | None
) -> tuple[dict[ComponentPath, object], ComponentPath | None]:
    """Return the new and changed components of ``adata``, keyed by the path they are written to.

    A new or changed ``X`` goes to ``x_to`` when given. Without ``x_to``, a
    changed ``X`` raises, and a new ``X``, for a stored table without one, goes
    to ``X``. ``X`` is classified first, so that these errors come before the
    other components are compared.

    Parameters
    ----------
    group
        The stored table, opened read-only by ``_open_table_group``. Used for
        what is stored: the values and element identities that decide whether a
        component that exists changed. Whether a component exists is decided on
        disk, at the table's path (``_exists_on_disk``).
    adata
        The table to write back. It is read, never modified.
    x_to
        The destination of ``X``, already validated by the caller.

    Returns
    -------
    updates
        The components to write, keyed by destination: ``X`` under ``x_to`` when
        redirected, every other component under its own path.
    x_destination
        The path that ``X`` is written to, or None if ``X`` is not written.

    Raises
    ------
    ValueError
        If ``X`` changed and storage has an ``X``, without ``x_to``; or if ``X``
        and ``adata.layers[key]`` would both be written to ``x_to``.
    """
    components = _table_components(adata)
    to_write: dict[ComponentPath, bool] = {}

    def will_be_written(path: ComponentPath) -> bool:
        if path not in to_write:
            to_write[path] = _new_or_changed(group, path, components[path])
        return to_write[path]

    updates = {}
    x_destination = None
    if ("X",) in components and will_be_written(("X",)):
        if x_to is None and _exists_on_disk(group, ("X",)):
            raise ValueError(
                "adata.X differs from the stored X, which write_table_updates never replaces implicitly. Pass "
                "x_to=('layers', key) to write it to a layer and keep the stored X, for example "
                "x_to=('layers', 'log1p') after normalisation, or x_to=('X',) to replace the stored X."
            )
        x_destination = ("X",) if x_to is None else x_to
        if x_destination != ("X",) and x_destination in components and will_be_written(x_destination):
            raise ValueError(
                f"x_to={x_destination!r} would receive the changed X, but adata.layers[{x_destination[1]!r}] is new "
                "or changed too: two values for one path. Remove that layer from adata, or choose another key."
            )
        updates[x_destination] = components[("X",)]
    for path, value in components.items():
        if path != ("X",) and path not in updates and will_be_written(path):
            updates[path] = value
    return updates, x_destination


def _table_components(adata: AnnData) -> dict[ComponentPath, object]:
    """Return every component of ``adata`` at the path that ``write_table_components`` writes it to.

    ``X``, when present; ``obs`` and ``var`` whole; each entry of ``layers``,
    ``obsm``, ``varm``, ``obsp`` and ``varp``; each top-level ``uns`` key except
    the SpatialData annotation, which is checked separately and never written;
    and ``raw``'s ``X``, ``var`` and ``varm`` entries.
    """
    components: dict[ComponentPath, object] = {} if adata.X is None else {("X",): adata.X}
    components[("obs",)] = adata.obs
    components[("var",)] = adata.var
    for slot in _MATRIX_MAPPINGS:
        components.update({(slot, key): value for key, value in getattr(adata, slot).items()})
    components.update({("uns", key): value for key, value in adata.uns.items() if key != TableModel.ATTRS_KEY})
    if adata.raw is not None:
        components[("raw", "X")] = adata.raw.X
        components[("raw", "var")] = adata.raw.var
        components.update({("raw", "varm", key): value for key, value in adata.raw.varm.items()})
    return components


def _new_or_changed(group: zarr.Group, path: ComponentPath, value: object) -> bool:
    """Return whether a component will be written: new if its path does not exist in storage, else if it changed.

    Existence is checked on disk (``_exists_on_disk``). A path on disk that Zarr
    does not recognise as an element is written too; the writer then decides
    whether it can be replaced.
    """
    if not _exists_on_disk(group, path) or "/".join(path) not in group:
        return True
    return _component_changed(group, path, value)


def _exists_on_disk(group: zarr.Group, path: ComponentPath) -> bool:
    """Return whether a component's path exists on disk, where the writer decides existence for ``overwrite``.

    A path can exist on disk without Zarr recognising it as an element, so
    existence is checked the way the writer checks it. ``group`` is opened from
    the SpatialData root by ``_open_table_group``, so its path, such as
    ``tables/counts``, is relative to that root: the table's path on disk is the
    one the writer builds from the store.
    """
    return (Path(group.store.root) / group.path).joinpath(*path).exists()


def _component_changed(group: zarr.Group, path: ComponentPath, value: object) -> bool:
    """Return whether a value of an AnnData table differs from its stored element.

    The element at ``path`` must exist in the table ``group``; a path that does
    not exist is new, which the caller decides before calling this. The rules:

    - ``uns`` entries, per top-level key, are compared by value
      (``_same_uns_value``);
    - dataframes, ``obs``, ``var``, ``raw.var`` and DataFrame-valued ``obsm`` or
      ``varm`` entries, are compared strictly (``_same_dataframe``);
    - a Dask array is unchanged only if it is a registered lazy read of exactly
      this element (``_lazy_read_source``); any other Dask array, derived or a
      read of another element, is changed, without computing it;
    - a backed handle (a ``zarr.Array`` or a CSR/CSC dataset) is unchanged only
      if it points to exactly this element, wherever its store was opened;
    - an in-memory NumPy or SciPy matrix is compared with the stored values,
      metadata first, then block by block (``_matrix_equals_stored``);
    - any other value counts as changed, and the writer validates it.

    Elements are identified by their resolved path on disk
    (``_element_identity``). Only a ``LocalStore`` gives elements one, so
    registered reads and backed handles of elements in other stores count as
    changed.
    """
    if path[0] == "uns":
        return not _same_uns_value(value, _read_anndata_element(group, path, mode="eager"))
    element = group["/".join(path)]
    stored_dataframe = element.attrs.get("encoding-type") == "dataframe"
    if isinstance(value, pd.DataFrame) or stored_dataframe:
        # A dataframe replacing a matrix, or the reverse, is changed without reading the stored value.
        if not (isinstance(value, pd.DataFrame) and stored_dataframe):
            return True
        return not _same_dataframe(value, _read_anndata_element(group, path, mode="eager"))
    identity = _element_identity(element)
    if isinstance(value, da.Array):
        return identity is None or _lazy_read_source(value) != identity
    if isinstance(value, (zarr.Array, CSRDataset, CSCDataset)):
        return identity is None or _backed_identity(value) != identity
    if not (isinstance(value, np.ndarray) or (sparse.issparse(value) and value.format in {"csr", "csc"})):
        # Not compared: the writer accepts or rejects it with its own validation.
        return True
    return not _matrix_equals_stored(value, element)


def _backed_identity(value: zarr.Array | CSRDataset | CSCDataset) -> Path | None:
    """Return the identity of the stored element a backed handle points to."""
    return _element_identity(value.group if isinstance(value, (CSRDataset, CSCDataset)) else value)


def _same_dataframe(value: pd.DataFrame, stored: pd.DataFrame) -> bool:
    """Compare two dataframes strictly.

    The same index, the same columns in the same order, the same dtypes,
    including categorical categories and their order, and the same values, with
    NaN equal to NaN. Anything else differs: a false "changed" only rewrites a
    small dataframe, while a false "unchanged" would lose data.
    """
    try:
        pd.testing.assert_frame_equal(
            value,
            stored,
            check_exact=True,
            check_index_type=True,
            # Compare the column labels, not the class of the columns index: AnnData
            # stores only the names, and its constructor turns the empty RangeIndex of
            # a dataframe without columns into an empty object index.
            check_column_type=False,
            check_like=False,
        )
    except (AssertionError, TypeError, ValueError):
        return False
    return True


def _same_uns_value(value: object, stored: object) -> bool:
    """Compare one ``uns`` value with its stored value, by value.

    Mappings are compared key by key, recursively; dataframes with
    ``_same_dataframe``; everything else as arrays with ``np.array_equal``,
    with NaN equal to NaN for floating point values, so a list equals the
    array AnnData stores it as. A value whose comparison raises or is ambiguous
    counts as different. Separate from ``_same_metadata``, which also guards
    ``uns["spatialdata_attrs"]`` and must keep its own semantics.
    """
    if isinstance(value, Mapping) or isinstance(stored, Mapping):
        if not (isinstance(value, Mapping) and isinstance(stored, Mapping)):
            return False
        return value.keys() == stored.keys() and all(_same_uns_value(value[key], stored[key]) for key in value)
    if isinstance(value, pd.DataFrame) or isinstance(stored, pd.DataFrame):
        return isinstance(value, pd.DataFrame) and isinstance(stored, pd.DataFrame) and _same_dataframe(value, stored)
    if value is None or stored is None:
        return value is None and stored is None
    try:
        left, right = np.asarray(value), np.asarray(stored)
        if left.shape != right.shape:
            return False
        return bool(np.array_equal(left, right, equal_nan=_floating(left) and _floating(right)))
    except (TypeError, ValueError):
        return False


def _matrix_equals_stored(value: _InMemoryMatrix, element: zarr.Array | zarr.Group) -> bool:
    """Compare an in-memory matrix with its stored element: metadata first, then block by block.

    1. Metadata, without reading values: a different encoding (dense, CSR,
       CSC), shape or dtype differs, and so does, for a sparse matrix, a
       different number of stored values (the length of ``data``). This is
       conservative: a sparse matrix that differs only in explicit zeros has
       another number of stored values.
    2. Values: the stored element is read lazily with the readers, in their
       ``"auto"`` layout, whose blocks lie along one axis: rows for dense and
       CSR matrices, columns for CSC matrices. Each block is computed and
       compared with the matching slice of ``value`` in turn, stopping at the
       first difference. Memory stays at ``value`` plus about one block.

    A string array always differs at step 1: AnnData stores strings with the
    ``string-array`` encoding, so a stored ``array`` has another dtype.
    """
    encoding = element.attrs.get("encoding-type")
    if isinstance(value, np.ndarray):
        if encoding != "array" or tuple(element.shape) != value.shape or element.dtype != value.dtype:
            return False
        axis = 0
    else:
        if (
            encoding != f"{value.format}_matrix"
            or tuple(element.attrs["shape"]) != value.shape
            or element["data"].dtype != value.dtype
            or element["data"].shape[0] != value.nnz
        ):
            return False
        axis = 0 if value.format == "csr" else 1
    stored = _decode_anndata_element(element, mode="lazy")
    # One compute per block, deliberately: it stops at the first differing block
    # and holds about one block in memory. It is not slower than batching blocks
    # or computing the whole matrix at once: the blocks are plain reads that share
    # no upstream work, and Zarr already decodes the stored chunks of one block in
    # parallel.
    start = 0
    for block_index, length in enumerate(stored.chunks[axis]):
        if axis == 0:
            value_block, stored_block = value[start : start + length], stored.blocks[block_index]
        else:
            value_block, stored_block = value[:, start : start + length], stored.blocks[:, block_index]
        if not _same_block(value_block, stored_block.compute()):
            return False
        start += length
    return True


def _same_block(value_block: object, stored_block: object) -> bool:
    """Compare one block of an in-memory matrix with the matching stored block.

    Dense blocks are compared directly. Sparse blocks are compared in canonical
    form (sorted indices, summed duplicates), so that two blocks with the same
    stored values in a different order are equal; a block whose structure
    differs, for example only in explicit zeros, differs. NaN equals NaN.
    """
    if sparse.issparse(value_block):
        if not sparse.issparse(stored_block) or value_block.format != stored_block.format:
            return False
        left, right = value_block.copy(), stored_block.copy()
        left.sum_duplicates()
        right.sum_duplicates()
        return (
            left.shape == right.shape
            and np.array_equal(left.indptr, right.indptr)
            and np.array_equal(left.indices, right.indices)
            and np.array_equal(left.data, right.data, equal_nan=_floating(left.data))
        )
    left, right = np.asarray(value_block), np.asarray(stored_block)
    return left.shape == right.shape and bool(np.array_equal(left, right, equal_nan=_floating(left)))


def _floating(array: np.ndarray) -> bool:
    """Return whether NaN can occur in ``array``, so that ``equal_nan`` applies."""
    return array.dtype.kind in "fc"
