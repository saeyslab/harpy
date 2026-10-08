from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
from anndata import AnnData
from loguru import logger as log
from spatialdata import SpatialData
from spatialdata.models import TableModel

from harpy._storage._anndata import _MATRIX_MAPPINGS, _read_anndata_table
from harpy.table.io._write import _write_table_operation
from harpy.utils._keys import _INSTANCE_KEY, _REGION_KEY


def add_table(
    sdata: SpatialData,
    adata: AnnData,
    output_table_name: str,
    region: list[str] | None,
    instance_key: str = _INSTANCE_KEY,
    region_key: str = _REGION_KEY,
    overwrite: bool = False,
) -> SpatialData:
    """Add an AnnData table to SpatialData.

    If ``sdata`` is backed by a Zarr store, also write the table to that store.
    Otherwise, attach the table to ``sdata`` only in memory.

    The prepared table is attached at ``sdata.tables[output_table_name]``.
    After writing to a Zarr store, only that table is reopened with lazy matrices.

    Parameters
    ----------
    sdata
        The :class:`~spatialdata.SpatialData` object to update.
    adata
        Parsed or unparsed :class:`~anndata.AnnData`. When ``region`` is provided,
        ``adata.obs`` must already contain the columns named by ``region_key``
        and ``instance_key``; their values are not inferred or created.
        For an already-parsed table, these key names must match its existing
        ``adata.uns["spatialdata_attrs"]`` metadata. Custom key names must be
        supplied explicitly. Parsing updates a separate prepared table;
        ``adata`` remains unchanged.
    output_table_name
        Name of the table in ``sdata.tables`` and, when backed, in the store.
    region
        Names of spatial elements annotated by the resulting table, corresponding
        to the values in ``adata.obs[region_key]``. Set to ``None`` to create an
        unannotated table, even if ``adata`` is already parsed. This removes
        linkage metadata from the prepared result, not its observation columns.
    instance_key
        Name of the ``adata.obs`` column containing instance IDs within each region.
        Ignored if ``region`` is ``None``.
    region_key
        Name of the ``adata.obs`` column containing each observation's region.
        Ignored if ``region`` is ``None``.
    overwrite
        Allow replacement of a table that exists in the store of backed
        ``sdata``. It only concerns the store: a table attached to backed
        ``sdata`` but never saved is replaced without it. Ignored for unbacked
        ``sdata``, where existing tables are always replaced.

    Returns
    -------
    spatialdata.SpatialData
        The same ``sdata`` object, with the prepared table attached at
        ``sdata.tables[output_table_name]``.

    Notes
    -----
    Annotation preparation copies metadata, not entire matrices. For unbacked
    ``sdata``, the attached table may share matrix data with ``adata``; this is
    not a deep-copy API.

    Backed writes finish serialization before replacing old data. Installation and
    metadata finalization remain covered by rollback; handled failures restore the
    previous stored table and attached entry. Unrelated elements are unchanged.
    External references to a replaced table are not refreshed; use
    ``sdata.tables[output_table_name]`` after success.

    Backed writes store matrices as :func:`harpy.table.io.write_table` does; its
    Notes list which elements are stored as matrices.

    See Also
    --------
    harpy.table.io.write_table : Write a complete table without updating ``sdata``.

    Examples
    --------
    In this example, ``sdata`` is backed by a Zarr store. Each row in
    ``adata.obs`` has a ``"region"`` value of ``"cells"`` and an
    ``"instance_id"`` identifying the corresponding cell.

    The call writes the table to the store as ``"processed"`` and updates
    ``sdata.tables["processed"]`` with the saved table. Its matrices are
    reopened lazily, without loading their values into memory.

    .. code-block:: python

        sdata = hp.tb.io.add_table(
            sdata,
            adata=adata,
            output_table_name="processed",
            region=["cells"],
            region_key="region",
            instance_key="instance_id",
            overwrite=True,
        )
        processed = sdata.tables["processed"]
    """
    return _add_table(
        sdata,
        adata=adata,
        output_table_name=output_table_name,
        region=region,
        instance_key=instance_key,
        region_key=region_key,
        overwrite=overwrite,
    )


def _needs_stringdtype_copy_workaround() -> bool:
    """Return whether NumPy still needs the gh-28609 StringDType copy workaround."""
    # Upstream NumPy issue: https://github.com/numpy/numpy/issues/28609
    return np.lib.NumpyVersion(np.__version__) < np.lib.NumpyVersion("2.2.5")


def _add_table(
    sdata: SpatialData,
    adata: AnnData,
    output_table_name: str,
    region: list[str] | None,  # list of labels elements
    instance_key: str = _INSTANCE_KEY,  # ignored if region is None
    region_key: str = _REGION_KEY,  # ignored if region is None
    overwrite: bool = False,
) -> SpatialData:
    if not isinstance(adata, AnnData):
        raise TypeError("adata must be an AnnData.")
    if region is not None:
        # Supplied keys must agree with any existing SpatialData annotation.
        if TableModel.ATTRS_KEY in adata.uns:
            spatialdata_attrs = adata.uns[TableModel.ATTRS_KEY]
            if (
                TableModel.REGION_KEY_KEY in spatialdata_attrs
                and region_key != spatialdata_attrs[TableModel.REGION_KEY_KEY]
            ):
                raise ValueError(
                    f"The provided region key '{region_key}' is not equal to the region key "
                    f"in the AnnData object ({spatialdata_attrs[TableModel.REGION_KEY_KEY]}). "
                    "This is not allowed."
                )
            if (
                TableModel.INSTANCE_KEY in spatialdata_attrs
                and instance_key != spatialdata_attrs[TableModel.INSTANCE_KEY]
            ):
                raise ValueError(
                    f"The provided instance key '{instance_key}' is not equal to the instance key "
                    f"in the AnnData object ({spatialdata_attrs[TableModel.INSTANCE_KEY]}). "
                    "This is not allowed."
                )

        if region_key not in adata.obs.columns:
            raise ValueError(
                f"Provided 'AnnData' object should contain a column '{region_key}' in 'adata.obs'. "
                "Linking the observations to a region (e.g. a labels element) in 'sdata'."
            )
        if instance_key not in adata.obs.columns:
            raise ValueError(
                f"Provided 'AnnData' object should contain a column '{instance_key}' in 'adata.obs'. "
                "Linking the observations to a region (e.g. a labels element) in 'sdata'."
            )

    # Parsing can replace obs columns and uns metadata. Isolate those annotations
    # without AnnData.copy(), which can copy or materialize entire matrices.
    # Pass matrices through without copying or materializing them here.
    # The writer prepares lazy/backed matrices for chunked serialization.
    raw = None
    if adata.raw is not None:
        raw = {"X": adata.raw.X, "var": adata.raw.var.copy(deep=True), "varm": dict(adata.raw.varm)}
    prepared = AnnData(
        X=adata.X,
        obs=adata.obs.copy(deep=True),
        var=adata.var.copy(deep=True),
        uns=deepcopy(adata.uns),
        raw=raw,
        **{slot: dict(getattr(adata, slot)) for slot in _MATRIX_MAPPINGS},
    )
    prepared.uns.pop(TableModel.ATTRS_KEY, None)
    prepared = TableModel.parse(
        prepared,
        region=deepcopy(region),
        region_key=region_key if region is not None else None,
        instance_key=instance_key if region is not None else None,
    )

    store = sdata.path
    previous_table = sdata.tables.get(output_table_name)
    # overwrite only concerns the store: a table attached but never saved is
    # replaced without it, as for unbacked sdata.
    if store is not None and not overwrite and (Path(store) / "tables" / output_table_name).exists():
        raise ValueError(
            f"Attempting to overwrite 'sdata.tables[\"{output_table_name}\"]', but overwrite is set to False. "
            "Set overwrite to True to overwrite the .zarr store."
        )
    try:
        if store is not None:
            with _write_table_operation(
                store, table_name=output_table_name, adata=prepared, overwrite=overwrite
            ) as published:
                # _write_table_operation() is paused at its yield while this with-body runs.
                # Reopen only this table from permanent paths and attach it to sdata.
                # After this body succeeds, the writer resumes after yield to consolidate
                # metadata and discard backups. Errors in this body instead propagate
                # back into the writer's rollback handling.
                reopened = _read_anndata_table(published, mode="lazy")
                if _needs_stringdtype_copy_workaround():
                    _cast_stringdtype_uns(reopened)
                sdata.tables[output_table_name] = reopened
        else:
            # Preserve the existing unbacked behavior: replacement does not
            # require overwrite=True, and no serialization is performed.
            if _needs_stringdtype_copy_workaround():
                _cast_stringdtype_uns(prepared)
            sdata.tables[output_table_name] = prepared
    except BaseException:
        # The shared writer restores disk state. Restore this object's entry
        # too, including when attachment changed it before raising an error.
        if sdata.tables.get(output_table_name) is not previous_table:
            if previous_table is None:
                del sdata.tables[output_table_name]
            else:
                sdata.tables[output_table_name] = previous_table
        raise

    return sdata


def _cast_stringdtype_uns(adata: AnnData, target_dtype="U7"):
    """Normalize top-level StringDType arrays in this table's uns only."""
    target_dtype = np.dtype(target_dtype)
    target_len = target_dtype.itemsize // np.dtype("U1").itemsize if target_dtype.kind == "U" else None

    for key, value in list(adata.uns.items()):
        if isinstance(value, np.ndarray) and (
            "StringDType" in str(value.dtype) or getattr(value.dtype, "kind", None) == "T"
        ):
            value_list = value.tolist()
            flat_values = np.asarray(value_list, dtype=object).ravel()
            required_len = max((len(str(v)) for v in flat_values), default=1)

            if target_len is not None and required_len > target_len:
                cast_dtype = np.dtype(f"U{required_len}")
                log.info(f"Casting key {key} to '{cast_dtype}' to avoid truncation.")
            else:
                cast_dtype = target_dtype
                log.info(f"Casting key {key} to '{cast_dtype}'.")

            adata.uns[key] = np.asarray(value_list, dtype=cast_dtype)
    return
