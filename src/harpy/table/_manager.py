from copy import deepcopy

import numpy as np
from anndata import AnnData
from loguru import logger as log
from spatialdata import SpatialData
from spatialdata.models import TableModel

from harpy._storage._anndata import _MATRIX_MAPPINGS, _read_anndata_table
from harpy.table._write import _write_table_operation
from harpy.utils._keys import _INSTANCE_KEY, _REGION_KEY


def _needs_stringdtype_copy_workaround() -> bool:
    """Return whether NumPy still needs the gh-28609 StringDType copy workaround."""
    # Upstream NumPy issue: https://github.com/numpy/numpy/issues/28609
    return np.lib.NumpyVersion(np.__version__) < np.lib.NumpyVersion("2.2.5")


class TableElementManager:
    def add_table(
        self,
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
        if store is not None and previous_table is not None and not overwrite:
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
