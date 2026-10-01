"""Regional component updates with selective installation into an attached table."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
from spatialdata import SpatialData
from spatialdata.models import TableModel

from harpy._storage._anndata import _read_anndata_element
from harpy.table.io._components import (
    _ABSENT,
    _check_in_memory_versus_storage_axes,
    _component_update_destination,
    _install_memory_updates,
    _memory_component,
    _prepare_memory_updates,
)
from harpy.table.io._read import ComponentPath
from harpy.table.io._write_by_region import (
    _DEFAULT_REGIONAL_CHUNK_SIZE,
    _matrix_format,
    _prepare_regional_matrix,
    _regional_row_positions,
    _validate_regional_request,
    _write_table_components_by_region_operation,
)
from harpy.table.io._write_validation import _validate_spatialdata_attrs_unchanged


def add_table_components_by_region(
    sdata: SpatialData,
    *,
    table_name: str,
    components: Mapping[ComponentPath, object],
    obs_identity: pd.DataFrame,
    fill_values: Mapping[ComponentPath, object] | None = None,
    chunk_size: int = _DEFAULT_REGIONAL_CHUNK_SIZE,
    overwrite: bool = False,
) -> SpatialData:
    """Update regional `.obsm` measurements in SpatialData and, when backed, its store.

    Keep the same AnnData object and change only requested components, preserving
    unrelated local annotations and matrix references. Unselected observations
    retain their measurements from the destination matrix::

        SpatialData mode    Destination used for merging    Installed result
        Backed              Stored matrix                   Lazy reopened matrix
        Unbacked            Attached matrix                 Lazy merged matrix

    Parameters
    ----------
    sdata
        SpatialData to update. Its path determines whether changes are persisted,
        independently of matrix representations or AnnData.isbacked.
    table_name
        Existing SpatialData-annotated table attached to sdata, also required in
        storage when backed. AnnData views and HDF5-backed destinations are rejected.
    components
        Nonempty mapping containing individual ``("obsm", key)`` matrices and
        optional whole-value ``uns`` replacements. For example,
        ``{("obsm", "morphology"): regional_features}``. Every matrix contains
        only the rows described by obs_identity, in matching order.

        Supports numeric two-dimensional dense, CSR and CSC matrices, in memory,
        lazy or Zarr-backed. DataFrame matrices and None matrix values are rejected.
        Existing destinations require matching formats, unchanged column counts
        and safe casts into their dtype. No dense/CSR/CSC conversion occurs.
        Prepare accompanying scientific metadata for the complete resulting
        matrix, including retained observations; metadata is not merged by region.
    obs_identity
        Nonempty region/instance dataframe describing every submitted matrix's
        rows, for example ``adata.obs.loc[selected, [region_key, instance_key]]``.
        Use exactly the destination's two identity columns, with categorical
        regions. May cover a subset of the table, but must include every observation
        of each selected region exactly once, in destination-table order, including
        interleaved regions. Actual values select regions, not unused categories.
        The index is ignored; no automatic row reordering occurs.
    fill_values
        For example, ``{("obsm", "morphology"): np.nan}`` fills unselected rows
        of a new dense matrix. Required only for new matrices with unselected rows;
        fills are ignored for existing entries. Scalars must be dtype-compatible,
        and sparse matrices allow only zero fills. Keys must name submitted matrices.
    chunk_size
        Positive integer controlling computational chunk shapes::

            dense / CSR: (chunk_size rows, all columns)
            CSC:         (all rows, chunk_size columns)

        Used for in-memory inputs, backed sparse reads and new partial-region
        matrices. Existing Dask and dense Zarr chunks are retained where compatible
        with merging. Attached CSR matrices with split column chunks, or CSC
        matrices with split row chunks, use a lazily rechunked working array;
        their original chunks are unchanged. Not an on-disk chunk size or byte limit.
    overwrite
        Ignored for unbacked SpatialData; validation still applies. When backed,
        allow replacement of targets present in memory or storage. Without it,
        additions must be absent from both.

    Returns
    -------
    spatialdata.SpatialData
        The supplied sdata, with requested components updated in its existing table.

    Notes
    -----
    Backed updates require the complete attached observation identities and order
    to match storage, including unselected regions. Storage supplies retained
    measurements, not unsaved local replacements of the requested matrix. A target
    present only in memory is new on disk and follows the new-entry-fill rules.

    For unbacked SpatialData, the attached Dask array represents the complete
    updated matrix: selected rows use the supplied measurements; unselected rows
    retain existing measurements, or receive the specified fill for a new entry.
    The numerical merge remains deferred until the array is evaluated, for example
    through `.compute()` or writing. This adapter attaches the result without
    executing the merge or writing to disk, avoiding forced materialization of the
    complete updated matrix merely to attach it. Subsequent processing or writing
    can evaluate its chunks as needed. Inputs already in memory still occupy memory.
    Inputs are not modified, but may be shared: this is not an immutable-snapshot
    or deep-copy API.

    Backed updates stage complete affected matrices in chunks and reopen only
    requested components, with lazy matrices and eager metadata. Installation runs
    inside the shared rollback window. Handled failures restore stored data and
    affected in-memory references. External aliases are not refreshed; callers own
    dirty/stale tracking. No crash recovery or concurrent-access isolation is provided.

    See Also
    --------
    harpy.table.write_table_components_by_region : Update regional measurements on disk only.
    harpy.table.add_table_components : Replace complete components in an attached table.

    Examples
    --------
    .. code-block:: python

        adata = sdata.tables["cell_features"]
        selected = adata.obs["region"].eq("cells_sample_a")
        hp.tb.add_table_components_by_region(
            sdata, table_name="cell_features",
            components={
                ("obsm", "morphology"): regional_features,
                ("uns", "feature_matrices", "morphology"): updated_metadata,
            },
            obs_identity=adata.obs.loc[selected, ["region", "instance_id"]],
            fill_values={("obsm", "morphology"): np.nan},
            overwrite=True,
        )
    """
    paths = _validate_regional_request(components, fill_values=fill_values, chunk_size=chunk_size, overwrite=overwrite)
    chunk_size = int(chunk_size)
    table, group = _component_update_destination(sdata, table_name=table_name)
    spatialdata_attrs = table.uns.get(TableModel.ATTRS_KEY)
    if not isinstance(spatialdata_attrs, Mapping):
        raise ValueError("Regional updates require a SpatialData-annotated table.")
    _validate_spatialdata_attrs_unchanged(spatialdata_attrs, components)
    for path in paths:
        # Check in-memory parents; their values are not the merge source
        # for a backed update.
        in_memory_value = _memory_component(table, path)
        # For backed SpatialData, replacing a component requires overwrite=True
        # even when it exists only in memory, not on disk. Installation would
        # otherwise replace potentially unsaved local data without permission.
        if group is not None and in_memory_value is not _ABSENT and not overwrite:
            raise FileExistsError(f"In-memory component {path!r} already exists; use overwrite=True.")

    if group is not None:
        # Backed SpatialData: group is the destination table in the Zarr store.
        # Installation assigns the complete matrix by row position, so even
        # observations outside the submitted regions must match storage.
        _check_in_memory_versus_storage_axes(
            table=table,
            group=group,
            indices={"obs": table.obs.index},
            stored_new_raw_var=None,
        )
    else:
        # Unbacked SpatialData: use the attached table's observations because
        # there is no backing store from which to read the destination identities.
        table_row_positions = _regional_row_positions(table.obs, spatialdata_attrs, obs_identity)
        replacements = dict(components)
        for path in paths:
            if path[0] != "obsm":
                continue
            existing = None
            if path[1] in table.obsm:
                existing = table.obsm[path[1]]
                _matrix_format(existing, label=f"Attached component {path!r}")
            replacements[path] = _prepare_regional_matrix(
                path,
                components[path],
                existing=existing,
                table_row_positions=table_row_positions,
                n_obs=table.n_obs,
                fill_values=fill_values,
                chunk_size=chunk_size,
            )
        # Unlike full updates in _update_table_components(), we do not call
        # _validate_component_values_against_memory() here: regional preparation
        # already validates identities and matrix compatibility, and SpatialData
        # annotation protection ran above.

    previous = {slot: getattr(table, f"_{slot}") for slot in {path[0] for path in paths}}
    try:
        if group is None:
            updates = _prepare_memory_updates(table, replacements, (), new_raw_var=None)
            _install_memory_updates(table, updates)
        else:
            with _write_table_components_by_region_operation(
                sdata.path,
                table_name=table_name,
                components=components,
                obs_identity=obs_identity,
                fill_values=fill_values,
                chunk_size=chunk_size,
                overwrite=overwrite,
            ) as published:
                # Serialization is complete (same pattern as in _update_table_components()).
                # Reopen requested components from permanent paths and attach them inside
                # _write_table_operation()'s rollback window, kept open by the regional operation.
                # Its nested _publish_staged_paths() context retains disk backups until
                # this with-body and metadata consolidation succeed. On failure, that
                # context restores disk components; the except block below restores
                # the previous in-memory slot objects.
                reopened = {path: _read_anndata_element(published, path, mode="lazy") for path in paths}
                updates = _prepare_memory_updates(table, reopened, (), new_raw_var=None)
                _install_memory_updates(table, updates)
    except BaseException:
        for slot, value in previous.items():
            setattr(table, f"_{slot}", value)
        raise
    return sdata
