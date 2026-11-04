from __future__ import annotations

from loguru import logger as log
from spatialdata import SpatialData

from harpy.table._validation import _validate_table_without_canonical
from harpy.table.canonical_centers import CANONICAL_OBSM_KEY, SPATIAL_COORDINATES_KEY
from harpy.table.canonical_centers._calculation import (
    _assemble_canonical_components,
    _calculate_canonical_centers,
)
from harpy.table.canonical_centers._schema import _build_canonical_region_binding
from harpy.table.io._components import add_table_components

_CANONICAL_MATRIX_PATH = ("obsm", CANONICAL_OBSM_KEY)
_CANONICAL_METADATA_PATH = ("uns", SPATIAL_COORDINATES_KEY, CANONICAL_OBSM_KEY)


def add_canonical_centers(
    sdata: SpatialData,
    *,
    table_name: str,
    overwrite: bool = False,
) -> SpatialData:
    """Add label-derived canonical centers to an existing SpatialData table.

    Every region the table annotates must be a labels element. Each row gets
    the center of mass of its instance in the intrinsic ``scale0`` frame of its
    labels element, stored as dense ``float64`` ``(z, y, x)`` coordinates in
    ``adata.obsm["spatial_canonical"]``; a 2D labels element gives ``z=0``.
    Their metadata is stored in
    ``adata.uns["spatial_coordinates"]["spatial_canonical"]``.

    Both are written with :func:`harpy.table.io.add_table_components`, whose
    guarantees apply: when ``sdata`` is backed, they are also written to its
    store and reattached lazily, and a handled failure restores the previous
    components. Other table components, including other entries under
    ``adata.uns["spatial_coordinates"]``, remain unchanged.

    Before calculation, the table is checked as :func:`harpy.tb.validate_table`
    checks it, with the default feature-matrix registry, except for existing
    canonical centers: they are replaced, so they may be stale or invalid.

    Parameters
    ----------
    sdata
        SpatialData object containing the table and its labels elements. When
        backed, the table must exist in the store, with the same observations
        in the same order.
    table_name
        Existing regions table to update.
    overwrite
        Allow replacing canonical centers that exist in the store. Ignored for
        unbacked SpatialData, as in :func:`harpy.table.io.add_table_components`.

    Returns
    -------
    The same SpatialData object with its table updated.

    Raises
    ------
    ValueError
        If the table, its annotation or its labels elements are invalid.
    FileExistsError
        If ``overwrite`` is ``False`` and the store has canonical centers.
    """
    if not isinstance(overwrite, bool):
        raise ValueError(f"Parameter 'overwrite' must be a bool, found {type(overwrite).__name__}.")
    annotation = _validate_table_without_canonical(sdata, table_name)
    if annotation is None:
        raise ValueError(f"Table element {table_name!r} is not a SpatialData regions table.")
    region_key, instance_key, regions = annotation
    not_labels = sorted(name for name in regions if name not in sdata.labels)
    if not_labels:
        raise ValueError(f"Table {table_name!r} references non-label spatial elements: {not_labels!r}.")

    table = sdata.tables[table_name]
    log.info(f"Calculating canonical label centers for table {table_name!r} from {len(regions)} labels region(s).")
    centers, metadata = _assemble_canonical_components(
        table.n_obs,
        [
            _calculate_canonical_centers(sdata, _build_canonical_region_binding(table, labels_name=labels_name))
            for labels_name in regions
        ],
    )
    log.info(f"Finished calculating canonical label centers for table {table_name!r}.")

    return add_table_components(
        sdata,
        table_name=table_name,
        components={_CANONICAL_MATRIX_PATH: centers, _CANONICAL_METADATA_PATH: metadata},
        obs_identity=table.obs[[region_key, instance_key]],
        overwrite=overwrite,
    )
