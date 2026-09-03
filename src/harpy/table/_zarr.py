"""Low-level Zarr helpers shared by Harpy table writers."""

from collections.abc import Sequence

import zarr

# Harpy owns these literals as the compatibility boundary for tables assembled
# with the low-level AnnData writers, without depending on private SpatialData
# writer APIs.
_SPATIALDATA_TABLE_ENCODING_TYPE = "ngff:regions_table"
_SPATIALDATA_TABLE_FORMAT_VERSION = "0.2"


def _write_spatialdata_table_attrs(
    group: zarr.Group,
    *,
    regions: Sequence[str],
    region_key: str,
    instance_key: str,
) -> None:
    """Write SpatialData's disk-level regions-table contract.

    ``TableModel.parse()`` records the semantic table relationship in
    ``adata.uns["spatialdata_attrs"]``. A SpatialData Zarr store also requires
    attributes on the AnnData group itself so that its reader recognizes the
    group as a regions table. Harpy writes the AnnData components directly in
    its out-of-core table path, so this helper adds that second, on-disk
    representation without calling SpatialData's private writer APIs.

    Parameters
    ----------
    group
        AnnData Zarr group that will become a SpatialData table element.
    regions
        Labels elements annotated by the table.
    region_key
        Column in ``adata.obs`` that identifies the labels element.
    instance_key
        Column in ``adata.obs`` that identifies an instance within that labels
        element.
    """
    group.attrs["spatialdata-encoding-type"] = _SPATIALDATA_TABLE_ENCODING_TYPE
    group.attrs["region"] = list(regions)
    group.attrs["region_key"] = region_key
    group.attrs["instance_key"] = instance_key
    group.attrs["version"] = _SPATIALDATA_TABLE_FORMAT_VERSION
