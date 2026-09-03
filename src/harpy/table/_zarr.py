"""Low-level Zarr helpers shared by Harpy table readers and writers."""

from collections.abc import Mapping, Sequence

import numpy as np
import zarr
from anndata import AnnData
from anndata.io import read_elem, sparse_dataset
from spatialdata.models import TableModel

# Harpy owns these literals as the compatibility boundary for tables assembled
# with the low-level AnnData writers, without depending on private SpatialData
# writer APIs.
_SPATIALDATA_TABLE_ENCODING_TYPE = "ngff:regions_table"
_SPATIALDATA_TABLE_FORMAT_VERSION = "0.2"


def _read_backed_table(group: zarr.Group) -> AnnData:
    """Read a Harpy-written AnnData table from its published Zarr group.

    ``obs``, ``var`` and ``uns`` are decoded through AnnData's component
    reader because AnnData represents them as in-memory pandas and Python
    objects. Dense Zarr arrays and sparse CSR/CSC datasets in ``X`` and
    ``obsm`` remain storage-backed. This gives the returned SpatialData object
    lightweight matrix handles while making the serialized table the source of
    truth for every component.

    The direct reader intentionally covers the components written by Harpy's
    out-of-core table path: ``X``, ``obs``, ``var``, ``uns`` and ``obsm``. It
    is not a replacement for ``anndata.read_zarr()`` for arbitrary AnnData
    stores.

    Parameters
    ----------
    group
        Published AnnData Zarr group at
        ``sdata.zarr/tables/<table_name>``. The group must already be at this
        persistent element path because the returned ``zarr.Array`` and
        ``CSRDataset`` objects retain its location for subsequent reads.

    Returns
    -------
    AnnData
        Table with storage-backed array components.
    """
    uns = read_elem(group["uns"])
    if not isinstance(uns, Mapping):
        raise ValueError("AnnData Zarr component 'uns' must decode to a mapping.")
    uns = dict(uns)
    spatialdata_attrs = uns.get(TableModel.ATTRS_KEY)
    if isinstance(spatialdata_attrs, Mapping):
        spatialdata_attrs = dict(spatialdata_attrs)
        region = spatialdata_attrs.get(TableModel.REGION_KEY)
        if isinstance(region, np.ndarray):
            spatialdata_attrs[TableModel.REGION_KEY] = region.tolist()
        uns[TableModel.ATTRS_KEY] = spatialdata_attrs

    obsm_group = group["obsm"]
    obsm = {key: _read_backed_array(obsm_group[key]) for key in obsm_group.keys()}
    return AnnData(
        X=_read_backed_array(group["X"]),
        obs=read_elem(group["obs"]),
        var=read_elem(group["var"]),
        uns=uns,
        obsm=obsm,
    )


def _read_backed_array(element: zarr.Array | zarr.Group) -> object:
    """Preserve storage backing for one dense or sparse AnnData array."""
    if isinstance(element, zarr.Array):
        return element
    if element.attrs.get("encoding-type") in {"csr_matrix", "csc_matrix"}:
        return sparse_dataset(element)
    return read_elem(element)


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
