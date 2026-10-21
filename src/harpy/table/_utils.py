from collections.abc import Mapping, Sequence

import numpy as np
from anndata import AnnData
from spatialdata.models import TableModel

from harpy.utils._names import _names, _selected_names


def _spatialdata_annotation(adata: object, *, function: str) -> Mapping:
    """Return the table's SpatialData annotation, raising unless ``adata`` is an AnnData that has one.

    ``function`` names the public function in the error raised for another type
    than an AnnData, such as a SpatialData object.
    """
    if not isinstance(adata, AnnData):
        raise TypeError(
            f"{function} takes the table, an AnnData such as sdata.tables[name], found {type(adata).__name__}."
        )
    attrs = adata.uns.get(TableModel.ATTRS_KEY)
    if not isinstance(attrs, Mapping):
        raise ValueError(f"The table has no SpatialData annotation (uns[{TableModel.ATTRS_KEY!r}]).")
    return attrs


def _region_rows(adata: AnnData, region: str | Sequence[str] | None) -> np.ndarray | None:
    """Return a boolean mask of the table's rows of the named regions, or None, for every row, for ``region=None``.

    The names are those of the table's SpatialData annotation,
    ``uns["spatialdata_attrs"]["region"]``, whose rows ``obs[region_key]``
    identifies, whatever the type of the annotated elements.
    """
    if region is None:
        return None
    attrs = adata.uns.get(TableModel.ATTRS_KEY)
    if not isinstance(attrs, Mapping):
        raise ValueError(
            f"region selects the regions of the table's SpatialData annotation, uns[{TableModel.ATTRS_KEY!r}], "
            "which this table has not; pass region=None to use every row."
        )
    selected = _selected_names(
        region, recorded=_names(attrs[TableModel.REGION_KEY]), parameter="region", kind="regions"
    )
    return adata.obs[attrs[TableModel.REGION_KEY_KEY]].isin(selected).to_numpy()


def _sanity_check_append_region(adata: AnnData, region_key: str, instance_key: str, region: str):
    # raise a ValueError if region key or instance key is different than the one in adata
    _region_key_current = adata.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY]
    if region_key != _region_key_current:
        raise ValueError(
            f"Provided region key '{region_key}' is different than the region key of the AnnData object you wish to append to ('{_region_key_current}'). "
            f"This is not allowed. Please set 'region_key' to '{_region_key_current}'."
        )
    _instance_key_current = adata.uns[TableModel.ATTRS_KEY][TableModel.INSTANCE_KEY]
    if instance_key != _instance_key_current:
        raise ValueError(
            f"Provided instance key '{instance_key}' is different than the instance key of the AnnData object you wish to append to ('{_instance_key_current}'). "
            f"This is not allowed. Please set 'instance_key' to '{_instance_key_current}'."
        )
    # raise a valueerror if labels element already exists in table
    if region in adata.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY]:
        raise ValueError(
            f"'{region}' already exists as a region in the AnnData object you wish to append to. "
            "Please choose a different labels element, choose a different 'output_table_name' or set append to False and overwrite to True to overwrite the existing table."
        )
