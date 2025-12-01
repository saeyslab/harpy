from anndata import AnnData
from spatialdata.models import TableModel


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
    # raise a valueerror if labels layer already exists in table
    if region in adata.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY]:
        raise ValueError(
            f"'{region}' already exists as a region in the AnnData object you wish to append to. "
            "Please choose a different labels layer, choose a different 'output_layer' or set append to False and overwrite to True to overwrite the existing table."
        )
