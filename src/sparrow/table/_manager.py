import spatialdata
from anndata import AnnData
from spatialdata import SpatialData, read_zarr

from sparrow.utils._keys import _INSTANCE_KEY, _REGION_KEY
from sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class TableLayerManager:
    def add_table(
        self,
        sdata: SpatialData,
        adata: AnnData,
        output_layer: str,
        region: list[str],  # list of labels_layers , TODO, check what to do with shapes layers
        overwrite: bool = False,
    ) -> SpatialData:
        assert (
            _REGION_KEY in adata.obs.columns
        ), f"Provided 'AnnData' object should contain a column '{_REGION_KEY}' in 'adata.obs'. Linking the observations to a labels layer in 'sdata'."
        assert (
            _INSTANCE_KEY in adata.obs.columns
        ), f"Provided 'AnnData' object should contain a column '{_INSTANCE_KEY}' in 'adata.obs'. Linking the observations to a labels layer in 'sdata'."

        if output_layer in [*sdata.tables]:
            if overwrite:
                log.warning(f"Table with name '{output_layer}' already exists. Overwriting...")
                element_type = sdata._element_type_from_element_name(output_layer)
                del getattr(sdata, element_type)[output_layer]
                if sdata.is_backed():
                    sdata.delete_element_from_disk(output_layer)
            else:
                if sdata.is_backed():
                    raise ValueError(
                        f"Attempting to overwrite sdata.tables[{output_layer}], but overwrite is set to False. Set overwrite to True to overwrite the .zarr store."
                    )

        # need to remove spatialdata_attrs, otherwise parsing gives error (TableModel.parse will add spatialdata_attrs back)
        if "spatialdata_attrs" in adata.uns.keys():
            adata.uns.pop("spatialdata_attrs")

        sdata.tables[output_layer] = spatialdata.models.TableModel.parse(
            adata,
            region_key=_REGION_KEY,
            region=region,
            instance_key=_INSTANCE_KEY,
        )

        if sdata.is_backed():
            sdata.write_element(output_layer)
            sdata = read_zarr(sdata.path)

        return sdata
