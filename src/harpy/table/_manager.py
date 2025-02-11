import spatialdata
from anndata import AnnData
from spatialdata import SpatialData, read_zarr
from spatialdata.models import TableModel

from harpy.utils._io import _incremental_io_on_disk
from harpy.utils._keys import _INSTANCE_KEY, _REGION_KEY
from harpy.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class TableLayerManager:
    def add_table(
        self,
        sdata: SpatialData,
        adata: AnnData,
        output_layer: str,
        region: list[str] | None,  # list of labels_layers
        overwrite: bool = False,
    ) -> SpatialData:
        if region is not None:
            assert _REGION_KEY in adata.obs.columns, (
                f"Provided 'AnnData' object should contain a column '{_REGION_KEY}' in 'adata.obs'. Linking the observations to a labels layer in 'sdata'."
            )
            assert _INSTANCE_KEY in adata.obs.columns, (
                f"Provided 'AnnData' object should contain a column '{_INSTANCE_KEY}' in 'adata.obs'. Linking the observations to a labels layer in 'sdata'."
            )

            # need to remove spatialdata_attrs, otherwise parsing gives error (TableModel.parse will add spatialdata_attrs back)
            if TableModel.ATTRS_KEY in adata.uns.keys():
                adata.uns.pop(TableModel.ATTRS_KEY)

            adata = spatialdata.models.TableModel.parse(
                adata,
                region_key=_REGION_KEY,
                region=region,
                instance_key=_INSTANCE_KEY,
            )
        else:
            if TableModel.ATTRS_KEY in adata.uns.keys():
                adata.uns.pop(TableModel.ATTRS_KEY)

            adata = spatialdata.models.TableModel.parse(
                adata,
            )

        if output_layer in [*sdata.tables]:
            if sdata.is_backed():
                if overwrite:
                    sdata = _incremental_io_on_disk(
                        sdata, output_layer=output_layer, element=adata, element_type="tables"
                    )
                else:
                    raise ValueError(
                        f"Attempting to overwrite 'sdata.tables[\"{output_layer}\"]', but overwrite is set to False. Set overwrite to True to overwrite the .zarr store."
                    )
            else:
                sdata[output_layer] = adata
        else:
            sdata[output_layer] = adata
            if sdata.is_backed():
                sdata.write_element(output_layer)
                sdata_temp = read_zarr(sdata.path, selection=["tables"])
                del sdata[output_layer]
                sdata[output_layer] = sdata_temp[output_layer]
                del sdata_temp

        return sdata
