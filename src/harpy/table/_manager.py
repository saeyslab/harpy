import numpy as np
import spatialdata
from anndata import AnnData
from loguru import logger as log
from spatialdata import SpatialData, read_zarr
from spatialdata.models import TableModel

from harpy.utils._io import _incremental_io_on_disk, _write_element_with_cleanup
from harpy.utils._keys import _INSTANCE_KEY, _REGION_KEY


class TableLayerManager:
    def add_table(
        self,
        sdata: SpatialData,
        adata: AnnData,
        output_layer: str,
        region: list[str] | None,  # list of labels_layers
        instance_key: str = _INSTANCE_KEY,  # ignored if region is None
        region_key: str = _REGION_KEY,  # ignored if region is None
        overwrite: bool = False,
    ) -> SpatialData:
        if region is not None:
            # do some sanity checks on the provided instance and region keys
            # e.g. to catch case that adata object would already be annotated with another region or instance key
            if TableModel.ATTRS_KEY in adata.uns.keys():
                if TableModel.REGION_KEY_KEY in adata.uns[TableModel.ATTRS_KEY]:
                    if region_key != adata.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY]:
                        raise ValueError(
                            f"The provided region key '{region_key}' is not equal to the region key in the AnnData object ({adata.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY]}). This is not allowed."
                        )
                if TableModel.INSTANCE_KEY in adata.uns[TableModel.ATTRS_KEY]:
                    if instance_key != adata.uns[TableModel.ATTRS_KEY][TableModel.INSTANCE_KEY]:
                        raise ValueError(
                            f"The provided instance key '{instance_key}' is not equal to the instance key in the AnnData object ({adata.uns[TableModel.ATTRS_KEY][TableModel.INSTANCE_KEY]}). This is not allowed."
                        )
            if region_key not in adata.obs.columns:
                raise ValueError(
                    f"Provided 'AnnData' object should contain a column '{region_key}' in 'adata.obs'. Linking the observations to a region (e.g. a labels layer) in 'sdata'."
                )
            if instance_key not in adata.obs.columns:
                raise ValueError(
                    f"Provided 'AnnData' object should contain a column '{instance_key}' in 'adata.obs'. Linking the observations to a region (e.g. a labels layer) in 'sdata'."
                )
            # need to remove spatialdata_attrs, otherwise parsing gives error (TableModel.parse will add spatialdata_attrs back)
            if TableModel.ATTRS_KEY in adata.uns.keys():
                adata.uns.pop(TableModel.ATTRS_KEY)

            adata = spatialdata.models.TableModel.parse(
                adata,
                region_key=region_key,
                region=region,
                instance_key=instance_key,
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
                _write_element_with_cleanup(sdata, output_layer)
                del sdata[output_layer]
                sdata_temp = read_zarr(sdata.path, selection=["tables"])
                sdata[output_layer] = sdata_temp[output_layer]
                del sdata_temp

        # If reading from the zarr store, colors are saved as Numpy StringDtype color arrays.
        # Leading to .copy() failing. Therefore, as a temporary workaround we cast to a safer dtype.
        _cast_colors_sdata(sdata)

        return sdata


def _cast_colors_sdata(sdata, target_dtype="U7"):
    """Normalize NumPy StringDType color arrays in all sdata tables to copy-safe dtype."""
    target_dtype = np.dtype(target_dtype)
    target_len = target_dtype.itemsize // np.dtype("U1").itemsize if target_dtype.kind == "U" else None

    for _, adata in sdata.tables.items():
        for key, value in list(adata.uns.items()):
            if key.endswith("_colors") and isinstance(value, np.ndarray):
                dt = value.dtype
                if "StringDType" in str(dt) or getattr(dt, "kind", None) == "T":
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
