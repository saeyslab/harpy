from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from spatialdata import SpatialData, read_zarr
from spatialdata.models import TableModel
from spatialdata_io._constants._constants import VisiumKeys
from spatialdata_io.readers.visium import visium as sdata_visium

import harpy as hp
from harpy.utils._keys import _INSTANCE_KEY, _REGION_KEY


def visium(
    path: str | Path,
    dataset_id: str | None = None,
    counts_file: str = VisiumKeys.FILTERED_COUNTS_FILE,
    fullres_image_file: str | Path | None = None,
    output: str | Path | None = None,
) -> SpatialData:
    """
    Read *10x Genomics* Visium formatted dataset.

    Wrapper around `spatialdata.io.readers.visium.visium`, but with the resulting table annotated by a labels layer.

    .. see also::

        - `Space Ranger output <https://support.10xgenomics.com/spatial-gene-expression/software/pipelines/latest/output/overview>`_.

    Parameters
    ----------
    path
        Path to directory containing the *10x Genomics* Visium output.
    dataset_id
        Unique identifier of the dataset. If `None`, it tries to infer it from the file name of the feature slice file.
    counts_file
        Name of the counts file, defaults to `'filtered_feature_bc_matrix.h5'`; a common alternative is
        `'raw_feature_bc_matrix.h5'`.
    fullres_image_file
        Path to the full-resolution image.
    output
        The path where the resulting `SpatialData` object will be backed. If None, it will not be backed to a zarr store.
    """
    sdata = sdata_visium(
        path=path,
        dataset_id=dataset_id,
        counts_file=counts_file,
        fullres_image_file=fullres_image_file,
    )

    for table_layer in [*sdata.tables]:
        adata = sdata[table_layer]
        adata.var_names_make_unique()
        adata.X = adata.X.tocsc()

        _old_instance_key = sdata[table_layer].uns[TableModel.ATTRS_KEY][TableModel.INSTANCE_KEY]
        _old_region_key = sdata[table_layer].uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY]
        adata.obs[_REGION_KEY] = pd.Categorical(adata.obs[_old_region_key].astype(str) + "_labels")
        if adata.obs[_old_instance_key].astype(int).min() == 0:  # Make sure index starts from 1
            adata.obs[_INSTANCE_KEY] = adata.obs[_old_instance_key].astype(int) + 1
        else:
            adata.obs[_INSTANCE_KEY] = adata.obs[_old_instance_key].astype(int)

        adata.uns.pop(TableModel.ATTRS_KEY)
        adata = TableModel.parse(
            adata,
            region_key=_REGION_KEY,
            region=adata.obs[_REGION_KEY].cat.categories.to_list(),
            instance_key=_INSTANCE_KEY,
        )

        assert len(sdata.shapes[dataset_id]) == len(adata), (
            f"Shapes layer '{dataset_id}' and corresponding table '{table_layer}' should have same length."
        )

        sdata.shapes[dataset_id].index = (
            adata.obs.set_index(_old_instance_key).loc[sdata.shapes[dataset_id].index, _INSTANCE_KEY].values
        )
        sdata.shapes[dataset_id].index.name = _INSTANCE_KEY

        if _old_region_key in adata.obs.columns:
            adata.obs.drop(columns=_old_region_key, inplace=True)

        if _old_instance_key in adata.obs.columns:
            adata.obs.drop(columns=_old_instance_key, inplace=True)

        # Convert Points to Polygons
        if "radius" not in sdata.shapes[dataset_id].columns:
            raise ValueError("Shapes layer is missing 'radius' column required for polygon buffering.")

        radius = sdata.shapes[dataset_id]["radius"].mean()
        polygons = sdata.shapes[dataset_id].buffer(radius, cap_style="round")
        sdata = hp.sh.add_shapes_layer(
            sdata, gpd.GeoDataFrame(geometry=polygons), output_layer=dataset_id, overwrite=True
        )

        # Create labels layer
        sdata = hp.im.rasterize(
            sdata,
            shapes_layer=dataset_id,
            output_layer=f"{dataset_id}_labels",
            chunks=5000,
            overwrite=True,
        )

        del sdata[table_layer]

        sdata[table_layer] = adata

    if output is not None:
        sdata.write(output)
        sdata = read_zarr(sdata.path)

    return sdata
