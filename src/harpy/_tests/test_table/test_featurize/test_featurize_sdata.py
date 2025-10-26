import dask.array as da
import numpy as np
import pytest
from spatialdata import SpatialData
from spatialdata.models import TableModel

from harpy.image import add_image_layer, add_labels_layer
from harpy.table._table import add_table_layer
from harpy.table.featurization.featurize import featurize
from harpy.utils._keys import _INSTANCE_KEY, _REGION_KEY
from harpy.utils.utils import _dummy_embedding


@pytest.mark.parametrize(
    "table_layer",
    ["table_transcriptomics", None],
)
def test_featurize_sdata(sdata_transcripts_no_backed: SpatialData, table_layer):
    sdata = sdata_transcripts_no_backed

    chunksize_spatial = 2048
    embedding_dim = 384
    embedding_obsm_key = "embedding_dummy"

    image = sdata["raw_image"].data

    sdata = add_image_layer(
        sdata,
        arr=da.concatenate([image] * 6, axis=0).rechunk((3, chunksize_spatial, chunksize_spatial)),
        output_layer="raw_image",
        overwrite=True,
    )

    sdata = add_labels_layer(
        sdata,
        arr=sdata["segmentation_mask"].data.rechunk(chunksize_spatial),
        output_layer="segmentation_mask",
        overwrite=True,
    )

    sdata = featurize(
        sdata,
        img_layer="raw_image",
        labels_layer="segmentation_mask",
        table_layer=table_layer,
        output_layer="table_transcriptomics_embedding",
        depth=100,
        diameter=75,
        embedding_dimension=embedding_dim,
        model=_dummy_embedding,
        batch_size=250,
        embedding_obsm_key=embedding_obsm_key,
        overwrite=True,
    )

    assert embedding_obsm_key in sdata["table_transcriptomics_embedding"].obsm
    if table_layer is None:
        assert sdata["table_transcriptomics_embedding"].obsm[embedding_obsm_key].shape == (
            da.unique(sdata["segmentation_mask"].data).compute().size - 1,
            embedding_dim,
        )
    else:
        assert sdata["table_transcriptomics_embedding"].obsm[embedding_obsm_key].shape == (
            sdata["table_transcriptomics"].shape[0],
            embedding_dim,
        )


def test_featurize_sdata_blobs(sdata: SpatialData):
    img_layer = "blobs_image"
    labels_layer = "blobs_labels"
    table_layer = "table"
    output_layer = "table_embedding"

    embedding_dim = 20
    embedding_obsm_key = "embedding_sdata"

    sdata = add_image_layer(
        sdata,
        arr=sdata[img_layer].data.astype(np.float32).rechunk(512),
        output_layer=img_layer,
        overwrite=True,
    )

    sdata = add_labels_layer(
        sdata,
        arr=sdata[labels_layer].data.rechunk(512),
        output_layer=labels_layer,
        overwrite=True,
    )
    sdata[table_layer].uns.pop(TableModel.ATTRS_KEY)
    sdata[table_layer]
    sdata[table_layer].obs.rename(columns={"instance_id": _INSTANCE_KEY, "region": _REGION_KEY}, inplace=True)

    sdata = add_table_layer(
        sdata,
        adata=sdata[table_layer],
        output_layer=table_layer,
        region=[labels_layer],
        overwrite=True,
    )

    sdata = featurize(
        sdata,
        img_layer=img_layer,
        labels_layer=labels_layer,
        table_layer=table_layer,
        output_layer=output_layer,
        depth=250,
        diameter=1000,
        embedding_dimension=embedding_dim,
        model=_dummy_embedding,
        batch_size=250,
        embedding_obsm_key=embedding_obsm_key,
        overwrite=True,
    )

    assert embedding_obsm_key in sdata[output_layer].obsm
    assert sdata[output_layer].obsm[embedding_obsm_key].shape == (sdata[table_layer].shape[0], embedding_dim)
