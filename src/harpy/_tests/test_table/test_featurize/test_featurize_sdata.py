import dask
import dask.array as da
import numpy as np
import pytest
from spatialdata import SpatialData
from spatialdata.models import TableModel

from harpy.image import add_image_layer, add_labels_layer
from harpy.table._table import add_table_layer
from harpy.table.featurization._featurize import extract_instances, featurize
from harpy.utils._keys import _INSTANCE_KEY, _REGION_KEY
from harpy.utils.utils import _dummy_embedding


def test_extract_instances_sdata(sdata_transcripts_no_backed: SpatialData):
    sdata = sdata_transcripts_no_backed

    chunksize_spatial = 2048
    img_layer = "raw_image"
    labels_layer = "segmentation_mask"

    image = sdata[img_layer].data

    sdata = add_image_layer(
        sdata,
        arr=da.concatenate([image] * 6, axis=0).rechunk((3, chunksize_spatial, chunksize_spatial)),
        output_layer=img_layer,
        overwrite=True,
    )

    sdata = add_labels_layer(
        sdata,
        arr=sdata[labels_layer].data.rechunk(chunksize_spatial),
        output_layer=labels_layer,
        overwrite=True,
    )

    instances_ids, out = extract_instances(
        sdata,
        img_layer=img_layer,
        labels_layer=labels_layer,
        depth=100,
        diameter=75,
        zarr_output_path=None,
        extract_mask=True,  # extract the mask, so we can perform sanity checks
        batch_size=250,
    )

    assert out[0].shape == (657, 1, 1, 75, 75)
    assert out[1].shape == (657, 6, 1, 75, 75)

    mask_instances, image_instances = dask.compute(*out)

    assert image_instances.shape == (657, 6, 1, 75, 75)
    assert mask_instances.shape == (657, 1, 1, 75, 75)

    # check that mask of each instance contains the index corresponding to instances_ids
    for _index, _item in zip(instances_ids, mask_instances, strict=True):
        _item_labels = np.unique(_item[0])
        _item_labels = _item_labels[_item_labels != 0]
        assert len(_item_labels) == 1
        assert _index == _item_labels[0]

    # check that all labels are extracted
    index = da.unique(mask_instances[:, 0, ...]).compute()
    index = index[index != 0]

    assert np.array_equal(index, np.sort(instances_ids))


@pytest.mark.parametrize(
    "table_layer",
    ["table_transcriptomics", None],
)
def test_featurize_sdata(sdata_transcripts_no_backed: SpatialData, table_layer):
    sdata = sdata_transcripts_no_backed

    chunksize_spatial = 2048
    embedding_dim = 384
    embedding_obsm_key = "embedding_dummy"

    img_layer = "raw_image"
    labels_layer = "segmentation_mask"
    output_layer = "table_transcriptomics_embedding"

    image = sdata[img_layer].data

    sdata = add_image_layer(
        sdata,
        arr=da.concatenate([image] * 6, axis=0).rechunk((3, chunksize_spatial, chunksize_spatial)),
        output_layer=img_layer,
        overwrite=True,
    )

    sdata = add_labels_layer(
        sdata,
        arr=sdata[labels_layer].data.rechunk(chunksize_spatial),
        output_layer=labels_layer,
        overwrite=True,
    )

    sdata = featurize(
        sdata,
        img_layer=img_layer,
        labels_layer=labels_layer,
        table_layer=table_layer,
        output_layer=output_layer,
        depth=100,
        diameter=75,
        embedding_dimension=embedding_dim,
        model=_dummy_embedding,
        batch_size=250,
        embedding_obsm_key=embedding_obsm_key,
        overwrite=True,
    )

    assert embedding_obsm_key in sdata[output_layer].obsm
    if table_layer is None:
        assert sdata[output_layer].obsm[embedding_obsm_key].shape == (
            da.unique(sdata[labels_layer].data).compute().size - 1,
            embedding_dim,
        )
    else:
        assert sdata[output_layer].obsm[embedding_obsm_key].shape == (
            sdata[table_layer].shape[0],
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
        instance_key=_INSTANCE_KEY,
        region_key=_REGION_KEY,
        overwrite=True,
    )

    sdata = featurize(
        sdata,
        img_layer=img_layer,
        labels_layer=labels_layer,
        table_layer=table_layer,
        output_layer=output_layer,
        diameter=1000,
        embedding_dimension=embedding_dim,
        model=_dummy_embedding,
        batch_size=250,
        embedding_obsm_key=embedding_obsm_key,
        overwrite=True,
    )

    assert embedding_obsm_key in sdata[output_layer].obsm
    assert sdata[output_layer].obsm[embedding_obsm_key].shape == (sdata[table_layer].shape[0], embedding_dim)
