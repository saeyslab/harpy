from pathlib import Path

import anndata as ad
import dask
import dask.array as da
import numpy as np
import pandas as pd
import pytest
from spatialdata import SpatialData, read_zarr
from spatialdata.models import Image2DModel, Labels2DModel, TableModel

from harpy.image import add_image, add_labels
from harpy.table.featurization._featurize import extract_instances, featurize
from harpy.table.io._add_table import add_table
from harpy.utils._keys import _INSTANCE_KEY, _REGION_KEY
from harpy.utils.utils import _dummy_embedding, _to_numpy


def test_extract_instances_sdata(sdata_transcripts_no_backed: SpatialData):
    sdata = sdata_transcripts_no_backed

    chunksize_spatial = 2048
    image_name = "raw_image"
    labels_name = "segmentation_mask"

    image = sdata[image_name].data

    sdata = add_image(
        sdata,
        arr=da.concatenate([image] * 6, axis=0).rechunk((3, chunksize_spatial, chunksize_spatial)),
        output_image_name=image_name,
        overwrite=True,
    )

    sdata = add_labels(
        sdata,
        arr=sdata[labels_name].data.rechunk(chunksize_spatial),
        output_labels_name=labels_name,
        overwrite=True,
    )

    instances_ids, out = extract_instances(
        sdata,
        image_name=image_name,
        labels_name=labels_name,
        depth=100,
        diameter=75,
        zarr_output_path=None,
        extract_mask=True,  # extract the mask, so we can perform sanity checks
        batch_size=250,
    )

    assert out[0].shape == (657, 1, 1, 75, 75)
    assert out[1].shape == (657, 6, 1, 75, 75)

    mask_instances, image_instances = dask.compute(*out)
    mask_instances = _to_numpy(mask_instances)
    image_instances = _to_numpy(image_instances)

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
    "table_name",
    ["table_transcriptomics", None],
)
def test_featurize_sdata(sdata_transcripts_no_backed: SpatialData, table_name):
    sdata = sdata_transcripts_no_backed

    chunksize_spatial = 2048
    embedding_dim = 384
    embedding_obsm_key = "embedding_dummy"

    image_name = "raw_image"
    labels_name = "segmentation_mask"
    output_table_name = "table_transcriptomics_embedding"

    image = sdata[image_name].data

    sdata = add_image(
        sdata,
        arr=da.concatenate([image] * 6, axis=0).rechunk((3, chunksize_spatial, chunksize_spatial)),
        output_image_name=image_name,
        overwrite=True,
    )

    sdata = add_labels(
        sdata,
        arr=sdata[labels_name].data.rechunk(chunksize_spatial),
        output_labels_name=labels_name,
        overwrite=True,
    )

    sdata = featurize(
        sdata,
        image_name=image_name,
        labels_name=labels_name,
        table_name=table_name,
        output_table_name=output_table_name,
        depth=100,
        diameter=75,
        embedding_dimension=embedding_dim,
        model=_dummy_embedding,
        batch_size=250,
        embedding_obsm_key=embedding_obsm_key,
        overwrite=True,
    )

    assert embedding_obsm_key in sdata[output_table_name].obsm
    if table_name is None:
        assert sdata[output_table_name].obsm[embedding_obsm_key].shape == (
            da.unique(sdata[labels_name].data).compute().size - 1,
            embedding_dim,
        )
    else:
        assert sdata[output_table_name].obsm[embedding_obsm_key].shape == (
            sdata[table_name].shape[0],
            embedding_dim,
        )


@pytest.mark.parametrize(
    ("store_intermediate", "backed"),
    [
        (False, False),
        (True, True),
        (True, False),
    ],
)
def test_featurize_sdata_blobs(sdata: SpatialData, tmp_path: Path, store_intermediate: bool, backed: bool):
    image_name = "blobs_image"
    labels_name = "blobs_labels"
    table_name = "table"
    output_table_name = "table_embedding"

    embedding_dim = 20
    embedding_obsm_key = "embedding_sdata"

    if backed:
        backed_path = tmp_path / "sdata_blobs.zarr"
        sdata.write(backed_path)
        sdata = read_zarr(backed_path)

    sdata = add_image(
        sdata,
        arr=sdata[image_name].data.astype(np.float32).rechunk(512),
        output_image_name=image_name,
        overwrite=True,
    )

    sdata = add_labels(
        sdata,
        arr=sdata[labels_name].data.rechunk(512),
        output_labels_name=labels_name,
        overwrite=True,
    )
    sdata[table_name].uns.pop(TableModel.ATTRS_KEY)
    sdata[table_name]
    sdata[table_name].obs.rename(columns={"instance_id": _INSTANCE_KEY, "region": _REGION_KEY}, inplace=True)

    sdata = add_table(
        sdata,
        adata=sdata[table_name],
        output_table_name=table_name,
        region=[labels_name],
        instance_key=_INSTANCE_KEY,
        region_key=_REGION_KEY,
        overwrite=True,
    )

    if store_intermediate and not backed:
        with pytest.raises(ValueError, match="store_intermediate=True' is only supported for backed SpatialData"):
            _ = featurize(
                sdata,
                image_name=image_name,
                labels_name=labels_name,
                table_name=table_name,
                output_table_name=output_table_name,
                diameter=1000,
                embedding_dimension=embedding_dim,
                model=_dummy_embedding,
                batch_size=250,
                embedding_obsm_key=embedding_obsm_key,
                store_intermediate=store_intermediate,
                overwrite=True,
            )
        return

    sdata = featurize(
        sdata,
        image_name=image_name,
        labels_name=labels_name,
        table_name=table_name,
        output_table_name=output_table_name,
        diameter=1000,
        embedding_dimension=embedding_dim,
        model=_dummy_embedding,
        batch_size=250,
        embedding_obsm_key=embedding_obsm_key,
        store_intermediate=store_intermediate,
        overwrite=True,
    )

    assert embedding_obsm_key in sdata[output_table_name].obsm
    assert sdata[output_table_name].obsm[embedding_obsm_key].shape == (sdata[table_name].shape[0], embedding_dim)


def _mean_intensity_model(batch: np.ndarray, embedding_dimension: int = 1) -> np.ndarray:
    means = batch.mean(axis=(1, 2, 3, 4), keepdims=False)[:, None].astype(np.float32)
    return np.repeat(means, embedding_dimension, axis=1)


@pytest.mark.parametrize(
    ("obs_rows", "labels_order", "image_order"),
    [
        # labels_name passed in reverse order relative to adata.obs
        ([("lbl0", 1), ("lbl1", 1)], ["lbl1", "lbl0"], ["img1", "img0"]),
        # regions interleaved in adata.obs
        ([("lbl0", 1), ("lbl1", 1), ("lbl0", 2)], ["lbl0", "lbl1"], ["img0", "img1"]),
    ],
)
def test_featurize_multi_region_aligns_embeddings_by_region_and_instance(obs_rows, labels_order, image_order):
    img0 = np.full((1, 16, 16), 10.0, dtype=np.float32)
    img1 = np.full((1, 16, 16), 99.0, dtype=np.float32)

    lbl0 = np.zeros((16, 16), dtype=np.uint16)
    lbl0[2:5, 2:5] = 1
    lbl0[10:13, 10:13] = 2

    lbl1 = np.zeros((16, 16), dtype=np.uint16)
    lbl1[4:7, 4:7] = 1

    obs = pd.DataFrame(
        {
            _REGION_KEY: pd.Categorical([r for r, _ in obs_rows]),
            _INSTANCE_KEY: [i for _, i in obs_rows],
        },
        index=[f"{r}_{i}" for r, i in obs_rows],
    )
    adata = ad.AnnData(X=np.zeros((len(obs_rows), 1), dtype=np.float32), obs=obs)
    adata = TableModel.parse(adata, region=["lbl0", "lbl1"], region_key=_REGION_KEY, instance_key=_INSTANCE_KEY)

    sdata = SpatialData(
        images={
            "img0": Image2DModel.parse(da.from_array(img0, chunks=(1, 16, 16)), dims=("c", "y", "x")),
            "img1": Image2DModel.parse(da.from_array(img1, chunks=(1, 16, 16)), dims=("c", "y", "x")),
        },
        labels={
            "lbl0": Labels2DModel.parse(da.from_array(lbl0, chunks=(16, 16)), dims=("y", "x")),
            "lbl1": Labels2DModel.parse(da.from_array(lbl1, chunks=(16, 16)), dims=("y", "x")),
        },
        tables={"tbl": adata},
    )

    sdata = featurize(
        sdata,
        image_name=image_order,
        labels_name=labels_order,
        table_name="tbl",
        output_table_name="tbl_out",
        depth=4,
        diameter=5,
        embedding_dimension=1,
        model=_mean_intensity_model,
        to_coordinate_system=["global", "global"],
        embedding_obsm_key="emb",
        overwrite=True,
    )

    out_table = sdata.tables["tbl_out"]
    emb = out_table.obsm["emb"][:, 0]
    expected = np.array([10.0 * (9 / 25) if r == "lbl0" else 99.0 * (9 / 25) for r in out_table.obs[_REGION_KEY]])
    np.testing.assert_allclose(emb, expected, rtol=1e-5)
