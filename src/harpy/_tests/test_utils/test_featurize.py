import dask.array as da
import numpy as np
import pytest
from scipy import ndimage
from spatialdata import SpatialData

from harpy.image import add_image_layer
from harpy.utils._featurize import Featurizer
from harpy.utils._keys import _INSTANCE_KEY


def test_featurize(sdata_transcripts_no_backed: SpatialData):
    sdata = sdata_transcripts_no_backed

    chunksize_spatial = 2048
    depth = 100
    embedding_dimension = 50
    mask = sdata["segmentation_mask"].data[None, ...].rechunk(chunksize_spatial)

    image = sdata["raw_image"].data.rechunk(chunksize_spatial)

    sdata = add_image_layer(
        sdata,
        arr=da.concatenate([image, image, image, image], axis=0),
        output_layer="raw_image",
        overwrite=True,
    )
    image = sdata["raw_image"].data[:, None, ...].rechunk((3, 1, chunksize_spatial, chunksize_spatial))

    featurizer = Featurizer(mask_dask_array=mask, image_dask_array=image)

    instances_ids, dask_chunks = featurizer.featurize(
        depth=depth,
        diameter=75,
        embedding_dimension=embedding_dimension,
    )

    result = dask_chunks.compute()
    # check that all labels are extracted
    index = da.unique(mask).compute()
    index = index[index != 0]

    assert np.array_equal(index, np.sort(instances_ids))
    assert result.shape[0] == index.shape[0]
    assert result.shape[1] == embedding_dimension


@pytest.mark.parametrize("extract_mask", [True, False])
def test_extract_instances(sdata_transcripts_no_backed, extract_mask):
    sdata = sdata_transcripts_no_backed

    chunksize_spatial = 2048
    depth = 100
    mask = sdata["segmentation_mask"].data[None, ...].rechunk(chunksize_spatial)

    image = sdata["raw_image"].data.rechunk(chunksize_spatial)

    sdata = add_image_layer(
        sdata,
        arr=da.concatenate([image, image, image, image], axis=0),
        output_layer="raw_image",
        overwrite=True,
    )
    image = sdata["raw_image"].data[:, None, ...].rechunk((3, 1, chunksize_spatial, chunksize_spatial))

    featurizer = Featurizer(mask_dask_array=mask, image_dask_array=image)

    instances_ids, dask_chunks = featurizer.extract_instances(
        depth=depth,
        diameter=75,
        extract_mask=extract_mask,
    )

    instances = dask_chunks.compute()
    assert instances.dtype == np.uint32 if extract_mask else image.dtype
    assert instances.shape == (657, 5, 1, 75, 75) if extract_mask else (657, 4, 1, 75, 75)

    if extract_mask:
        # check that mask of each instance contains the index corresponding to instances_ids
        for _index, _item in zip(instances_ids, instances, strict=True):
            _item_labels = np.unique(_item[0])
            _item_labels = _item_labels[_item_labels != 0]
            assert len(_item_labels) == 1
            assert _index == _item_labels[0]

        # check that all labels are extracted
        index = da.unique(instances[:, 0, ...]).compute()
        index = index[index != 0]

        assert np.array_equal(index, np.sort(instances_ids))


def test_extract_instances_mask(sdata_transcripts_no_backed):
    sdata = sdata_transcripts_no_backed

    chunksize_spatial = 2048
    depth = 100
    mask = sdata["segmentation_mask"].data[None, ...].rechunk(chunksize_spatial)

    featurizer = Featurizer(mask_dask_array=mask, image_dask_array=None)

    instances_ids, dask_chunks = featurizer.extract_instances(
        depth=depth,
        diameter=75,
        extract_mask=True,
    )

    instances = dask_chunks.compute()
    assert instances.dtype == mask.dtype
    assert instances.shape == (657, 1, 1, 75, 75)  # we only extract the mask

    # check that mask of each instance contains the index corresponding to instances_ids
    for _index, _item in zip(instances_ids, instances, strict=True):
        _item_labels = np.unique(_item[0])
        _item_labels = _item_labels[_item_labels != 0]
        assert len(_item_labels) == 1
        assert _index == _item_labels[0]

    # check that all labels are extracted
    index = da.unique(instances[:, 0, ...]).compute()
    index = index[index != 0]

    assert np.array_equal(index, np.sort(instances_ids))


def test_extract_instances_mean(sdata_transcripts_no_backed):
    sdata = sdata_transcripts_no_backed

    chunksize_spatial = 2048
    mask = sdata["segmentation_mask"].data[None, ...].rechunk(chunksize_spatial)

    image = sdata["raw_image"].data[:, None, ...].rechunk(chunksize_spatial)

    labels = da.unique(mask).compute()
    labels = labels[labels != 0]

    scipy_mean = ndimage.labeled_comprehension(
        input=image[0].compute(),
        labels=mask.compute(),
        index=labels,
        func=np.mean,
        out_dtype=np.float32,
        default=0,
    )

    featurizer = Featurizer(mask_dask_array=mask, image_dask_array=image)

    df_mean_featurizer = featurizer._mean(diameter=150)

    assert df_mean_featurizer.shape[1] - 1 == image.shape[0]

    assert np.allclose(df_mean_featurizer[0].values, scipy_mean, rtol=0, atol=1e-3)


def test_extract_instances_duplicates_blobs(sdata):
    chunksize_spatial = 512
    depth = 250

    mask = sdata["blobs_labels"].data[None, ...].rechunk(chunksize_spatial)
    image = sdata["blobs_image"].data[:, None, ...].astype(np.float32).rechunk(chunksize_spatial)

    featurizer = Featurizer(mask_dask_array=mask, image_dask_array=image)

    instances_ids, dask_chunks = featurizer.extract_instances(
        depth=depth,
        diameter=1000,
        extract_mask=True,
    )

    instances = dask_chunks.compute()

    # check that mask of each instance contains the index corresponding to instances_ids
    for _index, _item in zip(instances_ids, instances, strict=True):
        _item_labels = np.unique(_item[0])
        _item_labels = _item_labels[_item_labels != 0]
        assert len(_item_labels) == 1
        assert _index == _item_labels[0]

    # check that all labels are extracted
    index = da.unique(instances[:, 0, ...]).compute()
    index = index[index != 0]

    assert np.array_equal(index, np.sort(instances_ids))

    assert np.array_equal(
        instances_ids,
        np.array(
            [3, 5, 11, 12, 18, 19, 25, 29, 30, 31, 1, 6, 13, 17, 22, 26, 4, 8, 9, 16, 20, 23, 24, 27, 2, 10],
            dtype=np.int16,
        ),
    )


@pytest.mark.parametrize("fov_nr", [0, 1])
def test_instance_statistics_quantiles(sdata_pixie, fov_nr):
    image = sdata_pixie[f"raw_image_fov{fov_nr}"].data[:, None, ...].rechunk(100)
    mask = sdata_pixie[f"label_whole_fov{fov_nr}"].data[None, ...].rechunk(100)
    featurizer = Featurizer(
        mask_dask_array=mask,
        image_dask_array=image,
    )
    q = [0.3, 0.5]
    dfs = featurizer.quantiles(q=q, diameter=75, depth=50, batch_size=100, instance_key=_INSTANCE_KEY)
    assert len(dfs) == len(q)
    image = image.compute()
    mask = mask.compute()

    labels = np.unique(mask)
    labels = labels[labels != 0]

    C = image.shape[0]

    for _label in labels:
        quantiles_label = np.quantile(image[:, mask == _label], q=q, axis=1)  # quantiles_label is of shape len(q), c
        for i, _df_q in enumerate(dfs):
            assert _df_q.shape == (len(labels), C + 1)
            _q_label = quantiles_label[i]
            _q_label_computed = _df_q[_df_q[_INSTANCE_KEY] == _label][np.arange(0, C)].values
            assert np.allclose(_q_label, _q_label_computed)


def test_instance_statistics_quantiles_blobs(sdata):
    chunksize_spatial = 1000

    mask = (
        sdata["blobs_labels"].data[None, ...].rechunk(chunksize_spatial)
    )  # 1 chunk (labels in blobs_labels are non-local, they span complete image, therefore process as one chunk)
    image = sdata["blobs_image"].data[:, None, ...].rechunk(chunksize_spatial)

    featurizer = Featurizer(
        mask_dask_array=mask,
        image_dask_array=image,
    )
    q = [0.3, 0.5]
    dfs = featurizer.quantiles(
        q=q,
        diameter=chunksize_spatial,
        depth=50,  # FIXME depth 0 results in error, fix this, should work if there is only one chunk
        batch_size=5,
        instance_key=_INSTANCE_KEY,
    )  # one chunk
    assert len(dfs) == len(q)
    image = image.compute()
    mask = mask.compute()

    labels = np.unique(mask)
    labels = labels[labels != 0]

    C = image.shape[0]

    for _label in labels:
        quantiles_label = np.quantile(image[:, mask == _label], q=q, axis=1)  # quantiles_label is of shape len(q), c
        for i, _df_q in enumerate(dfs):
            assert _df_q.shape == (len(labels), C + 1)
            _q_label = quantiles_label[i]
            _q_label_computed = _df_q[_df_q[_INSTANCE_KEY] == _label][np.arange(0, C)].values
            assert np.allclose(_q_label, _q_label_computed)


def test_extract_instances_mean_blobs(sdata):
    chunksize_spatial = 1000

    mask = sdata["blobs_labels"].data[None, ...].rechunk(chunksize_spatial)  # 1 chunk
    image = sdata["blobs_image"].data[:, None, ...].astype(np.float32).rechunk(chunksize_spatial)

    labels = da.unique(mask).compute()
    labels = labels[labels != 0]

    scipy_mean = ndimage.labeled_comprehension(
        input=image[0].compute(),
        labels=mask.compute(),
        index=labels,
        func=np.mean,
        out_dtype=image.dtype,
        default=0,
    )

    featurizer = Featurizer(
        mask_dask_array=mask,
        image_dask_array=image,
    )
    df_mean_featurizer = featurizer._mean(diameter=1000)

    assert df_mean_featurizer.shape[1] - 1 == image.shape[0]

    assert np.allclose(df_mean_featurizer[0].values, scipy_mean, rtol=0, atol=1e-5)
