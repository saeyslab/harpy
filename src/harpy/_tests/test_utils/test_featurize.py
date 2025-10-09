import dask.array as da
import numpy as np
from scipy import ndimage

from harpy.image import add_image_layer
from harpy.utils._featurize import Featurizer


def test_featurize(sdata_transcripts_no_backed):
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


def test_featurize_mean(sdata):
    se_image = sdata["blobs_image"]
    se_labels = sdata["blobs_labels"]

    image = se_image.data[:, None, ...]
    mask = se_labels.data[None, ...]

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
        mask_dask_array=se_labels.data[None, ...],
        image_dask_array=se_image.data.astype(np.float32)[:, None, ...],
    )
    df_mean_featurizer = featurizer._mean(diameter=1000)

    assert df_mean_featurizer.shape[1] - 1 == image.shape[0]

    assert np.allclose(df_mean_featurizer[0].values, scipy_mean, rtol=0, atol=1e-5)
