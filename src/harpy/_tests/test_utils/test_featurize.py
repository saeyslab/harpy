import dask.array as da
import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import ndimage
from spatialdata import SpatialData

from harpy.image import add_image_layer
from harpy.utils._featurize import Featurizer, _region_radii_and_axes
from harpy.utils._keys import _INSTANCE_KEY
from harpy.utils.utils import _get_xp, _to_numpy


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
    # put it on cpu
    instances = _to_numpy(instances)
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
    instances = _to_numpy(instances)
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
    depth = 500

    mask = sdata["blobs_labels"].data[None, ...].rechunk(chunksize_spatial)
    image = sdata["blobs_image"].data[:, None, ...].astype(np.float32).rechunk(chunksize_spatial)

    featurizer = Featurizer(mask_dask_array=mask, image_dask_array=image)

    instances_ids, dask_chunks = featurizer.extract_instances(
        depth=depth,
        diameter=1000,
        extract_mask=True,
    )

    instances = dask_chunks.compute()
    instances = _to_numpy(instances)
    # check that mask of each instance contains the index corresponding to instances_ids
    for _index, _item in zip(instances_ids, instances, strict=True):
        _item_labels = np.unique(_item[0])
        _item_labels = _item_labels[_item_labels != 0]
        assert len(_item_labels) == 1
        assert _index == _item_labels[0]

    # check that all labels are extracted
    index = da.unique(instances[:, 0, ...]).compute()
    index = index[index != 0]

    uniq_labels = da.unique(mask).compute()
    uniq_labels = uniq_labels[uniq_labels != 0]

    assert np.array_equal(uniq_labels, index)
    assert np.array_equal(uniq_labels, np.sort(instances_ids))
    assert np.array_equal(
        instances_ids,
        uniq_labels,
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
    dfs = featurizer.quantiles(
        q=q,
        diameter=75,
        depth=50,
        batch_size=100,
        instance_key=_INSTANCE_KEY,
    )
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
    diameter = 1990

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
        diameter=diameter,
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


def test_instance_statistics_dummy_statistic_image(sdata_pixie):
    # Load example dataset
    sdata = sdata_pixie

    # Prepare image and mask arrays
    image_array = sdata["raw_image_fov0"].data[:, None, ...]
    mask_array = sdata["label_whole_fov0"].data[None, ...]

    C, _, _, _ = image_array.shape

    def _dummy_statistic_image(
        array: NDArray,
        value: int,
        run_on_gpu: bool = True,
    ):
        xp, _ = _get_xp(array, run_on_gpu=run_on_gpu)
        xp.random.seed(42)
        # shape of array=(c, number of pixels corresponding to non zero mask for instance i)
        assert array.ndim == 2
        C = array.shape[0]
        _statistic_dimension = 3
        # return dummy statistic of shape (C, statistic_dimension)
        return xp.random.rand(C, _statistic_dimension) + value

    # Create featurizer
    featurizer = Featurizer(
        mask_dask_array=mask_array,
        image_dask_array=image_array,
    )

    value = 100
    statistic_dimension = 3
    fn_kwargs = {"value": value}

    # Compute instance statistics
    instance_ids, calculated_statistic_lazy = featurizer.calculate_instance_statistics(
        diameter=50,
        depth=100,
        statistic_dimension=statistic_dimension,
        fn=_dummy_statistic_image,
        fn_kwargs=fn_kwargs,
        extract_image=True,
        batch_size=500,
    )

    result = calculated_statistic_lazy.compute()

    labels = da.unique(mask_array).compute()
    labels = labels[labels != 0]
    I = labels.shape[0]

    assert set(labels) == set(instance_ids)
    assert result.shape == (I, C, statistic_dimension)
    # some sanity check on the result
    assert (result > value).all()


@pytest.mark.parametrize("fov_nr", [0, 1])
def test_instance_statistics_radii_and_principal_axes(sdata_pixie, fov_nr):
    mask = sdata_pixie[f"label_whole_fov{fov_nr}"].data[None, ...].rechunk(100)

    featurizer = Featurizer(
        mask_dask_array=mask,
        image_dask_array=None,  # one could specify the image here, but anyway it is not used for calculation of radii and axes
        run_on_gpu=False,
    )
    df = featurizer.radii_and_principal_axes(
        calculate_axes=True,
        diameter=75,
        depth=50,
        batch_size=100,
        instance_key=_INSTANCE_KEY,
    )

    mask = mask.compute()
    labels = np.unique(mask)
    labels = labels[labels != 0]

    for _label in labels:
        radii, axes = _region_radii_and_axes(mask, label=_label)
        assert np.allclose(radii, df[df[_INSTANCE_KEY] == _label][range(3)].values.flatten())
        # NOTE: for fov2 the following test will fail. For instance with ID=287,
        # the featurizer.radii_and_principal_axes will give the vector:
        # (0.707107	-0.707107	0.0) for the first principal axes, while _region_radii_and_axes gives:
        # (-0.707107	0.707107	0.0) for the first principal axes, these are equivalent axes
        assert np.allclose(axes.flatten(), df[df[_INSTANCE_KEY] == _label][range(3, 12)].values.flatten())


def test_instance_statistics_radii_and_principal_axes_blobs(sdata_blobs):
    chunksize_spatial = 1000

    mask = (
        sdata_blobs["blobs_labels"].data[None, ...].rechunk(chunksize_spatial)
    )  # 1 chunk (labels in blobs_labels are non-local, they span complete image, therefore process as one chunk)

    featurizer = Featurizer(
        mask_dask_array=mask,
        image_dask_array=None,  # one could specify the image here, but anyway it is not used for calculation of radii and axes
        run_on_gpu=False,
    )
    df = featurizer.radii_and_principal_axes(
        calculate_axes=True,
        diameter=chunksize_spatial,
        batch_size=5,
        instance_key=_INSTANCE_KEY,
    )

    mask = mask.compute()
    labels = np.unique(mask)
    labels = labels[labels != 0]

    for _label in labels:
        radii, axes = _region_radii_and_axes(mask, label=_label)
        assert np.allclose(radii, df[df[_INSTANCE_KEY] == _label][range(3)].values.flatten())
        assert np.allclose(axes.flatten(), df[df[_INSTANCE_KEY] == _label][range(3, 12)].values.flatten())


def test_instance_statistics_dummy_statistic_mask(sdata_pixie):
    sdata = sdata_pixie

    mask_array = sdata["label_whole_fov0"].data[None, ...]

    def _dummy_statistic_mask(
        array: NDArray,
        value: int,
        run_on_gpu: bool = True,
    ) -> NDArray:
        xp, _ = _get_xp(array, run_on_gpu=run_on_gpu)
        # array should be of dtype int
        xp.random.seed(42)
        assert xp.issubdtype(array.dtype, xp.integer)
        # array is of shape = z,y,x, with y and x the size of the instance window.
        assert array.ndim == 3
        statistic_dimension = 5
        result = xp.random.rand(statistic_dimension) + value
        # return array containing float of shape (statistic_dimension,)
        return result[None, ...]

    featurizer = Featurizer(mask_dask_array=mask_array, image_dask_array=None)

    value = 100
    statistic_dimension = 5
    fn_kwargs = {"value": value}

    instance_ids, calculated_statistic_lazy = featurizer.calculate_instance_statistics(
        diameter=100,
        depth=50,
        statistic_dimension=statistic_dimension,
        fn=_dummy_statistic_mask,
        fn_kwargs=fn_kwargs,
        extract_image=True,
        batch_size=500,
    )

    result = calculated_statistic_lazy.compute()

    labels = da.unique(mask_array).compute()
    labels = labels[labels != 0]
    I = labels.shape[0]

    assert set(labels) == set(instance_ids)
    assert result.shape == (I, 1, statistic_dimension)
    # some sanity check on the result
    assert (result > value).all()


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
    df_mean_featurizer = featurizer._mean(diameter=1990)

    assert df_mean_featurizer.shape[1] - 1 == image.shape[0]

    assert np.allclose(df_mean_featurizer[0].values, scipy_mean, rtol=0, atol=1e-5)


def test_region_radii_and_axes():
    mask = np.array(
        [
            [1, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )

    mask = mask[None, ...]

    radii, axis = _region_radii_and_axes(mask=mask, label=1)

    assert np.array_equal(np.array([1.0, 0.0, 0.0]), radii)

    assert np.allclose(np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), axis)
