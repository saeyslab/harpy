import numpy as np

from harpy.image._normalize import normalize


def test_normalize(sdata_blobs):
    eps = 1e-20
    p_min = 5
    p_max = 95
    sdata_blobs = normalize(
        sdata_blobs,
        img_layer="blobs_image",
        output_layer="blobs_image_normalized",
        p_min=p_min,
        p_max=p_max,
        eps=eps,
    )
    arr_original = sdata_blobs["blobs_image"].data.compute()
    arr_normalized = sdata_blobs["blobs_image_normalized"].data.compute()

    mi = np.percentile(arr_original, q=p_min)
    ma = np.percentile(arr_original, q=p_max)
    arr_normalized_redo = (arr_original - mi) / (ma - mi + eps)
    arr_normalized_redo = np.clip(arr_normalized_redo, 0, 1)
    assert np.allclose(arr_normalized, arr_normalized_redo, rtol=0, atol=0.1)


def test_normalize_channels(sdata_blobs):
    # test for normalization on each channel individually
    eps = 1e-20
    p_min = 5
    p_max = 95
    sdata_blobs = normalize(
        sdata_blobs,
        img_layer="blobs_image",
        output_layer="blobs_image_normalized",
        p_min=sdata_blobs["blobs_image"].c.data.shape[0] * [p_min],
        p_max=sdata_blobs["blobs_image"].c.data.shape[0] * [p_max],
    )
    arr_original = sdata_blobs["blobs_image"].data.compute()
    arr_normalized = sdata_blobs["blobs_image_normalized"].data.compute()

    # check for channel 0
    mi = np.percentile(arr_original[0], q=p_min)
    ma = np.percentile(arr_original[0], q=p_max)
    arr_normalized_redo_channel_0 = (arr_original[0] - mi) / (ma - mi + eps)
    arr_normalized_redo_channel_0 = np.clip(arr_normalized_redo_channel_0, 0, 1)
    assert np.allclose(arr_normalized[0], arr_normalized_redo_channel_0, rtol=0, atol=0.1)
