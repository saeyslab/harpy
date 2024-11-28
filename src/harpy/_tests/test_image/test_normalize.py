import numpy as np

from harpy.image._normalize import normalize


def test_normalize(sdata_blobs):
    eps = 1e-20
    q_min = 5
    q_max = 95
    sdata_blobs = normalize(
        sdata_blobs,
        img_layer="blobs_image",
        output_layer="blobs_image_normalized",
        q_min=q_min,
        q_max=q_max,
        eps=eps,
    )
    arr_original = sdata_blobs["blobs_image"].data.compute()
    arr_normalized = sdata_blobs["blobs_image_normalized"].data.compute()

    mi = np.percentile(arr_original, q=q_min)
    ma = np.percentile(arr_original, q=q_max)
    arr_normalized_redo = (arr_original - mi) / (ma - mi + eps)
    arr_normalized_redo = np.clip(arr_normalized_redo, 0, 1)
    assert np.allclose(arr_normalized, arr_normalized_redo, rtol=0, atol=0.1)


def test_normalize_channels(sdata_blobs):
    # test for normalization on each channel individually
    eps = 1e-20
    q_min = 5
    q_max = 95
    sdata_blobs = normalize(
        sdata_blobs,
        img_layer="blobs_image",
        output_layer="blobs_image_normalized",
        q_min=sdata_blobs["blobs_image"].c.data.shape[0] * [q_min],
        q_max=sdata_blobs["blobs_image"].c.data.shape[0] * [q_max],
    )
    arr_original = sdata_blobs["blobs_image"].data.compute()
    arr_normalized = sdata_blobs["blobs_image_normalized"].data.compute()

    # check for channel 0
    mi = np.percentile(arr_original[0], q=q_min)
    ma = np.percentile(arr_original[0], q=q_max)
    arr_normalized_redo_channel_0 = (arr_original[0] - mi) / (ma - mi + eps)
    arr_normalized_redo_channel_0 = np.clip(arr_normalized_redo_channel_0, 0, 1)
    assert np.allclose(arr_normalized[0], arr_normalized_redo_channel_0, rtol=0, atol=0.1)
