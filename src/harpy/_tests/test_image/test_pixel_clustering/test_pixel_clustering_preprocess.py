from harpy.image.pixel_clustering._preprocess import pixel_clustering_preprocess


def test_pixel_clustering_preprocess_blobs(sdata_blobs):
    img_layer = "blobs_image"
    channels = ["lineage_0", "lineage_1", "lineage_5", "lineage_9"]

    sdata_blobs = pixel_clustering_preprocess(
        sdata_blobs,
        img_layer=[img_layer],
        output_layer=[f"{img_layer}_preprocessed"],
        channels=channels,
        q=99,
        q_sum=5,
        q_post=99.9,
        sigma=2.0,
        norm_sum=True,
        chunks=200,
        overwrite=True,
    )

    assert f"{img_layer}_preprocessed" in sdata_blobs.images
    assert (sdata_blobs[f"{img_layer}_preprocessed"].c.data == channels).all()
