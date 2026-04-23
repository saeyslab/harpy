from harpy.image.pixel_clustering._preprocess import pixel_clustering_preprocess


def test_pixel_clustering_preprocess_blobs(sdata_blobs):
    image_name = "blobs_image"
    channels = ["lineage_0", "lineage_1", "lineage_5", "lineage_9"]

    sdata_blobs = pixel_clustering_preprocess(
        sdata_blobs,
        image_name=[image_name],
        output_image_name=[f"{image_name}_preprocessed"],
        channels=channels,
        p=99,
        p_sum=5,
        p_post=99.9,
        sigma=2.0,
        norm_sum=True,
        chunks=200,
        overwrite=True,
    )

    assert f"{image_name}_preprocessed" in sdata_blobs.images
    assert (sdata_blobs[f"{image_name}_preprocessed"].c.data == channels).all()
