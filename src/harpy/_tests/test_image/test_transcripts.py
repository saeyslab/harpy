from sparrow.image._transcripts import transcript_density


def test_transcripts(sdata_blobs):
    sdata_blobs = transcript_density(
        sdata_blobs,
        img_layer="blobs_image",
        points_layer="blobs_points",
        output_layer="blobs_points_density",
        overwrite=True,
    )
    assert "blobs_points_density" in [*sdata_blobs.images]
