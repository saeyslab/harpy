from harpy.image._transcripts import transcript_density


def test_transcripts(sdata_blobs):
    sdata_blobs = transcript_density(
        sdata_blobs,
        image_name="blobs_image",
        points_name="blobs_points",
        output_image_name="blobs_points_density",
        overwrite=True,
    )
    assert "blobs_points_density" in [*sdata_blobs.images]
