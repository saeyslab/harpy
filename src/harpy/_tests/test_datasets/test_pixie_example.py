from spatialdata import SpatialData


def test_pixie_example(sdata_pixie):
    assert len([*sdata_pixie.images]) != 0
    assert len([*sdata_pixie.labels]) != 0
    assert isinstance(sdata_pixie, SpatialData)
