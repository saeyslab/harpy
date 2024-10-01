from spatialdata import SpatialData

from sparrow.datasets.transcriptomics import visium_hd_example


def test_visium_hd_example():
    sdata = visium_hd_example(bin_size=16)
    assert isinstance(sdata, SpatialData)
