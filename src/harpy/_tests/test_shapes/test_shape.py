import importlib

import dask.array as da
import numpy as np
import pytest

from harpy.image._rasterize import rasterize
from harpy.shape._shape import vectorize


def test_vectorize(sdata):
    sdata = vectorize(
        sdata,
        labels_layer="blobs_labels",
        output_layer="blobs_labels_boundaries",
        overwrite=True,
    )

    assert "blobs_labels_boundaries" in [*sdata.shapes]
    unique_labels = da.unique(sdata["blobs_labels"].data).compute()
    unique_labels = unique_labels[unique_labels != 0]
    assert np.array_equal(unique_labels, sdata["blobs_labels_boundaries"].index)


@pytest.mark.skipif(
    not importlib.util.find_spec("rasterio"),
    reason="requires the rasterio library",
)
def test_vectorize_roundtrip(sdata):
    # we do roundtrip. labels->shapes->labels_redo->shapes_redo. And check if shapes and shapes_redo are equal
    # roundtrip unit test only works when doing vectorize with rasterio backend.
    sdata = vectorize(
        sdata,
        labels_layer="blobs_labels",
        output_layer="blobs_labels_boundaries",
        overwrite=True,
    )
    sdata = rasterize(
        sdata,
        shapes_layer="blobs_labels_boundaries",
        output_layer="blobs_labels_redo",
        overwrite=True,
    )

    sdata = vectorize(
        sdata,
        labels_layer="blobs_labels_redo",
        output_layer="blobs_labels_boundaries_redo",
        overwrite=True,
    )

    assert sdata["blobs_labels_boundaries"].geometry.equals(sdata["blobs_labels_boundaries_redo"].geometry)
