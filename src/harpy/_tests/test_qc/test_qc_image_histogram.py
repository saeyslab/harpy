import os

import dask.array as da
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from spatialdata import SpatialData
from spatialdata.models import Image2DModel

from harpy.qc._qc_image_histogram import image_histogram


def test_image_histogram(sdata_blobs, tmp_path):
    matplotlib.use("Agg")

    image_histogram(
        sdata_blobs,
        image_name="blobs_image",
        channel="lineage_1",
        bins=100,
        range=(0, 50),
        fig_kwargs={
            "figsize": (10, 10),
        },
        bar_kwargs={"ahlpa": 0.1, "color": "red"},
        output=os.path.join(tmp_path, "histogram_1"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    image_histogram(
        sdata_blobs,
        image_name="blobs_image",
        channel="lineage_2",
        bins=100,
        range=(0, 50),
        ax=axes[0],
        bar_kwargs={"ahlpa": 0.1, "color": "red"},
    )

    image_histogram(
        sdata_blobs,
        image_name="blobs_image",
        channel="lineage_3",
        bins=100,
        range=(0, 50),
        ax=axes[1],
        bar_kwargs={"ahlpa": 0.1, "color": "red"},
    )
    axes[1].set_ylabel("")
    fig.savefig(os.path.join(tmp_path, "histogram_2_3"))


def test_image_histogram_ecdf_and_guides(sdata_blobs, tmp_path):
    matplotlib.use("Agg")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    image_histogram(
        sdata_blobs,
        image_name="blobs_image",
        channel="lineage_1",
        bins=100,
        ax=axes[0],
        density=True,
        log_y=True,
        exclude_zeros=True,
        percentile_lines=[0.1, 99.9],
    )

    image_histogram(
        sdata_blobs,
        image_name="blobs_image",
        channel="lineage_2",
        bins=100,
        ax=axes[1],
        kind="ecdf",
        exclude_zeros=True,
        percentile_lines=[5, 95],
    )

    fig.savefig(os.path.join(tmp_path, "histogram_ecdf"))


def test_image_histogram_percentile_guides_are_independent_of_other_percentiles():
    matplotlib.use("Agg")

    rng = np.random.default_rng(42)
    image = da.from_array(
        rng.lognormal(mean=3, sigma=2, size=(1, 10, 10)).astype(np.float32),
        chunks=(1, 1, 10),
    )
    sdata = SpatialData(
        images={
            "image": Image2DModel.parse(
                image,
                dims=("c", "y", "x"),
                c_coords=["marker"],
            )
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    image_histogram(
        sdata,
        image_name="image",
        channel="marker",
        bins=10,
        ax=axes[0],
        percentile_lines=[80, 100],
    )
    image_histogram(
        sdata,
        image_name="image",
        channel="marker",
        bins=10,
        ax=axes[1],
        percentile_lines=[80, 95],
    )

    p80_with_p100 = axes[0].lines[0].get_xdata()[0]
    p80_with_p95 = axes[1].lines[0].get_xdata()[0]
    expected_p80 = da.percentile(image.ravel(), q=80, internal_method="tdigest").compute()

    np.testing.assert_allclose(p80_with_p100, expected_p80)
    np.testing.assert_allclose(p80_with_p95, expected_p80)


def test_image_histogram_multiple_channels(sdata_blobs, tmp_path):
    matplotlib.use("Agg")

    axes = image_histogram(
        sdata_blobs,
        image_name="blobs_image",
        channel=["lineage_1", 2, "lineage_3"],
        bins=50,
        density=True,
        exclude_zeros=True,
        fig_kwargs={"figsize": (9, 6)},
        output=os.path.join(tmp_path, "histogram_multi"),
    )

    assert axes.size >= 3
    assert axes.ravel()[0].get_title() == "lineage_1"


def test_image_histogram_scale_parameter(sdata, tmp_path):
    matplotlib.use("Agg")

    image_histogram(
        sdata,
        image_name="blobs_multiscale_image",
        channel=0,
        bins=25,
        scale="scale2",
        output=os.path.join(tmp_path, "histogram_scale2"),
    )
