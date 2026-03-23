import os

import matplotlib
import matplotlib.pyplot as plt

from harpy.qc._qc_image_histogram import image_histogram


def test_image_histogram(sdata_blobs, tmp_path):
    matplotlib.use("Agg")

    image_histogram(
        sdata_blobs,
        img_layer="blobs_image",
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
        img_layer="blobs_image",
        channel="lineage_2",
        bins=100,
        range=(0, 50),
        ax=axes[0],
        bar_kwargs={"ahlpa": 0.1, "color": "red"},
    )

    image_histogram(
        sdata_blobs,
        img_layer="blobs_image",
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
        img_layer="blobs_image",
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
        img_layer="blobs_image",
        channel="lineage_2",
        bins=100,
        ax=axes[1],
        kind="ecdf",
        exclude_zeros=True,
        percentile_lines=[5, 95],
    )

    fig.savefig(os.path.join(tmp_path, "histogram_ecdf"))


def test_image_histogram_multiple_channels(sdata_blobs, tmp_path):
    matplotlib.use("Agg")

    axes = image_histogram(
        sdata_blobs,
        img_layer="blobs_image",
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
        img_layer="blobs_multiscale_image",
        channel=0,
        bins=25,
        scale="scale2",
        output=os.path.join(tmp_path, "histogram_scale2"),
    )
