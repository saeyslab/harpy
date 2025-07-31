import os

import matplotlib
import matplotlib.pyplot as plt

from harpy.plot._qc_image import histogram, snr_ratio


def test_plot_histogram(sdata_blobs, tmp_path):
    matplotlib.use("Agg")

    histogram(
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

    histogram(
        sdata_blobs,
        img_layer="blobs_image",
        channel="lineage_2",
        bins=100,
        range=(0, 50),
        ax=axes[0],
        bar_kwargs={"ahlpa": 0.1, "color": "red"},
    )

    histogram(
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


def test_plot_snr_ratio(sdata_blobs, tmp_path):
    # matplotlib.use("Agg") # What is this for?

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    snr_ratio(sdata_blobs, ax=axes[0], channel_names=None)

    snr_ratio(
        sdata_blobs,
        ax=axes[1],
        channel_names=["nucleus", "lineage_0", "lineage_2", "lineage_3", "lineage_5", "lineage_7", "lineage_9"],
    )
    fig.savefig(os.path.join(tmp_path, "snr_ratio"))
