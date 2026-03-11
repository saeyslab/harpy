import os

import matplotlib.pyplot as plt

from harpy.plot._qc_image import snr_ratio


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
