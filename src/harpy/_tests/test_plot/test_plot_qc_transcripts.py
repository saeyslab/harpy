import matplotlib
import matplotlib.pyplot as plt

import harpy as hp


def test_qc_metrics_histogram(sdata_transcripts_no_backed, tmp_path):
    matplotlib.use("Agg")

    fig = plt.figure()
    try:
        axes = hp.pl.qc_metrics_histogram(
            sdata_transcripts_no_backed,
            table_layer="table_transcriptomics_preprocessed",
            quantile_range=(0.1, 0.95),
        )

        axes = axes.ravel()

        assert axes.size == 6
        assert axes[0].get_xlabel() == "Total Counts"
        assert axes[1].get_xlabel() == "N Genes By Counts"
        fig = axes[0].figure
        fig.savefig(tmp_path / "qc_metrics_histogram.png")
    finally:
        plt.close(fig)


def test_qc_obs_scatter(sdata_transcripts_no_backed, tmp_path):
    matplotlib.use("Agg")

    fig, ax = plt.subplots()
    try:
        result = hp.pl.qc_obs_scatter(
            sdata_transcripts_no_backed,
            table_layer="table_transcriptomics_preprocessed",
            column_x="shapeSize",
            column_y="total_counts",
            ax=ax,
        )

        assert result is ax
        assert ax.get_xlabel() == "Shapesize"
        assert ax.get_ylabel() == "Total Counts"
        fig.savefig(tmp_path / "qc_obs_scatter.png")
    finally:
        plt.close(fig)
