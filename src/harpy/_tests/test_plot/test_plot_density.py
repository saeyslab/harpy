import matplotlib.pyplot as plt
import pytest
from loguru import logger
from matplotlib.axes import Axes

from harpy.plot._plot_density import plot_density
from harpy.utils._keys import _GENES_KEY


def _get_test_gene(sdata_transcripts_no_backed) -> str:
    return sdata_transcripts_no_backed.points["transcripts"].head(1)[_GENES_KEY].iloc[0]


def test_plot_density_returns_input_ax(sdata_transcripts_no_backed):
    fig, ax = plt.subplots()
    gene = _get_test_gene(sdata_transcripts_no_backed)

    result = plot_density(
        sdata_transcripts_no_backed,
        bin_size=100,
        points_layer="transcripts",
        genes=gene,
        frac=0.1,
        ax=ax,
    )

    assert result is ax
    assert isinstance(result, Axes)
    plt.close(fig)


def test_plot_density_warns_when_computing_many_points(monkeypatch, sdata_transcripts_no_backed):
    monkeypatch.setattr("harpy.plot._plot_density._MAX_POINTS_IN_MEMORY", 1)
    gene = _get_test_gene(sdata_transcripts_no_backed)
    messages: list[str] = []
    sink_id = logger.add(messages.append, level="WARNING", format="{message}")

    fig, ax = plt.subplots()
    try:
        plot_density(
            sdata_transcripts_no_backed,
            bin_size=100,
            points_layer="transcripts",
            genes=gene,
            ax=ax,
        )
    finally:
        logger.remove(sink_id)

    assert any("points into memory for plotting" in message for message in messages)
    plt.close(fig)


def test_plot_density_warns_when_creating_large_grid(monkeypatch, sdata_transcripts_no_backed):
    monkeypatch.setattr("harpy.plot._plot_density._MAX_HEATMAP_CELLS", 1)
    gene = _get_test_gene(sdata_transcripts_no_backed)
    messages: list[str] = []
    sink_id = logger.add(messages.append, level="WARNING", format="{message}")

    fig, ax = plt.subplots()
    try:
        plot_density(
            sdata_transcripts_no_backed,
            bin_size=1,
            points_layer="transcripts",
            genes=gene,
            ax=ax,
        )
    finally:
        logger.remove(sink_id)

    assert any("density grid" in message for message in messages)
    plt.close(fig)


def test_plot_density_gene_subset_empty_raises(sdata_transcripts_no_backed):
    with pytest.raises(ValueError, match="No transcripts found for specified gene"):
        plot_density(
            sdata_transcripts_no_backed,
            bin_size=100,
            points_layer="transcripts",
            genes="__not_a_gene__",
        )


@pytest.mark.parametrize("frac", [-0.1, 1.1])
def test_plot_density_invalid_frac_raises(sdata_transcripts_no_backed, frac):
    with pytest.raises(ValueError, match="Please set 'frac' to a value between 0 and 1"):
        plot_density(
            sdata_transcripts_no_backed,
            bin_size=100,
            points_layer="transcripts",
            frac=frac,
        )
