import dask.dataframe as dd
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from loguru import logger
from matplotlib.axes import Axes
from spatialdata import SpatialData
from spatialdata.transformations import Identity, Scale

from harpy.plot._plot_density import plot_density
from harpy.points._points import add_points_layer
from harpy.utils._keys import _GENES_KEY


def _get_test_gene(sdata_transcripts_no_backed) -> str:
    return sdata_transcripts_no_backed.points["transcripts"].head(1)[_GENES_KEY].iloc[0]


def test_plot_density_returns_input_ax(sdata_transcripts_no_backed, tmp_path):
    fig, ax = plt.subplots()
    gene = _get_test_gene(sdata_transcripts_no_backed)
    try:
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
        fig.savefig(tmp_path / "plot_density_returns_input_ax.png")
    finally:
        plt.close(fig)


def test_plot_density_warns_when_computing_many_points(monkeypatch, sdata_transcripts_no_backed):
    # this sets the constant _MAX_POINTS_IN_MEMORY to 1; otherwise we would need to use a points layer with > MAX_POINTS_IN_MEMORY for the unit test
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


def test_plot_density_without_colorbar(sdata_transcripts_no_backed):
    fig, ax = plt.subplots()
    try:
        plot_density(
            sdata_transcripts_no_backed,
            bin_size=100,
            points_layer="transcripts",
            frac=0.1,
            colorbar=False,
            ax=ax,
        )
        assert len(fig.axes) == 1
    finally:
        plt.close(fig)


def test_plot_density_filters_requested_z_plane(tmp_path):
    sdata = SpatialData()
    ddf = dd.from_pandas(
        pd.DataFrame(
            {
                "x": [0.0, 1.0],
                "y": [0.0, 1.0],
                "z": [0, 1],
                _GENES_KEY: pd.Categorical(["A", "A"]),
            }
        ),
        npartitions=1,
    )
    sdata = add_points_layer(
        sdata,
        ddf=ddf,
        output_layer="transcripts",
        coordinates={"x": "x", "y": "y", "z": "z"},
        transformations={"global": Identity()},
    )

    fig, ax = plt.subplots()
    try:
        plot_density(
            sdata,
            bin_size=1,
            points_layer="transcripts",
            crd=(0, 2, 0, 2),
            z_plane=1,
            ax=ax,
        )
        image = ax.images[0]
        # Sum all heatmap bins to verify that only the single transcript in z-plane 1 contributed.
        assert image.get_array().sum() == 1
        fig.savefig(tmp_path / "plot_density_filters_requested_z_plane.png")
    finally:
        plt.close(fig)


def test_plot_density_z_plane_without_z_column_raises():
    sdata = SpatialData()
    ddf = dd.from_pandas(
        pd.DataFrame(
            {
                "x": [0.0],
                "y": [0.0],
                _GENES_KEY: pd.Categorical(["A"]),
            }
        ),
        npartitions=1,
    )
    sdata = add_points_layer(
        sdata,
        ddf=ddf,
        output_layer="transcripts",
        coordinates={"x": "x", "y": "y"},
        transformations={"global": Identity()},
    )

    with pytest.raises(ValueError, match="does not contain a 'z' column"):
        plot_density(
            sdata,
            bin_size=1,
            points_layer="transcripts",
            z_plane=0,
        )


def test_plot_density_bin_size_in_target_coordinate_system(tmp_path):
    sdata = SpatialData()
    ddf = dd.from_pandas(
        pd.DataFrame(
            {
                # With the 0.5 scale to global_micron, these map as:
                # (2.0, 2.0) -> (1.0, 1.0) microns
                # (4.0, 4.0) -> (2.0, 2.0) microns
                "x": [2.0, 4.0],
                "y": [2.0, 4.0],
                _GENES_KEY: pd.Categorical(["A", "A"]),
            }
        ),
        npartitions=1,
    )
    sdata = add_points_layer(
        sdata,
        ddf=ddf,
        output_layer="transcripts",
        coordinates={"x": "x", "y": "y"},
        transformations={"global": Identity(), "global_micron": Scale(axes=("x", "y"), scale=[0.5, 0.5])},
    )

    fig, ax = plt.subplots()
    try:
        plot_density(
            sdata,
            bin_size=1,
            points_layer="transcripts",
            crd=(0, 3, 0, 3),
            to_coordinate_system="global_micron",
            ax=ax,
        )
        image = ax.images[0]
        # Both points fall inside the micron-space crd and should be counted after transforming for histogramming.
        assert image.get_array().sum() == 2
        assert image.get_extent() == [0, 3, 0, 3]
        fig.savefig(tmp_path / "plot_density_bin_size_in_target_coordinate_system.png")
    finally:
        plt.close(fig)
