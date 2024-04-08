import os

from sparrow.plot import cluster
from sparrow.table._clustering import leiden


def test_plot_cluster(sdata_multi_c, tmp_path):
    sdata_multi_c = leiden(
        sdata_multi_c,
        labels_layer="masks_whole",
        table_layer="table_intensities",
        output_layer="table_intensities_clustered",
        key_added="leiden",
        rank_genes=True,
        random_state=100,
        overwrite=True,
    )

    cluster(
        sdata_multi_c,
        table_layer="table_intensities_clustered",
        output=os.path.join(tmp_path, "cluster"),
    )
