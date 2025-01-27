import os

from harpy.plot import cluster
from harpy.table._clustering import leiden


def test_plot_cluster(sdata_multi_c_no_backed, tmp_path):
    sdata_multi_c_no_backed = leiden(
        sdata_multi_c_no_backed,
        labels_layer="masks_whole",
        table_layer="table_intensities",
        output_layer="table_intensities_clustered",
        key_added="leiden",
        rank_genes=True,
        random_state=100,
        overwrite=True,
    )

    cluster(
        sdata_multi_c_no_backed,
        table_layer="table_intensities_clustered",
        output=os.path.join(tmp_path, "cluster"),
    )
