from sparrow.table._preprocess import preprocess_proteomics
from sparrow.table._table import filter_on_size


def test_filter_on_size(sdata_multi_c_no_backed):
    sdata_multi_c_no_backed = preprocess_proteomics(
        sdata_multi_c_no_backed,
        labels_layer=["masks_whole", "masks_nuclear_aligned"],
        table_layer="table_intensities",
        output_layer="table_intensities_preprocessed",
        overwrite=True,
    )

    sdata_multi_c_no_backed = filter_on_size(
        sdata_multi_c_no_backed,
        table_layer="table_intensities_preprocessed",
        labels_layer=["masks_whole"],
        output_layer="table_intensities_filter",
        min_size=100,
        max_size=100000,
        overwrite=True,
        update_shapes_layers=True,
    )
    assert sdata_multi_c_no_backed.tables["table_intensities_filter"].shape == (643, 22)

    sdata_multi_c_no_backed = filter_on_size(
        sdata_multi_c_no_backed,
        table_layer="table_intensities_preprocessed",
        labels_layer=["masks_whole", "masks_nuclear_aligned"],
        output_layer="table_intensities_filter",
        min_size=100,
        max_size=100000,
        overwrite=True,
        update_shapes_layers=True,
    )
    assert sdata_multi_c_no_backed.tables["table_intensities_filter"].shape == (1136, 22)
