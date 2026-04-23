import dask.array as da
import pytest
from spatialdata.models import TableModel

from harpy.image._image import add_labels
from harpy.table._regionprops import add_regionprop_features, add_regionprops
from harpy.table._table import add_table

ALL_REGIONPROPS = [
    "area",
    "eccentricity",
    "major_axis_length",
    "minor_axis_length",
    "perimeter",
    "centroid",
    "convex_area",
    "equivalent_diameter",
    "major_minor_axis_ratio",
    "perim_square_over_area",
    "major_axis_equiv_diam_ratio",
    "convex_hull_resid",
    "centroid_dif",
]


@pytest.mark.parametrize(
    "properties_to_calculate",
    [
        ALL_REGIONPROPS,
        ["area"],
    ],
)
def test_add_regionprops(sdata_pixie_intensities, properties_to_calculate):
    table_name = "table_intensities"

    sdata_pixie_intensities = add_regionprops(
        sdata_pixie_intensities,
        labels_name=["label_whole_fov0", "label_whole_fov1"],
        table_name=table_name,
        output_table_name=table_name,
        properties=properties_to_calculate,
        overwrite=True,
    )

    shape_obs_axis_1 = (
        2 + len(properties_to_calculate) + 1  # +1 for centroid_x and centroid_y
        if "centroid" in properties_to_calculate
        else 2 + len(properties_to_calculate)
    )
    assert sdata_pixie_intensities.tables[table_name].obs.shape == (1414, shape_obs_axis_1)

    for _prop in properties_to_calculate:
        if _prop == "centroid":
            assert (
                f"{_prop}_x" in sdata_pixie_intensities[table_name].obs.columns
                and f"{_prop}_y" in sdata_pixie_intensities[table_name].obs.columns
            )
        else:
            assert _prop in sdata_pixie_intensities[table_name].obs.columns


# test if ValueErrors raised correctly
# test case where labels layer not annotated by the table element
def test_add_regionprops_raises(sdata_pixie_intensities):
    table_name = "table_intensities"
    labels_name = "label_whole_fov0"

    sdata_pixie_intensities = add_labels(
        sdata_pixie_intensities,
        arr=sdata_pixie_intensities[labels_name].data,
        output_labels_name=f"{labels_name}_not_annotated",
        overwrite=True,
    )

    # labels layer not annotated by the table layer
    with pytest.raises(
        ValueError,
        match=f"labels element '{labels_name}_not_annotated' not annotated by table element '{table_name}'",
    ):
        sdata_pixie_intensities = add_regionprops(
            sdata_pixie_intensities,
            labels_name=f"{labels_name}_not_annotated",
            table_name=table_name,
            output_table_name=table_name,
            properties=["area"],
            overwrite=True,
        )

    property_not_supported = "dummy_property"
    with pytest.raises(
        ValueError,
        match=f"Cell property {property_not_supported} is not supported. Please choose properties from the following list",
    ):
        sdata_pixie_intensities = add_regionprops(
            sdata_pixie_intensities,
            labels_name=labels_name,
            table_name=table_name,
            output_table_name=table_name,
            properties=[property_not_supported],
            overwrite=True,
        )


ALL_REGIONPROPS_3D = [
    "area",
    "major_axis_length",
    "minor_axis_length",
    "centroid",
    "convex_area",
    "equivalent_diameter",
    "major_minor_axis_ratio",
    "major_axis_equiv_diam_ratio",
    "convex_hull_resid",
    "centroid_dif",
]


@pytest.mark.parametrize(
    "properties_to_calculate",
    [
        ALL_REGIONPROPS_3D,
        ["area"],
    ],
)
def test_add_regionprops_3D(sdata_pixie_intensities, properties_to_calculate):
    table_name = "table_intensities"
    labels_name = "label_whole_fov0"

    array = da.stack([sdata_pixie_intensities[labels_name].data, sdata_pixie_intensities[labels_name].data])

    # add artificial 3D labels layer
    sdata_pixie_intensities = add_labels(
        sdata_pixie_intensities,
        arr=array,
        output_labels_name="label_whole_fov0_3D",
        overwrite=True,
    )
    # annotate the table layer with label_whole_fov0_3D
    adata = sdata_pixie_intensities[table_name]
    region_key = adata.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY]
    adata = adata[adata.obs[region_key] == "label_whole_fov0"].copy()  # copy, otherwise pop will not work
    adata.obs[region_key] = "label_whole_fov0_3D"
    adata.uns.pop(TableModel.ATTRS_KEY)

    sdata_pixie_intensities = add_table(
        sdata_pixie_intensities,
        adata=adata,
        region=["label_whole_fov0_3D"],
        output_table_name=table_name,
        overwrite=True,
    )

    sdata_pixie_intensities = add_regionprops(
        sdata_pixie_intensities,
        labels_name=["label_whole_fov0_3D"],
        table_name=table_name,
        output_table_name=table_name,
        properties=properties_to_calculate,
        overwrite=True,
    )

    shape_obs_axis_1 = (
        2 + len(properties_to_calculate) + 2  # +2 for centroid_x, centroid_y and centroid_z
        if "centroid" in properties_to_calculate
        else 2 + len(properties_to_calculate)
    )
    assert sdata_pixie_intensities.tables[table_name].obs.shape == (669, shape_obs_axis_1)

    for _prop in properties_to_calculate:
        if _prop == "centroid":
            assert (
                f"{_prop}_x" in sdata_pixie_intensities[table_name].obs.columns
                and f"{_prop}_y" in sdata_pixie_intensities[table_name].obs.columns
            )
        else:
            assert _prop in sdata_pixie_intensities[table_name].obs.columns


def test_add_regionprop_features_alias():
    assert add_regionprop_features is add_regionprops
