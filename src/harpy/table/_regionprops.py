from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame
from skimage.measure import moments, regionprops, regionprops_table
from skimage.measure._regionprops import RegionProperties
from spatialdata import SpatialData
from spatialdata.models import TableModel

from harpy.image._image import get_dataarray
from harpy.table._table import add_table_layer
from harpy.utils._keys import _CELL_INDEX, _INSTANCE_KEY
from harpy.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def add_regionprop_features(
    sdata: SpatialData,
    labels_layer: str,
    table_layer: str,
):
    """
    Enhances a SpatialData object with region property features calculated from the specified labels layer, updating its table attribute (`sdata.tables[table_layer]`)..

    This function computes various geometric and morphological properties for each instance
    found in the specified labels layer of the SpatialData object. These properties include the following measures:

        - **area**
        - **eccentricity**
        - **major_axis_length**
        - **minor_axis_length**
        - **perimeter**
        - **centroid**
        - **convex_area**
        - **equivalent_diameter**
        - **convex_area**
        - **equivalent_diameter**

    These features are added to the `.obs` attribute of the :class:`anndata.AnnData` table at slot `table_layer`.

    Parameters
    ----------
    sdata
        The SpatialData object.
    labels_layer
        The name of the layer in `sdata` that contains the labeled regions, typically derived from a segmentation
        process. Each distinct label corresponds to a different instance, and properties will be calculated for these
        labeled regions.
    table_layer
        The table layer in `sdata.tables` to which the features will be added.

    Returns
    -------
    The original SpatialData object, updated to include a range of new region-specific property measurements
    in the `.obs` attribute of the :class:`anndata.AnnData` table at slot `table_layer`

    Notes
    -----
    - The function operates by pulling the required data (masks) into memory for processing, as the underlying 'skimage.measure.regionprops'
      functionality does not support lazy loading. Consequently, sufficient memory must be available for large datasets.
    - Computed properties are merged (using the instance and region keys in `sdata.tables[table_layer].obs`) with the existing observations within the SpatialData's table (`sdata.tables[table_layer].obs`).

    Example
    -------
    >>> sdata = harpy.im.align_labels_layers(
    ...     sdata,
    ...     labels_layer_1="masks_nuclear",
    ...     labels_layer_2="masks_whole",
    ...     output_labels_layer="masks_nuclear_aligned",
    ...     output_shapes_layer=None,
    ...     overwrite=True,
    ...     chunks=256,
    ...     depth=100,
    ... )
    >>>
    >>> sdata = harpy.tb.allocate_intensity(
    ...     sdata, img_layer="raw_image", labels_layer="masks_whole", output_layer="table_intensities", chunks=100
    ... )
    >>>
    >>> sdata = harpy.tb.allocate_intensity(
    ...     sdata, img_layer="raw_image", labels_layer="masks_nuclear_aligned", output_layer="table_intensities", chunks=100, append=True
    ... )
    >>>
    >>> sdata = harpy.tb.add_regionprop_features(
    ...     sdata, labels_layer="masks_whole", table_layer="table_intensities",
    ... )
    >>>
    >>> sdata = harpy.tb.add_regionprop_features(
    ...     sdata, labels_layer="masks_nuclear_aligned", table_layer="table_intensities",
    ... )
    """
    se = get_dataarray(sdata, layer=labels_layer)
    region_key = sdata.tables[table_layer].uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY]
    instance_key = sdata.tables[table_layer].uns[TableModel.ATTRS_KEY][TableModel.INSTANCE_KEY]
    cell_index_name = sdata.tables[table_layer].obs.index.name

    # pull masks in in memory. skimage.measure.regionprops does not work with lazy objects.
    masks = se.data.compute()

    cell_props = _calculate_regionprop_features(masks, instance_key=instance_key, cell_index_name=cell_index_name)

    assert instance_key in cell_props.columns, f"'cell_props' should contain '{instance_key}' column"
    assert region_key not in cell_props.columns, f"'cell_props' should not contain '{region_key}' columns."
    assert region_key in sdata.tables[table_layer].obs, (
        f"Please link observation to a labels_layer using the '{region_key}' column in 'sdata.tables[{table_layer}].obs'"
    )
    assert instance_key in sdata.tables[table_layer].obs, (
        f"Please add unique {instance_key} (uint) for every observation in 'sdata.tables[{table_layer}]', e.g. see 'harpy.table.allocate_intensity'."
    )

    cell_props[region_key] = pd.Categorical([labels_layer] * len(cell_props))

    extra_cells = sum(~cell_props[instance_key].isin(sdata.tables[table_layer].obs[instance_key].values))
    if extra_cells:
        log.warning(
            f"Calculated properties of {extra_cells} cells/nuclei obtained from labels layer '{labels_layer}' "
            f"will not be added to 'sdata.tables[{table_layer}].obs', because their '{instance_key}' are not in 'sdata.tables[{table_layer}].obs.{instance_key}'. "
            "Please first append their intensities with the 'allocate_intensity' function "
            "if they should be included."
        )

    # sanity check (check that _INSTANCE_KEY unique for given labels_layer, otherwise unexpected behaviour when merging)
    for _df in [sdata.tables[table_layer].obs, cell_props]:
        assert not _df[_df[region_key] == labels_layer][instance_key].duplicated().any(), (
            f"{instance_key} should be unique for given '{region_key}'"
        )

    # make copy, otherwise we would update inplace, which could give issues if we are not allowed to overwrite in the next step
    # (i.e. disagreement between on-disk `sdata` and in memory `sdata``.)
    adata = sdata.tables[table_layer].copy()  # TODO think about this, maybe copy not necessary, as we always overwrite
    adata.obs.reset_index(inplace=True)
    adata.obs = adata.obs.merge(cell_props, on=[region_key, instance_key], how="left", suffixes=("", "_y"))
    adata.obs[region_key] = adata.obs[region_key].astype("category")

    for _column_name in adata.obs.columns:
        if _column_name in [region_key, instance_key, cell_index_name]:
            continue
        if f"{_column_name}_y" in adata.obs.columns:
            adata.obs[_column_name] = adata.obs[f"{_column_name}_y"].fillna(adata.obs[_column_name])
            adata.obs.drop(columns=f"{_column_name}_y", inplace=True)

    adata.obs.set_index(cell_index_name, inplace=True, drop=True)

    sdata = add_table_layer(
        sdata,
        adata=adata,
        output_layer=table_layer,
        region=adata.obs[region_key].cat.categories.to_list(),
        instance_key=instance_key,
        region_key=region_key,
        overwrite=True,  # always overwrite, because we only add a .obs features.
    )

    return sdata


def _calculate_regionprop_features(
    masks: NDArray,
    instance_key: str = _INSTANCE_KEY,
    cell_index_name: str = _CELL_INDEX,
) -> DataFrame:
    properties = [
        "label",
        "area",
        "eccentricity",
        "major_axis_length",
        "minor_axis_length",
        "perimeter",
        "centroid",
        "convex_area",
        "equivalent_diameter",
    ]

    cell_props = DataFrame(regionprops_table(masks, properties=properties))

    props = regionprops(masks)

    for _func in [
        _major_minor_axis_ratio,
        _perim_square_over_area,
        _major_axis_equiv_diam_ratio,
        _convex_hull_resid,
        _centroid_dif,
    ]:
        results = []
        for prop in props:
            results.append(_func(prop))
        cell_props[_func.__name__] = results

    _cells_id = cell_props.label.astype(int)
    cell_props[instance_key] = _cells_id
    cell_props = cell_props.set_index("label")
    cell_props.index.name = cell_index_name
    cell_props.index = cell_props.index.map(str)

    return cell_props


# following helper functions taken from
# https://github.com/angelolab/ark-analysis/blob/main/src/ark/segmentation/regionprops_extraction.py


def _major_minor_axis_ratio(prop: RegionProperties) -> float:
    if prop.minor_axis_length == 0:
        return float("NaN")
    else:
        return prop.major_axis_length / prop.minor_axis_length


def _perim_square_over_area(prop: RegionProperties) -> float:
    return np.square(prop.perimeter) / prop.area


def _major_axis_equiv_diam_ratio(prop: RegionProperties) -> float:
    return prop.major_axis_length / prop.equivalent_diameter


def _convex_hull_resid(prop: RegionProperties) -> float:
    return (prop.convex_area - prop.area) / prop.convex_area


def _centroid_dif(prop: RegionProperties) -> float:
    cell_image = prop.image
    cell_M = moments(cell_image)
    cell_centroid = np.array([cell_M[1, 0] / cell_M[0, 0], cell_M[0, 1] / cell_M[0, 0]])

    convex_image = prop.convex_image
    convex_M = moments(convex_image)
    convex_centroid = np.array([convex_M[1, 0] / convex_M[0, 0], convex_M[0, 1] / convex_M[0, 0]])

    centroid_dist = np.linalg.norm(cell_centroid - convex_centroid) / np.sqrt(prop.area)

    return centroid_dist
