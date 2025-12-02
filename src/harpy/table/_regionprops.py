from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger as log
from numpy.typing import NDArray
from pandas import DataFrame
from skimage.measure import moments, regionprops, regionprops_table
from skimage.measure._regionprops import RegionProperties
from spatialdata import SpatialData
from spatialdata.models import TableModel

from harpy.image._image import get_dataarray
from harpy.table._table import ProcessTable, add_table_layer
from harpy.utils._keys import _INSTANCE_KEY
from harpy.utils.utils import _make_list


def add_regionprop_features(
    sdata: SpatialData,
    labels_layer: str | list[str],
    table_layer: str,
    output_layer: str,
    properties: str | tuple[str] = (
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
    ),
    overwrite: bool = False,
):
    """
    Calculates region property features from the specified labels layer, and adds the results to the :class:`~anndata.AnnData` object that annotates the labels layer.

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
        - **major_minor_axis_ratio**
        - **perim_square_over_area**
        - **major_axis_equiv_diam_ratio**
        - **convex_hull_resid**
        - **centroid_dif**

    These features are added to the `.obs` attribute of the :class:`anndata.AnnData` table at slot `output_layer`.

    Note that calculation of **perimeter** and **perim_square_over_area** is not supported for 3D labels layers.

    Parameters
    ----------
    sdata
        The SpatialData object.
    labels_layer
        The name of the layer in `sdata` that contains the labeled regions, typically derived from a segmentation
        process. Each distinct label corresponds to a different instance, and properties will be calculated for these
        labeled regions.
    table_layer
        The table layer in `sdata.tables` that annotates the labels layer, and to which the calculated features will be added.
    output_layer
        Output table layer.
    properties
        The properties to calculate.
    overwrite
        If True, overwrites the output layer if it already exists in `sdata`.

    Returns
    -------
    The updated `sdata` object.

    Notes
    -----
    The function operates by pulling the required labels layers (masks) into memory for processing, as the underlying :func:'skimage.measure.regionprops'
    functionality does not support lazy loading. Consequently, sufficient memory must be available for large datasets.

    Example
    -------
    >>> import harpy as hp
    >>>
    >>> sdata = hp.datasets.pixie_example()
    >>>
    >>> sdata = hp.tb.allocate_intensity(
    ...     sdata,
    ...     img_layer="raw_image_fov0",
    ...     labels_layer="label_whole_fov0",
    ...     to_coordinate_system="fov0",
    ...     mode="sum",
    ...     output_layer="table_intensities",
    ...     overwrite=True,
    ... )
    >>>
    >>> sdata = hp.tb.allocate_intensity(
    ...     sdata,
    ...     img_layer="raw_image_fov1",
    ...     labels_layer="label_whole_fov1",
    ...     to_coordinate_system="fov1",
    ...     mode="sum",
    ...     output_layer="table_intensities",
    ...     append=True,
    ...     overwrite=True,
    ... )
    >>>
    >>> sdata = hp.tb.add_regionprop_features(
    ...     sdata,
    ...     labels_layer=["label_whole_fov0", "label_whole_fov1"],
    ...     table_layer="table_intensities",
    ...     output_layer="table_intensities",
    ...     properties=["perimeter", "equivalent_diameter"],
    ...     overwrite=True,
    ... )
    """
    if labels_layer is None:
        labels_layer = sdata[table_layer].uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY]
    labels_layer = _make_list(labels_layer)
    properties = _make_list(properties)
    table_processor = ProcessTable(sdata, labels_layer=labels_layer, table_layer=table_layer)
    adata = table_processor._get_adata()
    instance_key = table_processor.instance_key
    region_key = table_processor.region_key

    new_properties = []
    for _property in properties:
        if _property in adata.obs.columns or (
            _property == "centroid"
            and (
                "centroid_x" in adata.obs.columns
                or "centroid_y" in adata.obs.columns
                or "centroid_z" in adata.obs.columns
            )
        ):
            log.warning(
                f"Cell property '{_property}' already exists in '.obs' attribute of table layer '{table_layer}'. "
                "Skipping recomputation. Remove the column to trigger recalculation."
            )
        else:
            new_properties.append(_property)
    properties = tuple(new_properties)

    cell_props = []
    for _labels_layer in labels_layer:
        se = get_dataarray(sdata, layer=_labels_layer)
        # pull masks in in memory. skimage.measure.regionprops does not work with lazy objects.
        masks = se.data.compute()
        _cell_props = _calculate_regionprop_features(
            masks,
            properties=properties,
            instance_key=instance_key,
        )
        assert instance_key in _cell_props.columns, f"'cell_props' should contain '{instance_key}' column"
        assert region_key not in _cell_props.columns, f"'cell_props' should not contain '{region_key}' columns."
        if _labels_layer not in adata.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY]:
            raise ValueError(
                f"The labels layer '{_labels_layer}' does not seem to be annotated by the table layer {table_layer}."
            )

        _cell_props[region_key] = pd.Categorical([_labels_layer] * len(_cell_props))
        cell_props.append(_cell_props)
    cell_props = pd.concat(cell_props)

    # Sanity checks before merging cell_props and adata.obs.
    for _labels_layer in labels_layer:
        _cell_props = cell_props[cell_props[region_key] == _labels_layer]
        adata_view = adata[adata.obs[region_key] == _labels_layer]
        # 1) check that all cells in adata_view.obs could be matched to _cell_props
        missing = set(adata_view.obs[instance_key]) - set(_cell_props[instance_key])
        if missing:
            raise ValueError(
                f"{len(missing)} '{instance_key}' values in .obs not found in calculated cell properties (e.g. {list(missing)[:5]}). Please report this bug."
            )
        # 2) check that there are no duplicates in cell_props. If there would be duplicates in cell_props, then rows in adata.obs would be multiplied, we want to catch this early
        duplicates = _cell_props[_cell_props.duplicated(subset=instance_key, keep=False)]
        if not duplicates.empty:
            raise ValueError(
                f"Duplicate {instance_key} values found in calculated cell properties (e.g. {duplicates[instance_key].unique()[:5]}). Please report this bug."
            )
        # 3) log a warning if some cells in cell props could not be found in adata_view; this could happen if adata was filtered befote calling this function.
        extra_cells = sum(~_cell_props[instance_key].isin(adata_view.obs[instance_key].values))
        if extra_cells:
            log.warning(
                f"Calculated properties of '{extra_cells}' instances obtained from labels layer '{labels_layer}' "
                f"will not be added to 'sdata.tables[{table_layer}].obs', because their instance IDs ('{instance_key}') are not in 'sdata.tables[{table_layer}].obs.[{instance_key}]'. "
                "This can happen if some instances in the table were filtered prior to calling this function."
                "If they should be included, then first append their intensities with the 'harpy.tb.allocate_intensity' function."
            )
    cell_props[region_key] = cell_props[region_key].astype(
        "category"
    )  # concatenating results in region key no longer categorical
    # NOTE: we already checked that instance key is unique for given labels layer, in the get_adata() function.
    old_index = adata.obs.index
    old_index_name = adata.obs.index.name
    adata.obs = pd.merge(
        adata.obs,
        cell_props,
        how="inner",
        on=[instance_key, region_key],
    )
    adata.obs.index = old_index
    adata.obs.index.name = old_index_name

    sdata = add_table_layer(
        sdata,
        adata=adata,
        output_layer=output_layer,
        region=adata.obs[region_key].cat.categories.to_list(),
        instance_key=instance_key,
        region_key=region_key,
        overwrite=overwrite,
    )
    return sdata


def _calculate_regionprop_features(
    masks: NDArray,
    properties: tuple[str],
    instance_key: str = _INSTANCE_KEY,
) -> DataFrame:
    supported_properties = [
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

    for _property in properties:
        if _property not in supported_properties:
            raise ValueError(
                f"Cell property {_property} is not supported. Please choose properties from the following list: '{supported_properties}'."
            )
    properties = list(properties)
    properties.append("label")  # for the label ID

    properties_own_implementation = {
        "major_minor_axis_ratio": _major_minor_axis_ratio,
        "perim_square_over_area": _perim_square_over_area,
        "major_axis_equiv_diam_ratio": _major_axis_equiv_diam_ratio,
        "convex_hull_resid": _convex_hull_resid,
        "centroid_dif": _centroid_dif,
    }

    cell_props = DataFrame(
        regionprops_table(
            masks,
            properties=[_property for _property in properties if _property not in properties_own_implementation.keys()],
        )
    )
    props = regionprops(masks)

    for key, _func in properties_own_implementation.items():
        if key not in properties:
            # case where we should not calculate the property
            continue
        results = []
        for prop in props:
            results.append(_func(prop))
        cell_props[key] = results

    _cells_id = cell_props.label.astype(int)
    cell_props[instance_key] = _cells_id
    cell_props = cell_props.set_index("label")
    cell_props.index = cell_props.index.map(int)

    # rename centroid, if calculated
    if "centroid-0" in cell_props.columns and "centroid-1" in cell_props.columns:
        if "centroid-2" in cell_props.columns:
            cell_props.rename(
                columns={"centroid-0": "centroid_z", "centroid-1": "centroid_y", "centroid-2": "centroid_x"},
                inplace=True,
            )
        else:
            cell_props.rename(
                columns={"centroid-0": "centroid_y", "centroid-1": "centroid_x"},
                inplace=True,
            )

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
