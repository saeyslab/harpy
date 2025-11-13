import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon
from spatialdata import SpatialData

from harpy.shape._shape import add_shapes_layer


def filter_by_morphology(
    sdata: SpatialData,
    shapes_layer: str,
    output_shapes_layer: str,
    filters: dict[str, tuple] | None = None,
    keep_unsupported: bool = False,
    calculate_all_features: bool = False,
    calculate_all_features_grouped: bool = False,
    shape_names_column: str = "name",
    pixel_size_um: float = 1.0,
    overwrite: bool = True,
):
    """
    Filter polygons in a SpatialData shapes layer based on morphological features.
    It is recommended to run `hp.sh.prep_region_annotations` first when filtering region annotations that
    may contain multipolygons, unnamed annotations, etc.

    Parameters
    ----------
    sdata
        SpatialData object containing the input shapes layer.
    shapes_layer
        Name of the shapes layer in `sdata.shapes` containing polygons.
    output_shapes_layer
        Name of the shapes layer to store the filtered polygons.
    filters
        Dictionary specifying filtering thresholds.
        Each entry should be of the form:
            `{ "feature_name": (min_value, max_value) }`
        where `None` can be used to skip a bound, e.g.:
            `{ "area": (50, 200), "convexity": (None, 0.8) }`

        Supported features:
        - `"area"`: Area of the polygon (px²).
            → Use to filter out very small or very large polygons.
        - `"perimeter"`: Perimeter length (px).
            → Use to detect irregular boundaries or fragmented shapes.
        - `"circularity"`: 4π * area / perimeter².
            → 1 for a perfect circle; lower = irregular shape.
        - `"compactness"`: perimeter² / area.
            → Shape compactness measure.
        - `"convex_area"`: Area of convex hull (px²).
            → Useful with solidity and convexity.
        - `"solidity"`: area / convex_area.
            → Low values mean concave or fragmented shapes.
        - `"convexity"`: convex_perimeter / perimeter.
            → 1 for perfectly convex shapes. Lower for rough or spiky boundaries.
        - `"centroid_dif"`: Distance between polygon and convex hull centroids normalized by the polygon area.
            → Captures off-centered concavity or asymmetry.
        - `"major_axis_length"`: Length of the longest side of the minimum rotated bounding box (px).
            → Use to filter very long or very short shapes.
        - `"minor_axis_length"`: Length of the shortest side of the minimum rotated bounding box (px).
            → Use to filter very wide or very skinny shapes.
        - `"major_minor_axis_ratio"`: major_axis_length / minor_axis_length.
            → High values indicate elongated polygons; ~1 for round shapes.

        You can use grouped filters by suffixing features with `-grouped`.
        Grouped filters merge polygons sharing the same name in `shape_names_column` before computing morphological features and
        can be used interchangibly with regular filters.

    keep_unsupported
        Only Polygon and MultiPolygon are supported. Set keep_unsupported to True to skip any unssupported geometries types (e.g. Point), but
        keep them in the output_shapes_layer. Set to False to remove them from output_shapes_layer.
    calculate_all_features
        If True, computes all supported morphological features regardless of which ones are used in `filters` and saves them to `output_shapes_layer`.
        If False, only computes and saves morphological features needed for `filters`.
    calculate_all_features_grouped
        If True, computes all morphological features for merged group geometries (by {shape_names_column}), regardless
        of which ones are used in grouped filters.
    shape_names_column
        Column name in shapes layer containing geometry names. Required when using grouped filters.

    pixel_size_um
        Scale factor to convert geometric measurements from pixel units to microns.
        - Applied to:
            * "area" and "convex_area" → scaled by (pixel_size_um)²
            * "perimeter", "major_axis_length", and "minor_axis_length" → scaled by (pixel_size_um)
        - Dimensionless ratios (e.g., "circularity", "compactness", "solidity", "convexity", "major_minor_axis_ratio")
            are unaffected by scaling but are computed using the scaled geometric quantities.
        Defaults to 1.0 (no scaling, i.e. units remain in pixels).
        Note that this affects the min/max values that need to be specified in `filters`.
    overwrite
        Whether to overwrite an existing shapes layer.

    Returns
    -------
    SpatialData object with updated shapes layer containing only the filtered polygons.
    """
    # Get filters and split into individual and grouped filters
    filters = filters or {}

    grouped_filters = {k.replace("-grouped", ""): v for k, v in filters.items() if k.endswith("-grouped")}
    individual_filters = {k: v for k, v in filters.items() if not k.endswith("-grouped")}

    required_individual = set(individual_filters.keys())
    required_grouped = set(grouped_filters.keys())

    if calculate_all_features:
        required_individual.update(
            [
                "area",
                "perimeter",
                "convex_area",
                "circularity",
                "compactness",
                "solidity",
                "convexity",
                "centroid_dif",
                "major_axis_length",
                "minor_axis_length",
                "major_minor_axis_ratio",
            ]
        )
        print("Calculating all supported morphological features (per polygon).")

    if calculate_all_features_grouped:
        required_grouped.update(
            [
                "area",
                "perimeter",
                "convex_area",
                "circularity",
                "compactness",
                "solidity",
                "convexity",
                "centroid_dif",
                "major_axis_length",
                "minor_axis_length",
                "major_minor_axis_ratio",
            ]
        )
        print("Calculating all supported morphological features (grouped).")

    # Create copy of shapes layer
    gdf = sdata.shapes[shapes_layer].copy()

    # Filter out geometries that are not Polygon or MultiPolygon
    supported_mask = gdf.geometry.apply(lambda x: isinstance(x, (Polygon, MultiPolygon)))

    unssuported = len(gdf) - supported_mask.sum()
    if unssuported > 0:
        print(f"Found {unssuported} non-polygon geometries in {shapes_layer}.")
    if unssuported == len(gdf):
        print("No supported geometries (Polygon, MultiPolygon) found. Exiting...")
        return sdata

    if keep_unsupported:
        gdf_skipped = gdf[~supported_mask].copy()
    else:
        gdf_skipped = gpd.GeoDataFrame(columns=gdf.columns, geometry=[])

    gdf = gdf[supported_mask].copy()

    # Dissolve polygons by name
    grouped_gdf = None
    if grouped_filters or calculate_all_features_grouped:
        if shape_names_column not in gdf.columns:
            raise ValueError("Grouped filters require a 'name' column in the shapes layer.")

        grouped_gdf = gdf.dissolve(by=shape_names_column, as_index=False, aggfunc="first").copy()
        grouped_gdf["geometry"] = gdf.groupby(shape_names_column).geometry.apply(lambda x: x.unary_union).values
        print(f"Polygons merged by {shape_names_column}, resulting in {len(grouped_gdf)} groups.")

    # Compute metrics
    def _compute_morphological_features(gdf, required, pixel_size_um, suffix):
        if any(filter in required for filter in ["area", "circularity", "compactness", "solidity", "centroid_dif"]):
            gdf[f"area{suffix}"] = gdf.geometry.area * pixel_size_um**2

        if any(filter in required for filter in ["perimeter", "circularity", "compactness", "convexity"]):
            gdf[f"perimeter{suffix}"] = gdf.geometry.length * pixel_size_um

        if any(filter in required for filter in ["convex_area", "solidity"]):
            gdf[f"convex_area{suffix}"] = gdf.geometry.convex_hull.area * pixel_size_um**2

        if "circularity" in required:
            gdf[f"circularity{suffix}"] = 4 * np.pi * gdf[f"area{suffix}"] / (gdf[f"perimeter{suffix}"] ** 2)

        if "compactness" in required:
            gdf[f"compactness{suffix}"] = (gdf[f"perimeter{suffix}"] ** 2) / gdf[f"area{suffix}"]

        if "solidity" in required:
            gdf[f"solidity{suffix}"] = gdf[f"area{suffix}"] / gdf[f"convex_area{suffix}"]

        if "centroid_dif" in required:
            gdf[f"centroid_x{suffix}"] = gdf.geometry.centroid.x
            gdf[f"centroid_y{suffix}"] = gdf.geometry.centroid.y
            hull_centroids = gdf.geometry.convex_hull.centroid
            gdf[f"centroid_dif{suffix}"] = np.sqrt(
                (gdf[f"centroid_x{suffix}"] - hull_centroids.x) ** 2
                + (gdf[f"centroid_y{suffix}"] - hull_centroids.y) ** 2
            ) / np.sqrt(gdf[f"area{suffix}"])

        if "convexity" in required:
            gdf[f"convexity{suffix}"] = (gdf.geometry.convex_hull.length * pixel_size_um) / gdf[f"perimeter{suffix}"]

        # Get bounding box lengths from minimum rotated rectangle
        if any(filter in required for filter in ["major_axis_length", "minor_axis_length", "major_minor_axis_ratio"]):

            def _rotated_rect_axes(geom):
                try:
                    rect = geom.minimum_rotated_rectangle
                    x, y = rect.exterior.coords.xy
                    # Compute edge lengths
                    edges = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
                    edges = np.sort(edges[:-1])  # drop duplicate closing edge
                    minor, major = edges[0], edges[1]
                    ratio = major / minor if minor > 0 else np.nan
                    return major, minor, ratio
                except Exception:
                    return np.nan, np.nan, np.nan

            results = np.array([_rotated_rect_axes(g) for g in gdf.geometry])
            major_axis_length, minor_axis_length, major_minor_axis_ratio = results.T
            if any(filter in required for filter in ["major_axis_length", "major_minor_axis_ratio"]):
                gdf[f"major_axis_length{suffix}"] = major_axis_length * pixel_size_um
            if any(filter in required for filter in ["minor_axis_length", "major_minor_axis_ratio"]):
                gdf[f"minor_axis_length{suffix}"] = minor_axis_length * pixel_size_um
            if "major_minor_axis_ratio" in required:
                gdf[f"major_minor_axis_ratio{suffix}"] = major_minor_axis_ratio

        return gdf

    gdf = _compute_morphological_features(gdf, required_individual, pixel_size_um, "")
    grouped_gdf = _compute_morphological_features(grouped_gdf, required_grouped, pixel_size_um, "-grouped")

    # Merge grouped metrics back into main gdf
    if grouped_gdf is not None and not grouped_gdf.empty:
        grouped_metrics = [col for col in grouped_gdf.columns if col.endswith("-grouped")]
        gdf = gdf.merge(grouped_gdf[[shape_names_column] + grouped_metrics], on=shape_names_column, how="left")

    # Apply filters
    print(f"Applying morphological filter(s) on {len(gdf)} polygons...")
    mask = np.ones(len(gdf), dtype=bool)
    if filters:
        for feature, (min_val, max_val) in filters.items():
            print(f"\nFiltering by '{feature}': {min_val} ≤ value ≤ {max_val}")
            if feature not in gdf.columns:
                raise KeyError(f"Feature '{feature}' was not computed. Check your spelling or supported feature list.")

            # Apply lower bound
            if min_val is not None:
                to_remove = (gdf[feature] < min_val) & mask
                removed_low = to_remove.sum()
                mask &= gdf[feature] >= min_val
                print(f"  - Removed {removed_low} polygons with {feature} ≤ {min_val}")

            # Apply upper bound
            if max_val is not None:
                to_remove = (gdf[feature] > max_val) & mask
                removed_high = to_remove.sum()
                mask &= gdf[feature] <= max_val
                print(f"  - Removed {removed_high} polygons with {feature} ≥ {max_val}")

            remaining = mask.sum()
            print(f"  → Remaining after filtering '{feature}': {remaining} polygons")

    filtered_gdf = gdf[mask]
    print(f"\nKept {len(filtered_gdf)} / {len(gdf)} polygons after morphological filters.")

    # Add filtered shapes layer
    input_gdf = pd.concat([filtered_gdf, gdf_skipped], ignore_index=False)
    sdata = add_shapes_layer(
        sdata,
        input=input_gdf,
        output_layer=output_shapes_layer,
        overwrite=overwrite,
    )

    return sdata


def filter_by_shapes(
    sdata: SpatialData,
    target_shapes_layer: str,
    mask_shapes_layer: str,
    output_shapes_layer: str,
    shape_names_column: str | None = None,
    shape_names: str | list[str] | None = None,
    keep_intersecting: bool = True,
    overwrite: bool = False,
):
    """
    Filter polygons in a target shapes layer (typically containg segmention boundaries) based on intersection with polygons in a mask layer
    (typically containing region annotations).

    Parameters
    ----------
    sdata
        SpatialData object containing the target and mask shapes layers.
    target_shapes_layer
        Name of shapes layer whose polygons will be filtered.
    mask_shapes_layer
        Name of shapes layer used as mask for filtering.
    output_shapes_layer
        Name of the output shapes layer to store filtered polygons.
    shape_names_column
        Optional column in mask_shapes_layer to select specific polygons.
    shape_names
        Name or list of names of polygons in mask_shapes_layer to use for filtering. Ignored if shape_names_column is None.
    keep_intersecting
        If True, keeps polygons that intersect the mask.
        If False, removes polygons that intersect the mask.
    overwrite
        If True, overwrites the output shapes layer if it exists.

    Returns
    -------
    SpatialData object with filtered shapes layer.
    """
    # Copy target layer
    target_gdf = sdata.shapes[target_shapes_layer].copy()

    # Copy mask layer
    mask_gdf = sdata.shapes[mask_shapes_layer].copy()

    # Optionally select subset of mask polygons
    if shape_names_column is not None and shape_names is not None:
        if shape_names_column not in mask_gdf.columns:
            raise ValueError(f"Column '{shape_names_column}' not found in mask layer '{mask_shapes_layer}'.")
        if isinstance(shape_names, str):
            shape_names = [shape_names]
        mask_gdf = mask_gdf[mask_gdf[shape_names_column].isin(shape_names)].copy()
        if mask_gdf.empty:
            raise ValueError(f"No geometries found in '{mask_shapes_layer}' matching {shape_names}.")

    # Build mask union
    mask_union = mask_gdf.unary_union

    # Compute intersection boolean
    target_gdf["intersects_mask"] = target_gdf.geometry.intersects(mask_union)

    # Filter depending on mode
    if keep_intersecting:
        filtered_gdf = target_gdf[target_gdf["intersects_mask"]].copy()
    else:
        filtered_gdf = target_gdf[~target_gdf["intersects_mask"]].copy()

    filtered_gdf.drop(columns=["intersects_mask"], inplace=True)
    removed = len(target_gdf) - len(filtered_gdf)

    if keep_intersecting:
        print(
            f"Kept {len(filtered_gdf)} / {len(target_gdf)} geometries intersecting '{mask_shapes_layer}' (removed {removed})."
        )
    else:
        print(
            f"Removed {removed} / {len(target_gdf)} geometries intersecting '{mask_shapes_layer}' (kept {len(filtered_gdf)})."
        )

    # Add to SpatialData
    sdata = add_shapes_layer(
        sdata,
        input=filtered_gdf,
        output_layer=output_shapes_layer,
        overwrite=overwrite,
    )

    return sdata
