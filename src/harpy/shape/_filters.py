from collections.abc import Sequence
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import unary_union
from spatialdata import SpatialData
from spatialdata.transformations import get_transformation

from harpy.shape._shape import add_shapes_layer
from harpy.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


# Extract exterior and interior boundaries
def _extract_exterior_lines(geom):
    if isinstance(geom, Polygon):
        return [geom.exterior]
    elif isinstance(geom, MultiPolygon):
        return [poly.exterior for poly in geom.geoms]
    return []


def _extract_interior_lines(geom):
    if isinstance(geom, Polygon):
        return list(geom.interiors)
    elif isinstance(geom, MultiPolygon):
        lines = []
        for poly in geom.geoms:
            lines.extend(poly.interiors)
        return lines
    return []


def morphological_features(
    sdata: SpatialData,
    shapes_layer: str,
    output_shapes_layer: str,
    shape_names_column: str = "name",
    grouped_features: bool = False,
    pixel_size_um: float = 1.0,
    overwrite: bool = False,
) -> SpatialData:
    """
    Compute morphological features for polygons in a shapes layer. Only Polygon and MultiPolygon are supported.
    It is recommended to run `hp.sh.prep_region_annotations` first when computing features for region annotations that
    may contain multipolygons, point geometries, unnamed annotations, etc.

        - `area`: Area of the polygon (px² or µm²).
        - `perimeter`: Perimeter length (px or µm).
        - `equivalent_diameter`: Diameter of a circle with the same area (px or µm).
        - `convex_area`: Area of convex hull (px² or µm²).
        - `convex_perimeter`: Perimeter of convex hull (px or µm).
        - `circularity`: 4π * area / perimeter².
            → 1 for a perfect circle; lower = irregular shape.
        - `compactness`: perimeter² / area.
        - `solidity`: area / convex_area.
            → Low values mean concave or fragmented shapes.
        - `convexity`: convex_perimeter / perimeter.
            → 1 for perfectly convex shapes. Lower for rough or spiky boundaries.
        - `centroid_x`: X-coordinate of the polygon centroid (px).
        - `centroid_y`: Y-coordinate of the polygon centroid (px).
        - `centroid_dif`: Distance between polygon and convex hull centroids normalized by the polygon area.
            → Captures off-centered concavity or asymmetry.
        - `major_axis_length`: Length of the longest side of the minimum rotated bounding box (px or µm).
        - `minor_axis_length`: Length of the shortest side of the minimum rotated bounding box (px or µm).
        - `major_minor_axis_ratio`: major_axis_length / minor_axis_length.
        - `num_vertices`: Number of vertices along exterior boundaries.
        - `boundary_complexity`: num_vertices / perimeter.
            → Normalized measure of boundary irregularity.
        - `num_holes`: Count of internal holes.
        - `hole_area`: Area covered by internal holes (px² or µm²)

    Parameters
    ----------
    sdata
        SpatialData object containing the shapes layer.
    shapes_layer
        Name of the input shapes layer.
    output_shapes_layer
        Name of the output shapes layer to store results.
    shape_names_column
        Column name in shapes layer containing geometry names. Required when using grouped filters.
    grouped_features
        If True, also computes grouped morphological features by dissolving polygons sharing
        the same name in `shape_names_column`.
    pixel_size_um
        Scale factor to convert geometric measurements from pixel to micron units.
        - Applied to:
            * "perimeter", "major_axis_length", and "minor_axis_length" → scaled by (pixel_size_um)
            * "area" and "convex_area" → scaled by (pixel_size_um)²
        - Dimensionless ratios (e.g., "circularity", "compactness", "solidity", "convexity", "centroid_dif", "major_minor_axis_ratio")
            are unaffected by scaling but are computed using the scaled geometric quantities.
        - XY-coordinates are kept in original units.
        Defaults to 1.0 (no scaling, i.e. units remain in pixels).
    overwrite
        Whether to overwrite existing shapes layer.

    Returns
    -------
    SpatialData
        Object with updated shapes layer containing morphological feature columns.
    """
    # Create copy of shapes layer
    gdf = sdata.shapes[shapes_layer].copy()

    # Filter out geometries that are not Polygon or MultiPolygon
    supported_mask = gdf.geometry.apply(lambda x: isinstance(x, (Polygon, MultiPolygon)))

    unssuported = len(gdf) - supported_mask.sum()
    if unssuported > 0:
        raise ValueError(
            f"Found {unssuported} non-polygon geometries in {shapes_layer}. Consider running `hp.sh.prep_region_annotations` first to clean up geometries."
        )
    if unssuported == len(gdf):
        log.warning(
            f"No supported geometries (Polygon, MultiPolygon) found. Skipping addition of shapes layer '{output_shapes_layer}' to sdata."
        )
        return sdata

    gdf = gdf[supported_mask].copy()

    # Multipolygon check
    n_multipolygons = gdf.geometry.apply(lambda x: isinstance(x, MultiPolygon)).sum()
    if n_multipolygons > 0:
        log.warning(
            f"Detected {n_multipolygons} MultiPolygon geometries in '{shapes_layer}'. "
            "Consider running `hp.sh.prep_region_annotations` first to split MultiPolygons before feature extraction."
        )

    # Dissolve polygons by name
    grouped_gdf = None
    if grouped_features:
        if shape_names_column not in gdf.columns:
            raise ValueError(f"Grouped computation requires column '{shape_names_column}'.")
        grouped_gdf = gdf.dissolve(by=shape_names_column, as_index=False, aggfunc="first").copy()
        grouped_gdf["geometry"] = gdf.groupby(shape_names_column).geometry.apply(lambda x: x.unary_union).values
        log.info(f"Merged {len(gdf)} polygons into {len(grouped_gdf)} groups by '{shape_names_column}'.")

    # Compute features
    def _compute_features(gdf: gpd.GeoDataFrame, pixel_size_um, suffix: str = ""):
        # Base area & perimeter
        gdf[f"area{suffix}"] = gdf.geometry.area * pixel_size_um**2
        gdf[f"perimeter{suffix}"] = gdf.geometry.length * pixel_size_um

        # Equivalent diameter
        gdf[f"equivalent_diameter{suffix}"] = np.sqrt(4 * gdf[f"area{suffix}"] / np.pi)

        # Convex geometry
        convex_hulls = gdf.geometry.convex_hull
        gdf[f"convex_area{suffix}"] = convex_hulls.area * pixel_size_um**2
        gdf[f"convex_perimeter{suffix}"] = convex_hulls.length * pixel_size_um

        # Derived metrics
        gdf[f"circularity{suffix}"] = 4 * np.pi * gdf[f"area{suffix}"] / (gdf[f"perimeter{suffix}"] ** 2)
        gdf[f"compactness{suffix}"] = (gdf[f"perimeter{suffix}"] ** 2) / gdf[f"area{suffix}"]
        gdf[f"solidity{suffix}"] = gdf[f"area{suffix}"] / gdf[f"convex_area{suffix}"]
        gdf[f"convexity{suffix}"] = gdf[f"convex_perimeter{suffix}"] / gdf[f"perimeter{suffix}"]

        # Centroid-based metrics
        gdf[f"centroid_x{suffix}"] = gdf.geometry.centroid.x
        gdf[f"centroid_y{suffix}"] = gdf.geometry.centroid.y
        hull_centroids = convex_hulls.centroid
        gdf[f"centroid_dif{suffix}"] = np.sqrt(
            (gdf[f"centroid_x{suffix}"] - hull_centroids.x) ** 2 + (gdf[f"centroid_y{suffix}"] - hull_centroids.y) ** 2
        ) / np.sqrt(gdf[f"area{suffix}"])

        # Major/minor axis (from rotated rectangle)
        def _rotated_rect_axes(geom):
            try:
                rect = geom.minimum_rotated_rectangle
                x, y = rect.exterior.coords.xy
                edges = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
                edges = np.sort(edges[:-1])  # remove duplicate closing edge
                minor, major = edges[0], edges[1]
                ratio = major / minor if minor > 0 else np.nan
                return major, minor, ratio
            except Exception:
                return np.nan, np.nan, np.nan

        results = np.array([_rotated_rect_axes(g) for g in gdf.geometry])
        major_axis_length, minor_axis_length, major_minor_axis_ratio = results.T
        gdf[f"major_axis_length{suffix}"] = major_axis_length * pixel_size_um
        gdf[f"minor_axis_length{suffix}"] = minor_axis_length * pixel_size_um
        gdf[f"major_minor_axis_ratio{suffix}"] = major_minor_axis_ratio

        # Vertex metrics
        def _num_vertices(geom):
            exteriors = _extract_exterior_lines(geom)
            return sum(len(line.coords) for line in exteriors)

        gdf[f"num_vertices{suffix}"] = gdf.geometry.apply(_num_vertices)
        gdf[f"boundary_complexity{suffix}"] = gdf[f"num_vertices{suffix}"] / gdf[f"perimeter{suffix}"]

        # Hole topology
        def _num_holes(geom):
            return len(_extract_interior_lines(geom))

        def _hole_area(g):
            interiors = _extract_interior_lines(g)
            if not interiors:
                return 0.0
            try:
                holes = [Polygon(ring) for ring in interiors]
                total_hole_area = sum(h.area for h in holes)
                return total_hole_area
            except Exception:
                return np.nan

        gdf[f"num_holes{suffix}"] = gdf.geometry.apply(_num_holes)
        gdf[f"hole_area{suffix}"] = gdf.geometry.apply(_hole_area) * pixel_size_um**2

        return gdf

    gdf = _compute_features(gdf, pixel_size_um, "")

    if grouped_features and not grouped_gdf.empty:
        grouped_gdf = _compute_features(grouped_gdf, pixel_size_um, "-grouped")

        grouped_metrics = [col for col in grouped_gdf.columns if col.endswith("-grouped")]

        grouped_lookup = grouped_gdf.set_index(shape_names_column)
        for col in grouped_metrics:
            gdf[col] = gdf[shape_names_column].map(grouped_lookup[col])

    # sanity check. If this sanity check would fail in spatialdata at some point, then pass transformation to transformations parameter of add_shapes_layer.
    assert get_transformation(gdf, get_all=True) == get_transformation(sdata[shapes_layer], get_all=True)

    sdata = add_shapes_layer(
        sdata,
        input=gdf,
        output_layer=output_shapes_layer,
        overwrite=overwrite,
    )

    return sdata


def filter_shapes_numerical(
    sdata: SpatialData,
    shapes_layer: str,
    output_shapes_layer: str,
    numerical_column: str,
    min_value: float | int | None = None,
    max_value: float | int | None = None,
    overwrite: bool = False,
) -> SpatialData:
    """
    All polygons with a size outside of the min and max size range are removed using the `numerical_column` in `shapes_layer`.
    Polygons are kept if `min_value` ≤ `numerical_column` ≤ `max_value`.

    Parameters
    ----------
    sdata
        SpatialData object containing the shapes layer.
    shapes_layer
        Name of the input shapes layer.
    output_shapes_layer
        Name of the output shapes layer to store the filtered polygons.
    numerical_column
        Name of numerical column in the shapes layer used for filtering.
    min_value
        minimum value of `numerical_column` (inclusive). If None, lower bound is ignored.
    max_value
        maximum value of `numerical_column` (inclusive). If None, upper bound is ignored.
    overwrite
        Whether to overwrite existing output layer.

    Returns
    -------
    SpatialData
        Object with updated shapes layer containing only polygons meeting criteria.
    """
    gdf = sdata.shapes[shapes_layer].copy()

    if numerical_column not in gdf.columns:
        raise ValueError(f"Column '{numerical_column}' not found in '{shapes_layer}.obs'. ")

    if not np.issubdtype(gdf[numerical_column].dtype, np.number):
        raise ValueError(f"Column '{numerical_column}' must be numeric, but dtype is {gdf[numerical_column].dtype}.")

    # Filter cells based on min and max values
    start = len(gdf)
    mask = pd.Series(True, index=gdf.index)

    if min_value is not None:
        below = (gdf[numerical_column] < min_value).sum()
        log.info(f"Removed {below} cells below {min_value}.")
        mask &= gdf[numerical_column] >= min_value
    if max_value is not None:
        above = (gdf[numerical_column] > max_value).sum()
        log.info(f"Removed {above} cells above {max_value}.")
        mask &= gdf[numerical_column] <= max_value

    filtered_gdf = gdf[mask].copy()
    kept = len(filtered_gdf)
    removed = start - len(filtered_gdf)
    log.info(
        f"Removed {removed}/{start} polygons outside {min_value}–{max_value} for '{numerical_column}' (kept {kept})."
    )

    # sanity check. If this sanity check would fail in spatialdata at some point, then pass transformation to transformations parameter of add_shapes_layer.
    assert get_transformation(filtered_gdf, get_all=True) == get_transformation(sdata[shapes_layer], get_all=True)

    # Add to SpatialData
    sdata = add_shapes_layer(
        sdata,
        input=filtered_gdf,
        output_layer=output_shapes_layer,
        overwrite=overwrite,
    )

    return sdata


def filter_shapes_categorical(
    sdata: SpatialData,
    shapes_layer: str,
    output_shapes_layer: str,
    categorical_column: str | None = None,
    include_values: str | Sequence[str] | None = None,
    exclude_values: str | Sequence[str] | None = None,
    overwrite: bool = False,
) -> SpatialData:
    """
    Filter polygons in a shapes layer based on categorical values or index.

    Parameters
    ----------
    sdata
        SpatialData object containing the shapes layer.
    shapes_layer
        Name of the input shapes layer.
    output_shapes_layer
        Name of the output shapes layer to store filtered polygons.
    categorical_column
        Name of the categorical column in the shapes layer. If None, filtering is performed based on the index.
    include_values
        Value(s) to keep. Only polygons whose `categorical_column` matches one of these
        values will be kept. Mutually exclusive with `exclude_values`.
    exclude_values
        Value(s) to remove. polygons whose `categorical_column` matches one of these
        values will be removed. Mutually exclusive with `include_values`.
    overwrite
        If True, overwrites the `output_shapes_layer` if it already exists in `sdata`.

    Returns
    -------
    SpatialData
        Object with updated shapes layer containing filtered polygons.
    """
    if include_values is not None and exclude_values is not None:
        raise ValueError("Specify only one of 'include_values' or 'exclude_values'.")

    gdf = sdata.shapes[shapes_layer].copy()

    if categorical_column is not None and categorical_column not in gdf.columns:
        raise ValueError(f"Column '{categorical_column}' not found in '{shapes_layer}'.")

    # Ensure include/exclude are lists
    if isinstance(include_values, (str, int)):
        include_values = [include_values]
    if isinstance(exclude_values, (str, int)):
        exclude_values = [exclude_values]

    # Filter
    start = len(gdf)
    mask = pd.Series(True, index=gdf.index)

    if categorical_column is not None:
        # Filtering by column
        if include_values is not None:
            mask &= gdf[categorical_column].isin(include_values)
        elif exclude_values is not None:
            mask &= ~gdf[categorical_column].isin(exclude_values)
    else:
        # Filtering by index
        if include_values is not None:
            mask &= gdf.index.isin(include_values)
        elif exclude_values is not None:
            mask &= ~gdf.index.isin(exclude_values)

    filtered_gdf = gdf[mask].copy()
    kept = len(filtered_gdf)
    removed = start - kept
    if categorical_column is None:
        categorical_column = "index"
    log.info(f"Removed {removed}/{start} polygons based on '{categorical_column}' filtering (kept {kept}).")

    # sanity check. If this sanity check would fail in spatialdata at some point, then pass transformation to transformations parameter of add_shapes_layer.
    assert get_transformation(filtered_gdf, get_all=True) == get_transformation(sdata[shapes_layer], get_all=True)

    # Add to SpatialData
    sdata = add_shapes_layer(
        sdata,
        input=filtered_gdf,
        output_layer=output_shapes_layer,
        overwrite=overwrite,
    )

    return sdata


def filter_shapes_by_shapes(
    sdata: SpatialData,
    target_shapes_layer: str,
    mask_shapes_layer: str,
    output_shapes_layer: str,
    shape_names_column: str | None = None,
    shape_names: str | list[str] | None = None,
    mode: Literal[
        "intersects",
        "within",
        "centroid_within",
        "touches",
        "disjoint",
        "overlap_fraction",
        "edge",
        "outer_edge",
        "inner_edge",
    ] = "centroid_within",
    keep: bool = True,
    threshold: float | None = None,
    overwrite: bool = False,
):
    """
    Filter polygons in a target shapes layer based on geometric relationships to polygons of a mask shapes layer.
    A typical use-case would be to filter cell segmentations based on their location within annotated regions.

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
    mode
        Geometric relationship to use for filtering.
        Supported modes:
            - `"intersects"`: Keep/remove polygons that intersect mask polygons (i.e. have partial or full overlap).
            - `"within"`: Keep/remove polygons that are fully contained within mask polygons.
            - `"centroid_within"`: Keep/remove polygons whose centroids are within mask polygons.
            - `"touches"`: Keep/remove polygons that touch the boundary of mask polygons but do not overlap.
            - `"disjoint"`: Keep/remove polygons that have no overlap with mask polygons at all.
            - `"overlap_fraction"`: Keep/remove polygons whose overlap fraction with mask polygons exceeds a threshold.
            - `"edge"`: Keep/remove polygons that partially overlap either the outer or inner edge of the mask polygons.
            - `"outer_edge"`: Keep/remove polygons that overlap the outer edge (shell) of the mask polygons.
            - `"inner_edge"`: Keep/remove polygons that overlap the inner edge (holes) of the mask polygons.
    keep
        Whether to keep (True) or remove (False) polygons matching the condition.
    threshold
        Threshold to use for overlap_fraction.
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

    # Compute condition
    if mode == "intersects":
        condition = target_gdf.geometry.intersects(mask_union)
    elif mode == "within":
        condition = target_gdf.geometry.within(mask_union)
    elif mode == "centroid_within":
        condition = target_gdf.geometry.centroid.within(mask_union)
    elif mode == "touches":
        condition = target_gdf.geometry.touches(mask_union)
    elif mode == "disjoint":
        condition = target_gdf.geometry.disjoint(mask_union)
    elif mode == "overlap_fraction":
        if threshold is None:
            raise ValueError("Must specify 'threshold' for overlap_fraction mode.")
        intersecting = target_gdf[target_gdf.geometry.intersects(mask_union)]
        overlaps = intersecting.geometry.intersection(mask_union)
        fraction = overlaps.area / intersecting.geometry.area
        condition = pd.Series(False, index=target_gdf.index)
        condition.loc[intersecting.index] = fraction > threshold
    elif mode == "edge":
        condition = target_gdf.geometry.intersects(mask_union) & ~target_gdf.geometry.within(mask_union)
    elif mode in ["outer_edge", "inner_edge"]:
        outer_boundaries = []
        inner_boundaries = []

        if isinstance(mask_union, gpd.GeoSeries):
            for geom in mask_union:
                outer_boundaries.extend(_extract_exterior_lines(geom))
                inner_boundaries.extend(_extract_interior_lines(geom))
        else:
            outer_boundaries.extend(_extract_exterior_lines(mask_union))
            inner_boundaries.extend(_extract_interior_lines(mask_union))

        outer_boundary_union = (
            unary_union([LineString(boundary) for boundary in outer_boundaries]) if outer_boundaries else None
        )
        inner_boundary_union = (
            unary_union([LineString(boundary) for boundary in inner_boundaries]) if inner_boundaries else None
        )

        if mode == "outer_edge":
            condition = target_gdf.geometry.intersects(outer_boundary_union)
        elif mode == "inner_edge":
            condition = (
                target_gdf.geometry.intersects(inner_boundary_union) if inner_boundary_union is not None else False
            )

    else:
        raise ValueError(f"Unsupported mode '{mode}'.")

    # Apply keep/remove
    filtered_gdf = target_gdf[condition if keep else ~condition].copy()
    removed = len(target_gdf) - len(filtered_gdf)

    if keep:
        log.info(f"Kept {len(filtered_gdf)} / {len(target_gdf)} geometries (removed {removed}).")
    else:
        log.info(f"Removed {removed} / {len(target_gdf)} geometries (kept {len(filtered_gdf)}).")

    # sanity check. If this sanity check would fail in spatialdata at some point, then pass transformation to transformations parameter of add_shapes_layer.
    assert get_transformation(filtered_gdf, get_all=True) == get_transformation(
        sdata[target_shapes_layer], get_all=True
    )

    # Add to SpatialData
    sdata = add_shapes_layer(
        sdata,
        input=filtered_gdf,
        output_layer=output_shapes_layer,
        overwrite=overwrite,
    )

    return sdata
