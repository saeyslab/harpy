from typing import Literal

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon
from spatialdata import SpatialData

from harpy.table._table import add_table_layer
from harpy.utils._keys import _REGION_KEY


def assign_cells_to_shapes(
    sdata: SpatialData,
    shapes_layer: str,
    table_layer: str,
    output_table_layer: str,
    shape_names_column: str = "name",
    unique_shape_names_column: str = "name-unique",
    output_column: str = None,
    mode: Literal["original_names", "unique_names", "both"] = "original_names",
    create_column_per_shape: bool = False,
    overlap_tolerance: float = 0.1,
    spatial_key: str = "spatial",
    xy_columns: tuple = None,
    overwrite: bool = False,
):
    """
    Assign cells to polygons in a shapes layer and update the sdata table layer. It is recommended to run `hp.sh.prep_region_annotations` first.

    Parameters
    ----------
    sdata
        The SpatialData object containing the input table layer and shapes layer.
    shapes_layer
        The shapes layer in `sdata.shapes` to use as input.
    table_layer
        The table layer in `sdata.tables` to use as input.
    output_table_layer
        The output table layer in `sdata.tables` to which the updated table layer will be written.
    shape_names_column
        Column name in shapes layer containing geometry names.
    unique_shape_names_column
        Column name in shapes layer containing unique geometry names.
    output_column
        Name of the output column in `sdata.tables[table_layer].obs` to store the shape name of the geometries a cell was found in (if `create_column_per_shape` is False).
        For the `unique_names` mode, the output will be stored in `{output_column}-unique` in `sdata.tables[table_layer].obs` (if `create_column_per_shape` is False).
        If create_column_per_shape is True and output_column is not None, then column names will be in the format `{output_colum}-{shape_name}`.
        If create_column_per_shape is True and output_column is None, then column names will be in the format `{shape_name}`.
    mode
        When set to `original_names`, original polygon names from `shape_names_column` will be used.
        When set to `unique_names`, unique polygon names from `unique_shape_names_column` will be used. Use `both`, to run both modes at the same time.
    create_column_per_shape
        If True, create one column (named according to the shape names) per shape indicating whether a cell is located inside it.
    overlap_tolerance
        Tolerance for detecting overlapping polygons (area units of geometry CRS).
    spatial_key
        Key in `sdata.tables[table_layer].obsm` containing spatial coordinates. Ignored if `xy_columns` is provided.
    xy_columns
        Tuple of column names in `sdata.tables[table_layer].obs` containing the x and y coordinates the cells.
        If None, defaults to using coordinates from `sdata.tables[table_layer].obsm[spatial_key]`.
    overwrite
        If True, overwrites the `output_table_layer` and/or `output_shapes_layer` if it already exists in `sdata`.

    Notes
    -----
    - Only `Polygon` and `MultiPolygon` geometries are supported. Non-polygon geometries (e.g., `Point`, `LineString`) are skipped.

    Returns
    -------
    Modified `sdata` object with updated table layer.
    """
    if not output_column and not create_column_per_shape:
        raise ValueError("Specify `output_column` or set `create_column_per_shape=True`.")

    # Create copy of shapes layer
    gdf = sdata.shapes[shapes_layer].copy()

    # Create copy of table layer
    adata = sdata.tables[table_layer].copy()

    # Filter out geometries that are not Polygon or MultiPolygon
    supported_mask = gdf.geometry.apply(lambda x: isinstance(x, (Polygon, MultiPolygon)))
    skipped = len(gdf) - supported_mask.sum()
    if skipped > 0:
        print(f"Skipped {skipped} non-polygon geometries in {shapes_layer}.")
    if skipped == len(gdf):
        print("No supported geometries (Polygon, MultiPolygon) found.")
        return sdata

    gdf = gdf[supported_mask].copy().reset_index(drop=True)

    # Check for overlapping polygons
    total_area = gdf.geometry.area.sum()
    union_area = gdf.geometry.unary_union.area
    if total_area - union_area > overlap_tolerance and not create_column_per_shape:
        raise ValueError(
            f"Overlapping polygons detected in {shapes_layer}. Correct polygons or use create_column_per_shape."
        )
    elif total_area - union_area > 0 and not create_column_per_shape:
        print(f"Overlaps detected (Δ={total_area - union_area:.3f}), below tolerance threshold {overlap_tolerance}.")

    # Get cell coordinates
    if xy_columns is not None:
        x_col, y_col = xy_columns
        coords = adata.obs[[x_col, y_col]].to_numpy()
    else:
        if spatial_key not in adata.obsm:
            raise KeyError(f"No spatial coordinates found in `obsm['{spatial_key}']` and `xy_columns` not provided.")
        coords = adata.obsm[spatial_key]
        if coords.shape[1] != 2:
            raise ValueError(f"`obsm['{spatial_key}']` must have shape (n_cells, 2).")

    # Function to assign cell to a polygon
    def _assign_region(x, y, gdf, column, sindex):
        point = Point(x, y)

        # Quick bounding box search to prefilter the cells
        candidate_idx = list(sindex.intersection(point.bounds))
        if not candidate_idx:
            return None

        # Slower point‑in‑polygon search
        candidate_gdf = gdf.iloc[candidate_idx]
        match = candidate_gdf[candidate_gdf.contains(point)]

        if match.empty:
            return None

        return match[column].iloc[0]

    # Assign cells to polygons
    if create_column_per_shape:
        # Collect all relevant name columns based on mode
        name_columns = []
        if mode in ("original_names", "both"):
            name_columns.append(shape_names_column)
        if mode in ("unique_names", "both"):
            name_columns.append(unique_shape_names_column)

        # Collect all unique names across both columns
        name_to_column = {}
        for name_column in name_columns:
            for name in gdf[name_column].dropna().astype(str).unique():
                if name not in name_to_column:
                    name_to_column[name] = name_column

        # Create column per shape name
        for shape_name, name_column in name_to_column.items():
            col_name = f"{output_column}-{shape_name}" if output_column else shape_name

            gdf_shape = gdf[gdf[name_column] == shape_name]
            if gdf_shape.empty:
                continue

            sidx_shape = gdf_shape.sindex  # Build spatial index
            assigned = [_assign_region(x, y, gdf_shape, name_column, sidx_shape) for x, y in coords]
            adata.obs[col_name] = assigned

    else:
        # One output column per mode
        run_assign_dict = {}
        if mode in ("original_names", "both"):
            run_assign_dict[output_column] = shape_names_column
        if mode in ("unique_names", "both"):
            run_assign_dict[f"{output_column}-unique"] = unique_shape_names_column

        for output_column_name, name_column in run_assign_dict.items():
            # Assign one shape name per cell (to be avoided with overlapping regions)
            sidx = gdf.sindex  # Build spatial index
            assigned = [_assign_region(x, y, gdf, name_column, sidx) for x, y in coords]
            adata.obs[output_column_name] = assigned

    # Add table layer
    sdata = add_table_layer(
        sdata,
        adata=adata,
        output_layer=output_table_layer,
        region=adata.obs[_REGION_KEY].cat.categories.to_list(),
        overwrite=overwrite,
    )

    return sdata


def compute_distance_to_shapes(
    sdata: SpatialData,
    shapes_layer: str,
    table_layer: str,
    output_table_layer: str,
    shape_names_column: str = "name",
    unique_shape_names_column: str = "name-unique",
    output_name: str = None,
    pixel_size_um: float = 1.0,
    modes: list[
        Literal[
            "nearest_edge",
            "nearest_edge_grouped",
            "all_edges",
            "nearest_outer_edge",
            "nearest_outer_edge_grouped",
            "all_outer_edges",
            "nearest_inner_edge",
            "nearest_inner_edge_grouped",
            "nearest_inner_edge_grouped_unique",
            "all_inner_edges",
            "nearest_centroid",
            "nearest_centroid_grouped",
            "all_centroids",
            "nearest_point",
            "nearest_point_grouped",
            "all_points",
        ]
    ] = None,
    spatial_key: str = "spatial",
    xy_columns: tuple = None,
    overwrite: bool = False,
):
    """
    Compute distances from cells to polygons in a shapes layer and update the sdata table layer.
    It is recommended to run `hp.sh.prep_region_annotations` first.

    Parameters
    ----------
    sdata
        The SpatialData object containing the input table layer and shapes layer.
    shapes_layer
        The shapes layer in `sdata.shapes` to use as input.
    table_layer
        The table layer in `sdata.tables` to use as input.
    output_table_layer
        The output table layer in `sdata.tables` to which the updated table layer will be written.
    shape_names_column
        Column name in shapes layer containing geometry names.
    unique_shape_names_column
        Column name in shapes layer containing unique geometry names.
    output_name
        Prefix for new distance columns in `.obs`. If None, no prefix is added.
    pixel_size_um
        Scale factor to convert distances to microns. Defaults to 1 (i.e. distances are in pixels).
    modes
        Which distance features to calculate. Options include:

        Edge distances (For Polygon and MultiPolygon geometries)
        - `"nearest_edge"`: Distance to the nearest edge (outer or inner) of any polygon.
                Creates `{output_name}-distance_to_nearest_edge` and `{output_name}-name_of_nearest_edge` columns.
        - `"nearest_edge_grouped"`: For each group of shapes with the same name, compute the distance to the edge that is nearest.
                Creates `{output_name}-distance_to_nearest_edge_of_<name>` and `{output_name}-name_of_nearest_edge_of_<name>` columns.
        - `"all_edges"`: For each individual polygon, compute the distance to its edge.
                Creates `{output_name}-distance_to_edge_of_<shape_unique_name>` column.

        Outer edge distances (For Polygon and MultiPolygon geometries)
        - `"nearest_outer_edge"`: Distance to the nearest outer edge of any polygon.
                Creates `{output_name}-distance_to_nearest_outer_edge` and `{output_name}-name_of_nearest_outer_edge` columns.
        - `"nearest_outer_edge_grouped"`: For each group of shapes with the same name, compute the distance to the outer edge that is nearest.
                Creates `{output_name}-distance_to_nearest_outer_edge_of_<name>` and `{output_name}-name_of_nearest_outer_edge_of_<name>` columns.
        - `"all_outer_edges"`: For each individual polygon, compute the distance to its outer edge.
                Creates `{output_name}-distance_to_outer_edge_of_<shape_unique_name>` column.

        Inner edge (hole) distances (For Polygon and MultiPolygon geometries)
        - `"nearest_inner_edge"`: Distance to the nearest interior edge (“hole”) of any polygon.
                Creates `{output_name}-distance_to_nearest_inner_edge` and `{output_name}-name_of_nearest_inner_edge` columns.
        - `"nearest_inner_edge_grouped"`: For each group of shapes with the same name, compute the distance to the nearest inner edge of that group.
                Creates `{output_name}-distance_to_nearest_inner_edge_of_<name>` and `{output_name}-name_of_nearest_inner_edge_of_<name>` columns.
        - `"nearest_inner_edge_grouped_unique"`: For each individual polygon, compute the distance to the nearest inner edge of that polygon.
                Creates `{output_name}-distance_to_nearest_inner_edge_of_<shape_unique_name>` and `{output_name}-name_of_nearest_inner_edge_of_<shape_unique_name>` columns.
        - `"all_inner_edges"`: For all holes, compute the distance to each inner edge.
                Creates `{output_name}-distance_to_inner_edge_of_<shape_unique_name>-hole<i>` column.

        Centroid distances (For Polygon and MultiPolygon geometries)
        - `"nearest_centroid"`: Distance to the nearest centroid of any polygon.
                Creates `{output_name}-distance_to_nearest_centroid` and `{output_name}-name_of_nearest_centroid` columns.
        - `"nearest_centroid_grouped"`: For each group of shapes with the same name, compute the distance to the centroid that is nearest.
                Creates `{output_name}-distance_to_nearest_centroid_of_<name>` and `{output_name}-name_of_nearest_centroid_of_<name>` columns.
        - `"all_centroids"`: For each individual polygon, compute the distance to its centroid.
                Creates `{output_name}-distance_to_centroid_of_<shape_unique_name>` column.

        Point distances (For Point geometries)
        - `"nearest_point"`: Distance to the nearest point.
                Creates `{output_name}-distance_to_nearest_point` and `{output_name}-name_of_nearest_point` columns.
        - `"nearest_point_grouped"`: For each group of points with the same name, compute the nearest distance.
                Creates `{output_name}-distance_to_nearest_point_of_<name>` and `{output_name}-name_of_nearest_point_of_<name>` columns.
        - `"all_points"`: For each individual Point, compute the distance to point coordinates.
                Creates `{output_name}-distance_to_point_<shape_unique_name>` column.

    spatial_key
        Key in `sdata.tables[table_layer].obsm` containing spatial coordinates. Ignored if `xy_columns` is provided.
    xy_columns
        Tuple of column names in `sdata.tables[table_layer].obs` containing the x and y coordinates the cells.
        If None, defaults to using coordinates from `sdata.tables[table_layer].obsm[spatial_key]`.
    overwrite
        If True, overwrites the `output_table_layer` if it already exists in `sdata`.

    Notes
    -----
    - Only `Polygon`, `MultiPolygon` and `Point` geometries are supported. Other geometries (e.g., `LineString`, `MultiPoint`) are skipped.

    Returns
    -------
    Modified `sdata` object with updated table layer.
    """
    if modes is None:
        modes = ["all_edges"]
    if output_name is not None:
        output_name = f"{output_name}-"
    elif output_name is None:
        output_name = ""

    # Create copy of shapes layer
    gdf = sdata.shapes[shapes_layer].copy()

    # Create copy of table layer
    adata = sdata.tables[table_layer].copy()

    # Filter out geometries that are not Polygon, MultiPolygon or Point
    supported_mask = gdf.geometry.apply(lambda x: isinstance(x, (Polygon, MultiPolygon, Point)))
    skipped = len(gdf) - supported_mask.sum()
    if skipped > 0:
        print(f"Skipped {skipped} geometries in {shapes_layer} that are not Polygon, MultiPolygon or Point.")

    gdf = gdf[supported_mask].copy().reset_index(drop=True)

    # Separate gdf by type
    gdf_polygons = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    gdf_points = gdf[gdf.geometry.geom_type == "Point"].copy()

    has_polygons = not gdf_polygons.empty
    has_points = not gdf_points.empty

    if not has_polygons and not has_points:
        print("No supported geometries (Polygon, MultiPolygon, Point) found.")
        return sdata

    # Get cell coordinates
    if xy_columns is not None:
        x_col, y_col = xy_columns
        coords = adata.obs[[x_col, y_col]].to_numpy()
    else:
        if spatial_key not in adata.obsm:
            raise KeyError(f"No spatial coordinates found in `obsm['{spatial_key}']` and `xy_columns` not provided.")
        coords = adata.obsm[spatial_key]
        if coords.shape[1] != 2:
            raise ValueError(f"`obsm['{spatial_key}']` must have shape (n_cells, 2).")

    # Build point GeoDataFrame
    pts = gpd.GeoSeries([Point(x, y) for x, y in coords], crs=gdf.crs)

    pts_gdf = gpd.GeoDataFrame({"geometry": pts}, crs=gdf.crs).set_index(adata.obs.index)

    # Polygon and MultiPolygon
    if has_polygons:
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

        names = gdf_polygons[unique_shape_names_column].tolist()
        exteriors = [MultiLineString(_extract_exterior_lines(geom)) for geom in gdf_polygons.geometry]
        interiors = [MultiLineString(_extract_interior_lines(geom)) for geom in gdf_polygons.geometry]

        ext_gdf = gpd.GeoDataFrame({unique_shape_names_column: names}, geometry=exteriors, crs=gdf.crs)
        int_gdf = gpd.GeoDataFrame({unique_shape_names_column: names}, geometry=interiors, crs=gdf.crs)

        # Edges (outer and inner)
        if "nearest_edge" in modes:
            print("Calculating 'nearest_edge' distances'")
            all_edges_gdf = gpd.GeoDataFrame(pd.concat([ext_gdf, int_gdf], ignore_index=True), crs=gdf.crs)
            joined = gpd.sjoin_nearest(pts_gdf, all_edges_gdf, how="left", distance_col="dist")
            adata.obs[f"{output_name}distance_to_nearest_edge"] = joined["dist"].to_numpy() * pixel_size_um
            adata.obs[f"{output_name}name_of_nearest_edge"] = joined[unique_shape_names_column]

        if "nearest_edge_grouped" in modes:
            print("Calculating 'nearest_edge_grouped' distances'")
            for name, group in gdf_polygons.groupby(shape_names_column):
                edges = []
                labels = []
                for _idx, row in group.iterrows():
                    edge_lines = _extract_exterior_lines(row.geometry) + _extract_interior_lines(row.geometry)
                    edges.extend(edge_lines)
                    labels.extend([row[unique_shape_names_column]] * len(edge_lines))

                edge_gdf = gpd.GeoDataFrame({"geometry": edges, unique_shape_names_column: labels}, crs=gdf.crs)
                joined = gpd.sjoin_nearest(pts_gdf, edge_gdf, how="left", distance_col="dist")

                adata.obs[f"{output_name}distance_to_nearest_edge_of_{name}"] = (
                    joined["dist"].to_numpy() * pixel_size_um
                )
                adata.obs[f"{output_name}name_of_nearest_edge_of_{name}"] = joined[unique_shape_names_column]

        if "all_edges" in modes:
            print("Calculating 'all_edges' distances'")
            for _, feat in gdf_polygons.reset_index(drop=True).iterrows():
                adata.obs[f"{output_name}distance_to_edge_of_{feat[unique_shape_names_column]}"] = (
                    pts.distance(feat.geometry.boundary).to_numpy() * pixel_size_um
                )

        # Outer edges
        if "nearest_outer_edge" in modes:
            print("Calculating 'nearest_outer_edge' distances'")
            joined = gpd.sjoin_nearest(pts_gdf, ext_gdf, how="left", distance_col="dist")
            adata.obs[f"{output_name}distance_to_nearest_outer_edge"] = joined["dist"].to_numpy() * pixel_size_um
            adata.obs[f"{output_name}name_of_nearest_outer_edge"] = joined[unique_shape_names_column]

        if "nearest_outer_edge_grouped" in modes:
            print("Calculating 'nearest_outer_edge_grouped' distances'")
            for name, group in gdf_polygons.groupby(shape_names_column):
                edges = []
                labels = []
                for _idx, row in group.iterrows():
                    edge_lines = _extract_exterior_lines(row.geometry)
                    edges.extend(edge_lines)
                    labels.extend([row[unique_shape_names_column]] * len(edge_lines))

                edge_gdf = gpd.GeoDataFrame({"geometry": edges, unique_shape_names_column: labels}, crs=gdf.crs)
                joined = gpd.sjoin_nearest(pts_gdf, edge_gdf, how="left", distance_col="dist")

                adata.obs[f"{output_name}distance_to_nearest_outer_edge_of_{name}"] = (
                    joined["dist"].to_numpy() * pixel_size_um
                )
                adata.obs[f"{output_name}name_of_nearest_outer_edge_of_{name}"] = joined[unique_shape_names_column]

        if "all_outer_edges" in modes:
            print("Calculating 'all_outer_edges' distances'")
            for _, feat in gdf_polygons.reset_index(drop=True).iterrows():
                adata.obs[f"{output_name}distance_to_outer_edge_of_{feat[unique_shape_names_column]}"] = (
                    pts.distance(feat.geometry.exterior).to_numpy() * pixel_size_um
                )

        # Inner edges
        hole_geoms = []
        hole_names = []

        for _, row in gdf_polygons.reset_index(drop=True).iterrows():
            base = row[unique_shape_names_column]
            geom = row.geometry

            holes = _extract_interior_lines(geom)

            if len(holes) == 0:
                continue

            if len(holes) == 1:
                hole = holes[0]
                hole_geoms.append(LineString(hole.coords))
                hole_names.append(base)

            else:
                for i, hole in enumerate(holes, start=1):
                    hole_geoms.append(LineString(hole.coords))
                    hole_names.append(f"{base}-hole{i}")

        holes_gdf = gpd.GeoDataFrame({unique_shape_names_column: hole_names}, geometry=hole_geoms, crs=gdf.crs)

        if "nearest_inner_edge" in modes and not holes_gdf.empty:
            print("Calculating 'nearest_inner_edge' distances'")
            joined = gpd.sjoin_nearest(pts_gdf, holes_gdf, how="left", distance_col="dist")
            adata.obs[f"{output_name}distance_to_nearest_inner_edge"] = joined["dist"].to_numpy() * pixel_size_um
            adata.obs[f"{output_name}name_of_nearest_inner_edge"] = joined[unique_shape_names_column]

        if "nearest_inner_edge_grouped" in modes:
            print("Calculating 'nearest_inner_edge_grouped' distances'")
            for name, group in gdf_polygons.groupby(shape_names_column):
                hole_lines = []
                hole_labels = []

                for _idx, row in group.iterrows():
                    base = row[unique_shape_names_column]
                    for i, hole in enumerate(_extract_interior_lines(row.geometry), start=1):
                        hole_lines.append(LineString(hole.coords))
                        hole_labels.append(f"{base}-hole{i}")

                if not hole_lines:
                    continue

                hole_gdf = gpd.GeoDataFrame({"geometry": hole_lines, "hole_name": hole_labels}, crs=gdf.crs)

                joined = gpd.sjoin_nearest(pts_gdf, hole_gdf, how="left", distance_col="dist")

                adata.obs[f"{output_name}distance_to_nearest_inner_edge_of_{name}"] = (
                    joined["dist"].to_numpy() * pixel_size_um
                )
                adata.obs[f"{output_name}name_of_nearest_inner_edge_of_{name}"] = joined["hole_name"]

        if "nearest_inner_edge_grouped_unique" in modes:
            print("Calculating 'nearest_inner_edge_grouped_unique' distances'")
            for _idx, row in gdf_polygons.iterrows():
                base = row[unique_shape_names_column]
                holes = _extract_interior_lines(row.geometry)

                if not holes:
                    continue

                hole_geoms = [LineString(hole.coords) for hole in holes]
                hole_names = [f"{base}-hole{i + 1}" for i in range(len(holes))]

                hole_gdf = gpd.GeoDataFrame({"geometry": hole_geoms, "hole_name": hole_names}, crs=gdf.crs)

                joined = gpd.sjoin_nearest(pts_gdf, hole_gdf, how="left", distance_col="dist")

                adata.obs[f"{output_name}distance_to_nearest_inner_edge_of_{base}"] = (
                    joined["dist"].to_numpy() * pixel_size_um
                )
                adata.obs[f"{output_name}name_of_nearest_inner_edge_of_{base}"] = joined["hole_name"]

        if "all_inner_edges" in modes and not holes_gdf.empty:
            print("Calculating 'all_inner_edges' distances'")
            for _, hole in holes_gdf.iterrows():
                hole_name = hole[unique_shape_names_column]
                col = f"{output_name}distance_to_inner_edge_of_{hole_name}"
                adata.obs[col] = pts.distance(hole.geometry).to_numpy() * pixel_size_um

        # Centroids
        centroids_df = gdf_polygons.reset_index(drop=True)[[unique_shape_names_column]].copy()
        centroids_df["geometry"] = gdf_polygons.geometry.centroid.reset_index(drop=True)

        centroids_gdf = gpd.GeoDataFrame(centroids_df, geometry="geometry", crs=gdf.crs)

        if "nearest_centroid" in modes:
            print("Calculating 'nearest_centroid' distances'")
            joined = gpd.sjoin_nearest(pts_gdf, centroids_gdf, how="left", distance_col="dist")
            adata.obs.loc[joined.index, f"{output_name}distance_to_nearest_centroid"] = (
                joined["dist"].to_numpy() * pixel_size_um
            )
            adata.obs.loc[joined.index, f"{output_name}name_of_nearest_centroid"] = joined[
                unique_shape_names_column
            ].to_numpy()

        if "nearest_centroid_grouped" in modes:
            print("Calculating 'nearest_centroid_grouped' distances'")
            for name, group in gdf_polygons.groupby(shape_names_column):
                centroids = group.geometry.centroid
                labels = group[unique_shape_names_column].tolist()

                centroid_gdf = gpd.GeoDataFrame({"geometry": centroids, unique_shape_names_column: labels}, crs=gdf.crs)

                joined = gpd.sjoin_nearest(pts_gdf, centroid_gdf, how="left", distance_col="dist")

                adata.obs[f"{output_name}distance_to_nearest_centroid_of_{name}"] = (
                    joined["dist"].to_numpy() * pixel_size_um
                )
                adata.obs[f"{output_name}name_of_nearest_centroid_of_{name}"] = joined[unique_shape_names_column]

        if "all_centroids" in modes:
            print("Calculating 'all_centroids' distances'")
            for _idx, feat in gdf_polygons.reset_index(drop=True).iterrows():
                adata.obs[f"{output_name}distance_to_centroid_of_{feat[unique_shape_names_column]}"] = (
                    pts.distance(feat.geometry.centroid).to_numpy() * pixel_size_um
                )

    # Point
    if has_points:
        if "nearest_point" in modes:
            print("Calculating 'nearest_point' distances'")
            joined = gpd.sjoin_nearest(pts_gdf, gdf_points, how="left", distance_col="dist")
            adata.obs[f"{output_name}distance_to_nearest_point"] = joined["dist"].to_numpy() * pixel_size_um
            adata.obs[f"{output_name}name_of_nearest_point"] = joined[f"{shape_names_column}-unique"]

        if "nearest_point_grouped" in modes:
            print("Calculating 'nearest_point_grouped' distances'")
            for name, group in gdf_points.groupby(shape_names_column):
                joined = gpd.sjoin_nearest(pts_gdf, group, how="left", distance_col="dist")
                adata.obs[f"{output_name}distance_to_nearest_point_of_{name}"] = (
                    joined["dist"].to_numpy() * pixel_size_um
                )
                adata.obs[f"{output_name}name_of_nearest_point_of_{name}"] = joined[f"{shape_names_column}-unique"]

        if "all_points" in modes:
            print("Calculating 'all_points' distances'")
            for _, row in gdf_points.iterrows():
                adata.obs[f"{output_name}distance_to_point_{row[unique_shape_names_column]}"] = (
                    pts.distance(row.geometry).to_numpy() * pixel_size_um
                )

    # Add table layer
    sdata = add_table_layer(
        sdata,
        adata=adata,
        output_layer=output_table_layer,
        region=adata.obs[_REGION_KEY].cat.categories.to_list(),
        overwrite=overwrite,
    )

    return sdata
