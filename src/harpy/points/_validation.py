"""Read-only validation of points and their registered feature panels."""

import pandas as pd
from spatialdata import SpatialData
from spatialdata.models import PointsModel

from harpy._feature_panels import (
    _feature_panel_partition_errors,
    _resolve_points_feature_panel,
    _validate_feature_class_dtype,
)


def validate_points(sdata: SpatialData, points_name: str) -> None:
    """Validate one points element against its registered feature panel.

    Requires a panel registered in ``sdata.attrs["harpy"]``; use
    :func:`harpy.pt.add_feature_panel` to register one first. Scans feature/class
    columns partition-wise without collecting the complete points dataframe.
    Panel features and classes may have zero detections.

    Read-only for backed and unbacked objects: no normalization, repairs or
    writes. Checks assay compatibility, not sample identity or spatial registration.

    Parameters
    ----------
    sdata
        SpatialData containing the points and their registered feature panel.
    points_name
        Name of the existing points element to validate.

    Returns
    -------
    None
        Return without modifying the object when all checks pass.

    Raises
    ------
    ValueError
        If the points or their panel metadata are missing or incompatible.

    See Also
    --------
    harpy.pt.add_feature_panel : Register a panel and normalize classes when needed.
    harpy.tb.validate_table : Validate a table and its recognized Harpy metadata.
    """
    if points_name not in sdata.points:
        raise ValueError(f"Points element {points_name!r} does not exist.")
    points = sdata.points[points_name]
    PointsModel.validate(points)
    _, panel, _ = _resolve_points_feature_panel(sdata, points_name)
    columns = [panel.feature_key, panel.feature_class_key]
    for key in columns:
        if key not in points.columns:
            raise ValueError(f"Points element {points_name!r} does not contain panel column {key!r}.")
    _validate_feature_class_dtype(points, points_name=points_name, panel=panel)

    # Each partition returns at most one error, including when categories are
    # unknown. Inspect observed values without normalizing or collecting points.
    errors = (
        points[columns]
        .map_partitions(
            _feature_panel_partition_errors,
            feature_key=panel.feature_key,
            feature_class_key=panel.feature_class_key,
            class_by_feature=panel.class_by_feature,
            meta=pd.Series(name="error", dtype="object"),
        )
        .compute()
    )
    if not errors.empty:
        raise ValueError(f"Points element {points_name!r} disagrees with its feature panel: {errors.iloc[0]}")
