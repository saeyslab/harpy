"""Fixed output column and dimension names shared by the QC summaries of points and tables."""

from typing import Final

# These are summary output names, not the panel's source points column names
# nor a table's own .obs columns.
_FEATURE_KEY: Final = "feature"
_FEATURE_CLASS_KEY: Final = "feature_class"
_N_FEATURES_KEY: Final = "n_features"
_N_POINTS_KEY: Final = "n_points"
# The region column of TableSummary.per_region. Not harpy.utils._keys._REGION_KEY,
# the default name of a table's own region column in .obs.
_REGION_COLUMN_KEY: Final = "region"
