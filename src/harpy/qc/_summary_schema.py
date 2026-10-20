"""Fixed output column and dimension names shared by the QC summaries of points and tables."""

from typing import Final

# These are summary output names, not the panel's source points column names
# nor a table's own .obs columns.
_FEATURE_KEY: Final = "feature"
_FEATURE_CLASS_KEY: Final = "feature_class"
_N_FEATURES_KEY: Final = "n_features"
_N_POINTS_KEY: Final = "n_points"
_PCT_POINTS_KEY: Final = "pct_points"
# The rows of the selected regions, the population column of TableSummary.per_class.
_N_INSTANCES_KEY: Final = "n_instances"
# The per-class columns of TableSummary.per_instance, formatted with the class label.
_N_CLASS_POINTS_TEMPLATE: Final = "n_{label}_points"
_N_CLASS_DETECTED_FEATURES_TEMPLATE: Final = "n_{label}_detected_features"
_PCT_CLASS_POINTS_TEMPLATE: Final = "pct_{label}_points"
# The region column of TableSummary.per_region_class and per_region_feature. Not harpy.utils._keys._REGION_KEY,
# the default name of a table's own region column in .obs.
_REGION_COLUMN_KEY: Final = "region"
