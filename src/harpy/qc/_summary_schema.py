"""Every output column and dimension name of the QC summaries of points and tables.

The one inventory of the strings that the summaries, and the QC metrics that
``harpy.tb.pp.calculate_qc_metrics`` adds to a table, write: a module that
writes or reads an output name takes it from here. The statistics, such as
``mean_points_per_instance``, are named by ``_statistic_name`` of
``harpy.qc._summary_statistics``. The docstrings of the result classes describe
each column.
"""

from typing import Final

# These are summary output names, not the panel's source points column names
# nor a table's own .obs columns.
_FEATURE_KEY: Final = "feature"
_FEATURE_CLASS_KEY: Final = "feature_class"
_N_FEATURES_KEY: Final = "n_features"
_N_POINTS_KEY: Final = "n_points"
_PCT_POINTS_KEY: Final = "pct_points"
# A feature's points over its class's points, in PointsSummary.per_feature and TableSummary.per_feature.
_WITHIN_CLASS_FRACTION_KEY: Final = "within_class_fraction"

# PointsSummary.per_class: the class's features without points and their percentage, the requested top_n,
# the features it covers, min(top_n, panel size), and the percentage of the class's points in them.
_N_ZERO_FEATURES_KEY: Final = "n_zero_features"
_PCT_ZERO_FEATURES_KEY: Final = "pct_zero_features"
_TOP_N_KEY: Final = "top_n"
_N_TOP_FEATURES_KEY: Final = "n_top_features"
_PCT_POINTS_TOP_N_FEATURES_KEY: Final = "pct_points_top_n_features"

# The spatial bins of a points summary: the dimensions and bin-center coordinates of spatial_counts and of the
# retained-bin mask, and the columns of SpatialBinSummary.per_bin, with each bin's row and column index, also
# the levels of the binned counts before the grid, and its area, also in square micrometres when calibrated.
_X_KEY: Final = "x"
_Y_KEY: Final = "y"
_X_BIN_KEY: Final = "x_bin"
_Y_BIN_KEY: Final = "y_bin"
_BIN_AREA_KEY: Final = "bin_area"
_BIN_AREA_UM2_KEY: Final = "bin_area_um2"
# The population of the per_class and per_feature frames of the bin summaries: every bin of the grid, those
# with points of a selected class or feature, the others, and 100 times excluded / total.
_N_TOTAL_BINS_KEY: Final = "n_total_bins"
_N_RETAINED_BINS_KEY: Final = "n_retained_bins"
_N_EXCLUDED_BINS_KEY: Final = "n_excluded_bins"
_PCT_EXCLUDED_BINS_KEY: Final = "pct_excluded_bins"
# The retained bins with zero points of the row's group, a class or a feature, and their percentage,
# formatted with the group: "class" or "feature".
_N_RETAINED_BINS_WITHOUT_GROUP_TEMPLATE: Final = "n_retained_bins_without_{group}"
_PCT_RETAINED_BINS_WITHOUT_GROUP_TEMPLATE: Final = "pct_retained_bins_without_{group}"

# The rows of the selected regions, the population column of TableSummary.per_class.
_N_INSTANCES_KEY: Final = "n_instances"
# The instances with zero points of a class, and 100 times their number / n_instances, in
# TableSummary.per_class and per_region_class.
_N_INSTANCES_WITHOUT_CLASS_KEY: Final = "n_instances_without_class"
_PCT_INSTANCES_WITHOUT_CLASS_KEY: Final = "pct_instances_without_class"
# The instances with zero points of a feature, and 100 times their number / n_instances, in the per_feature
# frames of TableSummary and FeatureTableSummary.
_N_INSTANCES_WITHOUT_FEATURE_KEY: Final = "n_instances_without_feature"
_PCT_INSTANCES_WITHOUT_FEATURE_KEY: Final = "pct_instances_without_feature"
# The per-class columns of TableSummary.per_instance, formatted with the class label.
_N_CLASS_POINTS_TEMPLATE: Final = "n_{label}_points"
_N_CLASS_DETECTED_FEATURES_TEMPLATE: Final = "n_{label}_detected_features"
_PCT_CLASS_POINTS_TEMPLATE: Final = "pct_{label}_points"
# The rate per panel feature of a class, formatted with the class label, that harpy.tb.pp.calculate_qc_metrics
# adds to .obs.
_N_CLASS_POINTS_PER_PANEL_FEATURE_TEMPLATE: Final = "n_{label}_points_per_panel_feature"
# The region column of TableSummary.per_region_class and per_region_feature. Not harpy.utils._keys._REGION_KEY,
# the default name of a table's own region column in .obs.
_REGION_COLUMN_KEY: Final = "region"
