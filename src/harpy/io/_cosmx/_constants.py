from __future__ import annotations

import re
from enum import unique

from spatialdata_io._constants._enum import ModeEnum


@unique
class _CosmxKeys(ModeEnum):
    """Format-defined paths, filename tokens, and column names for CosMx."""

    CELL_STATS_DIR = "CellStatsDir"
    MORPHOLOGY_DIR = "Morphology2D"
    ANALYSIS_RESULTS_DIR = "AnalysisResults"

    TRANSCRIPT_SUFFIX = "target_call_coord.csv"
    INSTANCE_LABEL_PREFIX = "celllabels_f"
    COMPARTMENT_LABEL_PREFIX = "compartmentlabels_f"

    PLEX_FEATURE_COLUMN = "DisplayName"
    PLEX_CLASS_COLUMN = "CodeClass"
    FEATURE_KEY = "gene"
    FEATURE_CLASS_KEY = "code_class"
    ACQUISITION_TIMESTAMP = "OrigTimeStamp"


_FOV_DIR_RE = re.compile(r"^FOV0*(\d+)$", re.IGNORECASE)
_FOV_FILE_RE = re.compile(r"(?:^|[_-])(?:FOV|F)0*(\d+)(?=[_.-]|$)", re.IGNORECASE)
_MORPHOLOGY_FILE_RE = re.compile(
    r"^\d{8}_\d{6}_S\d+.*_F0*(?P<fov>\d+)\.(?:tif|tiff)$",
    re.IGNORECASE,
)
_PLEX_FILE_RE = re.compile(r"^plex(?:[-_].*)?\.txt$", re.IGNORECASE)

_DEFAULT_PIXEL_SIZE_UM = 0.120280945
_TIFF_SUFFIXES = (".tif", ".tiff")
_MORPHOLOGY_GEOMETRY_INVARIANT_KEYS = ("NFov", "ImPixelSize_nm", "ImRows", "ImCols")
_MORPHOLOGY_IMAGE_INVARIANT_KEYS = ("ChannelOrder",)
_INSTANCE_ID_FORMULA = "global_id = (fov - 1) * base + local_id"
# Element-record keys that are read outside the record builder, by the validator and assign_cosmx_fovs.
_SAMPLE_ID_KEY = "sample_id"
_FOV_BOUNDS_KEY = "fov_bounds_px"
# The fields of each FOV's half-open pixel bounds in an element record's fov_bounds_px.
_FOV_BOUNDS_FIELDS = ("y0", "x0", "y1", "x1")
_COMPARTMENT_CATEGORIES = {
    0: "background",
    1: "nuclear",
    2: "membrane",
    3: "cytoplasmic",
}
