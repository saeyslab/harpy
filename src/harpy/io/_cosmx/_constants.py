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
    FEATURE_COLUMN = "gene"
    FEATURE_CLASS_COLUMN = "code_class"


_FOV_DIR_RE = re.compile(r"^FOV0*(\d+)$", re.IGNORECASE)
_FOV_FILE_RE = re.compile(r"(?:^|[_-])(?:FOV|F)0*(\d+)(?=[_.-]|$)", re.IGNORECASE)
_MORPHOLOGY_FILE_RE = re.compile(
    r"^\d{8}_\d{6}_S\d+.*_F0*(?P<fov>\d+)\.(?:tif|tiff)$",
    re.IGNORECASE,
)
_PLEX_FILE_RE = re.compile(r"^plex(?:[-_].*)?\.txt$", re.IGNORECASE)

_DEFAULT_PIXEL_SIZE_UM = 0.120280945
_TIFF_SUFFIXES = (".tif", ".tiff")
_MORPHOLOGY_INVARIANT_KEYS = ("NFov", "ChannelOrder", "ImPixelSize_nm", "ImRows", "ImCols")
