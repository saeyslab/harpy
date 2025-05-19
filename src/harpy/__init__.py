"""Define package version"""

import importlib.metadata

__version__ = importlib.metadata.version("harpy-analysis")

import os

os.environ["USE_PYGEOS"] = "0"

try:
    import rasterio
except ImportError:
    pass

from harpy import (
    datasets,
    io,
    utils,
)
from harpy import image as im
from harpy import plot as pl
from harpy import points as pt
from harpy import shape as sh
from harpy import table as tb
