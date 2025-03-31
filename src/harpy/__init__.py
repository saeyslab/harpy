"""Define package version"""
import importlib.metadata

__version__ = importlib.metadata.version("harpy-analysis")

import os

os.environ["USE_PYGEOS"] = "0"

try:
    import rasterio
except ImportError:
    pass

from harpy import (  # noqa: E402
    datasets,
    io,
    utils,
)
from harpy import image as im  # noqa: E402
from harpy import plot as pl  # noqa: E402
from harpy import points as pt  # noqa: E402
from harpy import shape as sh  # noqa: E402
from harpy import table as tb  # noqa: E402
