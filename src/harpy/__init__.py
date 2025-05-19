"""Define package version"""

import importlib.metadata
import os

__version__ = importlib.metadata.version("harpy-analysis")
os.environ["USE_PYGEOS"] = "0"

# silence dask future warnings
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import dask

dask.config.set({'dataframe.query-planning': False})


from harpy import datasets
from harpy import image as im  # noqa: E402
from harpy import io
from harpy import plot as pl  # noqa: E402
from harpy import points as pt  # noqa: E402
from harpy import shape as sh  # noqa: E402
from harpy import table as tb  # noqa: E402
from harpy import utils

__all__ = [
    "datasets",
    "io",
    "utils",
    "im",
    "pl",
    "pt",
    "sh",
    "tb",
    "__version__",
]
