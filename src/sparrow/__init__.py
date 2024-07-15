"""Define package version"""

__version__ = "0.0.1"

import os

os.environ["USE_PYGEOS"] = "0"

from . import (  # noqa: E402
    datasets,
    io,
    utils,
)
from . import image as im  # noqa: E402
from . import plot as pl  # noqa: E402
from . import points as pt
from . import shape as sh  # noqa: E402
from . import table as tb  # noqa: E402
