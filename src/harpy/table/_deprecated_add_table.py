"""``harpy.tb.add_table``, deprecated in favour of ``harpy.tb.io.add_table``."""

from __future__ import annotations

import warnings

from anndata import AnnData
from spatialdata import SpatialData

from harpy.table.io._add_table import add_table as _add_table
from harpy.utils._keys import _INSTANCE_KEY, _REGION_KEY


def add_table(
    sdata: SpatialData,
    adata: AnnData,
    output_table_name: str,
    region: list[str] | None,
    instance_key: str = _INSTANCE_KEY,
    region_key: str = _REGION_KEY,
    overwrite: bool = False,
) -> SpatialData:
    """Add an AnnData table to SpatialData.

    .. deprecated:: 0.5.0
       `harpy.tb.add_table` moved to :func:`harpy.tb.io.add_table`, which takes the same arguments, and will be
       removed in a future release.
    """
    warnings.warn(
        "harpy.tb.add_table is deprecated since version 0.5.0 and will be removed in a future release. "
        "Use harpy.tb.io.add_table, which takes the same arguments.",
        FutureWarning,
        stacklevel=2,
    )
    return _add_table(
        sdata,
        adata,
        output_table_name,
        region,
        instance_key=instance_key,
        region_key=region_key,
        overwrite=overwrite,
    )
