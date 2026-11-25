"""Harpy's metadata in ``sdata.attrs["harpy"]``: its registries, and functions to read and validate them.

The package imports nothing from Harpy's other subpackages, only SpatialData and pandas, so that every
subpackage can import it without circular imports. Its public functions are exported at the top level, as
``hp.get_coordinate_system_metadata``, ``hp.get_element_coordinate_system`` and
``hp.validate_coordinate_system_metadata``.
"""

from harpy.metadata._coordinate_systems import (
    get_coordinate_system_metadata,
    get_element_coordinate_system,
    validate_coordinate_system_metadata,
)

__all__ = ["get_coordinate_system_metadata", "get_element_coordinate_system", "validate_coordinate_system_metadata"]
