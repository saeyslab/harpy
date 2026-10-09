"""Reading and writing tables, in a store and attached to SpatialData."""

from harpy.table.io._add_table import add_table
from harpy.table.io._components import add_table_components, remove_table_components
from harpy.table.io._components_by_region import add_table_components_by_region
from harpy.table.io._read import read_table, read_table_components
from harpy.table.io._updates import add_table_updates, write_table_updates
from harpy.table.io._write import delete_table_components, write_table, write_table_components
from harpy.table.io._write_by_region import write_table_components_by_region

__all__ = [
    "read_table",
    "read_table_components",
    "write_table",
    "write_table_updates",
    "write_table_components",
    "write_table_components_by_region",
    "delete_table_components",
    "add_table",
    "add_table_updates",
    "add_table_components",
    "add_table_components_by_region",
    "remove_table_components",
]
