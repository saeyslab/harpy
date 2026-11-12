"""Names and versions of the metadata Harpy stores in a table's ``uns``, and how it is compared and merged.

``docs/development/metadata.md`` describes the layout and its rules. The names
are laid out as the stored metadata. Their suffix says what they hold: ``_KEY``
a key of ``uns``, ``_FIELD`` the name of a field, ``_SCHEMA_VERSION`` and
``_SOURCE_KIND`` a value. Code reads and writes the metadata through these
names; docstrings and messages spell the names out.
"""

from collections.abc import Collection, Mapping, Sequence
from copy import deepcopy

import numpy as np

# Fields that several kinds of metadata share.
_SCHEMA_VERSION_FIELD = "schema_version"
_SOURCE_KIND_FIELD = "source_kind"
# The metadata of each region, keyed by labels element name.
_REGIONS_FIELD = "regions"
# In a regions entry: the SpatialData coordinate system of the region's source.
_COORDINATE_SYSTEM_FIELD = "coordinate_system"

# uns["feature_matrices"][<obsm key>]: the metadata of the matrix adata.obsm[<obsm key>].
_FEATURE_MATRICES_KEY = "feature_matrices"
# Version 2: add_feature_matrix keeps the metadata of each region under "regions".
_FEATURE_MATRIX_SCHEMA_VERSION = 2
_FEATURE_COLUMNS_FIELD = "feature_columns"
# Written by hp.tb.add_feature_matrix.
_ADD_FEATURE_MATRIX_SOURCE_KIND = "harpy_add_feature_matrix"
_SOURCE_CHANNELS_FIELD = "source_channels"
_FEATURES_FIELD = "features"
# In a regions entry: the image element, None for morphology features only.
_IMAGE_ELEMENT_FIELD = "image_element"

# uns["point_aggregation"], written by hp.tb.aggregate_points: the sources of adata.X and the class matrices.
_POINT_AGGREGATION_KEY = "point_aggregation"
_POINT_AGGREGATION_SCHEMA_VERSION = 1
_AGGREGATE_POINTS_SOURCE_KIND = "harpy_aggregate_points"
_FEATURE_KEY_FIELD = "feature_key"
_FEATURE_CLASS_KEY_FIELD = "feature_class_key"
# In a regions entry: the points element aggregated into the region.
_POINTS_ELEMENT_FIELD = "points_element"

# uns["spatial_coordinates"]["spatial_canonical"]: a public contract, named in harpy.table.canonical_centers.

_MISSING = object()


def _comparable_metadata(value: object) -> object:
    """Return a metadata value with arrays as lists and NumPy scalars as Python values, to compare metadata.

    Metadata read from storage holds NumPy arrays and scalars where the same
    metadata in memory holds lists and Python values.
    """
    if isinstance(value, Mapping):
        return {key: _comparable_metadata(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_comparable_metadata(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_comparable_metadata(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _differing_fields(first: Mapping, other: Mapping) -> list[str]:
    """Return the fields whose values differ between two mappings; a field missing from one of them differs."""
    return [
        field for field in dict.fromkeys([*first, *other]) if first.get(field, _MISSING) != other.get(field, _MISSING)
    ]


def _metadata_difference(per_table: Sequence[Mapping]) -> tuple[int, list[str]] | None:
    """Return the first table whose metadata of a matrix differs from ``per_table[0]`` outside ``regions``.

    ``per_table`` holds the metadata of one matrix, one per table. Harpy's
    metadata of a matrix keeps the metadata of each region under ``regions``;
    its other fields describe the whole matrix, and must be equal in the
    tables that concat joins. Returns the table's position and the fields that
    differ, or ``None`` when every table has the same fields. Arrays compare
    as lists.
    """
    first = _comparable_metadata(_without_regions(per_table[0]))
    for position, metadata in enumerate(per_table[1:], start=1):
        fields = _differing_fields(first, _comparable_metadata(_without_regions(metadata)))
        if fields:
            return position, fields
    return None


def _merged_metadata(
    per_table: Sequence[Mapping], *, declared_regions: Sequence[Collection[str]], region_order: Sequence[str]
) -> dict:
    """Merge the metadata of one matrix from the tables that concat joins, one per table in ``per_table``.

    The fields other than ``regions`` come from ``per_table[0]``: check first
    with :func:`_metadata_difference` that they are equal. ``regions`` holds
    each table's entries of the regions it declares, in ``region_order``, and
    leaves out the others. Metadata without ``regions`` has no entries.
    """
    merged = deepcopy(dict(per_table[0]))
    if not any(_REGIONS_FIELD in metadata for metadata in per_table):
        return merged
    entries = {}
    for metadata, regions in zip(per_table, declared_regions, strict=True):
        table_regions = metadata.get(_REGIONS_FIELD, {})
        entries.update({region: table_regions[region] for region in regions if region in table_regions})
    merged[_REGIONS_FIELD] = {region: deepcopy(entries[region]) for region in region_order if region in entries}
    return merged


def _without_regions(metadata: Mapping) -> dict:
    return {field: value for field, value in metadata.items() if field != _REGIONS_FIELD}
