"""Find the coordinate systems that Harpy records metadata for, and the one of an element."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from spatialdata import SpatialData
from spatialdata.transformations import get_transformation

from harpy.metadata._registry import _COORDINATE_SYSTEMS_METADATA_KEY, _HARPY_METADATA_KEY


def _coordinate_system_records(sdata: SpatialData) -> Mapping[str, Any]:
    """Return ``sdata.attrs["harpy"]["coordinate_systems"]``, or an empty mapping if it is absent.

    Raises
    ------
    ValueError
        If ``sdata.attrs["harpy"]`` or its ``coordinate_systems`` is present but not a mapping.
    """
    harpy_metadata = sdata.attrs.get(_HARPY_METADATA_KEY)
    if harpy_metadata is None:
        return {}
    if not isinstance(harpy_metadata, Mapping):
        raise ValueError(f"SpatialData attribute {_HARPY_METADATA_KEY!r} must be a mapping.")
    records = harpy_metadata.get(_COORDINATE_SYSTEMS_METADATA_KEY)
    if records is None:
        return {}
    if not isinstance(records, Mapping):
        raise ValueError(
            f"SpatialData attribute '{_HARPY_METADATA_KEY}.{_COORDINATE_SYSTEMS_METADATA_KEY}' must be a mapping."
        )
    return records


def _element_coordinate_system(sdata: SpatialData, element_name: str) -> str | None:
    """Return the coordinate system with Harpy metadata that an element has a transformation to.

    A reader records metadata for the coordinate systems it writes, such as one
    per CosMx mosaic, in ``sdata.attrs["harpy"]["coordinate_systems"]``.
    Harpy's functions copy the transformations of an element to the elements
    they derive from it, so a derived element, such as labels segmented on a
    crop of a morphology image, lies in the same coordinate system as its
    source::

        image   s1_morphology_image_mosaic_1   {"s1_global_1": Identity()}
        labels  s1_cellpose_labels_mosaic_1    {"s1_global_1": Translation(...)}
            -> "s1_global_1", for both

    Parameters
    ----------
    sdata
        SpatialData object with the element.
    element_name
        Name of an image, labels, points or shapes element of ``sdata``.

    Returns
    -------
    str or None
        The one coordinate system with Harpy metadata to which the element has a
        transformation, or ``None`` if it has none.

    Raises
    ------
    ValueError
        If ``element_name`` is a table, which has no transformations, or not an
        element of ``sdata``; or if the element has transformations to several
        coordinate systems with Harpy metadata.
    """
    if element_name in sdata.tables:
        raise ValueError(
            f"{element_name!r} is a table, which has no transformations. Look up the labels elements that it "
            "annotates, its regions, instead."
        )
    if element_name not in sdata:
        raise ValueError(f"{element_name!r} is not an element of sdata.")
    records = _coordinate_system_records(sdata)
    matches = sorted(set(get_transformation(sdata[element_name], get_all=True)) & set(records))
    if len(matches) > 1:
        raise ValueError(
            f"Element {element_name!r} has transformations to several coordinate systems with Harpy metadata, "
            f"{matches}, so it is ambiguous which one describes it. Keep its transformation to only one of them; "
            "to align elements across them, use a coordinate system without Harpy metadata."
        )
    return matches[0] if matches else None
