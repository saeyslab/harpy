"""Find the coordinate systems that Harpy records metadata for, and the one of an element."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import pandas as pd
from spatialdata import SpatialData
from spatialdata.transformations import get_transformation

from harpy.metadata._registry import _COORDINATE_SYSTEMS_METADATA_KEY, _HARPY_METADATA_KEY

# The name of the index of get_coordinate_system_metadata.
_INDEX_NAME = "coordinate_system"


def get_coordinate_system_metadata(sdata: SpatialData) -> pd.DataFrame:
    """Return the Harpy metadata of the coordinate systems of ``sdata``, one row per coordinate system.

    Readers record this metadata in ``sdata.attrs["harpy"]["coordinate_systems"]``;
    :func:`harpy.io.cosmx` records one coordinate system per mosaic. Find the
    coordinate system of an element with :func:`harpy.get_element_coordinate_system`.

    Parameters
    ----------
    sdata
        SpatialData object.

    Returns
    -------
    pandas.DataFrame
        Indexed by coordinate system, ``coordinate_system``, with one column
        per field of the metadata, in the order the fields first occur, as
        their writer documents them, such as :func:`harpy.io.cosmx`. A field
        that a coordinate system's metadata lacks is missing, and nested values
        are copies of those in ``sdata.attrs``. Without coordinate systems with
        Harpy metadata, the DataFrame is empty.

    Raises
    ------
    ValueError
        If ``sdata.attrs["harpy"]``, its ``coordinate_systems``, or the metadata
        of a coordinate system is present but not a mapping.

    Examples
    --------
    The mosaic, sample and pixel size of the labels that a table annotates:

    .. code-block:: python

        metadata = hp.get_coordinate_system_metadata(sdata)
        coordinate_system = hp.get_element_coordinate_system(sdata, "s1_cellpose_labels_mosaic_1")
        metadata.loc[coordinate_system, ["sample_id", "mosaic", "microns_per_unit"]]
    """
    records = _coordinate_system_records(sdata)
    rows = [deepcopy(dict(record)) for record in records.values()]
    return pd.DataFrame(
        rows,
        index=pd.Index(list(records), name=_INDEX_NAME, dtype=object),
        columns=list(dict.fromkeys(key for row in rows for key in row)),
    )


def get_element_coordinate_system(sdata: SpatialData, element_name: str) -> str | None:
    """Return the coordinate system with Harpy metadata that an element lies in.

    A reader records metadata for the coordinate systems it writes, such as
    one per CosMx mosaic, in ``sdata.attrs["harpy"]["coordinate_systems"]``;
    see :func:`harpy.get_coordinate_system_metadata`. The coordinate system
    of an element is the one with Harpy metadata to which it has a
    transformation. Harpy's functions copy the transformations of an element
    to the elements they derive from it, so a segmentation of a morphology
    image, also of a crop of it, lies in the image's coordinate system, without
    metadata of its own::

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
        The coordinate system with Harpy metadata to which the element has a
        transformation, or ``None`` if it has none. Transformations to
        coordinate systems without Harpy metadata, such as one added to align
        samples, are ignored.

    Raises
    ------
    ValueError
        If ``element_name`` is a table, which has no transformations, or not an
        element of ``sdata``; or if the element has transformations to two or
        more coordinate systems with Harpy metadata. It checks only what
        concerns this element; validate the whole store with
        :func:`harpy.validate_coordinate_system_metadata`.

    Examples
    --------
    The mosaic of a table's labels, and the images of that mosaic, found
    without parsing element names:

    .. code-block:: python

        coordinate_system = hp.get_element_coordinate_system(sdata, "s1_cellpose_labels_mosaic_1")
        mosaic = hp.get_coordinate_system_metadata(sdata).loc[coordinate_system, "mosaic"]
        images = [name for name in sdata.images if hp.get_element_coordinate_system(sdata, name) == coordinate_system]
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


def validate_coordinate_system_metadata(sdata: SpatialData) -> None:
    """Validate the Harpy metadata of the coordinate systems of ``sdata``, without modifying it.

    A reader records metadata for the coordinate systems it writes, such as
    one per CosMx mosaic, in ``sdata.attrs["harpy"]["coordinate_systems"]``.
    Whoever writes it, the metadata follows these rules:

    1. ``sdata.attrs["harpy"]["coordinate_systems"]`` maps names of
       coordinate systems to mappings of their fields.
    2. Every coordinate system with Harpy metadata exists in ``sdata``: some
       element has a transformation to it. Renaming the coordinate system, or
       deleting its last element, leaves metadata that describes nothing.
    3. Every image, labels, points and shapes element, read or derived, has
       transformations to at most one coordinate system with Harpy metadata,
       which :func:`harpy.get_element_coordinate_system` then returns. To
       align elements across such coordinate systems, use a coordinate system
       without Harpy metadata.

    The fields of the metadata belong to its writer, which validates them:
    :func:`harpy.io.validate_cosmx_store` calls this function, then checks the
    fields that :func:`harpy.io.cosmx` writes. Without Harpy metadata of
    coordinate systems, ``sdata`` is valid.

    Parameters
    ----------
    sdata
        SpatialData object, in memory or backed.

    Raises
    ------
    ValueError
        If a rule is broken. The message names the coordinate systems or
        elements that break it, and how to fix them.

    Examples
    --------
    .. code-block:: python

        hp.validate_coordinate_system_metadata(sdata)
    """
    records = _coordinate_system_records(sdata)
    if not records:
        return
    existing_coordinate_systems = set(sdata.coordinate_systems)
    stale = [name for name in records if name not in existing_coordinate_systems]
    if stale:
        paths = ", ".join(f"{_HARPY_METADATA_KEY}.{_COORDINATE_SYSTEMS_METADATA_KEY}[{name!r}]" for name in stale)
        raise ValueError(
            f"Harpy metadata {paths} records coordinate system {', '.join(repr(name) for name in stale)}, to which "
            "no element of sdata has a transformation. Restore its elements, rename the coordinate system back, "
            "or remove its metadata."
        )
    ambiguous = {}
    for _, element_name, element in sdata.gen_spatial_elements():
        matches = sorted(set(get_transformation(element, get_all=True)) & set(records))
        if len(matches) > 1:
            ambiguous[element_name] = matches
    if ambiguous:
        described = "; ".join(f"{name!r} to {matches}" for name, matches in ambiguous.items())
        raise ValueError(
            f"Elements have transformations to several coordinate systems with Harpy metadata: {described}. Keep "
            "each element's transformation to only one of them; to align elements across them, use a coordinate "
            "system without Harpy metadata."
        )


def _coordinate_system_records(sdata: SpatialData) -> Mapping[str, Any]:
    """Return ``sdata.attrs["harpy"]["coordinate_systems"]``, or an empty mapping if it is absent.

    Raises
    ------
    ValueError
        If ``sdata.attrs["harpy"]``, its ``coordinate_systems``, or the metadata of a coordinate system is
        present but not a mapping.
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
    for name, record in records.items():
        if not isinstance(record, Mapping):
            raise ValueError(
                f"SpatialData attribute '{_HARPY_METADATA_KEY}.{_COORDINATE_SYSTEMS_METADATA_KEY}' must map each "
                f"coordinate system to a mapping, found {type(record).__name__} for {name!r}."
            )
    return records
