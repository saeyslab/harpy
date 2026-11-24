from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from spatialdata import SpatialData
from spatialdata.models import TableModel, get_table_keys
from spatialdata.transformations import get_transformation

from harpy.io._cosmx._constants import _FOV_BOUNDS_KEY
from harpy.metadata._coordinate_systems import _coordinate_system_records, _element_coordinate_system
from harpy.metadata._registry import _COORDINATE_SYSTEMS_METADATA_KEY, _HARPY_METADATA_KEY, _SAMPLE_ID_KEY
from harpy.table.canonical_centers import CANONICAL_OBSM_KEY


def assign_cosmx_fovs(
    sdata: SpatialData,
    *,
    table_name: str,
    fov_key: str = "fov",
    sample_key: str = "sample_id",
) -> None:
    """Add the CosMx sample and FOV of each row of a table to its ``.obs``, in place.

    For each segmentation mask that the table annotates, such as a custom
    segmentation, on a mosaic (a group of adjacent FOVs stitched into one
    image) written by :func:`harpy.io.cosmx`::

        table.obsm["spatial_canonical"]          centers, in the mask's intrinsic frame
                  │
                  │  mask -> the mosaic's coordinate system, the one with a
                  │  record in sdata.attrs["harpy"]["coordinate_systems"]
                  v
        centers in the mosaic's coordinate system
                  │
                  │  half-open bounds test against the record's fov_bounds
                  v
        table.obs[fov_key], table.obs[sample_key]

    The canonical centers come from :func:`harpy.tb.aggregate_points` or
    :func:`harpy.tb.add_canonical_centers`. The mask may lie on a crop or
    rescaled copy of the morphology image, as long as its transformations place
    it in the mosaic's coordinate system.

    The columns are added to ``sdata.tables[table_name].obs`` in place, in
    memory, and the function returns ``None``; nothing is written to the
    store. To store them, write ``obs`` back with
    :func:`harpy.tb.io.add_table_updates` and ``overwrite=True``, as ``obs``
    already exists in the store.

    Parameters
    ----------
    sdata
        SpatialData with the table and its segmentation masks, and the records
        of the mosaics' coordinate systems, as written by
        :func:`harpy.io.cosmx`.
    table_name
        Regions table whose regions are segmentation masks, with canonical
        centers.
    fov_key
        Column of ``.obs`` to add the FOV numbers to, as pandas' nullable
        ``Int64``: ``<NA>`` when the center lies in a gap between FOVs or
        outside the mosaic. An existing column is replaced.
    sample_key
        Column of ``.obs`` to add the CosMx samples to, categorical, also for
        rows without a FOV. FOV numbers repeat across samples, so the two
        columns together identify a FOV. An existing column is replaced.

    Raises
    ------
    ValueError
        If the table is missing, is not a regions table or has no canonical
        centers; if a column to add would replace the table's region or
        instance column, or the two columns share a name; if the store records
        no coordinate systems; if a segmentation mask is missing from
        ``sdata``, or has a transformation to no recorded coordinate system,
        or to several; or if the record of its coordinate system has no
        ``fov_bounds``. The table is unchanged then.

    Notes
    -----
    The reader records each mosaic's facts, such as its sample and the bounds
    of its FOVs, in ``sdata.attrs["harpy"]["coordinate_systems"]``, under the
    name of the mosaic's pixel coordinate system, such as
    ``<sample>_global_1``. A mask segmented on the mosaic's morphology image,
    or on a crop or copy of it, keeps the image's transformation to that
    coordinate system, and so identifies its mosaic. The bounds are in units
    of that coordinate system, the intrinsic frame of the reader's elements. A
    coordinate system without a record, such as one added to align samples, is
    ignored.

    Examples
    --------
    .. code-block:: python

        hp.io.assign_cosmx_fovs(sdata, table_name="table")
        hp.tb.io.add_table_updates(sdata, table_name="table", overwrite=True)
    """
    if table_name not in sdata.tables:
        raise ValueError(f"Table {table_name!r} is not a table of sdata.")
    table = sdata.tables[table_name]
    if TableModel.ATTRS_KEY not in table.uns:
        raise ValueError(f"Table {table_name!r} is not a SpatialData regions table.")
    _, region_key, instance_key = get_table_keys(table)
    if fov_key == sample_key:
        raise ValueError(f"fov_key and sample_key must differ, found {fov_key!r} for both.")
    for key in (fov_key, sample_key):
        if key in (region_key, instance_key):
            raise ValueError(f"Column {key!r} would replace the region or instance column of table {table_name!r}.")
    if CANONICAL_OBSM_KEY not in table.obsm:
        raise ValueError(
            f"Table {table_name!r} has no obsm[{CANONICAL_OBSM_KEY!r}]; "
            "add its canonical centers with harpy.tb.add_canonical_centers first."
        )
    records = _coordinate_system_records(sdata)
    if not records:
        raise ValueError(
            f"SpatialData records no coordinate systems in "
            f"sdata.attrs[{_HARPY_METADATA_KEY!r}][{_COORDINATE_SYSTEMS_METADATA_KEY!r}]; write the store with "
            "harpy.io.cosmx."
        )

    regions = table.obs[region_key].astype(str).to_numpy()
    # (y, x) of each row's center, in the intrinsic frame of that row's own mask.
    centers = np.asarray(table.obsm[CANONICAL_OBSM_KEY])[:, 1:]
    fovs = np.zeros(table.n_obs, dtype=np.int64)
    sample_ids = np.empty(table.n_obs, dtype=object)
    for labels_name in pd.unique(regions):
        rows = regions == labels_name
        fovs[rows], sample_ids[rows] = _assign_region(sdata, labels_name, centers[rows], records=records)

    table.obs[fov_key] = pd.arrays.IntegerArray(fovs, mask=fovs == 0)
    table.obs[sample_key] = pd.Categorical(sample_ids)


def _assign_region(
    sdata: SpatialData,
    labels_name: str,
    centers: np.ndarray,
    *,
    records: Mapping[str, Mapping[str, object]],
) -> tuple[np.ndarray, str]:
    """Return the FOV of each ``(y, x)`` center of one mask, 0 outside every FOV, and the mask's sample."""
    if labels_name not in sdata.labels:
        raise ValueError(f"Segmentation mask {labels_name!r} of the table is not a labels element of sdata.")
    coordinate_system = _element_coordinate_system(sdata, labels_name)
    if coordinate_system is None:
        raise ValueError(
            f"Segmentation mask {labels_name!r} has no transformation to a coordinate system recorded in "
            f"sdata.attrs[{_HARPY_METADATA_KEY!r}][{_COORDINATE_SYSTEMS_METADATA_KEY!r}], {sorted(records)}. "
            "Place it in its mosaic's coordinate system, as hp.im.segment does for labels of a CosMx image."
        )
    record = records[coordinate_system]
    bounds = record.get(_FOV_BOUNDS_KEY)
    if not isinstance(bounds, Mapping):
        raise ValueError(
            f"The record of coordinate system {coordinate_system!r} has no {_FOV_BOUNDS_KEY}; run the CosMx reader "
            "again to rewrite the store."
        )

    # From the mask's frame to the mosaic's coordinate system, in which the bounds are stored.
    transformation = get_transformation(sdata[labels_name], to_coordinate_system=coordinate_system)
    matrix = transformation.to_affine_matrix(input_axes=("y", "x"), output_axes=("y", "x"))
    yx = centers @ matrix[:2, :2].T + matrix[:2, 2]

    # Integer indices are pixel centers, so the pixel containing a center c is floor(c + 0.5).
    y, x = np.floor(yx + 0.5).astype(np.int64).T
    fovs = np.zeros(len(yx), dtype=np.int64)
    for fov, b in bounds.items():
        fovs[(b["y0"] <= y) & (y < b["y1"]) & (b["x0"] <= x) & (x < b["x1"])] = int(fov)
    return fovs, str(record[_SAMPLE_ID_KEY])
