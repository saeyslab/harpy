from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from spatialdata import SpatialData
from spatialdata.models import TableModel, get_table_keys
from spatialdata.transformations import Identity, get_transformation, get_transformation_between_coordinate_systems

from harpy._metadata import _HARPY_METADATA_KEY, _IMAGES_METADATA_KEY
from harpy.io._cosmx._constants import _FOV_BOUNDS_KEY, _SAMPLE_ID_KEY
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
                  │  mask -> CosMx morphology image of the same mosaic,
                  │  through a coordinate system the two share
                  v
        centers in the image's intrinsic frame
                  │
                  │  half-open bounds test against the image's record,
                  │  sdata.attrs["harpy"]["images"][image]["fov_bounds_px"]
                  v
        table.obs[fov_key], table.obs[sample_key]

    The canonical centers come from :func:`harpy.tb.aggregate_points` or
    :func:`harpy.tb.add_canonical_centers`. The mask may lie on a crop or
    rescaled copy of the morphology image, as long as its transformations place
    it in the mosaic's coordinate systems.

    The columns are added to ``sdata.tables[table_name].obs`` in place, in
    memory, and the function returns ``None``; nothing is written to the
    store. To store them, write ``obs`` back with
    :func:`harpy.tb.io.add_table_updates` and ``overwrite=True``, as ``obs``
    already exists in the store.

    Parameters
    ----------
    sdata
        SpatialData with the table, its segmentation masks and the CosMx
        morphology images of their mosaics, as written by
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
        instance column, or the two columns share a name; if the store has no
        CosMx morphology images; if a segmentation mask is missing from
        ``sdata``, or shares a coordinate system with no CosMx morphology
        image, or only with several; or if the matched image's record has no
        ``fov_bounds_px``. The table is unchanged then.

    Notes
    -----
    The bounds for a segmentation mask are read from one CosMx morphology
    image: the image that shares a coordinate system with the mask and with no
    other CosMx morphology image. The reader gives each morphology image its
    own pixel and micron coordinate systems, such as ``<sample>_global_1`` and
    ``<sample>_global_1_micron``, and a mask segmented on that image, or on a
    crop or copy of it, places itself in them as well. Only images written by
    the reader, registered in ``sdata.attrs["harpy"]["images"]``, count;
    derived images, such as preprocessed copies, are ignored. A coordinate
    system that several CosMx morphology images share, such as one added to
    align samples, cannot tell them apart, so it is ignored.

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
    image_records = (sdata.attrs.get(_HARPY_METADATA_KEY) or {}).get(_IMAGES_METADATA_KEY) or {}
    if not image_records:
        raise ValueError("SpatialData has no CosMx morphology images in sdata.attrs['harpy']['images'].")

    regions = table.obs[region_key].astype(str).to_numpy()
    # (y, x) of each row's center, in the intrinsic frame of that row's own mask.
    centers = np.asarray(table.obsm[CANONICAL_OBSM_KEY])[:, 1:]
    fovs = np.zeros(table.n_obs, dtype=np.int64)
    sample_ids = np.empty(table.n_obs, dtype=object)
    for labels_name in pd.unique(regions):
        rows = regions == labels_name
        fovs[rows], sample_ids[rows] = _assign_region(sdata, labels_name, centers[rows], image_records=image_records)

    table.obs[fov_key] = pd.arrays.IntegerArray(fovs, mask=fovs == 0)
    table.obs[sample_key] = pd.Categorical(sample_ids)


def _assign_region(
    sdata: SpatialData,
    labels_name: str,
    centers: np.ndarray,
    *,
    image_records: Mapping[str, Mapping[str, object]],
) -> tuple[np.ndarray, str]:
    """Return the FOV of each ``(y, x)`` center of one mask, 0 outside every FOV, and the mask's sample."""
    if labels_name not in sdata.labels:
        raise ValueError(f"Segmentation mask {labels_name!r} of the table is not a labels element of sdata.")
    image_name, coordinate_system = _cosmx_image_of(sdata, labels_name, image_records=image_records)
    record = image_records[image_name]
    bounds = record.get(_FOV_BOUNDS_KEY)
    if not isinstance(bounds, Mapping):
        raise ValueError(
            f"CosMx image {image_name!r} has no {_FOV_BOUNDS_KEY} in its record. Stores written by older Harpy "
            "versions lack it; run the CosMx reader again to rewrite the store."
        )

    # From the mask's frame to the CosMx image's frame, in which the bounds are stored.
    transformation = get_transformation_between_coordinate_systems(
        sdata, sdata[labels_name], sdata[image_name], intermediate_coordinate_systems=coordinate_system
    )
    matrix = transformation.to_affine_matrix(input_axes=("y", "x"), output_axes=("y", "x"))
    yx = centers @ matrix[:2, :2].T + matrix[:2, 2]

    # Integer indices are pixel centers, so the pixel containing a center c is floor(c + 0.5).
    y, x = np.floor(yx + 0.5).astype(np.int64).T
    fovs = np.zeros(len(yx), dtype=np.int64)
    for fov, b in bounds.items():
        fovs[(b["y0"] <= y) & (y < b["y1"]) & (b["x0"] <= x) & (x < b["x1"])] = int(fov)
    return fovs, str(record[_SAMPLE_ID_KEY])


def _cosmx_image_of(
    sdata: SpatialData,
    labels_name: str,
    *,
    image_records: Mapping[str, Mapping[str, object]],
) -> tuple[str, str]:
    """Return the CosMx morphology image of the mosaic that ``labels_name`` lies on, and a coordinate system they share.

    Each mosaic has its own pixel and micron coordinate systems, so a
    coordinate system that the mask shares with exactly one CosMx image
    identifies its mosaic. Registered images missing from a partially read
    store are not candidates. Of the shared coordinate systems, the image's
    own pixel grid is preferred.
    """
    image_names_by_coordinate_system: dict[str, set[str]] = {}
    for image_name in sorted(set(image_records) & set(sdata.images)):
        for coordinate_system in get_transformation(sdata[image_name], get_all=True):
            image_names_by_coordinate_system.setdefault(coordinate_system, set()).add(image_name)
    links = {
        coordinate_system: next(iter(candidates))
        for coordinate_system in get_transformation(sdata[labels_name], get_all=True)
        if len(candidates := image_names_by_coordinate_system.get(coordinate_system, set())) == 1
    }
    if len(set(links.values())) != 1:
        raise ValueError(
            f"Segmentation mask {labels_name!r} must share a coordinate system with exactly one CosMx morphology "
            f"image, found {links or 'none'}."
        )
    image_name = next(iter(links.values()))
    # Prefer the image's own pixel grid, the coordinate system in which its transformation is the
    # identity: the composed transformation is then the mask's own, which keeps a crop exact.
    on_pixel_grid = [
        coordinate_system
        for coordinate_system in links
        if isinstance(get_transformation(sdata[image_name], to_coordinate_system=coordinate_system), Identity)
    ]
    return image_name, min(on_pixel_grid or links)
