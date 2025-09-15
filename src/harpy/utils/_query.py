from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import dask.array as da
import numpy as np
from spatialdata import SpatialData, read_zarr
from spatialdata import bounding_box_query as bounding_box_query_spatialdata
from spatialdata.transformations import get_transformation

from harpy.image._image import _get_spatial_element, add_labels_layer
from harpy.table._table import add_table_layer
from harpy.utils._keys import _INSTANCE_KEY, _REGION_KEY
from harpy.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def bounding_box_query(
    sdata: SpatialData,
    labels_layer: str | Iterable[str],
    to_coordinate_system: str | Iterable[str] | None,
    crd: tuple[int, int, int, int] | Iterable[tuple[int, int, int, int] | None] | None,
    copy_img_layer: bool = True,
    copy_shapes_layer: bool = True,
    copy_points_layer: bool = True,
    output: str | Path | None = None,
) -> SpatialData:
    """
    Query the labels layers of a SpatialData object and the corresponding instances it annotates in `sdata.tables` via a bounding box query.

    Parameters
    ----------
    sdata
        The SpatialData object to query.
    labels_layer
        The labels layer(s) to query, which can be a single string or an iterable of strings.
    to_coordinate_system
        The coordinate system(s) to which the query provided via `crd` is defined. If `None`, will use 'global'.
    crd
        Coordinates defining the bounding box, specified as a tuple of four integers (x_min, y_min, x_max, y_max), or an iterable of such tuples.
        Setting `crd` to `None` can be usefull if you want to filter elments in tables layers that are annotated by specific labels layers.
        E.g. setting `labels_layer=layer_1` and `crd=None`, will result in AnnData objects in `sdata.tables` containing only instances annotated by `layer_1`.
    copy_img_layer
        Whether to copy all image layers to the new SpatialData object. If set to `False`, image layers will not be included in the result.
    copy_shapes_layer
        Whether to copy all shapes layers to the new SpatialData object. If set to `False`, shapes layers will not be included in the result.
    copy_points_layer
        Whether to copy all points layers to the new SpatialData object. If set to `False`, points layers will not be included in the result.
    output
        Path to the zarr store where the resulting SpatialData object will be backed.
        If None, the new resulting SpatialData object will be persisted in memory.

    Returns
    -------
    A new SpatialData object containing the extracted bounding box region and associated data layers.

    Raises
    ------
    AssertionError
        If the number of provided `labels_layer`, `crd` and `to_coordinate_system` is not equal.
    """

    def _fix_name(name: str | Iterable[str]):
        return list(name) if isinstance(name, Iterable) and not isinstance(name, str) else [name]

    # make all input iterables
    labels_layer = _fix_name(labels_layer)
    crd = _crd_to_iterable_of_iterables(crd)
    to_coordinate_system = _fix_name(to_coordinate_system)
    assert len(labels_layer) == len(crd) == len(to_coordinate_system), (
        "The number of 'labels_layer', 'crd' and 'to_coordinate_system' specified should all be equal."
    )

    sdata_queried = SpatialData()
    # back resulting sdata to zarr store if output is specified
    if output is not None:
        sdata_queried.write(output)

    # first add queried labels layer to sdata, so we do not have to query them + calculate unique labels in them len[*sdata.tables] times.
    labels_ids = []
    # labels_layer_queried=[]
    for _labels_layer, _crd, _to_coordinate_system in zip(labels_layer, crd, to_coordinate_system, strict=True):
        se = _get_spatial_element(sdata, layer=_labels_layer)

        if _crd is None:
            se_queried = se
        else:
            se_queried = bounding_box_query_spatialdata(
                se,
                axes=("y", "x"),  # for now only support bounding box query in y and x via crd
                min_coordinate=[_crd[2], _crd[0]],
                max_coordinate=[_crd[3], _crd[1]],
                target_coordinate_system=_to_coordinate_system,
            )

        if se_queried is None:
            # set labels_id to empty array, so for this labels layer, all instances will be removed from table.
            labels_ids.append(np.array([]))
            log.warning(
                f"query with crd {crd} to coordinate system '{to_coordinate_system}' for labels layer '{labels_layer}' resulted in empty labels layer."
                f"Therefore labels layer '{labels_layer}' will not be present in resulting spatialdata object. "
                f"Instances in tables annotated by '{labels_layer}' will also be removed from the tables."
            )
        else:
            labels_ids.append(da.unique(se_queried.data).compute())
            sdata_queried = add_labels_layer(
                sdata_queried,
                arr=se_queried.data.rechunk(
                    se_queried.data.chunksize
                ),  # rechunk to avoid irregular chunks when writing to zarr store.
                output_layer=_labels_layer,
                transformations=get_transformation(se_queried, get_all=True),
                # note that if labels layer was multiscale, it will now not be multiscale.
            )

    # now query the associated table layer
    for _table_layer in [*sdata.tables]:
        region = []
        adata = sdata.tables[_table_layer]
        remove = np.ones(len(adata), dtype=bool)

        for _labels_layer, _crd, _to_coordinate_system, _labels_id in zip(
            labels_layer,
            crd,
            to_coordinate_system,
            labels_ids,
            strict=True,
        ):
            # the code also handles case when labels layer does not annotate the table
            # (i.e. _labels_layer not in adata.obs[_REGION_KEY].values), because remove already set to True for all instances.
            if _labels_layer in adata.obs[_REGION_KEY].values:
                mask_label = adata.obs[_REGION_KEY] == _labels_layer

                remove_label = ~((mask_label) & (adata.obs[_INSTANCE_KEY].isin(_labels_id))).to_numpy()
                remove = remove & remove_label
                if remove_label[
                    (mask_label).to_numpy()
                ].all():  # if all instances from this label layer will be removed from adata, then we do not append _labels_layer to region
                    continue
                else:
                    region.append(_labels_layer)

        if remove.all():
            log.info(
                f"Query removed all instances from table layer '{_table_layer}'. "
                "Therefore this table layer will not be present in resulting spatialdata object."
            )
            continue
        # subset
        adata = adata[~remove].copy()
        # add subsetted adata to sdata
        sdata_queried = add_table_layer(
            sdata_queried,
            adata=adata,
            output_layer=_table_layer,
            region=region,
        )

    # now copy image, shapes and points layer if copy is True.
    layers_to_copy = []

    if copy_img_layer:
        layers_to_copy.extend([*sdata.images])
    if copy_shapes_layer:
        layers_to_copy.extend([*sdata.shapes])
    if copy_points_layer:
        layers_to_copy.extend(*[sdata.points])

    for _layer_to_copy in layers_to_copy:
        sdata_queried[_layer_to_copy] = sdata[_layer_to_copy]
        if sdata_queried.is_backed():
            sdata_queried.write_element(_layer_to_copy)

    # if backed, and if there were layers copied, we read back from zarr, otherwise sdata_queried not self contained
    if sdata_queried.is_backed() and layers_to_copy:
        sdata_queried = read_zarr(sdata_queried.path)

    return sdata_queried


def _crd_to_iterable_of_iterables(
    crd: tuple[int, int, int, int] | Iterable[tuple[int, int, int, int] | None] | None,
) -> Iterable[tuple[int, int, int, int]]:
    _iterable_elements = False
    if crd is not None:
        for element in crd:
            # check if any of the elements is iterable, if so, we are probably dealing with an Iterable of Iterables.
            if (isinstance(element, Iterable) and not isinstance(element, str)) or element is None:
                _iterable_elements = True
                break

    if not _iterable_elements:
        crd = [crd]

    return crd
