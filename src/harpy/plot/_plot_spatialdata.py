"""Light wrapper around spatialdata-plot"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import dask.array as da
import spatialdata_plot  # noqa: F401
from matplotlib.axes import Axes
from spatialdata import SpatialData, bounding_box_query
from spatialdata.models import TableModel

from harpy.image._image import _get_spatial_element
from harpy.utils._keys import _INSTANCE_KEY, _REGION_KEY


def plot_spatialdata(
    sdata: SpatialData,
    img_layer: str,
    channel: str,
    labels_layer: str | None,
    table_layer: str | None,
    ax: Axes | None = None,
    crd: tuple[int, int, int, int] | None = None,
    to_coordinate_system: str = "global",
    render_images_kwargs: Mapping[str, Any] = MappingProxyType({}),
    render_labels_kwargs: Mapping[str, Any] = MappingProxyType({}),
    show_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> Axes:
    if table_layer is not None and labels_layer is None:
        raise ValueError(
            f"Please specify a labels layer (which annotates the table layer {table_layer}) if 'table_layer' is specified."
        )

    if table_layer is not None:
        adata = sdata[table_layer]
        # subset
        # TODO need to check that we do not alter underlying spatialdata object.
        adata = adata[adata.obs[_REGION_KEY] == labels_layer]
        if adata.shape[0] == 0:
            raise ValueError(
                f"The labels layer '{labels_layer}' does not seem to annotate the table layer '{table_layer}."
            )

    img_layer_crop = f"{img_layer}_{uuid.uuid4()}"
    labels_layer_crop = f"{labels_layer}_{uuid.uuid4()}"
    table_layer_crop = f"{table_layer}_{uuid.uuid4()}"
    if crd is not None:
        sdata[img_layer_crop] = bounding_box_query(
            sdata[img_layer],
            axes=["x", "y"],
            min_coordinate=[crd[0], crd[2]],
            max_coordinate=[crd[1], crd[3]],
            target_coordinate_system=to_coordinate_system,
        )
        if labels_layer is not None:
            sdata[labels_layer_crop] = bounding_box_query(
                sdata[labels_layer],
                axes=["x", "y"],
                min_coordinate=[crd[0], crd[2]],
                max_coordinate=[crd[1], crd[3]],
                target_coordinate_system=to_coordinate_system,
            )
            if table_layer is not None:
                # annotate with labels_layer_crop
                adata.obs[_REGION_KEY] = labels_layer_crop
                adata.obs[_REGION_KEY] = adata.obs[_REGION_KEY].astype("category")
                # subset adata
                se = _get_spatial_element(sdata, layer=labels_layer_crop)
                labels_crop = da.unique(se.data).compute()
                adata = adata[adata.obs[_INSTANCE_KEY].isin(labels_crop)]
                adata.uns.pop(TableModel.ATTRS_KEY)
                adata = TableModel.parse(
                    adata=adata,
                    region=labels_layer_crop,
                    region_key=_REGION_KEY,
                    instance_key=_INSTANCE_KEY,
                )
                # add subsetted adata to spatialdata object
                sdata[table_layer_crop] = adata

    if labels_layer is None:
        ax = sdata.pl.render_images(
            img_layer if crd is None else img_layer_crop,
            channel=channel,
            **render_images_kwargs,
        ).pl.show(
            **show_kwargs,
            ax=ax,
            return_ax=True,
        )

    else:
        assert "table_name" not in render_labels_kwargs.keys()  # TODO raise valueerror if provided.
        ax = (
            sdata.pl.render_images(
                img_layer if crd is None else img_layer_crop,
                channel=channel,
                **render_images_kwargs,
            )
            .pl.render_labels(
                labels_layer if crd is None else labels_layer_crop,
                table_name=table_layer if crd is None else table_layer_crop,
                **render_labels_kwargs,
            )
            .pl.show(
                **show_kwargs,
                ax=ax,
                return_ax=True,
            )
        )

    if crd is not None:
        del sdata[img_layer_crop]
        if labels_layer is not None:
            del sdata[labels_layer_crop]
            if table_layer is not None:
                del sdata[table_layer_crop]

    return ax
