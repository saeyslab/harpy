"""Light wrapper around spatialdata-plot"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import pandas as pd
from matplotlib.axes import Axes
from spatialdata import SpatialData, bounding_box_query

from harpy.utils._keys import _REGION_KEY
from harpy.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


try:
    import spatialdata_plot  # noqa: F401
except ImportError:
    log.warning(
        "Module 'spatialdata-plot' not installed, please install 'spatialdata-plot' if you want to use 'hp.pl.plot_spatialdata'."
    )


def plot_spatialdata(
    sdata: SpatialData,
    img_layer: str,
    channel: str,
    labels_layer: str | None = None,
    table_layer: str | None = None,
    ax: Axes | None = None,
    crd: tuple[int, int, int, int] | None = None,
    to_coordinate_system: str = "global",
    region_key: str = _REGION_KEY,
    render_images_kwargs: Mapping[str, Any] = MappingProxyType({}),
    render_labels_kwargs: Mapping[str, Any] = MappingProxyType({}),
    show_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> Axes:
    # TODO write docstring. Write unit tests.
    if table_layer is not None and labels_layer is None:
        raise ValueError(
            f"Please specify a labels layer (which annotates the table layer '{table_layer}') if 'table_layer' is specified."
        )

    if table_layer is not None:
        adata = sdata[table_layer]
        # sanity check
        mask = adata.obs[region_key] == labels_layer
        if not mask.any():
            raise ValueError(
                f"The labels layer '{labels_layer}' does not seem to annotate the table layer '{table_layer}."
            )

    sdata_to_plot = sdata
    queried = False
    if crd is not None:
        queried = True
        sdata_to_plot = bounding_box_query(
            sdata,
            axes=["x", "y"],
            min_coordinate=[crd[0], crd[2]],
            max_coordinate=[crd[1], crd[3]],
            target_coordinate_system=to_coordinate_system,
            filter_table=True,
        )

    if labels_layer is None:
        ax = sdata_to_plot.pl.render_images(
            img_layer,
            channel=channel,
            **render_images_kwargs,
        ).pl.show(
            **show_kwargs,
            ax=ax,
            return_ax=True,
        )

    else:
        if "table_name" in render_labels_kwargs.keys():
            raise ValueError(
                "Please specify 'table_name' via the keyword argument 'table_layer' of 'plot_spatialdata.'"
            )
        # workaround for https://github.com/scverse/spatialdata-plot/issues/414, also see
        # https://github.com/ArneDefauw/spatialdata-plot/blob/5af65aa118f7abf87e47470038ecdbddb27ef1ca/tests/pl/test_render_labels.py#L250
        if "color" in render_labels_kwargs.keys():
            if queried:
                color = render_labels_kwargs["color"]
                if (
                    color is not None
                    and color in sdata_to_plot[table_layer].obs
                    and pd.api.types.is_categorical_dtype(sdata_to_plot[table_layer].obs[color])
                ):
                    sdata_to_plot[table_layer].obs[color] = (
                        sdata_to_plot[table_layer].obs[color].cat.remove_unused_categories()
                    )

        ax = (
            sdata_to_plot.pl.render_images(
                img_layer,
                channel=channel,
                **render_images_kwargs,
            )
            .pl.render_labels(
                labels_layer,
                table_name=table_layer,
                **render_labels_kwargs,
            )
            .pl.show(
                **show_kwargs,
                ax=ax,
                return_ax=True,
            )
        )

    return ax
