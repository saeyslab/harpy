from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import dask.dataframe as dd
import numpy as np
import pandas as pd
from spatialdata import SpatialData, read_zarr
from spatialdata.models import TableModel
from spatialdata.transformations import get_transformation, set_transformation
from spatialdata_io import xenium as sdata_xenium
from spatialdata_io._constants._constants import XeniumKeys

from sparrow.io._transcripts import read_transcripts
from sparrow.utils._keys import _INSTANCE_KEY, _REGION_KEY
from sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def xenium(
    path: str | Path | list[str] | list[Path],
    to_coordinate_system: str | list[str] = "global",
    aligned_images: bool = True,
    cells_labels: bool = False,
    nucleus_labels: bool = False,
    cells_table: bool = False,
    filter_gene_names: str | list[str] = None,
    output: str | Path | None = None,
) -> SpatialData:
    """
    Read a *10X Genomics Xenium* dataset into a SpatialData object.

    Wrapper around `spatialdata_io.xenium`, but with support for reading multiple samples into one spatialdata object.
    This function reads images, transcripts, masks (cell and nuclei) and tables, so it can be used for analysis.

    This function reads the following files:

        - ``{xx.XENIUM_SPECS!r}``: File containing specifications.
        - ``{xx.NUCLEUS_BOUNDARIES_FILE!r}``: Polygons of nucleus boundaries.
        - ``{xx.CELL_BOUNDARIES_FILE!r}``: Polygons of cell boundaries.
        - ``{xx.TRANSCRIPTS_FILE!r}``: File containing transcripts.
        - ``{xx.CELL_FEATURE_MATRIX_FILE!r}``: File containing cell feature matrix.
        - ``{xx.CELL_METADATA_FILE!r}``: File containing cell metadata.
        - ``{xx.MORPHOLOGY_MIP_FILE!r}``: File containing morphology mip.
        - ``{xx.MORPHOLOGY_FOCUS_FILE!r}``: File containing morphology focus.

    .. seealso::

        - `10X Genomics Xenium file format  <https://cf.10xgenomics.com/supp/xenium/xenium_documentation.html>`_.

    Parameters
    ----------
    path
        Specifies the location of the dataset. This can either be a single path or a list of paths, where each path corresponds to a different experiment/roi.
    to_coordinate_system
        The coordinate system to which the images, segmentation masks and transcripts will be added for each item in path.
        If provided as a list, its length should be equal to the number of paths specified in `path`.
    aligned_images
        Whether to also parse, when available, additional H&E or IF aligned images. For more control over the aligned
        images being read, in particular, to specify the axes of the aligned images, please set this parameter to
        `False` and use the `xenium_aligned_image` function directly.
    cells_labels
        Whether to read cell labels (raster) provided by Xenium. The polygonal version of the cell labels are simplified
        for visualization purposes, and using the raster version is recommended for analysis.
    nucleus_labels
        Whether to read nucleus labels (raster) provided by Xenium. The polygonal version of the nucleus labels are simplified
        for visualization purposes, and using the raster version is recommended for analysis.
    cells_table
        Whether to read the cell annotations in the `AnnData` table.
        Will be added to the `f"table_{to_coordinate_system}"` slot in `sdata.tables`, or  f"table_{to_coordinate_system[i]}" if `to_coordinate_system` is a list.
        If `True`, labels layer annotating the table will also be added to `sdata`.
    filter_gene_names
        Gene names that need to be filtered out (via `str.contains`), mostly control genes that were added, and which you don't want to use.
        Filtering is case insensitive. Also see `sparrow.read_transcripts`.
    output
        The path where the resulting `SpatialData` object will be backed. If `None`, it will not be backed to a zarr store.

    Raises
    ------
    AssertionError
        Raised when the number of elements in `path` and `to_coordinate_system` are not the same.
    AssertionError
        If elements in `to_coordinate_system` are not unique.
    AssertionError
        If `cells_table` is `True`, but the labels layer annotating the table is not found.

    Returns
    -------
    A SpatialData object.
    """

    def _fix_name(item: str | Iterable[str]):
        return list(item) if isinstance(item, Iterable) and not isinstance(item, str) else [item]

    path = _fix_name(path)
    to_coordinate_system = _fix_name(to_coordinate_system)
    assert len(path) == len(
        to_coordinate_system
    ), "If parameters 'path' and/or 'to_coordinate_system' are specified as a list, their length should be equal."
    assert len(to_coordinate_system) == len(
        set(to_coordinate_system)
    ), "All elements specified via 'to_coordinate_system' should be unique."
    if cells_table:
        log.info(
            "Setting 'cells_labels' to True, in order to being able to annotate the table with corresponding labels layer."
        )
        cells_labels = True

    for _path, _to_coordinate_system in zip(path, to_coordinate_system):
        sdata = sdata_xenium(
            path=_path,
            cells_boundaries=False,
            nucleus_boundaries=False,
            cells_labels=cells_labels,
            nucleus_labels=nucleus_labels,
            morphology_focus=True,
            morphology_mip=False,
            cells_as_circles=False,
            transcripts=False,
            cells_table=cells_table,
            aligned_images=aligned_images,
        )

        layers = [*sdata.images] + [*sdata.labels]

        for _layer in layers:
            # rename coordinate system "global" to _to_coordinate_system
            transformation = {_to_coordinate_system: get_transformation(sdata[_layer], to_coordinate_system="global")}
            set_transformation(sdata[_layer], transformation=transformation, set_all=True)
            sdata[f"{_layer}_{_to_coordinate_system}"] = sdata[_layer]
            del sdata[_layer]

        if cells_table:
            adata = sdata["table"]
            assert f"cell_labels_{_to_coordinate_system}" in [
                *sdata.labels
            ], "labels layer annotating the table is not found in SpatialData object."
            # remove "cell_id" column in table, to avoid confusion with _INSTANCE_KEY.
            if "cell_id" in adata.obs.columns:
                adata.obs.drop(columns=["cell_id"], inplace=True)

            adata.obs.rename(columns={"region": _REGION_KEY, "cell_labels": _INSTANCE_KEY}, inplace=True)
            adata.obs[_REGION_KEY] = pd.Categorical(adata.obs[_REGION_KEY].astype(str) + f"_{_to_coordinate_system}")
            adata.uns.pop(TableModel.ATTRS_KEY)
            adata = TableModel.parse(
                adata,
                region_key=_REGION_KEY,
                region=adata.obs[_REGION_KEY].cat.categories.to_list(),
                instance_key=_INSTANCE_KEY,
            )

            del sdata["table"]

            sdata[f"table_{_to_coordinate_system}"] = adata

    # back the images to the zarr store, so we avoid having to persist the transcripts in memory in the read_transcripts step.
    if output is not None:
        sdata.write(output)
        sdata = read_zarr(output)

    # now read the transcripts
    for _path, _to_coordinate_system in zip(path, to_coordinate_system):
        table = dd.read_parquet(os.path.join(_path, XeniumKeys.TRANSCRIPTS_FILE))

        with open(os.path.join(_path, XeniumKeys.XENIUM_SPECS)) as f:
            specs = json.load(f)

        # Create a 3x3 identity matrix
        affine_matrix = np.eye(3)

        affine_matrix[0, 0] = 1 / specs["pixel_size"]  # Scaling in x
        affine_matrix[1, 1] = 1 / specs["pixel_size"]  # Scaling in y

        column_x_name = XeniumKeys.TRANSCRIPTS_X
        column_y_name = XeniumKeys.TRANSCRIPTS_Y
        column_gene_name = XeniumKeys.FEATURE_NAME

        column_x = table.columns.get_loc(column_x_name)
        column_y = table.columns.get_loc(column_y_name)
        column_gene = table.columns.get_loc(column_gene_name)

        sdata = read_transcripts(
            sdata,
            path_count_matrix=os.path.join(_path, XeniumKeys.TRANSCRIPTS_FILE),
            transform_matrix=affine_matrix,
            output_layer=f"transcripts_{_to_coordinate_system}",
            column_x=column_x,
            column_y=column_y,
            column_z=None,
            column_gene=column_gene,
            to_coordinate_system=_to_coordinate_system,
            filter_gene_names=filter_gene_names,
            overwrite=False,
        )

    return sdata
