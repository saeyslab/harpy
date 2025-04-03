from __future__ import annotations

from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pyarrow
from numpy.typing import NDArray
from spatialdata import SpatialData
from spatialdata.transformations import Identity

from harpy.points._points import add_points_layer
from harpy.utils._keys import _GENES_KEY
from harpy.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def read_resolve_transcripts(
    sdata: SpatialData,
    path_count_matrix: str | Path,
    output_layer: str = "transcripts",
    to_coordinate_system: str = "global",
    overwrite: bool = False,
) -> SpatialData:
    """
    Reads and adds Resolve transcript information to a SpatialData object.

    Parameters
    ----------
    sdata
        The SpatialData object to which the transcripts will be added.
    path_count_matrix
        Path to the file containing the transcripts information specific to Resolve.
        Expected to contain x, y coordinates and a gene name.
    output_layer: str, default='transcripts'.
        Name of the points layer of the SpatialData object to which the transcripts will be added.
    to_coordinate_system
        Coordinate system to which `output_layer` will be added.
    overwrite
        If True overwrites the `output_layer` (a points layer) if it already exists.

    Returns
    -------
    The updated SpatialData object containing the transcripts.
    """
    args = (sdata, path_count_matrix)
    kwargs = {
        "column_x": 0,
        "column_y": 1,
        "column_gene": 3,
        "delimiter": "\t",
        "header": None,
        "overwrite": overwrite,
        "output_layer": output_layer,
        "to_coordinate_system": to_coordinate_system,
    }

    sdata = read_transcripts(*args, **kwargs)
    return sdata


def read_merscope_transcripts(
    sdata: SpatialData,
    path_count_matrix: str | Path,
    transform_matrix: str | Path,
    output_layer: str = "transcripts",
    to_coordinate_system: str = "global",
    overwrite: bool = False,
) -> SpatialData:
    """
    Reads and adds merscope transcript information to a SpatialData object.

    Parameters
    ----------
    sdata
        The SpatialData object to which the transcripts will be added.
    path_count_matrix
        Path to the file containing the transcripts information specific to Vizgen.
        Expected to contain x, y coordinates and a gene name.
    output_layer: str, default='transcripts'.
        Name of the points layer of the SpatialData object to which the transcripts will be added.
    to_coordinate_system
        Coordinate system to which `output_layer` will be added.
    transform_matrix
        Path to the transformation matrix for the affine transformation.
    overwrite: bool, default=False
        If True overwrites the `output_layer` (a points layer) if it already exists.

    Returns
    -------
    The updated SpatialData object containing the transcripts.
    """
    args = (sdata, path_count_matrix, transform_matrix)
    kwargs = {
        "column_x": 2,
        "column_y": 3,
        "column_gene": 8,
        "delimiter": ",",
        "header": 0,
        "overwrite": overwrite,
        "output_layer": output_layer,
        "to_coordinate_system": to_coordinate_system,
    }

    sdata = read_transcripts(*args, **kwargs)
    return sdata


def read_stereoseq_transcripts(
    sdata: SpatialData,
    path_count_matrix: str | Path,
    output_layer: str = "transcripts",
    to_coordinate_system: str = "global",
    overwrite: bool = False,
) -> SpatialData:
    """
    Reads and adds Stereoseq transcript information to a SpatialData object.

    Parameters
    ----------
    sdata
        The SpatialData object to which the transcripts will be added.
    path_count_matrix
        Path to the file containing the transcripts information specific to Stereoseq.
        Expected to contain x, y coordinates, gene name, and a midcount column.
    output_layer: str, default='transcripts'.
        Name of the points layer of the SpatialData object to which the transcripts will be added.
    to_coordinate_system
        Coordinate system to which `output_layer` will be added.
    overwrite: bool, default=False
        If True overwrites the `output_layer` (a points layer) if it already exists.

    Returns
    -------
    The updated SpatialData object containing the transcripts.
    """
    args = (sdata, path_count_matrix)
    kwargs = {
        "column_x": 1,
        "column_y": 2,
        "column_gene": 0,
        "column_midcount": 3,
        "delimiter": ",",
        "header": 0,
        "overwrite": overwrite,
        "output_layer": output_layer,
        "to_coordinate_system": to_coordinate_system,
    }

    sdata = read_transcripts(*args, **kwargs)
    return sdata


def read_transcripts(
    sdata: SpatialData,
    path_count_matrix: str | Path,
    transform_matrix: str | Path | NDArray | None = None,
    output_layer: str = "transcripts",
    overwrite: bool = False,
    debug: bool = False,
    column_x: int = 0,
    column_y: int = 1,
    column_z: int | None = None,
    column_gene: int = 3,
    column_midcount: int | None = None,
    delimiter: str = ",",
    header: int | None = None,
    comment: str | None = None,
    crd: tuple[int, int, int, int] | None = None,
    to_coordinate_system: str = "global",
    filter_gene_names: str | list[str] | None = None,
    blocksize: str = "64MB",
) -> SpatialData:
    """
    Reads transcript information from a file with each row listing the x and y coordinates, along with the gene name.

    If a transform matrix is provided an affine transformation is applied to the coordinates of the transcripts.
    The transformation is applied to the dask dataframe before adding it to `sdata`.
    The SpatialData object is augmented with a points layer named `output_layer` that contains the transcripts.

    Parameters
    ----------
    sdata
        The SpatialData object to which the transcripts will be added.
    path_count_matrix
        Path to a `.parquet` file or `.csv` file containing the transcripts information. Each row should contain an `x` (`column_x`), `y` (`column_y`) coordinate and a gene name (`column_gene`).
        Optional a count column (see `column_midcount`) is provided.
    transform_matrix
        This `numpy` array should contain a 3x3 transformation matrix for the affine transformation.
        The matrix defines the linear transformation to be applied to the coordinates of the transcripts before adding it as a points layer to `sdata`.
        E.g.:
        | Sx  0  Tx |
        |  0  Sy  Ty |
        |  0   0   1 |
        If no transform matrix is specified, the identity matrix will be used.
        If `transform_matrix` is specified as a path to a file, it will be read via `numpy.loadtext`.
    output_layer
        Name of the points layer of the SpatialData object to which the transcripts will be added.
    overwrite
        If True overwrites the `output_layer` (a points layer) if it already exists.
    debug
        If True, a sample of the data is processed for debugging purposes.
    column_x
        Column index of the X coordinate in the count matrix.
    column_y
        Column index of the Y coordinate in the count matrix.
    column_z
        Column index of the Z coordinate in the count matrix.
    column_gene
        Column index of the gene information in the count matrix.
    column_midcount
        Specifies the column index that contains the count of how many times the gene is detected at that particular location.
        Ignored when set to None.
    delimiter
        Delimiter used to separate values in the `.csv` file. Ignored if `path_count_matrix` is a `.parquet` file.
    header
        Row number to use as the header in the `.csv` file. If `None`, no header is used. Ignored if `path_count_matrix` is a `.parquet` file.
    comment
        Character indicating that the remainder of line should not be parsed.
        If found at the beginning of a line, the line will be ignored altogether.
        This parameter must be a single character.
        Ignored if `path_count_matrix` is a `.parquet` file.
    crd
        The coordinates (in pixels) for the region of interest in the format (xmin, xmax, ymin, ymax).
        If None, all transcripts are considered.
    to_coordinate_system
        Coordinate system to which `output_layer` will be added.
    filter_gene_names
        Gene names that need to be filtered out (via `str.contains`), mostly control genes that were added, and which you don't want to use.
        Filtering is case insensitive.
    blocksize
        Block size of the partions of the dask dataframe stored as `points_layer` in `sdata`.

    Returns
    -------
    The updated SpatialData object containing the transcripts.
    """

    def _read_parquet_file(path_count_matrix):
        try:
            # Try reading the file as a Parquet file
            ddf = dd.read_parquet(path_count_matrix, blocksize=blocksize)
            return ddf
        except pyarrow.ArrowInvalid:
            return None

    # first try to read it as a parquet file.
    ddf = _read_parquet_file(path_count_matrix=path_count_matrix)

    if ddf is None:
        # if not parquet file, consider it to be csv file
        ddf = dd.read_csv(
            path_count_matrix,
            delimiter=delimiter,
            header=header,
            comment=comment,
            blocksize=blocksize,
        )

    def filter_names(ddf, column_gene, filter_name):
        # filter out control genes that you don't want ending up in the dataset

        ddf = ddf[~ddf.iloc[:, column_gene].str.contains(filter_name, case=False, na=False)]
        return ddf

    if filter_gene_names:
        if isinstance(filter_gene_names, list):
            for i in filter_gene_names:
                ddf = filter_names(ddf, column_gene, i)

        elif isinstance(filter_gene_names, str):
            ddf = filter_names(ddf, column_gene, filter_gene_names)
        else:
            log.info(
                "instance to filter on isn't a string nor a list. No genes are filtered out based on the gene name. "
            )

    # Read the transformation matrix
    if transform_matrix is None:
        log.info("No transform matrix given, will use identity matrix.")
        transform_matrix = np.identity(3)
    elif isinstance(transform_matrix, Path | str):
        transform_matrix = np.loadtxt(transform_matrix)

    log.info(f"Transform matrix used:\n {transform_matrix}")

    if debug:
        n = 100000
        fraction = n / len(ddf)
        ddf = ddf.sample(frac=fraction)

    # Function to repeat rows based on MIDCount value
    def repeat_rows(df):
        repeat_df = df.reindex(df.index.repeat(df.iloc[:, column_midcount])).reset_index(drop=True)
        return repeat_df

    # Apply the row repeat function if column_midcount is not None (e.g. for Stereoseq)
    if column_midcount is not None:
        ddf = ddf.map_partitions(repeat_rows, meta=ddf)

    def transform_coordinates(df):
        micron_coordinates = df.iloc[:, [column_x, column_y]].values
        micron_coordinates = np.column_stack((micron_coordinates, np.ones(len(micron_coordinates))))
        pixel_coordinates = np.dot(micron_coordinates, transform_matrix.T)[:, :2]
        result_df = df.iloc[:, [column_gene]].copy()
        result_df["pixel_x"] = pixel_coordinates[:, 0]
        result_df["pixel_y"] = pixel_coordinates[:, 1]
        return result_df

    # Apply the transformation to the Dask DataFrame
    transformed_ddf = ddf.map_partitions(transform_coordinates)

    # Rename the columns
    transformed_ddf.columns = [_GENES_KEY, "pixel_x", "pixel_y"]

    columns = ["pixel_x", "pixel_y", _GENES_KEY]
    coordinates = {"x": "pixel_x", "y": "pixel_y"}

    if column_z is not None:
        transformed_ddf["pixel_z"] = ddf.iloc[:, column_z]
        columns.append("pixel_z")
        coordinates["z"] = "pixel_z"

    # Reorder
    transformed_ddf = transformed_ddf[columns]
    # Save genes key as categorical
    transformed_ddf[_GENES_KEY] = transformed_ddf[_GENES_KEY].astype("str").astype("category")

    if crd is not None:
        transformed_ddf = transformed_ddf.query(f"{crd[0]} <= pixel_x < {crd[1]} and {crd[2]} <= pixel_y < {crd[3]}")

    sdata = add_points_layer(
        sdata,
        ddf=transformed_ddf,
        output_layer=output_layer,
        coordinates=coordinates,
        transformations={to_coordinate_system: Identity()},
        overwrite=overwrite,
    )

    return sdata
