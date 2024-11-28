from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import shapely
from geopandas import GeoDataFrame
from numpy.typing import NDArray
from pandas import DataFrame as PandasDataFrame
from PIL import Image
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, shape
from shapely.validation import make_valid
from skimage.segmentation import relabel_sequential

from harpy.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def _baysor(
    img: NDArray,
    df: PandasDataFrame,
    name_x: str,
    name_y: str,
    name_gene: str,
    config_path: str | Path,  # path to config.toml file of baysor
    output_dir: str | Path,
    threads: int,
    diameter: int = 40,  # this is scale in baysor, should be approx equal to the expected cell ,
    min_size: int | None = None,
    use_prior_segmentation: bool = True,
    prior_confidence: int = 0.2,  # expected quality of the prior (i.e. masks). 0.0 will make algorithm ignore the prior, while 1.0 restricts the algorithm from contradicting the prior.
) -> NDArray:
    # img is (z,y,x,c), and returned masks are also (z,y,x,c)
    log.info(f"Prior segmentation: {use_prior_segmentation}")
    log.info(f"Prior confidence: {prior_confidence}")
    assert img.ndim == 4, "Please provide img with (z,y,x,c) dimension. z and c dimension should be 1."
    assert img.shape[0] == 1, "Currently only 2D segmentation is supported. I.e. z-dimension should be 1"
    assert img.shape[-1] == 1, "Channel dimension should be 1."
    assert name_x in df.columns, f"DataFrame should contain 'x' coordinate '{name_x}'."
    assert name_y in df.columns, f"DataFrame should contain 'y' coordinate '{name_y}'."
    assert name_gene in df.columns, f"DataFrame should contain 'gene' column. '{name_gene}'."

    if df.shape[0] < 50:
        log.warning("Chunk contains less than 50 transcripts, returning array containing zeros.")
        return np.zeros((img.shape), dtype="uint32")

    # squeeze the trivial c-channel
    img = img.squeeze(-1)
    # currently only support 2D segmentation with baysor.
    img = img.squeeze(0)

    temp_dir = tempfile.mkdtemp(dir=output_dir)
    # Define paths for the stdout and stderr files
    stdout_path = os.path.join(temp_dir, "stdout.log")
    stderr_path = os.path.join(temp_dir, "stderr.log")

    df.to_csv(os.path.join(temp_dir, "transcripts.csv"))
    # Save the image as a TIFF file
    if use_prior_segmentation:
        image = Image.fromarray(img)
        masks_path = os.path.join(temp_dir, "masks.tiff")
        image.save(masks_path, format="TIFF")
    else:
        masks_path = ""

    output_html = os.path.join(temp_dir, "output.html")

    # command to run baysor
    command = (
        f"JULIA_NUM_THREADS={threads} baysor run "
        f"-s {diameter} "
        f"-x {name_x} "
        f"-y {name_y} "
        f"-g {name_gene} "
        f"-c {config_path} "
        f"-o {output_html} "
        f"--prior-segmentation-confidence={prior_confidence} "
        f"{os.path.join( temp_dir, 'transcripts.csv' ) } "
        "--save-polygons=geojson "
        f"{masks_path}"
    )

    with open(stdout_path, "w") as stdout_file, open(stderr_path, "w") as stderr_file:
        try:
            # Run the command and capture output
            subprocess.run(
                command,
                shell=True,
                check=True,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
            )

        except subprocess.CalledProcessError as e:
            log.error(f"Command failed with error: {e}.")

            log.info("Retrying...")
            try:
                # Second attempt to run the command
                subprocess.run(
                    command,
                    shell=True,
                    check=True,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                log.error(f"Second attempt failed with error: {e}.")
                log.error(f"Temporary directory kept (with transcripts and masks) at: {temp_dir}")
                # Re-raise the exception to be handled by the calling code or terminate the program with the traceback.
                raise e

    polygons = _read_baysor(path_polygons=os.path.join(temp_dir, "output_polygons.json"))

    if min_size is None:
        min_size = _calculate_area(diameter=diameter * 0.60)
        log.warning(
            f"Min size was set to 'None', setting min size to '{min_size}' (size of circle with diameter equal to 0.6*{diameter})."
        )

    polygons = polygons[polygons.geometry.area > min_size]

    # clean up temp_dir
    shutil.rmtree(temp_dir)

    # convert polygons to masks
    masks = rasterio.features.rasterize(
        zip(
            polygons.geometry,
            polygons.index.values.astype(float),
        ),
        out_shape=[img.shape[0], img.shape[1]],
        dtype="uint32",
        fill=0,
    )

    # add z and c dimension to masks
    masks = masks[None, ..., None]

    return masks


def _dummy(
    img: NDArray,
    df: PandasDataFrame,
    name_x: str,
    name_y: str,
    name_gene: str,
    c_dim: int = 1,
) -> NDArray:
    img, _, _ = relabel_sequential(
        img,
    )
    img = [img] * c_dim
    # dummy points segmentation, just return c_dims times the labels layer
    # (used for benchmarking, and unit tests)
    return np.concatenate(img, axis=-1)


def _read_baysor(path_polygons: str | Path, min_vertices: int = 4) -> GeoDataFrame:
    with open(path_polygons) as f:
        polygons_dict = json.load(f)
        polygons_dict = {c["cell"]: c for c in polygons_dict["geometries"]}

    polygons_dict = {num: data for num, data in polygons_dict.items() if len(data["coordinates"][0]) >= min_vertices}

    polygons = [shape(data) for _, data in polygons_dict.items()]

    gdf = gpd.GeoDataFrame(geometry=polygons)

    gdf.geometry = gdf.geometry.map(lambda cell: _ensure_polygon(make_valid(cell)))
    gdf = gdf[~gdf.geometry.isna()]

    return gdf


# taken from sopa https://github.com/gustaveroussy/sopa
def _ensure_polygon(cell: Polygon | MultiPolygon | GeometryCollection) -> Polygon:
    """Ensures that the provided cell becomes a Polygon

    Args:
        cell: A shapely Polygon or MultiPolygon or GeometryCollection

    Returns
    -------
        The shape as a Polygon
    """
    cell = shapely.make_valid(cell)

    if isinstance(cell, Polygon):
        if cell.interiors:
            cell = Polygon(list(cell.exterior.coords))
        return cell

    if isinstance(cell, MultiPolygon):
        return max(cell.geoms, key=lambda polygon: polygon.area)

    if isinstance(cell, GeometryCollection):
        geoms = [geom for geom in cell.geoms if isinstance(geom, Polygon)]

        if not geoms:
            log.info(f"Removing cell of type {type(cell)} as it contains no Polygon geometry")
            return None

        return max(geoms, key=lambda polygon: polygon.area)

    log.info(f"Removing cell of unknown type {type(cell)}")
    return None


def _calculate_area(diameter: int):
    radius = diameter / 2
    area = math.pi * (radius**2)
    return area
