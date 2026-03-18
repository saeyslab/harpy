from __future__ import annotations

import dask.array as da
import numpy as np
from dask import delayed
from loguru import logger as log
from scipy.ndimage import gaussian_filter
from spatialdata import SpatialData
from spatialdata.models.models import ScaleFactors_t
from spatialdata.transformations import Translation

from harpy.image._image import _get_boundary, add_image_layer, get_dataarray
from harpy.utils._transformations import _identity_check_transformations_points


def _build_density_chunk(
    x_values: np.ndarray,
    y_values: np.ndarray,
    values: np.ndarray,
    x_offset: int,
    y_offset: int,
    shape: tuple[int, int],
) -> np.ndarray:
    """Build a single dense chunk from local transcript coordinates."""
    block = np.zeros(shape, dtype=np.int32)
    block[x_values - x_offset, y_values - y_offset] = values
    return block


def transcript_density(
    sdata: SpatialData,
    img_layer: str = "raw_image",
    points_layer: str = "transcripts",
    n_sample: int | None = 15000000,
    name_x: str = "x",
    name_y: str = "y",
    name_z: str | None = None,
    z_index: int | None = None,
    chunks: int = 1024,
    crd: tuple[int, int, int, int] | None = None,
    to_coordinate_system: str = "global",
    scale_factors: ScaleFactors_t | None = None,
    output_layer: str = "transcript_density",
    overwrite: bool = False,
) -> SpatialData:
    """
    Calculate the transcript density using a Gaussian filter and add it to the provided :class:`spatialdata.SpatialData` object.

    This function computes the density of transcripts in the
    :class:`spatialdata.SpatialData` object, scales and smooths it, and then adds the
    resulting density image to the same :class:`spatialdata.SpatialData` object.

    Parameters
    ----------
    sdata
        :class:`spatialdata.SpatialData` object containing spatial information.
    img_layer
        The layer of the SpatialData object used for determining image boundary.
        `img_layer` and `points_layer` should be registered in coordinate system `to_coordinate_system`.
    points_layer
        The layer name that contains the transcript data points, by default "transcripts".
    n_sample
        The number of transcripts to sample for calculation of transcript density.
    name_x
        Column name for x-coordinates of the transcripts in the points layer, by default "x".
    name_y
        Column name for y-coordinates of the transcripts in the points layer, by default "y".
    name_z
        Column name for z-coordinates of the transcripts in the points layer, by default None.
    z_index
        The z index in the points layer for which to calculate transcript density. If set to None for a 3D points layer
        (and `name_z` is not equal to None), an y-x transcript density projection will be calculated.
    chunks
        Chunksize for calculation of density using gaussian filter.
    crd
        The coordinates for a region of interest in the format (xmin, xmax, ymin, ymax).
        If provided, the density is computed only for this region, by default None.
    to_coordinate_system
        The coordinate system that holds `img_layer` and `points_layer`.
        This coordinate system should be the intrinsic coordinate system in pixels.
    scale_factors
        Scale factors to apply for multiscale.
    output_layer
        The name of the output image layer in the :class:`spatialdata.SpatialData` object
        where the transcript density will be added, by default "transcript_density".
    overwrite
        If True overwrites the element if it already exists.

    Returns
    -------
    Updated :class:`spatialdata.SpatialData` object with the added transcript density layer
    as an image layer.

    Examples
    --------
    >>> sdata = SpatialData(...)
    >>> sdata = transcript_density(sdata, points_layer="transcripts", crd=(2000, 4000, 2000, 4000))

    """
    if z_index is not None and name_z is None:
        raise ValueError(
            "Please specify column name for the z-coordinates of the transcripts in the points layer "
            "when specifying z_index."
        )

    ddf = sdata.points[points_layer]

    _identity_check_transformations_points(ddf, to_coordinate_system=to_coordinate_system)

    ddf[name_x] = np.floor(ddf[name_x]).astype(int)
    ddf[name_y] = np.floor(ddf[name_y]).astype(int)
    if name_z is not None:
        ddf[name_z] = np.floor(ddf[name_z]).astype(int)

    # get image boundary from last image layer if img_layer is None
    if img_layer is None:
        img_layer = [*sdata.images][-1]

    se = get_dataarray(sdata, layer=img_layer)

    img_boundary = _get_boundary(se, to_coordinate_system=to_coordinate_system)

    # if crd is None, get boundary from image at img_layer if given,
    if crd is None:
        crd = img_boundary
    else:
        # fix crd so it falls in boundaries of img_layer, otherwise possible issues with registration transcripts, and size of the generated image.
        _crd = [
            max(img_boundary[0], crd[0]),
            min(img_boundary[1], crd[1]),
            max(img_boundary[2], crd[2]),
            min(img_boundary[3], crd[3]),
        ]
        if _crd != crd:
            log.warning(
                f"Provided crd didn't fully fit within the image layer '{img_layer}' with image boundary '{img_boundary}'. "
                f"The crd was updated from '{crd}' to '{_crd}'."
            )
        crd = _crd
    ddf = ddf.query(f"{crd[0]} <= {name_x} < {crd[1]} and {crd[2]} <= {name_y} < {crd[3]}")

    if z_index is not None:
        ddf = ddf.query(f"{name_z} == {z_index}")

    # subsampling:
    if n_sample is not None:
        size = len(ddf)
        if size > n_sample:
            log.info(f"The number of transcripts ( {size} ) is larger than n_sample, sampling {n_sample} transcripts.")
            fraction = n_sample / size
            ddf = ddf.sample(frac=fraction)
            log.info("sampling finished")

    counts_location_transcript = (
        ddf.groupby([name_x, name_y]).size().reset_index().rename(columns={0: "__count__"}).compute()
    )

    # crd is set to img boundary if None
    counts_location_transcript[name_x] = counts_location_transcript[name_x] - crd[0]
    counts_location_transcript[name_y] = counts_location_transcript[name_y] - crd[2]

    chunks = (chunks, chunks)
    width = crd[1] - crd[0]
    height = crd[3] - crd[2]

    if counts_location_transcript.empty:
        image = da.zeros((width, height), chunks=chunks, dtype=np.int32)
    else:
        x_values = counts_location_transcript[name_x].to_numpy(dtype=np.int64, copy=False)
        y_values = counts_location_transcript[name_y].to_numpy(dtype=np.int64, copy=False)
        values = counts_location_transcript["__count__"].to_numpy(dtype=np.int32, copy=False)

        chunk_x, chunk_y = chunks
        nblocks_x = (width + chunk_x - 1) // chunk_x
        nblocks_y = (height + chunk_y - 1) // chunk_y

        block_x = x_values // chunk_x
        block_y = y_values // chunk_y
        block_ids = block_x * nblocks_y + block_y

        order = np.argsort(block_ids, kind="mergesort")
        x_values = x_values[order]
        y_values = y_values[order]
        values = values[order]
        block_ids = block_ids[order]

        unique_ids, start_idx = np.unique(block_ids, return_index=True)
        stop_idx = np.append(start_idx[1:], len(block_ids))

        non_empty_blocks: dict[int, da.Array] = {}
        for block_id, start, stop in zip(unique_ids.tolist(), start_idx.tolist(), stop_idx.tolist(), strict=False):
            bx = block_id // nblocks_y
            by = block_id % nblocks_y
            x_offset = bx * chunk_x
            y_offset = by * chunk_y
            shape = (min(chunk_x, width - x_offset), min(chunk_y, height - y_offset))

            non_empty_blocks[block_id] = da.from_delayed(
                delayed(_build_density_chunk)(
                    np.ascontiguousarray(x_values[start:stop]),
                    np.ascontiguousarray(y_values[start:stop]),
                    np.ascontiguousarray(values[start:stop]),
                    x_offset,
                    y_offset,
                    shape,
                ),
                shape=shape,
                dtype=np.int32,
            )

        image = da.block(
            [
                [
                    non_empty_blocks.get(
                        bx * nblocks_y + by,
                        da.zeros(
                            (min(chunk_x, width - bx * chunk_x), min(chunk_y, height - by * chunk_y)),
                            chunks=(
                                min(chunk_x, width - bx * chunk_x),
                                min(chunk_y, height - by * chunk_y),
                            ),
                            dtype=np.int32,
                        ),
                    )
                    for by in range(nblocks_y)
                ]
                for bx in range(nblocks_x)
            ]
        )  # we assume never more than np.iinfo(np.int32).max transcripts per pixel

    image = image / da.maximum(da.max(image), 1)
    image = image.astype(np.float32)  # some small precision loss by casting

    sigma = 7

    def chunked_gaussian_filter(chunk):
        return gaussian_filter(chunk, sigma=sigma)

    # take overlap to be 3 times sigma
    overlap = sigma * 3

    blurred_transcripts = image.map_overlap(
        chunked_gaussian_filter, depth=overlap, boundary="reflect", dtype=np.float32
    )

    blurred_transcripts = blurred_transcripts.T
    # rechunk, otherwise possible issues when saving to zarr
    blurred_transcripts = blurred_transcripts.rechunk(blurred_transcripts.chunksize)

    translation = Translation([crd[0], crd[2]], axes=("x", "y"))

    arr = blurred_transcripts[None,]

    sdata = add_image_layer(
        sdata,
        arr=arr,
        output_layer=output_layer,
        chunks=arr.chunksize,
        transformations={to_coordinate_system: translation},
        scale_factors=scale_factors,
        overwrite=overwrite,
    )

    return sdata
