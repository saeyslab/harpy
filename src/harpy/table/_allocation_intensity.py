from __future__ import annotations

import uuid
from collections.abc import Iterable

import anndata as ad
import numpy as np
import pandas as pd
from anndata import AnnData
from dask_image.ndmeasure import center_of_mass
from spatialdata import SpatialData

from sparrow.image._image import _get_spatial_element, _get_translation
from sparrow.table._table import add_table_layer
from sparrow.utils._aggregate import RasterAggregator
from sparrow.utils._keys import _CELL_INDEX, _INSTANCE_KEY, _REGION_KEY, _SPATIAL
from sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def allocate_intensity(
    sdata: SpatialData,
    img_layer: str | None = None,
    labels_layer: str | None = None,
    output_layer: str = "table_intensities",
    channels: int | str | Iterable[int] | Iterable[str] | None = None,
    to_coordinate_system: str = "global",
    chunks: str | int | tuple[int, ...] | None = 10000,
    append: bool = False,
    calculate_center_of_mass: bool = True,
    overwrite: bool = True,
) -> SpatialData:
    """
    Allocates intensity values from a specified image layer to corresponding cells in a SpatialData object and returns an updated SpatialData object augmented with a table layer (`sdata.tables[output_layer]`) AnnData object with intensity values for each cell and each (specified) channel.

    It requires that the image layer and the labels layer have the same shape and alignment.

    Parameters
    ----------
    sdata
        The SpatialData object containing spatial information about cells.
    img_layer
        The name of the layer in `sdata` that contains the image data from which to extract intensity information.
        Both the `img_layer` and `labels_layer` should have the same shape and alignment. If not provided,
        will use last img_layer.
    labels_layer
        The name of the layer in `sdata` containing the labels (segmentation) used to define the boundaries of cells.
        These labels correspond with regions in the `img_layer`. If not provided, will use last labels_layer.
    output_layer: str, optional
        The table layer in `sdata` in which to save the AnnData object with the intensity values per cell.
    channels
        Specifies the channels to be considered when extracting intensity information from the `img_layer`.
        This parameter can take a single integer or string or an iterable of integers or strings representing specific channels.
        If set to None (the default), intensity data will be aggregated from all available channels within the image layer.
    to_coordinate_system
        The coordinate system that holds `img_layer` and `labels_layer`.
    chunks
        The chunk size for processing the image data. If provided as a tuple, desired chunksize for (z), y, x should be provided.
    append
        If set to True, and the `labels_layer` does not yet exist as a `_REGION_KEY` in `sdata.tables[output_layer].obs`,
        the intensity values extracted during the current function call will be appended (along axis=0) to any existing intensity data
        within the SpatialData object's table attribute. If False, and overwrite is set to True any existing data in `sdata.tables[output_layer]` will be overwritten by the newly extracted intensity values.
    calculate_center_of_mass
        If `True`, the center of mass of the labels in `labels_layer` will be calculated and added to `sdata.tables[ output_layer ].obsm[_SPATIAL]`.
        To calculate center of mass, we use `dask_image.ndmeasure.center_of_mass`.
    overwrite
        If `True`, overwrites the `output_layer` if it already exists in `sdata`.

    Returns
    -------
    An updated version of the input SpatialData object augmented with a table layer (`sdata.tables[output_layer]`) AnnData object.

    Notes
    -----
    - The function currently supports scenarios where the `img_layer` and `labels_layer` are aligned and have the same
      shape. Misalignments or differences in shape must be handled prior to invoking this function.
    - Intensity calculation is performed per channel for each cell. The function aggregates this information and
      attaches it as a table (AnnData object) within the SpatialData object.
    - Due to the memory-intensive nature of the operation, especially for large datasets, the function implements
      chunk-based processing, aided by Dask. The `chunks` parameter allows for customization of the chunk sizes used
      during processing.

    Example
    -------
    >>> sdata = sp.im.align_labels_layers(
    ...     sdata,
    ...     labels_layer_1="masks_nuclear",
    ...     labels_layer_2="masks_whole",
    ...     output_labels_layer="masks_nuclear_aligned",
    ...     output_shapes_layer=None,
    ...     overwrite=True,
    ...     chunks=256,
    ...     depth=100,
    ... )
    >>>
    >>> sdata = sp.tb.allocate_intensity(
    ...     sdata, img_layer="raw_image", labels_layer="masks_whole", output_layer="table_intensities", chunks=100
    ... )
    >>>
    >>> sdata = sp.tb.allocate_intensity(
    ...     sdata, img_layer="raw_image", labels_layer="masks_nuclear_aligned", output_later="table_intensities", chunks=100, append=True
    ... )
    >>> # alternatively, save to different tables
    >>> sdata = sp.tb.allocate_intensity(
    ...     sdata, img_layer="raw_image", labels_layer="masks_whole", output_layer="table_intensities_masks_whole", chunks=100
    ... )
    >>>
    >>> sdata = sp.tb.allocate_intensity(
    ...     sdata, img_layer="raw_image", labels_layer="masks_nuclear_aligned", output_later="table_intensities_masks_nuclear_aligned", chunks=100, append=True
    ... )
    """
    if img_layer is None:
        img_layer = [*sdata.images][-1]
        log.warning(
            f"No image layer specified. "
            f"Extracting intensities from the last image layer '{img_layer}' of the provided SpatialData object."
        )

    if labels_layer is None:
        labels_layer = [*sdata.labels][-1]
        log.warning(
            f"No labels layer specified. "
            f"Using mask from labels layer '{labels_layer}' of the provided SpatialData object."
        )

    if channels is not None:
        channels = list(channels) if isinstance(channels, Iterable) and not isinstance(channels, str) else [channels]

    # currently this function will only work if img_layer and labels_layer have the same shape.
    # And are in same position, i.e. if one is translated, other should be translated with same offset
    se_image = _get_spatial_element(sdata, layer=img_layer)
    se_labels = _get_spatial_element(sdata, layer=labels_layer)

    assert (
        se_image.data.shape[1:] == se_labels.data.shape
    ), "Only arrays with same spatial shape are currently supported, "
    f"but image layer with name {img_layer} has shape {se_image.data.shape}, "
    f"while labels layer with name {labels_layer} has shape {se_labels.data.shape}  "

    t1x, t1y = _get_translation(se_image, to_coordinate_system=to_coordinate_system)
    t2x, t2y = _get_translation(se_labels, to_coordinate_system=to_coordinate_system)

    assert (t1x, t1y) == (t2x, t2y), f"image layer with name {img_layer} should "
    f"be registered to labels layer with name {labels_layer} in coordinate system {to_coordinate_system}."

    if channels is None:
        channels = se_image.c.data

    _array_mask = se_labels.data
    _array_img = se_image.data

    to_squeeze = False
    if se_image.ndim == 3:
        to_squeeze = True
        _array_mask = _array_mask[None, ...]
        _array_img = _array_img[:, None, ...]

    chunks_masks = None
    if chunks is not None:
        if not isinstance(chunks, (int, str)):
            if to_squeeze:
                assert len(chunks) == _array_img.ndim - 2
                chunks = (_array_img.chunksize[0], 1, chunks[0], chunks[1])
                chunks_masks = (1, chunks[2], chunks[3])
            else:
                assert len(chunks) == _array_img.ndim - 1
                chunks = (_array_img.chunksize[0], chunks[0], chunks[1], chunks[2])
                chunks_masks = (chunks[1], chunks[2], chunks[3])
        else:
            chunks_masks = chunks

    _array_img = _array_img.rechunk(chunks) if chunks is not None else _array_img
    _array_mask_rechunked = _array_mask.rechunk(chunks_masks) if chunks_masks is not None else _array_mask

    assert all(
        element in se_image.c.data for element in channels
    ), f"Some channels specified via 'channels' could not be found in image layer '{img_layer}'. Please choose 'channels' from '{list( se_image.c.data )}'."
    channel_indices = [list(se_image.c.data).index(channel) for channel in channels]
    _array_img = _array_img[channel_indices]
    aggregator = RasterAggregator(image_dask_array=_array_img, mask_dask_array=_array_mask_rechunked)
    df_sum = aggregator.aggregate_sum()

    _cells_id = df_sum[_INSTANCE_KEY].values
    channel_intensities = df_sum.drop([_INSTANCE_KEY], axis=1).values

    channels = list(map(str, channels))
    var = pd.DataFrame(index=channels)
    var.index = var.index.map(str)
    var.index.name = "channels"

    # _cells_id = unique(se_labels.data).compute()  # two times computation of unique labels, this is not necessary.
    cells = pd.DataFrame(index=_cells_id)
    _uuid_value = str(uuid.uuid4())[:8]
    cells.index = cells.index.map(lambda x: f"{x}_{labels_layer}_{_uuid_value}")
    cells.index.name = _CELL_INDEX
    adata = AnnData(X=channel_intensities, obs=cells, var=var)

    adata.obs[_INSTANCE_KEY] = _cells_id.astype(int)
    adata.obs[_REGION_KEY] = pd.Categorical([labels_layer] * len(adata.obs))
    # remove background intensity
    adata = adata[adata.obs[_INSTANCE_KEY] != 0]
    _cells_id = _cells_id[_cells_id != 0]

    if calculate_center_of_mass:
        # add center of cells here (via the masks).
        _array_mask = _array_mask.squeeze(0) if to_squeeze else _array_mask
        coordinates = center_of_mass(
            image=_array_mask,  # do not use rechunked array mask here, leads to significant increase in required ram.
            label_image=_array_mask,
            index=_cells_id,
        )

        coordinates = coordinates.compute()
        coordinates += np.array([t1y, t1x]) if to_squeeze else np.array([0, t1y, t1x])

        adata.obsm[_SPATIAL] = coordinates

    if append:
        region = []
        if output_layer in [*sdata.tables]:
            if labels_layer in sdata.tables[output_layer].obs[_REGION_KEY].cat.categories:
                raise ValueError(
                    f"'{labels_layer}' already exists as a region in the 'sdata.tables[{output_layer}]' object. Please choose a different labels layer, choose a different 'output_layer' or set append to False and overwrite to True to overwrite the existing table."
                )
            adata = ad.concat([sdata.tables[output_layer], adata], axis=0)
            # get the regions already in sdata, and append the new one
            region = sdata.tables[output_layer].obs[_REGION_KEY].cat.categories.to_list()
        region.append(labels_layer)

    else:
        region = [labels_layer]

    sdata = add_table_layer(
        sdata,
        adata=adata,
        output_layer=output_layer,
        region=region,
        overwrite=overwrite,
    )

    return sdata
