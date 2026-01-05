from __future__ import annotations

import uuid
from collections.abc import Iterable
from functools import reduce
from typing import Literal

import anndata as ad
import dask.array as da
import numpy as np
import pandas as pd
from anndata import AnnData
from loguru import logger as log
from spatialdata import SpatialData
from spatialdata.models import TableModel

from harpy.image._image import _get_translation, get_dataarray
from harpy.table._table import add_table_layer
from harpy.table._utils import _sanity_check_append_region
from harpy.utils._aggregate import RasterAggregator
from harpy.utils._keys import _CELL_INDEX, _CELLSIZE_KEY, _INSTANCE_KEY, _REGION_KEY, _SPATIAL


def allocate_intensity(
    sdata: SpatialData,
    img_layer: str | None = None,
    labels_layer: str | None = None,
    output_layer: str = "table_intensities",
    channels: int | str | Iterable[int] | Iterable[str] | None = None,
    mode: Literal["sum", "mean"] = "mean",
    obs_stats: list[str] | None = None,
    to_coordinate_system: str = "global",
    chunks: str | int | tuple[int, ...] | None = None,
    append: bool = False,
    calculate_center_of_mass: bool = True,
    region_key: str = _REGION_KEY,
    instance_key: str = _INSTANCE_KEY,
    spatial_key: str = _SPATIAL,
    instance_size_key: str = _CELLSIZE_KEY,
    cell_index_name: str = _CELL_INDEX,
    overwrite: bool = True,
) -> SpatialData:
    """
    Allocates intensity values from a specified image layer to corresponding cells in a SpatialData object and returns an updated SpatialData object augmented with a table layer (`sdata.tables[output_layer]`) :class:`~anndata.AnnData` object with intensity values for each cell and each (specified) channel.

    It requires that the image layer and the labels layer have the same shape and alignment.

    Internally this function uses :func:`harpy.utils.RasterAggregator`.

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
        The table layer in `sdata` in which to save the :class:`~anndata.AnnData` object with the intensity values per cell.
    channels
        Specifies the channels to be considered when extracting intensity information from the `img_layer`.
        This parameter can take a single integer or string or an iterable of integers or strings representing specific channels.
        If set to None (the default), intensity data will be aggregated from all available channels within the image layer.
    mode
        When mode is set to `"sum"`, the total intensity for each label will be added to `.X` of the resulting `output_layer`; if set to `"mean"`, it calculates the average intensity per label.
    obs_stats
        Statistics to add to `.obs` of `output_layer`.
        Supported values: `["sum", "mean", "count", "var", "kurtosis", "skew", "max", "min"]`.

        - If `obs_stats` contains `"mode"`, it will **not** be added to `.obs`.
        - For each `stat` in `["sum", "mean", "var", "kurtosis", "skew", "max", "min"]`, the result is stored as: `{channel_name}_{stat}`.
        - `"count"` is stored in `.obs` using the name given by `instance_size_key`.

    to_coordinate_system
        The coordinate system that holds `img_layer` and `labels_layer`.
        This should be the intrinsic coordinate system in pixels.
    chunks
        The chunk size for processing the image data. If provided as a tuple, desired chunksize for (z), y, x should be provided.
    append
        If set to True, and the `labels_layer` does not yet exist as a `region_key` in `sdata.tables[output_layer].obs`,
        the intensity values extracted during the current function call will be appended (along axis=0) to any existing intensity data
        within the SpatialData object's table attribute. If False, and overwrite is set to True any existing data in `sdata.tables[output_layer]` will be overwritten by the newly extracted intensity values.
        Note that we join the :class:`~anndata.AnnData` objects using :func:`~anndata.concat` with `join="inner"`.
    calculate_center_of_mass
        If `True`, the center of mass of the labels in `labels_layer` will be calculated and added to `sdata.tables[ output_layer ].obsm[spatial_key]`.
        The center of mass is computed using `scipy.ndimage.center_of_mass`. Enabling `calculate_center_of_mass` will cause the `labels_layer` to be loaded into memory.
    instance_key
        Instance key. The name of the column in :class:`~anndata.AnnData` table `.obs` that will hold the instance ids.
    region_key
        Region key. The name of the column in  :class:`~anndata.AnnData` table `.obs` that will hold the name of the element(s) that are annotated by the resulting table.
    spatial_key
        The key in the :class:`~anndata.AnnData` table `.obsm` that will hold the `x` and `y` center of the instances.
        This center is calculated by calculating the center of mass of each cell in `labels_layer`.
    instance_size_key
        The key in the :class:`~anndata.AnnData` table `.obs` that will hold the size of the instances. Ignored if "count" not in `obs_stats`.
    cell_index_name
        The name of the index of the resulting :class:`~anndata.AnnData` table.
    overwrite
        If `True`, overwrites the `output_layer` if it already exists in `sdata`.

    Returns
    -------
    An updated version of the input SpatialData object augmented with a table layer (`sdata.tables[output_layer]`) :class:`~anndata.AnnData` object.

    Notes
    -----
    - The function currently supports scenarios where the `img_layer` and `labels_layer` are aligned and have the same
      shape. Misalignments or differences in shape must be handled prior to invoking this function.
    - Intensity calculation is performed per channel for each cell. The function aggregates this information and
      attaches it as a table (:class:`~anndata.AnnData` object) within the SpatialData object.
    - Due to the memory-intensive nature of the operation, especially for large datasets, the function implements
      chunk-based processing, aided by Dask.
      If sdata is backed by a Zarr store, we recommend using `chunks=None` and ensuring that the on-disk Dask array chunks are optimized for both storage efficiency and computational performance.


    Examples
    --------
    Allocate intensity statistics into an AnnData table:

    .. code-block:: python

        import harpy as hp

        sdata = hp.datasets.pixie_example()

        # Compute intensity statistics in coordinate system "fov0"
        sdata = hp.tb.allocate_intensity(
            sdata,
            img_layer="raw_image_fov0",
            labels_layer="label_whole_fov0",
            to_coordinate_system="fov0",
            output_layer="my_table",
            mode="sum",
            obs_stats="count",  # cell size
            overwrite=True,
        )

        # Append intensity statistics in coordinate system "fov1"
        sdata = hp.tb.allocate_intensity(
            sdata,
            img_layer="raw_image_fov1",
            labels_layer="label_whole_fov1",
            to_coordinate_system="fov1",
            output_layer="my_table",
            mode="sum",
            obs_stats="count",  # cell size
            append=True,
            overwrite=True,
        )

    See Also
    --------
    harpy.utils.RasterAggregator : out of core calculation of statistics from raster data.
    """
    assert mode in ["sum", "mean"], "'mode' must be either 'sum' or 'mean'."
    if obs_stats is not None:
        if isinstance(obs_stats, str):
            obs_stats = [obs_stats]
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
    se_image = get_dataarray(sdata, layer=img_layer)
    se_labels = get_dataarray(sdata, layer=labels_layer)

    if se_image.data.shape[1:] != se_labels.data.shape:
        raise ValueError(
            "Only arrays with same spatial shape are currently supported, "
            f"but image layer with name {img_layer} has shape {se_image.data.shape}, "
            f"while labels layer with name {labels_layer} has shape {se_labels.data.shape}."
        )

    t1x, t1y = _get_translation(se_image, to_coordinate_system=to_coordinate_system)
    t2x, t2y = _get_translation(se_labels, to_coordinate_system=to_coordinate_system)

    if (t1x, t1y) != (t2x, t2y):
        raise ValueError(
            f"image layer with name {img_layer} should "
            f"be registered to labels layer with name {labels_layer} in coordinate system {to_coordinate_system}."
        )

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
        if not isinstance(chunks, int | str):
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

    assert all(element in se_image.c.data for element in channels), (
        f"Some channels specified via 'channels' could not be found in image layer '{img_layer}'. Please choose 'channels' from '{list(se_image.c.data)}'."
    )
    channel_indices = [list(se_image.c.data).index(channel) for channel in channels]
    _array_img = _array_img[channel_indices]
    aggregator = RasterAggregator(
        image_dask_array=_array_img,
        mask_dask_array=_array_mask_rechunked,
        instance_key=instance_key,
        instance_size_key=instance_size_key,
    )
    index = da.unique(_array_mask_rechunked).compute()

    if obs_stats is not None:
        stats_funcs = [mode] + [stat for stat in obs_stats if stat != mode]
    else:
        stats_funcs = [mode]

    def rename_columns(_df: pd.DataFrame, prefix: str):
        # helper function to rename column of a dataframe to prefix_channel_name
        new_columns = []
        for _name in _df.columns:
            if _name != instance_key:
                new_columns.append(f"{prefix}_{channels[_name]}")
            else:
                new_columns.append(_name)
        _df.columns = new_columns

    # Calculate max and min if necessary.
    df_max = df_min = None
    if "max" in stats_funcs:
        stats_funcs.remove("max")
        df_max = aggregator.aggregate_max(index=index)
        rename_columns(df_max, prefix="max")

    if "min" in stats_funcs:
        stats_funcs.remove("min")
        df_min = aggregator.aggregate_min(index=index)
        rename_columns(df_min, prefix="min")

    dfs = aggregator.aggregate_stats(stats_funcs=stats_funcs, index=index)

    df_X = dfs[0]  # the first one is the mode

    if obs_stats is not None and mode not in obs_stats:
        df_obs = dfs[1:]
        for _df, _prefix in zip(df_obs, stats_funcs[1:], strict=True):
            if _prefix != "count":
                rename_columns(_df, prefix=_prefix)
            else:
                _df.rename(columns={0: instance_size_key}, inplace=True)
        df_obs = df_obs + [x for x in [df_max, df_min] if x is not None]
        # merge
        df_obs = reduce(lambda left, right: pd.merge(left, right, how="inner", on=instance_key), df_obs)

    _cells_id = df_X[instance_key].values
    channel_intensities = df_X.drop([instance_key], axis=1).values

    channels = list(map(str, channels))
    var = pd.DataFrame(index=channels)
    var.index = var.index.map(str)
    var.index.name = "channels"

    # _cells_id = unique(se_labels.data).compute()  # two times computation of unique labels, this is not necessary.
    cells = pd.DataFrame(index=_cells_id)
    _uuid_value = str(uuid.uuid4())[:8]
    cells.index = cells.index.map(lambda x: f"{x}_{labels_layer}_{_uuid_value}")
    cells.index.name = cell_index_name
    adata = AnnData(X=channel_intensities, obs=cells, var=var)

    adata.obs[instance_key] = _cells_id.astype(int)
    adata.obs[region_key] = pd.Categorical([labels_layer] * len(adata.obs))
    # remove background intensity
    adata = adata[adata.obs[instance_key] != 0]
    _cells_id = _cells_id[_cells_id != 0]

    if calculate_center_of_mass:
        # add center of cells here (via the masks).
        coordinates = aggregator._center_of_mass(index=index[index != 0])
        """
        _array_mask = _array_mask.squeeze(0) if to_squeeze else _array_mask

        # dask image center of mass for masks seems bugged (very slow), use in memory scipy.ndimage.center_of_mass for now.
        in_memory = True
        if not in_memory:
            coordinates = center_of_mass(
                image=_array_mask,  # do not use rechunked array mask here, leads to significant increase in required ram.
                label_image=_array_mask,
                index=_cells_id,
            )

            coordinates = coordinates.compute()
        else:
            from scipy import ndimage

            _array_mask_in_memory = _array_mask.compute()
            coordinates = np.array(
                ndimage.center_of_mass(input=_array_mask_in_memory, labels=_array_mask_in_memory, index=_cells_id)
            )
        """
        coordinates += np.array([0, t1y, t1x])  # we account for possible translation in y and x
        # swap y and x, because adata.obsm["SPATIAL"] requires x,y.
        coordinates[:, [-2, -1]] = coordinates[:, [-1, -2]]
        if to_squeeze:
            coordinates = coordinates[:, -2:]
        adata.obsm[spatial_key] = coordinates

    # merge the obs
    if obs_stats is not None and mode not in obs_stats:
        adata.obs.reset_index(inplace=True)
        df_obs = df_obs[df_obs[instance_key] != 0]
        assert adata.obs.shape[0] == df_obs.shape[0], "Number of observations in `adata.obs` and `df_obs` do not match."
        adata.obs = adata.obs.merge(
            df_obs,
            on=[instance_key],
            how="inner",
        )
        adata.obs.set_index(cell_index_name, inplace=True, drop=True)

    if append:
        region = []
        if output_layer in [*sdata.tables]:
            _sanity_check_append_region(
                adata=sdata.tables[output_layer], region_key=region_key, instance_key=instance_key, region=labels_layer
            )
            adata = ad.concat(
                [
                    sdata.tables[output_layer],
                    adata,
                ],
                axis=0,
                join="inner",
            )
            # get the regions already in sdata, and append the new one
            region = sdata.tables[output_layer].uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY]
        region.append(labels_layer)

    else:
        region = [labels_layer]

    sdata = add_table_layer(
        sdata,
        adata=adata,
        output_layer=output_layer,
        region=region,
        instance_key=instance_key,
        region_key=region_key,
        overwrite=overwrite,
    )

    return sdata
