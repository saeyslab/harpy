from __future__ import annotations

import uuid
from collections.abc import Iterable
from functools import reduce
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
from anndata import AnnData
from loguru import logger as log
from spatialdata import SpatialData
from spatialdata.models import TableModel

from harpy.image._image import _get_translation, get_dataarray
from harpy.table._utils import _sanity_check_append_region
from harpy.table.io._add_table import add_table
from harpy.utils._deprecation import _renamed_functions
from harpy.utils._keys import _CELL_INDEX, _CELLSIZE_KEY, _INSTANCE_KEY, _REGION_KEY, _SPATIAL
from harpy.utils._moments import MomentAggregator


def aggregate_image(
    sdata: SpatialData,
    image_name: str | None = None,
    labels_name: str | None = None,
    output_table_name: str = "table_intensities",
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
    Aggregate image intensities per labeled instance into a table.

    For every instance in `labels_name`, the intensities of the selected
    channels of `image_name` are aggregated with `mode` into `.X` of
    `sdata.tables[output_table_name]`: one row per instance, one column per
    channel. The statistics in `obs_stats` are added to `.obs`, and the center
    of mass of each instance to `.obsm[spatial_key]`. All of them are computed
    in one pass over the image and the labels, with
    :class:`harpy.utils.MomentAggregator`.

    The image and labels elements must have the same spatial shape and the
    same translation in `to_coordinate_system`. Misalignments must be resolved
    before calling this function.

    **Writes** one table at `sdata.tables[output_table_name]`::

        adata.X                          float32: mode of each channel, per instance
        adata.var_names                  channel names; index named "channels"
        adata.obs index                  "<instance>_<labels_name>_<token>", named
                                         cell_index_name
        adata.obs[region_key]            categorical: labels_name
        adata.obs[instance_key]          int64: the row's label value
        adata.obs[instance_size_key]     if obs_stats contains "count": float32, pixels
                                         per instance
        adata.obs[f"{stat}_{channel}"]   each other statistic in obs_stats, per channel:
                                         float32, min and max in the image dtype
        adata.obsm[spatial_key]          if calculate_center_of_mass: float64 center of
                                         mass in to_coordinate_system, (x, y) for 2D
                                         labels, (z, x, y) for 3D labels
        adata.uns["spatialdata_attrs"]   region, region_key, instance_key

    The background, label 0, gets no row.

    Parameters
    ----------
    sdata
        The SpatialData object.
    image_name
        Image element from which intensities are aggregated. If `None`, the
        last image element in `sdata` is used, with a warning.
    labels_name
        Labels element whose instances are aggregated over. If `None`, the last
        labels element in `sdata` is used, with a warning.
    output_table_name
        Table element in which to store the resulting AnnData object.
    channels
        Channel or channels of `image_name` to aggregate, given as values of
        its `c` coordinate. If `None`, all channels are used.
    mode
        Statistic stored in `.X`: `"mean"` for the mean intensity per instance,
        `"sum"` for the total intensity.
    obs_stats
        Statistics to add to `.obs`, as a list or a single string. Choose from
        `"sum"`, `"mean"`, `"count"`, `"var"`, `"kurtosis"`, `"skew"`, `"max"`
        and `"min"`.

        - Each statistic except `"count"` is stored per channel as
          `{stat}_{channel_name}`.
        - `"count"`, the number of pixels per instance, is stored as
          `instance_size_key`.
        - The statistic given as `mode` is already in `.X` and is not added to
          `.obs` again.
    to_coordinate_system
        Coordinate system that holds `image_name` and `labels_name`. It should
        be in pixels: only the translation of the elements in it is used.
    chunks
        Optional chunk size used to rechunk the image and labels arrays. A
        tuple gives the chunk size of (z,) y and x. Rechunking on disk ahead of
        time is often more efficient.
    append
        If `True`, append the rows to `sdata.tables[output_table_name]` and add
        `labels_name` to its regions, or create the table if it does not exist.
        The tables are joined with :func:`anndata.concat` and `join="inner"`.
        The existing table must use the same `region_key` and `instance_key`,
        and `labels_name` must not be one of its regions yet. If `False`, the
        table holds only the rows of `labels_name`.
    calculate_center_of_mass
        If `True`, store the center of mass of each instance in
        `.obsm[spatial_key]`. It is computed in the same pass over the labels
        as the intensity statistics.
    region_key
        Column in `.obs` holding the labels element of each row.
    instance_key
        Column in `.obs` holding the label value of each row.
    spatial_key
        Key in `.obsm` for the center of mass of each instance.
    instance_size_key
        Column in `.obs` for the number of pixels of each instance. Only used if
        `obs_stats` contains `"count"`.
    cell_index_name
        Name of the index of `.obs`.
    overwrite
        If `True`, replace `sdata.tables[output_table_name]` if it already
        exists.

    Returns
    -------
    The updated SpatialData object.

    Notes
    -----
    The image and labels are processed chunk by chunk with Dask. For a
    Zarr-backed `sdata`, keep `chunks=None` and store the elements with chunks
    that suit both storage and computation.

    Examples
    --------
    Aggregate image intensity statistics into an AnnData table:

    .. code-block:: python

        import harpy as hp

        sdata = hp.datasets.pixie_example()

        # Compute intensity statistics in coordinate system "fov0"
        sdata = hp.tb.aggregate_image(
            sdata,
            image_name="raw_image_fov0",
            labels_name="label_whole_fov0",
            to_coordinate_system="fov0",
            output_table_name="my_table",
            mode="sum",
            obs_stats="count",  # cell size
            overwrite=True,
        )

        # Append intensity statistics in coordinate system "fov1"
        sdata = hp.tb.aggregate_image(
            sdata,
            image_name="raw_image_fov1",
            labels_name="label_whole_fov1",
            to_coordinate_system="fov1",
            output_table_name="my_table",
            mode="sum",
            obs_stats="count",  # cell size
            append=True,
            overwrite=True,
        )

    See Also
    --------
    harpy.tb.add_feature_matrix : per-instance intensity and morphology features in `.obsm`.
    harpy.utils.MomentAggregator : the per-instance statistics used here, from Dask arrays.
    """
    assert mode in ["sum", "mean"], "'mode' must be either 'sum' or 'mean'."
    if obs_stats is not None:
        if isinstance(obs_stats, str):
            obs_stats = [obs_stats]
    if image_name is None:
        image_name = [*sdata.images][-1]
        log.warning(
            f"No image element specified. "
            f"Extracting intensities from the last image element '{image_name}' of the provided SpatialData object."
        )

    if labels_name is None:
        labels_name = [*sdata.labels][-1]
        log.warning(
            f"No labels element specified. "
            f"Using mask from labels element '{labels_name}' of the provided SpatialData object."
        )

    if channels is not None:
        channels = list(channels) if isinstance(channels, Iterable) and not isinstance(channels, str) else [channels]

    # currently this function will only work if image_name and labels_name have the same shape.
    # And are in same position, i.e. if one is translated, other should be translated with same offset
    se_image = get_dataarray(sdata, element_name=image_name)
    se_labels = get_dataarray(sdata, element_name=labels_name)

    if se_image.data.shape[1:] != se_labels.data.shape:
        raise ValueError(
            "Only arrays with same spatial shape are currently supported, "
            f"but image element with name {image_name} has shape {se_image.data.shape}, "
            f"while labels element with name {labels_name} has shape {se_labels.data.shape}."
        )

    t1x, t1y = _get_translation(se_image, to_coordinate_system=to_coordinate_system)
    t2x, t2y = _get_translation(se_labels, to_coordinate_system=to_coordinate_system)

    if (t1x, t1y) != (t2x, t2y):
        raise ValueError(
            f"image element with name {image_name} should "
            f"be registered to labels element with name {labels_name} in coordinate system {to_coordinate_system}."
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
        f"Some channels specified via 'channels' could not be found in image element '{image_name}'. Please choose 'channels' from '{list(se_image.c.data)}'."
    )
    channel_indices = [list(se_image.c.data).index(channel) for channel in channels]
    _array_img = _array_img[channel_indices]

    # MomentAggregator requires the labels and the image to share spatial (z, y, x) chunks.
    # The labels and image already share spatial shape (validated above), so align the
    # (cheaper) single-band labels onto the image's spatial chunks when they differ.
    # Compare the chunks, not the chunk sizes: a crop off the chunk grid has the same chunk
    # size with other chunk boundaries.
    if _array_mask_rechunked.chunks != _array_img.chunks[1:]:
        log.info(
            "Image and labels have different chunks; rechunking labels to match "
            f"the image's spatial chunks {_array_img.chunksize[1:]} for aggregation."
        )
        _array_mask_rechunked = _array_mask_rechunked.rechunk(_array_img.chunks[1:])

    aggregator = MomentAggregator(
        mask_dask_array=_array_mask_rechunked,
        image_dask_array=_array_img,
        instance_key=instance_key,
    )

    # Every statistic in one pass: the mode, then the statistics for .obs with max and min
    # last, as .obs orders them, then the center of mass.
    other_stats = [stat for stat in obs_stats if stat != mode] if obs_stats is not None else []
    obs_stats_funcs = [stat for stat in other_stats if stat not in ("max", "min")]
    obs_stats_funcs += [stat for stat in ("max", "min") if stat in other_stats]
    stats_funcs = [mode, *obs_stats_funcs]
    if calculate_center_of_mass:
        stats_funcs.append("center_of_mass")
    # Without an index, the rows are every label of the mask, background included, in ascending order.
    dfs = aggregator.aggregate_stats(stats_funcs=stats_funcs)

    def rename_columns(_df: pd.DataFrame, prefix: str):
        # helper function to rename column of a dataframe to prefix_channel_name
        new_columns = []
        for _name in _df.columns:
            if _name != instance_key:
                new_columns.append(f"{prefix}_{channels[_name]}")
            else:
                new_columns.append(_name)
        _df.columns = new_columns

    df_X = dfs[0]  # the first one is the mode

    if obs_stats_funcs:
        df_obs = dfs[1 : 1 + len(obs_stats_funcs)]
        for _df, _prefix in zip(df_obs, obs_stats_funcs, strict=True):
            if _prefix != "count":
                rename_columns(_df, prefix=_prefix)
            else:
                _df.rename(columns={0: instance_size_key}, inplace=True)
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
    cells.index = cells.index.map(lambda x: f"{x}_{labels_name}_{_uuid_value}")
    cells.index.name = cell_index_name
    adata = AnnData(X=channel_intensities, obs=cells, var=var)

    adata.obs[instance_key] = _cells_id.astype(int)
    adata.obs[region_key] = pd.Categorical([labels_name] * len(adata.obs))
    # remove background intensity
    adata = adata[adata.obs[instance_key] != 0]
    _cells_id = _cells_id[_cells_id != 0]

    if calculate_center_of_mass:
        # center of the cells, (z, y, x), in the row order of df_X, without background
        df_center = dfs[-1]
        coordinates = df_center[df_center[instance_key] != 0].drop(columns=[instance_key]).to_numpy()
        coordinates += np.array([0, t1y, t1x])  # we account for possible translation in y and x
        # swap y and x, because adata.obsm["SPATIAL"] requires x,y.
        coordinates[:, [-2, -1]] = coordinates[:, [-1, -2]]
        if to_squeeze:
            coordinates = coordinates[:, -2:]
        adata.obsm[spatial_key] = coordinates

    # merge the obs
    if obs_stats_funcs:
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
        if output_table_name in [*sdata.tables]:
            _sanity_check_append_region(
                adata=sdata.tables[output_table_name],
                region_key=region_key,
                instance_key=instance_key,
                region=labels_name,
            )
            adata = ad.concat(
                [
                    sdata.tables[output_table_name],
                    adata,
                ],
                axis=0,
                join="inner",
            )
            # get the regions already in sdata, and append the new one
            region = sdata.tables[output_table_name].uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY]
        region.append(labels_name)

    else:
        region = [labels_name]

    sdata = add_table(
        sdata,
        adata=adata,
        output_table_name=output_table_name,
        region=region,
        instance_key=instance_key,
        region_key=region_key,
        overwrite=overwrite,
    )

    return sdata


__getattr__ = _renamed_functions(__name__, namespace="harpy.tb", renamed={"allocate_intensity": "aggregate_image"})
