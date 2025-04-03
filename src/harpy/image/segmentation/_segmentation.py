from __future__ import annotations

import os
import shutil
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from pathlib import Path
from types import MappingProxyType
from typing import Any

import dask.array as da
import numpy as np
from dask.array import Array
from nptyping import NDArray, Shape
from spatialdata import SpatialData, bounding_box_query
from spatialdata.models._utils import MappingToCoordinateSystem_t
from spatialdata.models.models import ScaleFactors_t
from spatialdata.transformations import get_transformation
from upath import UPath
from xarray import DataArray

from harpy.image._image import (
    _fix_dimensions,
    _get_spatial_element,
    _get_translation,
    add_labels_layer,
)
from harpy.image.segmentation._align_masks import align_labels_layers
from harpy.image.segmentation._utils import (
    _SEG_DTYPE,
    _add_depth_to_chunks_size,
    _check_boundary,
    _clean_up_masks,
    _get_block_position,
    _link_labels,
    _merge_masks,
    _rechunk_overlap,
    _substract_depth_from_chunks_size,
)
from harpy.image.segmentation.segmentation_models._baysor import _baysor as _model_points
from harpy.image.segmentation.segmentation_models._cellpose import cellpose_callable as _model
from harpy.points._points import add_points_layer
from harpy.shape._shape import add_shapes_layer
from harpy.utils._keys import _GENES_KEY
from harpy.utils._transformations import _identity_check_transformations_points
from harpy.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def segment(
    sdata: SpatialData,
    img_layer: str,
    model: Callable[..., NDArray] = _model,
    output_labels_layer: str | list[str] = "segmentation_mask",
    output_shapes_layer: str | list[str] | None = "segmentation_mask_boundaries",
    labels_layer_align: str | None = None,
    depth: tuple[int, int] | int = 100,
    chunks: str | int | tuple[int, int] | None = None,
    boundary: str = "reflect",
    trim: bool = False,
    iou: bool = True,
    iou_depth: tuple[int, int] | int = 2,
    iou_threshold: float = 0.7,
    crd: tuple[int, int, int, int] | None = None,
    to_coordinate_system: str = "global",
    scale_factors: ScaleFactors_t | None = None,
    overwrite: bool = False,
    **kwargs: Any,
) -> SpatialData:
    """
    Segment images using a provided model and add segmentation results (labels layer and shapes layer) to the SpatialData object.

    Parameters
    ----------
    sdata
        The SpatialData object containing the image layer to segment.
    img_layer
        The image layer in `sdata` to be segmented.
    model
        The segmentation model function used to process the images.
        Callable should take as input numpy arrays of dimension `(z,y,x,c)` and return labels of dimension `(z,y,x,c)`. It can have an arbitrary number of other parameters.
    output_labels_layer
        Name of the label layer in which segmentation results will be stored in `sdata`.
        Can be a list of strings, if `model` returns multi channel mask.
        If provided as a list, its length should match the `c` dimension of the output of `model`.
    output_shapes_layer
        Name of the shapes layer where boundaries obtained output_labels_layer will be stored. If set to None, shapes won't be stored.
        Can be a list of strings, if `model` returns multi channel mask.
        If provided as a list, its length should match the `c` dimension of the output of `model`.
    labels_layer_align
        Name of the labels layer in `output_labels_layer` to align to if `model` retuns multi channel mask.
    depth
        The depth in `y` and x dimension. The depth parameter is passed to `dask.array.map_overlap`. If trim is set to `False`,
        it's recommended to set the depth to a value greater than twice the estimated diameter of the cells/nulcei.
    chunks
        Chunk sizes for processing. Can be a string, integer or tuple of integers. If chunks is a `tuple`,
        they  contain the chunk size that will be used in `y` and `x` dimension. Chunking in `z` or `c` dimension is not supported.
    boundary
        Boundary parameter passed to `dask.array.map_overlap`.
    trim
        If set to `True`, overlapping regions will be processed using the `squidpy` algorithm.
        If set to `False`, the `harpy` algorithm will be employed instead. For dense cell distributions,
        we recommend setting trim to `False`.
    iou
        If set to `True`, will try to harmonize labels across chunks using a label adjacency graph with an iou threshold (see `harpy.image.segmentation.utils._link_labels`). If set to `False`, conflicts will be resolved using an algorithm that only retains masks with the center in the chunk.
        Setting `iou` to `False` gives good results if there is reasonable agreement of the predicted labels across adjacent chunks.
    iou_depth
        iou depth used for harmonizing labels across chunks. Note that if `labels_layer_align` is specified, `iou_depth` will also be used for harmonizing labels between different chunks.
    iou_threshold
        iou threshold used for harmonizing labels across chunks. Note that if `labels_layer_align` is specified, `iou_threshold` will also be used for harmonizing labels between different chunks.
    crd
        The coordinates specifying the region of the image to be segmented. Defines the bounds `(x_min, x_max, y_min, y_max)`.
    to_coordinate_system
        The coordinate system to which the `crd` is specified. Ignored if `crd` is None.
    scale_factors
        Scale factors to apply for multiscale.
    overwrite
        If `True`, overwrites the existing layers if they exist. Otherwise, raises an error if the layers exist.
    **kwargs
        Additional keyword arguments passed to the provided `model`.

    Returns
    -------
    Updated `sdata` object containing the segmentation results.

    Raises
    ------
    TypeError
        If the provided `model` is not a callable.
    """
    fn_kwargs = kwargs

    if not callable(model):
        raise TypeError(f"Expected `model` to be a callable, found `{type(model)}`.")

    # kwargs to be passed to map_overlap/map_blocks
    kwargs = {}
    kwargs.setdefault("depth", depth)
    kwargs.setdefault("boundary", boundary)
    kwargs.setdefault("chunks", chunks)
    kwargs.setdefault("trim", trim)
    kwargs.setdefault("iou", iou)
    kwargs.setdefault("iou_depth", iou_depth)
    kwargs.setdefault("iou_threshold", iou_threshold)

    segmentation_model = SegmentationModelStains(model)

    sdata = segmentation_model._segment_layer(
        sdata,
        img_layer=img_layer,
        output_labels_layer=output_labels_layer,
        output_shapes_layer=output_shapes_layer,
        labels_layer_align=labels_layer_align,
        crd=crd,
        to_coordinate_system=to_coordinate_system,
        scale_factors=scale_factors,
        overwrite=overwrite,
        fn_kwargs=fn_kwargs,
        **kwargs,
    )
    return sdata


def segment_points(
    sdata: SpatialData,
    labels_layer: str,  # the prior
    points_layer: str,
    name_x: str = "x",
    name_y: str = "y",
    name_gene: str = _GENES_KEY,
    model: Callable[..., NDArray] = _model_points,
    output_labels_layer: str | list[str] = "segmentation_mask",
    output_shapes_layer: str | list[str] | None = "segmentation_mask_boundaries",
    labels_layer_align: str | None = None,
    depth: tuple[int, int] | int = 100,
    chunks: str | int | tuple[int, int] | None = None,
    boundary: str = "reflect",
    trim: bool = False,
    iou: bool = True,
    iou_depth: tuple[int, int] | int = 2,
    iou_threshold: float = 0.7,
    crd: tuple[int, int, int, int] | None = None,
    to_coordinate_system: str = "global",
    scale_factors: ScaleFactors_t | None = None,
    overwrite: bool = False,
    **kwargs: Any,
) -> SpatialData:
    """
    Segment images using a `points_layer` and a prior (`labels_layer`) and add segmentation results (labels layer and shapes layer) to the SpatialData object.

    Currently only segmentation using a prior is supported (i.e. `labels_layer` should be provided).
    The `points_layer` and the `labels_layer` should be registered (i.e. same coordinate space in `sdata`).

    Parameters
    ----------
    sdata
        The SpatialData object containing the image layer to segment.
    labels_layer
        The labels layer in `sdata` to be used as a prior.
    points_layer
        The points layer in `sdata` to be used for segmentation.
    name_x
        Column name for x-coordinates of the transcripts in the points layer, by default "x".
    name_y
        Column name for y-coordinates of the transcripts in the points layer, by default "y".
    name_gene
        Column name in the points_layer representing gene information.
    model
        The segmentation model function used to process the images.
        Callable should take as input numpy arrays of dimension `(z,y,x,c)`, a pandas dataframe with the transcripts,
        and parameters 'name_x', 'name_y' and 'name_gene' with the column names of the x and y location and the column
        name for the transcripts. It should return labels of dimension `(z,y,x,c)`.
        Currently only 2D segmentation is supported `(y,x)`.
        It can have an arbitrary number of other parameters.
    output_labels_layer
        Name of the labels layer in which segmentation results will be stored in `sdata`.
        Can be a list of strings, if `model` returns multi channel mask.
        If provided as a list, its length should match the `c` dimension of the output of `model`.
    output_shapes_layer
        Name of the shapes layer where boundaries obtained output_labels_layer will be stored. If set to None, shapes won't be stored.
        Can be a list of strings, if `model` returns multi channel mask.
        If provided as a list, its length should match the `c` dimension of the output of `model`.
    labels_layer_align
        Name of the labels layer in `output_labels_layer` to align to if `model` retuns multi channel mask.
    depth
        The depth in `y` and `x` dimension. The depth parameter is passed to `dask.array.map_overlap`. If trim is set to `False`,
        it's recommended to set the depth to a value greater than twice the estimated diameter of the cells/nulcei.
    chunks
        Chunk sizes for processing. Can be a string, integer or tuple of integers. If chunks is a Tuple,
        they contain the chunk size that will be used in `y` and `x` dimension. Chunking in `z` or `c` dimension is not supported.
    boundary
        Boundary parameter passed to `dask.array.map_overlap`.
    trim
        If set to True, overlapping regions will be processed using the `squidpy` algorithm.
        If set to False, the `harpy` algorithm will be employed instead. For dense cell distributions,
        we recommend setting trim to False.
    iou
        If set to True, will try to harmonize labels across chunks using a label adjacency graph with an iou threshold (see `harpy.image.segmentation.utils._link_labels`). If set to False, conflicts will be resolved using an algorithm that only retains masks with the center in the chunk.
        Setting `iou` to False gives good results if there is reasonable agreement of the predicted labels accross adjacent chunks.
    iou_depth
        iou depth used for harmonizing labels across chunks. Note that if `labels_layer_align` is specified, `iou_depth` will also be used for harmonizing labels between different chunks.
    iou_threshold
        iou threshold used for harmonizing labels across chunks. Note that if `labels_layer_align` is specified, `iou_threshold` will also be used for harmonizing labels between different chunks.
    crd
        The coordinates specifying the region of the image to be segmented. Defines the bounds `(x_min, x_max, y_min, y_max)`.
    to_coordinate_system
        The coordinate system to which the `crd` is specified. Ignored if `crd` is None.
    scale_factors
        Scale factors to apply for multiscale.
    overwrite
        If `True`, overwrites the existing layers if they exist. Otherwise, raises an error if the layers exist.
    **kwargs
        Additional keyword arguments passed to the provided `model`.

    Returns
    -------
    Updated `sdata` object containing the segmentation results.

    Raises
    ------
    TypeError
        If the provided `model` is not callable.
    """
    fn_kwargs = kwargs

    if not callable(model):
        raise TypeError(f"Expected `model` to be a callable, found `{type(model)}`.")

    # kwargs to be passed to map_overlap/map_blocks
    kwargs = {}
    kwargs.setdefault("depth", depth)
    kwargs.setdefault("boundary", boundary)
    kwargs.setdefault("chunks", chunks)
    kwargs.setdefault("trim", trim)
    kwargs.setdefault("iou", iou)
    kwargs.setdefault("iou_depth", iou_depth)
    kwargs.setdefault("iou_threshold", iou_threshold)

    segmentation_model = SegmentationModelPoints(model)

    sdata = segmentation_model._segment_layer(
        sdata,
        labels_layer=labels_layer,
        points_layer=points_layer,
        name_x=name_x,
        name_y=name_y,
        name_gene=name_gene,
        output_labels_layer=output_labels_layer,
        output_shapes_layer=output_shapes_layer,
        labels_layer_align=labels_layer_align,
        crd=crd,
        to_coordinate_system=to_coordinate_system,
        scale_factors=scale_factors,
        overwrite=overwrite,
        fn_kwargs=fn_kwargs,
        **kwargs,
    )
    return sdata


class SegmentationModel(ABC):
    def __init__(self, model: Callable[..., NDArray]):
        self._model = model

    @abstractmethod
    def _segment_layer(
        self,
        sdata: SpatialData,
        *args,
        **kwargs,
    ) -> SpatialData:
        pass

    def _precondition_output_layers_name(
        self, output_labels_layer: str | list[str] | None, output_shapes_layer: str | list[str] | None
    ) -> tuple[list[str] | None, list[str] | None]:
        def _fix_name(name: str | Iterable[str]):
            return list(name) if isinstance(name, Iterable) and not isinstance(name, str) else [name]

        if output_labels_layer is not None:
            output_labels_layer = _fix_name(output_labels_layer)

        if output_shapes_layer is not None:
            output_shapes_layer = _fix_name(output_shapes_layer)

        if output_labels_layer is not None and output_shapes_layer is not None:
            assert (
                len(output_labels_layer) == len(output_shapes_layer)
            ), "It 'output_labels_layer' or 'output_shapes_layer' is provided as a list, they should be of the same length."

        return output_labels_layer, output_shapes_layer

    def _precondition(self, se: DataArray, kwargs: dict[Any, Any]) -> tuple[Array, dict[Any, Any]]:
        # take dask array and put channel dimension last,
        # so we have ( z, y, x, c ), also do some checks on depth and chunks

        if se.data.ndim == 4:
            assert se.dims == (
                "c",
                "z",
                "y",
                "x",
            ), "dimension should be in order: ('c', 'z' , 'y', 'x')."
            # transpose x, so channel dimension is last
            x = _fix_dimensions(se.data, dims=se.dims, target_dims=("z", "y", "x", "c"))

        elif se.data.ndim == 3:
            assert se.dims == (
                "c",
                "y",
                "x",
            ), "dimension should be in order: ('c', 'y', 'x')."
            # transpose x, so channel dimension is last
            x = _fix_dimensions(se.data, dims=se.dims, target_dims=("y", "x", "c"))
            # add trivial z dimension.
            x = x[None, ...]
        else:
            raise ValueError("Only 3D and 4D arrays are supported, i.e. (c, (z), y, x).")

        def _fix_depth(kwargs: dict, key: str) -> dict:
            if key in kwargs:
                depth = kwargs[key]
                if isinstance(depth, int):
                    kwargs[key] = {0: 0, 1: depth, 2: depth, 3: 0}
                else:
                    assert len(depth) == x.ndim - 2, f"Please (only) provide '{key}' for ( 'y', 'x')."
                    # set depth for every dimension
                    kwargs[key] = {0: 0, 1: depth[0], 2: depth[1], 3: 0}
            return kwargs

        kwargs = _fix_depth(kwargs, key="depth")
        kwargs = _fix_depth(kwargs, key="iou_depth")

        if "chunks" in kwargs:
            chunks = kwargs["chunks"]
            if chunks is not None:
                if not isinstance(chunks, int | str):
                    assert len(chunks) == x.ndim - 2, "Please (only) provide chunks for ( 'y', 'x')."
                    chunks = (x.shape[0], chunks[0], chunks[1], x.shape[-1])
                    kwargs["chunks"] = chunks
                elif isinstance(chunks, int):
                    chunks = (x.shape[0], chunks, chunks, x.shape[-1])
                    kwargs["chunks"] = chunks
            else:
                if x.chunksize[0] != x.shape[0] or x.chunksize[-1] != x.shape[-1]:
                    log.info(
                        "Provided array is chunked in 'z' and/or 'c' dimension. Will rechunk in 'z' and/or 'c' dimension."
                    )
                    kwargs["chunks"] = (x.shape[0], x.chunksize[1], x.chunksize[2], x.shape[-1])

        return x, kwargs

    def _add_to_sdata(
        self,
        sdata: SpatialData,
        x_labels: Array,  # z,y,x,c
        output_labels_layer: list[str],
        output_shapes_layer: list[str] | None,
        labels_layer_align: str | None = None,
        transformations: MappingToCoordinateSystem_t | None = None,
        scale_factors: ScaleFactors_t | None = None,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> SpatialData:
        # x_labels c dimension should be equal to the number of labels layers specified.
        # note that this assert will never fail due to how chunks parameter works in map_overlap.
        # i.e. we set chunks=(,...(len( output_labels_layer ),).
        assert x_labels.shape[-1] == len(output_labels_layer), (
            f"Expected {len(output_labels_layer)} segmentation masks (based on 'output_labels_layer'), "
            f"but got {x_labels.shape[-1]} masks from the segmentation model."
        )
        # squeeze the z-dim if it is 1 (i.e. case where you did not do 3D segmentation),
        # otherwise 2D labels layer would be saved as 3D
        if x_labels.shape[0] == 1:
            x_labels = x_labels.squeeze(0)

        # now iterate over x_labels and add to sdata
        for c_index in range(x_labels.shape[-1]):
            _x_labels = x_labels[..., c_index]
            _output_labels_layer = output_labels_layer[c_index]
            # do not write to multiscale if we have to align afterwards, would be waste of write operations.
            if labels_layer_align is not None and _output_labels_layer != labels_layer_align:
                _scale_factors = None
            else:
                _scale_factors = scale_factors
            sdata = add_labels_layer(
                sdata,
                arr=_x_labels,
                output_layer=_output_labels_layer,
                chunks=_x_labels.chunksize,
                transformations=transformations,
                scale_factors=_scale_factors,
                overwrite=overwrite,
            )

        # align the labels layers if labels_layer_align is specified, and if there is more than one labels layer.
        if labels_layer_align is not None and len(output_labels_layer) > 1:
            log.info(f"Aligning labels layers: {output_labels_layer}")
            depth = kwargs["depth"]
            iou_depth = kwargs["iou_depth"]
            chunks = kwargs["chunks"]

            for _output_labels_layer in output_labels_layer:
                if _output_labels_layer == labels_layer_align:
                    # we do not need to align labels_layer_align with labels_layer_align
                    continue
                sdata = align_labels_layers(
                    sdata,
                    labels_layer_1=_output_labels_layer,
                    labels_layer_2=labels_layer_align,
                    depth=(
                        depth[1],
                        depth[2],
                    ),
                    chunks=chunks
                    if isinstance(chunks, str | type(None))
                    else (chunks[1], chunks[2]),  # get this from kwargs. Make a copy of kwargs before it is popped
                    iou_depth=(iou_depth[1], iou_depth[2]),
                    iou_threshold=kwargs["iou_threshold"],
                    output_labels_layer=_output_labels_layer,
                    output_shapes_layer=None,
                    scale_factors=scale_factors,
                    overwrite=True,
                )

        # only calculate shapes layer if it is specified
        if output_shapes_layer is not None:
            for i, _output_labels_layer in enumerate(output_labels_layer):
                se_labels = _get_spatial_element(sdata, layer=_output_labels_layer)
                _output_shapes_layer = output_shapes_layer[i]
                # convert the labels to polygons and add them as shapes layer to sdata
                sdata = add_shapes_layer(
                    sdata,
                    input=se_labels.data,
                    output_layer=_output_shapes_layer,
                    transformations=transformations,
                    overwrite=overwrite,
                )

        return sdata

    def _segment(
        self,
        x: Array,  # array with dimension z,y,x,c
        temp_path: str | Path,
        c_dim_output_labels: int,
        fn_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any,  # keyword arguments to be passed to map_overlap/map_blocks
    ) -> Array:  # array with dimension z,y,x,c
        assert x.ndim == 4, "Please provide a 4D array (('z', 'y', 'x', 'c'))."
        chunks = kwargs.pop("chunks", None)
        depth = kwargs.pop("depth", {0: 0, 1: 100, 2: 100, 3: 0})
        iou_depth = kwargs.pop("iou_depth", {0: 0, 1: 2, 2: 2, 3: 0})
        for _depth in [depth, iou_depth]:
            assert len(_depth) == 4, "Please provide depth for (('z', 'y', 'x', 'c'))"
            assert _depth[0] == 0, "Depth not equal to 0 for 'z' dimension is not supported"
            assert _depth[3] == 0, "Depth not equal to 0 for 'c' dimension is not supported"
        boundary = kwargs.pop("boundary", "reflect")
        trim = kwargs.pop("trim", False)
        iou = kwargs.pop("iou", True)
        iou_threshold = kwargs.pop("iou_threshold", 0.7)

        if not trim and depth[1] == 0 or depth[2] == 0:
            log.warning("Depth equal to zero not supported with trim==False, setting trim to True.")
            trim = True

        _check_boundary(boundary)

        # make depth uniform (dict with depth for z,y and x)
        # + rechunk so that we ensure minimum chunksize, in order to control output_chunks sizes.
        x = _rechunk_overlap(
            x,
            depth=depth,
            chunks=chunks,
        )

        # remove trivial depth==0 for c dimension
        depth.pop(3)
        iou_depth.pop(3)

        output_chunks = _add_depth_to_chunks_size(x.chunks, depth)
        #  size of output chunks in channel dimension is c_dim_output_labels, because we do not allow chunking in that dimension
        output_chunks = output_chunks[:-1] + ((c_dim_output_labels,),)

        # shift added to results of every chunk (i.e. if shift is 4, then 0 0 0 0 will be added to every label).
        # These 0's are then filled in with block_id number. This way labels are unique accross the different chunks.
        # not that if x.numblocks.bit_length() would be close to 16 bit, and each chunks has labels up to 16 bits,
        # this could lead to collisions.
        # ignore channel dim (num_blocks[3]), because num_blocks[3]==1, because we do not allow chunking in c dimension.
        num_blocks = x.numblocks
        shift = int(np.prod(num_blocks[0] * num_blocks[1] * num_blocks[2]) - 1).bit_length()

        x_labels = da.map_overlap(
            self._segment_chunk,
            x,
            dtype=_SEG_DTYPE,
            num_blocks=num_blocks,
            shift=shift,
            # need to pass _output_chunks because we want to query the dask dataframe in case we segment points layer,
            # so we need to know exact location of block in full array,
            _output_chunks=output_chunks,
            _depth=depth,
            trim=trim,
            allow_rechunk=False,  # already dealed with correcting for case where depth > chunksize
            chunks=output_chunks,  # e.g. ((7,) ,(1024+60, 1024+60, 452+60), (1024+60, 1024+60, 452+60), (1,) ),
            depth=depth,
            boundary=boundary,
            fn_kwargs=fn_kwargs,
            **kwargs,
        )

        # write to intermediate zarr store if sdata is backed to reduce ram memory.
        if temp_path is not None:
            zarr_path = os.path.join(temp_path, f"labels_{uuid.uuid4()}.zarr")
            _chunks = x_labels.chunks
            x_labels.rechunk(x_labels.chunksize).to_zarr(
                zarr_path,
                overwrite=False,
            )
            x_labels = da.from_zarr(zarr_path)
            x_labels = x_labels.rechunk(_chunks)
        else:
            x_labels = x_labels.persist()

        # we support multi channel labels layer
        _all_labels = []
        for c_index in range(x_labels.shape[-1]):
            _x_labels = x_labels[..., c_index]
            # if trim==True --> use squidpy's way of handling neighbouring blocks
            if trim:
                from dask_image.ndmeasure._utils._label import (
                    connected_components_delayed,
                    label_adjacency_graph,
                    relabel_blocks,
                )

                # max because labels are not continuous (and won't be continuous)
                label_groups = label_adjacency_graph(_x_labels, None, _x_labels.max())
                new_labeling = connected_components_delayed(label_groups)
                _x_labels = relabel_blocks(_x_labels, new_labeling)
                _x_labels = _x_labels.rechunk(_x_labels.chunksize)

            elif iou:
                iou_depth = da.overlap.coerce_depth(len(depth), iou_depth)

                if any(iou_depth[ax] > depth[ax] for ax in depth.keys()):
                    raise ValueError(f"iou_depth {iou_depth} > depth {depth}")

                trim_depth = {k: depth[k] - iou_depth[k] for k in depth.keys()}
                _x_labels = da.overlap.trim_internal(_x_labels, trim_depth, boundary=boundary)
                _x_labels = _link_labels(
                    _x_labels,
                    _x_labels.max(),
                    iou_depth,
                    iou_threshold=iou_threshold,
                )

                _x_labels = da.overlap.trim_internal(_x_labels, iou_depth, boundary=boundary)

                _x_labels = _x_labels.rechunk(_x_labels.chunksize)

            else:
                _x_labels = da.map_blocks(
                    _clean_up_masks,
                    _x_labels,
                    dtype=_SEG_DTYPE,
                    depth=depth,
                    **kwargs,
                )

                output_chunks = _substract_depth_from_chunks_size(_x_labels.chunks, depth=depth)

                _x_labels = da.map_overlap(
                    _merge_masks,
                    _x_labels,
                    dtype=_SEG_DTYPE,
                    num_blocks=_x_labels.numblocks,
                    trim=False,
                    allow_rechunk=False,  # already dealed with correcting for case where depth > chunksize
                    chunks=output_chunks,  # e.g. ((7,) ,(1024, 1024, 452), (1024, 1024, 452),)
                    depth=depth,
                    boundary="reflect",
                    _depth=depth,
                )

                _x_labels = _x_labels.rechunk(_x_labels.chunksize)

            _all_labels.append(_x_labels)

        # returns a dask array containing labels with dimension (z,y,x,c)
        x_labels = da.stack(_all_labels, axis=-1)

        log.info("Linking labels across chunks.")

        if x_labels.shape[-1] > 1:
            # write to intermediate zarr store, otherwise will redo solving of chunks for each label channel.
            if temp_path is not None:
                zarr_path = os.path.join(temp_path, f"labels_{uuid.uuid4()}.zarr")
                _chunks = x_labels.chunks
                x_labels.rechunk(x_labels.chunksize).to_zarr(
                    zarr_path,
                    overwrite=False,
                )
                x_labels = da.from_zarr(zarr_path)
            else:
                x_labels = x_labels.persist()

        return x_labels

    def _segment_chunk(
        self,
        block: NDArray[Shape[Any, Any, Any, Any]],
        block_id: tuple[int, ...],
        num_blocks: tuple[int, ...],
        shift: int,
        fn_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs,
    ) -> NDArray[Shape[Any, Any, Any, Any]]:
        """Method should be implemented in each subclass to handle the segmentation logic on each chunk"""
        if len(num_blocks) == 4:
            if num_blocks[0] != 1:
                raise ValueError(
                    f"Expected the number of blocks in the Z-dimension to be `1`, found `{num_blocks[0]}`."
                )
            if num_blocks[-1] != 1:
                raise ValueError(
                    f"Expected the number of blocks in the c-dimension to be `1`, found `{num_blocks[-1]}`."
                )

            # note: ignore num_blocks[3]==1 and block_id[3]==0, because we assume there is not chunking in c dimension
            # actually also not in z dimension, so we could remove block_id[0] (always 0)
            block_num = block_id[0] * (num_blocks[1] * num_blocks[2]) + block_id[1] * (num_blocks[2]) + block_id[2]

        else:
            raise ValueError(f"Expected `4` dimensional chunks, found `{len(num_blocks)}`.")

        labels = self._custom_segment_chunk(block, block_id, fn_kwargs=fn_kwargs, **kwargs)

        mask: NDArray = labels > 0
        labels[mask] = (labels[mask] << shift) | block_num

        return labels

    @abstractmethod
    def _custom_segment_chunk(
        self,
        block: NDArray[Shape[Any, Any, Any, Any]],
        block_id: tuple[int, ...],
        fn_kwargs: Mapping[str, Any],
        **kwargs,
    ) -> NDArray[Shape[Any, Any, Any, Any]]:
        """
        Implement the unique part of _segment_chunk in each subclass.

        Takes as input a numpy array (('z', 'y', 'x', 'c')),
        and returns a numpy array containing predicted labels (('z', 'y', 'x', 'c')), with c-dimension==1
        """
        pass


class SegmentationModelStains(SegmentationModel):
    def __init__(
        self,
        model: Callable[..., NDArray],
    ):
        self._model = model

    def _segment_layer(
        self,
        sdata: SpatialData,
        img_layer: str,
        output_labels_layer: str | list[str] = "segmentation_mask",
        output_shapes_layer: str | list[str] | None = "segmentation_mask_boundaries",
        labels_layer_align: str | None = None,
        crd: tuple[int, int, int, int] | None = None,
        to_coordinate_system: str = "global",
        scale_factors: ScaleFactors_t | None = None,
        overwrite: bool = False,
        fn_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any,
    ) -> SpatialData:
        output_labels_layer, output_shapes_layer = self._precondition_output_layers_name(
            output_labels_layer, output_shapes_layer
        )
        if labels_layer_align is not None and labels_layer_align not in output_labels_layer:
            raise ValueError(
                f"'labels_layer_align' ('{labels_layer_align}') should be one of the values in 'output_labels_layer' ({output_labels_layer})."
            )
        c_dim_output_labels = len(output_labels_layer)

        se = _get_spatial_element(sdata, layer=img_layer)
        se_crop = None
        if crd is not None:
            se_crop = bounding_box_query(
                se,
                axes=["x", "y"],
                min_coordinate=[crd[0], crd[2]],
                max_coordinate=[crd[1], crd[3]],
                target_coordinate_system=to_coordinate_system,
            )
            if se_crop is not None:
                se = se_crop
            else:
                log.warning(
                    f"Cropped spatial element using crd '{crd}' is None. Falling back to processing on full dataset."
                )

        x, kwargs = self._precondition(se, kwargs)

        # rechunk to ensure we do not have irregular chunksize after taking a crop
        if se_crop is not None:
            x = x.rechunk(x.chunksize)

        if sdata.is_backed():
            _temp_path = UPath(sdata.path).parent / f"tmp_{uuid.uuid4()}"
        else:
            _temp_path = None

        x_labels = self._segment(
            x,
            temp_path=_temp_path,
            c_dim_output_labels=c_dim_output_labels,
            fn_kwargs=fn_kwargs,
            **deepcopy(kwargs),
        )

        sdata = self._add_to_sdata(
            sdata,
            x_labels,
            output_labels_layer=output_labels_layer,
            output_shapes_layer=output_shapes_layer,
            labels_layer_align=labels_layer_align,
            transformations=get_transformation(se, get_all=True),
            scale_factors=scale_factors,
            overwrite=overwrite,
            **kwargs,
        )

        if _temp_path is not None:
            # TODO this will not work if sdata is remote (e.g. s3 bucket).
            shutil.rmtree(_temp_path)

        return sdata

    def _custom_segment_chunk(
        self,
        block: NDArray[Shape[Any, Any, Any, Any]],
        block_id: tuple[int, ...],
        fn_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs,
    ) -> NDArray[Shape[Any, Any, Any, Any]]:
        labels = self._model(block, **fn_kwargs).astype(_SEG_DTYPE)
        return labels


class SegmentationModelPoints(SegmentationModel):
    def __init__(
        self,
        model: Callable[..., NDArray],
    ):
        self._model = model

    def _segment_layer(
        self,
        sdata: SpatialData,
        labels_layer: str,  # prior, required for now
        points_layer: str,
        name_x: str = "x",
        name_y: str = "y",
        name_gene: str = _GENES_KEY,
        output_labels_layer: str | list[str] = "segmentation_mask",
        output_shapes_layer: str | list[str] | None = "segmentation_mask_boundaries",
        labels_layer_align: str | None = None,
        crd: tuple[int, int, int, int] | None = None,
        to_coordinate_system: str = "global",
        scale_factors: ScaleFactors_t | None = None,
        overwrite: bool = False,
        fn_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs: Any,
    ) -> SpatialData:
        fn_kwargs["name_x"] = name_x
        fn_kwargs["name_y"] = name_y
        fn_kwargs["name_gene"] = name_gene

        output_labels_layer, output_shapes_layer = self._precondition_output_layers_name(
            output_labels_layer, output_shapes_layer
        )
        if labels_layer_align is not None and labels_layer_align not in output_labels_layer:
            raise ValueError(
                f"'labels_layer_align' ('{labels_layer_align}') should be one of the values in 'output_labels_layer' ({output_labels_layer})."
            )
        c_dim_output_labels = len(output_labels_layer)

        se = _get_spatial_element(sdata, layer=labels_layer)

        # Now we check that there are no scaling and rotations defined on se; and that points layer has identiy transformation associated.
        # We do not allow a transformation other than translation in y and x defined on labels layer.
        _get_translation(se, to_coordinate_system=to_coordinate_system)
        # We do not allow that a transformation other than identity is defined on points layer.
        _identity_check_transformations_points(sdata.points[points_layer], to_coordinate_system=to_coordinate_system)

        se_crop = None
        if crd is not None:
            se_crop = bounding_box_query(
                se,
                axes=["x", "y"],
                min_coordinate=[crd[0], crd[2]],
                max_coordinate=[crd[1], crd[3]],
                target_coordinate_system=to_coordinate_system,
            )
            if se_crop is not None:
                se = se_crop
            else:
                log.warning(
                    f"Cropped spatial element using crd '{crd}' is None. Falling back to processing on full dataset."
                )

        # add trivial channel dimension
        se = se.expand_dims("c")

        x, kwargs = self._precondition(se, kwargs)

        _crd_points = None
        if se_crop is not None:
            # rechunk to ensure we do not have irregular chunksize after taking a crop
            x = x.rechunk(x.chunksize)
            tx, ty = _get_translation(se_crop, to_coordinate_system=to_coordinate_system)
            # define crd to original coordinates
            _crd_points = [tx, tx + se_crop.sizes["x"], ty, ty + se_crop.sizes["y"]]
        else:
            _crd_points = None

        # handle taking crops
        if _crd_points is not None:
            # need to account for fact that there can be a translation defined on the labels layer
            # query the dask dataframe. We use this query, because spatialdata query pulls query in memory.
            _ddf = sdata.points[points_layer].query(
                f"{_crd_points[0]} <= {name_x} < {_crd_points[1]} and {_crd_points[2]} <= {name_y} < {_crd_points[3]}"
            )
            coordinates = {name_x: name_x, name_y: name_y}

            # we write to points layer,
            # otherwise we would need to do this query again for every chunk we process later on
            _crd_points_layer = f"{points_layer}_{'_'.join(str(int(item)) for item in _crd_points)}"

            sdata = add_points_layer(
                sdata,
                ddf=_ddf,
                output_layer=_crd_points_layer,
                coordinates=coordinates,
                overwrite=True,
            )

            self._ddf = sdata.points[_crd_points_layer]

        else:
            # or do sdata.points[ points_layer ].to_delayed() # and then apply everything on this.
            self._ddf = sdata.points[points_layer]

        # need this original crd for when we do the query
        self._crd_points = _crd_points

        if sdata.is_backed():
            _temp_path = UPath(sdata.path).parent / f"tmp_{uuid.uuid4()}"
        else:
            _temp_path = None

        x_labels = self._segment(
            x,
            temp_path=_temp_path,
            c_dim_output_labels=c_dim_output_labels,
            fn_kwargs=fn_kwargs,
            **deepcopy(kwargs),
        )

        sdata = self._add_to_sdata(
            sdata,
            x_labels,
            output_labels_layer=output_labels_layer,
            output_shapes_layer=output_shapes_layer,
            labels_layer_align=labels_layer_align,
            transformations=get_transformation(se, get_all=True),
            scale_factors=scale_factors,
            overwrite=overwrite,
            **kwargs,
        )

        if _temp_path is not None:
            # TODO this will not work if sdata is remote (e.g. s3 bucket).
            shutil.rmtree(_temp_path)

        return sdata

    def _custom_segment_chunk(
        self,
        block: NDArray[Shape[Any, Any, Any, Any]],
        block_id: tuple[int, ...],
        _output_chunks: tuple[tuple[int, ...], ...],
        _depth: dict[int, int],
        fn_kwargs: Mapping[str, Any] = MappingProxyType({}),
        **kwargs,
    ) -> NDArray[Shape[Any, Any, Any, Any]]:
        name_x = fn_kwargs.setdefault("name_x", "x")
        name_y = fn_kwargs.setdefault("name_y", "y")
        _ = fn_kwargs.setdefault("name_gene", _GENES_KEY)

        # first calculate original chunks position (i.e. without the overlap)
        original_chunks = _substract_depth_from_chunks_size(_output_chunks, _depth)

        # find position in larger array
        y_start, y_stop, x_start, x_stop = _get_block_position(original_chunks, block_id)

        # shape size of original image
        shape_size = (
            sum(original_chunks[0]),
            sum(original_chunks[1]),
            sum(original_chunks[2]),
            sum(original_chunks[3]),
        )

        if self._crd_points is None:
            _crd_points = [0, shape_size[2], 0, shape_size[1]]
        else:
            _crd_points = self._crd_points

        x_start = x_start + _crd_points[0]
        x_stop = x_stop + _crd_points[0]
        y_start = y_start + _crd_points[2]
        y_stop = y_stop + _crd_points[2]

        if y_start != _crd_points[2]:
            y_start = y_start - _depth[1]
            assert y_start >= _crd_points[2], "Provided query not inside labels region."
        if y_stop != shape_size[1] + _crd_points[2]:
            y_stop = y_stop + _depth[1]
            assert y_stop <= shape_size[1] + _crd_points[2], "Provided query not inside labels region."
        if x_start != _crd_points[0]:
            x_start = x_start - _depth[2]
            assert x_start >= _crd_points[0], "Provided query not inside labels region."
        if x_stop != shape_size[2] + _crd_points[0]:
            x_stop = x_stop + _depth[2]
            assert x_stop <= shape_size[2] + _crd_points[0], "Provided query not inside labels region."

        # query the dask dataframe
        _ddf = self._ddf.query(f"{x_start} <= {name_x} < {x_stop} and {y_start} <= {name_y} < {y_stop}")

        df = _ddf.compute()
        # account for the fact that we do a reflect at the boundaries,
        # i.e. the labels layer does a reflect,
        # so we need to set relative position of points layer to labels layer correct.
        # therefore add depth if y_start or x_start is equal to 0/touches crd boundary.
        df[name_y] -= y_start
        df[name_x] -= x_start
        if y_start == _crd_points[2]:
            df[name_y] += _depth[1]
        if x_start == _crd_points[0]:
            df[name_x] += _depth[2]

        labels = self._model(block, df, **fn_kwargs).astype(_SEG_DTYPE)
        # for debug
        # labels = block.astype(_SEG_DTYPE)

        return labels
