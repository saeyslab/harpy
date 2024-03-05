from collections import defaultdict
from types import MappingProxyType
from typing import Any, Mapping, Optional

import dask
import numpy as np
import pandas as pd
import scanpy as sc
import spatialdata
from dask.array import Array
from spatialdata import SpatialData

from sparrow.image._image import _get_spatial_element
from sparrow.shape._shape import _filter_shapes_layer
from sparrow.table._table import ProcessTable, _back_sdata_table_to_zarr
from sparrow.utils._keys import _CELL_INDEX, _CELLSIZE_KEY, _INSTANCE_KEY, _REGION_KEY
from sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def preprocess_transcriptomics(
    sdata: SpatialData,
    labels_layer: str,
    min_counts: int = 10,
    min_cells: int = 5,
    size_norm: bool = True,
    max_value_scale: int = 10,
    n_comps: int = 50,
) -> SpatialData:
    """
    Preprocess the table (AnnData) attribute of a SpatialData object for transcriptomics data.

    Performs filtering (via `scanpy.pp.filter_cells` and `scanpy.pp.filter_genes` ) and optional normalization (on size or via `scanpy.sc.pp.normalize_total`), log transformation, scaling, and PCA calculation for transcriptomics data
    contained in the `sdata`. qc metrics are added to `sdata.table.obs` using `scanpy.pp.calculate_qc_metrics`.

    Parameters
    ----------
    sdata : SpatialData
        The input SpatialData object.
    labels_layer : str
        The labels layer of `sdata` used to select the cells via the _REGION_KEY.
        Note that cells in `sdata.table` linked to other labels_layer (via the _REGION_KEY), will be removed from `sdata.table`.
    min_counts : int, default=10
        Minimum number of genes a cell should contain to be kept (passed to `scanpy.pp.filter_cells`).
    min_cells : int, default=5
        Minimum number of cells a gene should be in to be kept (passed to `scanpy.pp.filter_genes`).
    size_norm : bool, default=True
        If True, normalization is based on the size of the nucleus/cell. If False, `scanpy.sc.pp.normalize_total` is used for normalization.
    max_value_scale : float, default=10
        The maximum value to which data will be scaled.
    n_comps : int, default=50
        Number of principal components to calculate.

    Returns
    -------
    SpatialData
        The preprocessed `sdata` containing the preprocessed AnnData object as an attribute (`sdata.table`).

    Raises
    ------
    ValueError
        - If `sdata` does not have labels attribute.
        - If `sdata` does not have table attribute.
        - If `labels_layer` is not a labels layer in `sdata`.

    Warnings
    --------
    - If `max_value_scale` is set too low, it may overly constrain the variability of the data,
      potentially impacting downstream analyses.
    - If the dimensionality of `sdata.table` is smaller than the desired number of principal components, `n_comps` is set to the minimum dimensionality, and a message is printed.
    """
    preprocess_instance = Preprocess(sdata, labels_layer=labels_layer)
    sdata = preprocess_instance.preprocess(
        filter_cells=True,
        filter_genes=True,
        calculate_cell_size=True,
        size_norm=size_norm,
        log1p=True,
        scale=True,
        max_value_scale=max_value_scale,
        calculate_pca=True,
        update_shapes_layers=True,
        qc_kwargs={"percent_top": [2, 5]},
        filter_cells_kwargs={"min_counts": min_counts},
        filter_genes_kwargs={"min_cells": min_cells},
        pca_kwargs={"n_comps": n_comps},
    )
    return sdata


def preprocess_proteomics(
    sdata: SpatialData,
    labels_layer: str,
    size_norm: bool = True,
    log1p: bool = True,
    scale: bool = False,
    max_value_scale: float = 10,
    calculate_pca: bool = False,
    n_comps: int = 50,
) -> SpatialData:
    """
    Preprocess the table (AnnData) attribute of a SpatialData object for proteomics data.

    Performs optional normalization (on size or via `scanpy.sc.pp.normalize_total`), log transformation, scaling, and PCA calculation for proteomics data
    contained in the `sdata`.
    qc metrics are added to `sdata.table.obs` using `scanpy.pp.calculate_qc_metrics`.

    Parameters
    ----------
    sdata : SpatialData
        The input SpatialData object.
    labels_layer : str
        The labels layer of `sdata` used to select the cells via the _REGION_KEY.
        Note that cells in `sdata.table` linked to other labels_layer (via the _REGION_KEY), will be removed from `sdata.table`.
    size_norm : bool, default=True
        If True, normalization is based on the size of the nucleus/cell. If False, `scanpy.sc.pp.normalize_total` is used for normalization.
    log1p : bool, default=True
        If True, applies log1p transformation to the data.
    scale : bool, default=False
        If True, scales the data to have zero mean and a variance of one. The scaling is capped at `max_value_scale`.
    max_value_scale : float, default=10
        The maximum value to which data will be scaled. Ignored if `scale` is False.
    calculate_pca : bool, default=False
        If True, calculates principal component analysis (PCA) on the data.
    n_comps : int, default=50
        Number of principal components to calculate. Ignored if `calculate_pca` is False.

    Returns
    -------
    SpatialData
        The preprocessed `sdata` containing the preprocessed AnnData object as an attribute (`sdata.table`).

    Raises
    ------
    ValueError
        - If `sdata` does not have labels attribute.
        - If `sdata` does not have table attribute.
        - If `labels_layer` is not a labels layer in `sdata`.

    Warnings
    --------
    - If `scale` is True and `max_value_scale` is set too low, it may overly constrain the variability of the data,
      potentially impacting downstream analyses.
    - If the dimensionality of `sdata.table` is smaller than the desired number of principal components
      when `calculate_pca` is True, `n_comps` is set to the minimum dimensionality, and a message is printed.
    """
    preprocess_instance = Preprocess(sdata, labels_layer=labels_layer)
    sdata = preprocess_instance.preprocess(
        filter_cells=False,
        filter_genes=False,
        calculate_cell_size=True,
        size_norm=size_norm,
        log1p=log1p,
        scale=scale,
        max_value_scale=max_value_scale,
        calculate_pca=calculate_pca,
        update_shapes_layers=False,
        qc_kwargs={"percent_top": [2, 5]},
        pca_kwargs={"n_comps": n_comps},
    )
    return sdata


class Preprocess(ProcessTable):
    def preprocess(
        self,
        filter_cells: bool = True,
        filter_genes: bool = True,
        calculate_cell_size: bool = True,
        size_norm: bool = True,
        log1p: bool = True,
        scale: bool = True,
        max_value_scale: Optional[float] = 10,  # ignored if scale is False,
        calculate_pca: bool = True,
        update_shapes_layers: bool = True,  # whether to update the shapes layer based on the items filtered out in sdata.table.
        qc_kwargs: Mapping[str, Any] = MappingProxyType({}),  # keyword arguments passed to sc.pp.calculate_qc_metrics
        filter_cells_kwargs: Mapping[str, Any] = MappingProxyType({}),  # keyword arguments passed to sc.pp.filter_cells
        filter_genes_kwargs: Mapping[str, Any] = MappingProxyType({}),  # keyword arguments passed to sc.pp.filter_genes
        norm_kwargs: Mapping[str, Any] = MappingProxyType(
            {}
        ),  # keyword arguments passed to sc.pp.normalize_total, ignored if size_norm is True.
        pca_kwargs: Mapping[str, Any] = MappingProxyType({}),  # keyword arguments passed to sc.tl.pca
    ) -> SpatialData:
        adata = self._get_adata()
        # Calculate QC Metrics
        sc.pp.calculate_qc_metrics(adata, inplace=True, **qc_kwargs)

        # Filter cells and genes
        if filter_cells:
            sc.pp.filter_cells(adata, **filter_cells_kwargs)
        if filter_genes:
            sc.pp.filter_genes(adata, **filter_genes_kwargs)

        if calculate_cell_size:
            log.info(f"Calculating cell size from provided labels_layer '{self.labels_layer}'")
            se = _get_spatial_element(self.sdata, layer=self.labels_layer)
            shapesize = _get_mask_area(se.data)
            # we do not want to loose the index (_CELL_INDEX)
            if _CELLSIZE_KEY in adata.obs.columns:
                log.warning(f"Column with name '{_CELLSIZE_KEY}' already exists. Removing column '{_CELLSIZE_KEY}'.")
                adata.obs = adata.obs.drop(columns=_CELLSIZE_KEY)
            old_index = adata.obs.index
            adata.obs = pd.merge(adata.obs.reset_index(), shapesize, on=_INSTANCE_KEY, how="left")
            adata.obs.index = old_index
            adata.obs = adata.obs.drop(columns=[_CELL_INDEX])

        adata.layers["raw_counts"] = adata.X

        if size_norm:
            adata.X = (adata.X.T * 100 / adata.obs[_CELLSIZE_KEY].values).T
        else:
            sc.pp.normalize_total(adata, **norm_kwargs)

        if log1p:
            sc.pp.log1p(adata)

        if scale:
            # need to do .copy() here to set .raw value, because .scale still overwrites this .raw, which is unexpected behaviour
            adata.raw = adata.copy()
            sc.pp.scale(adata, max_value=max_value_scale)

        if calculate_pca:
            # calculate the max amount of pc's possible
            n_comps = pca_kwargs.pop("n_comps", None)
            if n_comps is not None:
                if min(adata.shape) < n_comps:
                    n_comps = min(adata.shape) - 1
                    log.warning(
                        f"amount of pc's was set to {min( adata.shape)-1} because of the dimensionality of 'sdata.table'."
                    )
            if not scale:
                log.warning("Please consider scaling the data by passing scale=True, before calculating pca.")
            sc.tl.pca(adata, n_comps=n_comps, **pca_kwargs)

        # Update the SpatialData object
        if self.sdata.table:
            del self.sdata.table
        self.sdata.table = spatialdata.models.TableModel.parse(adata)

        indexes_to_keep = self.sdata.table.obs[_INSTANCE_KEY].values.astype(int)

        if update_shapes_layers:
            self.sdata = _filter_shapes_layer(
                self.sdata,
                indexes_to_keep=indexes_to_keep,
                prefix_filtered_shapes_layer="filtered_low_counts",
            )

        return self.sdata


def preprocess(
    sdata: SpatialData,
    labels_layer: str,
    shapes_layer: str = None,
    min_counts: int = 10,
    min_cells: int = 5,
    size_norm: bool = True,
    n_comps: int = 50,
) -> SpatialData:
    """
    Preprocess the table (AnnData) attribute of a SpatialData object.

    Calculates nucleus/cell size from either shapes_layer or labels_layer, and adds it
    to sdata.table.obs as column _CELLSIZE_KEY.
    Filters cells and genes, normalizes based on nucleus/cell size, calculates QC metrics and principal components.

    Parameters
    ----------
    sdata : SpatialData
        The input SpatialData object.
    labels_layer : str
        The labels layer of `sdata` that will be used for preprocessing.
    shapes_layer : str, optional
        The shapes layer of `sdata` that will be used to calculate cell size for normalization.
        If None, labels_layer will be used for calculation of cell size.
    min_counts : int, default=10
        Minimum number of genes a cell should contain to be kept.
    min_cells : int, default=5
        Minimum number of cells a gene should be in to be kept.
    size_norm : bool, default=True
        If True, normalization is based on the size of the nucleus/cell. Else the normalize_total function of scanpy is used.
    n_comps : int, default=50
        Number of principal components to calculate.

    Returns
    -------
    SpatialData
        The preprocessed `sdata` containg the preprocessed AnnData object as an attribute (sdata.table).

    Raises
    ------
    ValueError
        - If `labels_layer` is not a labels layer in `sdata`.
        - If `shapes_layer` contains 3D polygons.

    Notes
    -----
    - All cells with less than `min_counts` genes and all genes with less than `min_cells` cells are removed.
    - The QC metrics are calculated using scanpy's `calculate_qc_metrics` function.

    Warnings
    --------
    - If the dimensionality of the table attribute is smaller than the desired number of principal components,
      `n_comps` is set to the minimum dimensionality and a message is printed.
    """
    if labels_layer not in [*sdata.labels]:
        raise ValueError(
            f"Provided labels layer '{labels_layer}' not in 'sdata', please specify a labels layer from '{[*sdata.labels]}'"
        )

    adata = sdata.table[sdata.table.obs[_REGION_KEY] == labels_layer].copy()
    adata.uns["spatialdata_attrs"]["region"] = [labels_layer]

    del sdata.table
    sdata.table = spatialdata.models.TableModel.parse(adata)

    # Calculate QC Metrics
    sc.pp.calculate_qc_metrics(sdata.table, inplace=True, percent_top=[2, 5])

    # Filter cells and genes
    sc.pp.filter_cells(sdata.table, min_counts=min_counts)
    sc.pp.filter_genes(sdata.table, min_cells=min_cells)

    if shapes_layer is not None:
        has_z = sdata.shapes[shapes_layer]["geometry"].apply(lambda geom: geom.has_z)
        if any(has_z):
            raise ValueError(
                f"The shapes layer {shapes_layer} contains 3D polygons for calculation of nucleus/cell size. "
                "At present, support for computing the size of nuclei or cells in 3D is confined to a labels layer. "
                "It is advisable to designate a labels layer for this purpose."
            )

        sdata.table.obs[_CELLSIZE_KEY] = sdata[shapes_layer].area

    else:
        se = _get_spatial_element(sdata, layer=labels_layer)
        shapesize = _get_mask_area(se.data)

        old_index = sdata.table.obs.index
        sdata.table.obs = pd.merge(sdata.table.obs.reset_index(), shapesize, on=_INSTANCE_KEY, how="left")
        sdata.table.obs.index = old_index
        sdata.table.obs = sdata.table.obs.drop(columns=[_CELL_INDEX])

    sdata.table.layers["raw_counts"] = sdata.table.X

    if size_norm:
        sdata.table.X = (sdata.table.X.T * 100 / sdata.table.obs[_CELLSIZE_KEY].values).T
        sc.pp.log1p(sdata.table)
        # need to do .copy() here to set .raw value, because .scale still overwrites this .raw, which is unexpected behaviour
        sdata.table.raw = sdata.table.copy()
        sc.pp.scale(sdata.table, max_value=10)

    else:
        sc.pp.normalize_total(sdata.table)
        sc.pp.log1p(sdata.table)
        sdata.table.raw = sdata.table.copy()
        sc.pp.scale(sdata.table, max_value=10)

    # calculate the max amount of pc's possible
    if min(sdata.table.shape) < n_comps:
        n_comps = min(sdata.table.shape) - 1
        log.warning(
            f"amount of pc's was set to {min( sdata.table.shape)-1} because of the dimensionality of the AnnData object."
        )
    sc.tl.pca(sdata.table, svd_solver="arpack", n_comps=n_comps)

    # Is this the best way of doing it? Every time you subset your data, the polygons should be subsetted too!
    indexes_to_keep = sdata.table.obs[_INSTANCE_KEY].values.astype(int)
    sdata = _filter_shapes_layer(
        sdata,
        indexes_to_keep=indexes_to_keep,
        prefix_filtered_shapes_layer="filtered_low_counts",
    )

    # need to update sdata.table via .parse, otherwise it will not be backed by zarr store
    _back_sdata_table_to_zarr(sdata)

    return sdata


def _get_mask_area(mask: Array) -> pd.DataFrame:
    """Calculate area of each label in mask. Return as pd.Series."""

    @dask.delayed
    def calculate_area(mask_chunk: np.ndarray) -> tuple:
        unique, counts = np.unique(mask_chunk, return_counts=True)

        return unique, counts

    delayed_results = [calculate_area(chunk) for chunk in mask.to_delayed().flatten()]

    results = dask.compute(*delayed_results, scheduler="threads")

    combined_counts = defaultdict(int)

    # aggregate
    for unique, counts in results:
        for label, count in zip(unique, counts):
            if label > 0:
                combined_counts[int(label)] += count

    combined_counts = pd.Series(combined_counts)
    combined_counts.index.name = _INSTANCE_KEY

    combined_counts.name = _CELLSIZE_KEY
    combined_counts = combined_counts.to_frame().reset_index()

    return combined_counts
