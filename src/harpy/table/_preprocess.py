from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from spatialdata import SpatialData

from harpy.image._image import _get_spatial_element
from harpy.shape._shape import filter_shapes_layer
from harpy.table._table import ProcessTable, add_table_layer
from harpy.utils._aggregate import _get_mask_area
from harpy.utils._keys import _CELLSIZE_KEY, _INSTANCE_KEY, _RAW_COUNTS_KEY, _REGION_KEY
from harpy.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def preprocess_transcriptomics(
    sdata: SpatialData,
    labels_layer: str | Iterable[str],
    table_layer: str,
    output_layer: str,
    percent_top: tuple[int, ...] = (2, 5),
    min_counts: int = 10,
    min_cells: int = 5,
    size_norm: bool = True,
    highly_variable_genes: bool = False,
    highly_variable_genes_kwargs: Mapping[str, Any] = MappingProxyType({}),
    max_value_scale: float | None = 10,
    n_comps: int = 50,
    update_shapes_layers: bool = True,
    overwrite: bool = False,
) -> SpatialData:
    """
    Preprocess a table (AnnData) attribute of a SpatialData object for transcriptomics data.

    Performs filtering (via `scanpy.pp.filter_cells` and `scanpy.pp.filter_genes` ) and optional normalization (on size or via `scanpy.sc.pp.normalize_total`),
    log transformation (`scanpy.pp.log1p`), highly variable genes selection (`scanpy.pp.highly_variable_genes`),
    scaling (`scanpy.pp.scale`), and PCA calculation (`scanpy.tl.pca`) for transcriptomics data
    contained in the `sdata.tables[table_layer]`. QC metrics are added to `sdata.tables[output_layer].obs` using `scanpy.pp.calculate_qc_metrics`.

    Parameters
    ----------
    sdata
        The input SpatialData object.
    labels_layer
        The labels layer(s) of `sdata` used to select the cells via the `_REGION_KEY` in `sdata.tables[table_layer].obs`.
        Note that if `output_layer` is equal to `table_layer` and overwrite is `True`,
        cells in `sdata.tables[table_layer]` linked to other `labels_layer` (via the `_REGION_KEY`), will be removed from `sdata.tables[table_layer]`
        (also from the backing zarr store if it is backed).
    table_layer
        The table layer in `sdata` on which to perform preprocessing on.
    output_layer
        The output table layer in `sdata` to which preprocessed table layer will be written.
    percent_top
        List of ranks (where genes are ranked by expression) at which the cumulative proportion of expression will be reported as a percentage.
        Passed to `scanpy.pp.calculate_qc_metrics`.
    min_counts
        Minimum number of genes a cell should contain to be kept (passed to `scanpy.pp.filter_cells`).
    min_cells
        Minimum number of cells a gene should be in to be kept (passed to `scanpy.pp.filter_genes`).
    size_norm
        If `True`, normalization is based on the size of the nucleus/cell. If `False`, `scanpy.sc.pp.normalize_total` is used for normalization.
    highly_variable_genes
        If `True`, will only retain highly variable genes, as calculated by `scanpy.pp.highly_variable_genes`.
    highly_variable_genes_kwargs
        Keyword arguments passed to `scanpy.pp.highly_variable_genes`. Ignored if `highly_variable_genes` is `False`.
    max_value_scale
        The maximum value to which data will be scaled, using `scanpy.pp.scale`.
    n_comps
        Number of principal components to calculate.
    update_shapes_layers
        Whether to filter the shapes layers associated with `labels_layer`.
        If set to `True`, cells that do not appear in resulting `output_layer` (with `_REGION_KEY` equal to `labels_layer`) will be removed from the shapes layers (via `_INSTANCE_KEY`) in the `sdata` object.
        Filtered shapes will be added to `sdata` with prefix 'filtered_low_counts'.
    overwrite
        If True, overwrites the `output_layer` if it already exists in `sdata`.

    Returns
    -------
    The `sdata` containing the preprocessed AnnData object as an attribute (`sdata.tables[output_layer]`).

    Raises
    ------
    ValueError
        If `sdata` does not have labels attribute.
    ValueError
        If `sdata` does not have tables attribute.
    ValueError
        If `labels_layer`, or one of the element of `labels_layer` is not a labels layer in `sdata`.
    ValueError
        If `table_layer` is not a table layer in `sdata`.


    Warnings
    --------
    - If `max_value_scale` is set too low, it may overly constrain the variability of the data,
      potentially impacting downstream analyses.
    - If the dimensionality of `sdata.tables[table_layer]` is smaller than the desired number of principal components, `n_comps` is set to the minimum dimensionality, and a message is printed.

    See Also
    --------
    harpy.tb.allocate : create an AnnData table in `sdata` using a `points_layer` and a `labels_layer`.
    """
    preprocess_instance = Preprocess(sdata, labels_layer=labels_layer, table_layer=table_layer)
    sdata = preprocess_instance.preprocess(
        output_layer=output_layer,
        calculate_qc_metrics=True,
        filter_cells=True,
        filter_genes=True,
        calculate_cell_size=True,
        size_norm=size_norm,
        log1p=True,
        highly_variable_genes=highly_variable_genes,
        highly_variable_genes_kwargs=highly_variable_genes_kwargs,
        scale=True,
        max_value_scale=max_value_scale,
        calculate_pca=True,
        update_shapes_layers=update_shapes_layers,
        qc_kwargs={"percent_top": percent_top},
        filter_cells_kwargs={"min_counts": min_counts},
        filter_genes_kwargs={"min_cells": min_cells},
        pca_kwargs={"n_comps": n_comps},
        overwrite=overwrite,
    )
    return sdata


def preprocess_proteomics(
    sdata: SpatialData,
    labels_layer: str | Iterable[str],
    table_layer: str,
    output_layer: str,
    size_norm: bool = True,
    log1p: bool = True,
    scale: bool = False,
    max_value_scale: float | None = 10,
    q: float | None = None,
    calculate_pca: bool = False,
    n_comps: int = 50,
    overwrite: bool = False,
) -> SpatialData:
    """
    Preprocess a table (AnnData) attribute of a SpatialData object for proteomics data.

    Performs optional normalization (on size or via `scanpy.sc.pp.normalize_total`), log transformation
    (`scanpy.pp.log1p`), scaling (`scanpy.pp.scale`)/ quantile normalization and PCA calculation (`scanpy.tl.pca`)
    for proteomics data contained in `sdata`.

    Parameters
    ----------
    sdata
        The input SpatialData object.
    labels_layer
        The labels layer(s) of `sdata` used to select the cells via the _REGION_KEY  in `sdata.tables[table_layer].obs`.
        Note that if `output_layer` is equal to `table_layer` and overwrite is True,
        cells in `sdata.tables[table_layer]` linked to other `labels_layer` (via the _REGION_KEY), will be removed from `sdata.tables[table_layer]`.
        If a list of labels layers is provided, they will therefore be preprocessed together (e.g. multiple samples).
    table_layer
        The table layer in sdata to apply preprocessing to.
        It is an AnnData object containing total intensities per cell in `.obs` (rows) and per channel in `.var` (columns).
    output_layer
        The output table layer in `sdata` to which preprocessed table layer will be written.
    size_norm
        If `True`, normalization is based on the size of the nucleus/cell. If False, `scanpy.sc.pp.normalize_total` is used for normalization.
    log1p
        If `True`, applies log1p transformation to the data.
    scale
        If `True`, scales the data to have zero mean and a variance of one. The scaling is capped at `max_value_scale`.
    max_value_scale
        The maximum value to which data will be scaled. Ignored if `scale` is `False`.
    q
        Quantile used for normalization. If specified, values are normalized by this quantile calculated for each `adata.var`. Values are multiplied by 100 after normalization. Typical value used is 0.999,
    calculate_pca
        If `True`, calculates principal component analysis (PCA) on the data.
    n_comps
        Number of principal components to calculate. Ignored if `calculate_pca` is False.
    overwrite
        If `True`, overwrites the `output_layer` if it already exists in `sdata`.

    Returns
    -------
    The `sdata` containing the preprocessed AnnData object as an attribute (`sdata.tables[output_layer]`).

    Raises
    ------
    ValueError
        - If `sdata` does not contains any labels layers.
        - If `sdata` does not contain any table layers.
        - If `labels_layer`, or one of the element of `labels_layer` is not a labels layer in `sdata`.
        - If `table_layer` is not a table layer in `sdata`.
        - If both `scale` is set to True and `q` is not None.

    Warnings
    --------
    - If `scale` is True and `max_value_scale` is set too low, it may overly constrain the variability of the data,
      potentially impacting downstream analyses.
    - If the dimensionality of `sdata.tables[table_layer]` is smaller than the desired number of principal components
      when `calculate_pca` is True, `n_comps` is set to the minimum dimensionality, and a message is printed.

    See Also
    --------
    harpy.tb.allocate_intensity : create an AnnData table in `sdata` using an `image_layer` and a `labels_layer`.
    """
    preprocess_instance = Preprocess(sdata, labels_layer=labels_layer, table_layer=table_layer)
    sdata = preprocess_instance.preprocess(
        output_layer=output_layer,
        calculate_qc_metrics=False,
        filter_cells=False,
        filter_genes=False,
        calculate_cell_size=True,
        size_norm=size_norm,
        log1p=log1p,
        scale=scale,
        q=q,
        max_value_scale=max_value_scale,
        highly_variable_genes=False,
        calculate_pca=calculate_pca,
        update_shapes_layers=False,
        pca_kwargs={"n_comps": n_comps},
        overwrite=overwrite,
    )
    return sdata


class Preprocess(ProcessTable):
    def preprocess(
        self,
        output_layer: str,
        calculate_qc_metrics: bool = True,
        filter_cells: bool = True,
        filter_genes: bool = True,
        calculate_cell_size: bool = True,
        size_norm: bool = True,
        log1p: bool = True,
        scale: bool = True,
        max_value_scale: float | None = 10,  # ignored if scale is False,
        q: float | None = None,  # quantile for normalization, typically 0.999
        highly_variable_genes: bool = False,
        calculate_pca: bool = True,
        update_shapes_layers: bool = True,  # whether to update the shapes layer based on the items filtered out in sdata.tables[self.table_layer].
        qc_kwargs: Mapping[str, Any] = MappingProxyType({}),  # keyword arguments passed to sc.pp.calculate_qc_metrics
        filter_cells_kwargs: Mapping[str, Any] = MappingProxyType({}),  # keyword arguments passed to sc.pp.filter_cells
        filter_genes_kwargs: Mapping[str, Any] = MappingProxyType({}),  # keyword arguments passed to sc.pp.filter_genes
        norm_kwargs: Mapping[str, Any] = MappingProxyType(
            {}
        ),  # keyword arguments passed to sc.pp.normalize_total, ignored if size_norm is True.
        highly_variable_genes_kwargs: Mapping[str, Any] = MappingProxyType(
            {}
        ),  # keyword arguments passed to sc.pp.highly_variable_genes
        pca_kwargs: Mapping[str, Any] = MappingProxyType({}),  # keyword arguments passed to sc.tl.pca
        overwrite: bool = False,
    ) -> SpatialData:
        adata = self._get_adata()
        # Calculate QC Metrics
        if calculate_qc_metrics:
            sc.pp.calculate_qc_metrics(adata, layer=None, inplace=True, **qc_kwargs)

            # Filter cells and genes
            if filter_cells:
                sc.pp.filter_cells(adata, inplace=True, copy=False, **filter_cells_kwargs)
            if filter_genes:
                sc.pp.filter_genes(adata, inplace=True, copy=False, **filter_genes_kwargs)

        if calculate_cell_size:
            # we do not want to loose the index (_CELL_INDEX)
            old_index = adata.obs.index
            index_name = adata.obs.index.name or "index"
            if _CELLSIZE_KEY in adata.obs.columns:
                log.warning(f"Column with name '{_CELLSIZE_KEY}' already exists. Removing column '{_CELLSIZE_KEY}'.")
                adata.obs = adata.obs.drop(columns=_CELLSIZE_KEY)
            for i, _labels_layer in enumerate(self.labels_layer):
                log.info(f"Calculating cell size from provided labels_layer '{_labels_layer}'")
                se = _get_spatial_element(self.sdata, layer=_labels_layer)
                _shapesize = _get_mask_area(se.data if se.data.ndim == 3 else se.data[None, ...])
                _shapesize[_REGION_KEY] = _labels_layer
                if i == 0:
                    shapesize = _shapesize
                else:
                    shapesize = pd.concat([shapesize, _shapesize], ignore_index=True)
            # note that we checked that adata.obs[ _INSTANCE_KEY ] is unique for given region (see self._get_adata())
            adata.obs = pd.merge(adata.obs.reset_index(), shapesize, on=[_INSTANCE_KEY, _REGION_KEY], how="left")
            adata.obs.index = old_index
            adata.obs = adata.obs.drop(columns=[index_name])

        adata.layers[_RAW_COUNTS_KEY] = adata.X.copy()

        if size_norm:
            adata.X = (adata.X.T * 100 / adata.obs[_CELLSIZE_KEY].values).T
            if issparse(adata.X):
                adata.X = adata.X.tocsr()
        else:
            sc.pp.normalize_total(adata, layer=None, layers=None, copy=False, inplace=True, **norm_kwargs)

        if log1p:
            sc.pp.log1p(adata, base=None, copy=False, layer=None, obsm=None)

        if highly_variable_genes:
            sc.pp.highly_variable_genes(adata, layer=None, inplace=True, subset=False, **highly_variable_genes_kwargs)
            adata.raw = adata.copy()
            adata = adata[:, adata.var.highly_variable]

        if scale and q is not None:
            raise ValueError(
                "Please choose between scaling via 'harpy.pp.scale' or normalization by q quantile, not both."
            )

        if scale:
            if adata.raw is None:
                adata.raw = adata.copy()
            sc.pp.scale(adata, copy=False, layer=None, obsm=None, zero_center=True, max_value=max_value_scale)

        if q is not None:
            if adata.raw is None:
                adata.raw = adata.copy()
            arr = adata.X.toarray() if issparse(adata.X) else adata.X  # np.where not defined on sparse matrix
            arr = np.where(arr == 0, np.nan, arr)
            arr_quantile = np.nanquantile(arr, q, axis=0)
            adata.X = (adata.X.T * 100 / arr_quantile.reshape(-1, 1)).T
            if issparse(adata.X):
                adata.X = adata.X.tocsr()

        if calculate_pca:
            # calculate the max amount of pc's possible
            n_comps = pca_kwargs.pop("n_comps", None)
            if n_comps is not None:
                if min(adata.shape) < n_comps:
                    n_comps = min(adata.shape) - 1
                    log.warning(
                        f"amount of pc's was set to {min(adata.shape) - 1} because of the dimensionality of 'sdata.tables[table_layer]'."
                    )
            if not scale:
                log.warning(
                    "Please consider scaling the data by passing 'scale=True', when passing 'calculate_pca=True'."
                )
            self._type_check_before_pca(adata)
            sc.pp.pca(adata, copy=False, n_comps=n_comps, **pca_kwargs)

        self.sdata = add_table_layer(
            self.sdata,
            adata=adata,
            output_layer=output_layer,
            region=self.labels_layer,
            overwrite=overwrite,
        )

        if update_shapes_layers:
            for _labels_layer in self.labels_layer:
                self.sdata = filter_shapes_layer(
                    self.sdata,
                    table_layer=output_layer,
                    labels_layer=_labels_layer,
                    prefix_filtered_shapes_layer="filtered_low_counts",
                )

        return self.sdata
