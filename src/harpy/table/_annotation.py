from __future__ import annotations

import os
from collections.abc import Mapping
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from spatialdata import SpatialData

from harpy.table._table import ProcessTable, add_table_layer
from harpy.utils._keys import _ANNOTATION_KEY, _CLEANLINESS_KEY, _UNKNOWN_CELLTYPE_KEY
from harpy.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def score_genes(
    sdata: SpatialData,
    labels_layer: list[str],
    table_layer: str,
    output_layer: str,
    path_marker_genes: str | Path | pd.DataFrame,
    delimiter=",",
    row_norm: bool = False,
    repl_columns: dict[str, str] | None = None,
    del_celltypes: dict[str] | None = None,
    input_dict: bool = False,
    celltype_column: str = _ANNOTATION_KEY,
    overwrite: bool = False,
    **kwargs: Any,
) -> tuple[SpatialData, list[str], list[str]]:
    """
    The function loads marker genes from a CSV file and scores cells for each cell type using those markers using scanpy's `sc.tl.score_genes` function.

    Function annotates cells to the celltype with the maximum score obtained through `sc.tl.score_genes`.
    Marker genes can be provided as a one-hot encoded matrix with cell types listed in the first row, and marker genes in the first column;
    or in dictionary format. The function further allows replacements of column names and deletions of specific marker genes.

    Parameters
    ----------
    sdata
        The SpatialData object.
    labels_layer
        The labels layer(s) of `sdata` used to select the cells via the _REGION_KEY in `sdata.tables[table_layer].obs`.
        Note that if `output_layer` is equal to `table_layer` and overwrite is True,
        cells in `sdata.tables[table_layer]` linked to other `labels_layer` (via the _REGION_KEY), will be removed from `sdata.tables[table_layer]`.
        If a list of labels layers is provided, they will therefore be scored together (e.g. multiple samples).
    table_layer
        The table layer in `sdata` on which to perform annotation on.
    output_layer
        The output table layer in `sdata` to which table layer with results of annotation will be written.
    path_marker_genes
        Path to the CSV file containing the marker genes or a pandas dataframe.
        It should be a one-hot encoded matrix with cell types listed in the first row, and marker genes in the first column.
    delimiter
        Delimiter used in the CSV file, default is ','.
    row_norm
        Flag to determine if row normalization is applied, default is False.
    repl_columns
        Dictionary containing cell types to be replaced. The keys are the original cell type names and
        the values are their replacements.
    del_celltypes
        List of cell types to be deleted from the list of possible cell type candidates.
        Cells are scored for these cell types, but will not be assigned a cell type from this list.
    input_dict
        If True, the marker gene list from the CSV file is treated as a dictionary with the first column being
        the cell type names and the subsequent columns being the marker genes for those cell types. Default is False.
    celltype_column
        The column name in the SpatialData object's table that specifies the cell type annotations.
        The default value is `_ANNOTATION_KEY`.
    overwrite
        If True, overwrites the `output_layer` if it already exists in `sdata`.
    **kwargs
        Additional keyword arguments passed to `scanpy.tl.score_genes`.

    Returns
    -------
    tuple:

        - Updated `sdata`.

        - list of strings, with all celltypes that are scored (but are not in the del_celltypes list).

        - list of strings, with all celltypes, some of which may not be scored, because their corresponding transcripts do not appear in the region of interest. _UNKNOWN_CELLTYPE_KEY, is also added if it is detected.

    Notes
    -----
    The cell type `_UNKNOWN_CELLTYPE_KEY` is reserved for cells that could not be assigned a specific cell type.

    """
    process_table_instance = ProcessTable(sdata, labels_layer=labels_layer, table_layer=table_layer)
    adata = process_table_instance._get_adata()
    # Load marker genes from csv
    if input_dict:
        log.warning(
            "'input_dict' is deprecated and will be removed in future versions. "
            "Please pass a pandas DataFrame or a path to a .csv file to 'path_marker_genes'. "
            "It should be a one-hot encoded matrix with cell types listed in the first row "
            "and marker genes in the first column."
        )
        df_markers = pd.read_csv(path_marker_genes, header=None, index_col=0, delimiter=delimiter)
        df_markers = df_markers.T
        genes_dict = df_markers.to_dict("list")
        for i in genes_dict:
            genes_dict[i] = [x for x in genes_dict[i] if str(x) != "nan"]
    # Replace column names in marker genes
    else:
        if isinstance(path_marker_genes, pd.DataFrame):
            df_markers = path_marker_genes
        elif isinstance(path_marker_genes, (str, Path)):
            df_markers = pd.read_csv(path_marker_genes, index_col=0, delimiter=delimiter)
        else:
            raise ValueError("Please pass either a path to a .csv file, or a pandas Dataframe to 'path_marker_genes'.")

        if repl_columns:
            for column, replace in repl_columns.items():
                df_markers.columns = df_markers.columns.str.replace(column, replace)

        # Create genes dict with all marker genes for every celltype
        genes_dict = {}
        for i in df_markers:
            genes = []
            for row, value in enumerate(df_markers[i]):
                if value > 0:
                    genes.append(df_markers.index[row])
            genes_dict[i] = genes

    assert _UNKNOWN_CELLTYPE_KEY not in genes_dict.keys(), (
        f"Cell type {_UNKNOWN_CELLTYPE_KEY} is reserved for cells that could not be assigned a specific cell type"
    )

    # sanity check
    unique_genes = {item for sublist in genes_dict.values() for item in sublist}
    if not set(adata.var.index).intersection(unique_genes):
        raise ValueError(
            f"No genes in provided marker genes file at '{path_marker_genes}' where found in .var of table layer '{table_layer}'."
        )

    # Score all cells for all celltypes
    for key, value in genes_dict.items():
        try:
            sc.tl.score_genes(adata, value, score_name=key, copy=False, **kwargs)
        except ValueError:
            log.warning(f"Markergenes '{value}' not present in region, celltype '{key}' not found.")

    # Delete genes from marker genes and genes dict
    if del_celltypes:
        for gene in del_celltypes:
            if gene in df_markers.columns:
                del df_markers[gene]
            if gene in genes_dict.keys():
                del genes_dict[gene]

    adata, celltypes_scored = _annotate_celltype(
        adata=adata,
        celltypes=df_markers.columns,
        row_norm=row_norm,
        celltype_column=celltype_column,
    )

    # add _UNKNOWN_CELLTYPE_KEY to the list of celltypes if it is detected.
    if _UNKNOWN_CELLTYPE_KEY in adata.obs[celltype_column].cat.categories:
        genes_dict[_UNKNOWN_CELLTYPE_KEY] = []

    celltypes_all = list(genes_dict.keys())

    sdata = add_table_layer(
        sdata,
        adata=adata,
        output_layer=output_layer,
        region=process_table_instance.labels_layer,
        overwrite=overwrite,
    )

    return sdata, celltypes_scored, celltypes_all


def score_genes_iter(
    sdata: SpatialData,
    labels_layer: list[str],
    table_layer: str,
    output_layer: str,
    path_marker_genes: str | Path | pd.DataFrame,
    delimiter: str = ",",
    min_score: Literal["Zero", "Quantile", None] = "Zero",
    min_score_p: float = 25,
    scaling: Literal[
        "MinMax",
        "ZeroMax",
        "Nmarkers",
        "Robust",
        "Rank",
    ] = "Nmarkers",
    scale_score_p: float = 1,
    n_iter: int = 5,
    calculate_umap=False,
    calculate_neighbors=False,
    neigbors_kwargs: Mapping[str, Any] = MappingProxyType({}),
    umap_kwargs: Mapping[str, Any] = MappingProxyType({}),
    output_dir=None,
    celltype_column: str = _ANNOTATION_KEY,
    overwrite: bool = False,
) -> tuple[SpatialData, list[str], list[str]]:
    """
    Iterative annotation algorithm.

    For each cell, a score is calculated for each cell type.

    In the 0-th iteration this is:

    First mean expression is substracted from expression levels.
    Score for each cell type is obtained via sum of these normalized expressions of the markers in the cell.

    And in following iterations:

    Expression levels are normalized by substracting the mean over all celltypes assigned in iteration i-1.
    Score for each cell type is obtained via sum of these normalized expressions of the markers in the cell.

    Function expects scaled data (obtained through e.g. `scanpy.pp.scale`).

    Parameters
    ----------
    sdata
        The SpatialData object.
    labels_layer
        The labels layer(s) of `sdata` used to select the cells via the _REGION_KEY in `sdata.tables[table_layer].obs`.
        Note that if `output_layer` is equal to `table_layer` and overwrite is True,
        cells in `sdata.tables[table_layer]` linked to other `labels_layer` (via the _REGION_KEY), will be removed from `sdata.tables[table_layer]`.
        If a list of labels layers is provided, they will therefore be scored together (e.g. multiple samples).
    table_layer
        The table layer in `sdata` on which to perform annotation on. We assume the data is already preprocessed by e.g. `harpy.tb.preprocess_transcriptomics`.
        Features should all have approximately same variance.
    output_layer
        The output table layer in `sdata` to which table layer with results of annotation will be written.
    path_marker_genes
        Path to the CSV file containing the marker genes or a pandas dataframe.
        It should be a one-hot encoded matrix with cell types listed in the first row, and marker genes in the first column.
    delimiter
        Delimiter used in the CSV file.
    min_score
        Min score method. Choose from one of these options: "Zero", "Quantile", None.
    min_score_p
        Min score percentile. Ignored if `min_score` is not set to "Quantile".
    scaling
        Scaling method. Choose from one of these options: "MinMax", "ZeroMax", "Nmarkers", "Robust", "Rank".
    scale_score_p
        Scale score percentile.
    n_iter
        Number of iterations.
    calculate_umap
        If `True`, calculates a UMAP via `scanpy.tl.umap` for visualization of obtained annotations per iteration.
        If `False` and 'umap' or 'X_umap' is not in .obsm, then no umap will be plotted.
    calculate_neighbors
        If `True`, calculates neighbors via `scanpy.pp.neighbors`. Ignored if `calculate_umap` is set to `False`.
    umap_kwargs
        Keyword arguments passed to `scanpy.tl.umap`. Ignored if `calculate_umap` is `False`.
    neigbors_kwargs
        Keyword arguments passed to `scanpy.pp.neighbors`. Ignored if `calculate_umap` is `False` or if `calculate_neighbors` is set to `False` and "neighbors" already in `.uns.keys()`.
    output_dir
        If specified, figures with umaps will be saved in this directory after each iteration. If None, the plots will be displayed directly without saving.
    celltype_column
        The column name in the SpatialData object's table that specifies the cell type annotations.
        The default value is `_ANNOTATION_KEY`.
    overwrite
        If True, overwrites the `output_layer` if it already exists in `sdata`.

    Returns
    -------
    tuple:

        - Updated `sdata`.

        - list of strings, with all celltypes that are scored (but are not in the del_celltypes list).

        - list of strings, with all celltypes, some of which may not be scored, because their corresponding transcripts do not appear in the region of interest. _UNKNOWN_CELLTYPE_KEY, is also added if it is detected.
    """
    kwargs = {}
    kwargs["min_score"] = min_score
    kwargs["min_score_p"] = min_score_p
    kwargs["scaling"] = scaling
    kwargs["scale_score_p"] = scale_score_p

    process_table_instance = ProcessTable(sdata, labels_layer=labels_layer, table_layer=table_layer)
    adata = process_table_instance._get_adata()

    adata, celltypes_scored, celltypes_all = _annotate_celltype_iter(
        adata=adata,
        path_marker_genes=path_marker_genes,
        delimiter=delimiter,
        n_iter=n_iter,
        calculate_umap=calculate_umap,
        calculate_neighbors=calculate_neighbors,
        neigbors_kwargs=neigbors_kwargs,
        umap_kwargs=umap_kwargs,
        output_dir=output_dir,
        celltype_column=celltype_column,
        **kwargs,  # keyword arguments passed to _annotate_celltype_weighted
    )

    sdata = add_table_layer(
        sdata,
        adata=adata,
        output_layer=output_layer,
        region=process_table_instance.labels_layer,
        overwrite=overwrite,
    )

    return sdata, celltypes_scored, celltypes_all


def _annotate_celltype_iter(
    adata: AnnData,
    path_marker_genes: str | Path | pd.DataFrame,
    delimiter: str = ",",
    n_iter=5,
    calculate_umap=False,
    calculate_neighbors=False,
    neigbors_kwargs: Mapping[str, Any] = MappingProxyType({}),  # keyword arguments passed to sc.pp.neighbors
    umap_kwargs: Mapping[str, Any] = MappingProxyType({}),  # keyword arguments passed to sc.tl.umap
    output_dir=None,
    celltype_column: str = _ANNOTATION_KEY,
    **kwargs,
) -> tuple[AnnData, pd.DataFrame]:
    # initial clustering: = typical own_score_genes
    # 'mean expression' is over all cells'
    # but you do MinMax scaling so 'mean expression' does not have an effect

    if celltype_column in adata.obs:
        log.warning(f"Column '{celltype_column}' already a column in '.obs'. This column will be overwritten.")

    if isinstance(path_marker_genes, (str, Path)):
        marker_genes = pd.read_csv(path_marker_genes, index_col=0, delimiter=delimiter)
    elif isinstance(path_marker_genes, pd.DataFrame):
        marker_genes = path_marker_genes
    else:
        raise ValueError("Please pass either a path to a .csv file, or a pandas Dataframe to 'path_marker_genes'.")

    # only retain marker genes that are in adata
    marker_genes = marker_genes[marker_genes.index.isin(adata.var_names)]

    celltypes_all = marker_genes.columns
    column_sums = marker_genes.sum(axis=0)
    columns_to_drop = column_sums[column_sums == 0].index.tolist()
    for _column in columns_to_drop:
        log.info(
            f"Celltype '{_column}' has no marker genes in given tissue, and will be removed from the marker genes file."
        )
    marker_genes = marker_genes.loc[:, column_sums != 0]

    # Remove celltype columns if they already exist in the DataFrame
    for column in marker_genes.columns:
        if column in adata.obs.columns:
            log.info(f"Column '{column}' already a column in '.obs'. Removing.")
            adata.obs.drop(column, axis=1, inplace=True)

    adata, _ = _annotate_celltype_weighted(
        adata,
        marker_genes=marker_genes,
        mean_values=None,
        celltype_column=celltype_column,
        **kwargs,
    )

    log.info((adata.obs[celltype_column].value_counts() / len(adata.obs[celltype_column])) * 100)

    if calculate_umap:
        if calculate_neighbors:
            if "neighbors" in adata.uns.keys():
                log.warning(
                    "'neighbors' already in 'adata.uns', recalculating neighbors. Consider passing 'calculate_neigbors=False'."
                )
            sc.pp.neighbors(adata, copy=False, **neigbors_kwargs)
        else:
            if "neighbors" not in adata.uns.keys():
                log.info("'neighbors not in 'adata.uns', computing neighborhood graph before calculating umap.")
                sc.pp.neighbors(adata, copy=False, **neigbors_kwargs)
            else:
                log.info("'neighbors already in 'adata.uns', reusing for calculating umap.")
        sc.tl.umap(adata, copy=False, **umap_kwargs)

    if not any(key in adata.obsm for key in ["umap", "X_umap"]):
        log.info(
            "Could not find 'umap' or 'X_umap' in .obsm. Will not plot umap. Set 'calculate_umap' to 'True' to enable visualization of the umap."
        )
    else:
        sc.pl.umap(adata, color=[celltype_column], show=not output_dir)
        if output_dir is not None:
            plt.savefig(os.path.join(output_dir, celltype_column), bbox_inches="tight")
            plt.close()
        del adata.uns[f"{celltype_column}_colors"]

    # iterative clustering:
    # own_score_genes again but now no (MinMax) scaling hence mean_expression has an effect
    # mean expression with fair contribution of each cell type (cell types are based on the previous clustering)
    # changes = []
    completed = 0
    for _iteration in range(n_iter):
        log.info(f"Iteration {_iteration}.")
        cell_types = np.unique(adata.obs[celltype_column]).tolist()
        if _UNKNOWN_CELLTYPE_KEY in cell_types:
            cell_types.remove(_UNKNOWN_CELLTYPE_KEY)
        mean_per_ct = []
        for ct in cell_types:
            l = pd.DataFrame(adata.obs[celltype_column] == ct)
            l = l.index[l[celltype_column]].tolist()
            ct_sel = adata[l, :]
            mean_per_ct.append(ct_sel.to_df().mean(axis=0))

        df = pd.concat(mean_per_ct, axis=1)
        next_mean = df.mean(axis=1)
        if f"{celltype_column}_previous" in adata.obs.columns:
            adata.obs.drop(columns=[f"{celltype_column}_previous"], inplace=True)

        adata.obs.rename(
            columns={celltype_column: f"{celltype_column}_previous"},
            inplace=True,
        )
        adata, scores = _annotate_celltype_weighted(
            adata,
            marker_genes=marker_genes,
            mean_values=next_mean,
            celltype_column=celltype_column,
            **kwargs,
        )
        t = adata.obs[celltype_column] == adata.obs[f"{celltype_column}_previous"]
        adata.obs["own_score_genes_diff_iter"] = [int(x) for x in t.to_list()]
        fr = adata.obs["own_score_genes_diff_iter"].value_counts() / len(adata.obs["own_score_genes_diff_iter"])
        completed = completed + 1
        if len(fr) > 1 and (fr[0] * 100) > 0.05:
            log.info("Percentage of cells with changed annotation: " + str(np.round((fr[0] * 100), 2)))
            # changes.append(fr[0] * 100)
            if not any(key in adata.obsm for key in ["umap", "X_umap"]):
                log.info(
                    "Could not find 'umap' or 'X_umap' in .obsm. Will not plot umap. Set 'calculate_umap' to 'True' to enable visualization of the umap."
                )
            else:
                for _name in ["own_score_genes_diff_iter", celltype_column]:
                    sc.pl.umap(adata, color=[_name], show=not output_dir)
                    if output_dir is not None:
                        plt.savefig(os.path.join(output_dir, f"{_name}_{_iteration}"), bbox_inches="tight")
                        plt.close()
                # remove colors added by sc.pl.umap
                del adata.uns[f"{celltype_column}_colors"]
            log.info((adata.obs[celltype_column].value_counts() / len(adata.obs[celltype_column])) * 100)
        else:
            if len(fr) > 1:
                log.info("Percentage of cells with changed annotation: " + str(np.round((fr[0] * 100), 2)))
            else:
                log.info("Percentage of cells with changed annotation: " + str(0.0))
            log.info("converged")
            # changes.append(0)
            break

    adata.obs.drop(columns=["own_score_genes_diff_iter"], inplace=True)
    adata.obs.drop(columns=[f"{celltype_column}_previous"], inplace=True)

    scores.index = adata.obs.index

    celltypes_scored = scores.columns

    adata.obs = pd.concat([adata.obs, scores], axis=1)

    adata.obs[celltype_column] = adata.obs[celltype_column].astype("category")

    return adata, celltypes_scored.to_list(), celltypes_all.to_list()


def _annotate_celltype_weighted(
    adata: AnnData,
    marker_genes: pd.DataFrame,
    min_score: Literal["Zero", "Quantile", None] = "Zero",
    min_score_p: float = 25,
    scaling: Literal[
        "MinMax",
        "ZeroMax",
        "Nmarkers",
        "Robust",
        "Rank",
    ] = "Nmarkers",
    scale_score_p: float = 1,
    mean_values=None,
    celltype_column: str = _ANNOTATION_KEY,
) -> tuple[AnnData, pd.DataFrame]:
    _min_score_options = (
        "Zero",
        "Quantile",
        None,
    )
    if min_score not in _min_score_options:
        raise ValueError(f"'min_score' should be one of {_min_score_options}")
    _scaling_options = (
        "MinMax",
        "ZeroMax",
        "Nmarkers",
        "Robust",
        "Rank",
    )
    if scaling not in _scaling_options:
        raise ValueError(f"'scaling' should be one of {_scaling_options}")

    scores_cell_celltype = pd.DataFrame()
    cell_types = marker_genes.columns.tolist()
    # get the counts out
    matrix = adata.to_df()

    if mean_values is None:
        mean_expression = matrix.mean(axis=0)
    else:
        mean_expression = mean_values

    # make sure the mean over all genes is zero, so no minus necessary anymore
    matrix_minus_mean = matrix - mean_expression
    genes_in_anndata = matrix.columns.to_list()
    # print time for the first part
    for cell_type in cell_types:
        adata.obs["score_" + cell_type] = 0
        for gene in marker_genes[marker_genes[cell_type] > 0].index.tolist():  # select marker genes per celltype
            if gene in genes_in_anndata:  # write
                adata.obs["score_" + cell_type] = (
                    adata.obs["score_" + cell_type] + matrix_minus_mean[gene]
                ) * marker_genes[cell_type][gene]

        scores_cell_celltype[cell_type] = adata.obs["score_" + cell_type]
        adata.obs = adata.obs.drop(columns=["score_" + cell_type])

    scores_cell_celltype.index.name = None
    scores_cell_celltype = scores_cell_celltype.reset_index(drop=True)

    # min score to obtain for a cell type, otherwise 'unknown'

    if min_score == "Zero":
        scores_cell_celltype_ok = scores_cell_celltype > 0
    if min_score == "Quantile":
        scores_cell_celltype_ok = scores_cell_celltype > scores_cell_celltype.quantile(min_score_p / 100)
    if min_score is None:
        scores_cell_celltype_ok = pd.DataFrame(
            True, index=scores_cell_celltype.index, columns=scores_cell_celltype.columns
        )

    # scale scores per cell type to make them more comparable between cell types (because some cell types have more markers etc.)
    # this scaling happens per celtype over the different cells
    if scaling == "MinMax":
        # if you chose this the '- mean_expression' you did before does not have an effect
        scores_cell_celltype = scores_cell_celltype.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

    if scaling == "ZeroMax":
        scores_cell_celltype = scores_cell_celltype.apply(
            lambda x: (x) / (np.max(x))
        )  # (~ min max scaling with min = 0)

    if scaling == "Nmarkers":
        Nmarkers = marker_genes.sum(axis=0).to_list()
        scores_cell_celltype = scores_cell_celltype.div(Nmarkers)
        log.info("Scaling based on number of markers per celltype.")
    if scaling == "Robust":
        for cell_type in cell_types:
            if np.percentile(scores_cell_celltype[cell_type], scale_score_p) < np.percentile(
                scores_cell_celltype[cell_type], 100 - scale_score_p
            ):
                scores_cell_celltype[cell_type] = (
                    scores_cell_celltype[cell_type] - np.percentile(scores_cell_celltype[cell_type], scale_score_p)
                ) / (
                    np.percentile(scores_cell_celltype[cell_type], 100 - scale_score_p)
                    - np.percentile(scores_cell_celltype[cell_type], scale_score_p)
                )

            else:  # MinMax scaling if percentiles are equal
                scores_cell_celltype[cell_type] = (
                    scores_cell_celltype[cell_type] - np.min(scores_cell_celltype[cell_type])
                ) / (np.max(scores_cell_celltype[cell_type]) - np.min(scores_cell_celltype[cell_type]))

    if scaling == "Rank":
        for cell_type in cell_types:
            scores_cell_celltype[cell_type] = scores_cell_celltype[cell_type].rank(pct=True)

    # cell is annotated with the cell type with the highest score (+ this highest score is above min_score)
    scores_cell_celltype[~scores_cell_celltype_ok] = 0

    # cleanliness of each annotation is calculated
    max_scores, second_scores = (
        np.sort(scores_cell_celltype.values)[:, -1:],
        np.sort(scores_cell_celltype.values)[:, -2:-1],
    )

    # Calculate the cleanliness
    difference = max_scores - second_scores
    sum_scores = max_scores + second_scores
    # avoid division by zero by setting a mask, if max_score same as min_score, set cleanliness to inf.
    mask = sum_scores == 0
    cleanliness = np.full(max_scores.shape, np.inf)
    cleanliness[~mask] = difference[~mask] / (sum_scores[~mask] / 2.0)

    scores_cell_celltype[~scores_cell_celltype_ok] = np.nan

    def assign_cell_type(row):
        # If all NaN, return _UNKNOWN_CELLTYPE_KEY for this cell
        if row.isna().all():
            return _UNKNOWN_CELLTYPE_KEY
        # Identify the cell type with the max score
        return row.idxmax()

    adata.obs[celltype_column] = scores_cell_celltype.apply(assign_cell_type, axis=1).values
    adata.obs[_CLEANLINESS_KEY] = cleanliness

    return adata, scores_cell_celltype


def cluster_cleanliness(
    sdata: SpatialData,
    labels_layer: list[str],
    table_layer: str,
    output_layer: str,
    celltypes: list[str],
    celltype_indexes: dict[str, int] | None = None,
    colors: list[str] | None = None,
    celltype_column: str = _ANNOTATION_KEY,
    overwrite: bool = False,
) -> tuple[SpatialData, dict | None]:
    """
    Re-calculates annotations, potentially following corrections to the list of celltypes, or after a manual update of the assigned scores per cell type via e.g. `correct_marker_genes`.

    Celltypes can also be grouped together via the celltype_indexes parameter.
    Returns a `SpatialData` object alongside a dictionary mapping cell types to colors.

    Parameters
    ----------
    sdata
        Data containing spatial information.
    labels_layer
        The labels layer(s) of `sdata` used to select the cells via the _REGION_KEY in `sdata.tables[table_layer].obs`.
        Note that if `output_layer` is equal to `table_layer` and overwrite is True,
        cells in `sdata.tables[table_layer]` linked to other `labels_layer` (via the _REGION_KEY), will be removed from `sdata.tables[table_layer]`.
        If a list of labels layers is provided, they will therefore be scored together (e.g. multiple samples).
    table_layer
        The table layer in `sdata` on which to perform cleaning on.
    output_layer
        The output table layer in `sdata` to which table layer with results of cleaned annotations will be written.
    celltypes
        List of celltypes that you want to use for annotation, can be a subset of what is available in .obs of corresponding table.
    celltype_indexes
        Dictionary with cell type as keys and indexes as values.
        Cell types with provided indexes will be grouped together under new cell type provided as key.
        E.g.:
        celltype_indexes = {"fibroblast": [4,5,23,25], "stellate": [28,29,30]} ->
        celltypes at index 4,5,23 and 25 in provided list of celltypes (after an alphabetic sort) will be grouped together as "fibroblast".
    colors
        List of colors to be used for visualizing different cell types. If not provided,
        a default colormap will be generated.
    celltype_column
        The column name in the SpatialData object's table that specifies the cell type annotations.
        The default value is `_ANNOTATION_KEY`.
    overwrite
        If True, overwrites the `output_layer` if it already exists in `sdata`.

    Returns
    -------
    tuple:

        - Updated spatial data after the cleanliness analysis.

        - Dictionary with cell types as keys and their corresponding colors as values.
    """
    process_table_instance = ProcessTable(sdata, labels_layer=labels_layer, table_layer=table_layer)
    adata = process_table_instance._get_adata()
    celltypes = np.array(sorted(celltypes), dtype=str)
    color_dict = None

    # recalculate annotation, because we possibly did correction on celltype score for certain cells via correct_marker_genes function,
    # or updated the list of celltypes.
    adata, _ = _annotate_celltype(
        adata=adata,
        celltypes=celltypes,
        row_norm=False,
        celltype_column=celltype_column,
    )

    # Create custom colormap for clusters
    if not colors:
        color = np.concatenate(
            (
                plt.get_cmap("tab20c")(np.arange(20)),
                plt.get_cmap("tab20b")(np.arange(20)),
            )
        )
        colors = [mpl.colors.rgb2hex(color[j * 4 + i]) for i in range(4) for j in range(10)]

    adata.uns[f"{celltype_column}_colors"] = colors

    if celltype_indexes:
        adata.obs[f"{celltype_column}Save"] = adata.obs[celltype_column]
        gene_celltypes = {}

        for key, value in celltype_indexes.items():
            gene_celltypes[key] = celltypes[value]

        for gene, _indexes in celltype_indexes.items():
            adata = _annotate_maxscore(adata, gene, gene_celltypes, celltype_column=celltype_column)

        for gene, _indexes in celltype_indexes.items():
            adata = _remove_celltypes(adata, gene, gene_celltypes, celltype_column=celltype_column)

        celltypes_f = np.delete(celltypes, list(chain(*celltype_indexes.values())))  # type: ignore
        celltypes_f = np.append(celltypes_f, list(celltype_indexes.keys()))
        color_dict = dict(zip(celltypes_f, adata.uns[f"{celltype_column}_colors"]))

    else:
        color_dict = dict(zip(celltypes, adata.uns[f"{celltype_column}_colors"]))

    for i, name in enumerate(color_dict.keys()):
        color_dict[name] = colors[i]
    adata.uns[f"{celltype_column}_colors"] = list(map(color_dict.get, adata.obs[celltype_column].cat.categories.values))

    sdata = add_table_layer(
        sdata,
        adata=adata,
        output_layer=output_layer,
        region=process_table_instance.labels_layer,
        overwrite=overwrite,
    )

    return sdata, color_dict


def _annotate_celltype(
    adata: AnnData,
    celltypes: list[str],
    row_norm: bool = False,
    celltype_column: str = _ANNOTATION_KEY,
) -> tuple[SpatialData, list[str]]:
    scoresper_cluster = adata.obs[[col for col in adata.obs if col in celltypes]]

    # Row normalization for visualisation purposes
    if row_norm:
        row_norm = scoresper_cluster.sub(scoresper_cluster.mean(axis=1).values, axis="rows").div(
            scoresper_cluster.std(axis=1).values, axis="rows"
        )
        adata.obs[scoresper_cluster.columns.values] = row_norm
        temp = np.sort(row_norm)[:, -2:]
    else:
        temp = np.sort(scoresper_cluster)[:, -2:]

    max_scores = temp[:, 1].reshape(-1, 1)
    second_scores = temp[:, 0].reshape(-1, 1)

    # Calculate the cleanliness
    difference = max_scores - second_scores
    sum_scores = max_scores + second_scores
    # avoid division by zero by setting a mask, if max_score same as min_score, set cleanliness to inf.
    mask = sum_scores == 0
    cleanliness = np.full(max_scores.shape, np.inf)
    cleanliness[~mask] = difference[~mask] / (sum_scores[~mask] / 2.0)

    adata.obs[_CLEANLINESS_KEY] = cleanliness

    def assign_cell_type(row):
        # Identify the cell type with the max score
        max_score_type = row.idxmax()
        # If max score is <= 0, assign _UNKNOWN_CELLTYPE_KEY
        if row[max_score_type] <= 0:
            return _UNKNOWN_CELLTYPE_KEY
        else:
            return max_score_type

    # Assign _UNKNOWN_CELLTYPE_KEY cell_type if no cell type could be found that has larger expression than random sample
    # as calculated by sc.tl.score_genes function of scanpy.
    adata.obs[celltype_column] = scoresper_cluster.apply(assign_cell_type, axis=1)
    adata.obs[celltype_column] = adata.obs[celltype_column].astype("category")
    # Set the Cleanliness score for UNKNOWN_CELLTYPE_KEY equal to 0 (i.e. not clean)
    adata.obs.loc[adata.obs[celltype_column] == _UNKNOWN_CELLTYPE_KEY, _CLEANLINESS_KEY] = 0

    return adata, list(scoresper_cluster.columns.values)


def _remove_celltypes(adata: AnnData, types: str, indexes: dict, celltype_column: str = _ANNOTATION_KEY) -> AnnData:
    """Returns the AnnData object."""
    for index in indexes[types]:
        if index in adata.obs[celltype_column].cat.categories:
            adata.obs[celltype_column] = adata.obs[celltype_column].cat.remove_categories(index)
    return adata


def _annotate_maxscore(adata: AnnData, types: str, indexes: dict, celltype_column: str = _ANNOTATION_KEY) -> AnnData:
    """Returns the AnnData object.

    Adds types to the Anndata maxscore category.
    """
    adata.obs[celltype_column] = adata.obs[celltype_column].cat.add_categories([types])
    for i, val in enumerate(adata.obs[celltype_column]):
        if val in indexes[types]:
            adata.obs[celltype_column][i] = types
    return adata
