from itertools import chain
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from spatialdata import SpatialData

from sparrow.table._table import ProcessTable, _add_table_layer
from sparrow.utils._keys import _ANNOTATION_KEY, _CLEANLINESS_KEY, _UNKNOWN_CELLTYPE_KEY
from sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def score_genes(
    sdata: SpatialData,
    labels_layer: list[str],
    table_layer: str,
    output_layer: str,
    path_marker_genes: str,
    delimiter=",",
    row_norm: bool = False,
    repl_columns: Optional[Dict[str, str]] = None,
    del_celltypes: Optional[List[str]] = None,
    input_dict: bool = False,
    overwrite: bool = False,
) -> Tuple[SpatialData, list[str], list[str]]:
    """
    The function loads marker genes from a CSV file and scores cells for each cell type using those markers using scanpy's score_genes function.

    Marker genes can be provided as a one-hot encoded matrix with cell types listed in the first row, and marker genes in the first column;
    or in dictionary format. The function further allows replacements of column names and deletions of specific marker genes.

    Parameters
    ----------
    sdata : SpatialData
        Data containing spatial information.
    labels_layer : str or Iterable[str]
        The labels layer(s) of `sdata` used to select the cells via the _REGION_KEY.
        Note that if `output_layer` is equal to `table_layer` and overwrite is True,
        cells in `sdata.tables[table_layer]` linked to other `labels_layer` (via the _REGION_KEY), will be removed from `sdata.tables[table_layer]`.
        If a list of labels layers is provided, they will therefore be scored together (e.g. multiple samples).
    table_layer: str, optional
        The table layer in `sdata` on which to perform annotation on.
    output_layer: str, optional
        The output table layer in `sdata` to which table layer with results of annotation will be written.
    path_marker_genes : str
        Path to the CSV file containing the marker genes.
        CSV file should be a one-hot encoded matrix with cell types listed in the first row, and marker genes in the first column.
    delimiter : str, optional
        Delimiter used in the CSV file, default is ','.
    row_norm : bool, optional
        Flag to determine if row normalization is applied, default is False.
    repl_columns : dict, optional
        Dictionary containing cell types to be replaced. The keys are the original cell type names and
        the values are their replacements.
    del_celltypes : list, optional
        List of cell types to be deleted from the list of possible cell type candidates.
        Cells are scored for these cell types, but will not be assigned a cell type from this list.
    input_dict : bool, optional
        If True, the marker gene list from the CSV file is treated as a dictionary with the first column being
        the cell type names and the subsequent columns being the marker genes for those cell types. Default is False.
    overwrite : bool, default=False
        If True, overwrites the `output_layer` if it already exists in `sdata`.

    Returns
    -------
    list[str]
        list of strings, with all celltypes that are scored (but are not in the del_celltypes list).
    list[str]
        list of strings, with all celltypes, some of which may not be scored, because their corresponding transcripts do not appear in the region of interest. _UNKNOWN_CELLTYPE_KEY, is also added if it is detected.

    Notes
    -----
    The cell type `_UNKNOWN_CELLTYPE_KEY` is reserved for cells that could not be assigned a specific cell type.

    """
    process_table_instance = ProcessTable(sdata, labels_layer=labels_layer, table_layer=table_layer)
    adata = process_table_instance._get_adata()
    # Load marker genes from csv
    if input_dict:
        df_markers = pd.read_csv(path_marker_genes, header=None, index_col=0, delimiter=delimiter)
        df_markers = df_markers.T
        genes_dict = df_markers.to_dict("list")
        for i in genes_dict:
            genes_dict[i] = [x for x in genes_dict[i] if str(x) != "nan"]
    # Replace column names in marker genes
    else:
        df_markers = pd.read_csv(path_marker_genes, index_col=0, delimiter=delimiter)
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

    assert (
        _UNKNOWN_CELLTYPE_KEY not in genes_dict.keys()
    ), f"Cell type {_UNKNOWN_CELLTYPE_KEY} is reserved for cells that could not be assigned a specific cell type"

    # Score all cells for all celltypes
    for key, value in genes_dict.items():
        try:
            sc.tl.score_genes(adata, value, score_name=key)
        except ValueError:
            log.warning(f"Markergenes {value} not present in region, celltype {key} not found")

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
        celltype_column=_ANNOTATION_KEY,
    )

    # add _UNKNOWN_CELLTYPE_KEY to the list of celltypes if it is detected.
    if _UNKNOWN_CELLTYPE_KEY in adata.obs[_ANNOTATION_KEY].cat.categories:
        genes_dict[_UNKNOWN_CELLTYPE_KEY] = []

    celltypes_all = list(genes_dict.keys())

    sdata = _add_table_layer(
        sdata,
        adata=adata,
        output_layer=output_layer,
        region=process_table_instance.labels_layer,
        overwrite=overwrite,
    )

    return sdata, celltypes_scored, celltypes_all


def cluster_cleanliness(
    sdata: SpatialData,
    labels_layer: list[str],
    table_layer: str,
    output_layer: str,
    celltypes: List[str],
    celltype_indexes: Optional[Dict[str, int]] = None,
    colors: Optional[List[str]] = None,
    overwrite: bool = False,
) -> Tuple[SpatialData, Optional[dict]]:
    """
    Re-calculates annotations, potentially following corrections to the list of celltypes, or after a manual update of the assigned scores per cell type via e.g. `correct_marker_genes`.

    Celltypes can also be grouped together via the celltype_indexes parameter.
    Returns a `SpatialData` object alongside a dictionary mapping cell types to colors.

    Parameters
    ----------
    sdata : SpatialData
        Data containing spatial information.
    labels_layer : str or Iterable[str]
        The labels layer(s) of `sdata` used to select the cells via the _REGION_KEY.
        Note that if `output_layer` is equal to `table_layer` and overwrite is True,
        cells in `sdata.tables[table_layer]` linked to other `labels_layer` (via the _REGION_KEY), will be removed from `sdata.tables[table_layer]`.
        If a list of labels layers is provided, they will therefore be scored together (e.g. multiple samples).
    table_layer: str, optional
        The table layer in `sdata` on which to perform cleaning on.
    output_layer: str, optional
        The output table layer in `sdata` to which table layer with results of cleaned annotations will be written.
    celltypes : List[str]
        List of celltypes that you want to use for annotation, can be a subset of what is available in .obs of corresponding table.
    celltype_indexes : dict, optional
        Dictionary with cell type as keys and indexes as values.
        Cell types with provided indexes will be grouped together under new cell type provided as key.
        E.g.:
        celltype_indexes = {"fibroblast": [4,5,23,25], "stellate": [28,29,30]} ->
        celltypes at index 4,5,23 and 25 in provided list of celltypes (after an alphabetic sort) will be grouped together as "fibroblast".
    colors : list, optional
        List of colors to be used for visualizing different cell types. If not provided,
        a default colormap will be generated.
    overwrite : bool, default=False
        If True, overwrites the `output_layer` if it already exists in `sdata`.

    Returns
    -------
    SpatialData
        Updated spatial data after the cleanliness analysis.
    dict
        Dictionary with cell types as keys and their corresponding colors as values.

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
        celltype_column=_ANNOTATION_KEY,
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

    adata.uns[f"{_ANNOTATION_KEY}_colors"] = colors

    if celltype_indexes:
        adata.obs[f"{_ANNOTATION_KEY}Save"] = adata.obs[_ANNOTATION_KEY]
        gene_celltypes = {}

        for key, value in celltype_indexes.items():
            gene_celltypes[key] = celltypes[value]

        for gene, _indexes in celltype_indexes.items():
            adata = _annotate_maxscore(adata, gene, gene_celltypes)

        for gene, _indexes in celltype_indexes.items():
            adata = _remove_celltypes(adata, gene, gene_celltypes)

        celltypes_f = np.delete(celltypes, list(chain(*celltype_indexes.values())))  # type: ignore
        celltypes_f = np.append(celltypes_f, list(celltype_indexes.keys()))
        color_dict = dict(zip(celltypes_f, adata.uns[f"{_ANNOTATION_KEY}_colors"]))

    else:
        color_dict = dict(zip(celltypes, adata.uns[f"{_ANNOTATION_KEY}_colors"]))

    for i, name in enumerate(color_dict.keys()):
        color_dict[name] = colors[i]
    adata.uns[f"{_ANNOTATION_KEY}_colors"] = list(map(color_dict.get, adata.obs[_ANNOTATION_KEY].cat.categories.values))

    sdata = _add_table_layer(
        sdata,
        adata=adata,
        output_layer=output_layer,
        region=process_table_instance.labels_layer,
        overwrite=overwrite,
    )

    return sdata, color_dict


def _annotate_celltype(
    adata: AnnData,
    celltypes: List[str],
    row_norm: bool = False,
    celltype_column: str = _ANNOTATION_KEY,
) -> Tuple[SpatialData, list[str]]:
    scoresper_cluster = adata.obs[[col for col in adata.obs if col in celltypes]]

    # Row normalization for visualisation purposes
    if row_norm:
        row_norm = scoresper_cluster.sub(scoresper_cluster.mean(axis=1).values, axis="rows").div(
            scoresper_cluster.std(axis=1).values, axis="rows"
        )
        adata.obs[scoresper_cluster.columns.values] = row_norm
        temp = pd.DataFrame(np.sort(row_norm)[:, -2:])
    else:
        temp = pd.DataFrame(np.sort(scoresper_cluster)[:, -2:])

    scores = (temp[1] - temp[0]) / ((temp[1] + temp[0]) / 2)
    adata.obs[_CLEANLINESS_KEY] = scores.values

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


def _remove_celltypes(adata: AnnData, types: str, indexes: dict) -> AnnData:
    """Returns the AnnData object."""
    for index in indexes[types]:
        if index in adata.obs[_ANNOTATION_KEY].cat.categories:
            adata.obs[_ANNOTATION_KEY] = adata.obs[_ANNOTATION_KEY].cat.remove_categories(index)
    return adata


def _annotate_maxscore(adata: AnnData, types: str, indexes: dict) -> AnnData:
    """Returns the AnnData object.

    Adds types to the Anndata maxscore category.
    """
    adata.obs[_ANNOTATION_KEY] = adata.obs[_ANNOTATION_KEY].cat.add_categories([types])
    for i, val in enumerate(adata.obs[_ANNOTATION_KEY]):
        if val in indexes[types]:
            adata.obs[_ANNOTATION_KEY][i] = types
    return adata
