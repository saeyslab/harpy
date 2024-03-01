from itertools import chain
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from spatialdata import SpatialData

from sparrow.table._table import _back_sdata_table_to_zarr
from sparrow.utils._keys import _ANNOTATION_KEY, _CLEANLINESS_KEY, _UNKNOWN_CELLTYPE_KEY
from sparrow.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def score_genes(
    sdata: SpatialData,
    path_marker_genes: str,
    delimiter=",",
    row_norm: bool = False,
    repl_columns: Optional[Dict[str, str]] = None,
    del_celltypes: Optional[List[str]] = None,
    input_dict=False,
) -> Tuple[dict, pd.DataFrame]:
    """
    The function loads marker genes from a CSV file and scores cells for each cell type using those markers using scanpy's score_genes function.

    Marker genes can be provided as a one-hot encoded matrix with cell types listed in the first row, and marker genes in the first column;
    or in dictionary format. The function further allows replacements of column names and deletions of specific marker genes.

    Parameters
    ----------
    sdata : SpatialData
        Data containing spatial information.
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

    Returns
    -------
    dict
        Dictionary with cell types as keys and their respective marker genes as values.
    pd.DataFrame
        Index:
            cells: The index corresponds to indivdual cells ID's.
        Columns:
            celltypes (as provided via the markers file).
        Values:
            Score obtained using scanpy's score_genes function for each celltype and for each cell.

    Notes
    -----
    The cell type `_UNKNOWN_CELLTYPE_KEY` is reserved for cells that could not be assigned a specific cell type.

    """
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
            sc.tl.score_genes(sdata.table, value, score_name=key)
        except ValueError:
            log.warning(f"Markergenes {value} not present in region, celltype {key} not found")

    # Delete genes from marker genes and genes dict
    if del_celltypes:
        for gene in del_celltypes:
            if gene in df_markers.columns:
                del df_markers[gene]
            if gene in genes_dict.keys():
                del genes_dict[gene]

    sdata, scoresper_cluster = _annotate_celltype(
        sdata=sdata,
        celltypes=df_markers.columns,
        row_norm=row_norm,
        celltype_column=_ANNOTATION_KEY,
    )

    # add _UNKNOWN_CELLTYPE_KEY to the list of celltypes if it is detected.
    if _UNKNOWN_CELLTYPE_KEY in sdata.table.obs[_ANNOTATION_KEY].cat.categories:
        genes_dict[_UNKNOWN_CELLTYPE_KEY] = []

    _back_sdata_table_to_zarr(sdata)

    return genes_dict, scoresper_cluster


def cluster_cleanliness(
    sdata: SpatialData,
    celltypes: List[str],
    celltype_indexes: Optional[Dict[str, int]] = None,
    colors: Optional[List[str]] = None,
) -> Tuple[SpatialData, Optional[dict]]:
    """
    Re-calculates annotations, potentially following corrections to the list of celltypes, or after a manual update of the assigned scores per cell type via e.g. `correct_marker_genes`.

    Celltypes can also be grouped together via the celltype_indexes parameter.
    Returns a `SpatialData` object alongside a dictionary mapping cell types to colors.

    Parameters
    ----------
    sdata : SpatialData
        Data containing spatial information.
    celltypes : List[str]
        List of celltypes used for annotation.
    celltype_indexes : dict, optional
        Dictionary with cell type as keys and indexes as values.
        Cell types with provided indexes will be grouped together under new cell type provided as key.
        E.g.:
        celltype_indexes = {"fibroblast": [4,5,23,25], "stellate": [28,29,30]} ->
        celltypes at index 4,5,23 and 25 in provided list of celltypes (after an alphabetic sort) will be grouped together as "fibroblast".
    colors : list, optional
        List of colors to be used for visualizing different cell types. If not provided,
        a default colormap will be generated.

    Returns
    -------
    SpatialData
        Updated spatial data after the cleanliness analysis.
    dict
        Dictionary with cell types as keys and their corresponding colors as values.

    """
    celltypes = np.array(sorted(celltypes), dtype=str)
    color_dict = None

    # recalculate annotation, because we possibly did correction on celltype score for certain cells via correct_marker_genes function,
    # or updated the list of celltypes.
    sdata, _ = _annotate_celltype(
        sdata=sdata,
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

    sdata.table.uns[f"{_ANNOTATION_KEY}_colors"] = colors

    if celltype_indexes:
        sdata.table.obs[f"{_ANNOTATION_KEY}Save"] = sdata.table.obs[_ANNOTATION_KEY]
        gene_celltypes = {}

        for key, value in celltype_indexes.items():
            gene_celltypes[key] = celltypes[value]

        for gene, _indexes in celltype_indexes.items():
            sdata = _annotate_maxscore(gene, gene_celltypes, sdata)

        for gene, _indexes in celltype_indexes.items():
            sdata = _remove_celltypes(gene, gene_celltypes, sdata)

        celltypes_f = np.delete(celltypes, list(chain(*celltype_indexes.values())))  # type: ignore
        celltypes_f = np.append(celltypes_f, list(celltype_indexes.keys()))
        color_dict = dict(zip(celltypes_f, sdata.table.uns[f"{_ANNOTATION_KEY}_colors"]))

    else:
        color_dict = dict(zip(celltypes, sdata.table.uns[f"{_ANNOTATION_KEY}_colors"]))

    for i, name in enumerate(color_dict.keys()):
        color_dict[name] = colors[i]
    sdata.table.uns[f"{_ANNOTATION_KEY}_colors"] = list(
        map(color_dict.get, sdata.table.obs[_ANNOTATION_KEY].cat.categories.values)
    )

    _back_sdata_table_to_zarr(sdata)

    return sdata, color_dict


def _annotate_celltype(
    sdata: SpatialData,
    celltypes: List[str],
    row_norm: bool = False,
    celltype_column: str = _ANNOTATION_KEY,
) -> Tuple[SpatialData, pd.DataFrame]:
    scoresper_cluster = sdata.table.obs[[col for col in sdata.table.obs if col in celltypes]]

    # Row normalization for visualisation purposes
    if row_norm:
        row_norm = scoresper_cluster.sub(scoresper_cluster.mean(axis=1).values, axis="rows").div(
            scoresper_cluster.std(axis=1).values, axis="rows"
        )
        sdata.table.obs[scoresper_cluster.columns.values] = row_norm
        temp = pd.DataFrame(np.sort(row_norm)[:, -2:])
    else:
        temp = pd.DataFrame(np.sort(scoresper_cluster)[:, -2:])

    scores = (temp[1] - temp[0]) / ((temp[1] + temp[0]) / 2)
    sdata.table.obs[_CLEANLINESS_KEY] = scores.values

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
    sdata.table.obs[celltype_column] = scoresper_cluster.apply(assign_cell_type, axis=1)
    sdata.table.obs[celltype_column] = sdata.table.obs[celltype_column].astype("category")
    # Set the Cleanliness score for UNKNOWN_CELLTYPE_KEY equal to 0 (i.e. not clean)
    sdata.table.obs.loc[sdata.table.obs[celltype_column] == _UNKNOWN_CELLTYPE_KEY, _CLEANLINESS_KEY] = 0

    return sdata, scoresper_cluster


def _remove_celltypes(types: str, indexes: dict, sdata):
    """Returns the AnnData object."""
    for index in indexes[types]:
        if index in sdata.table.obs[_ANNOTATION_KEY].cat.categories:
            sdata.table.obs[_ANNOTATION_KEY] = sdata.table.obs[_ANNOTATION_KEY].cat.remove_categories(index)
    return sdata


def _annotate_maxscore(types: str, indexes: dict, sdata):
    """Returns the AnnData object.

    Adds types to the Anndata maxscore category.
    """
    sdata.table.obs[_ANNOTATION_KEY] = sdata.table.obs[_ANNOTATION_KEY].cat.add_categories([types])
    for i, val in enumerate(sdata.table.obs[_ANNOTATION_KEY]):
        if val in indexes[types]:
            sdata.table.obs[_ANNOTATION_KEY][i] = types
    return sdata
