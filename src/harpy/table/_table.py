from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from anndata import AnnData
from spatialdata import SpatialData
from spatialdata.models import TableModel

from harpy.shape._shape import filter_shapes_layer
from harpy.table._manager import TableLayerManager
from harpy.utils._keys import _CELLSIZE_KEY, _INSTANCE_KEY, _REGION_KEY
from harpy.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class ProcessTable:
    def __init__(
        self,
        sdata: SpatialData,
        table_layer: str,
        labels_layer: str | Iterable[str] | None = None,
    ):
        """
        Base class for implementation of processing on tables.

        Parameters
        ----------
        spatial_data: SpatialData
            The SpatialData object containing spatial data.
        table_layer: str
            The table layer to use.
        labels_layer : str or Iterable[str] or None
            The label layer(s) to use.
        """
        if sdata.tables == {}:
            raise ValueError(
                "Provided SpatialData object 'sdata' does not contain any 'tables'. "
                "Please create tables via e.g. 'harpy.tb.allocation' or 'harpy.tb.allocation_intensity' functions."
            )

        if labels_layer is not None:
            if sdata.labels == {}:
                raise ValueError(
                    "Provided SpatialData object 'sdata' does not contain 'labels'. "
                    "Please create a labels layer via e.g. 'harpy.im.segment'."
                )
            labels_layer = (
                list(labels_layer)
                if isinstance(labels_layer, Iterable) and not isinstance(labels_layer, str)
                else [labels_layer]
            )

        self.sdata = sdata
        self.labels_layer = labels_layer
        self.table_layer = table_layer
        self._validated_table_layer()
        if self.labels_layer is not None:
            self._validate_layer(layer_list=self.labels_layer)
        if self.labels_layer is None:
            self._validate()

    def _validate_layer(self, layer_list, layer_type="labels"):
        """Generic layer validation helper to reduce code duplication."""
        for _layer in layer_list:
            if _layer not in [*getattr(self.sdata, layer_type)]:
                raise ValueError(f"'{layer_type}' layer '{_layer}' not in 'sdata.{layer_type}'.")
            if _layer not in self.sdata.tables[self.table_layer].obs[_REGION_KEY].cat.categories:
                raise ValueError(
                    f"'{layer_type}' layer '{_layer}' not in 'sdata.tables[\"{self.table_layer}\"].obs[_REGION_KEY].cat.categories'"
                )
            # Check for uniqueness of instance keys
            assert (
                self.sdata.tables[self.table_layer]
                .obs[self.sdata.tables[self.table_layer].obs[_REGION_KEY] == _layer][_INSTANCE_KEY]
                .is_unique
            ), f"'{_INSTANCE_KEY}' is not unique for '{_REGION_KEY}' == '{_layer}'. Please make sure these are unique."

    def _validate(self):
        assert self.sdata.tables[self.table_layer].obs[_INSTANCE_KEY].is_unique, (
            f"'{_INSTANCE_KEY}' is not unique. Please make sure these are unique, or specify a 'labels_layer' via '{_REGION_KEY}'."
        )

    def _validated_table_layer(self):
        """Validate if the specified table layer exists in the SpatialData object."""
        if self.table_layer not in [*self.sdata.tables]:
            raise ValueError(f"table layer '{self.table_layer}' not in 'sdata.tables'.")

    def _get_adata(
        self, index_names_var: Iterable[str] | None = None, index_positions_var: Iterable[int] | None = None
    ) -> AnnData:
        """Preprocess the data by filtering based on the table layer and setting spatialdata attributes."""
        if self.labels_layer is not None:
            adata = self.sdata.tables[self.table_layer][
                self.sdata.tables[self.table_layer].obs[_REGION_KEY].isin(self.labels_layer)
            ]
        else:
            adata = self.sdata.tables[self.table_layer]
        if index_names_var is not None or index_positions_var is not None:
            adata = self._subset_adata_var(
                adata, index_names_var=index_names_var, index_positions_var=index_positions_var
            )
        adata = adata.copy()
        if self.labels_layer is not None:
            adata.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY] = self.labels_layer

        return adata

    @staticmethod
    def _type_check_before_pca(adata: AnnData):
        # type check because pca raises error when adata.X is sparse and of dtype int.
        if np.issubdtype(adata.X.dtype, np.integer):
            raise ValueError(
                f"Data matrix of AnnData table is of type '{adata.X.dtype}', "
                "which indicates no preprocessing is performed. "
                "Please consider preprocessing the data first before calculating pca ('scanpy.tl.pca') or calculating neighborhood grap (`scanpy.pp.neighbors`), "
                "e.g. with 'scanpy.pp.scale'."
                ""
            )

    @staticmethod
    def _subset_adata_var(
        adata: AnnData, index_names_var: Iterable[str] | None = None, index_positions_var: Iterable[int] | None = None
    ) -> AnnData:
        """
        Subsets AnnData object by index names or index positions of `adata.var`.

        Parameters
        ----------
        adata: AnnData object.
        index_names_var: List of index names to subset. If None, the function will use index_positions.
        index_positions_var: List of integer positions to subset. Used if index_names_var is None.

        Returns
        -------
        - Subsetted AnnData object.
        """
        if index_names_var is not None:
            index_names_var = (
                list(index_names_var)
                if isinstance(index_names_var, Iterable) and not isinstance(index_names_var, str)
                else [index_names_var]
            )
            selected_var = adata.var.loc[index_names_var]
        elif index_positions_var is not None:
            index_names_var = (
                list(index_positions_var) if isinstance(index_positions_var, Iterable) else [index_positions_var]
            )
            selected_var = adata.var.iloc[index_positions_var]
        else:
            raise ValueError("Either index_names or index_positions must be provided.")

        selected_columns = adata.var_names.intersection(selected_var.index)

        adata = adata[:, selected_columns]

        return adata


def correct_marker_genes(
    sdata: SpatialData,
    labels_layer: list[str],
    table_layer: str,
    output_layer: str,
    celltype_correction_dict: dict[str, tuple[float, float]],
    overwrite: bool = False,
) -> SpatialData:
    """
    Correct celltype expression in `sdata.tables[table_layer]` using `celltype_correction_dict`.

    Corrects celltypes that are higher expessed by dividing them by a value if they exceed a certain threshold.
    The `celltype_correction_dict` has as keys the celltypes that should be corrected and as values the threshold and the divider.

    Parameters
    ----------
    sdata
        The SpatialData object.
    labels_layer
        The labels layer(s) of `sdata` used to select the cells via the _REGION_KEY  in `sdata.tables[table_layer].obs`.
        Note that if `output_layer` is equal to `table_layer` and overwrite is True,
        cells in `sdata.tables[table_layer]` linked to other `labels_layer` (via the _REGION_KEY), will be removed from `sdata.tables[table_layer]`
        (also from the backing zarr store if it is backed).
    table_layer
        The table layer in `sdata`.
    output_layer
        The output table layer in `sdata`.
    celltype_correction_dict
        The `celltype_correction_dict` has as keys the celltypes that should be corrected and as values the threshold and the divider.

    Returns
    -------
    The updated SpatialData object.
    """
    process_table_instance = ProcessTable(sdata, labels_layer=labels_layer, table_layer=table_layer)
    adata = process_table_instance._get_adata()
    # Correct for all the genes
    for celltype, values in celltype_correction_dict.items():
        if celltype not in adata.obs.columns:
            log.info(
                f"Cell type '{celltype}' not in obs of AnnData object. Skipping. Please first calculate gene expression for this cell type."
            )
            continue
        adata.obs[celltype] = np.where(
            adata.obs[celltype] > values[0],
            adata.obs[celltype] / values[1],
            adata.obs[celltype],
        )

    sdata = add_table_layer(
        sdata,
        adata=adata,
        output_layer=output_layer,
        region=process_table_instance.labels_layer,
        overwrite=overwrite,
    )

    return sdata


def filter_on_size(
    sdata: SpatialData,
    labels_layer: list[str],
    table_layer: str,
    output_layer: str,
    cellsize_key=_CELLSIZE_KEY,
    min_size: float | int | None = None,
    max_size: float | int | None = None,
    update_shapes_layers: bool = True,
    prefix_filtered_shapes_layer: str = "filtered_size",
    overwrite: bool = False,
) -> SpatialData:
    """Returns the updated SpatialData object.

    All cells with a size outside of the min and max size range are removed using the `cellsize_key` in `.obs`. Run e.g. `harpy.tb.preprocess_transcriptomics` or `harpy.tb.preprocess_proteomics` to obtain cell sizes.
    Cells are kept if min_size ≤ cellsize_key ≤ max_size.

    Parameters
    ----------
    sdata
        The SpatialData object.
    labels_layer
        The labels layer(s) of `sdata` used to select the cells via the _REGION_KEY  in `sdata.tables[table_layer].obs`.
        Note that if `output_layer` is equal to `table_layer` and overwrite is True,
        cells in `sdata.tables[table_layer]` linked to other `labels_layer` (via the _REGION_KEY), will be removed from `sdata.tables[table_layer]`
        (also from the backing zarr store if it is backed).
    table_layer
        The table layer in `sdata`.
    output_layer
        The output table layer in `sdata`.
    cellsize_key
        Column in `sdata.tables[table_layer].obs` containing cell sizes.
    min_size
        minimum size in pixels. If None, this value is not used for filtering.
    max_size
        maximum size in pixels. If None, this value is not used for filtering.
    update_shapes_layers
        Whether to filter the shapes layers associated with `labels_layer`.
        If set to `True`, cells that do not appear in resulting `output_layer` (with `_REGION_KEY` equal to `labels_layer`) will be removed from the shapes layers (via `_INSTANCE_KEY`) in the `sdata` object.
        Filtered shapes will be added to `sdata` with prefix `prefix_filtered_shapes_layer`.
    prefix_filtered_shapes_layer
        prefix to use for filtered shapes layer if update_shapes_layers is True. Defaults to 'filtered_size'.
    overwrite
        If True, overwrites the `output_layer` if it already exists in `sdata`.

    Returns
    -------
    The updated SpatialData object.
    """
    sdata = filter_numerical(
        sdata=sdata,
        labels_layer=labels_layer,
        table_layer=table_layer,
        output_layer=output_layer,
        numerical_column=cellsize_key,
        min_value=min_size,
        max_value=max_size,
        update_shapes_layers=update_shapes_layers,
        prefix_filtered_shapes_layer=prefix_filtered_shapes_layer,
        overwrite=overwrite,
    )

    return sdata


def filter_numerical(
    sdata: SpatialData,
    labels_layer: list[str],
    table_layer: str,
    output_layer: str,
    numerical_column: str,
    min_value: float | int | None = None,
    max_value: float | int | None = None,
    update_shapes_layers: bool = True,
    prefix_filtered_shapes_layer: str = "filtered",
    overwrite: bool = False,
) -> SpatialData:
    """Returns the updated SpatialData object.

    All cells with a size outside of the min and max size range are removed using the `numerical_column` in `.obs` (cells are kept if min_value ≤ numerical_column ≤ max_value).

    Parameters
    ----------
    sdata
        The SpatialData object.
    labels_layer
        The labels layer(s) of `sdata` used to select the cells via the _REGION_KEY  in `sdata.tables[table_layer].obs`.
        Note that if `output_layer` is equal to `table_layer` and overwrite is True,
        cells in `sdata.tables[table_layer]` linked to other `labels_layer` (via the _REGION_KEY), will be removed from `sdata.tables[table_layer]`
        (also from the backing zarr store if it is backed).
    table_layer
        The table layer in `sdata`.
    output_layer
        The output table layer in `sdata`.
    numerical_column
        Name of numerical column in `sdata.tables[table_layer].obs` to use for filtering with `min_value` and `max_value`.
    min_value
        minimum value of `numerical_column`. If None, this value is not used for filtering.
    max_value
        maximum value of `numerical_column`. If None, this value is not used for filtering.
    update_shapes_layers
        Whether to filter the shapes layers associated with `labels_layer`.
        If set to `True`, cells that do not appear in resulting `output_layer` (with `_REGION_KEY` equal to `labels_layer`) will be removed from the shapes layers (via `_INSTANCE_KEY`) in the `sdata` object.
        Filtered shapes will be added to `sdata` with prefix `prefix_filtered_shapes_layer`.
    prefix_filtered_shapes_layer
        prefix to use for filtered shapes layer if update_shapes_layers is True. Defaults to 'filtered'.
    overwrite
        If True, overwrites the `output_layer` if it already exists in `sdata`.

    Returns
    -------
    The updated SpatialData object.
    """
    process_table_instance = ProcessTable(sdata, labels_layer=labels_layer, table_layer=table_layer)
    adata = process_table_instance._get_adata().copy()
    start = adata.shape[0]

    if numerical_column not in adata.obs.columns:
        raise ValueError(f"Column '{numerical_column}' not found in '{table_layer}.obs'. ")

    if not np.issubdtype(adata.obs[numerical_column].dtype, np.number):
        raise ValueError(
            f"Column '{numerical_column}' must be numeric, but dtype is {adata.obs[numerical_column].dtype}."
        )

    # Filter cells based on min and max values
    mask = pd.Series(True, index=adata.obs.index)

    if min_value is not None:
        below = (adata.obs[numerical_column] < min_value).sum()
        log.info(f"Removed {below} cells below {min_value}.")
        mask &= adata.obs[numerical_column] >= min_value
    if max_value is not None:
        above = (adata.obs[numerical_column] > max_value).sum()
        log.info(f"Removed {above} cells above {max_value}.")
        mask &= adata.obs[numerical_column] <= max_value

    adata = adata[mask, :].copy()

    sdata = add_table_layer(
        sdata,
        adata=adata,
        output_layer=output_layer,
        region=process_table_instance.labels_layer,
        overwrite=overwrite,
    )

    if update_shapes_layers:
        for _labels_layer in process_table_instance.labels_layer:
            sdata = filter_shapes_layer(
                sdata,
                table_layer=output_layer,
                labels_layer=_labels_layer,
                prefix_filtered_shapes_layer=prefix_filtered_shapes_layer,
            )

    filtered = start - adata.shape[0]
    log.info(
        f"Removed {filtered} / {start} cells based on '{numerical_column}' (min={min_value}, max={max_value}) and kept {adata.shape[0]}."
    )

    return sdata


def filter_categorical(
    sdata: SpatialData,
    labels_layer: list[str],
    table_layer: str,
    output_layer: str,
    categorical_column: str,
    include_values: str | Sequence[str] | None = None,
    exclude_values: str | Sequence[str] | None = None,
    update_shapes_layers: bool = True,
    prefix_filtered_shapes_layer: str = "filtered",
    overwrite: bool = False,
) -> SpatialData:
    """Filter cells based on categorical values in `.obs`.

    Removes or keeps cells based on specific values in a categorical column of
    `sdata.tables[table_layer].obs`.

    Parameters
    ----------
    sdata
        The SpatialData object.
    labels_layer
        The labels layer(s) of `sdata` used to select the cells via the _REGION_KEY  in `sdata.tables[table_layer].obs`.
        Note that if `output_layer` is equal to `table_layer` and overwrite is True,
        cells in `sdata.tables[table_layer]` linked to other `labels_layer` (via the _REGION_KEY), will be removed from `sdata.tables[table_layer]`
        (also from the backing zarr store if it is backed).
    table_layer
        The table layer in `sdata`.
    output_layer
        The output table layer in `sdata`.
    categorical_column
        Name of the categorical column in `.obs` to use for filtering.
    include_values
        Value(s) to keep. Only cells whose `categorical_column` matches one of these
        values will be kept. Mutually exclusive with `exclude_values`.
    exclude_values
        Value(s) to remove. Cells whose `categorical_column` matches one of these
        values will be removed. Mutually exclusive with `include_values`.
    update_shapes_layers
        Whether to filter the shapes layers associated with `labels_layer`.
        If set to `True`, cells that do not appear in resulting `output_layer` (with `_REGION_KEY` equal to `labels_layer`) will be removed from the shapes layers (via `_INSTANCE_KEY`) in the `sdata` object.
        Filtered shapes will be added to `sdata` with prefix `prefix_filtered_shapes_layer`.
    prefix_filtered_shapes_layer
        prefix to use for filtered shapes layer if update_shapes_layers is True. Defaults to 'filtered'.
    overwrite
        If True, overwrites the `output_layer` if it already exists in `sdata`.

    Returns
    -------
    The updated SpatialData object.
    """
    if include_values is not None and exclude_values is not None:
        raise ValueError("Specify only one of 'include_values' or 'exclude_values'.")

    process_table_instance = ProcessTable(sdata, labels_layer=labels_layer, table_layer=table_layer)
    adata = process_table_instance._get_adata().copy()
    start = adata.shape[0]

    if categorical_column not in adata.obs.columns:
        raise ValueError(f"Column '{categorical_column}' not found in '{table_layer}.obs'.")

    # Ensure include/exclude are lists
    if isinstance(include_values, str):
        include_values = [include_values]
    if isinstance(exclude_values, str):
        exclude_values = [exclude_values]

    # Filter
    mask = pd.Series(True, index=adata.obs.index)

    if include_values is not None:
        kept = adata.obs[categorical_column].isin(include_values).sum()
        removed = (~adata.obs[categorical_column].isin(include_values)).sum()
        log.info(f"Found {kept} cells in {include_values} to keep.")
        mask &= adata.obs[categorical_column].isin(include_values)

    elif exclude_values is not None:
        removed = adata.obs[categorical_column].isin(exclude_values).sum()
        kept = (~adata.obs[categorical_column].isin(exclude_values)).sum()
        log.info(f"Found {removed} cells in {exclude_values} to remove.")
        mask &= ~adata.obs[categorical_column].isin(exclude_values)

    adata = adata[mask, :].copy()

    sdata = add_table_layer(
        sdata,
        adata=adata,
        output_layer=output_layer,
        region=process_table_instance.labels_layer,
        overwrite=overwrite,
    )

    if update_shapes_layers:
        for _labels_layer in process_table_instance.labels_layer:
            sdata = filter_shapes_layer(
                sdata,
                table_layer=output_layer,
                labels_layer=_labels_layer,
                prefix_filtered_shapes_layer=prefix_filtered_shapes_layer,
            )

    filtered = start - adata.shape[0]
    log.info(
        f"Removed {filtered} / {start} cells based on '{categorical_column}' "
        f"({'included' if include_values is not None else 'excluded'}: "
        f"{include_values if include_values is not None else exclude_values}) "
        f"and kept {adata.shape[0]}."
    )

    return sdata


def add_table_layer(
    sdata: SpatialData,
    adata: AnnData,
    output_layer: str,
    region: list[str] | None,
    overwrite: bool = False,
):
    """
    Add a table layer to a SpatialData object.

    This function allows you to add a table layer to `sdata`.
    If `sdata` is backed by a zarr store, the resulting table layer will be backed to the zarr store.

    Parameters
    ----------
    sdata
        The SpatialData object to which the new table layer will be added.
    adata
        The AnnData object containing the table data to be added. If `region` is not None, it should contain `_REGION_KEY` and `_INSTANCE_KEY` in adata.obs.
    output_layer
        The name of the output layer where the table data will be stored.
    region
        A list of regions to associate with the table data. Typically this is all unique elements in `adata.obs[_REGION_KEY]`.
    overwrite
        If True, overwrites the output layer if it already exists in `sdata`.

    Returns
    -------
    The updated `sdata` object.
    """
    manager = TableLayerManager()
    sdata = manager.add_table(
        sdata,
        adata=adata,
        output_layer=output_layer,
        region=region,
        overwrite=overwrite,
    )

    return sdata
