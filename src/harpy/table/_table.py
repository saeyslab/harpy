from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from anndata import AnnData
from loguru import logger as log
from spatialdata import SpatialData
from spatialdata.models import TableModel

from harpy.shape._shape import filter_shapes
from harpy.table._manager import TableLayerManager
from harpy.utils._keys import _CELLSIZE_KEY, _INSTANCE_KEY, _REGION_KEY


class ProcessTable:
    def __init__(
        self,
        sdata: SpatialData,
        table_name: str,
        labels_name: str
        | Iterable[str]
        | None = None,  # TODO: replace with region, and also support shapes and points
    ):
        """
        Base class for implementation of processing on tables.

        Parameters
        ----------
        sdata: SpatialData
            The SpatialData object containing spatial data.
        table_name: str
            The table element to use.
        labels_name : str or Iterable[str] or None
            The labels element(s) to use.
        """
        if sdata.tables == {}:
            raise ValueError(
                "Provided SpatialData object 'sdata' does not contain any 'tables'. "
                "Please create tables via e.g. 'harpy.tb.allocation' or 'harpy.tb.allocation_intensity' functions."
            )

        if labels_name is not None:
            if sdata.labels == {}:
                raise ValueError(
                    "Provided SpatialData object 'sdata' does not contain 'labels'. "
                    "Please create a labels element via e.g. 'harpy.im.segment'."
                )
            labels_name = (
                list(labels_name)
                if isinstance(labels_name, Iterable) and not isinstance(labels_name, str)
                else [labels_name]
            )

        self.sdata = sdata
        self.labels_name = labels_name
        self.table_name = table_name
        # Do not pass it here, get it straight from the anndata
        self._validated_table_layer()
        self.instance_key = None
        self.region_key = None
        if TableModel.ATTRS_KEY in self.sdata.tables[self.table_name].uns:
            self.instance_key = self.sdata.tables[self.table_name].uns[TableModel.ATTRS_KEY][TableModel.INSTANCE_KEY]
            self.region_key = self.sdata.tables[self.table_name].uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY]
        if self.labels_name is not None:
            self._validate_layer(layer_list=self.labels_name)
        # if self.labels_name is None:
        #    self._validate()

    def _validate_layer(self, layer_list, layer_type="labels"):
        """Generic element validation helper to reduce code duplication."""
        for _layer in layer_list:
            if _layer not in [*getattr(self.sdata, layer_type)]:
                raise ValueError(f"{layer_type} element '{_layer}' not in 'sdata.{layer_type}'.")
            if (
                _layer not in self.sdata.tables[self.table_name].obs[self.region_key].cat.categories
                or _layer not in self.sdata.tables[self.table_name].uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY]
            ):
                raise ValueError(f"{layer_type} element '{_layer}' not annotated by table element '{self.table_name}'.")
            # Check for uniqueness of instance keys
            assert (
                self.sdata.tables[self.table_name]
                .obs[self.sdata.tables[self.table_name].obs[self.region_key] == _layer][self.instance_key]
                .is_unique
            ), (
                f"'{self.instance_key}' is not unique for '{self.region_key}' == '{_layer}'. Please make sure these are unique."
            )

    def _validate(self):
        assert self.sdata.tables[self.table_name].obs[self.instance_key].is_unique, (
            f"'{self.instance_key}' is not unique. Please make sure these are unique, or specify a 'labels_name' via '{self.region_key}'."
        )

    def _validated_table_layer(self):
        """Validate if the specified table element exists in the SpatialData object."""
        if self.table_name not in [*self.sdata.tables]:
            raise ValueError(f"table element '{self.table_name}' not in 'sdata.tables'.")

    def _get_adata(
        self, index_names_var: Iterable[str] | None = None, index_positions_var: Iterable[int] | None = None
    ) -> AnnData:
        """Preprocess the data by filtering based on the table element and setting spatialdata attributes."""
        if self.labels_name is not None:
            adata = self.sdata.tables[self.table_name][
                self.sdata.tables[self.table_name].obs[self.region_key].isin(self.labels_name)
            ]
        else:
            adata = self.sdata.tables[self.table_name]
        if index_names_var is not None or index_positions_var is not None:
            adata = self._subset_adata_var(
                adata, index_names_var=index_names_var, index_positions_var=index_positions_var
            )
        # TODO: `adata.copy()` can be expensive when callers only need to update `.obs`.
        # Consider supporting a lighter-weight path in the future for table operations
        # that preserve the selected rows and only modify metadata columns.
        adata = adata.copy()
        if self.labels_name is not None:
            adata.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY] = self.labels_name

        return adata

    @staticmethod
    def _type_check_before_pca(adata: AnnData):
        # type check because pca raises error when adata.X is sparse and of dtype int.
        if np.issubdtype(adata.X.dtype, np.integer):
            raise ValueError(
                f"Data matrix of AnnData table is of type '{adata.X.dtype}', "
                "which indicates no preprocessing is performed. "
                "Please consider preprocessing the data first before calculating pca ('scanpy.tl.pca') or calculating neighborhood grap ('scanpy.pp.neighbors'), "
                "e.g. with 'scanpy.pp.scale'."
                ""
            )

    @staticmethod
    def _subset_adata_var(
        adata: AnnData, index_names_var: Iterable[str] | None = None, index_positions_var: Iterable[int] | None = None
    ) -> AnnData:
        """
        Subsets an :class:`~anndata.AnnData` object by index names or index positions of `adata.var`.

        Parameters
        ----------
        adata: :class:`~anndata.AnnData object.
        index_names_var: List of index names to subset. If `None`, the function will use index_positions.
        index_positions_var: List of integer positions to subset. Used if `index_names_var` is `None`.

        Returns
        -------
        Subsetted :class:`~anndata.AnnData` object.
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
    labels_name: list[str],
    table_name: str,
    output_table_name: str,
    celltype_correction_dict: dict[str, tuple[float, float]],
    overwrite: bool = False,
) -> SpatialData:
    """
    Correct celltype expression in `sdata.tables[table_name]` using `celltype_correction_dict`.

    Corrects celltype scores (found in `.obs` attribute of the :class:`~anndata.AnnData` table) that are higher expessed by dividing them by a value if they exceed a certain threshold.
    The `celltype_correction_dict` has as keys the celltypes that should be corrected and as values the threshold and the divider.

    .. deprecated:: 0.3.0
       `harpy.tb.correct_marker_genes` is deprecated and may be removed in a future release.

    Parameters
    ----------
    sdata
        The :class:`~spatialdata.SpatialData` object.
    labels_name
        The labels element(s) of `sdata` used to select the cells via the region key in `sdata.tables[table_name].obs`.
        Note that if `output_table_name` is equal to `table_name` and overwrite is True,
        cells in `sdata.tables[table_name]` linked to other `labels_name` (via the region key), will be removed from `sdata.tables[table_name]`
        (also from the backing Zarr store if it is backed).
    table_name
        The table element in `sdata`.
    output_table_name
        The output table element in `sdata`.
    celltype_correction_dict
        The `celltype_correction_dict` has as keys the celltypes that should be corrected and as values the threshold and the divider.

    Returns
    -------
    The updated :class:`~spatialdata.SpatialData` object.

    See Also
    --------
    harpy.tb.score_genes : score genes using :func:`~scanpy.tl.score_genes`.
    """
    process_table_instance = ProcessTable(sdata, labels_name=labels_name, table_name=table_name)
    adata = process_table_instance._get_adata()
    # Correct for all the genes
    for celltype, values in celltype_correction_dict.items():
        if celltype not in adata.obs.columns:
            log.info(
                f"Cell type '{celltype}' not in .obs of AnnData object. Skipping. Please first calculate gene expression for this cell type."
            )
            continue
        adata.obs[celltype] = np.where(
            adata.obs[celltype] > values[0],
            adata.obs[celltype] / values[1],
            adata.obs[celltype],
        )

    sdata = add_table(
        sdata,
        adata=adata,
        output_table_name=output_table_name,
        region=process_table_instance.labels_name,
        instance_key=process_table_instance.instance_key,
        region_key=process_table_instance.region_key,
        overwrite=overwrite,
    )

    return sdata


def filter_on_size(
    sdata: SpatialData,
    labels_name: list[str],
    table_name: str,
    output_table_name: str,
    min_size: int = 100,
    max_size: int = 100000,
    update_shapes_elements: bool = True,
    instance_size_key: str = _CELLSIZE_KEY,
    overwrite: bool = False,
) -> SpatialData:
    """Returns the updated SpatialData object.

    All cells with a size outside of the min and max size range are removed using the `instance_size_key` in `.obs`. Run e.g. :func:`~harpy.tb.preprocess_transcriptomics` or :func:`~harpy.tb.preprocess_proteomics` to obtain cell sizes.

    Parameters
    ----------
    sdata
        The SpatialData object.
    labels_name
        The labels element(s) of `sdata` used to select the cells via the region key in `sdata.tables[table_name].obs`.
        Note that if `output_table_name` is equal to `table_name` and overwrite is True,
        cells in `sdata.tables[table_name]` linked to other `labels_name` (via the region key), will be removed from `sdata.tables[table_name]`
        (also from the backing zarr store if it is backed).
    table_name
        The table element in `sdata`.
    output_table_name
        The output table element in `sdata`.
    min_size
        minimum size in pixels.
    max_size
        maximum size in pixels.
    update_shapes_elements
        Whether to filter the shapes elements associated with `labels_name`.
        If set to `True`, cells that do not appear in resulting `output_table_name` (with region key equal to `labels_name`) will be removed from the shapes elements (via instance key) in the `sdata` object.
        Filtered shapes will be added to `sdata` with prefix 'filtered_size'.
        This parameter is deprecated, and will be removed in a future version.
    instance_size_key
        Column in `sdata.tables[table_name].obs` containing instance sizes.
    overwrite
        If True, overwrites the `output_table_name` if it already exists in `sdata`.

    Returns
    -------
    The updated SpatialData object.
    """
    process_table_instance = ProcessTable(sdata, labels_name=labels_name, table_name=table_name)
    adata = process_table_instance._get_adata()
    start = adata.shape[0]

    # Filter cells based on size and distance
    # need to do the copy because we pop the spatialdata_attrs in add_table, otherwise it would not be updated inplace
    adata = adata[adata.obs[instance_size_key] < max_size, :].copy()
    adata = adata[adata.obs[instance_size_key] > min_size, :].copy()

    sdata = add_table(
        sdata,
        adata=adata,
        output_table_name=output_table_name,
        region=process_table_instance.labels_name,
        instance_key=process_table_instance.instance_key,
        region_key=process_table_instance.region_key,
        overwrite=overwrite,
    )

    if update_shapes_elements:
        for _labels_layer in process_table_instance.labels_name:
            sdata = filter_shapes(
                sdata,
                table_name=output_table_name,
                labels_name=_labels_layer,
                prefix_filtered_shapes_name="filtered_size",
            )

    filtered = start - adata.shape[0]
    log.info(f"{filtered} cells were filtered out based on size.")

    return sdata


def add_table(
    sdata: SpatialData,
    adata: AnnData,
    output_table_name: str,
    region: list[str] | None,
    instance_key: str = _INSTANCE_KEY,
    region_key: str = _REGION_KEY,
    overwrite: bool = False,
) -> SpatialData:
    """
    Add an :class:`~anndata.AnnData` object as a table element to a :class:`~spatialdata.SpatialData` object.

    This function stores the provided :class:`~anndata.AnnData` object in ``sdata.tables[output_table_name]``.
    When ``region`` is provided, the table is parsed as a SpatialData table and linked to one or more
    spatial elements via ``region_key`` and ``instance_key``. If ``region`` is ``None``, the AnnData
    object is added as a regular table without region annotations.

    If ``sdata`` is backed by a zarr store, the resulting table element is also written to that store.

    Parameters
    ----------
    sdata
        The :class:`~spatialdata.SpatialData` object to which the new table element will be added.
    adata
        The :class:`~anndata.AnnData` object to add. If ``region`` is not ``None``, ``adata.obs``
        must contain the columns specified by ``region_key`` and ``instance_key``.
    output_table_name
        Name of the output table element in ``sdata.tables``.
    region
        Regions annotated by the table. Typically this is the list of unique values in
        ``adata.obs[region_key]``. Set to ``None`` if the table should not annotate any spatial element.
    instance_key
        Name of the column in ``adata.obs`` that stores instance ids.
        Ignored if ``region`` is ``None``.
    region_key
        Name of the column in ``adata.obs`` that stores the region labels annotated by the table.
        Ignored if ``region`` is ``None``.
    overwrite
        If ``True``, overwrite ``output_table_name`` if it already exists in ``sdata``.

    Returns
    -------
    The updated :class:`~spatialdata.SpatialData` object.
    """
    manager = TableLayerManager()
    sdata = manager.add_table(
        sdata,
        adata=adata,
        output_table_name=output_table_name,
        region=region,
        instance_key=instance_key,
        region_key=region_key,
        overwrite=overwrite,
    )

    return sdata
