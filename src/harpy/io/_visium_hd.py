from __future__ import annotations

from pathlib import Path

from spatialdata import SpatialData, read_zarr
from spatialdata.models import TableModel
from spatialdata_io._constants._constants import VisiumHDKeys
from spatialdata_io.readers.visium_hd import visium_hd as sdata_visium_hd

from harpy.utils._keys import _INSTANCE_KEY, _REGION_KEY


def visium_hd(
    path: str | Path,
    dataset_id: str | None = None,
    bin_size: int | list[int] | None = None,
    bins_as_squares: bool = True,
    output: str | Path | None = None,
) -> SpatialData:
    """
    Read *10x Genomics* Visium HD formatted dataset.

    Wrapper around `spatialdata.io.readers.visium_hd.visium_hd`, but with the resulting table annotated by a labels layer.

    .. seealso::

        - `Space Ranger output <https://support.10xgenomics.com/spatial-gene-expression/software/pipelines/latest/output/overview>`_.

    Parameters
    ----------
    path
        Path to directory containing the *10x Genomics* Visium HD output.
    dataset_id
        Unique identifier of the dataset. If `None`, it tries to infer it from the file name of the feature slice file.
    bin_size
        When specified, load the data of a specific bin size, or a list of bin sizes. By default, it loads all the
        available bin sizes.
    bins_as_squares
        If `True`, the bins are represented as squares. If `False`, the bins are represented as circles. For a correct
        visualization one should use squares.
    output
        The path where the resulting `SpatialData` object will be backed. If None, it will not be backed to a zarr store.
    """
    sdata = sdata_visium_hd(
        path=path,
        bin_size=bin_size,
        dataset_id=dataset_id,
        filtered_counts_file=True,
        annotate_table_by_labels=True,
        bins_as_squares=bins_as_squares,
    )

    for tables_layer in [*sdata.tables]:
        adata = sdata[tables_layer]
        adata.var_names_make_unique()
        adata.X = adata.X.tocsc()

        adata.obs.rename(
            columns={VisiumHDKeys.REGION_KEY: _REGION_KEY, VisiumHDKeys.INSTANCE_KEY: _INSTANCE_KEY}, inplace=True
        )
        adata.uns.pop(TableModel.ATTRS_KEY)
        adata = TableModel.parse(
            adata,
            region_key=_REGION_KEY,
            region=adata.obs[_REGION_KEY].cat.categories.to_list(),
            instance_key=_INSTANCE_KEY,
        )

        del sdata[tables_layer]

        sdata[tables_layer] = adata

    for shapes_layer in [*sdata.shapes]:
        sdata[shapes_layer].index.name = _INSTANCE_KEY

    if output is not None:
        sdata.write(output)
        sdata = read_zarr(sdata.path)

    return sdata
