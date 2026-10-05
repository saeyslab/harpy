import geopandas as gpd
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse
from shapely.geometry import Point, box
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, Labels2DModel, ShapesModel, TableModel
from spatialdata.transformations import Identity, Scale, get_transformation
from spatialdata_io._constants._constants import VisiumHDKeys

import harpy.io._visium as visium_module
import harpy.io._visium_hd as visium_hd_module

# Counts as spatialdata-io returns them: scanpy's read_10x_h5 gives CSR.
_COUNTS = sparse.csr_matrix(np.array([[1, 0, 2], [0, 3, 0], [4, 0, 0], [0, 0, 5]], dtype=np.float32))
_GENES = pd.DataFrame(index=["GeneA", "GeneB", "GeneC"])


def _fake_visium_hd(**kwargs):
    """A minimal stand-in for spatialdata-io's Visium HD reader, with its table annotated by labels.

    As spatialdata-io does since 0.2.0, the full-resolution and CytAssist images,
    when requested, are registered in the dataset's coordinate system.
    """
    dataset_id = kwargs["dataset_id"]
    images = {}
    if kwargs.get("fullres_image_file") is not None:
        images[f"{dataset_id}_full_image"] = Image2DModel.parse(
            np.zeros((3, 4, 4)), dims=("c", "y", "x"), transformations={dataset_id: Identity()}
        )
    if kwargs.get("load_all_images"):
        images[f"{dataset_id}_cytassist_image"] = Image2DModel.parse(
            np.zeros((3, 4, 4)), dims=("c", "y", "x"), transformations={dataset_id: Scale([2.0, 2.0], axes=("y", "x"))}
        )
    location_ids = np.array([10, 11, 12, 13])
    shapes = ShapesModel.parse(gpd.GeoDataFrame(geometry=[box(i, 0, i + 1, 1) for i in range(4)], index=location_ids))
    labels = Labels2DModel.parse(np.array([[1, 2, 3, 4]], dtype=np.uint32), dims=("y", "x"))
    obs = pd.DataFrame(
        {
            VisiumHDKeys.REGION_KEY: pd.Categorical(["square_016um_labels"] * 4),
            VisiumHDKeys.INSTANCE_KEY: location_ids,
            "labels_id": [1, 2, 3, 4],
        },
        index=[f"bin{i}" for i in location_ids],
    )
    table = TableModel.parse(
        AnnData(X=_COUNTS.copy(), obs=obs, var=_GENES.copy()),
        region="square_016um_labels",
        region_key=VisiumHDKeys.REGION_KEY,
        instance_key="labels_id",
    )
    return SpatialData(
        images=images,
        shapes={"Visium_HD_square_016um": shapes},
        labels={"square_016um_labels": labels},
        tables={"square_016um": table},
    )


def _fake_visium(**kwargs):
    """A minimal stand-in for spatialdata-io's Visium reader: circular spots and a table annotating them."""
    dataset_id = kwargs["dataset_id"]
    spot_ids = np.arange(4)
    shapes = ShapesModel.parse(
        gpd.GeoDataFrame({"radius": np.full(4, 2.0)}, geometry=[Point(5 + 10 * i, 5) for i in range(4)], index=spot_ids)
    )
    obs = pd.DataFrame(
        {"region": pd.Categorical([dataset_id] * 4), "spot_id": spot_ids}, index=[f"spot{i}" for i in spot_ids]
    )
    table = TableModel.parse(
        AnnData(X=_COUNTS.copy(), obs=obs, var=_GENES.copy()),
        region=dataset_id,
        region_key="region",
        instance_key="spot_id",
    )
    return SpatialData(shapes={dataset_id: shapes}, tables={"table": table})


def test_visium_hd_reader_keeps_the_csr_counts(monkeypatch):
    """The Visium HD reader keeps spatialdata-io's CSR counts, the layout lazy scanpy and rapids-singlecell need."""
    monkeypatch.setattr(visium_hd_module, "sdata_visium_hd", _fake_visium_hd)

    sdata = visium_hd_module.visium_hd(path="unused", dataset_id="Visium_HD")

    matrix = sdata.tables["square_016um"].X
    assert isinstance(matrix, sparse.csr_matrix)
    np.testing.assert_array_equal(matrix.toarray(), _COUNTS.toarray())


def test_visium_hd_reader_keeps_images_in_the_dataset_coordinate_system(monkeypatch):
    """The full-resolution and CytAssist images keep the transformations spatialdata-io gave them.

    spatialdata-io registers both images in the dataset's coordinate system,
    without a "global" one, so the reader must not look for a "global"
    transformation to move.
    """
    monkeypatch.setattr(visium_hd_module, "sdata_visium_hd", _fake_visium_hd)

    sdata = visium_hd_module.visium_hd(
        path="unused", dataset_id="Visium_HD", fullres_image_file="unused.tif", load_all_images=True
    )

    assert get_transformation(sdata.images["Visium_HD_full_image"], get_all=True) == {"Visium_HD": Identity()}
    assert get_transformation(sdata.images["Visium_HD_cytassist_image"], get_all=True) == {
        "Visium_HD": Scale([2.0, 2.0], axes=("y", "x"))
    }


def test_visium_reader_keeps_the_csr_counts(monkeypatch):
    """The Visium reader keeps spatialdata-io's CSR counts, the layout lazy scanpy and rapids-singlecell need."""
    monkeypatch.setattr(visium_module, "sdata_visium", _fake_visium)

    sdata = visium_module.visium(path="unused", dataset_id="visium")

    matrix = sdata.tables["table"].X
    assert isinstance(matrix, sparse.csr_matrix)
    np.testing.assert_array_equal(matrix.toarray(), _COUNTS.toarray())
