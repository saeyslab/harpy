import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse
from shapely.geometry import box
from spatialdata import SpatialData
from spatialdata._io.format import SpatialDataContainerFormatV01
from spatialdata.models import Image2DModel, Labels2DModel, PointsModel, ShapesModel, TableModel
from spatialdata.transformations import Translation


@pytest.fixture(params=[2, 3])
def spatial_store(tmp_path, request):
    """A small mixed-element store with dense, CSR and CSC tables and no Harpy metadata."""
    transforms = {"sample": Translation([10, 20], axes=("x", "y"))}
    obs = pd.DataFrame({"region": pd.Categorical(["cells", "cells"]), "instance": [1, 2]}, index=["c1", "c2"])
    tables = {}
    for kind in ("dense", "csr", "csc"):
        matrix = np.array([[1, 0], [2, 3]], dtype=np.float32)
        if kind != "dense":
            matrix = getattr(sparse, f"{kind}_matrix")(matrix)
        tables[kind] = TableModel.parse(
            AnnData(
                X=matrix,
                obs=obs.copy(),
                var=pd.DataFrame(index=["A", "B"]),
                layers={"counts": matrix.copy()},
                obsm={"embedding": np.array([[0, 1], [2, 3]])},
                uns={"analysis": {"threshold": 3}},
            ),
            region="cells",
            region_key="region",
            instance_key="instance",
        )
    sdata = SpatialData(
        images={
            "image": Image2DModel.parse(
                np.arange(4, dtype=np.uint16).reshape(1, 2, 2),
                dims=("c", "y", "x"),
                c_coords=["DAPI"],
                transformations=transforms,
            )
        },
        labels={"cells": Labels2DModel.parse(np.array([[1, 1], [2, 2]]), transformations=transforms)},
        points={
            "calls": PointsModel.parse(
                pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0], "gene": ["A", "B"]}),
                transformations=transforms,
            )
        },
        shapes={
            "outlines": ShapesModel.parse(gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)]), transformations=transforms)
        },
        tables=tables,
        attrs={"note": {"sample": "example"}},
    )
    path = tmp_path / "sdata.zarr"
    options = {"sdata_formats": SpatialDataContainerFormatV01()} if request.param == 2 else {}
    sdata.write(path, **options)
    return path
