from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import zarr
from anndata import AnnData
from scipy import sparse
from spatialdata import SpatialData, read_zarr
from spatialdata._io.format import SpatialDataContainerFormatV01
from spatialdata.models import Labels2DModel, Labels3DModel, TableModel
from spatialdata.transformations import Identity

from harpy._tests.test_table.test_io.test_write import _store_bytes
from harpy.table import add_canonical_centers, validate_table
from harpy.table.canonical_centers import (
    CANONICAL_ALGORITHM_VERSION,
    CANONICAL_OBSM_KEY,
    SPATIAL_COORDINATES_KEY,
    CanonicalRegionMetadata,
    validate_canonical_centers,
)
from harpy.table.canonical_centers._calculation import _calculate_canonical_centers
from harpy.table.canonical_centers._models import _CanonicalRegionBinding
from harpy.table.canonical_centers._schema import (
    _build_canonical_metadata,
    _canonical_metadata_to_storage,
    _parse_canonical_metadata,
)

# Centers of the rows (cell 2, cell 1) of the labels [[1, 1, 2], [1, 2, 2]].
_EXPECTED_2D = [[0.0, 2 / 3, 5 / 3], [0.0, 1 / 3, 1 / 3]]


@pytest.mark.parametrize(
    ("dims", "shape"),
    [(("y", "x"), (4, 5)), (("z", "y", "x"), (3, 4, 5))],
)
def test_canonical_metadata_round_trip_supports_2d_and_3d(
    dims: tuple[str, ...],
    shape: tuple[int, ...],
) -> None:
    metadata = _build_canonical_metadata(
        regions={
            "labels": CanonicalRegionMetadata(
                dims=dims,
                shape=shape,
                dtype="uint32",
                algorithm_version=CANONICAL_ALGORITHM_VERSION,
            )
        },
    )

    storage = _canonical_metadata_to_storage(metadata)

    assert storage["axes"] == ["z", "y", "x"]
    assert storage["regions"]["labels"]["source"]["dims"] == list(dims)
    assert _parse_canonical_metadata(storage) == metadata


def test_calculate_canonical_centers_preserves_requested_instance_order_in_2d() -> None:
    labels = Labels2DModel.parse(
        np.array(
            [
                [1, 1, 0],
                [1, 0, 2],
                [0, 2, 2],
            ],
            dtype=np.uint32,
        ),
        dims=("y", "x"),
        transformations={"sample": Identity()},
    )
    sdata = SpatialData(labels={"labels": labels})
    binding = _CanonicalRegionBinding(
        labels_name="labels",
        row_positions=np.array([0, 1]),
        instance_ids=np.array([2, 1], dtype=np.uint32),
    )

    region = _calculate_canonical_centers(sdata, binding)

    np.testing.assert_allclose(
        region.centers,
        [[0.0, 5 / 3, 5 / 3], [0.0, 1 / 3, 1 / 3]],
    )
    assert region.centers.dtype == np.dtype(np.float64)
    assert region.binding is binding
    assert region.metadata == CanonicalRegionMetadata(
        dims=("y", "x"),
        shape=(3, 3),
        dtype="uint32",
        algorithm_version=CANONICAL_ALGORITHM_VERSION,
    )


def test_calculate_canonical_centers_uses_measured_z_in_3d() -> None:
    labels = Labels3DModel.parse(
        np.ones((2, 2, 2), dtype=np.uint32),
        dims=("z", "y", "x"),
        transformations={"volume": Identity()},
    )
    sdata = SpatialData(labels={"labels": labels})
    binding = _CanonicalRegionBinding(
        labels_name="labels",
        row_positions=np.array([0]),
        instance_ids=np.array([1], dtype=np.uint32),
    )

    region = _calculate_canonical_centers(sdata, binding)

    np.testing.assert_allclose(region.centers, [[0.5, 0.5, 0.5]])


def test_validate_canonical_centers_returns_metadata_or_none() -> None:
    sdata = _canonical_sdata()
    table = sdata.tables["table"]

    metadata = validate_canonical_centers(sdata, table_name="table")

    assert metadata is not None
    assert list(metadata.regions) == ["labels"]

    del table.obsm[CANONICAL_OBSM_KEY]
    del table.uns[SPATIAL_COORDINATES_KEY][CANONICAL_OBSM_KEY]
    assert validate_canonical_centers(sdata, table_name="table") is None


@pytest.mark.parametrize("missing", ["matrix", "metadata"])
def test_validate_canonical_centers_rejects_matrix_metadata_asymmetry(missing: str) -> None:
    sdata = _canonical_sdata()
    table = sdata.tables["table"]
    if missing == "matrix":
        del table.obsm[CANONICAL_OBSM_KEY]
    else:
        del table.uns[SPATIAL_COORDINATES_KEY][CANONICAL_OBSM_KEY]

    with pytest.raises(ValueError, match="both the matrix"):
        validate_canonical_centers(sdata, table_name="table")


def test_validate_canonical_centers_rejects_stale_centers() -> None:
    """The labels element was replaced after its centers were calculated, here by one of another shape."""
    sdata = _canonical_sdata()
    sdata.labels["labels"] = Labels2DModel.parse(
        np.array([[1, 1, 2, 2], [1, 1, 2, 2]], dtype=np.uint32),
        dims=("y", "x"),
        transformations={"sample": Identity()},
    )

    with pytest.raises(ValueError, match="are stale"):
        validate_canonical_centers(sdata, table_name="table")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("dtype", "float64"),
        ("shape", "shape"),
        ("nan", "finite"),
        ("schema", "Rebuild the canonical centers"),
        ("coordinates", "must use z=0"),
    ],
)
def test_validate_canonical_centers_rejects_malformed_components(mutation: str, message: str) -> None:
    sdata = _canonical_sdata()
    table = sdata.tables["table"]
    if mutation == "dtype":
        table.obsm[CANONICAL_OBSM_KEY] = table.obsm[CANONICAL_OBSM_KEY].astype(np.float32)
    elif mutation == "shape":
        table.obsm[CANONICAL_OBSM_KEY] = table.obsm[CANONICAL_OBSM_KEY][:, :2]
    elif mutation == "nan":
        table.obsm[CANONICAL_OBSM_KEY][0, 1] = np.nan
    elif mutation == "schema":
        table.uns[SPATIAL_COORDINATES_KEY][CANONICAL_OBSM_KEY]["schema_version"] = 1
    else:
        table.obsm[CANONICAL_OBSM_KEY][0, 0] = 1.0

    with pytest.raises(ValueError, match=message):
        validate_canonical_centers(sdata, table_name="table")


def test_validate_table_accepts_reordered_and_region_subset_tables() -> None:
    """Row subsets and reorders keep canonical centers valid.

    Per-cell QC before ``hp.tb.concat`` relies on this
    (https://github.com/saeyslab/harpy/issues/279), so no check may tie a region
    to its rows again. A subset to fewer regions keeps the canonical-center
    metadata of the regions it dropped; validation ignores it.
    """
    sdata = _multi_region_sdata()
    add_canonical_centers(sdata, table_name="table")
    table = sdata.tables["table"]

    sdata.tables["table"] = table[[2, 0]].copy()
    validate_table(sdata, "table")

    subset = table[(table.obs["region"] == "labels_a").to_numpy()].copy()
    # AnnData drops unused categories when subsetting, so the region column no
    # longer lists labels_b. Validation compares only the observed regions and
    # would not notice if AnnData stopped doing so; this assertion would.
    assert list(subset.obs["region"].cat.categories) == ["labels_a"]
    subset.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY] = ["labels_a"]
    sdata.tables["table"] = subset
    validate_table(sdata, "table")
    assert set(subset.uns[SPATIAL_COORDINATES_KEY][CANONICAL_OBSM_KEY]["regions"]) == {"labels_a", "labels_b"}
    assert list(validate_canonical_centers(sdata, table_name="table").regions) == ["labels_a"]


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_add_canonical_centers_round_trips_through_the_store(tmp_path, zarr_format) -> None:
    """Only the two canonical components change in the store, and they pass validation when reopened.

    The metadata goes to its nested ``uns`` path, so other entries under
    ``uns["spatial_coordinates"]`` stay.
    """
    sdata = _backed_external_sdata(tmp_path, zarr_format=zarr_format)
    table_path = sdata.path / "tables" / "table"
    before = _store_bytes(table_path)

    add_canonical_centers(sdata, table_name="table")

    updated_paths = (f"obsm/{CANONICAL_OBSM_KEY}/", f"uns/{SPATIAL_COORDINATES_KEY}/{CANONICAL_OBSM_KEY}/")
    after = _store_bytes(table_path)
    assert {key: value for key, value in after.items() if not key.startswith(updated_paths)} == before
    validate_table(sdata, "table")

    reopened = read_zarr(sdata.path)
    validate_table(reopened, "table")
    table = reopened.tables["table"]
    np.testing.assert_allclose(table.obsm[CANONICAL_OBSM_KEY], _EXPECTED_2D)
    expected_metadata = _canonical_sdata().tables["table"].uns[SPATIAL_COORDINATES_KEY][CANONICAL_OBSM_KEY]
    assert _parse_canonical_metadata(table.uns[SPATIAL_COORDINATES_KEY][CANONICAL_OBSM_KEY]) == (
        _parse_canonical_metadata(expected_metadata)
    )
    assert list(table.uns[SPATIAL_COORDINATES_KEY]["viewer"]["axes"]) == ["y", "x"]


def test_add_canonical_centers_on_unbacked_sdata() -> None:
    """Unbacked SpatialData is updated in memory, and ``overwrite`` is ignored."""
    sdata = _external_sdata()

    add_canonical_centers(sdata, table_name="table")

    table = sdata.tables["table"]
    assert isinstance(table.obsm[CANONICAL_OBSM_KEY], np.ndarray)
    np.testing.assert_allclose(table.obsm[CANONICAL_OBSM_KEY], _EXPECTED_2D)
    assert table.uns[SPATIAL_COORDINATES_KEY]["viewer"] == {"axes": ["y", "x"]}
    validate_table(sdata, "table")

    table.obsm[CANONICAL_OBSM_KEY] = np.zeros((table.n_obs, 3))
    add_canonical_centers(sdata, table_name="table")

    np.testing.assert_allclose(table.obsm[CANONICAL_OBSM_KEY], _EXPECTED_2D)


@pytest.mark.parametrize("backed", [False, True])
def test_add_canonical_centers_aligns_overlapping_instance_ids_across_regions(tmp_path, backed) -> None:
    sdata = _multi_region_sdata()
    if backed:
        sdata = _write_and_read(sdata, tmp_path / "multi.zarr")

    add_canonical_centers(sdata, table_name="table")

    np.testing.assert_allclose(
        sdata.tables["table"].obsm[CANONICAL_OBSM_KEY],
        [[0.0, 1.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.5]],
    )
    validate_table(sdata, "table")


@pytest.mark.parametrize("backed", [False, True])
def test_add_canonical_centers_supports_a_3d_labels_source(tmp_path, backed) -> None:
    labels = Labels3DModel.parse(
        np.ones((2, 2, 2), dtype=np.uint32),
        dims=("z", "y", "x"),
        transformations={"global": Identity()},
    )
    table = TableModel.parse(
        AnnData(
            X=np.ones((1, 1)),
            obs=pd.DataFrame({"region": ["labels"], "cell_ID": np.array([1], dtype=np.uint32)}, index=["cell-1"]),
        ),
        region_key="region",
        region="labels",
        instance_key="cell_ID",
    )
    sdata = SpatialData(labels={"labels": labels}, tables={"table": table})
    if backed:
        sdata = _write_and_read(sdata, tmp_path / "three-dimensional.zarr")

    add_canonical_centers(sdata, table_name="table")

    np.testing.assert_allclose(sdata.tables["table"].obsm[CANONICAL_OBSM_KEY], [[0.5, 0.5, 0.5]])
    validate_table(sdata, "table")


@pytest.mark.parametrize(
    ("instance_id", "message"),
    [(2, "no finite center"), (0, "labels element `labels` have a non-positive instance ID \\(0\\)")],
)
def test_add_canonical_centers_rejects_a_row_without_an_instance_in_its_labels(tmp_path, instance_id, message) -> None:
    """A table row without an instance in its labels element raises, before anything is written.

    Its instance ID either has no pixels in the labels element or is background
    ``0``. All regions are calculated before the single write, so the table gets no
    canonical centers at all, neither in memory nor in the store.
    """
    labels = Labels2DModel.parse(
        np.array([[1]], dtype=np.uint32),
        dims=("y", "x"),
        transformations={"global": Identity()},
    )
    table = TableModel.parse(
        AnnData(
            X=np.ones((1, 1)),
            obs=pd.DataFrame(
                {"region": ["labels"], "cell_ID": np.array([instance_id], dtype=np.uint32)}, index=["cell"]
            ),
        ),
        region_key="region",
        region="labels",
        instance_key="cell_ID",
    )
    output = tmp_path / "missing-instance.zarr"
    sdata = _write_and_read(SpatialData(labels={"labels": labels}, tables={"table": table}), output)

    with pytest.raises(ValueError, match=message):
        add_canonical_centers(sdata, table_name="table")

    assert CANONICAL_OBSM_KEY not in sdata.tables["table"].obsm
    assert SPATIAL_COORDINATES_KEY not in sdata.tables["table"].uns
    reopened = read_zarr(output)
    assert CANONICAL_OBSM_KEY not in reopened.tables["table"].obsm
    assert SPATIAL_COORDINATES_KEY not in reopened.tables["table"].uns


def test_add_canonical_centers_overwrite_repairs_incomplete_centers(tmp_path) -> None:
    """``overwrite=True`` replaces canonical centers that fail validation, here a matrix without its metadata.

    ``add_canonical_centers`` does not validate existing centers, so it can repair them;
    ``validate_table`` rejects the table before and accepts it after.
    """
    sdata = _backed_external_sdata(tmp_path)
    add_canonical_centers(sdata, table_name="table")

    with pytest.raises(FileExistsError):
        add_canonical_centers(sdata, table_name="table")

    # Leave only the matrix, in memory and in the store.
    del sdata.tables["table"].uns[SPATIAL_COORDINATES_KEY][CANONICAL_OBSM_KEY]
    root = zarr.open_group(store=str(sdata.path), mode="r+", use_consolidated=False)
    del root["tables"]["table"]["uns"][SPATIAL_COORDINATES_KEY][CANONICAL_OBSM_KEY]
    sdata.write_consolidated_metadata()
    with pytest.raises(ValueError, match="both the matrix"):
        validate_table(sdata, "table")

    add_canonical_centers(sdata, table_name="table", overwrite=True)

    validate_table(sdata, "table")
    validate_table(read_zarr(sdata.path), "table")


@pytest.mark.parametrize(
    ("problem", "message"),
    [
        ("non_categorical_region", "must be of type `categorical`"),
        ("feature_matrix", "declares 2 feature columns"),
        ("region_without_rows", "Regions.*do not match"),
    ],
)
def test_add_canonical_centers_validates_the_table_first(tmp_path, problem, message) -> None:
    """A table edited in place into an invalid one raises before anything is written, even with ``overwrite=True``."""
    sdata = _backed_external_sdata(tmp_path)
    table = sdata.tables["table"]
    if problem == "non_categorical_region":
        table.obs["region"] = table.obs["region"].astype("string")
    elif problem == "feature_matrix":
        table.obsm["negative"] = np.ones((table.n_obs, 1))
        table.uns["feature_matrices"] = {
            "negative": {
                "schema_version": 1,
                "source_kind": "harpy_aggregate_points",
                "feature_columns": ["Negative01", "Negative02"],
            }
        }
    else:
        sdata.labels["empty_labels"] = Labels2DModel.parse(
            np.array([[1]], dtype=np.uint32),
            dims=("y", "x"),
            transformations={"global": Identity()},
        )
        table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY] = ["labels", "empty_labels"]
    before = _store_bytes(sdata.path)

    with pytest.raises(ValueError, match=message):
        add_canonical_centers(sdata, table_name="table", overwrite=True)

    assert CANONICAL_OBSM_KEY not in table.obsm
    assert CANONICAL_OBSM_KEY not in table.uns[SPATIAL_COORDINATES_KEY]
    assert _store_bytes(sdata.path) == before


def test_canonical_centers_reject_in_memory_observations_reordered_relative_to_storage(tmp_path):
    """Centers calculated in live-table order must never be written onto different stored rows.

    A regression guard only: nothing here is specific to ``add_canonical_centers``.
    ``add_table_components`` compares the attached ``obs`` with the store for every
    obs-aligned write, and ``test_backed_updates_reject_live_axis_order_mismatch``
    tests that. This test fails if the centers are ever written another way.
    """
    sdata = _backed_external_sdata(tmp_path)
    table = sdata.tables["table"]
    table.obs = table.obs.iloc[::-1].copy()
    before = _store_bytes(sdata.path)
    with pytest.raises(ValueError, match="differ in order"):
        add_canonical_centers(sdata, table_name="table")
    assert _store_bytes(sdata.path) == before
    assert CANONICAL_OBSM_KEY not in table.obsm


def _canonical_sdata() -> SpatialData:
    labels = Labels2DModel.parse(
        np.array([[1, 1, 2], [1, 2, 2]], dtype=np.uint32),
        dims=("y", "x"),
        transformations={"sample": Identity()},
    )
    metadata = _build_canonical_metadata(
        regions={
            "labels": CanonicalRegionMetadata(
                dims=("y", "x"),
                shape=(2, 3),
                dtype="uint32",
                algorithm_version=CANONICAL_ALGORITHM_VERSION,
            )
        },
    )
    table = TableModel.parse(
        AnnData(
            X=np.ones((2, 1)),
            obs=pd.DataFrame(
                {"region": ["labels", "labels"], "cell_ID": [2, 1]},
                index=["row-2", "row-1"],
            ),
            obsm={CANONICAL_OBSM_KEY: np.array(_EXPECTED_2D)},
            uns={SPATIAL_COORDINATES_KEY: {CANONICAL_OBSM_KEY: _canonical_metadata_to_storage(metadata)}},
        ),
        region_key="region",
        region="labels",
        instance_key="cell_ID",
    )
    return SpatialData(labels={"labels": labels}, tables={"table": table})


def _multi_region_sdata() -> SpatialData:
    """Two labels elements sharing instance ID 1, with rows in neither region's order."""
    labels_a = Labels2DModel.parse(
        np.array([[1, 1], [2, 0]], dtype=np.uint32),
        dims=("y", "x"),
        transformations={"global": Identity()},
    )
    labels_b = Labels2DModel.parse(
        np.array([[0, 0], [0, 1]], dtype=np.uint32),
        dims=("y", "x"),
        transformations={"global": Identity()},
    )
    table = TableModel.parse(
        AnnData(
            X=np.ones((3, 1)),
            obs=pd.DataFrame(
                {
                    "region": ["labels_b", "labels_a", "labels_a"],
                    "cell_ID": np.array([1, 2, 1], dtype=np.uint32),
                },
                index=["b-1", "a-2", "a-1"],
            ),
        ),
        region_key="region",
        region=["labels_a", "labels_b"],
        instance_key="cell_ID",
    )
    return SpatialData(labels={"labels_a": labels_a, "labels_b": labels_b}, tables={"table": table})


def _external_sdata() -> SpatialData:
    labels = Labels2DModel.parse(
        np.array([[1, 1, 2], [1, 2, 2]], dtype=np.uint32),
        dims=("y", "x"),
        transformations={"global": Identity()},
    )
    table = AnnData(
        X=np.array([[1, 2], [3, 4]], dtype=np.float32),
        obs=pd.DataFrame(
            {"region": ["labels", "labels"], "cell_ID": np.array([2, 1], dtype=np.uint32)},
            index=["row-2", "row-1"],
        ),
        var=pd.DataFrame(index=["GeneA", "GeneB"]),
        layers={"counts": np.array([[5, 6], [7, 8]], dtype=np.uint32)},
        obsm={"other_coordinates": np.array([[10.0, 20.0], [30.0, 40.0]])},
        varm={"loadings": np.array([[1.0], [2.0]])},
        obsp={"connectivities": sparse.eye(2, format="csr")},
        varp={"correlations": np.array([[1.0, 0.5], [0.5, 1.0]])},
        uns={SPATIAL_COORDINATES_KEY: {"viewer": {"axes": ["y", "x"]}}, "other": {"keep": True}},
    )
    table.raw = table.copy()
    table = TableModel.parse(
        table,
        region_key="region",
        region="labels",
        instance_key="cell_ID",
    )
    return SpatialData(labels={"labels": labels}, tables={"table": table})


def _backed_external_sdata(tmp_path, *, zarr_format=3) -> SpatialData:
    return _write_and_read(_external_sdata(), tmp_path / "external.zarr", zarr_format=zarr_format)


def _write_and_read(sdata: SpatialData, output: Path, *, zarr_format=3) -> SpatialData:
    sdata.write(output, sdata_formats=SpatialDataContainerFormatV01() if zarr_format == 2 else None)
    return read_zarr(output)
