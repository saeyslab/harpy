from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import zarr
from anndata import AnnData
from scipy import sparse
from spatialdata import SpatialData, read_zarr
from spatialdata.models import Labels2DModel, Labels3DModel, TableModel
from spatialdata.transformations import Identity

from harpy.table import add_canonical_centers, validate_table
from harpy.table.canonical_centers import (
    CANONICAL_ALGORITHM_VERSION,
    CANONICAL_OBSM_KEY,
    SPATIAL_COORDINATES_KEY,
    CanonicalCacheReport,
    CanonicalCacheState,
    CanonicalRegionBinding,
    CanonicalRegionMetadata,
    CanonicalSourceSignature,
    build_canonical_metadata,
    build_canonical_source_signature,
    build_instance_set_digest,
    calculate_canonical_centers,
    canonical_metadata_to_storage,
    inspect_canonical_cache,
    parse_canonical_metadata,
    read_canonical_centers_from_cache,
    validate_canonical_payload,
)


def test_instance_set_digest_preserves_the_pinned_napari_harpy_encoding() -> None:
    expected = "sha256:1020a68ff134a26d0139cd20507546c0278f2c308da95133089a5a7c9c8a4718"

    assert build_instance_set_digest("nuclei", [3, 1, 2]) == expected
    assert build_instance_set_digest("nuclei", np.array([2, 3, 1], dtype=np.uint64)) == expected


@pytest.mark.parametrize(
    ("dims", "shape"),
    [(("y", "x"), (4, 5)), (("z", "y", "x"), (3, 4, 5))],
)
def test_schema_v1_metadata_round_trip_supports_2d_and_3d(
    dims: tuple[str, ...],
    shape: tuple[int, ...],
) -> None:
    source = CanonicalSourceSignature(
        labels_name="labels",
        source_scale="scale0",
        dims=dims,
        shape=shape,
        dtype="uint32",
    )
    metadata = build_canonical_metadata(
        region_key="region",
        instance_key="cell_ID",
        regions={
            "labels": CanonicalRegionMetadata(
                source_signature=source,
                n_obs=2,
                instance_set_digest=build_instance_set_digest("labels", [1, 2]),
                algorithm_version=CANONICAL_ALGORITHM_VERSION,
            )
        },
    )

    storage = canonical_metadata_to_storage(metadata)

    assert storage["axes"] == ["z", "y", "x"]
    assert storage["regions"]["labels"]["source"]["dims"] == list(dims)
    assert parse_canonical_metadata(storage) == metadata


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
    source = build_canonical_source_signature(sdata, "labels")
    binding = CanonicalRegionBinding(
        table_name="table",
        labels_name="labels",
        region_key="region",
        instance_key="cell_ID",
        row_positions=np.array([0, 1]),
        instance_ids=np.array([2, 1], dtype=np.uint32),
    )

    payload = calculate_canonical_centers(
        sdata,
        CanonicalCacheReport(stored_metadata=None, source_signature=source, binding=binding),
    )

    np.testing.assert_allclose(
        payload.centers,
        [[0.0, 5 / 3, 5 / 3], [0.0, 1 / 3, 1 / 3]],
    )
    assert payload.centers.dtype == np.dtype(np.float64)


def test_calculate_canonical_centers_uses_measured_z_in_3d() -> None:
    labels = Labels3DModel.parse(
        np.ones((2, 2, 2), dtype=np.uint32),
        dims=("z", "y", "x"),
        transformations={"volume": Identity()},
    )
    sdata = SpatialData(labels={"labels": labels})
    source = build_canonical_source_signature(sdata, "labels")
    binding = CanonicalRegionBinding(
        table_name="table",
        labels_name="labels",
        region_key="region",
        instance_key="cell_ID",
        row_positions=np.array([0]),
        instance_ids=np.array([1], dtype=np.uint32),
    )

    payload = calculate_canonical_centers(
        sdata,
        CanonicalCacheReport(stored_metadata=None, source_signature=source, binding=binding),
    )

    np.testing.assert_allclose(payload.centers, [[0.5, 0.5, 0.5]])


def test_validate_canonical_payload_and_inspector_accept_a_complete_contract() -> None:
    sdata = _canonical_sdata()

    metadata = validate_canonical_payload(
        sdata,
        sdata.tables["table"],
        table_name="table",
        region_key="region",
        instance_key="cell_ID",
        regions=("labels",),
    )
    report = inspect_canonical_cache(sdata, table_name="table", labels_name="labels")

    assert metadata is not None
    assert report.state is CanonicalCacheState.VALID
    assert report.binding.instance_ids.tolist() == [2, 1]
    np.testing.assert_array_equal(
        read_canonical_centers_from_cache(sdata, report).centers,
        sdata.tables["table"].obsm[CANONICAL_OBSM_KEY],
    )


@pytest.mark.parametrize("missing", ["matrix", "metadata"])
def test_validate_canonical_payload_rejects_matrix_metadata_asymmetry(missing: str) -> None:
    sdata = _canonical_sdata()
    table = sdata.tables["table"]
    if missing == "matrix":
        del table.obsm[CANONICAL_OBSM_KEY]
    else:
        del table.uns[SPATIAL_COORDINATES_KEY][CANONICAL_OBSM_KEY]

    with pytest.raises(ValueError, match="both the matrix"):
        validate_canonical_payload(
            sdata,
            table,
            table_name="table",
            region_key="region",
            instance_key="cell_ID",
            regions=("labels",),
        )


def test_validate_canonical_payload_rejects_a_stale_source_signature() -> None:
    sdata = _canonical_sdata()
    storage = sdata.tables["table"].uns[SPATIAL_COORDINATES_KEY][CANONICAL_OBSM_KEY]
    stale = deepcopy(storage)
    stale["regions"]["labels"]["source"]["shape"] = [99, 99]
    sdata.tables["table"].uns[SPATIAL_COORDINATES_KEY][CANONICAL_OBSM_KEY] = stale

    with pytest.raises(ValueError, match="source signature"):
        validate_canonical_payload(
            sdata,
            sdata.tables["table"],
            table_name="table",
            region_key="region",
            instance_key="cell_ID",
            regions=("labels",),
        )


def test_validate_table_rejects_an_incomplete_canonical_contract() -> None:
    sdata = _canonical_sdata()
    del sdata.tables["table"].uns[SPATIAL_COORDINATES_KEY][CANONICAL_OBSM_KEY]

    with pytest.raises(ValueError, match="both the matrix"):
        validate_table(sdata, "table")


@pytest.mark.parametrize("mutation", ["dtype", "shape", "schema", "coordinates"])
def test_validate_canonical_payload_rejects_malformed_components(mutation: str) -> None:
    sdata = _canonical_sdata()
    table = sdata.tables["table"]
    if mutation == "dtype":
        table.obsm[CANONICAL_OBSM_KEY] = table.obsm[CANONICAL_OBSM_KEY].astype(np.float32)
    elif mutation == "shape":
        table.obsm[CANONICAL_OBSM_KEY] = table.obsm[CANONICAL_OBSM_KEY][:, :2]
    elif mutation == "schema":
        table.uns[SPATIAL_COORDINATES_KEY][CANONICAL_OBSM_KEY]["schema_version"] = 2
    else:
        table.obsm[CANONICAL_OBSM_KEY][0, 0] = 1.0

    with pytest.raises(ValueError):
        validate_canonical_payload(
            sdata,
            table,
            table_name="table",
            region_key="region",
            instance_key="cell_ID",
            regions=("labels",),
        )


def test_add_canonical_centers_updates_only_the_canonical_components(tmp_path) -> None:
    sdata = _backed_external_sdata(tmp_path)
    table = sdata.tables["table"]
    previous_x = table.X
    previous_layer = table.layers["counts"]
    previous_raw = table.raw
    previous_obsp = table.obsp["connectivities"]
    previous_varm = table.varm["loadings"]
    previous_varp = table.varp["correlations"]

    result = add_canonical_centers(sdata, table_name="table")

    assert result is sdata
    assert result.tables["table"] is table
    assert table.X is previous_x
    assert table.layers["counts"] is previous_layer
    assert table.raw is previous_raw
    assert table.obsp["connectivities"] is previous_obsp
    assert table.varm["loadings"] is previous_varm
    assert table.varp["correlations"] is previous_varp
    assert list(table.uns[SPATIAL_COORDINATES_KEY]["viewer"]["axes"]) == ["y", "x"]
    np.testing.assert_allclose(
        table.obsm[CANONICAL_OBSM_KEY],
        [[0.0, 2 / 3, 5 / 3], [0.0, 1 / 3, 1 / 3]],
    )
    expected_metadata = _canonical_sdata().tables["table"].uns[SPATIAL_COORDINATES_KEY][CANONICAL_OBSM_KEY]
    assert parse_canonical_metadata(table.uns[SPATIAL_COORDINATES_KEY][CANONICAL_OBSM_KEY]) == (
        parse_canonical_metadata(expected_metadata)
    )
    validate_table(result, "table")

    reopened = read_zarr(result.path)
    reopened_table = reopened.tables["table"]
    validate_table(reopened, "table")
    np.testing.assert_array_equal(reopened_table.layers["counts"], [[5, 6], [7, 8]])
    np.testing.assert_array_equal(reopened_table.raw.X, [[1, 2], [3, 4]])
    np.testing.assert_array_equal(reopened_table.obsp["connectivities"].toarray(), np.eye(2))
    np.testing.assert_array_equal(reopened_table.varm["loadings"], [[1.0], [2.0]])
    np.testing.assert_array_equal(reopened_table.varp["correlations"], [[1.0, 0.5], [0.5, 1.0]])


def test_add_canonical_centers_aligns_overlapping_instance_ids_across_regions(tmp_path) -> None:
    labels_a = Labels2DModel.parse(
        np.array([[1, 1], [0, 0]], dtype=np.uint32),
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
            X=np.ones((2, 1)),
            obs=pd.DataFrame(
                {"region": ["labels_b", "labels_a"], "cell_ID": np.array([1, 1], dtype=np.uint32)},
                index=["b-1", "a-1"],
            ),
        ),
        region_key="region",
        region=["labels_a", "labels_b"],
        instance_key="cell_ID",
    )
    sdata = SpatialData(labels={"labels_a": labels_a, "labels_b": labels_b}, tables={"table": table})
    output = tmp_path / "multi.zarr"
    sdata.write(output)
    sdata = read_zarr(output)

    add_canonical_centers(sdata, table_name="table", labels_name=["labels_b", "labels_a"])

    np.testing.assert_allclose(
        sdata.tables["table"].obsm[CANONICAL_OBSM_KEY],
        [[0.0, 1.0, 1.0], [0.0, 0.0, 0.5]],
    )
    validate_table(sdata, "table")


def test_add_canonical_centers_supports_a_3d_labels_source(tmp_path) -> None:
    labels = Labels3DModel.parse(
        np.ones((2, 2, 2), dtype=np.uint32),
        dims=("z", "y", "x"),
        transformations={"global": Identity()},
    )
    table = TableModel.parse(
        AnnData(
            X=np.ones((1, 1)),
            obs=pd.DataFrame({"region": ["labels"], "cell_ID": np.array([1], dtype=np.uint32)}),
        ),
        region_key="region",
        region="labels",
        instance_key="cell_ID",
    )
    sdata = SpatialData(labels={"labels": labels}, tables={"table": table})
    output = tmp_path / "three-dimensional.zarr"
    sdata.write(output)
    sdata = read_zarr(output)

    add_canonical_centers(sdata, table_name="table")

    np.testing.assert_allclose(sdata.tables["table"].obsm[CANONICAL_OBSM_KEY], [[0.5, 0.5, 0.5]])
    validate_table(sdata, "table")


def test_add_canonical_centers_missing_instance_fails_without_writing_components(tmp_path) -> None:
    labels = Labels2DModel.parse(
        np.array([[1]], dtype=np.uint32),
        dims=("y", "x"),
        transformations={"global": Identity()},
    )
    table = TableModel.parse(
        AnnData(
            X=np.ones((1, 1)),
            obs=pd.DataFrame({"region": ["labels"], "cell_ID": np.array([2], dtype=np.uint32)}),
        ),
        region_key="region",
        region="labels",
        instance_key="cell_ID",
    )
    sdata = SpatialData(labels={"labels": labels}, tables={"table": table})
    output = tmp_path / "missing-instance.zarr"
    sdata.write(output)
    sdata = read_zarr(output)

    with pytest.raises(ValueError, match="no finite center"):
        add_canonical_centers(sdata, table_name="table")

    assert CANONICAL_OBSM_KEY not in sdata.tables["table"].obsm
    assert SPATIAL_COORDINATES_KEY not in sdata.tables["table"].uns
    reopened = read_zarr(output)
    assert CANONICAL_OBSM_KEY not in reopened.tables["table"].obsm
    assert SPATIAL_COORDINATES_KEY not in reopened.tables["table"].uns


@pytest.mark.parametrize(
    "labels_name",
    ["labels_a", ["labels_a", "labels_a"], ["labels_a", "labels_b", "labels_c"]],
)
def test_add_canonical_centers_rejects_an_incomplete_or_duplicate_labels_assertion(tmp_path, labels_name) -> None:
    labels = {
        "labels_a": Labels2DModel.parse(
            np.array([[1]], dtype=np.uint32),
            dims=("y", "x"),
            transformations={"global": Identity()},
        ),
        "labels_b": Labels2DModel.parse(
            np.array([[1]], dtype=np.uint32),
            dims=("y", "x"),
            transformations={"global": Identity()},
        ),
    }
    table = TableModel.parse(
        AnnData(
            X=np.ones((2, 1)),
            obs=pd.DataFrame(
                {"region": ["labels_a", "labels_b"], "cell_ID": np.array([1, 1], dtype=np.uint32)},
            ),
        ),
        region_key="region",
        region=["labels_a", "labels_b"],
        instance_key="cell_ID",
    )
    sdata = SpatialData(labels=labels, tables={"table": table})
    output = tmp_path / "selection.zarr"
    sdata.write(output)
    sdata = read_zarr(output)

    with pytest.raises(ValueError, match="duplicate|must match"):
        add_canonical_centers(sdata, table_name="table", labels_name=labels_name)

    assert CANONICAL_OBSM_KEY not in sdata.tables["table"].obsm


def test_add_canonical_centers_overwrite_repairs_an_asymmetric_payload(tmp_path) -> None:
    sdata = _backed_external_sdata(tmp_path)
    add_canonical_centers(sdata, table_name="table")

    with pytest.raises(ValueError, match="already contains"):
        add_canonical_centers(sdata, table_name="table")

    del sdata.tables["table"].uns[SPATIAL_COORDINATES_KEY][CANONICAL_OBSM_KEY]
    root = zarr.open_group(store=str(sdata.path), mode="r+", use_consolidated=False)
    del root["tables"]["table"]["uns"][SPATIAL_COORDINATES_KEY][CANONICAL_OBSM_KEY]
    sdata.write_consolidated_metadata()
    with pytest.raises(ValueError, match="both the matrix"):
        validate_table(sdata, "table")

    add_canonical_centers(sdata, table_name="table", overwrite=True)

    validate_table(sdata, "table")
    assert CANONICAL_OBSM_KEY in read_zarr(sdata.path).tables["table"].uns[SPATIAL_COORDINATES_KEY]


def test_add_canonical_centers_does_not_ignore_an_unrelated_table_error(tmp_path) -> None:
    sdata = _backed_external_sdata(tmp_path)
    table = sdata.tables["table"]
    table.obsm["auxiliary_feature_counts"] = np.ones((table.n_obs, 1))

    with pytest.raises(ValueError, match="without 'feature_class_aggregation' metadata"):
        add_canonical_centers(sdata, table_name="table", overwrite=True)

    assert CANONICAL_OBSM_KEY not in table.obsm


def test_add_canonical_centers_rejects_a_declared_region_without_rows(tmp_path) -> None:
    sdata = _backed_external_sdata(tmp_path)
    sdata.labels["empty_labels"] = Labels2DModel.parse(
        np.array([[1]], dtype=np.uint32),
        dims=("y", "x"),
        transformations={"global": Identity()},
    )
    sdata.tables["table"].uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY] = ["labels", "empty_labels"]

    with pytest.raises(ValueError, match="without observations"):
        add_canonical_centers(sdata, table_name="table")


def test_add_canonical_centers_rolls_back_disk_and_memory_when_consolidation_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    sdata = _backed_external_sdata(tmp_path)
    add_canonical_centers(sdata, table_name="table")
    table = sdata.tables["table"]
    previous = np.array([[0.0, 99.0, 99.0], [0.0, 98.0, 98.0]], dtype=np.float64)
    table.obsm[CANONICAL_OBSM_KEY][:] = previous

    def fail_consolidation(self) -> None:
        raise RuntimeError("synthetic consolidation failure")

    monkeypatch.setattr(SpatialData, "write_consolidated_metadata", fail_consolidation)
    with pytest.raises(RuntimeError, match="synthetic consolidation failure"):
        add_canonical_centers(sdata, table_name="table", overwrite=True)

    np.testing.assert_array_equal(table.obsm[CANONICAL_OBSM_KEY], previous)
    reopened = read_zarr(sdata.path)
    np.testing.assert_array_equal(reopened.tables["table"].obsm[CANONICAL_OBSM_KEY], previous)
    validate_table(reopened, "table")
    assert not list(tmp_path.glob(".external.zarr.harpy-canonical-*"))


def _canonical_sdata() -> SpatialData:
    labels = Labels2DModel.parse(
        np.array([[1, 1, 2], [1, 2, 2]], dtype=np.uint32),
        dims=("y", "x"),
        transformations={"sample": Identity()},
    )
    source = CanonicalSourceSignature(
        labels_name="labels",
        source_scale="scale0",
        dims=("y", "x"),
        shape=(2, 3),
        dtype="uint32",
    )
    metadata = build_canonical_metadata(
        region_key="region",
        instance_key="cell_ID",
        regions={
            "labels": CanonicalRegionMetadata(
                source_signature=source,
                n_obs=2,
                instance_set_digest=build_instance_set_digest("labels", [1, 2]),
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
            obsm={CANONICAL_OBSM_KEY: np.array([[0.0, 2 / 3, 5 / 3], [0.0, 1 / 3, 1 / 3]])},
            uns={SPATIAL_COORDINATES_KEY: {CANONICAL_OBSM_KEY: canonical_metadata_to_storage(metadata)}},
        ),
        region_key="region",
        region="labels",
        instance_key="cell_ID",
    )
    return SpatialData(labels={"labels": labels}, tables={"table": table})


def _backed_external_sdata(tmp_path) -> SpatialData:
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
    sdata = SpatialData(labels={"labels": labels}, tables={"table": table})
    output = tmp_path / "external.zarr"
    sdata.write(output)
    return read_zarr(output)
