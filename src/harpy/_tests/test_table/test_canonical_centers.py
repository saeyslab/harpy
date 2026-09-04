from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from spatialdata import SpatialData
from spatialdata.models import Labels2DModel, Labels3DModel, TableModel
from spatialdata.transformations import Identity

from harpy.table import validate_table
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
