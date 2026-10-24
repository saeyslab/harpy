from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from spatialdata import SpatialData, read_zarr
from spatialdata.models import TableModel
from spatialdata.transformations import Identity, Scale, Sequence, Translation, set_transformation

from harpy.image._image import get_dataarray
from harpy.io import CosmxSample, assign_cosmx_fovs, cosmx
from harpy.io._cosmx._fovs import _cosmx_image_of
from harpy.table import add_canonical_centers


@pytest.fixture(scope="module")
def cosmx_store(module_decoded_cosmx_path: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Two samples of one run, with a table over all their instance labels and its canonical centers."""
    output = tmp_path_factory.mktemp("cosmx") / "cosmx.zarr"
    sdata = cosmx(
        {
            "sample_a": CosmxSample(path=module_decoded_cosmx_path),
            "sample_b": CosmxSample(path=module_decoded_cosmx_path, fovs=[2, 3]),
        },
        output,
        compartment_labels=False,
        points=False,
        image_chunks=(1, 4, 4),
        labels_chunks=(4, 4),
    )
    labels_names = sorted(sdata.labels)
    regions: list[str] = []
    cell_ids = []
    for labels in labels_names:
        ids = np.unique(get_dataarray(sdata, labels).data.compute())
        ids = ids[ids > 0]
        regions += [labels] * len(ids)
        cell_ids.append(ids)
    cell_id = np.concatenate(cell_ids)
    obs = pd.DataFrame(
        {"region": pd.Categorical(regions, categories=labels_names), "cell_ID": cell_id},
        index=[f"{region}_{instance}" for region, instance in zip(regions, cell_id, strict=True)],
    )
    sdata.tables["table"] = TableModel.parse(
        AnnData(X=np.zeros((len(obs), 1)), obs=obs),
        region_key="region",
        region=labels_names,
        instance_key="cell_ID",
    )
    sdata.write_element("table")
    add_canonical_centers(read_zarr(output), table_name="table")
    return output


@pytest.fixture
def cosmx_table(cosmx_store: Path) -> SpatialData:
    """A fresh read of the shared store, so that a test's changes in memory do not reach the others."""
    return read_zarr(cosmx_store)


def test_assign_cosmx_fovs_matches_the_instance_id_encoding_across_samples(cosmx_table: SpatialData) -> None:
    assert len(cosmx_table.tables["table"].obs["region"].unique()) == 4

    assign_cosmx_fovs(cosmx_table, table_name="table")

    obs = cosmx_table.tables["table"].obs
    assert obs["fov"].dtype == pd.Int64Dtype()
    assert isinstance(obs["sample_id"].dtype, pd.CategoricalDtype)
    base = next(iter(cosmx_table.attrs["harpy"]["labels"].values()))["instance_id_encoding"]["base"]
    expected_fov = obs["cell_ID"].to_numpy().astype(np.int64) // base + 1
    np.testing.assert_array_equal(obs["fov"].to_numpy(dtype=np.int64), expected_fov)
    expected_sample = obs["region"].astype(str).str.split("_instance_labels").str[0]
    assert obs["sample_id"].astype(str).tolist() == expected_sample.tolist()
    # FOV numbers repeat across samples; the sample tells them apart.
    assert {("sample_a", 2), ("sample_b", 2)} <= set(zip(obs["sample_id"], obs["fov"], strict=True))


def test_assign_cosmx_fovs_follows_a_mask_segmented_on_a_crop(cosmx_table: SpatialData) -> None:
    labels = "sample_a_instance_labels_mosaic_1"
    table = cosmx_table.tables["table"]
    centers = np.asarray(table.obsm["spatial_canonical"])[(table.obs["region"] == labels).to_numpy()]
    # A mask segmented on a crop of the image that starts at FOV 2's tile, (y, x) = (0, 8).
    # FOV 2's cell lies at x = 13.5 in the image, but at x = 5.5, inside FOV 1's bounds,
    # in the crop: ignoring the crop's translation would assign it to FOV 1.
    offset_x = 8
    fov_2_centers = centers[centers[:, 2] > offset_x] - [0, 0, offset_x]
    cropped_labels = deepcopy(cosmx_table[labels])
    translation = Translation([offset_x, 0], axes=("x", "y"))
    pixel_size = cosmx_table.attrs["harpy"]["labels"][labels]["pixel_size_um"]
    set_transformation(
        cropped_labels,
        {
            "sample_a_global_1": translation,
            "sample_a_global_1_micron": Sequence([translation, Scale([pixel_size, pixel_size], axes=("x", "y"))]),
        },
        set_all=True,
    )
    sdata = _one_mask(cosmx_table, labels, labels_element=cropped_labels, centers=fov_2_centers)

    assign_cosmx_fovs(sdata, table_name="table")

    assert fov_2_centers[:, 2].tolist() == [5.5]
    assert sdata.tables["table"].obs["fov"].tolist() == [2]


def test_cosmx_image_of_goes_through_the_image_pixel_grid(cosmx_table: SpatialData) -> None:
    labels = "sample_a_instance_labels_mosaic_1"
    image = "sample_a_morphology_image_mosaic_1"
    # The physical coordinate system sorts first by name: the pixel grid must be chosen by its
    # identity transformation, not by name.
    transformations = {"pixels": Identity(), "a_micron": Scale([0.5, 0.5], axes=("x", "y"))}
    elements = {}
    for name in (image, labels):
        elements[name] = deepcopy(cosmx_table[name])
        set_transformation(elements[name], transformations, set_all=True)
    sdata = SpatialData(
        images={image: elements[image]},
        labels={labels: elements[labels]},
        attrs=deepcopy(cosmx_table.attrs),
    )

    assert _cosmx_image_of(sdata, labels, image_records=sdata.attrs["harpy"]["images"]) == (image, "pixels")


def test_assign_cosmx_fovs_leaves_centers_outside_every_fov_unassigned(cosmx_table: SpatialData) -> None:
    sdata = _one_mask(
        cosmx_table,
        "sample_a_instance_labels_mosaic_1",
        centers=np.array([[0.0, 1.5, 1.5], [0.0, 1.5, 100.0]]),
    )

    assign_cosmx_fovs(sdata, table_name="table", fov_key="cosmx_fov", sample_key="cosmx_sample")

    obs = sdata.tables["table"].obs
    assert obs["cosmx_fov"].tolist() == [1, pd.NA]
    assert obs["cosmx_sample"].tolist() == ["sample_a", "sample_a"]


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("no_centers", "add_canonical_centers"),
        ("no_shared_coordinate_system", "must share a coordinate system with exactly one CosMx morphology image"),
        ("region_column", "would replace the region or instance column"),
    ],
)
def test_assign_cosmx_fovs_rejects_invalid_input_without_changing_the_table(
    cosmx_table: SpatialData,
    case: str,
    message: str,
) -> None:
    labels = "sample_a_instance_labels_mosaic_1"
    labels_element = None
    if case == "no_shared_coordinate_system":
        labels_element = deepcopy(cosmx_table[labels])
        set_transformation(labels_element, {"other": Identity()}, set_all=True)
    sdata = _one_mask(cosmx_table, labels, labels_element=labels_element)
    table = sdata.tables["table"]
    if case == "no_centers":
        del table.obsm["spatial_canonical"]
    before = table.obs.copy()

    with pytest.raises(ValueError, match=message):
        assign_cosmx_fovs(sdata, table_name="table", fov_key="region" if case == "region_column" else "fov")

    pd.testing.assert_frame_equal(table.obs, before)


def _one_mask(
    source: SpatialData,
    labels: str,
    *,
    labels_element: object | None = None,
    centers: np.ndarray | None = None,
) -> SpatialData:
    """Return one mask of ``source`` and its mosaic's morphology image, with an in-memory table over the mask.

    The records of ``source`` are kept, so the result is a partial store: most registered images are missing.
    """
    table = source.tables["table"]
    rows = (table.obs["region"] == labels).to_numpy()
    if centers is None:
        centers = np.asarray(table.obsm["spatial_canonical"])[rows]
    obs = pd.DataFrame(
        {
            "region": pd.Categorical([labels] * len(centers)),
            "cell_ID": np.arange(1, len(centers) + 1, dtype=np.uint32),
        },
        index=[f"row_{index}" for index in range(len(centers))],
    )
    adata = TableModel.parse(
        AnnData(obs=obs, obsm={"spatial_canonical": centers}),
        region_key="region",
        region=labels,
        instance_key="cell_ID",
    )
    image = labels.replace("instance_labels", "morphology_image")
    return SpatialData(
        images={image: source[image]},
        labels={labels: source[labels] if labels_element is None else labels_element},
        tables={"table": adata},
        attrs=deepcopy(source.attrs),
    )
