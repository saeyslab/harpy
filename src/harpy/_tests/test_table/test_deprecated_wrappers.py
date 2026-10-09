"""The deprecated thin wrappers on lazy tables: the guarded ones warn, then refuse.

Kept light on purpose: these wrappers are removed in a future release, together
with this test. Their behaviour on in-memory tables is covered by their existing
tests.
"""

import pytest

import harpy as hp
from harpy.datasets.proteomics import mibi_example
from harpy.datasets.transcriptomics import resolve_example
from harpy.table._clustering import kmeans, leiden
from harpy.table._preprocess import preprocess_transcriptomics


@pytest.fixture(scope="module")
def stores(tmp_path_factory):
    """The example datasets in stores; the guarded wrappers raise before writing, so the cases can share them."""
    root = tmp_path_factory.mktemp("stores")
    paths = {"proteomics": root / "proteomics.zarr", "transcriptomics": root / "transcriptomics.zarr"}
    mibi_example().write(paths["proteomics"])
    resolve_example().write(paths["transcriptomics"])
    return paths


@pytest.mark.parametrize(
    ("store", "call"),
    [
        pytest.param(
            "transcriptomics",
            lambda sdata: preprocess_transcriptomics(
                sdata, labels_name="segmentation_mask", table_name="table_transcriptomics", output_table_name="out"
            ),
            id="preprocess_transcriptomics",
        ),
        pytest.param(
            "proteomics",
            lambda sdata: leiden(
                sdata, labels_name="masks_whole", table_name="table_intensities", output_table_name="out"
            ),
            id="leiden",
        ),
        pytest.param(
            "proteomics",
            lambda sdata: kmeans(
                sdata, labels_name="masks_whole", table_name="table_intensities", output_table_name="out"
            ),
            id="kmeans",
        ),
    ],
)
def test_guarded_wrappers_warn_then_refuse_lazy_tables(stores, store, call):
    """The deprecation warning comes first, then the guard raises, before anything is computed."""
    sdata = hp.io.read_zarr(stores[store], table_mode="lazy")
    with (
        pytest.warns(FutureWarning, match="deprecated since version 0.5.0"),
        pytest.raises(ValueError, match="needs an in-memory table"),
    ):
        call(sdata)
