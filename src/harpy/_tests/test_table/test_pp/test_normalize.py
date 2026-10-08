"""harpy.tb.pp.normalize_by_size and normalize_by_quantile, in memory and on lazy tables."""

import warnings

import dask.array as da
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

import harpy as hp
from harpy._tests.test_table.test_io.test_updates import _CountComputes
from harpy._tests.test_table.test_io.test_write_table_updates import _store, _table
from harpy.table._preprocess import preprocess_proteomics, preprocess_transcriptomics
from harpy.table.io import read_table
from harpy.table.pp import normalize_by_quantile, normalize_by_size
from harpy.utils._keys import _CELLSIZE_KEY, _RAW_COUNTS_KEY

_IN_MEMORY_FORMATS = {
    "dense": np.asarray,
    "csr": sparse.csr_matrix,
    "csc": sparse.csc_matrix,
    "csr_array": sparse.csr_array,
}


def _intensities(n_obs=30, n_vars=5, *, dtype=np.float32, seed=0):
    """A table of intensities with about 30 % zeros, and the area of each cell in obs."""
    rng = np.random.default_rng(seed)
    values = rng.gamma(2.0, 3.0, size=(n_obs, n_vars)).astype(dtype)
    values[rng.random((n_obs, n_vars)) < 0.3] = 0
    obs = pd.DataFrame({"area": rng.uniform(10, 50, n_obs)}, index=[f"cell_{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"channel_{i}" for i in range(n_vars)])
    return AnnData(X=values, obs=obs, var=var)


def _dense(matrix):
    matrix = matrix.compute() if isinstance(matrix, da.Array) else matrix
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def _nonzero_quantiles(values, q):
    return np.nanquantile(np.where(values == 0, np.nan, values), q, axis=0)


def _wrapper_input(expected):
    """The wrapper's output table, with the matrix it started from: the layer in which it kept that matrix."""
    return AnnData(X=expected.layers[_RAW_COUNTS_KEY].copy(), obs=expected.obs.copy(), var=expected.var.copy())


def test_the_normalisations_live_in_harpy_tb_pp():
    assert sorted(hp.tb.pp.__all__) == ["normalize_by_quantile", "normalize_by_size"]
    assert hp.tb.pp.normalize_by_size is normalize_by_size
    assert hp.tb.pp.normalize_by_quantile is normalize_by_quantile


@pytest.mark.filterwarnings("ignore:harpy.tb.preprocess_proteomics is deprecated:FutureWarning")
def test_normalize_by_size_equals_preprocess_proteomics(sdata_multi_c_no_backed):
    sdata = preprocess_proteomics(
        sdata_multi_c_no_backed,
        labels_name="masks_whole",
        table_name="table_intensities",
        output_table_name="preprocessed",
        size_norm=True,
        log1p=False,
        instance_size_key=_CELLSIZE_KEY,
        raw_counts_key=_RAW_COUNTS_KEY,
    )
    expected = sdata.tables["preprocessed"]
    adata = _wrapper_input(expected)
    normalize_by_size(adata, size_key=_CELLSIZE_KEY)
    np.testing.assert_allclose(_dense(adata.X), _dense(expected.X), rtol=1e-6)


@pytest.mark.filterwarnings("ignore:harpy.tb.preprocess_transcriptomics is deprecated:FutureWarning")
def test_normalize_by_size_equals_preprocess_transcriptomics(sdata_transcripts_no_backed):
    sdata = preprocess_transcriptomics(
        sdata_transcripts_no_backed,
        labels_name="segmentation_mask",
        table_name="table_transcriptomics",
        output_table_name="preprocessed",
        size_norm=True,
        instance_size_key=_CELLSIZE_KEY,
        raw_counts_key=_RAW_COUNTS_KEY,
    )
    expected = sdata.tables["preprocessed"]
    adata = _wrapper_input(expected)
    normalize_by_size(adata, size_key=_CELLSIZE_KEY)
    # The wrapper applies log1p next, and keeps that matrix in raw before it scales X.
    np.testing.assert_allclose(np.log1p(_dense(adata.X)), _dense(expected.raw.X), rtol=1e-6)


@pytest.mark.filterwarnings("ignore:harpy.tb.preprocess_proteomics is deprecated:FutureWarning")
def test_normalize_by_quantile_equals_preprocess_proteomics_divided_by_100(sdata_multi_c_no_backed):
    sdata = preprocess_proteomics(
        sdata_multi_c_no_backed,
        labels_name="masks_whole",
        table_name="table_intensities",
        output_table_name="preprocessed",
        size_norm=False,
        log1p=False,
        q=0.999,
        instance_size_key=_CELLSIZE_KEY,
        raw_counts_key=_RAW_COUNTS_KEY,
    )
    expected = sdata.tables["preprocessed"]
    adata = _wrapper_input(expected)
    normalize_by_quantile(adata, q=0.999)
    np.testing.assert_allclose(_dense(adata.X), _dense(expected.X) / 100, rtol=1e-6)


@pytest.mark.parametrize("matrix_format", list(_IN_MEMORY_FORMATS))
@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64])
def test_normalize_by_size_divides_rows_by_size_and_keeps_the_format(matrix_format, dtype):
    adata = _intensities()
    values = adata.X.astype(dtype)
    adata.X = _IN_MEMORY_FORMATS[matrix_format](values)
    matrix_type = type(adata.X)
    normalize_by_size(adata, size_key="area", scale_factor=10)
    assert type(adata.X) is matrix_type
    assert adata.X.dtype == (np.float32 if dtype is np.int64 else dtype)
    expected = values / adata.obs["area"].to_numpy()[:, None] * 10
    np.testing.assert_allclose(_dense(adata.X), expected, rtol=1e-6)


def test_normalize_by_size_on_a_layer_leaves_x_unchanged():
    adata = _intensities()
    adata.layers["raw"] = adata.X.copy()
    x = adata.X.copy()
    normalize_by_size(adata, size_key="area", layer="raw")
    np.testing.assert_array_equal(adata.X, x)
    np.testing.assert_allclose(adata.layers["raw"], x / adata.obs["area"].to_numpy()[:, None] * 100, rtol=1e-6)
    assert adata.uns["normalize_by_size"] == {"size_key": "area", "scale_factor": 100, "layer": "raw"}


def test_normalize_by_size_on_a_lazy_table_computes_nothing_and_keeps_chunks_and_csr_blocks(tmp_path):
    table = _table()
    table.obs["area"] = np.linspace(10, 50, table.n_obs)
    path = _store(tmp_path, table)
    adata = read_table(path, table_name="counts", mode="lazy", sparse_chunks=8)
    chunks = adata.X.chunks
    assert len(chunks[0]) > 1
    with _CountComputes() as computes:
        normalize_by_size(adata, size_key="area")
    assert computes.count == 0
    assert adata.X.chunks == chunks
    assert isinstance(adata.X._meta, sparse.csr_matrix)
    blocks = [adata.X.blocks[i, 0].compute() for i in range(adata.X.numblocks[0])]
    assert all(isinstance(block, sparse.csr_matrix) for block in blocks)
    in_memory = read_table(path, table_name="counts", mode="eager")
    normalize_by_size(in_memory, size_key="area")
    np.testing.assert_allclose(_dense(adata.X), _dense(in_memory.X), rtol=1e-6)


@pytest.mark.parametrize("chunks", [(8, -1), (8, 2)])
@pytest.mark.parametrize("matrix_format", ["dense", "csr"])
def test_lazy_size_normalisation_equals_in_memory(matrix_format, chunks):
    in_memory = _intensities()
    in_memory.X = _IN_MEMORY_FORMATS[matrix_format](in_memory.X)
    lazy = in_memory.copy()
    lazy.X = da.from_array(lazy.X, chunks=chunks, asarray=False)
    lazy_chunks = lazy.X.chunks
    normalize_by_size(in_memory, size_key="area")
    normalize_by_size(lazy, size_key="area")
    assert lazy.X.chunks == lazy_chunks
    np.testing.assert_allclose(_dense(lazy.X), _dense(in_memory.X), rtol=1e-6)


@pytest.mark.parametrize("size", [np.nan, 0, -1])
def test_a_missing_zero_or_negative_size_raises(size):
    adata = _intensities()
    adata.obs.loc["cell_3", "area"] = size
    with pytest.raises(ValueError, match="1 missing, zero or negative sizes"):
        normalize_by_size(adata, size_key="area")


def test_a_missing_size_column_raises():
    with pytest.raises(ValueError, match="no column 'size'"):
        normalize_by_size(_intensities(), size_key="size")


@pytest.mark.parametrize("max_value", [1, 2, None])
def test_normalize_by_quantile_maps_the_quantile_to_one_and_clips_at_max_value(max_value):
    adata = _intensities()
    values = adata.X.copy()
    normalize_by_quantile(adata, q=0.9, max_value=max_value)
    quantiles = _nonzero_quantiles(values, 0.9)
    expected = values / quantiles
    if max_value is not None:
        expected = np.minimum(expected, max_value)
    np.testing.assert_allclose(adata.X, expected, rtol=1e-6)
    assert adata.X.max() == (expected.max() if max_value is None else max_value)
    np.testing.assert_allclose(adata.var["quantile"], quantiles, rtol=1e-6)
    assert adata.uns["normalize_by_quantile"] == {"q": 0.9, "max_value": max_value, "layer": None}


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_lazy_quantile_normalisation_equals_in_memory_and_computes_the_quantiles_once(dtype):
    in_memory = _intensities(n_obs=100, dtype=dtype)
    lazy = in_memory.copy()
    lazy.X = da.from_array(lazy.X, chunks=(13, -1))
    normalize_by_quantile(in_memory, q=0.99)
    with _CountComputes() as computes:
        normalize_by_quantile(lazy, q=0.99)
    assert computes.count == 1
    assert isinstance(lazy.X, da.Array)
    assert lazy.X.chunks == ((13,) * 7 + (9,), (5,))
    assert lazy.X.dtype == dtype
    np.testing.assert_allclose(lazy.var["quantile"], in_memory.var["quantile"], rtol=1e-6)
    np.testing.assert_allclose(lazy.X.compute(), in_memory.X, rtol=1e-6)


@pytest.mark.parametrize("lazy", [False, True])
def test_a_channel_without_non_zero_values_stays_zero(lazy):
    adata = _intensities()
    adata.X[:, 2] = 0
    if lazy:
        adata.X = da.from_array(adata.X, chunks=(8, -1))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        normalize_by_quantile(adata)
    result = _dense(adata.X)
    assert np.all(result[:, 2] == 0)
    assert np.isfinite(result).all()
    assert np.isnan(adata.var["quantile"].iloc[2])


def test_in_memory_sparse_input_to_normalize_by_quantile_gives_the_dense_result():
    dense = _intensities()
    csr = dense.copy()
    csr.X = sparse.csr_matrix(csr.X)
    normalize_by_quantile(dense)
    normalize_by_quantile(csr)
    assert isinstance(csr.X, np.ndarray)
    np.testing.assert_array_equal(csr.X, dense.X)


def test_lazy_sparse_input_to_normalize_by_quantile_raises():
    adata = _intensities()
    adata.X = da.from_array(sparse.csr_matrix(adata.X), chunks=(8, -1), asarray=False)
    with pytest.raises(ValueError, match="lazy sparse matrix.*mode='eager'"):
        normalize_by_quantile(adata)


@pytest.mark.parametrize(
    ("normalize", "entry"),
    [
        (lambda adata: normalize_by_size(adata, size_key="area"), "normalize_by_size"),
        (lambda adata: normalize_by_size(adata, size_key="area", key_added="by_area"), "by_area"),
        (normalize_by_quantile, "normalize_by_quantile"),
        (lambda adata: normalize_by_quantile(adata, key_added="by_quantile"), "by_quantile"),
    ],
)
def test_a_second_run_warns_at_the_caller_and_replaces_the_entry(normalize, entry):
    adata = _intensities()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        normalize(adata)
    assert entry in adata.uns
    adata.uns[entry]["layer"] = "earlier"
    with pytest.warns(UserWarning, match="may already be normalised") as record:
        normalize(adata)
    assert record[0].filename == __file__
    assert adata.uns[entry]["layer"] is None


def test_key_added_keeps_the_records_of_two_normalised_matrices_apart():
    adata = _intensities()
    values = adata.X.copy()
    adata.layers["doubled"] = 2 * values
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        normalize_by_size(adata, size_key="area")
        normalize_by_size(adata, size_key="area", scale_factor=10, layer="doubled", key_added="doubled_size")
        normalize_by_quantile(adata, q=0.9)
        normalize_by_quantile(adata, q=0.5, layer="doubled", key_added="doubled_quantile")
    assert adata.uns["normalize_by_size"] == {"size_key": "area", "scale_factor": 100, "layer": None}
    assert adata.uns["doubled_size"] == {"size_key": "area", "scale_factor": 10, "layer": "doubled"}
    assert adata.uns["normalize_by_quantile"] == {"q": 0.9, "max_value": 1, "layer": None}
    assert adata.uns["doubled_quantile"] == {"q": 0.5, "max_value": 1, "layer": "doubled"}
    sizes = adata.obs["area"].to_numpy()[:, None]
    np.testing.assert_allclose(adata.var["quantile"], _nonzero_quantiles(values / sizes * 100, 0.9), rtol=1e-6)
    np.testing.assert_allclose(
        adata.var["doubled_quantile"], _nonzero_quantiles(2 * values / sizes * 10, 0.5), rtol=1e-6
    )


def test_supplied_quantiles_give_the_computed_result_and_record_no_q():
    computed = _intensities()
    supplied = computed.copy()
    normalize_by_quantile(computed, q=0.9)
    normalize_by_quantile(supplied, q=0.5, quantiles=computed.var["quantile"].to_numpy().tolist())
    np.testing.assert_array_equal(supplied.X, computed.X)
    np.testing.assert_array_equal(supplied.var["quantile"], computed.var["quantile"])
    assert supplied.uns["normalize_by_quantile"]["q"] is None


def test_supplied_quantiles_compute_nothing_on_a_lazy_table():
    adata = _intensities()
    adata.X = da.from_array(adata.X, chunks=(8, -1))
    with _CountComputes() as computes:
        normalize_by_quantile(adata, quantiles=np.arange(1, 6))
    assert computes.count == 0
    assert isinstance(adata.X, da.Array)


def test_supplied_quantiles_in_a_series_align_by_channel_name():
    adata = _intensities()
    values = adata.X.copy()
    quantiles = pd.Series(np.arange(1.0, 6.0), index=adata.var_names)
    normalize_by_quantile(adata, quantiles=quantiles.iloc[::-1], max_value=None)
    np.testing.assert_allclose(adata.X, values / quantiles.to_numpy(), rtol=1e-6)
    np.testing.assert_array_equal(adata.var["quantile"], quantiles.to_numpy())


@pytest.mark.parametrize(
    ("quantiles", "message"),
    [
        (pd.Series([1.0, 2.0], index=["channel_0", "channel_1"]), "no value for 3 channels"),
        ([1.0, 2.0], "one value per channel"),
    ],
)
def test_supplied_quantiles_that_miss_channels_raise(quantiles, message):
    with pytest.raises(ValueError, match=message):
        normalize_by_quantile(_intensities(), quantiles=quantiles)
