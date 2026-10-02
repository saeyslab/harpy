from collections import Counter
from pathlib import Path

import anndata as ad
import dask
import dask.array as da
import numpy as np
import pandas as pd
import pytest
import zarr
from anndata import AnnData, Raw
from anndata.abc import CSRDataset
from anndata.io import read_elem, write_elem
from scipy import sparse
from spatialdata import SpatialData
from spatialdata.models import Labels2DModel, TableModel
from spatialdata.transformations import Identity
from zarr.storage import LocalStore

import harpy._storage._anndata as anndata_storage
from harpy._storage._anndata import (
    _read_anndata_element,
    _read_backed_element,
    _read_backed_table,
    _write_anndata_element,
    _write_spatialdata_table_attrs,
)
from harpy.utils._keys import _INSTANCE_KEY, _REGION_KEY, _SPATIAL


def test_read_backed_table_satisfies_anndata_and_spatialdata_contract(tmp_path):
    table_group = _write_test_regions_table(tmp_path / "table")

    table = _read_backed_table(table_group)

    TableModel.validate(table)
    labels = Labels2DModel.parse(
        np.array([[1, 2]], dtype=np.uint32),
        dims=("y", "x"),
        transformations={"global": Identity()},
    )
    container = SpatialData(labels={"labels": labels}, tables={"table": table})

    assert container.tables["table"] is table
    assert isinstance(table.X, CSRDataset)
    assert isinstance(table.obsm[_SPATIAL], zarr.Array)
    assert isinstance(table.obsm["auxiliary_feature_counts"], CSRDataset)
    assert isinstance(table.obs, pd.DataFrame)
    assert isinstance(table.var, pd.DataFrame)
    assert isinstance(table.uns, dict)
    assert table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY] == ["labels"]


def test_write_spatialdata_table_attrs_writes_regions_table_contract(tmp_path):
    group = zarr.open_group(store=str(tmp_path / "table"), mode="w")

    _write_spatialdata_table_attrs(
        group,
        regions=["labels_a", "labels_b"],
        region_key="region",
        instance_key="instance",
    )

    assert dict(group.attrs) == {
        "instance_key": "instance",
        "region": ["labels_a", "labels_b"],
        "region_key": "region",
        "spatialdata-encoding-type": "ngff:regions_table",
        "version": "0.2",
    }


def test_read_backed_element_uses_the_stored_encoding(tmp_path):
    group = zarr.open_group(store=str(tmp_path / "elements.zarr"), mode="w")
    for path, value in {
        ("dense",): np.array([[1.0, 2.0]]),
        ("sparse",): sparse.csr_matrix([[0, 3]], dtype=np.uint32),
        ("frame",): pd.DataFrame({"value": [4]}),
    }.items():
        _write_anndata_element(group, path, value, logical_path=path, create_parents=False)
    _write_anndata_element(
        group,
        ("uns", "registry", "record"),
        {"value": 5},
        logical_path=("uns", "registry", "record"),
        create_parents=True,
    )

    dense = _read_backed_element(group["dense"])
    sparse_matrix = _read_backed_element(group["sparse"])

    assert isinstance(dense, zarr.Array)
    assert isinstance(sparse_matrix, CSRDataset)
    assert np.array_equal(dense[:], [[1.0, 2.0]])
    assert np.array_equal(sparse_matrix.to_memory().toarray(), [[0, 3]])
    assert _read_anndata_element(group, ("frame",)).equals(pd.DataFrame({"value": [4]}))
    assert _read_anndata_element(group, ("uns", "registry", "record")) == {"value": 5}


@pytest.mark.parametrize(
    "mode, dense_type, sparse_type",
    [("lazy", da.Array, da.Array), ("backed", zarr.Array, CSRDataset), ("eager", np.ndarray, sparse.csr_matrix)],
)
def test_mapping_decoder_preserves_matrix_mode(tmp_path, mode, dense_type, sparse_type):
    """Dictionary nesting must preserve the requested matrix mode and chunk settings."""
    group = zarr.open_group(str(tmp_path / "elements.zarr"), mode="w")
    _write_anndata_element(
        group,
        ("mapping",),
        {
            "dense": np.ones((2, 2)),
            "strings": np.array([["a", "b"], ["c", "d"]]),
            "nested": {"sparse": sparse.csr_matrix([[0, 3], [4, 0]]), "dense": np.ones((4, 2))},
        },
        logical_path=("mapping",),
        create_parents=False,
    )

    result = anndata_storage._decode_anndata_element(group["mapping"], mode=mode, sparse_chunks=1, dense_chunks=1)

    assert isinstance(result["dense"], dense_type)
    assert isinstance(result["strings"], dense_type)
    assert isinstance(result["nested"]["sparse"], sparse_type)
    if mode == "lazy":
        assert result["nested"]["sparse"].chunks == ((1, 1), (2,))
        stored_rows = group["mapping/nested/dense"].chunks[0]
        assert result["nested"]["dense"].chunks[0][0] == stored_rows


@pytest.mark.parametrize("mode", ["lazy", "backed", "eager"])
@pytest.mark.parametrize(
    "options, error, message",
    [
        ({"dense_chunks": "stored"}, ValueError, "dense_chunks must be 'auto' or 'storage'"),
        ({"dense_chunks": 1.5}, TypeError, "dense_chunks must be"),
        ({"dense_chunks": 0}, ValueError, "dense_chunks must be"),
        ({"sparse_chunks": "storage"}, ValueError, "sparse_chunks must be 'auto' or a positive integer"),
        ({"sparse_chunks": True}, TypeError, "sparse_chunks must be"),
        ({"sparse_chunks": -1}, ValueError, "sparse_chunks must be"),
    ],
)
def test_decoder_rejects_unknown_chunk_settings(tmp_path, mode, options, error, message):
    """Internal callers cannot fall back to "auto" by passing an unknown setting."""
    group = zarr.open_group(str(tmp_path / "elements.zarr"), mode="w")
    write_elem(group, "dense", np.ones((4, 2)))
    write_elem(group, "sparse", sparse.csr_matrix(np.eye(4)))
    for name in ("dense", "sparse"):
        with pytest.raises(error, match=message):
            anndata_storage._decode_anndata_element(group[name], mode=mode, **options)
        with pytest.raises(error, match=message):
            _read_anndata_element(group, (name,), mode=mode, **options)


def test_dense_lazy_chunks_cover_other_dimensions_empty_columns_strings_and_shards(tmp_path):
    """Row-only "auto" chunks for arrays that are not 2-D, without columns, of strings, or sharded.

    The target fits five rows of the 3-D array: two stored chunks of two rows. The
    sharded array aligns with its inner chunks of three rows, not its shards.
    """
    group = zarr.open_group(str(tmp_path / "elements.zarr"), mode="w", zarr_format=3)
    arrays = {
        "cube": (np.arange(60, dtype=np.float64).reshape(10, 3, 2), {"chunks": (2, 1, 1)}),
        "vector": (np.arange(10, dtype=np.float64), {"chunks": (3,)}),
        "empty": (np.zeros((10, 0)), {}),
        "strings": (np.array([["a", "bb"]] * 10), {"chunks": (2, 1)}),
        "sharded": (np.arange(48, dtype=np.float64).reshape(12, 4), {"chunks": (3, 4), "shards": (6, 4)}),
    }
    for name, (values, dataset_kwargs) in arrays.items():
        write_elem(group, name, values, dataset_kwargs=dataset_kwargs)
    assert group["sharded"].chunks == (3, 4) and group["sharded"].shards == (6, 4)

    with dask.config.set({"array.chunk-size": 5 * 3 * 2 * 8}):
        lazy = {name: anndata_storage._decode_anndata_element(group[name], mode="lazy") for name in arrays}

    assert lazy["cube"].chunks == ((4, 4, 2), (3,), (2,))
    # 240 bytes fit 30 rows of the vector; the result is clamped to its ten rows.
    assert lazy["vector"].chunks == ((10,),)
    assert lazy["empty"].chunks == ((10,), (0,))
    assert lazy["strings"].chunks == ((2,) * 5, (1, 1))
    # 240 bytes fit seven 32-byte rows: two inner chunks of three rows.
    assert lazy["sharded"].chunks == ((6, 6), (4,))
    for name, (values, _) in arrays.items():
        np.testing.assert_array_equal(lazy[name].compute(), values)


@pytest.mark.parametrize(
    "shape, expected",
    [
        # 4 MiB holds 20,971 rows of 50 float32 columns.
        ((100_000, 50), (20_971, 50)),
        # A smaller array is one stored chunk of its own rows, not a padded 4 MiB chunk.
        ((10, 50), (10, 50)),
        # A row larger than the constant is a stored chunk on its own.
        ((10, 2_000_000), (1, 2_000_000)),
        # Arrays that are not 2-D are chunked along the first axis only.
        ((2_000_000,), (1_048_576,)),
        ((1_000, 600, 4), (436, 600, 4)),
        # Zarr requires chunk edges of at least 1, also along empty axes.
        ((10, 0), (10, 1)),
        ((0, 5), (1, 5)),
        ((0, 0), (1, 1)),
    ],
)
def test_chosen_dense_stored_chunks_are_row_only_and_valid_for_zarr(shape, expected):
    assert anndata_storage._choose_dense_stored_chunks(shape, np.dtype(np.float32).itemsize) == expected


@pytest.mark.parametrize(
    "input_chunks, expected_rows",
    [
        # Input blocks smaller than a stored chunk of three rows: one stored chunk per write block.
        ((2, 2), (3, 3, 3, 3)),
        # Seven-row input blocks hold two whole stored chunks: their boundaries move down to row 6.
        ((7, 4), (6, 6)),
        # Input blocks of whole stored chunks are already write blocks.
        ((6, 4), (6, 6)),
    ],
)
def test_rechunk_to_write_blocks_keeps_the_input_size_in_whole_stored_chunks(input_chunks, expected_rows):
    values = da.from_array(np.arange(48, dtype=np.float64).reshape(12, 4), chunks=input_chunks)

    blocks = anndata_storage._rechunk_to_write_blocks(values, chosen_chunk_rows=3)

    assert blocks.chunks == (expected_rows, (4,))
    if blocks.chunks == values.chunks:
        assert blocks is values
    np.testing.assert_array_equal(blocks.compute(), values.compute())


def test_rechunk_to_write_blocks_leaves_arrays_without_values_unchanged():
    for values in (da.zeros((0, 4), chunks=(1, 2)), da.zeros((5, 0))):
        assert anndata_storage._rechunk_to_write_blocks(values, chosen_chunk_rows=1) is values


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_writer_stores_only_dense_matrices_in_row_only_chunks(tmp_path, monkeypatch, zarr_format):
    """Matrix paths get row-only stored chunks; other encoded arrays keep AnnData's defaults.

    The lowered constant holds two 48-byte matrix rows, or twelve values of a 1-D
    float64 array, so that the Harpy layout and AnnData's defaults differ for
    every array checked. Columns of a dataframe-valued obsm entry, obs and var
    columns, categorical codes, uns arrays, string arrays and sparse buffers must
    be stored as a plain write_elem stores them.
    """
    monkeypatch.setattr(anndata_storage, "_STORED_CHUNK_BYTES", 100)
    n_obs = 30
    values = np.arange(n_obs * 6, dtype=np.float64).reshape(n_obs, 6)
    obs = pd.DataFrame(
        {"score": np.arange(n_obs, dtype=float), "label": pd.Categorical(["a", "b"] * 15)},
        index=[f"c{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame({"mean": np.zeros(6)}, index=[f"g{i}" for i in range(6)])
    table = AnnData(
        X=values,
        obs=obs,
        var=var,
        layers={"lazy": da.from_array(values, chunks=(4, 2)), "sparse": sparse.csr_matrix(values)},
        obsm={
            "embedding": values.copy(),
            "frame": pd.DataFrame({"a": np.arange(n_obs, dtype=float)}, index=obs.index),
            "strings": np.array([["ab"] * 6] * n_obs),
        },
        varm={"loadings": np.ones((6, 6))},
        obsp={"distances": np.ones((n_obs, n_obs))},
        varp={"correlations": np.ones((6, 6))},
        uns={"values": np.arange(n_obs * 6, dtype=float), "nested": {"matrix": values.copy()}},
    )
    table.raw = AnnData(X=values.copy(), obs=obs, var=var, varm={"loadings": np.ones((6, 6))})
    root = zarr.open_group(str(tmp_path / "elements.zarr"), mode="w", zarr_format=zarr_format)

    _write_anndata_element(root, ("table",), table, logical_path=(), create_parents=False)
    write_elem(root, "reference", table)

    group, reference = root["table"], root["reference"]
    row_only = {
        "X": (2, 6),
        "layers/lazy": (2, 6),
        "obsm/embedding": (2, 6),
        "varm/loadings": (2, 6),
        # A 240-byte row exceeds the constant, so each stored chunk is one row.
        "obsp/distances": (1, n_obs),
        "varp/correlations": (2, 6),
        "raw/X": (2, 6),
        "raw/varm/loadings": (2, 6),
    }
    for name, chunks in row_only.items():
        assert group[name].chunks == chunks, name
    for name in (
        "obs/score",
        "obs/label/codes",
        "var/mean",
        "obsm/frame/a",
        "obsm/strings",
        "uns/values",
        "uns/nested/matrix",
        "layers/sparse/data",
        "layers/sparse/indices",
        "raw/var/mean",
    ):
        assert group[name].chunks == reference[name].chunks, name
    written = read_elem(group)
    np.testing.assert_array_equal(written.X, values)
    np.testing.assert_array_equal(written.layers["lazy"], values)
    np.testing.assert_array_equal(written.raw.X, values)


def test_writer_uses_the_logical_path_rather_than_the_staged_name(tmp_path, monkeypatch):
    """Components staged under temporary names are chunked by their logical path."""
    monkeypatch.setattr(anndata_storage, "_STORED_CHUNK_BYTES", 100)
    group = zarr.open_group(str(tmp_path / "elements.zarr"), mode="w", zarr_format=3)
    values = np.ones((30, 6))
    components = {
        "component-0": (("obsm", "embedding"), values),
        "component-1": (("uns", "matrix"), values),
        # Arrays that are not 2-D are chunked along the first axis, from 48-byte rows.
        "component-2": (("obsm", "cube"), np.ones((30, 3, 2))),
        "component-3": (("obsm", "vector"), np.ones(30)),
    }
    for name, (logical_path, value) in components.items():
        _write_anndata_element(group, (name,), value, logical_path=logical_path, create_parents=False)
    raw = Raw(AnnData(shape=(30, 0)), X=values, var=pd.DataFrame(index=list("abcdef")), varm={"loadings": values[:6]})
    _write_anndata_element(group, ("raw",), raw, logical_path=("raw",), create_parents=False)
    write_elem(group, "reference", values)

    assert group["component-0"].chunks == (2, 6)
    assert group["component-1"].chunks == group["reference"].chunks
    assert group["component-2"].chunks == (2, 3, 2)
    assert group["component-3"].chunks == (12,)
    assert group["raw/X"].chunks == group["raw/varm/loadings"].chunks == (2, 6)


@pytest.mark.parametrize("shape, expected", [((10, 0), (10, 1)), ((0, 5), (1, 5)), ((0, 0), (1, 1))])
@pytest.mark.parametrize("lazy", [False, True])
def test_writer_stores_dense_matrices_without_values(tmp_path, shape, expected, lazy):
    values = np.zeros(shape, dtype=np.float32)
    group = zarr.open_group(str(tmp_path / "elements.zarr"), mode="w", zarr_format=3)

    _write_anndata_element(
        group, ("X",), da.from_array(values) if lazy else values, logical_path=("X",), create_parents=False
    )

    assert group["X"].chunks == expected
    np.testing.assert_array_equal(read_elem(group["X"]), values)


class _CountingStore(LocalStore):
    """Count writes and reads of stored chunks, not of metadata."""

    def __init__(self, root):
        super().__init__(root)
        self.writes = Counter()
        self.reads = Counter()

    async def set(self, key, value):
        if "/c/" in key:
            self.writes[key] += 1
        return await super().set(key, value)

    async def get(self, key, prototype=None, byte_range=None):
        result = await super().get(key, prototype, byte_range)
        if "/c/" in key and result is not None:
            self.reads[key] += 1
        return result


def test_misaligned_dask_inputs_write_each_stored_chunk_once(tmp_path, monkeypatch):
    """Input blocks that split columns and stored chunks still write every stored chunk once, unread.

    Without the rechunk, AnnData writes the same layout with repeated
    read-modify-write cycles of shared stored chunks.
    """
    monkeypatch.setattr(anndata_storage, "_STORED_CHUNK_BYTES", 3 * 4 * 8)
    values = np.arange(48, dtype=np.float64).reshape(12, 4)
    lazy = da.from_array(values, chunks=(2, 2))
    store = _CountingStore(tmp_path / "harpy.zarr")
    reference_store = _CountingStore(tmp_path / "reference.zarr")

    _write_anndata_element(
        zarr.open_group(store=store, mode="w", zarr_format=3), ("X",), lazy, logical_path=("X",), create_parents=False
    )
    write_elem(
        zarr.open_group(store=reference_store, mode="w", zarr_format=3), "X", lazy, dataset_kwargs={"chunks": (3, 4)}
    )

    assert sorted(store.writes.values()) == [1, 1, 1, 1] and not store.reads
    assert max(reference_store.writes.values()) > 1 and reference_store.reads
    group = zarr.open_group(str(tmp_path / "harpy.zarr"), mode="r")
    assert group["X"].chunks == (3, 4)
    np.testing.assert_array_equal(group["X"][...], values)


def test_automatic_sharding_keeps_row_only_chunks_as_inner_chunks(tmp_path, monkeypatch):
    """Harpy passes no shards, so AnnData's opt-in automatic sharding still applies."""
    monkeypatch.setattr(anndata_storage, "_STORED_CHUNK_BYTES", 3 * 4 * 8)
    group = zarr.open_group(str(tmp_path / "elements.zarr"), mode="w", zarr_format=3)
    values = np.arange(120, dtype=np.float64).reshape(30, 4)

    # AnnData accepts automatic sharding only once it writes Zarr v3.
    with ad.settings.override(zarr_write_format=3), ad.settings.override(auto_shard_zarr_v3=True):
        with pytest.warns(UserWarning, match="Automatic shard shape inference is experimental"):
            _write_anndata_element(group, ("X",), values, logical_path=("X",), create_parents=False)

    assert group["X"].chunks == (3, 4) and group["X"].shards == (6, 4)
    np.testing.assert_array_equal(group["X"][...], values)


@pytest.mark.parametrize("lazy", [False, True])
def test_sparse_stored_chunks_stay_far_below_auto_blocks(tmp_path, lazy):
    """Guard the assumption behind sparse_chunks="auto" against changes in AnnData or Zarr.

    Harpy sets no sparse chunk sizes. Zarr's default, sized from the array or,
    for Dask writes, from the first block, must stay far below the about
    array.chunk-size bytes of an "auto" block.
    """
    matrix = sparse.random(20_000, 1_000, density=0.1, format="csr", dtype=np.float32, random_state=0)
    value = matrix
    if lazy:
        value = da.from_array(
            matrix, chunks=(2_000, -1), asarray=False, meta=sparse.csr_matrix((0, 0), dtype=np.float32)
        )
    group = zarr.open_group(str(tmp_path / "elements.zarr"), mode="w", zarr_format=3)

    _write_anndata_element(group, ("X",), value, logical_path=("X",), create_parents=False)

    limit = anndata_storage._chunk_size_target() // 8
    for name in ("data", "indices"):
        array = group[f"X/{name}"]
        assert array.chunks[0] * array.dtype.itemsize <= limit, name
    assert (read_elem(group["X"]) != matrix).nnz == 0


@pytest.mark.parametrize("mode", ["lazy", "backed"])
def test_lazy_and_backed_reads_reject_eager_only_encoding(tmp_path, monkeypatch, mode):
    """Harpy rejects structured arrays before dispatch, even if AnnData can decode them."""
    group = zarr.open_group(str(tmp_path / "elements.zarr"), mode="w", zarr_format=2)
    values = np.array([(1.0, 2.0)], dtype=[("a", "f8"), ("b", "f8")])
    component_path = ("obsm", "structured")
    _write_anndata_element(group, component_path, values, logical_path=component_path, create_parents=True)
    np.testing.assert_array_equal(_read_anndata_element(group, component_path, mode="eager"), values)

    def unexpected_decode(*args, **kwargs):
        pytest.fail("An unsupported matrix encoding must not reach an AnnData decoder.")

    for name in ("read_elem", "read_elem_lazy", "sparse_dataset"):
        monkeypatch.setattr(anndata_storage, name, unexpected_decode)
    with pytest.raises(ValueError, match=f"rec-array.*{mode}"):
        _read_anndata_element(group, component_path, mode=mode)


@pytest.mark.parametrize("mode", ["lazy", "backed"])
@pytest.mark.parametrize("version", ["0.3.0", None])
def test_dense_encoding_version_is_checked_before_decoding(tmp_path, monkeypatch, mode, version):
    """Harpy rejects unknown or missing dense versions before asking AnnData to read."""
    group = zarr.open_group(str(tmp_path / "elements.zarr"), mode="w")
    _write_anndata_element(group, ("X",), np.ones((2, 2)), logical_path=("X",), create_parents=False)
    if version is None:
        del group["X"].attrs["encoding-version"]
    else:
        group["X"].attrs["encoding-version"] = version

    def unexpected_decode(*args, **kwargs):
        pytest.fail("An unsupported dense encoding must not reach an AnnData decoder.")

    for name in ("read_elem", "read_elem_lazy", "sparse_dataset"):
        monkeypatch.setattr(anndata_storage, name, unexpected_decode)
    with pytest.raises(ValueError, match=f"array.*{version}.*{mode}"):
        _read_anndata_element(group, ("X",), mode=mode)


@pytest.mark.parametrize("mode", ["lazy", "eager", "backed"])
@pytest.mark.parametrize("matrix_kind", ["csr", "csc"])
@pytest.mark.parametrize("version", ["0.2.0", None])
def test_sparse_encoding_version_is_checked_before_decoding(tmp_path, monkeypatch, mode, matrix_kind, version):
    """Unknown or missing sparse versions fail in Harpy, regardless of AnnData support."""
    group = zarr.open_group(str(tmp_path / "table.zarr"), mode="w")
    matrix = getattr(sparse, f"{matrix_kind}_matrix")([[0, 3]], dtype=np.uint32)
    _write_anndata_element(group, ("X",), matrix, logical_path=("X",), create_parents=False)
    if version is None:
        del group["X"].attrs["encoding-version"]
    else:
        group["X"].attrs["encoding-version"] = version

    def unexpected_decode(*args, **kwargs):
        pytest.fail("An unsupported sparse encoding must not reach an AnnData decoder.")

    for name in ("read_elem", "read_elem_lazy", "sparse_dataset"):
        monkeypatch.setattr(anndata_storage, name, unexpected_decode)

    with pytest.raises(ValueError, match=f"Unsupported {matrix_kind}_matrix encoding version {version!r}; expected"):
        _read_anndata_element(group, ("X",), mode=mode)


def _write_test_regions_table(path: Path) -> zarr.Group:
    obs = pd.DataFrame(
        {
            _REGION_KEY: pd.Categorical(["labels", "labels"]),
            _INSTANCE_KEY: np.array([1, 2], dtype=np.uint32),
        },
        index=pd.Index(["labels_1", "labels_2"], name="observation"),
    )
    var = pd.DataFrame(index=pd.Index(["GeneA", "GeneB"], name="gene"))
    table = TableModel.parse(
        AnnData(
            X=sparse.csr_matrix(np.array([[1, 0], [0, 2]], dtype=np.uint32)),
            obs=obs,
            var=var,
            obsm={
                _SPATIAL: np.array([[0.0, 0.0], [1.0, 0.0]]),
                "auxiliary_feature_counts": sparse.csr_matrix(np.array([[0, 3], [4, 0]], dtype=np.uint32)),
            },
        ),
        region=["labels"],
        region_key=_REGION_KEY,
        instance_key=_INSTANCE_KEY,
    )
    table.write_zarr(path)
    group = zarr.open_group(store=str(path), mode="r+", use_consolidated=False)
    _write_spatialdata_table_attrs(
        group,
        regions=["labels"],
        region_key=_REGION_KEY,
        instance_key=_INSTANCE_KEY,
    )
    return group
