"""write_table_updates: writing back the components of a table that are new or changed against its store."""

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData
from scipy import sparse
from spatialdata import SpatialData
from spatialdata.models import TableModel

import harpy.table.io._updates as updates_module
from harpy._storage._anndata import _lazy_read_source
from harpy.table.io import read_table, write_table, write_table_components, write_table_updates

_IDENTITY = ["region", "instance"]


def _table(n_obs=20, n_vars=6, *, annotated=True, seed=0):
    rng = np.random.default_rng(seed)
    counts = rng.poisson(1.0, size=(n_obs, n_vars)).astype(np.float32)
    obs_names = [f"cell_{i}" for i in range(n_obs)]
    obs = pd.DataFrame(
        {
            "region": pd.Categorical(["cells"] * n_obs),
            "instance": np.arange(1, n_obs + 1),
            "cluster": pd.Categorical(rng.choice(["a", "b"], n_obs), categories=["a", "b"]),
        },
        index=obs_names,
    )
    adata = AnnData(
        X=sparse.csr_matrix(counts),
        obs=obs,
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)]),
        layers={"dense": counts.copy()},
        obsm={"spatial": rng.random((n_obs, 2))},
        uns={"pca": {"params": {"n_comps": 2, "solver": "arpack"}, "variance": np.array([1.5, np.nan])}},
    )
    if not annotated:
        return adata
    return TableModel.parse(adata, region="cells", region_key="region", instance_key="instance")


def _store(tmp_path, adata=None, *, name="sdata.zarr", table_name="counts"):
    path = tmp_path / name
    if not path.exists():
        SpatialData().write(path)
    write_table(path, table_name=table_name, adata=_table() if adata is None else adata)
    return path


def _stored(path, table_name="counts"):
    return read_table(path, table_name=table_name, mode="eager")


def _dense(matrix):
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


@pytest.fixture
def store(tmp_path):
    return _store(tmp_path)


@pytest.fixture
def written(monkeypatch):
    """The component mappings passed to write_table_components, one per call."""
    calls = []
    original_write = updates_module.write_table_components

    def record(*args, **kwargs):
        calls.append(dict(kwargs["components"]))
        return original_write(*args, **kwargs)

    monkeypatch.setattr(updates_module, "write_table_components", record)
    return calls


# The scanpy pipeline, end to end.


def test_scanpy_pipeline_writes_exactly_its_results_and_keeps_the_stored_counts(tmp_path, written):
    path = _store(tmp_path, _table(n_obs=200, n_vars=40))
    counts = _stored(path).X.toarray()
    adata = read_table(path, table_name="counts", mode="lazy")
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=20)
    sc.pp.pca(adata, n_comps=5, svd_solver="covariance_eigh")
    # As in the recommended pattern: compute the small projection once, rather
    # than in every step that uses it, the write-back included.
    adata.obsm["X_pca"] = adata.obsm["X_pca"].compute()
    # The table has 40 genes: without use_rep, scanpy's neighbors would use X
    # instead of X_pca (it uses X_pca only above settings.N_PCS = 50 genes), and
    # this X is a lazy Dask array with sparse blocks, which the neighbor search
    # cannot convert to an in-memory array.
    sc.pp.neighbors(adata, n_neighbors=10, use_rep="X_pca")
    sc.tl.leiden(adata, flavor="igraph", n_iterations=2)
    # Only for the assertions below: the expected values of the layer. Not part
    # of the scanpy pattern, where X stays lazy until it is written.
    log1p = adata.X.compute().toarray()

    # obs and var exist in storage, so writing them needs permission.
    with pytest.raises(FileExistsError, match=r"\('obs',\), \('var',\)"):
        write_table_updates(path, table_name="counts", adata=adata, x_to=("layers", "log1p"))
    assert written == []
    assert "leiden" not in _stored(path).obs

    write_table_updates(path, table_name="counts", adata=adata, x_to=("layers", "log1p"), overwrite=True)
    assert set(written[0]) == {
        ("obs",),
        ("var",),
        ("layers", "log1p"),
        ("obsm", "X_pca"),
        ("varm", "PCs"),
        ("obsp", "connectivities"),
        ("obsp", "distances"),
        ("uns", "log1p"),
        ("uns", "hvg"),
        ("uns", "pca"),
        ("uns", "neighbors"),
        ("uns", "leiden"),
    }
    stored = _stored(path)
    np.testing.assert_array_equal(stored.X.toarray(), counts)
    np.testing.assert_allclose(_dense(stored.layers["log1p"]), log1p)
    pd.testing.assert_series_equal(stored.obs["leiden"], adata.obs["leiden"])
    np.testing.assert_array_equal(stored.obsm["X_pca"], adata.obsm["X_pca"])

    # The reopened table matches the store: a second call writes nothing.
    write_table_updates(
        path, table_name="counts", adata=read_table(path, table_name="counts", mode="lazy"), x_to=("layers", "log1p")
    )
    assert len(written) == 1


# X and x_to.


def test_a_changed_x_without_x_to_raises_and_writes_nothing(store, written):
    adata = read_table(store, table_name="counts", mode="lazy")
    adata.X = adata.X * 2
    with pytest.raises(ValueError, match=r"x_to=\('layers', key\)"):
        write_table_updates(store, table_name="counts", adata=adata, overwrite=True)
    assert written == []


def test_x_to_x_replaces_the_stored_x_with_permission(store, written):
    adata = read_table(store, table_name="counts", mode="lazy")
    counts = _stored(store).X.toarray()
    adata.X = adata.X * 2
    with pytest.raises(FileExistsError, match=r"\('X',\)"):
        write_table_updates(store, table_name="counts", adata=adata, x_to=("X",))
    write_table_updates(store, table_name="counts", adata=adata, x_to=("X",), overwrite=True)
    assert set(written[0]) == {("X",)}
    np.testing.assert_array_equal(_stored(store).X.toarray(), counts * 2)


def test_x_to_a_layer_after_reopening(store, written):
    """The three cases of the collision rule, around one layer that receives X."""
    counts = _stored(store).X.toarray()
    adata = read_table(store, table_name="counts", mode="lazy")
    adata.X = adata.X * 2
    # A new layer needs no permission.
    write_table_updates(store, table_name="counts", adata=adata, x_to=("layers", "doubled"))
    assert set(written[0]) == {("layers", "doubled")}

    # X unchanged, the layer present from the first call: nothing to write.
    adata = read_table(store, table_name="counts", mode="lazy")
    write_table_updates(store, table_name="counts", adata=adata, x_to=("layers", "doubled"))
    assert len(written) == 1

    # A re-run: X changed again, the layer an unchanged read, so X replaces it.
    adata.X = adata.X * 3
    with pytest.raises(FileExistsError, match=r"\('layers', 'doubled'\)"):
        write_table_updates(store, table_name="counts", adata=adata, x_to=("layers", "doubled"))
    write_table_updates(store, table_name="counts", adata=adata, x_to=("layers", "doubled"), overwrite=True)
    assert set(written[1]) == {("layers", "doubled")}
    np.testing.assert_array_equal(_dense(_stored(store).layers["doubled"]), counts * 3)

    # X and the layer both changed: two values for one path, whatever overwrite.
    adata.layers["doubled"] = adata.layers["doubled"] * 5
    with pytest.raises(ValueError, match="two values for one path"):
        write_table_updates(store, table_name="counts", adata=adata, x_to=("layers", "doubled"), overwrite=True)
    assert len(written) == 2


@pytest.mark.parametrize(
    "x_to", [("obsm", "X_pca"), ("layers",), ("layers", "a", "b"), ("X", "Y"), "X", ["X"], ("layers", "")]
)
def test_x_to_other_than_a_layer_or_x_raises(store, x_to):
    adata = read_table(store, table_name="counts", mode="lazy")
    with pytest.raises(ValueError, match="x_to must be|Invalid"):
        write_table_updates(store, table_name="counts", adata=adata, x_to=x_to)


@pytest.mark.parametrize(("x_to", "destination"), [(None, ("X",)), (("layers", "counts"), ("layers", "counts"))])
def test_without_a_stored_x_a_new_x_goes_to_x_to_or_to_x(tmp_path, written, x_to, destination):
    table = _table()
    table.X = None
    path = _store(tmp_path, table)
    adata = read_table(path, table_name="counts", mode="lazy")
    adata.X = sparse.csr_matrix(adata.layers["dense"].compute())
    write_table_updates(path, table_name="counts", adata=adata, x_to=x_to)
    assert set(written[0]) == {destination}


# Axes and annotation.


def _renamed_observations(adata):
    adata.obs_names = adata.obs_names + "_renamed"
    return adata


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(lambda adata: adata[:-1].copy(), id="cells filtered"),
        pytest.param(lambda adata: adata[:, :-1].copy(), id="genes filtered"),
        pytest.param(lambda adata: adata[::-1].copy(), id="cells reordered"),
        pytest.param(_renamed_observations, id="obs_names renamed"),
    ],
)
def test_changed_axes_raise_with_the_ways_out_and_write_nothing(store, written, change):
    adata = change(read_table(store, table_name="counts", mode="eager"))
    with pytest.raises(ValueError, match="write_table_components_by_region") as error:
        write_table_updates(store, table_name="counts", adata=adata, overwrite=True)
    assert "write_table(..., overwrite=True)" in str(error.value)
    assert written == []


def test_changed_region_instance_pairs_raise(store, written):
    adata = read_table(store, table_name="counts", mode="eager")
    adata.obs["instance"] = adata.obs["instance"] + 100
    with pytest.raises(ValueError, match="region/instance pairs"):
        write_table_updates(store, table_name="counts", adata=adata, overwrite=True)
    assert written == []


@pytest.mark.parametrize("region", [["cells"], np.array(["cells"])])
def test_one_region_as_a_one_element_list_is_the_stored_annotation(store, written, region):
    """A region given as ["cells"] instead of "cells" is the same annotation.

    The stored annotation has the region "cells". Changing only its form in
    adata, to a one-element list or array, leaves the annotation unchanged:
    write_table_updates neither raises nor writes anything.
    """
    adata = read_table(store, table_name="counts", mode="lazy")
    adata.uns[TableModel.ATTRS_KEY] = {**adata.uns[TableModel.ATTRS_KEY], TableModel.REGION_KEY: region}
    write_table_updates(store, table_name="counts", adata=adata)
    assert written == []


def test_a_changed_or_added_annotation_raises_and_a_missing_one_is_left_alone(store, tmp_path, written):
    adata = read_table(store, table_name="counts", mode="lazy")
    annotation = adata.uns[TableModel.ATTRS_KEY]
    adata.uns[TableModel.ATTRS_KEY] = {**annotation, TableModel.REGION_KEY: ["nuclei"]}
    with pytest.raises(ValueError, match="differs from the stored SpatialData annotation"):
        write_table_updates(store, table_name="counts", adata=adata, overwrite=True)

    del adata.uns[TableModel.ATTRS_KEY]
    write_table_updates(store, table_name="counts", adata=adata)
    assert written == []
    assert _stored(store).uns[TableModel.ATTRS_KEY] == annotation

    unannotated = _store(tmp_path, _table(annotated=False), name="plain.zarr")
    plain = read_table(unannotated, table_name="counts", mode="lazy")
    plain.uns[TableModel.ATTRS_KEY] = annotation
    with pytest.raises(ValueError, match="would add SpatialData annotation"):
        write_table_updates(unannotated, table_name="counts", adata=plain, overwrite=True)


def test_an_unannotated_table_is_identified_by_its_obs_names(tmp_path, written):
    path = _store(tmp_path, _table(annotated=False))
    adata = read_table(path, table_name="counts", mode="lazy")
    adata.obsm["embedding"] = np.ones((adata.n_obs, 2))
    write_table_updates(path, table_name="counts", adata=adata)
    assert set(written[0]) == {("obsm", "embedding")}
    np.testing.assert_array_equal(_stored(path).obsm["embedding"], np.ones((adata.n_obs, 2)))


# What counts as changed.


@pytest.mark.parametrize(
    "read",
    [
        pytest.param({"mode": "lazy"}, id="lazy"),
        pytest.param({"mode": "lazy", "sparse_chunks": 3, "dense_chunks": 5}, id="lazy, other chunks"),
        pytest.param({"mode": "lazy", "dense_chunks": "storage"}, id="lazy, stored chunks"),
        pytest.param({"mode": "backed"}, id="backed"),
        pytest.param({"mode": "eager"}, id="eager"),
    ],
)
def test_an_unchanged_table_writes_nothing(store, written, read):
    adata = read_table(store, table_name="counts", **read)
    write_table_updates(store, table_name="counts", adata=adata)
    assert written == []


def test_a_real_rechunk_is_changed_and_a_rechunk_to_the_same_chunks_is_not(store, written):
    adata = read_table(store, table_name="counts", mode="lazy")
    dense = adata.layers["dense"]
    adata.layers["dense"] = dense.rechunk(dense.chunks)
    write_table_updates(store, table_name="counts", adata=adata)
    assert written == []
    adata.layers["dense"] = dense.rechunk({0: 5})
    write_table_updates(store, table_name="counts", adata=adata, overwrite=True)
    assert set(written[0]) == {("layers", "dense")}


@pytest.mark.parametrize(
    "replacement",
    [
        pytest.param(lambda dense: dense + 1, id="values"),
        pytest.param(lambda dense: dense.astype(np.float64), id="dtype"),
        pytest.param(lambda dense: sparse.csr_matrix(dense), id="format"),
    ],
)
def test_an_eager_table_writes_only_what_differs_from_storage(store, written, replacement):
    adata = read_table(store, table_name="counts", mode="eager")
    adata.layers["dense"] = replacement(adata.layers["dense"])
    adata.obsm["embedding"] = np.ones((adata.n_obs, 2))
    write_table_updates(store, table_name="counts", adata=adata, overwrite=True)
    assert set(written[0]) == {("layers", "dense"), ("obsm", "embedding")}


def test_an_eager_table_with_a_changed_x_needs_x_to(store, written):
    adata = read_table(store, table_name="counts", mode="eager")
    adata.X = adata.X * 2
    with pytest.raises(ValueError, match="x_to"):
        write_table_updates(store, table_name="counts", adata=adata, overwrite=True)
    assert written == []


def test_a_persisted_read_is_unchanged_and_never_writes_back_old_values(store, written):
    adata = read_table(store, table_name="counts", mode="lazy")
    adata.layers["dense"] = adata.layers["dense"].persist()
    assert _lazy_read_source(adata.layers["dense"]) is not None
    # Another writer replaces the stored layer after the read.
    newer = np.full(adata.shape, 7.0, dtype=np.float32)
    write_table_components(
        store,
        table_name="counts",
        components={("layers", "dense"): newer},
        obs_identity=adata.obs[_IDENTITY],
        var_names=adata.var_names,
        overwrite=True,
    )
    write_table_updates(store, table_name="counts", adata=adata)
    assert written == []
    np.testing.assert_array_equal(_stored(store).layers["dense"], newer)


def test_a_backed_table_writes_only_what_was_added_in_memory(store, written):
    adata = read_table(store, table_name="counts", mode="backed")
    adata.obs["leiden"] = pd.Categorical(["0", "1"] * (adata.n_obs // 2))
    adata.obsm["embedding"] = np.ones((adata.n_obs, 2))
    write_table_updates(store, table_name="counts", adata=adata, overwrite=True)
    assert set(written[0]) == {("obs",), ("obsm", "embedding")}


@pytest.mark.parametrize("source", ["other table", "other store"])
def test_backed_handles_to_other_elements_are_copied(store, tmp_path, written, source):
    other = _table(seed=1)
    if source == "other table":
        other_path = _store(tmp_path, other, table_name="other")
        other_name = "other"
    else:
        other_path = _store(tmp_path, other, name="other.zarr")
        other_name = "counts"
    handles = read_table(other_path, table_name=other_name, mode="backed")
    adata = read_table(store, table_name="counts", mode="lazy")
    adata.layers["external"] = handles.X
    adata.obsm["external"] = handles.obsm["spatial"]
    write_table_updates(store, table_name="counts", adata=adata)
    assert set(written[0]) == {("layers", "external"), ("obsm", "external")}
    stored = _stored(store)
    np.testing.assert_array_equal(_dense(stored.layers["external"]), other.X.toarray())
    np.testing.assert_array_equal(stored.obsm["external"], other.obsm["spatial"])


# What is written, and what is left alone.


def test_components_missing_from_adata_are_not_deleted(store, written):
    adata = read_table(store, table_name="counts", mode="lazy")
    adata.X = None
    del adata.layers["dense"]
    del adata.obsm["spatial"]
    del adata.uns["pca"]
    write_table_updates(store, table_name="counts", adata=adata)
    assert written == []
    stored = _stored(store)
    assert stored.X is not None
    assert "dense" in stored.layers and "spatial" in stored.obsm and "pca" in stored.uns


def test_obs_and_uns_keys_are_written_whole(store, written):
    """A removed obs column and a removed nested uns key disappear; a removed top-level uns key stays."""
    adata = read_table(store, table_name="counts", mode="lazy")
    adata.obs = adata.obs.drop(columns="cluster")
    del adata.uns["pca"]["params"]["solver"]
    adata.uns["pca"]["params"]["n_comps"] = 3
    write_table_updates(store, table_name="counts", adata=adata, overwrite=True)
    assert set(written[0]) == {("obs",), ("uns", "pca")}
    stored = _stored(store)
    assert "cluster" not in stored.obs
    assert stored.uns["pca"]["params"] == {"n_comps": 3}

    adata = read_table(store, table_name="counts", mode="lazy")
    del adata.uns["pca"]
    write_table_updates(store, table_name="counts", adata=adata)
    assert len(written) == 1
    assert "pca" in _stored(store).uns


def test_raw_is_written_whole_when_new_and_left_alone_when_unchanged(store, written):
    adata = read_table(store, table_name="counts", mode="lazy")
    adata.raw = adata
    # The table was written without raw, which AnnData stores as None: that counts as absent.
    write_table_updates(store, table_name="counts", adata=adata)
    assert set(written[0]) == {("raw", "X"), ("raw", "var")}
    stored = _stored(store)
    np.testing.assert_array_equal(stored.raw.X.toarray(), stored.X.toarray())
    assert list(stored.raw.var_names) == list(stored.var_names)

    write_table_updates(store, table_name="counts", adata=read_table(store, table_name="counts", mode="lazy"))
    assert len(written) == 1


def test_changed_raw_var_names_raise(store, written):
    adata = read_table(store, table_name="counts", mode="eager")
    adata.raw = adata
    write_table_updates(store, table_name="counts", adata=adata)
    adata = read_table(store, table_name="counts", mode="eager")
    adata.raw = adata[:, :3].copy()
    with pytest.raises(ValueError, match="raw.var_names"):
        write_table_updates(store, table_name="counts", adata=adata, overwrite=True)
    assert len(written) == 1


def test_a_failing_write_writes_nothing(store, written):
    """All new and changed components go through one write_table_components call, which rolls back as a whole."""
    adata = read_table(store, table_name="counts", mode="lazy")
    adata.obs["leiden"] = pd.Categorical(["0", "1"] * (adata.n_obs // 2))
    adata.uns["unwritable"] = object()
    # AnnData's IORegistryError is private; match its message instead.
    with pytest.raises(Exception, match="No method registered for writing <class 'object'>"):
        write_table_updates(store, table_name="counts", adata=adata, overwrite=True)
    assert set(written[0]) == {("obs",), ("uns", "unwritable")}
    assert "leiden" not in _stored(store).obs


def test_adata_is_not_modified(store):
    """write_table_updates leaves the caller's table as it is, also after x_to.

    Every slot holds the same objects as before: X is not replaced by the stored
    counts, and the x_to layer is not added. Reopen the table to match the store.
    """
    adata = read_table(store, table_name="counts", mode="lazy")
    adata.X = adata.X * 2
    adata.obs["leiden"] = pd.Categorical(["0", "1"] * (adata.n_obs // 2))
    before = {
        "X": adata.X,
        "obs": adata.obs,
        "var": adata.var,
        "layers": dict(adata.layers),
        "obsm": dict(adata.obsm),
        "uns": dict(adata.uns),
    }
    write_table_updates(store, table_name="counts", adata=adata, x_to=("layers", "log1p"), overwrite=True)
    assert adata.X is before["X"]
    assert adata.obs is before["obs"]
    assert adata.var is before["var"]
    for slot in ("layers", "obsm", "uns"):
        after = dict(getattr(adata, slot))
        assert after.keys() == before[slot].keys()
        assert all(after[key] is value for key, value in before[slot].items())
    assert "log1p" not in adata.layers
