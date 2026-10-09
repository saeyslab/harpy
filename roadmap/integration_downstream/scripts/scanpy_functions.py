"""Check scanpy functions on a Dask-backed X against the same data in memory.

Supports "Upstream limitations to document" in ``../scanpy_rapids_singlecell.md``.

Usage::

    scanpy_functions.py KIND [ROWS,COLUMNS]

``KIND`` is ``csr_matrix``, ``csr_array`` or ``dense``. The optional chunk shape
defaults to ``500,500`` for a 2000 × 500 matrix; ``500,250`` splits the feature axis.
``flavor="seurat_v3"`` uses a stand-in for scikit-misc's loess (see
``_common.install_fake_skmisc``), so only its Dask behavior is meaningful.
"""

import sys

import dask
import dask.array as da
import numpy as np
import pandas as pd
import scanpy as sc
from _common import describe, install_fake_skmisc, make_dask_adata, quiet, run_step
from scipy import sparse

quiet()
install_fake_skmisc()
kind = sys.argv[1] if len(sys.argv) > 1 else "csr_matrix"
chunks = tuple(int(size) for size in sys.argv[2].split(",")) if len(sys.argv) > 2 else (500, 500)
print(f"===== scanpy functions: kind={kind} chunks={chunks} =====")


def in_memory(adata):
    """Return a copy of ``adata`` with X computed, as CSR when sparse."""
    copy = adata.copy()
    copy.X = copy.X.compute()
    if sparse.issparse(copy.X):
        copy.X = sparse.csr_matrix(copy.X)
    return copy


def as_dense(x):
    """Compute a Dask array and convert sparse matrices to dense NumPy arrays."""
    if isinstance(x, da.Array):
        x = x.compute()
    if sparse.issparse(x):
        x = x.toarray()
    return np.asarray(x)


def report_close(actual, expected, label):
    """Print whether two matrices match within a small tolerance."""
    try:
        matches = np.allclose(as_dense(actual), as_dense(expected), rtol=1e-4, atol=1e-5, equal_nan=True)
    except Exception as error:  # noqa: BLE001 - report comparison failures as results
        matches = f"comparison failed: {error}"
    print(f"       matches in-memory ({label}): {matches}")


base = make_dask_adata(kind, chunks)

# calculate_qc_metrics
adata, reference = base.copy(), in_memory(base)
result, ok = run_step(
    "calculate_qc_metrics(inplace=False)",
    sc.pp.calculate_qc_metrics,
    adata,
    qc_vars=["mt"],
    percent_top=[50, 100],
)
if ok:
    expected = sc.pp.calculate_qc_metrics(reference, qc_vars=["mt"], percent_top=[50, 100])
    for index, axis in enumerate(["obs", "var"]):
        equal = all(
            np.allclose(
                result[index][column].to_numpy(dtype=float),
                expected[index][column].to_numpy(dtype=float),
                equal_nan=True,
            )
            for column in expected[index].columns
        )
        print(f"       {axis} columns equal: {equal}")

# filter_cells / filter_genes
for function, kwargs in [
    (sc.pp.filter_cells, {"min_genes": 45}),
    (sc.pp.filter_cells, {"min_counts": 180}),
    (sc.pp.filter_genes, {"min_cells": 205}),
    (sc.pp.filter_genes, {"min_counts": 800}),
]:
    adata = base.copy()
    run_step(f"{function.__name__}({kwargs})", function, adata, **kwargs)
    reference = in_memory(base)
    function(reference, **kwargs)
    print(f"       shape: {adata.shape} (in memory: {reference.shape}); X={describe(adata.X)}")

# normalize_total
for kwargs in [
    {},
    {"target_sum": 1e4},
    {"target_sum": 1e4, "key_added": "norm_factor"},
    {"exclude_highly_expressed": True, "max_fraction": 0.02},
    {"inplace": False},
]:
    adata, reference = base.copy(), in_memory(base)
    result, ok = run_step(f"normalize_total({kwargs})", sc.pp.normalize_total, adata, **kwargs)
    if ok:
        expected = sc.pp.normalize_total(reference, **kwargs)
        if kwargs.get("inplace", True):
            print(f"       X={describe(adata.X)}")
            report_close(adata.X, reference.X, "X")
        else:
            report_close(result["X"], expected["X"], "X")

# log1p
for kwargs in [{}, {"base": 2}]:
    adata, reference = base.copy(), in_memory(base)
    run_step(f"log1p({kwargs})", sc.pp.log1p, adata, **kwargs)
    sc.pp.log1p(reference, **kwargs)
    print(f"       X={describe(adata.X)}")
    report_close(adata.X, reference.X, "X")

# highly_variable_genes
logged = base.copy()
sc.pp.normalize_total(logged)
sc.pp.log1p(logged)
logged_reference = in_memory(logged)
for kwargs in [
    {"flavor": "seurat"},
    {"flavor": "seurat", "n_top_genes": 100},
    {"flavor": "cell_ranger", "n_top_genes": 100},
    {"flavor": "seurat", "batch_key": "group", "n_top_genes": 100},
]:
    adata, reference = logged.copy(), logged_reference.copy()
    _, ok = run_step(f"highly_variable_genes({kwargs})", sc.pp.highly_variable_genes, adata, **kwargs)
    if ok:
        sc.pp.highly_variable_genes(reference, **kwargs)
        equal = (adata.var["highly_variable"].to_numpy() == reference.var["highly_variable"].to_numpy()).all()
        print(f"       highly_variable equal: {equal}")
for kwargs in [
    {"flavor": "seurat_v3", "n_top_genes": 100},
    {"flavor": "seurat_v3", "n_top_genes": 100, "batch_key": "group"},
]:
    adata, reference = base.copy(), in_memory(base)
    _, ok = run_step(f"highly_variable_genes({kwargs}) [stand-in loess]", sc.pp.highly_variable_genes, adata, **kwargs)
    if ok:
        sc.pp.highly_variable_genes(reference, **kwargs)
        equal = (adata.var["highly_variable"].to_numpy() == reference.var["highly_variable"].to_numpy()).all()
        print(f"       highly_variable equal: {equal}")

# scale. Clipping (max_value) crashes Numba's workqueue layer under the threaded
# scheduler on machines without TBB or OpenMP, so those runs are synchronous.
for kwargs in [
    {},
    {"zero_center": False},
    {"max_value": 10},
    {"zero_center": False, "max_value": 10},
    {"mask_obs": np.arange(base.n_obs) % 2 == 0},
]:
    label = {key: ("boolean mask" if key == "mask_obs" else value) for key, value in kwargs.items()}
    scheduler = "synchronous" if "max_value" in kwargs else "threads"
    adata, reference = logged.copy(), logged_reference.copy()
    with dask.config.set(scheduler=scheduler):
        _, ok = run_step(f"scale({label}), scheduler={scheduler}", sc.pp.scale, adata, **kwargs)
        if ok:
            sc.pp.scale(reference, **kwargs)
            block = adata.X.blocks[0, 0].compute()
            print(f"       X={describe(adata.X)}; first computed block type={type(block).__name__}")
            report_close(adata.X, reference.X, "X")

# pca
for kwargs in [
    {},
    {"svd_solver": "covariance_eigh"},
    {"svd_solver": "randomized"},
    {"svd_solver": "arpack"},
    {"zero_center": False},
    {"chunked": True, "chunk_size": 500},
    {"mask_var": None},
    {"random_state": np.random.RandomState(0)},
]:
    adata, reference = logged.copy(), logged_reference.copy()
    _, ok = run_step(f"pca(n_comps=20, {kwargs})", sc.pp.pca, adata, n_comps=20, **kwargs)
    if ok:
        print(f"       obsm['X_pca']={describe(adata.obsm['X_pca'])}")
        reference_kwargs = {key: value for key, value in kwargs.items() if key != "svd_solver"}
        sc.pp.pca(reference, n_comps=20, svd_solver="covariance_eigh", **reference_kwargs)
        variance_equal = np.allclose(adata.uns["pca"]["variance"], reference.uns["pca"]["variance"], rtol=1e-3)
        print(f"       variance matches in-memory: {variance_equal}")

# regress_out
for keys in ["cont", "group"]:
    adata = logged.copy()
    run_step(f"regress_out({keys!r})", sc.pp.regress_out, adata, keys)
    print(f"       X after: {describe(adata.X)}")

# rank_genes_groups
for kwargs in [
    {"method": "t-test"},
    {"method": "t-test", "pts": True},
    {"method": "wilcoxon"},
    {"method": "logreg", "max_iter": 50},
]:
    adata, reference = logged.copy(), logged_reference.copy()
    _, ok = run_step(f"rank_genes_groups({kwargs})", sc.tl.rank_genes_groups, adata, "group", **kwargs)
    if ok:
        sc.tl.rank_genes_groups(reference, "group", **kwargs)
        names = pd.DataFrame(adata.uns["rank_genes_groups"]["names"]).to_numpy()
        reference_names = pd.DataFrame(reference.uns["rank_genes_groups"]["names"]).to_numpy()
        print(f"       fraction of equal gene names: {(names == reference_names).mean():.2f}")

# get.aggregate
run_step(
    "get.aggregate(by='group', func=['sum', 'mean'])", sc.get.aggregate, base.copy(), by="group", func=["sum", "mean"]
)

# sample / subsample
adata = base.copy()
run_step("pp.sample(fraction=0.5)", sc.pp.sample, adata, fraction=0.5, rng=0)
print(f"       shape={adata.shape} X={describe(adata.X)}")
adata = base.copy()
run_step("pp.subsample(fraction=0.5)", sc.pp.subsample, adata, fraction=0.5)
print(f"       shape={adata.shape} X={describe(adata.X)}")
