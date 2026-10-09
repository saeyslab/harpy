"""Show which scanpy helpers merge a whole Dask matrix into one task.

Supports "Upstream limitations to document" (whole-matrix tasks) and Gap 1 in
``../scanpy_rapids_singlecell.md``. The script registers spies on scanpy 1.11.1's
private ``singledispatch`` helpers and prints the block shapes they receive.
"""

import numpy as np
import scanpy._utils as scanpy_utils
from _common import make_dask_adata, quiet
from scanpy.preprocessing._qc import top_segment_proportions
from scipy import sparse

quiet()
seen = []


@scanpy_utils.axis_nnz.register(sparse.csr_matrix)
def _axis_nnz_csr(X, axis):
    seen.append(("csr", X.shape, axis))
    return X.getnnz(axis=axis)


@scanpy_utils.axis_nnz.register(np.ndarray)
def _axis_nnz_dense(X, axis):
    seen.append(("dense", X.shape, axis))
    return np.count_nonzero(X, axis=axis)


check_blocks = []
check_csr = scanpy_utils.check_nonnegative_integers.dispatch(sparse.csr_matrix)
check_dense = scanpy_utils.check_nonnegative_integers.dispatch(np.ndarray)


@scanpy_utils.check_nonnegative_integers.register(sparse.csr_matrix)
def _check_csr(X):
    check_blocks.append((type(X).__name__, X.shape))
    return check_csr(X)


@scanpy_utils.check_nonnegative_integers.register(np.ndarray)
def _check_dense(X):
    check_blocks.append((type(X).__name__, X.shape))
    return check_dense(X)


for kind in ["csr_matrix", "dense"]:
    for chunks in [(500, 500), (500, 250)]:
        X = make_dask_adata(kind, chunks).X
        for axis in (0, 1):
            seen.clear()
            try:
                result = scanpy_utils.axis_nnz(X, axis=axis).compute()
                print(f"{kind} {chunks} axis_nnz(axis={axis}): result shape={result.shape}; blocks received={seen}")
            except Exception as error:  # noqa: BLE001 - report and continue
                print(f"{kind} {chunks} axis_nnz(axis={axis}): FAIL {type(error).__name__}: {error}")
        check_blocks.clear()
        try:
            result = bool(scanpy_utils.check_nonnegative_integers(X))
            print(f"{kind} {chunks} check_nonnegative_integers: {result}; blocks received={check_blocks}")
        except Exception as error:  # noqa: BLE001 - report and continue
            print(f"{kind} {chunks} check_nonnegative_integers: FAIL {type(error).__name__}: {error}")
        try:
            proportions = top_segment_proportions(X, [50, 100])
            print(f"{kind} {chunks} top_segment_proportions: result shape={proportions.shape}")
        except Exception as error:  # noqa: BLE001 - report and continue
            print(f"{kind} {chunks} top_segment_proportions: FAIL {type(error).__name__}: {str(error)[:200]}")
