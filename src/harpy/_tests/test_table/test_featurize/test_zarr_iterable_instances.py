import importlib

import numpy as np
import pytest
import zarr

from harpy.table.featurization._zarr_iterable_instances import ZarrDataLoader, ZarrIterableInstances


@pytest.fixture()
def tiny_zarr(tmp_path):
    """
    Create a tiny Zarr array with shape (N, C, Z, Y, X) and chunks along axis-0.
    Values are constructed so we can infer the global row index from x[0,0,0,0].
    """
    N = 23
    C, Z, Y, X = 2, 1, 4, 4
    chunk_i = 5  # axis-0 chunking

    # Build data: each instance (row) in r contains constant r in channel 0, and r+1000 in channel 1
    arr_np = np.zeros((N, C, Z, Y, X), dtype=np.uint16)
    for r in range(N):
        arr_np[r, 0, :, :, :] = r
        arr_np[r, 1, :, :, :] = r + 1000

    zpath = tmp_path / "instances.zarr"
    z = zarr.open(str(zpath), mode="w", shape=arr_np.shape, chunks=(chunk_i, C, Z, Y, X), dtype=arr_np.dtype)
    z[:] = arr_np

    instance_ids = np.arange(N, dtype=np.int64) + 10_000

    # labels: mark some as unlabeled (-1)
    labels = np.full((N,), -1, dtype=np.int64)
    labeled_rows = [0, 1, 2, 7, 8, 9, 10, 17, 22]
    for i, r in enumerate(labeled_rows):
        labels[r] = i % 3  # 3 classes

    return {
        "zarr_path": str(zpath),
        "arr_np": arr_np,
        "instance_ids": instance_ids,
        "labels": labels,
        "chunk_i": chunk_i,
        "N": N,
        "shape": arr_np.shape,
        "labeled_rows": np.array(labeled_rows, dtype=np.int64),
    }


def _collect(loader, max_items=None):
    """Collect all outputs from a loader (or up to max_items)."""
    out = []
    for i, b in enumerate(loader):
        out.append(b)
        if max_items is not None and i + 1 >= max_items:
            break
    return out


@pytest.mark.skipif(
    not importlib.util.find_spec("torch"),
    reason="requires the torch library",
)
def test_len_training_drop_unlabeled(tiny_zarr):
    ds = ZarrIterableInstances(
        zarr_path=tiny_zarr["zarr_path"],
        instance_ids=tiny_zarr["instance_ids"],
        labels=tiny_zarr["labels"],
        drop_unlabeled=True,
        shuffle_chunks=False,
        shuffle_within_chunk=False,
        normalize="none",
        return_instance_id=False,
        return_row_index=True,
    )
    assert len(ds) == len(tiny_zarr["labeled_rows"])


@pytest.mark.skipif(
    not importlib.util.find_spec("torch"),
    reason="requires the torch library",
)
def test_len_training_keep_unlabeled(tiny_zarr):
    # typically you do not want the unlabeled in you train set, i.e. you would set drop_unlabeled to True
    ds = ZarrIterableInstances(
        zarr_path=tiny_zarr["zarr_path"],
        instance_ids=tiny_zarr["instance_ids"],
        labels=tiny_zarr["labels"],
        drop_unlabeled=False,
        shuffle_chunks=False,
        shuffle_within_chunk=False,
        normalize="none",
        return_instance_id=False,
        return_row_index=True,
    )
    assert len(ds) == tiny_zarr["N"]


@pytest.mark.skipif(
    not importlib.util.find_spec("torch"),
    reason="requires the torch library",
)
def test_inference_yields_expected_tuple(tiny_zarr):
    import torch
    from torch.utils.data import DataLoader

    ds = ZarrIterableInstances(
        zarr_path=tiny_zarr["zarr_path"],
        instance_ids=tiny_zarr["instance_ids"],
        labels=None,  # inference
        shuffle_chunks=False,
        shuffle_within_chunk=False,
        normalize="none",
        return_instance_id=True,
        return_row_index=True,
    )
    loader = DataLoader(ds, batch_size=None, num_workers=0)

    x, inst, row = next(iter(loader))
    assert isinstance(x, torch.Tensor)
    assert isinstance(inst, int)
    assert isinstance(row, int)
    assert inst == int(tiny_zarr["instance_ids"][row])


@pytest.mark.skipif(
    not importlib.util.find_spec("torch"),
    reason="requires the torch library",
)
def test_training_yields_expected_tuple(tiny_zarr):
    import torch
    from torch.utils.data import DataLoader

    ds = ZarrIterableInstances(
        zarr_path=tiny_zarr["zarr_path"],
        instance_ids=tiny_zarr["instance_ids"],
        labels=tiny_zarr["labels"],
        drop_unlabeled=True,
        shuffle_chunks=False,
        shuffle_within_chunk=False,
        normalize="none",
        return_instance_id=True,
        return_row_index=True,
    )
    loader = DataLoader(ds, batch_size=None, num_workers=0)
    x, y, inst, row = next(iter(loader))
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor) and y.dtype == torch.long
    assert isinstance(inst, int)
    assert isinstance(row, int)
    assert inst == int(tiny_zarr["instance_ids"][row])
    assert int(y.item()) == int(tiny_zarr["labels"][row])


@pytest.mark.skipif(
    not importlib.util.find_spec("torch"),
    reason="requires the torch library",
)
def test_allowed_chunk_indexes_restricts(tiny_zarr):
    from torch.utils.data import DataLoader

    # chunk_i=5 and N=23 => chunks: 0..4 (0:0-4, 1:5-9, 2:10-14, 3:15-19, 4:20-22)
    allowed = np.array([1, 4], dtype=np.int64)

    ds = ZarrIterableInstances(
        zarr_path=tiny_zarr["zarr_path"],
        instance_ids=tiny_zarr["instance_ids"],
        labels=None,
        shuffle_chunks=False,
        shuffle_within_chunk=False,
        normalize="none",
        return_instance_id=False,
        return_row_index=True,
        allowed_chunk_indexes=allowed,
    )
    loader = DataLoader(ds, batch_size=None, num_workers=0)

    rows = []
    for _, row in loader:
        rows.append(row)

    rows = np.array(rows, dtype=np.int64)

    # All rows must come from chunks 1 or 4
    chunk_i = tiny_zarr["chunk_i"]  # chunk_i is the chunksize in i
    chunks_seen = np.unique(rows // chunk_i)  # get the chunk_id for every instance (row)
    assert set(chunks_seen.tolist()).issubset({1, 4})


@pytest.mark.skipif(
    not importlib.util.find_spec("torch"),
    reason="requires the torch library",
)
def test_allowed_chunk_indexes_empty_raises(tiny_zarr):
    # choose chunk ids that don't exist
    allowed = np.array([999, 1000], dtype=np.int64)

    with pytest.raises(ValueError):
        _ = ZarrIterableInstances(
            zarr_path=tiny_zarr["zarr_path"],
            instance_ids=tiny_zarr["instance_ids"],
            labels=None,
            shuffle_chunks=False,
            shuffle_within_chunk=False,
            normalize="none",
            allowed_chunk_indexes=allowed,
        )


@pytest.mark.skipif(
    not importlib.util.find_spec("torch"),
    reason="requires the torch library",
)
def test_normalize_minmax01_per_channel(tiny_zarr):
    import torch
    from torch.utils.data import DataLoader

    # Each channel is constant within a sample -> min=max -> output should be ~0 everywhere due to +1e-8
    ds = ZarrIterableInstances(
        zarr_path=tiny_zarr["zarr_path"],
        instance_ids=tiny_zarr["instance_ids"],
        labels=None,
        shuffle_chunks=False,
        shuffle_within_chunk=False,
        normalize="minmax01",
        return_instance_id=False,
        return_row_index=False,
    )
    loader = DataLoader(ds, batch_size=None, num_workers=0)
    x = next(iter(loader))
    assert x.dtype == torch.float32
    assert torch.allclose(x, torch.zeros_like(x), atol=1e-6)


@pytest.mark.skipif(
    not importlib.util.find_spec("torch"),
    reason="requires the torch library",
)
def test_unknown_normalize_raises(tiny_zarr):
    from torch.utils.data import DataLoader

    ds = ZarrIterableInstances(
        zarr_path=tiny_zarr["zarr_path"],
        instance_ids=tiny_zarr["instance_ids"],
        labels=None,
        shuffle_chunks=False,
        shuffle_within_chunk=False,
        normalize="not supported",
    )
    loader = DataLoader(ds, batch_size=None, num_workers=0)
    with pytest.raises(ValueError):
        _ = next(iter(loader))


@pytest.mark.skipif(
    not importlib.util.find_spec("torch"),
    reason="requires the torch library",
)
def test_epoch_advances_and_changes_order_single_worker(tiny_zarr):
    """
    Strict determinism test: use num_workers=0 so ordering is fully deterministic.
    Verify:
      - epoch 0 and epoch 1 yield different row orders when shuffling enabled
      - each epoch yields the same set of rows (inference mode over all rows)
    """
    ds = ZarrIterableInstances(
        zarr_path=tiny_zarr["zarr_path"],
        instance_ids=tiny_zarr["instance_ids"],
        labels=None,  # so drop unlabeld is ignored, and we should return all the rows
        shuffle_chunks=True,
        chunk_seed=123,
        shuffle_within_chunk=True,
        buffer_seed=456,
        normalize="none",
        return_instance_id=False,
        return_row_index=True,
    )

    loader = ZarrDataLoader(ds, batch_size=None, num_workers=0, start_epoch=0)

    assert loader._get_epoch() == 0
    assert ds._get_epoch() == 0
    # epoch 0
    out0 = _collect(loader)
    assert loader._get_epoch() == 1  # epoch in ds was set to 0, and then epoch in loader was increased by 1
    assert ds._get_epoch() == 0
    rows0 = [row for (x, row) in out0]
    # epoch 1 (new iteration over loader advances epoch)
    out1 = _collect(loader)
    assert loader._get_epoch() == 2
    assert ds._get_epoch() == 1
    rows1 = [row for (x, row) in out1]

    assert set(rows0) == set(range(tiny_zarr["N"]))
    assert set(rows1) == set(range(tiny_zarr["N"]))
    assert rows0 != rows1  # order changes for new epoch


@pytest.mark.skipif(
    not importlib.util.find_spec("torch"),
    reason="requires the torch library",
)
def test_multiworker_epoch_covers_all_rows_no_duplicates(tiny_zarr):
    """
    With multiple workers, we don't assert exact order, but we do assert:
      - we get all rows exactly once per epoch
    """
    ds = ZarrIterableInstances(
        zarr_path=tiny_zarr["zarr_path"],
        instance_ids=tiny_zarr["instance_ids"],
        labels=None,
        shuffle_chunks=True,
        chunk_seed=7,
        shuffle_within_chunk=True,
        buffer_seed=9,
        normalize="none",
        return_instance_id=False,
        return_row_index=True,
    )

    loader = ZarrDataLoader(
        ds,
        batch_size=None,
        num_workers=2,
        persistent_workers=False,  # keep this False for test stability across platforms
        start_epoch=0,
    )

    out = _collect(loader)
    rows = [row for (x, row) in out]

    assert len(rows) == tiny_zarr["N"]
    assert len(set(rows)) == tiny_zarr["N"]
    assert set(rows) == set(range(tiny_zarr["N"]))
