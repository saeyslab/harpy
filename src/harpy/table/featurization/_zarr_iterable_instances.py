import math
import multiprocessing as mp

import numpy as np
import torch
import zarr
from numpy.typing import NDArray
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


class ZarrIterableInstances(IterableDataset):
    """
    Chunk-wise iterable dataset for:

      - iterates chunk-by-chunk
      - shuffles chunks deterministically
      - partitions chunks across DDP ranks and dataloader workers
      - supervised training (labels provided): yields (x, y) or (x, y, instance_id, row_idx)
      - inference (no labels): yields (x, instance_id, row_idx)
    inspired on https://gitlab.in2p3.fr/ipsl/espri/espri-ia/projects/zarr-torch-dataset
    """

    def __init__(
        self,
        zarr_path: str,
        instance_ids: NDArray,
        labels: NDArray | None = None,  # None => inference
        chunk_i: int | None = None,
        shuffle_chunks: bool = False,  # default False for inference
        chunk_seed: int = 0,
        shuffle_within_chunk: bool = False,  # default False for inference
        buffer_seed: int | None = None,  # if None, shuffling within chunks will not be deterministic
        normalize: str = "minmax01",
        x_dtype: torch.dtype = torch.float32,
        return_instance_id: bool = True,  # for inference
        return_row_index: bool = True,  # for inference
        drop_unlabeled: bool = True,  # will drop labels that are <1 only relevant if labels is not None, otherwise ignored
        allowed_chunk_indexes: NDArray = None,  # optionally restrict to a chunk subset (train/test split)
    ):
        self.zarr_path = zarr_path
        self.instance_ids = np.asarray(instance_ids).astype(np.int64)
        self.labels_all = None if labels is None else np.asarray(labels).astype(np.int64)

        self.shuffle_chunks = shuffle_chunks
        self.chunk_seed = int(chunk_seed)
        self.shuffle_within_chunk = shuffle_within_chunk
        self.buffer_seed = buffer_seed

        self.normalize = normalize
        self.x_dtype = x_dtype

        self.return_instance_id = return_instance_id
        self.return_row_index = return_row_index
        self.drop_unlabeled = drop_unlabeled

        # epoch shared across workers (so it also works if persistent_workers=True in DataLoader)
        self._epoch = mp.Value("i", 0)  # initial value 0

        arr = zarr.open(self.zarr_path, mode="r")
        self.N = arr.shape[0]
        self.chunk_i = arr.chunks[0] if chunk_i is None else int(chunk_i)  # e.g. self.chunk_i = 500

        if len(self.instance_ids) != self.N:
            raise ValueError("instance_ids must be aligned to zarr rows (length N).")

        if self.labels_all is not None and len(self.labels_all) != self.N:
            raise ValueError("labels must be aligned to zarr rows (length N).")

        # Choose which rows we will iterate
        if self.labels_all is None:  # -> inference
            # inference: all rows
            self.valid_rows = np.arange(self.N, dtype=np.int64)  # e.g. [0,1,2,3], length is equal to i
        else:
            # supervised: only labeled rows (or keep all if drop_unlabeled=False)
            if self.drop_unlabeled:
                self.valid_rows = np.flatnonzero(
                    self.labels_all >= 0
                ).astype(
                    np.int64
                )  # e.g. [12,487,501,1234,...], length is equal to the i dimension minus the not nonzero labels (i-not nonzero,)
            else:
                self.valid_rows = np.arange(self.N, dtype=np.int64)

        # Which chunks are relevant
        # Figure out which chunks have at least 1 labeled row
        # e.g.:
        # valid_rows=[12,487,501,1234,...] (i-not nonzero,) -> if we set self.drop_unlabeld to True.
        # chunk_i = 500
        # valid_chunk_idx = [0,0,1,2,...] (i-not nonzero,)  # corresponding chunk_idx
        valid_chunk_idx = self.valid_rows // self.chunk_i
        self.valid_chunk_indexes = np.unique(valid_chunk_idx).astype(np.int64)

        if allowed_chunk_indexes is not None:
            allowed = np.asarray(allowed_chunk_indexes, dtype=np.int64)
            _valid_chunk_indexes = np.intersect1d(self.valid_chunk_indexes, allowed, assume_unique=False).astype(
                np.int64
            )
            if len(_valid_chunk_indexes) == 0:
                raise ValueError(
                    f"No chunks left after subsetting to valid chunks. Please choose from '{self.valid_chunk_indexes}'."
                )
            self.valid_chunk_indexes = _valid_chunk_indexes

        # remap to local coordinates inside the chunk using a dictionary ({chunk_id: ...)
        self._chunk_to_local = {}
        for ch_idx in self.valid_chunk_indexes:  # valid_chunk_idx = [0,0,1,2,...] and length is equal to the i dimension minus the not nonzero labels (i-not nonzero,)
            mask = valid_chunk_idx == ch_idx
            rows = self.valid_rows[mask]
            local = (rows - ch_idx * self.chunk_i).astype(
                np.int64
            )  # only keep the rows that are in chunk with id ch_idx

            if self.labels_all is not None:
                labs = self.labels_all[rows].astype(np.int64)
            else:
                labs = None

            self._chunk_to_local[int(ch_idx)] = (local, labs, rows)

    def __len__(self):
        total = 0
        for local, _, _ in self._chunk_to_local.values():
            total += len(local)
        return total

    def set_epoch(self, epoch: int) -> None:
        """Call this once per epoch."""
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    def _get_epoch(self) -> int:
        with self._epoch.get_lock():
            return int(self._epoch.value)

    def __iter__(self):
        rank, world = _ddp_info()
        wid, nworkers = _worker_info()

        epoch = self._get_epoch()

        # 1) shuffle chunks (same across all workers/ranks if seed fixed)
        chunks = self.valid_chunk_indexes.copy()

        if self.shuffle_chunks:
            if self.chunk_seed is None:
                raise ValueError("Please specify 'chunk_seed' if 'shuffle_chunks' is True.")
            # epoch-dependent shuffle, but identical across ranks/workers
            rng = np.random.default_rng(self.chunk_seed + epoch)
            rng.shuffle(chunks)

        # partition across ranks, then workers
        # if both world>1 and nworkers>1 -> then we subset chunks two times
        if world > 1:
            chunks = _select_subset(chunks, rank, world)
        if nworkers > 1:
            chunks = _select_subset(chunks, wid, nworkers)

        if self.shuffle_within_chunk:
            if self.buffer_seed is None:
                # random seed
                rng_buf = np.random.default_rng()
            else:
                # make shuffle within chunks deterministic per worker,rank and epoch
                # so for each worker, rank and epoch we get a different shuffle of the chunks
                rng_buf = np.random.default_rng(self.buffer_seed + epoch + 1_000_000 * rank + wid)
        else:
            rng_buf = None

        arr = zarr.open(self.zarr_path, mode="r")

        for ch in chunks:
            ch = int(ch)
            i0 = ch * self.chunk_i
            i1 = min(i0 + self.chunk_i, self.N)
            # note that here we assume regular chunk size
            # (i.e. last chunk can have different chunk size, fixed with the min, but if chunks[0] would be 500,300,200,
            # then we would extract overlapping parts of the zarr array.) -> FIXME need to add check that checks for regular chunks

            # load chunk
            chunk_np = np.asarray(arr[i0:i1])  # (<=chunk_i, c, z, y, x)

            local_idx, local_labels, global_rows = self._chunk_to_local[ch]

            # shuffle within chunk
            order = np.arange(len(local_idx))
            if rng_buf is not None:
                rng_buf.shuffle(order)

            for j in order:
                li = int(local_idx[j])
                row = int(global_rows[j])

                x = torch.from_numpy(chunk_np[li])  # (c,z,y,x)

                # normalization/cast
                if self.normalize == "minmax01":
                    x = _minmax01_per_channel(x)
                elif self.normalize == "unit":
                    x = x.to(torch.float32).div_(65535.0)
                elif self.normalize in (None, "none"):
                    x = x.to(torch.float32)
                else:
                    raise ValueError(f"Unknown normalize mode: {self.normalize}")

                x = x.to(self.x_dtype)

                inst = int(self.instance_ids[row]) if self.return_instance_id else None

                if self.labels_all is None:
                    # inference: (x, instance_id, row_idx)
                    out = [x]
                    if self.return_instance_id:
                        out.append(inst)
                    if self.return_row_index:
                        out.append(row)
                    yield tuple(out) if len(out) > 1 else x
                else:
                    # training: (x, y) (+ optional instance ids)
                    y = torch.tensor(int(local_labels[j]), dtype=torch.long)
                    out = [x, y]
                    if self.return_instance_id:
                        out.append(inst)
                    if self.return_row_index:
                        out.append(row)
                    yield tuple(out)


class ZarrDataLoader(DataLoader):
    def __init__(self, *args, start_epoch: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self._epoch = int(start_epoch)

    def __iter__(self):
        ds = self.dataset
        if hasattr(ds, "set_epoch"):
            ds.set_epoch(self._epoch)
        it = super().__iter__()
        self._epoch += 1
        return it


def _ddp_info():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _worker_info():
    wi = get_worker_info()
    if wi is None:
        return 0, 1
    return wi.id, wi.num_workers


def _select_subset(indexes: np.ndarray, part_id: int, n_parts: int) -> np.ndarray:
    n = len(indexes)
    per = int(math.ceil(n / n_parts))
    start = part_id * per
    stop = min(start + per, n)
    return indexes[start:stop]


def _minmax01_per_channel(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    flat = x.view(x.shape[0], -1)
    mn = flat.min(dim=1)[0].view(x.shape[0], *([1] * (x.ndim - 1)))
    mx = flat.max(dim=1)[0].view(x.shape[0], *([1] * (x.ndim - 1)))
    return (x - mn) / (mx - mn + 1e-8)
