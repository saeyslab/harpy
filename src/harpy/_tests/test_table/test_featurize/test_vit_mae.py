import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from harpy.table.featurization import _vit_mae as mae  # noqa: E402


def test_traincfg_validation():
    with pytest.raises(ValueError):
        mae.TrainCfg(n_train=1.0)
    with pytest.raises(ValueError):
        mae.TrainCfg(n_train=0.0)
    with pytest.raises(ValueError):
        mae.TrainCfg(keep_channels=(0, 1))
    with pytest.raises(ValueError):
        mae.TrainCfg(image_size=128)


def test_make_chunk_splits_deterministic_and_sizes():
    all_chunks = np.arange(10, dtype=np.int64)
    train1, val1, test1 = mae.make_chunk_splits(all_chunks, seed=123, n_train=0.6)
    train2, val2, test2 = mae.make_chunk_splits(all_chunks, seed=123, n_train=0.6)

    assert np.array_equal(train1, train2)
    assert np.array_equal(val1, val2)
    assert np.array_equal(test1, test2)

    assert len(train1) == 6
    assert len(val1) == 2
    assert len(test1) == 2

    combined = np.concatenate([train1, val1, test1])
    assert set(combined.tolist()) == set(all_chunks.tolist())


def test_make_chunk_splits_invalid_fraction():
    with pytest.raises(ValueError):
        mae.make_chunk_splits(np.arange(3), n_train=1.0)


def test_preprocess_batch_basic():
    cfg = mae.TrainCfg()
    device = torch.device("cpu")

    x = torch.zeros((4, 1, 128, 128), dtype=torch.float32)  # c,z,y,x
    x[0] = 1.0
    x[1] = 2.0
    x[2] = 3.0
    x[3] = 4.0

    batch = [x, (x, "id")]  # B == 2
    out = mae.preprocess_batch(batch=batch, cfg=cfg, device=device)

    assert out.shape == (2, 3, 224, 224)  # C==3 because we only take the first 3 channels
    assert out.dtype == torch.float32
    assert torch.allclose(out[:, 0].mean(), torch.tensor(1.0))
    assert torch.allclose(out[:, 1].mean(), torch.tensor(2.0))
    assert torch.allclose(out[:, 2].mean(), torch.tensor(3.0))


def test_preprocess_batch_invalid_shapes():
    cfg = mae.TrainCfg()
    device = torch.device("cpu")

    bad_z = torch.zeros((3, 2, 128, 128))
    with pytest.raises(ValueError):
        mae.preprocess_batch([bad_z], cfg=cfg, device=device)

    bad_channels = torch.zeros((2, 1, 128, 128))
    with pytest.raises(ValueError):
        mae.preprocess_batch([bad_channels], cfg=cfg, device=device)

    bad_hw = torch.zeros((3, 1, 64, 64))
    with pytest.raises(ValueError):
        mae.preprocess_batch([bad_hw], cfg=cfg, device=device)


def test_preprocess_batch_imagenet_norm_requires_stats():
    cfg = mae.TrainCfg(use_imagenet_norm=True)
    device = torch.device("cpu")
    x = torch.zeros((3, 1, 128, 128))
    with pytest.raises(ValueError):
        mae.preprocess_batch([x], cfg=cfg, device=device)


class _DummyConfig:
    image_size = 224
    patch_size = 16
    hidden_size = 8


class _DummyModel:
    config = _DummyConfig()

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, pixel_values, noise=None):
        batch = pixel_values.shape[0]
        num_patches = (self.config.image_size // self.config.patch_size) ** 2
        cls = torch.full((batch, 1, self.config.hidden_size), 2.0, device=pixel_values.device)
        patches = torch.full((batch, num_patches, self.config.hidden_size), 4.0, device=pixel_values.device)
        return types.SimpleNamespace(last_hidden_state=torch.cat([cls, patches], dim=1))

    @classmethod
    def from_pretrained(cls, ckpt_dir):
        return cls()


def test_mae_embedding_pools(monkeypatch):
    monkeypatch.setattr(mae, "ViTMAEModel", _DummyModel)

    array = np.zeros((2, 3, 1, 8, 8), dtype=np.float32)

    emb_cls = mae.mae_embedding(array, ckpt_dir="dummy", embedding_dimension=8, device="cpu", pool="cls")
    assert emb_cls.shape == (2, 8)
    assert np.allclose(emb_cls, 2.0)

    emb_patches = mae.mae_embedding(array, ckpt_dir="dummy", embedding_dimension=8, device="cpu", pool="mean_patches")
    assert np.allclose(emb_patches, 4.0)

    emb_all = mae.mae_embedding(array, ckpt_dir="dummy", embedding_dimension=8, device="cpu", pool="mean_all")
    expected = (2.0 + 196 * 4.0) / 197.0
    assert np.allclose(emb_all, expected)


def test_mae_embedding_invalid_pool(monkeypatch):
    monkeypatch.setattr(mae, "ViTMAEModel", _DummyModel)
    array = np.zeros((1, 3, 1, 8, 8), dtype=np.float32)
    with pytest.raises(ValueError):
        mae.mae_embedding(array, ckpt_dir="dummy", embedding_dimension=8, device="cpu", pool="nope")


def test_mae_embedding_dimension_matches_hidden_size_check(monkeypatch):
    monkeypatch.setattr(mae, "ViTMAEModel", _DummyModel)
    array = np.zeros((1, 3, 1, 8, 8), dtype=np.float32)
    with pytest.raises(ValueError):
        mae.mae_embedding(array, ckpt_dir="dummy", embedding_dimension=7, device="cpu", pool="cls")
