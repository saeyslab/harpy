from __future__ import annotations

from pathlib import Path

import torch
from kronos import create_model_from_pretrained
from kronos.vision_transformer import DinoVisionTransformer
from numpy.typing import NDArray


def _dummy_patch_embedding(array: NDArray, embedding_dimension: int) -> NDArray:
    import torch

    random_torch_array = torch.rand(array.shape[0], embedding_dimension, dtype=torch.float32)
    return random_torch_array.cpu().numpy()


def _kronos_embedding(
    array: NDArray,
    embedding_dimension: int,
    checkpoint_path: str | Path,
    hf_auth_token: str | None = None,
    cfg_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    cfg: dict | None = None,
) -> NDArray:
    assert array.ndim == 5  # (i,c,z,y,x)
    if array.shape[2] != 1:
        raise ValueError("Currently only arrays with Z-dimension equal to 1 are supported.")
    # squeeze the z dimension
    array = array.squeeze(2)
    array = torch.from_numpy(array)

    # load a separate model for every worker
    model, precision, embedding_dim = create_model_from_pretrained(
        checkpoint_path=checkpoint_path,  # "hf_hub:MahmoodLab/kronos",  # or provide a local path
        cfg_path=None,  # or provide a local path
        hf_auth_token=hf_auth_token,  # provide authentication token for Hugging Face Hub
        cache_dir="./model_assets",
        cfg={"model_type": "vits16", "token_overlap": False},  # or provide None if using cfg_path
    )

    # sanity checks
    assert precision == torch.float32
    assert embedding_dim == embedding_dimension

    if not isinstance(model, DinoVisionTransformer):
        raise TypeError(f"Loaded model is not of type {DinoVisionTransformer}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    marker_count = array.shape[1]
    low, high = 0, marker_count
    dummy_marker_ids = torch.randperm(high - low) + low
    dummy_marker_ids = dummy_marker_ids.reshape(1, -1).to(device)

    # TODO: std and mean normalization of array
    # TODO: get marker id from given pandas dataframe (passed to _kronos_embedding), which contains the marker ids.

    model.to(device)
    array = array.to(device)

    with torch.no_grad():
        patch_embeddings, _, _ = model(array, marker_ids=dummy_marker_ids)

    # patch_embeddings is of dimension (i, embedding_dimension)
    return patch_embeddings.cpu().numpy()
