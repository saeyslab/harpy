from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from kronos import create_model_from_pretrained
from kronos.vision_transformer import DinoVisionTransformer
from numpy.typing import NDArray


def kronos_embedding(
    array: NDArray,
    matched_channels: pd.DataFrame,  # dataframe that matches channel_id of data specific channels to channel id's pretrained, it also contains mean and std values.
    embedding_dimension: int,
    do_instance_embedding: bool = True,  # if True does instance embedding, otherwise, per channel embedding
    checkpoint_path: str | Path = "hf_hub:MahmoodLab/kronos",
    hf_auth_token: str | None = None,
    cache_dir: str | Path | None = None,
    model_type: str = "vits16",  # Type of pre-trained model to use (e.g., vits16)
    token_overlap: bool = False,  # whether to use token overlap during feature extraction; in tutorial token overlap False only used for unsupervised phenotyping
    max_value: int = 1,  # max value of larger dask array. Depends on image type. We will do array / max, so they are in range 0-1
    channel_id_pretrained_name: str = "marker_id",  # Change this to channel_id_pretrained
    channel_id_data_specific_name: str = "channel_id",  # And this to channel_id_data_specific, only keep the channels from array that could be matched to pretrained dataset.
    channel_mean_name: str = "marker_mean",  # change this to channel_mean, ideally comes from pretrained
    channel_std_name: str = "marker_std",  # change this to channel_std, ideally comes from pretrained
) -> NDArray:
    assert array.ndim == 5  # (i,c,z,y,x)
    if array.shape[2] != 1:
        raise ValueError("Currently only arrays with Z-dimension equal to 1 are supported.")
    # squeeze the z dimension
    array = array.squeeze(2)

    # only keep the channels that could be matched
    keep_channels = matched_channels[channel_id_data_specific_name].values[
        matched_channels[channel_id_data_specific_name].values < array.shape[1]
    ]
    matched_channels = matched_channels[matched_channels[channel_id_data_specific_name].isin(keep_channels)]
    marker_ids = torch.from_numpy(matched_channels[channel_id_pretrained_name].values)
    array = array[:, keep_channels, ...]
    # put in range 0-1
    array = array / max_value
    # mean std scaling
    mean = matched_channels[channel_mean_name].values.astype("float32")  # otherwise they would be float64
    std = matched_channels[channel_std_name].values.astype("float32")
    # array is of shape (i,c,y,x) (we squeezed the z dimension)
    array = (array - mean[None, :, None, None]) / std[None, :, None, None]

    array = torch.from_numpy(array).to(torch.float32)

    # load a separate model for every worker
    model, precision, embedding_dim_model = create_model_from_pretrained(
        checkpoint_path=checkpoint_path,  # "hf_hub:MahmoodLab/kronos",  # or provide a local path
        cfg_path=None,
        hf_auth_token=hf_auth_token,  # provide authentication token for Hugging Face Hub if checkpoint path is link to hugging face model
        cache_dir=cache_dir,
        cfg={"model_type": model_type, "token_overlap": token_overlap},
    )

    # sanity checks
    assert precision == torch.float32
    if do_instance_embedding:
        if embedding_dim_model != embedding_dimension:
            raise ValueError(
                "Value of parameter 'embedding_dimension' "
                f"should be equal to the embedding dimension of the kronos model ({embedding_dim_model})."
            )
    else:
        if embedding_dimension != array.shape[1] * embedding_dim_model:
            raise ValueError(
                f"Value for parameter 'embedding_dimension' should be '#channels*{embedding_dim_model}': {array.shape[1] * embedding_dim_model}, "
                f"while {embedding_dimension} was provided."
            )

    if not isinstance(model, DinoVisionTransformer):
        raise TypeError(f"Loaded model is not of type {DinoVisionTransformer}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()
    array = array.to(device)
    marker_ids = marker_ids.reshape(1, -1).to(device, dtype=torch.int64)

    with torch.no_grad():
        instance_embeddings, marker_embeddings, _ = model(array, marker_ids=marker_ids)

    if do_instance_embedding:
        # returned array is of dimension (i, embedding_dimension)
        return instance_embeddings.cpu().numpy()
    else:
        # returned array is of dimension (i, c*embedding_dimension)
        return marker_embeddings.reshape(marker_embeddings.shape[0], -1).cpu().numpy()
