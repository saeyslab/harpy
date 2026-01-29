from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger as log
from numpy.typing import NDArray
from torch.amp import GradScaler
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, ViTMAEForPreTraining, ViTMAEModel
from transformers.utils import logging as hf_logging

from harpy.table.featurization._zarr_iterable_instances import ZarrDataLoader, ZarrIterableInstances

try:
    import torch
    from torch.utils.data import DataLoader

except ImportError:
    log.warning("'torch' not installed, to use 'harpy.tb.ZarrIterableInstances' please install this library.")


def train_autoencoder(
    instance_ids: NDArray,
    instances_path: str | Path,
    output_dir: str | Path,  # location where the trained autoencoder will be saved.
    visualize_reconstruction: bool = True,
    epochs: int = 20,
    batch_size: int = 128,
    num_workers: int = 4,  # number of workers for dataloader
    n_train: float = 0.9,
):
    """
    Train a ViT-MAE autoencoder on Zarr instances and save the best checkpoint.

    Uses a chunk-level train/val/test split, logs losses, and optionally
    visualizes reconstructions after training.

    Parameters
    ----------
    instance_ids
        One-dimensional array of instance identifiers aligned to the first
        dimension of the Zarr array.
    instances_path
        Path to the Zarr array of instances with shape ``(i, c, z, y, x)``.
        Currently only ``c,z,y,x == 3,1,128,128`` is supported.
        Input intensities are expected to be scaled to ``[0, 1]``.
    output_dir
        Directory where the best checkpoint and processor are saved.
    visualize_reconstruction
        If ``True``, show a reconstruction panel from the validation loader
        after training.
    epochs
        Number of training epochs.
    batch_size
        Batch size for training/evaluation.
    num_workers
        Number of dataloader workers.
    n_train
        Fraction of chunks used for training (remainder is split equally
        between validation and test).
    """
    cfg = TrainCfg(
        model_name="facebook/vit-mae-base",  # hard code this for now, as we only support vit-mae-base at this point
        output_dir=output_dir,  # "/data/groups/technologies/spatial.catalyst/Arne/xenium_human_ovarian_cancer_model",
        epochs=epochs,
        batch_size=batch_size,
        n_train=n_train,
        use_imagenet_norm=False,
    )
    set_seed(cfg.seed)

    device = _get_default_device()

    # HF image processor gives the reference mean/std used by pretrained ViT checkpoints
    image_mean = None
    image_std = None

    # only get this if cfg.use_imagenet_norm is True.
    processor = AutoImageProcessor.from_pretrained(cfg.model_name)
    if cfg.use_imagenet_norm:
        image_mean = torch.tensor(processor.image_mean).view(3, 1, 1).to(device)
        image_std = torch.tensor(processor.image_std).view(3, 1, 1).to(device)

    # 1) Load the model

    model = ViTMAEForPreTraining.from_pretrained(cfg.model_name).to(device)
    model.config.mask_ratio = cfg.mask_ratio  # 0.75,

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
    )

    scaler = torch.amp.GradScaler(device, enabled=cfg.amp)

    # 2) Create the DataLoaders.

    # create this iter_ds instance to get the allowed chunk indices.
    base_ds = ZarrIterableInstances(
        zarr_path=instances_path,
        instance_ids=instance_ids,
        labels=None,  # unsupervised
        shuffle_chunks=True,
        chunk_seed=0,  # fixed seed for reproducibility
        shuffle_within_chunk=True,
        buffer_seed=0,  # set for deterministic within-chunk shuffle, but each epoch different shuffle
        normalize="none",  # do normalization upfront.
        x_dtype=torch.float32,  # FIXME: check if this is correct.
        return_instance_id=False,
        return_row_index=False,
        allowed_chunk_indexes=None,
    )

    train_chunks, val_chunks, test_chunks = make_chunk_splits(
        base_ds.valid_chunk_indexes, seed=cfg.seed, n_train=cfg.n_train
    )
    if len(train_chunks) == 0 or len(val_chunks) == 0 or len(test_chunks) == 0:
        raise ValueError(
            "Invalid chunk split: one or more splits are empty. "
            f"Got train={len(train_chunks)}, val={len(val_chunks)}, test={len(test_chunks)}. "
            "Check base_ds.valid_chunk_indexes and your split settings."
        )

    train_ds = ZarrIterableInstances(
        zarr_path=instances_path,
        instance_ids=instance_ids,
        labels=None,  # unsupervised
        shuffle_chunks=True,
        chunk_seed=0,  # fixed seed for reproducibility
        shuffle_within_chunk=True,
        buffer_seed=0,  # set for deterministic within-chunk shuffle, but each epoch different shuffle
        normalize="none",  # do normalization upfront.
        x_dtype=torch.float32,
        return_instance_id=False,
        return_row_index=False,
        allowed_chunk_indexes=train_chunks,
    )

    val_ds = ZarrIterableInstances(
        zarr_path=instances_path,
        instance_ids=instance_ids,
        labels=None,  # unsupervised
        shuffle_chunks=False,
        chunk_seed=0,  # fixed seed for reproducibility
        shuffle_within_chunk=False,
        buffer_seed=0,  # set for deterministic within-chunk shuffle, but each epoch different shuffle
        normalize="none",
        x_dtype=torch.float32,
        return_instance_id=False,
        return_row_index=False,
        allowed_chunk_indexes=val_chunks,
    )

    test_ds = ZarrIterableInstances(
        zarr_path=instances_path,
        instance_ids=instance_ids,
        labels=None,  # unsupervised
        shuffle_chunks=False,
        chunk_seed=0,  # fixed seed for reproducibility
        shuffle_within_chunk=False,
        buffer_seed=0,  # set for deterministic within-chunk shuffle, but each epoch different shuffle
        normalize="none",
        x_dtype=torch.float32,
        return_instance_id=False,
        return_row_index=False,
        allowed_chunk_indexes=test_chunks,
    )

    train_loader = ZarrDataLoader(
        dataset=train_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=num_workers > 0,  # persistent workers will fail if num_workers == 0
        prefetch_factor=4,
        start_epoch=0,
    )

    val_loader = DataLoader(  # do not increase epoch index if run on val_loader or test_loader, so we use DataLoader
        dataset=val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=num_workers > 0,
        prefetch_factor=4,
    )

    test_loader = DataLoader(  # do not increase epoch index if run on val_loader or test_loader, so we use DataLoader
        dataset=test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=num_workers > 0,
        prefetch_factor=4,
    )

    log.info("Start training the model.")
    best_val = float("inf")
    for epoch in range(cfg.epochs):
        log.info(f"\nEpoch {epoch + 1}/{cfg.epochs}")
        tr_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            cfg=cfg,
            device=device,
            image_mean=image_mean,
            image_std=image_std,
        )
        log.info(f"Evaluating Epoch {epoch + 1}.")
        va_loss = eval_one_epoch(
            model=model,
            loader=val_loader,
            cfg=cfg,
            device=device,
            image_mean=image_mean,
            image_std=image_std,
        )

        log.info(f"train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}")

        # save the best model
        if va_loss < best_val:
            best_val = va_loss
            model.save_pretrained(cfg.output_dir)
            processor.save_pretrained(cfg.output_dir)
            log.info(f"Saved -> {cfg.output_dir} (best val so far).")

    if visualize_reconstruction:
        visualize_mae_reconstructions(
            model=model,
            cfg=cfg,
            val_loader=val_loader,
            device=device,
            n_show=4,
        )

    log.info("Evaluating on the test dataset.")
    test_loss = eval_one_epoch(
        model=model,
        loader=test_loader,
        cfg=cfg,
        device=device,
        image_mean=image_mean,
        image_std=image_std,
    )
    log.info(f"test_loss={test_loss:.4f}.")
    log.info(f"Finished. Model is saved here -> {cfg.output_dir}.")


@dataclass
class TrainCfg:
    """Train config"""

    model_name: str = "facebook/vit-mae-base"
    output_dir: str = "./vitmae_sc_cells"
    epochs: int = 119
    batch_size: int = 128
    lr: float = 1.5e-4
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.95)
    num_workers: int = 8
    mask_ratio: float = 0.75
    image_size: int = 224
    keep_channels: tuple[int, int, int] = (0, 1, 2)
    use_imagenet_norm: bool = False  # already in 0,1, range
    amp: bool = True
    log_every: int = 50
    seed: int = 0
    n_train: float = 0.9

    def __post_init__(self) -> None:
        if not (0.0 < self.n_train < 1.0):
            raise ValueError(
                f"n_train must be a float strictly between 0 and 1 (0 < n_train < 1). Got n_train={self.n_train!r}."
            )
        # keep_channels check
        if len(self.keep_channels) != 3:
            raise ValueError(
                f"keep_channels must contain exactly 3 channel indices, "
                f"but got {len(self.keep_channels)}: {self.keep_channels!r}."
            )
        # image_size check
        if self.image_size != 224:
            raise ValueError(f"image_size must be 224 for this ViT-MAE setup, but got image_size={self.image_size}.")


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def _get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def preprocess_batch(
    batch: list[torch.Tensor | tuple],
    cfg: TrainCfg,
    device: torch.device,
    image_mean: torch.Tensor | None = None,  # shape (1,3,1,1) recommended
    image_std: torch.Tensor | None = None,  # shape (1,3,1,1) recommended
) -> torch.Tensor:
    xs = []
    for item in batch:
        x = (
            item[0] if isinstance(item, (tuple, list)) else item
        )  # x: (C,Z,H,W) or (C,H,W) # dataset may yield (x, id, row) or just x

        if x.ndim == 4:
            if x.shape[1] != 1:
                raise ValueError(f"Expected Z==1 for (C,Z,H,W), got shape {tuple(x.shape)}")
            x = x[:, 0]  # -> (C,H,W) # squeeze z

        if x.ndim != 3:
            raise ValueError(f"Expected (C,H,W) after squeeze, got {tuple(x.shape)}")

        # select channels (expects len(cfg.keep_channels)==3)
        if x.shape[0] < 3:
            raise ValueError(f"Expected at least 3 channels, but got {x.shape[0]}.")

        if x.shape[0] > 3:
            log.info(f"Input has {x.shape[0]} channels; keeping channels with indices {cfg.keep_channels}.")
        assert len(cfg.keep_channels) == 3
        x = x[list(cfg.keep_channels), :, :]  # (3,H,W)
        xs.append(x)

    x = torch.stack(xs, dim=0)  # (B,3,H,W)

    # FIXME: this step not really necessary, we can support other spatial sizes.
    if x.shape[2] != 128 or x.shape[3] != 128:
        raise ValueError(f"Expected spatial size 128x128 before resize, got {tuple(x.shape)}")

    # resize in one go
    x = x.to(device, non_blocking=True, dtype=torch.float32)
    x = F.interpolate(x, size=(cfg.image_size, cfg.image_size), mode="bilinear", align_corners=False)  # (B,3,224,224)

    # normalization in one go
    if cfg.use_imagenet_norm:
        if image_mean is None or image_std is None:
            raise ValueError("cfg.use_imagenet_norm=True but image_mean/image_std is None.")
        # expect image_mean/std broadcastable: (1,3,1,1) or (3,1,1)
        x = (x - image_mean) / image_std

    return x


def make_chunk_splits(all_chunk_indexes: NDArray, seed: int = 0, n_train: float = 0.9):
    if not (0.0 < n_train < 1.0):
        raise ValueError(f"n_train must be in (0,1), got {n_train}.")

    rng = np.random.default_rng(seed)
    chunks = np.array(all_chunk_indexes, copy=True)
    rng.shuffle(chunks)

    n = len(chunks)
    n_train_i = int(round(n_train * n))
    n_rem = n - n_train_i

    n_val = n_rem // 2
    # n_test = n_rem - n_val  # ensures val+test==remainder

    train = chunks[:n_train_i]
    val = chunks[n_train_i : n_train_i + n_val]
    test = chunks[n_train_i + n_val :]

    return train, val, test


def train_one_epoch(
    model: ViTMAEForPreTraining,
    loader: ZarrDataLoader,
    optimizer: Optimizer,
    scaler: GradScaler,
    cfg: TrainCfg,
    device: torch.device,
    image_mean: torch.Tensor | None = None,
    image_std: torch.Tensor | None = None,
):
    """Train one epoch."""
    if not isinstance(loader, ZarrDataLoader):
        raise TypeError(f"loader must be an instance of hp.tb.ZarrDataLoader, but got {type(loader).__name__}.")
    model.train()
    running = 0.0
    steps = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for step, batch in enumerate(pbar):
        pixel_values = preprocess_batch(
            batch=batch,
            cfg=cfg,
            image_mean=image_mean,
            image_std=image_std,
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)

        if cfg.amp:
            amp_dtype = torch.float16
            if device.type == "cpu":
                amp_dtype = torch.bfloat16
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                out = model(pixel_values=pixel_values)
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(pixel_values=pixel_values)
            loss = out.loss
            loss.backward()
            optimizer.step()

        running += float(loss.detach().cpu())
        steps += 1

        avg_loss = running / steps
        pbar.set_postfix(loss=f"{avg_loss:.4f}")

        if (step + 1) % cfg.log_every == 0:
            tqdm.write(f"  step {step + 1}: loss={avg_loss:.4f}")

    return running / max(1, steps)


@torch.no_grad()
def eval_one_epoch(
    model: ViTMAEForPreTraining,
    loader: DataLoader,
    cfg: TrainCfg,
    device: torch.device,
    image_mean: torch.Tensor | None = None,
    image_std: torch.Tensor | None = None,
):
    """Eval one epoch."""
    if not isinstance(loader, DataLoader):
        raise TypeError(f"loader must be an instance of torch.utils.data.DataLoader, but got {type(loader).__name__}.")
    model.eval()
    running = 0.0
    steps = 0
    pbar = tqdm(loader, desc="eval", leave=False)
    for batch in pbar:
        pixel_values = preprocess_batch(
            batch=batch,
            cfg=cfg,
            image_mean=image_mean,
            image_std=image_std,
            device=device,
        )
        if cfg.amp:
            amp_dtype = torch.float16
            if device.type == "cpu":
                amp_dtype = torch.bfloat16
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                out = model(pixel_values=pixel_values)
        else:
            out = model(pixel_values=pixel_values)

        running += float(out.loss.detach().cpu())
        steps += 1
        pbar.set_postfix(loss=f"{running / steps:.4f}")
    return running / max(1, steps)


def mae_embedding(
    array: torch.Tensor | NDArray,
    ckpt_dir: str | Path,  # e.g. "/.../autoencoder_model"
    embedding_dimension: int = 768,  # hp.tb.featurize needs this embedding_dimension.
    device: str | None = None,
    use_imagenet_norm: bool = False,  # set to True if trained with umagenet norm
    pool: str = "mean_patches",  # "mean_all" or "mean_patches" or "cls"
) -> NDArray:
    """Returns embedding: torch.Tensor shape (768,)"""
    assert array.ndim == 5  # (i,3,1,D,D)
    if array.shape[2] != 1:
        raise ValueError("Currently only arrays with Z-dimension equal to 1 are supported.")
    if array.shape[1] != 3:
        raise ValueError("Instance should have exactly 3 channels.")

    array = array.squeeze(2)  # squeeze z

    if device is None:
        device = _get_default_device()
    log.info(f"Using device: {device}")

    # Silence transformers info/warning logs about unused decoder weights:
    # the checkpoint contains encoder + decoder, but ViTMAEModel is encoder-only.
    prev_verbosity = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    try:
        # 1) Load encoder-only model from checkpoint
        model = ViTMAEModel.from_pretrained(ckpt_dir).to(device)
        # Sanity check: make sure we really got a ViTMAEModel instance
        if not isinstance(model, ViTMAEModel):
            raise TypeError(
                f"Expected model of type ViTMAEModel, but got {type(model).__name__!r}. "
                "Check that the checkpoint at ckpt_dir is compatible."
            )
    finally:
        # Restore previous verbosity to avoid changing global logging state permanently
        hf_logging.set_verbosity(prev_verbosity)
        hf_logging.enable_progress_bar()

    model.eval()
    model.config.mask_ratio = 0.0  # disable random masking during inference

    # sanity check
    if embedding_dimension != model.config.hidden_size:
        raise ValueError(
            f"Unsupported embedding_dimension={embedding_dimension}. "
            f"This MAE embedding function currently assumes ViT-MAE base with hidden_size={model.config.hidden_size}. "
            "If you are using a different checkpoint (e.g., vit-mae-large with hidden_size=1024), "
            "either set embedding_dimension to match model.config.hidden_size or remove this check "
            "and infer the dimension from the loaded model."
        )

    # 2) To tensor, squeeze Z
    x = torch.as_tensor(array)  # (B,3,D,D)

    # 3) Resize to 224x224 -> that is the input the autoencoder needs.
    x = x.to(torch.float32)

    if x.ndim != 4:
        raise ValueError(f"Expected x to be 4D (B,3,H,W) before interpolate, got shape {tuple(x.shape)}.")
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

    # 4) Normalization must match training
    # - If trained with ImageNet norm, set use_imagenet_norm=True
    # - Otherwise keep False
    if use_imagenet_norm:
        processor = AutoImageProcessor.from_pretrained(ckpt_dir)
        image_mean = processor.image_mean
        image_std = processor.image_std
        mean = torch.tensor(image_mean, device=device).view(1, 3, 1, 1)
        std = torch.tensor(image_std, device=device).view(1, 3, 1, 1)
        x = (x.to(device) - mean) / std
    else:
        x = x.to(device)

    # 5) Forward
    num_patches = (model.config.image_size // model.config.patch_size) ** 2  # 224/16 -> 14*14=196
    noise = torch.zeros((x.shape[0], num_patches), device=x.device)  # deterministic noise

    with torch.no_grad():
        out = model(pixel_values=x, noise=noise)
        h = out.last_hidden_state  # (B, 1+196, 768) typically (CLS + nr of patches(=196))

        if pool == "mean_all":
            emb = h.mean(dim=1)  # (B,768) includes CLS
        elif pool == "mean_patches":
            emb = h[:, 1:, :].mean(dim=1)  # (B,768) patches only
        elif pool == "cls":
            emb = h[:, 0, :]  # (B,768) CLS token
        else:
            raise ValueError("pool must be one of: mean_all, mean_patches, cls")

    result = emb.detach().cpu().numpy()

    if result.shape[1] != model.config.hidden_size:
        raise ValueError(
            "Unexpected embedding dimension in batched output. "
            f"Got {result.shape} (so hidden_dim={result.shape[1]}), "
            f"but expected hidden_dim={model.config.hidden_size} from model.config.hidden_size."
        )
    return result


@torch.no_grad()
def visualize_mae_reconstructions(
    model: ViTMAEForPreTraining,
    cfg: TrainCfg,
    val_loader: DataLoader,
    device: torch.device,
    n_show: int = 4,
):
    """
    Visualize mae reconstructions.

    Code generated by Codex 5.2.

    Let:

    pixel_values: (B, C, 224, 224), with C=3

    patch size p = 16

    number of patches L = (224/16)*(224/16) = 14*14 = 196

    a) outputs = model(pixel_values=pixel_values)

    This runs MAE forward.

    b) pred_patches = outputs.logits

    Shape: (B, L, p*p*3) = (B, 196, 16*16*3) = (B, 196, 768), with p the patch size==16 (model.config.patch_size)
    """
    if not isinstance(model, ViTMAEForPreTraining):
        raise TypeError(
            f"Reconstruction requires a MAE *pretraining* model with a decoder head. "
            f"Expected ViTMAEForPreTraining, but got {model.__class__.__name__}. "
            f"Load it with:\n"
            f"  model = ViTMAEForPreTraining.from_pretrained(ckpt_dir)\n"
            f"(ViTMAEModel only returns encoder hidden states; it does not produce pixel reconstructions.)"
        )

    def patchify(imgs: torch.Tensor, patch_size: int) -> torch.Tensor:
        """
        Patch

        Takes the real image and turns it into patch vectors:

        input: (B, C, H, W), e.g. (B, 3, 224, 224)

        output: (B, L, patch_size*patch_size*C), e.g. (B, 14*14, 16*16*3)=(B, 196, 768)

        imgs: (B, C, H, W). with C==3
        returns: (B, L, patch_size*patch_size*3) where L = (H/patch_size)*(W/patch_size)
        """
        B, C, H, W = imgs.shape
        assert C == 3
        assert H % patch_size == 0 and W % patch_size == 0
        h = H // patch_size
        w = W // patch_size
        x = imgs.reshape(B, C, h, patch_size, w, patch_size)
        x = x.permute(0, 2, 4, 3, 5, 1)  # (B, h, w, p, p, C)
        x = x.reshape(B, h * w, patch_size * patch_size * C)
        return x

    def unpatchify(patches: torch.Tensor, patch_size: int, img_size: int) -> torch.Tensor:
        """
        Unpatch

        Inverse of patchify

        Takes the patch vectors and turns them into real images.

        Input: (B, L, patch_size*patch_size*3), e.g. (B, 14*14, 16*16*3) = (B, 196, 768)

        Output: (B, C, H, W), e.g. (B, 3, 224, 224)

        patches: (B, L, patch_size*patch_size*3)
        returns: (B, C, img_size, img_size), with C==3
        """
        B, L, D = patches.shape
        C = 3
        assert D == patch_size * patch_size * C
        h = w = img_size // patch_size
        assert L == h * w
        x = patches.reshape(B, h, w, patch_size, patch_size, C)
        x = x.permute(0, 5, 1, 3, 2, 4)  # (B, C, h, p, w, p)
        imgs = x.reshape(B, C, img_size, img_size)
        return imgs

    # TODO: now the batch selected from the val loader is not really random. Fix it, so this batch is random.
    model.eval()

    batch = next(iter(val_loader))
    pixel_values = preprocess_batch(batch, cfg=cfg, device=device)  # pixel values is of shape (3,224,224)

    outputs = model(pixel_values=pixel_values)

    if not hasattr(outputs, "logits"):
        raise TypeError("This model does not output `logits`. For reconstructions you need ViTMAEForPreTraining.")
    if not hasattr(outputs, "mask"):
        raise TypeError("This model does not output `mask`. For reconstructions you need ViTMAEForPreTraining.")

    pred_patches = outputs.logits  # (B, L, p*p*3)
    mask = outputs.mask  # (B, L), 1 = masked, 0 = keep

    p = model.config.patch_size
    img_size = model.config.image_size

    # Original patches
    target_patches = patchify(pixel_values, patch_size=p)

    # Combine: keep original visible patches, fill masked patches with predictions
    mask_ = mask.unsqueeze(-1).type_as(pred_patches)  # (B, L, 1)
    recon_patches = target_patches * (1.0 - mask_) + pred_patches * mask_
    recon_imgs = unpatchify(recon_patches, patch_size=p, img_size=img_size)

    # Build a pixel-level mask image for visualization
    mask_patches = mask_.repeat(1, 1, p * p * 3)  # (B, L, p*p*3)
    mask_imgs = unpatchify(mask_patches, patch_size=p, img_size=img_size)  # (B,3,H,W)
    mask_imgs = mask_imgs[:, :1]  # (B,1,H,W) just one channel

    # Masked input (what encoder sees): gray out masked regions
    masked_imgs = pixel_values * (1.0 - mask_imgs) + 0.5 * mask_imgs

    # Move to CPU for plotting
    orig = pixel_values.detach().cpu()
    masked = masked_imgs.detach().cpu()
    recon = recon_imgs.detach().cpu()
    mimg = mask_imgs.detach().cpu()

    # Clamp for display
    orig = orig.clamp(0, 1)
    masked = masked.clamp(0, 1)
    recon = recon.clamp(0, 1)

    B = orig.shape[0]
    n = min(n_show, B)

    _, axes = plt.subplots(n, 4, figsize=(8, 2 * n))
    if n == 1:
        axes = axes[None, :]

    for i in range(n):
        o = orig[i].permute(1, 2, 0)  # HWC, e.g. (224,224,3)
        ms = masked[i].permute(1, 2, 0)
        r = recon[i].permute(1, 2, 0)
        mi = mimg[i, 0]  # HW, e.g. (224,224)

        axes[i, 0].imshow(o)
        axes[i, 0].set_title("Original")
        axes[i, 1].imshow(ms)
        axes[i, 1].set_title("Masked input")
        axes[i, 2].imshow(r)
        axes[i, 2].set_title("Reconstruction")
        axes[i, 3].imshow(mi)
        axes[i, 3].set_title("Mask (1=masked)")

        for j in range(4):
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.show()
