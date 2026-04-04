"""Resume training from an existing checkpoint.

Usage: python resume_training.py
Edit the settings below before running.
"""

# ============================================================================
# >>> EDIT THESE BEFORE RUNNING <<<
# ============================================================================
CHECKPOINT      = "../checkpoints/best_model.pth"
DATASET         = "../data/DIV2K_train_HR"
VAL_DATASET     = "../data/DIV2K_valid_HR"
EPOCHS          = 200       # total epochs (continues from checkpoint epoch)
OUTPUT_DIR      = "../checkpoints"
CHECKPOINT_NAME = "best_model.pth"
# ============================================================================

import logging
import math
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import TrainConfig
from dataset import ImageDataset
from model import NeuralCodec
from utils import Metrics


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_LOG2 = math.log(2)


@dataclass(frozen=True)
class ValidationResult:
    avg_loss: float
    avg_psnr: float
    avg_ms_ssim: float


def compute_rate(out_net: dict, num_pixels: int) -> torch.Tensor:
    likelihoods = out_net["likelihoods"]
    return (
        torch.log(likelihoods["y"].sum() + 1e-10)
        + torch.log(likelihoods["z"].sum() + 1e-10)
    ) / (-_LOG2 * num_pixels)


def train_one_epoch(
    codec: NeuralCodec,
    dataloader: DataLoader,
    main_optimizer: torch.optim.Adam,
    aux_optimizer: torch.optim.Adam,
    scaler: GradScaler,
    config: TrainConfig,
    device: torch.device,
) -> float:
    codec.model.train()
    total_loss = 0.0
    num_batches = 0
    main_params = [p for group in main_optimizer.param_groups for p in group["params"]]

    for batch in tqdm(dataloader, desc="Training", leave=False):
        batch = batch.to(device)
        pad_result = codec.pad_image(batch, config.pad_factor)
        padded = pad_result.tensor
        num_pixels = padded.shape[0] * padded.shape[2] * padded.shape[3]

        main_optimizer.zero_grad()
        aux_optimizer.zero_grad()

        with autocast(device_type=device.type):
            out_net = codec.model(padded)
            distortion = nn.functional.mse_loss(out_net["x_hat"], padded)
            rate = compute_rate(out_net, num_pixels)
            rd_loss = distortion + config.lambda_val * rate

        scaler.scale(rd_loss).backward()
        scaler.unscale_(main_optimizer)
        nn.utils.clip_grad_norm_(main_params, config.grad_clip_max_norm)
        scaler.step(main_optimizer)
        scaler.update()

        aux_loss = codec.model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        total_loss += rd_loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def validate(
    codec: NeuralCodec,
    dataloader: DataLoader,
    config: TrainConfig,
    device: torch.device,
) -> ValidationResult:
    codec.model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ms_ssim = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            batch = batch.to(device)
            pad_result = codec.pad_image(batch, config.pad_factor)
            padded = pad_result.tensor
            orig_h, orig_w = pad_result.original_height, pad_result.original_width
            num_pixels = padded.shape[0] * padded.shape[2] * padded.shape[3]

            with autocast(device_type=device.type):
                out_net = codec.model(padded)

            distortion = nn.functional.mse_loss(out_net["x_hat"], padded)
            rate = compute_rate(out_net, num_pixels)
            rd_loss = distortion + config.lambda_val * rate
            total_loss += rd_loss.item()

            x_hat = out_net["x_hat"].clamp(0.0, 1.0)
            original_unpadded = codec.unpad_image(batch, orig_h, orig_w)
            recon_unpadded = codec.unpad_image(x_hat, orig_h, orig_w)

            batch_size = original_unpadded.shape[0]
            batch_psnr = sum(
                Metrics.compute_psnr(
                    original_unpadded[i].cpu().float().numpy().transpose(1, 2, 0),
                    recon_unpadded[i].cpu().float().numpy().transpose(1, 2, 0),
                )
                for i in range(batch_size)
            )
            batch_ms_ssim = sum(
                Metrics.compute_ms_ssim(
                    original_unpadded[i].cpu().float().numpy().transpose(1, 2, 0),
                    recon_unpadded[i].cpu().float().numpy().transpose(1, 2, 0),
                )
                for i in range(batch_size)
            )

            total_psnr += batch_psnr / batch_size
            total_ms_ssim += batch_ms_ssim / batch_size
            num_batches += 1

    n = max(num_batches, 1)
    return ValidationResult(
        avg_loss=total_loss / n,
        avg_psnr=total_psnr / n,
        avg_ms_ssim=total_ms_ssim / n,
    )


def main() -> None:
    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    logger.info(f"Loading checkpoint: {CHECKPOINT}")
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)

    saved_config = ckpt.get("config", {})
    config = TrainConfig()
    if "N" in saved_config:
        config.N = saved_config["N"]
    if "M" in saved_config:
        config.M = saved_config["M"]
    if "lambda" in saved_config:
        config.lambda_val = saved_config["lambda"]
    config.max_epochs = EPOCHS

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    start_epoch = ckpt.get("epoch", 0) + 1
    logger.info(f"=== Resume Training ===")
    logger.info(f"Device: {device}")
    logger.info(f"Resuming from epoch {start_epoch}, best val loss: {ckpt.get('best_val_loss', 'unknown'):.6f}")
    logger.info(f"Model: N={config.N}, M={config.M}, Lambda: {config.lambda_val}")
    logger.info(f"Target epochs: {config.max_epochs}")
    logger.info(f"Output: {OUTPUT_DIR}/{CHECKPOINT_NAME}")

    # Data
    train_dataset = ImageDataset(DATASET, patch_size=config.patch_size, augment=config.augment)
    val_dataset = ImageDataset(VAL_DATASET, patch_size=config.patch_size, augment=False)
    use_cuda = device.type == "cuda"
    num_workers = 4 if use_cuda else 0
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
    )

    # Model + optimizers
    codec = NeuralCodec(N=config.N, M=config.M)
    codec.model.load_state_dict(ckpt["model_state_dict"])
    codec.model.to(device)

    main_optimizer, aux_optimizer = codec.get_optimizers(
        config.learning_rate, config.aux_learning_rate,
    )
    main_optimizer.load_state_dict(ckpt["main_optimizer_state_dict"])
    aux_optimizer.load_state_dict(ckpt["aux_optimizer_state_dict"])

    if config.use_onecycle:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            main_optimizer, max_lr=config.learning_rate * 10,
            epochs=config.max_epochs - start_epoch + 1,
            steps_per_epoch=len(train_loader),
            pct_start=0.1, anneal_strategy='cos'
        )
    else:
        scheduler = CosineAnnealingLR(main_optimizer, T_max=config.max_epochs)
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    scaler = GradScaler(device.type)

    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    epochs_without_improvement = 0

    for epoch in range(start_epoch, config.max_epochs + 1):
        logger.info(f"--- Epoch {epoch}/{config.max_epochs} ---")
        if device.type == "cuda":
            torch.cuda.empty_cache()

        train_loss = train_one_epoch(
            codec, train_loader, main_optimizer, aux_optimizer, scaler, config, device,
        )
        val = validate(codec, val_loader, config, device)

        if not config.use_onecycle:
            scheduler.step()

        logger.info(
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val.avg_loss:.6f} | "
            f"Val PSNR: {val.avg_psnr:.2f} dB | "
            f"Val MS-SSIM: {val.avg_ms_ssim:.4f}"
        )

        if val.avg_loss < best_val_loss:
            best_val_loss = val.avg_loss
            epochs_without_improvement = 0
            ckpt_path = os.path.join(OUTPUT_DIR, CHECKPOINT_NAME)
            torch.save({
                "model_state_dict": codec.model.state_dict(),
                "main_optimizer_state_dict": main_optimizer.state_dict(),
                "aux_optimizer_state_dict": aux_optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "config": {"N": config.N, "M": config.M, "lambda": config.lambda_val},
            }, ckpt_path)
            logger.info(f"New best model saved: {ckpt_path} (val_loss={best_val_loss:.6f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stop_patience:
                logger.info(
                    f"Early stopping triggered at epoch {epoch} — "
                    f"no improvement for {config.early_stop_patience} consecutive epochs. "
                    f"Best val loss was {best_val_loss:.6f}."
                )
                break

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
