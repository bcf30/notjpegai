"""Training loop for the Neural Image Compression Pipeline.

Implements mixed-precision training with dual optimizers (main + auxiliary),
gradient clipping on main params only, and CosineAnnealingLR scheduling.
"""

DEFAULT_OUTPUT_DIR = "../checkpoints"
CHECKPOINT_NAME = "best_model.pth"

import argparse
import logging
import math
import os
from dataclasses import dataclass
from typing import Tuple

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


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

_LOG2 = math.log(2)


# ============================================================================
# Result types
# ============================================================================

@dataclass(frozen=True)
class ValidationResult:
    """Validation metrics for a single epoch."""
    avg_loss: float
    avg_psnr: float
    avg_ms_ssim: float


# ============================================================================
# Core training functions
# ============================================================================

def compute_rate(out_net: dict, num_pixels: int) -> torch.Tensor:
    """Compute estimated bitrate from model likelihoods.

    Formula (per spec Req 5.5):
        rate = (log(sum(y) + eps) + log(sum(z) + eps)) / (-log(2) * num_pixels)
    """
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
    """Run one training epoch with AMP and dual optimizers.

    Per-batch steps:
        1. Pad → autocast forward → MSE distortion + rate
        2. rd_loss = distortion + lambda_val * rate
        3. Scaled backward → unscale → clip main params → step main → update scaler
        4. aux_loss backward → step aux (outside autocast, after scaler update)

    Returns:
        Average RD loss over the epoch.
    """
    codec.model.train()
    total_loss = 0.0
    num_batches = 0

    # Collect main params once for gradient clipping (excludes aux params)
    main_params = [p for group in main_optimizer.param_groups for p in group["params"]]

    for batch in tqdm(dataloader, desc="Training", leave=False):
        batch = batch.to(device)

        pad_result = codec.pad_image(batch, config.pad_factor)
        padded = pad_result.tensor
        num_pixels = padded.shape[0] * padded.shape[2] * padded.shape[3]

        # --- Main RD loss with AMP ---
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

        # --- Auxiliary loss (outside autocast, after scaler update) ---
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
    """Run validation: compute average loss, PSNR, and MS-SSIM."""
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

            # RD loss (same formula as training)
            distortion = nn.functional.mse_loss(out_net["x_hat"], padded)
            rate = compute_rate(out_net, num_pixels)
            rd_loss = distortion + config.lambda_val * rate
            total_loss += rd_loss.item()

            # Per-image metrics on unpadded, clamped reconstructions
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


# ============================================================================
# CLI entry point
# ============================================================================

def main() -> None:
    """Parse CLI args, set up training, run loop, save best checkpoint."""
    parser = argparse.ArgumentParser(description="Train Neural Image Compression model")
    parser.add_argument("--dataset", type=str, required=True, help="Training image directory")
    parser.add_argument("--val_dataset", type=str, required=True, help="Validation image directory")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--lambda", type=float, default=None, dest="lambda_val", help="RD lambda")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Checkpoint directory")
    parser.add_argument("--N", type=int, default=None, help="Model N channels")
    parser.add_argument("--M", type=int, default=None, help="Model M channels")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")
    parser.add_argument("--onecycle", action="store_true", help="Use OneCycleLR scheduler")
    parser.add_argument("--early_stop", type=int, default=None, help="Early stopping patience")
    args = parser.parse_args()

    # Apply CLI overrides to config
    config = TrainConfig()
    if args.lambda_val is not None:
        config.lambda_val = args.lambda_val
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.epochs is not None:
        config.max_epochs = args.epochs
    if args.N is not None:
        config.N = args.N
    if args.M is not None:
        config.M = args.M
    if args.augment:
        config.augment = True
    if args.onecycle:
        config.use_onecycle = True
    if args.early_stop is not None:
        config.early_stop_patience = args.early_stop

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # Turing conv autotuning
    logger.info(f"=== Training Configuration ===")
    logger.info(f"Device: {device}")
    logger.info(f"Lambda: {config.lambda_val}, Batch size: {config.batch_size}, Epochs: {config.max_epochs}")
    logger.info(f"Model: N={config.N}, M={config.M}")
    logger.info(f"Learning rate: {config.learning_rate}, Aux LR: {config.aux_learning_rate}")
    logger.info(f"Patch size: {config.patch_size}, Pad factor: {config.pad_factor}")
    logger.info(f"Augmentation: {config.augment}, OneCycleLR: {config.use_onecycle}")
    logger.info(f"Early stopping patience: {config.early_stop_patience}")
    logger.info(f"Output directory: {args.output_dir}")

    # Data
    logger.info(f"Loading training dataset: {args.dataset}")
    train_dataset = ImageDataset(args.dataset, patch_size=config.patch_size, augment=config.augment)
    logger.info(f"Loading validation dataset: {args.val_dataset}")
    val_dataset = ImageDataset(args.val_dataset, patch_size=config.patch_size, augment=False)
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
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model + optimizers + scheduler + scaler
    logger.info(f"Initializing model (N={config.N}, M={config.M})")
    codec = NeuralCodec(N=config.N, M=config.M)
    codec.model.to(device)
    main_optimizer, aux_optimizer = codec.get_optimizers(
        config.learning_rate, config.aux_learning_rate,
    )
    
    if config.use_onecycle:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            main_optimizer, max_lr=config.learning_rate * 10,
            epochs=config.max_epochs, steps_per_epoch=len(train_loader),
            pct_start=0.1, anneal_strategy='cos'
        )
    else:
        scheduler = CosineAnnealingLR(main_optimizer, T_max=config.max_epochs)
    
    scaler = GradScaler(device.type)
    logger.info("Model and optimizers initialized")

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, config.max_epochs + 1):
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
            ckpt_path = os.path.join(args.output_dir, CHECKPOINT_NAME)
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
                logger.info(f"Early stopping triggered after {epoch} epochs (no improvement for {config.early_stop_patience} epochs)")
                break

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
