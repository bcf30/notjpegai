"""Evaluator CLI — benchmark neural codec vs JPEG baseline.

Produces rate-distortion curves and CSV output comparing a single
neural codec checkpoint against JPEG at multiple quality levels.

Usage:
    python evaluate.py -i /path/to/test -c checkpoint.pth -rd rd.png -csv results.csv
"""

import io
import logging
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor

from config import TrainConfig
from model import NeuralCodec
from utils import Metrics, pack_bitstream, unpack_bitstream


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Result types
# ============================================================================

@dataclass(frozen=True)
class NeuralResult:
    """Aggregate evaluation result for the neural codec."""
    mean_bpp: float
    mean_psnr: float
    mean_ms_ssim: float


@dataclass(frozen=True)
class JpegResult:
    """Aggregate evaluation result for a single JPEG quality level."""
    quality: int
    mean_bpp: float
    mean_psnr: float
    mean_ms_ssim: float


# ============================================================================
# Evaluation functions
# ============================================================================

def evaluate_neural(
    codec: NeuralCodec,
    image_paths: List[str],
    config: TrainConfig,
    device: torch.device,
) -> NeuralResult:
    """Evaluate a single checkpoint across all test images.

    For each image: compress → pack → measure byte size for BPP →
    unpack → decompress → compute PSNR and MS-SSIM.
    All operations happen entirely in memory.
    """
    codec.model.update(force=True)
    codec.model.eval()
    codec.model.to(device)

    bpps: List[float] = []
    psnrs: List[float] = []
    ms_ssims: List[float] = []
    to_tensor = ToTensor()

    with torch.no_grad():
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            original_np = np.array(img).astype(np.float32) / 255.0
            tensor = to_tensor(img).unsqueeze(0).to(device)

            # Pad → compress → pack
            pad_result = codec.pad_image(tensor, config.pad_factor)
            padded = pad_result.tensor
            orig_h, orig_w = pad_result.original_height, pad_result.original_width
            pad_h, pad_w = padded.shape[2], padded.shape[3]

            out = codec.model.compress(padded)
            packed_bytes = pack_bitstream(orig_h, orig_w, pad_h, pad_w, out["strings"])

            bpps.append(Metrics.compute_bpp(len(packed_bytes), orig_h, orig_w))

            # Unpack → decompress → unpad → clamp
            header = unpack_bitstream(packed_bytes)
            latent_shape = (
                1, config.M,
                header.padded_height // config.pad_factor,
                header.padded_width // config.pad_factor,
            )
            recon = codec.model.decompress(header.strings, latent_shape)
            x_hat = recon["x_hat"].clamp(0, 1)
            x_hat = codec.unpad_image(x_hat, orig_h, orig_w)

            recon_np = x_hat[0].cpu().permute(1, 2, 0).numpy().astype(np.float32)
            psnrs.append(Metrics.compute_psnr(original_np, recon_np))
            ms_ssims.append(Metrics.compute_ms_ssim(original_np, recon_np))

    return NeuralResult(
        mean_bpp=float(np.mean(bpps)),
        mean_psnr=float(np.mean(psnrs)),
        mean_ms_ssim=float(np.mean(ms_ssims)),
    )


def evaluate_jpeg(
    image_paths: List[str],
    quality_levels: List[int],
) -> List[JpegResult]:
    """Evaluate JPEG baseline at multiple quality levels.

    Images are loaded once and reused across all quality levels.
    All JPEG compression/decompression happens in-memory via BytesIO.
    """
    # Pre-load all images once (avoids N * Q disk reads)
    loaded_images: List[Tuple[Image.Image, np.ndarray, int, int]] = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        original_np = np.array(img).astype(np.float32) / 255.0
        h, w = original_np.shape[:2]
        loaded_images.append((img, original_np, h, w))

    results: List[JpegResult] = []

    for quality in quality_levels:
        bpps: List[float] = []
        psnrs: List[float] = []
        ms_ssims: List[float] = []

        for img, original_np, h, w in loaded_images:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            size_bytes = buf.tell()

            buf.seek(0)
            recon_np = (
                np.array(Image.open(buf).convert("RGB")).astype(np.float32) / 255.0
            )

            bpps.append(Metrics.compute_bpp(size_bytes, h, w))
            psnrs.append(Metrics.compute_psnr(original_np, recon_np))
            ms_ssims.append(Metrics.compute_ms_ssim(original_np, recon_np))

        results.append(JpegResult(
            quality=quality,
            mean_bpp=float(np.mean(bpps)),
            mean_psnr=float(np.mean(psnrs)),
            mean_ms_ssim=float(np.mean(ms_ssims)),
        ))

    return results


# ============================================================================
# Plotting
# ============================================================================

def plot_rd_curves(
    neural: NeuralResult,
    jpeg_results: List[JpegResult],
    output_path: str,
) -> None:
    """Create matplotlib figure: BPP vs PSNR (left) and BPP vs MS-SSIM (right).

    JPEG: dotted blue line with markers.
    Neural: single large red dot.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    jpeg_bpps = [r.mean_bpp for r in jpeg_results]
    jpeg_psnrs = [r.mean_psnr for r in jpeg_results]
    jpeg_ms_ssims = [r.mean_ms_ssim for r in jpeg_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, jpeg_y, neural_y, ylabel in [
        (ax1, jpeg_psnrs, neural.mean_psnr, "PSNR (dB)"),
        (ax2, jpeg_ms_ssims, neural.mean_ms_ssim, "MS-SSIM"),
    ]:
        ax.plot(jpeg_bpps, jpeg_y, "b--o", label="JPEG", markersize=4)
        ax.plot(neural.mean_bpp, neural_y, "ro", markersize=12, label="Neural")
        ax.set_xlabel("BPP")
        ax.set_ylabel(ylabel)
        ax.set_title(f"BPP vs {ylabel}")
        ax.legend()
        ax.grid(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"RD curve saved to {output_path}")


# ============================================================================
# CLI entry point
# ============================================================================

JPEG_QUALITY_LEVELS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def _discover_images(root: str) -> List[str]:
    """Recursively find image files under a directory."""
    paths: List[str] = [
        os.path.join(dirpath, fname)
        for dirpath, _, filenames in os.walk(root)
        for fname in filenames
        if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS
    ]
    paths.sort()
    return paths


def main() -> None:
    import argparse
    import csv

    parser = argparse.ArgumentParser(description="Evaluate neural codec vs JPEG baseline")
    parser.add_argument("-i", required=True, help="Test image directory")
    parser.add_argument("-c", required=True, help="Single checkpoint path")
    parser.add_argument("-rd", default=None, help="Output path for RD curve plot")
    parser.add_argument("-csv", default=None, help="Output path for results CSV")
    args = parser.parse_args()

    logger.info("=== Evaluation Started ===")
    logger.info(f"Test directory: {args.i}")
    logger.info(f"Checkpoint: {args.c}")

    image_paths = _discover_images(args.i)
    if not image_paths:
        logger.warning(f"No images found in {args.i}")
        return

    logger.info(f"Found {len(image_paths)} test images")

    config = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Neural evaluation
    logger.info("Loading neural codec checkpoint...")
    codec = NeuralCodec(checkpoint_path=args.c, N=config.N, M=config.M)
    logger.info("Evaluating neural codec...")
    neural = evaluate_neural(codec, image_paths, config, device)
    logger.info(
        f"Neural result — BPP: {neural.mean_bpp:.4f}, "
        f"PSNR: {neural.mean_psnr:.2f} dB, "
        f"MS-SSIM: {neural.mean_ms_ssim:.4f}"
    )

    # JPEG evaluation
    logger.info("Evaluating JPEG baseline...")
    jpeg_results = evaluate_jpeg(image_paths, JPEG_QUALITY_LEVELS)
    for r in jpeg_results:
        logger.info(
            f"JPEG q={r.quality:3d} — BPP: {r.mean_bpp:.4f}, "
            f"PSNR: {r.mean_psnr:.2f} dB, "
            f"MS-SSIM: {r.mean_ms_ssim:.4f}"
        )

    # Plot
    if args.rd:
        logger.info(f"Generating RD curve: {args.rd}")
        plot_rd_curves(neural, jpeg_results, args.rd)

    # CSV
    if args.csv:
        logger.info(f"Writing CSV: {args.csv}")
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["codec", "quality_or_lambda", "avg_bpp", "avg_psnr", "avg_ms_ssim"])
            writer.writerow(["neural", config.lambda_val, neural.mean_bpp, neural.mean_psnr, neural.mean_ms_ssim])
            for r in jpeg_results:
                writer.writerow(["jpeg", r.quality, r.mean_bpp, r.mean_psnr, r.mean_ms_ssim])
        logger.info(f"CSV saved: {args.csv}")

    logger.info("=== Evaluation Complete ===")


if __name__ == "__main__":
    main()
