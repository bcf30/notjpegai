"""Compressor — compress a single image to .Ramiro binary format.

Usage:
    python compress.py input.png output.ramiro checkpoint.pth
"""

import logging
import os
import sys

import torch
from PIL import Image
from torchvision.transforms import ToTensor

from config import TrainConfig
from model import NeuralCodec
from utils import Metrics, pack_bitstream


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def compress(input_path: str, output_path: str, checkpoint_path: str) -> None:
    """Compress an image to .Ramiro format."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image not found: {input_path}")

    logger.info(f"Loading input image: {input_path}")
    config = TrainConfig()

    img = Image.open(input_path).convert("RGB")
    tensor = ToTensor()(img).unsqueeze(0)
    logger.info(f"Image loaded, dimensions: {img.size[0]}x{img.size[1]}")

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    codec = NeuralCodec(checkpoint_path=checkpoint_path, N=config.N, M=config.M)
    codec.model.update(force=True)
    codec.model.eval()

    with torch.no_grad():
        pad_result = codec.pad_image(tensor, config.pad_factor)
        padded = pad_result.tensor
        orig_h = pad_result.original_height
        orig_w = pad_result.original_width
        pad_h, pad_w = padded.shape[2], padded.shape[3]

        logger.info(f"Compressing ({orig_h}x{orig_w}) -> padded ({pad_h}x{pad_w})...")
        out = codec.model.compress(padded)
        latent_h, latent_w = out["shape"]

    data = pack_bitstream(orig_h, orig_w, pad_h, pad_w, latent_h, latent_w, out["strings"])
    with open(output_path, "wb") as f:
        f.write(data)

    file_size = len(data)
    bpp = Metrics.compute_bpp(file_size, orig_h, orig_w)
    logger.info(f"Output: {output_path}")
    logger.info(f"Size: {file_size} bytes ({file_size / 1024:.2f} KB), BPP: {bpp:.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python compress.py <input_image> <output.ramiro> <checkpoint.pth>")
        sys.exit(1)
    try:
        compress(sys.argv[1], sys.argv[2], sys.argv[3])
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error(f"Compression failed: {exc}")
        sys.exit(1)
