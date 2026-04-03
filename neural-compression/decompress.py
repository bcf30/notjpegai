"""Decompressor — decompress a .Ramiro file back to PNG.

Usage:
    python decompress.py input.ramiro output.png checkpoint.pth
"""

import logging
import os
import sys

import numpy as np
import torch
from PIL import Image

from config import TrainConfig
from model import NeuralCodec
from utils import unpack_bitstream


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def decompress(input_path: str, output_path: str, checkpoint_path: str) -> None:
    """Decompress a .Ramiro file to PNG."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input .Ramiro file not found: {input_path}")

    logger.info(f"Reading .Ramiro file: {input_path}")
    config = TrainConfig()

    with open(input_path, "rb") as f:
        data = f.read()

    header = unpack_bitstream(data)
    logger.info(f"Header: original {header.original_height}x{header.original_width}, "
                f"padded {header.padded_height}x{header.padded_width}")

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    codec = NeuralCodec(checkpoint_path=checkpoint_path, N=config.N, M=config.M)
    codec.model.update(force=True)
    codec.model.eval()

    with torch.no_grad():
        latent_shape = (header.latent_height, header.latent_width)
        logger.info(f"Decompressing latent shape: {latent_shape}...")
        out = codec.model.decompress(header.strings, latent_shape)

        x_hat = out["x_hat"].clamp(0, 1)
        x_hat = codec.unpad_image(x_hat, header.original_height, header.original_width)

    img_array = (x_hat[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(img_array).save(output_path)

    logger.info(f"Output: {output_path} ({header.original_height}x{header.original_width})")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python decompress.py <input.ramiro> <output.png> <checkpoint.pth>")
        sys.exit(1)
    try:
        decompress(sys.argv[1], sys.argv[2], sys.argv[3])
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error(f"Decompression failed: {exc}")
        sys.exit(1)
