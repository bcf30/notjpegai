"""Decompressor CLI — decompress a .Ramiro file back to PNG.

Usage:
    python decompress.py -i input.ramiro -o output.png -c checkpoint.pth
"""

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Decompress a .Ramiro file to PNG")
    parser.add_argument("-i", required=True, help="Input .ramiro file path")
    parser.add_argument("-o", required=True, help="Output .png file path")
    parser.add_argument("-c", required=True, help="Model checkpoint path")
    args = parser.parse_args()

    try:
        # Validate input
        if not os.path.exists(args.i):
            raise FileNotFoundError(f"Input .Ramiro file not found: {args.i}")

        logger.info(f"Reading .Ramiro file: {args.i}")
        config = TrainConfig()

        # Read and parse .Ramiro file
        with open(args.i, "rb") as f:
            data = f.read()

        header = unpack_bitstream(data)
        logger.info(f".Ramiro header parsed: original {header.original_height}x{header.original_width}, "
                    f"padded {header.padded_height}x{header.padded_width}")

        # Load model
        logger.info(f"Loading checkpoint: {args.c}")
        codec = NeuralCodec(checkpoint_path=args.c, N=config.N, M=config.M)
        codec.model.update(force=True)
        codec.model.eval()
        logger.info("Model loaded and ready for decompression")

        with torch.no_grad():
            # CompressAI decompress expects 4D latent-space shape:
            # (batch=1, channels=M, latent_H, latent_W)
            latent_shape = (
                1,
                config.M,
                header.padded_height // config.pad_factor,
                header.padded_width // config.pad_factor,
            )
            logger.info(f"Decompressing latent shape: {latent_shape}...")
            out = codec.model.decompress(header.strings, latent_shape)
            logger.info("Decompression complete")

            x_hat = out["x_hat"].clamp(0, 1)
            x_hat = codec.unpad_image(
                x_hat, header.original_height, header.original_width,
            )

        # Convert to PIL and save
        img_array = (
            x_hat[0].cpu().permute(1, 2, 0).numpy() * 255
        ).astype(np.uint8)
        Image.fromarray(img_array).save(args.o)

        logger.info(f"Output written to: {args.o}")
        logger.info(f"Output dimensions: {header.original_height}x{header.original_width}")

    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error(f"Decompression failed: {exc}")
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
