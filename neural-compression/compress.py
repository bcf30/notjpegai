"""Compressor CLI — compress a single image to .Ramiro binary format.

Usage:
    python compress.py -i input.png -o output.ramiro -c checkpoint.pth
"""

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Compress an image to .Ramiro format")
    parser.add_argument("-i", required=True, help="Input image path")
    parser.add_argument("-o", required=True, help="Output .ramiro file path")
    parser.add_argument("-c", required=True, help="Model checkpoint path")
    args = parser.parse_args()

    try:
        # Validate input
        if not os.path.exists(args.i):
            raise FileNotFoundError(f"Input image not found: {args.i}")

        logger.info(f"Loading input image: {args.i}")
        config = TrainConfig()

        # Load image → RGB → [1, 3, H, W] float32 in [0, 1]
        img = Image.open(args.i).convert("RGB")
        tensor = ToTensor()(img).unsqueeze(0)
        logger.info(f"Image loaded, dimensions: {img.size[0]}x{img.size[1]}")

        # Load model
        logger.info(f"Loading checkpoint: {args.c}")
        codec = NeuralCodec(checkpoint_path=args.c, N=config.N, M=config.M)
        codec.model.update(force=True)
        codec.model.eval()
        logger.info("Model loaded and ready for compression")

        with torch.no_grad():
            pad_result = codec.pad_image(tensor, config.pad_factor)
            padded = pad_result.tensor
            orig_h = pad_result.original_height
            orig_w = pad_result.original_width
            pad_h, pad_w = padded.shape[2], padded.shape[3]

            logger.info(f"Compressing ({orig_h}x{orig_w}) -> padded ({pad_h}x{pad_w})...")
            out = codec.model.compress(padded)
            logger.info("Compression complete")

        # Serialize and write
        data = pack_bitstream(orig_h, orig_w, pad_h, pad_w, out["strings"])
        with open(args.o, "wb") as f:
            f.write(data)

        # Report
        file_size = len(data)
        bpp = Metrics.compute_bpp(file_size, orig_h, orig_w)
        logger.info(f"Output written to: {args.o}")
        logger.info(f"Original dimensions: {orig_h}x{orig_w}")
        logger.info(f"Compressed file size: {file_size} bytes ({file_size / 1024:.2f} KB)")
        logger.info(f"BPP: {bpp:.4f}")

    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error(f"Compression failed: {exc}")
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
