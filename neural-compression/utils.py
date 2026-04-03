"""Serialization and metrics utilities for the Neural Image Compression Pipeline.

Contains:
    - .Ramiro binary format serialization (pack/unpack)
    - Image quality metrics (PSNR, MS-SSIM, BPP)
"""

import struct
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# ============================================================================
# .Ramiro Binary Format
# ============================================================================

class RamiroFormat:
    """Constants for the .Ramiro binary file format.

    Layout: 24-byte header + length-prefixed byte streams.
    All integers are big-endian unsigned 32-bit.
    """

    MAGIC: bytes = b".Ramiro\x00"
    HEADER_STRUCT: str = ">4sIIIII"  # magic + orig_h + orig_w + pad_h + pad_w + num_streams
    HEADER_SIZE: int = 24
    STREAM_LENGTH_STRUCT: str = ">I"
    STREAM_LENGTH_SIZE: int = 4


@dataclass(frozen=True)
class RamiroHeader:
    """Parsed .Ramiro file header — like a C# record for the deserialized header."""
    original_height: int
    original_width: int
    padded_height: int
    padded_width: int
    strings: List[List[bytes]]  # CompressAI batch format: [[stream_0, stream_1, ...]]


def pack_bitstream(
    orig_h: int,
    orig_w: int,
    pad_h: int,
    pad_w: int,
    strings: List[List[bytes]],
) -> bytes:
    """Serialize compressed output to .Ramiro binary format.

    Args:
        orig_h, orig_w: Original image dimensions before padding.
        pad_h, pad_w: Padded image dimensions (multiples of 16).
        strings: CompressAI output, shape [[bytes_y, bytes_z]] (batch=1).

    Returns:
        Raw bytes: 24-byte header + length-prefixed byte strings.
    """
    # strings is [[y_bytes], [z_bytes]] - flatten to get all streams
    inner = []
    for stream_list in strings:
        inner.extend(stream_list)  # Add each stream's bytes

    # Build header manually: 8-byte magic + 5 uint32be integers = 28 bytes
    header = (
        b".Ramiro\x00"
        + struct.pack(">I", orig_h)
        + struct.pack(">I", orig_w)
        + struct.pack(">I", pad_h)
        + struct.pack(">I", pad_w)
        + struct.pack(">I", len(inner))
    )

    # Build payload: [header, len0, data0, len1, data1, ...]
    parts = [header]
    for stream in inner:
        parts.append(struct.pack(">I", len(stream)))
        parts.append(stream)

    return b"".join(parts)


def unpack_bitstream(data: bytes) -> RamiroHeader:
    """Deserialize .Ramiro binary format back to components.

    Returns:
        .RamiroHeader with original dims, padded dims, and CompressAI-format strings.

    Raises:
        ValueError: If magic number doesn't match or file is truncated.
    """
    if len(data) < 28:
        raise ValueError(
            f"Invalid .Ramiro file: file too short for header "
            f"(expected >= 28 bytes)"
        )

    magic = data[:8]
    if magic != b".Ramiro\x00":
        raise ValueError(
            f"Invalid .Ramiro file: magic number mismatch "
            f"(expected b'.Ramiro\\x00', got {magic!r})"
        )

    orig_h, orig_w, pad_h, pad_w, num_streams = struct.unpack(
        ">IIIII", data[8:28]
    )

    offset = 28
    inner: List[bytes] = []

    for _ in range(num_streams):
        if offset + 4 > len(data):
            raise ValueError("Invalid .Ramiro file: unexpected end of stream data")

        (length,) = struct.unpack(">I", data[offset : offset + 4])
        offset += 4

        if offset + length > len(data):
            raise ValueError("Invalid .Ramiro file: unexpected end of stream data")

        inner.append(data[offset : offset + length])
        offset += length

    # CompressAI decompress expects strings as [y_bytes, z_bytes] (flat list of 2)
    return RamiroHeader(
        original_height=orig_h,
        original_width=orig_w,
        padded_height=pad_h,
        padded_width=pad_w,
        strings=[inner[0], inner[1]] if len(inner) >= 2 else [b'', b''],
    )


# ============================================================================
# Image Quality Metrics
# ============================================================================

class Metrics:
    """Static helper methods for image quality measurement.

    All methods expect float32 HxWx3 numpy arrays in [0, 1].
    """

    @staticmethod
    def compute_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
        """PSNR via skimage with data_range=1.0."""
        return float(
            peak_signal_noise_ratio(original, reconstructed, data_range=1.0)
        )

    @staticmethod
    def compute_ms_ssim(original: np.ndarray, reconstructed: np.ndarray) -> float:
        """MS-SSIM via skimage with gaussian weights, sigma=1.5."""
        return float(structural_similarity(
            original, reconstructed,
            data_range=1.0,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
            channel_axis=2,
        ))

    @staticmethod
    def compute_bpp(size_bytes: int, height: int, width: int) -> float:
        """Bits per pixel from physical compressed size."""
        return (size_bytes * 8) / (height * width)


# ============================================================================
# Backward-compatible module-level aliases
# ============================================================================

compute_psnr = Metrics.compute_psnr
compute_ms_ssim = Metrics.compute_ms_ssim
compute_bpp = Metrics.compute_bpp
