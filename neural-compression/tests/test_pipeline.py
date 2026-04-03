"""Pipeline correctness tests for the Neural Image Compression Pipeline.

Four focused tests validating the highest-value correctness properties:
    1. Pad/unpad round-trip (Property 3)
    2. Bitstream serialization round-trip (Property 1)
    3. BPP formula correctness (Property 9)
    4. Invalid magic number rejection (Property 2)
"""

import sys
import os

import pytest
import torch

# Add parent directory to path for module imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model import NeuralCodec
from utils import pack_bitstream, unpack_bitstream, Metrics


class TestPadUnpadRoundTrip:
    """Property 3: pad_image → unpad_image preserves tensor content exactly.

    Validates: Requirements 3.4, 3.5, 7.4
    """

    def test_round_trip_preserves_content(self) -> None:
        codec = NeuralCodec()
        original = torch.randn(1, 3, 100, 150)

        pad_result = codec.pad_image(original, factor=16)
        restored = codec.unpad_image(
            pad_result.tensor,
            pad_result.original_height,
            pad_result.original_width,
        )

        assert restored.shape == original.shape
        assert torch.equal(restored, original)


class TestBitstreamRoundTrip:
    """Property 1: pack_bitstream → unpack_bitstream is lossless.

    Validates: Requirements 4.2, 4.3, 4.4, 4.6
    """

    def test_round_trip_preserves_all_fields(self) -> None:
        orig_h, orig_w = 100, 150
        pad_h, pad_w = 112, 160
        strings = [[b"stream_y", b"stream_z"]]

        packed = pack_bitstream(orig_h, orig_w, pad_h, pad_w, strings)
        header = unpack_bitstream(packed)

        assert header.original_height == orig_h
        assert header.original_width == orig_w
        assert header.padded_height == pad_h
        assert header.padded_width == pad_w

        # Must be in CompressAI batch format [[...]]
        assert isinstance(header.strings, list)
        assert len(header.strings) == 1
        assert isinstance(header.strings[0], list)
        assert header.strings[0] == strings[0]


class TestBppFormula:
    """Property 9: BPP = (size_bytes * 8) / (height * width).

    Validates: Requirements 8.5
    """

    def test_known_value(self) -> None:
        assert Metrics.compute_bpp(100, 10, 10) == 8.0


class TestInvalidMagicRejection:
    """Property 2: unpack_bitstream raises ValueError on wrong magic bytes.

    Validates: Requirements 4.5, 10.1
    """

    def test_wrong_magic_raises_value_error(self) -> None:
        bad_data = b"XXXX" + b"\x00" * 20
        with pytest.raises(ValueError):
            unpack_bitstream(bad_data)
