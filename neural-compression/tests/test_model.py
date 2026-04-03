"""Unit tests for model.py — NeuralCodec class."""

import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model import NeuralCodec


class TestNeuralCodecInit:
    """Tests for NeuralCodec construction and checkpoint loading."""

    def test_instantiation(self) -> None:
        codec = NeuralCodec()
        assert codec.model is not None
        assert codec.model.N == 128
        assert codec.model.M == 128

    def test_checkpoint_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            NeuralCodec(checkpoint_path="nonexistent_checkpoint.pth")

    def test_none_checkpoint_uses_random_init(self) -> None:
        codec = NeuralCodec(checkpoint_path=None)
        assert codec.model is not None

    def test_checkpoint_loading(self, tmp_path) -> None:
        codec1 = NeuralCodec()
        ckpt_path = str(tmp_path / "test_ckpt.pth")
        torch.save({"model_state_dict": codec1.model.state_dict()}, ckpt_path)

        codec2 = NeuralCodec(checkpoint_path=ckpt_path)
        for (k1, v1), (k2, v2) in zip(
            codec1.model.state_dict().items(),
            codec2.model.state_dict().items(),
        ):
            assert k1 == k2
            assert torch.equal(v1, v2), f"Mismatch at {k1}"


class TestPadImage:
    """Tests for pad_image / unpad_image."""

    def test_dimensions_are_multiples_of_factor(self) -> None:
        codec = NeuralCodec()
        x = torch.randn(1, 3, 100, 150)
        result = codec.pad_image(x, 16)

        assert result.original_height == 100
        assert result.original_width == 150
        assert result.tensor.shape[2] % 16 == 0
        assert result.tensor.shape[3] % 16 == 0
        assert result.tensor.shape[2] >= 100
        assert result.tensor.shape[3] >= 150

    def test_already_aligned_adds_no_padding(self) -> None:
        codec = NeuralCodec()
        x = torch.randn(1, 3, 128, 256)
        result = codec.pad_image(x, 16)

        assert result.tensor.shape == x.shape
        assert torch.equal(result.tensor, x)

    def test_round_trip(self) -> None:
        codec = NeuralCodec()
        x = torch.randn(1, 3, 100, 150)
        result = codec.pad_image(x, 16)
        unpadded = codec.unpad_image(
            result.tensor, result.original_height, result.original_width,
        )

        assert unpadded.shape == x.shape
        assert torch.equal(unpadded, x)


class TestGetOptimizers:
    """Tests for dual optimizer creation."""

    def test_returns_two_adam_optimizers(self) -> None:
        codec = NeuralCodec()
        main_opt, aux_opt = codec.get_optimizers(lr=1e-4, aux_lr=1e-3)

        assert isinstance(main_opt, torch.optim.Adam)
        assert isinstance(aux_opt, torch.optim.Adam)
        assert main_opt.defaults["lr"] == 1e-4
        assert aux_opt.defaults["lr"] == 1e-3

    def test_no_parameter_overlap(self) -> None:
        codec = NeuralCodec()
        main_opt, aux_opt = codec.get_optimizers(lr=1e-4, aux_lr=1e-3)

        main_ids = {id(p) for p in main_opt.param_groups[0]["params"]}
        aux_ids = {id(p) for p in aux_opt.param_groups[0]["params"]}
        assert main_ids.isdisjoint(aux_ids)
