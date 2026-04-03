"""Unit tests for ImageDataset class."""

import os
import tempfile

import pytest
import torch
from PIL import Image

from dataset import ImageDataset


@pytest.fixture
def image_dir():
    """Create a temp directory with test images of various sizes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Large enough image (64x64)
        img_large = Image.new("RGB", (64, 64), color=(128, 64, 32))
        img_large.save(os.path.join(tmpdir, "large.png"))

        # Another large image as JPEG
        img_jpg = Image.new("RGB", (100, 80), color=(10, 20, 30))
        img_jpg.save(os.path.join(tmpdir, "photo.jpg"))

        # Too-small image (should be filtered out)
        img_small = Image.new("RGB", (30, 30), color=(255, 0, 0))
        img_small.save(os.path.join(tmpdir, "tiny.png"))

        # Image in a subdirectory (recursive discovery)
        subdir = os.path.join(tmpdir, "subdir")
        os.makedirs(subdir)
        img_sub = Image.new("RGB", (64, 64), color=(0, 255, 0))
        img_sub.save(os.path.join(subdir, "nested.JPEG"))

        yield tmpdir


def test_discovers_valid_images(image_dir):
    """Should find 3 valid images (>=64x64), filtering out the 30x30 one."""
    ds = ImageDataset(image_dir, patch_size=64)
    assert len(ds) == 3


def test_filters_small_images(image_dir):
    """With patch_size=100, only the 100x80 image qualifies on width but not height."""
    # Only the 100x80 image has width>=100, but height=80 < 100, so it's filtered.
    # No images qualify → RuntimeError
    with pytest.raises(RuntimeError, match="No valid images found"):
        ImageDataset(image_dir, patch_size=101)


def test_getitem_shape_and_dtype(image_dir):
    """__getitem__ should return [3, patch_size, patch_size] float32 in [0,1]."""
    patch_size = 64
    ds = ImageDataset(image_dir, patch_size=patch_size)
    tensor = ds[0]
    assert tensor.shape == (3, patch_size, patch_size)
    assert tensor.dtype == torch.float32
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def test_converts_non_rgb_to_3_channels():
    """Grayscale and RGBA images should be converted to 3-channel RGB."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Grayscale image
        gray = Image.new("L", (64, 64), color=128)
        gray.save(os.path.join(tmpdir, "gray.png"))

        # RGBA image
        rgba = Image.new("RGBA", (64, 64), color=(128, 64, 32, 200))
        rgba.save(os.path.join(tmpdir, "alpha.png"))

        ds = ImageDataset(tmpdir, patch_size=64)
        assert len(ds) == 2
        for i in range(len(ds)):
            t = ds[i]
            assert t.shape[0] == 3


def test_runtime_error_on_empty_dir():
    """Should raise RuntimeError when no valid images exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(RuntimeError, match="No valid images found"):
            ImageDataset(tmpdir, patch_size=64)


def test_case_insensitive_extensions():
    """Should discover images with uppercase extensions like .PNG, .JPG."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img = Image.new("RGB", (64, 64), color=(0, 0, 255))
        img.save(os.path.join(tmpdir, "upper.PNG"))

        ds = ImageDataset(tmpdir, patch_size=64)
        assert len(ds) == 1
