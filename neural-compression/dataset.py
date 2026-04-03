"""Image dataset loader for the Neural Image Compression Pipeline.

Recursively discovers images, filters by minimum size, and yields
normalized random crops as training tensors.
"""

import logging
import os
from pathlib import Path
from typing import List

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


logger = logging.getLogger(__name__)


class ImageDataset(Dataset):
    """PyTorch Dataset that discovers and serves random image crops.

    Walks a directory tree for .png/.jpg/.jpeg files, filters out images
    smaller than patch_size in either dimension, and yields [3, P, P]
    float32 tensors in [0, 1] via RandomCrop + ToTensor.
    """

    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

    def __init__(self, root: str, patch_size: int = 256, augment: bool = False) -> None:
        self.root = root
        self.patch_size = patch_size
        self.augment = augment

        self.image_paths: List[str] = self._discover_images(root, patch_size)

        if not self.image_paths:
            raise RuntimeError(
                f"No valid images found in {root} "
                f"(minimum size: {patch_size}x{patch_size})"
            )

        logger.info(f"Found {len(self.image_paths)} valid images in {root}")

        if augment:
            self._transform = transforms.Compose([
                transforms.RandomCrop(patch_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
            ])
        else:
            self._transform = transforms.Compose([
                transforms.RandomCrop(patch_size),
                transforms.ToTensor(),
            ])

    # region Private helpers

    @classmethod
    def _discover_images(cls, root: str, min_size: int) -> List[str]:
        """Recursively find images meeting the minimum dimension requirement."""
        valid_paths: List[str] = []

        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                ext = Path(fname).suffix.lower()
                if ext not in cls.SUPPORTED_EXTENSIONS:
                    continue

                full_path = os.path.join(dirpath, fname)
                try:
                    with Image.open(full_path) as img:
                        width, height = img.size
                        if width >= min_size and height >= min_size:
                            valid_paths.append(full_path)
                except Exception:
                    continue  # Skip corrupt or unreadable files

        return valid_paths

    # endregion

    # region Dataset interface

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Returns a [3, patch_size, patch_size] float32 tensor in [0, 1]."""
        img = Image.open(self.image_paths[index]).convert("RGB")
        return self._transform(img)

    # endregion
