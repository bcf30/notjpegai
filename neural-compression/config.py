"""Centralized configuration for the Neural Image Compression Pipeline."""

from dataclasses import dataclass


@dataclass
class TrainConfig:
    """Hyperparameters and constants for the neural compression pipeline.

    Analogous to an appsettings/options class — single source of truth
    for all tunable values across the pipeline.
    """

    # Rate-distortion trade-off weight (higher = more compression)
    lambda_val: float = 0.01

    # Training crop size and minimum image dimension
    patch_size: int = 256

    # Training batch size (8 fits in 6 GB VRAM with 256×256 crops on CUDA)
    batch_size: int = 8

    # Main network learning rate
    learning_rate: float = 1e-4

    # Auxiliary (entropy model) learning rate
    aux_learning_rate: float = 1e-3

    # Maximum training epochs
    max_epochs: int = 100

    # Spatial dimension alignment factor for encoder/decoder
    # MeanScaleHyperprior has 4 downsampling layers (2^4=16) plus hyperprior needs factor of 2
    pad_factor: int = 128

    # Maximum gradient norm for main optimizer clipping
    grad_clip_max_norm: float = 1.0

    # CompressAI MeanScaleHyperprior channel parameters
    # N=128, M=128 keeps peak VRAM under ~4 GB on a 6 GB card
    N: int = 128
    M: int = 128

    # Enable data augmentation (flips, rotations)
    augment: bool = False

    # Early stopping patience (epochs without improvement)
    early_stop_patience: int = 20

    # Use OneCycleLR scheduler instead of CosineAnnealingLR
    use_onecycle: bool = False
