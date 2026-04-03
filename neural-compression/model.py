"""Neural codec model wrapper for the Neural Image Compression Pipeline.

Provides a clean facade over CompressAI's MeanScaleHyperprior, handling
padding, unpadding, checkpoint loading, and dual-optimizer creation.
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from compressai.models import MeanScaleHyperprior


# region Result types (like C# records / DTOs)

@dataclass(frozen=True)
class PadResult:
    """Result of pad_image — carries the padded tensor and original dimensions."""
    tensor: torch.Tensor
    original_height: int
    original_width: int

# endregion


class NeuralCodec:
    """Wrapper around CompressAI MeanScaleHyperprior(N=192, M=192).

    Responsibilities:
        - Model instantiation with optional checkpoint loading
        - Asymmetric mirror-padding / unpadding for factor-of-16 alignment
        - Dual-optimizer creation (main vs. auxiliary parameters)
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        N: int = 128,
        M: int = 128,
    ) -> None:
        self.N = N
        self.M = M
        self.model = MeanScaleHyperprior(N=N, M=M)

        if checkpoint_path is not None:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            state_dict = state.get("model_state_dict", state)
            self.model.load_state_dict(state_dict)

    # region Padding

    def pad_image(self, x: torch.Tensor, factor: int) -> PadResult:
        """Apply asymmetric reflect-padding so H and W become multiples of factor.

        Padding is applied exclusively to the right and bottom edges.
        """
        _, _, h, w = x.shape
        pad_h = (factor - h % factor) % factor
        pad_w = (factor - w % factor) % factor
        padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return PadResult(tensor=padded, original_height=h, original_width=w)

    @staticmethod
    def unpad_image(x: torch.Tensor, orig_h: int, orig_w: int) -> torch.Tensor:
        """Slice a padded tensor back to its original spatial dimensions."""
        return x[:, :, :orig_h, :orig_w]

    # endregion

    # region Optimizers

    def get_optimizers(
        self, lr: float, aux_lr: float
    ) -> Tuple[torch.optim.Adam, torch.optim.Adam]:
        """Create dual Adam optimizers with non-overlapping parameter sets.

        Returns:
            (main_optimizer, aux_optimizer) — main covers network weights,
            aux covers entropy bottleneck quantile parameters.
        """
        aux_params = self._collect_aux_parameters()
        aux_param_ids = {id(p) for p in aux_params}
        main_params = [p for p in self.model.parameters() if id(p) not in aux_param_ids]

        return (
            torch.optim.Adam(main_params, lr=lr),
            torch.optim.Adam(aux_params, lr=aux_lr),
        )

    def _collect_aux_parameters(self) -> List[torch.nn.Parameter]:
        """Collect auxiliary parameters from entropy bottleneck modules."""
        seen: set = set()
        params: List[torch.nn.Parameter] = []

        for module in self.model.modules():
            if not hasattr(module, "quantiles"):
                continue
            for p in module.parameters():
                if p.requires_grad and id(p) not in seen:
                    params.append(p)
                    seen.add(id(p))

        return params

    # endregion
