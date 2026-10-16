from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn

from .point_encoder import DP3InspiredPointEncoder
from .image_encoder import WristResNet18Encoder


@dataclass
class FrontendOutput:
    visual_tokens: torch.Tensor
    head_token_count: int
    left_token_count: int
    right_token_count: int


class GeneralizedDAPIERACTFrontend(nn.Module):
    """Generalization-ready multimodal frontend.

    This is a refactor scaffold, not a claim of compatibility with an existing
    private checkpoint. Connect it to the verified ACT/CVAE backend only after a
    tensor-by-tensor regression test.
    """

    def __init__(self, embedding_dim: int = 256, pretrained_rgb: bool = True) -> None:
        super().__init__()
        self.head = DP3InspiredPointEncoder(embedding_dim)
        self.left = WristResNet18Encoder(embedding_dim, pretrained_rgb)
        self.right = WristResNet18Encoder(embedding_dim, pretrained_rgb)

    def forward(self, head_points, left_wrist_rgb, right_wrist_rgb) -> FrontendOutput:
        head = self.head(head_points)
        left = self.left(left_wrist_rgb)
        right = self.right(right_wrist_rgb)
        tokens = torch.cat([head, left, right], dim=1)
        return FrontendOutput(
            visual_tokens=tokens,
            head_token_count=head.shape[1],
            left_token_count=left.shape[1],
            right_token_count=right.shape[1],
        )
