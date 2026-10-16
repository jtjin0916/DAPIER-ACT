from __future__ import annotations
import torch
from torch import nn


class DP3InspiredPointEncoder(nn.Module):
    """Simple global PointCloud encoder inspired by DP3-style point features.

    Input:  [B, N, 3]
    Output: [B, 1, D]
    """

    def __init__(self, out_dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError(f"Expected [B,N,3], got {tuple(points.shape)}")
        feat = self.mlp(points)
        global_feat = feat.max(dim=1).values
        return global_feat.unsqueeze(1)
