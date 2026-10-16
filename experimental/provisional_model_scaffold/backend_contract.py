from __future__ import annotations
from typing import Protocol
import torch


class ACTBackend(Protocol):
    """Expected adapter interface for the verified historical ACT/CVAE backend."""

    def __call__(
        self,
        visual_tokens: torch.Tensor,
        qpos: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return [B, chunk_size, action_dim]."""
        ...
