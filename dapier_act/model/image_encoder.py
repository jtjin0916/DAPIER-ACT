from __future__ import annotations
import torch
from torch import nn


class WristResNet18Encoder(nn.Module):
    """Independent wrist-camera encoder.

    Exact image preprocessing must be matched to historical training code before
    loading old weights.
    """

    def __init__(self, out_dim: int = 256, pretrained: bool = True) -> None:
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        net = resnet18(weights=weights)
        self.backbone = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool,
            net.layer1, net.layer2, net.layer3, net.layer4,
        )
        self.proj = nn.Conv2d(512, out_dim, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"Expected [B,3,H,W], got {tuple(image.shape)}")
        feat = self.proj(self.backbone(image))
        return feat.flatten(2).transpose(1, 2)
