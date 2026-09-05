"""Single-trunk Omnivore classifier using its native four-channel RGB-D path."""

from __future__ import annotations

import torch
from torch import nn


class OmnivoreRGBDClassifier(nn.Module):
    def __init__(self, trunk: nn.Module, num_classes: int = 40) -> None:
        super().__init__()
        self.trunk = trunk
        self.classifier = nn.Sequential(
            nn.LayerNorm(768), nn.Dropout(0.2), nn.Linear(768, num_classes)
        )

    def forward(self, rgbd: torch.Tensor) -> torch.Tensor:
        if rgbd.ndim != 5 or rgbd.shape[1] != 4:
            raise ValueError("Omnivore RGB-D input must have shape [B,4,T,H,W]")
        return self.classifier(self.trunk(rgbd))
