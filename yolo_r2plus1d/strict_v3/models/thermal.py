"""Small ImageNet-initialized temporal segment network for thermal video."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18


def shift_frames(x: torch.Tensor, segments: int) -> torch.Tensor:
    """Exchange 1/8 of channels each way, without crossing clip boundaries.

    Temporal Shift Module, Lin et al., ICCV 2019, arXiv:1811.08383.
    """
    if x.shape[0] % segments:
        raise ValueError("frame batch is not divisible by temporal segments")
    sequence = x.reshape(-1, segments, *x.shape[1:])
    width = x.shape[1] // 8
    result = torch.zeros_like(sequence)
    result[:, :-1, :width] = sequence[:, 1:, :width]
    result[:, 1:, width:2 * width] = sequence[:, :-1, width:2 * width]
    result[:, :, 2 * width:] = sequence[:, :, 2 * width:]
    return result.reshape_as(x)


class ShiftConv(nn.Module):
    def __init__(self, convolution: nn.Module) -> None:
        super().__init__()
        self.convolution = convolution
        self.segments = 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convolution(shift_frames(x, self.segments))


class ThermalTSN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = resnet18(weights=None)
        self.encoder.fc = nn.Identity()
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, 40))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def enable_shift(self) -> None:
        for stage in (self.encoder.layer1, self.encoder.layer2,
                      self.encoder.layer3, self.encoder.layer4):
            for block in stage:
                if isinstance(block.conv1, ShiftConv):
                    raise ValueError("temporal shift already installed")
                block.conv1 = ShiftConv(block.conv1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        batch, time, channels, height, width = frames.shape
        for module in self.modules():
            if isinstance(module, ShiftConv):
                module.segments = time
        x = frames.reshape(batch * time, channels, height, width).float() / 255.0
        features = self.encoder((x - self.mean) / self.std)
        return self.head(features.reshape(batch, time, -1).mean(1))


def thermal_fallback(
    baseline: torch.Tensor, thermal: torch.Tensor, no_primary: torch.Tensor,
    thermal_valid: torch.Tensor,
) -> torch.Tensor:
    """Use the independent sensor only where both primary inputs are absent."""
    if baseline.shape != thermal.shape or baseline.ndim != 2:
        raise ValueError("baseline and thermal must be aligned [rows, classes] logits")
    for mask in (no_primary, thermal_valid):
        if mask.dtype != torch.bool or mask.shape != (len(baseline),):
            raise ValueError("validity masks must be aligned bool[rows]")
    return torch.where((no_primary & thermal_valid)[:, None], thermal, baseline)
