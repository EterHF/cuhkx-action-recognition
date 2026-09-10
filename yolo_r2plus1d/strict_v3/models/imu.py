"""Compact InceptionTime-inspired network for paired local/global IMU views."""

from __future__ import annotations

import torch
from torch import nn


class Inception(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.bottleneck = nn.Conv1d(channels, 16, 1, bias=False)
        self.scales = nn.ModuleList(nn.Conv1d(16, 16, k, padding=k // 2, bias=False)
                                    for k in (9, 19, 39))
        self.pool = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1),
                                  nn.Conv1d(channels, 16, 1, bias=False))
        self.norm = nn.BatchNorm1d(64)

    def forward(self, x):
        z = self.bottleneck(x)
        return torch.relu(self.norm(torch.cat([layer(z) for layer in self.scales] + [self.pool(x)], 1)))


class IMUClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Inception(60), *[Inception(64) for _ in range(5)]])
        self.skips = nn.ModuleList([
            nn.Sequential(nn.Conv1d(60, 64, 1, bias=False), nn.BatchNorm1d(64)),
            nn.Sequential(nn.Conv1d(64, 64, 1, bias=False), nn.BatchNorm1d(64)),
        ])
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(64, 40))

    def forward(self, x):
        residual = x
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i % 3 == 2:
                x = torch.relu(x + self.skips[i // 3](residual))
                residual = x
        return self.head(x.mean(-1))


def select_view(signals: torch.Tensor, view: str) -> torch.Tensor:
    local = signals[:, 0].flatten(1, 2)
    other = local if view == "local" else signals[:, 1].flatten(1, 2)
    if view not in {"local", "global"}:
        raise ValueError(f"unknown view: {view}")
    return torch.cat((local, other), dim=1)


def imu_blend(baseline: torch.Tensor, imu: torch.Tensor, eligible: torch.Tensor) -> torch.Tensor:
    """Fixed 10% probability contribution, restricted to available primary/IMU rows."""
    if eligible.dtype != torch.bool or eligible.shape != (len(baseline),):
        raise ValueError("eligible must be a row boolean mask")
    if imu.shape != baseline.shape:
        raise ValueError("IMU/baseline shape mismatch")
    result = baseline.clone()
    result[eligible] = (
        .9 * baseline[eligible].softmax(1) + .1 * imu[eligible].softmax(1)
    ).log()
    return result
