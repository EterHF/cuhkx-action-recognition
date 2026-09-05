"""Leakage-free high-rate skeleton baseline for research OOF evaluation."""

from __future__ import annotations

import torch
from torch import nn

PARENTS = (0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15)


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(channels, channels * 2, kernel_size=1)
        self.output = nn.Sequential(
            nn.GELU(), nn.Dropout(dropout), nn.Conv1d(channels * 2, channels, 1)
        )

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = inputs
        value = self.norm(inputs).transpose(1, 2)
        value = self.output(self.pointwise(self.depthwise(value))).transpose(1, 2)
        return (residual + value) * mask[..., None].to(value.dtype)


class HighRateSkeletonClassifier(nn.Module):
    """Preserve native skeleton frames and classify without target-pretrained state."""

    def __init__(
        self,
        hidden_dim: int = 128,
        joint_dim: int = 64,
        num_classes: int = 40,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.register_buffer("parents", torch.tensor(PARENTS), persistent=False)
        self.joint_input = nn.Linear(9, joint_dim)
        self.joint_embedding = nn.Parameter(torch.empty(1, 1, 17, joint_dim))
        self.spatial_norm = nn.LayerNorm(joint_dim)
        self.spatial_attention = nn.MultiheadAttention(
            joint_dim, num_heads=4, dropout=dropout, batch_first=True
        )
        self.spatial_pool = nn.Linear(joint_dim, 1)
        self.frame_projection = nn.Linear(joint_dim, hidden_dim)
        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.temporal = nn.ModuleList(
            TemporalBlock(hidden_dim, dilation, dropout) for dilation in (1, 2, 4, 8, 16, 32)
        )
        self.temporal_pool = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes)
        )
        nn.init.trunc_normal_(self.joint_embedding, std=0.02)

    def forward(
        self,
        skeleton: torch.Tensor,
        mask: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if skeleton.ndim != 4 or skeleton.shape[-2:] != (17, 3):
            raise ValueError("skeleton must have shape [B,T,17,3]")
        if mask.shape != skeleton.shape[:2] or positions.shape != mask.shape:
            raise ValueError("mask and positions must have shape [B,T]")
        mask = mask.bool()
        velocity = torch.zeros_like(skeleton)
        velocity[:, 1:] = skeleton[:, 1:] - skeleton[:, :-1]
        adjacent = torch.zeros_like(mask)
        adjacent[:, 1:] = mask[:, 1:] & mask[:, :-1]
        velocity *= adjacent[:, :, None, None].to(velocity.dtype)
        bones = skeleton - skeleton[:, :, self.parents]
        features = torch.cat((skeleton, velocity, bones), dim=-1)

        batch, frames = skeleton.shape[:2]
        joints = self.joint_input(features) + self.joint_embedding
        flat = self.spatial_norm(joints).reshape(batch * frames, 17, -1)
        attended, _ = self.spatial_attention(flat, flat, flat, need_weights=False)
        joints = (flat + attended).reshape(batch, frames, 17, -1)
        joint_weights = self.spatial_pool(joints).softmax(dim=2)
        value = self.frame_projection((joints * joint_weights).sum(dim=2))
        value = value + self.time_embedding(positions[..., None])
        value *= mask[..., None].to(value.dtype)
        for block in self.temporal:
            value = block(value, mask)

        scores = self.temporal_pool(value).squeeze(-1).masked_fill(~mask, -1e4)
        weights = scores.softmax(dim=1) * mask.to(scores.dtype)
        weights /= weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        summary = (value * weights[..., None]).sum(dim=1)
        return self.classifier(summary)
