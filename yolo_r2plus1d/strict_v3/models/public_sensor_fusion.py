"""High-rate skeleton queries attending to independent Depth and IR encoders."""

from __future__ import annotations

import torch
from torch import nn

from yolo_r2plus1d.strict_v3.models.research_skeleton import HighRateSkeletonClassifier


class PublicSensorFusion(nn.Module):
    def __init__(
        self,
        depth_dim: int = 768,
        ir_dim: int = 768,
        hidden_dim: int = 128,
        num_classes: int = 40,
    ) -> None:
        super().__init__()
        self.skeleton = HighRateSkeletonClassifier(
            hidden_dim=hidden_dim, num_classes=num_classes
        )
        self.depth_input = nn.Sequential(nn.LayerNorm(depth_dim), nn.Linear(depth_dim, hidden_dim))
        self.ir_input = nn.Sequential(nn.LayerNorm(ir_dim), nn.Linear(ir_dim, hidden_dim))
        self.sensor_embedding = nn.Parameter(torch.empty(1, 2, hidden_dim))
        self.temporal_embedding = nn.Parameter(torch.empty(1, 8, hidden_dim))
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, dropout=0.1, batch_first=True
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Dropout(0.2), nn.Linear(hidden_dim, num_classes)
        )
        self.depth_classifier = nn.Linear(hidden_dim, num_classes)
        self.ir_classifier = nn.Linear(hidden_dim, num_classes)
        nn.init.trunc_normal_(self.sensor_embedding, std=0.02)
        nn.init.trunc_normal_(self.temporal_embedding, std=0.02)

    def encode_sensor_tokens(
        self, depth: torch.Tensor, infrared: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if depth.shape[1] != infrared.shape[1] or depth.shape[1] > len(self.temporal_embedding[0]):
            raise ValueError("Depth and IR must have the same number of at most 8 tokens")
        length = depth.shape[1]
        temporal = self.temporal_embedding[:, :length]
        depth_tokens = self.depth_input(depth) + temporal + self.sensor_embedding[:, :1]
        ir_tokens = self.ir_input(infrared) + temporal + self.sensor_embedding[:, 1:]
        return depth_tokens, ir_tokens, torch.cat((depth_tokens, ir_tokens), dim=1)

    def forward(
        self,
        depth: torch.Tensor,
        infrared: torch.Tensor,
        skeleton: torch.Tensor,
        skeleton_mask: torch.Tensor,
        skeleton_positions: torch.Tensor,
        return_aux: bool = False,
    ):
        sequence, mask = self.skeleton.forward_sequence(
            skeleton, skeleton_mask, skeleton_positions
        )
        depth_tokens, ir_tokens, sensors = self.encode_sensor_tokens(depth, infrared)
        attended, _ = self.cross_attention(sequence, sensors, sensors, need_weights=False)
        fused = self.fusion_norm(sequence + attended)
        fused *= mask[..., None].to(fused.dtype)
        skeleton_summary = self.skeleton.pool_sequence(sequence, mask)
        fused_summary = self.skeleton.pool_sequence(fused, mask)
        depth_summary = depth_tokens.mean(dim=1)
        ir_summary = ir_tokens.mean(dim=1)
        sensor_summary = sensors.mean(dim=1)
        present = mask.any(dim=1, keepdim=True)
        summary = torch.where(present, fused_summary, sensor_summary)
        logits = self.classifier(summary)
        if not return_aux:
            return logits
        return (
            logits,
            self.skeleton.classifier(skeleton_summary),
            self.depth_classifier(depth_summary),
            self.ir_classifier(ir_summary),
        )
