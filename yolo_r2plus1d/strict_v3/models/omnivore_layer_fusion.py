"""Multi-stage temporal fusion of independent Depth and IR Omnivore trunks."""

from __future__ import annotations

import torch
from torch import nn


class OmnivoreLayerFusion(nn.Module):
    stage_keys = ("interim0", "interim1", "interim2", "interim3")
    stage_dims = (192, 384, 768, 768)

    def __init__(
        self,
        depth_trunk: nn.Module,
        ir_trunk: nn.Module,
        hidden_dim: int = 128,
        num_classes: int = 40,
    ) -> None:
        super().__init__()
        self.depth_trunk = depth_trunk
        self.ir_trunk = ir_trunk
        self.depth_projection = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden_dim))
            for dim in self.stage_dims
        )
        self.ir_projection = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden_dim))
            for dim in self.stage_dims
        )
        self.cross_attention = nn.ModuleList(
            nn.MultiheadAttention(hidden_dim, 4, dropout=0.1, batch_first=True)
            for _ in self.stage_dims
        )
        self.stage_norm = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in self.stage_dims)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * len(self.stage_dims)),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * len(self.stage_dims), hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )
        self.depth_classifier = nn.Linear(hidden_dim, num_classes)
        self.ir_classifier = nn.Linear(hidden_dim, num_classes)

    @staticmethod
    def temporal_tokens(features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 5:
            raise ValueError("Omnivore stage features must have shape [B,C,T,H,W]")
        return features.mean(dim=(-2, -1)).permute(0, 2, 1)

    def forward(
        self, depth: torch.Tensor, infrared: torch.Tensor, return_aux: bool = False
    ):
        depth_stages = self.depth_trunk(depth, out_feat_keys=list(self.stage_keys))
        ir_stages = self.ir_trunk(infrared, out_feat_keys=list(self.stage_keys))
        summaries = []
        depth_last = ir_last = None
        for index, (depth_stage, ir_stage) in enumerate(zip(depth_stages, ir_stages, strict=True)):
            depth_tokens = self.depth_projection[index](self.temporal_tokens(depth_stage))
            ir_tokens = self.ir_projection[index](self.temporal_tokens(ir_stage))
            depth_context, _ = self.cross_attention[index](
                depth_tokens, ir_tokens, ir_tokens, need_weights=False
            )
            ir_context, _ = self.cross_attention[index](
                ir_tokens, depth_tokens, depth_tokens, need_weights=False
            )
            fused = self.stage_norm[index](
                (depth_tokens + depth_context + ir_tokens + ir_context) * 0.5
            )
            summaries.append(fused.mean(dim=1))
            depth_last = depth_tokens.mean(dim=1)
            ir_last = ir_tokens.mean(dim=1)
        logits = self.classifier(torch.cat(summaries, dim=-1))
        if not return_aux:
            return logits
        return logits, self.depth_classifier(depth_last), self.ir_classifier(ir_last)
