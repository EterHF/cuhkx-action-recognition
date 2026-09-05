"""High-rate skeleton queries attending to one shallow visual feature stream."""

from __future__ import annotations

import torch
from torch import nn


class HighRateCrossAttention(nn.Module):
    """Add a compact cross-modal correction to frozen temporal logits."""

    def __init__(
        self,
        visual_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 40,
        heads: int = 4,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.skeleton_input = nn.Linear(17 * 3 * 2, hidden_dim)
        self.visual_input = nn.Linear(visual_dim, hidden_dim)
        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.local = nn.Sequential(
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=1,
                groups=hidden_dim,
                bias=False,
            ),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
        )
        self.local_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=0.1, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.pool = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Dropout(0.2), nn.Linear(hidden_dim, num_classes)
        )
        nn.init.zeros_(self.classifier[-1].weight)
        nn.init.zeros_(self.classifier[-1].bias)

    def forward(
        self,
        visual: torch.Tensor,
        skeleton: torch.Tensor,
        skeleton_mask: torch.Tensor,
        skeleton_positions: torch.Tensor,
        baseline_logits: torch.Tensor,
    ) -> torch.Tensor:
        if skeleton.ndim != 4 or skeleton.shape[-2:] != (17, 3):
            raise ValueError("skeleton must have shape [B,T,17,3]")
        mask = skeleton_mask.bool()
        velocity = torch.zeros_like(skeleton)
        velocity[:, 1:] = skeleton[:, 1:] - skeleton[:, :-1]
        pair_mask = torch.zeros_like(mask)
        pair_mask[:, 1:] = mask[:, 1:] & mask[:, :-1]
        velocity *= pair_mask[:, :, None, None].to(velocity.dtype)

        pose = torch.cat((skeleton, velocity), dim=-1).flatten(2)
        queries = self.skeleton_input(pose)
        queries = queries + self.time_embedding(skeleton_positions[..., None])
        local = self.local(queries.transpose(1, 2)).transpose(1, 2)
        queries = self.local_norm(queries + local)
        queries *= mask[..., None].to(queries.dtype)

        visual_positions = torch.linspace(
            0.0, 1.0, visual.shape[1], device=visual.device, dtype=visual.dtype
        )
        visual_tokens = self.visual_input(visual)
        visual_tokens = visual_tokens + self.time_embedding(
            visual_positions[None, :, None]
        )
        attended, _ = self.cross_attention(queries, visual_tokens, visual_tokens)
        queries = self.cross_norm(queries + attended)
        queries *= mask[..., None].to(queries.dtype)

        scores = self.pool(queries).squeeze(-1).masked_fill(~mask, -1e4)
        weights = scores.softmax(dim=1) * mask.to(scores.dtype)
        weights /= weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        summary = (queries * weights[..., None]).sum(dim=1)
        residual = self.classifier(summary)
        present = mask.any(dim=1, keepdim=True).to(residual.dtype)
        return (
            baseline_logits.float()
            + self.residual_scale * residual.float() * present
        )
