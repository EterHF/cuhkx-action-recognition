"""Small Visual residual head conditioned on frozen Temporal logits."""

from __future__ import annotations

import torch
from torch import nn


class ConditionalCorrector(nn.Module):
    def __init__(
        self, visual_dim: int = 512, classes: int = 40, hidden_dim: int = 128
    ) -> None:
        super().__init__()
        self.visual_norm = nn.LayerNorm(visual_dim)
        self.temporal_norm = nn.LayerNorm(classes)
        self.hidden = nn.Sequential(
            nn.Linear(visual_dim + classes, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.output = nn.Linear(hidden_dim, classes)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self, visual_features: torch.Tensor, temporal_logits: torch.Tensor
    ) -> torch.Tensor:
        if visual_features.ndim != 2 or temporal_logits.ndim != 2:
            raise ValueError("corrector inputs must be two-dimensional")
        if visual_features.shape[0] != temporal_logits.shape[0]:
            raise ValueError("corrector inputs must have equal batch size")
        value = torch.cat(
            (
                self.visual_norm(visual_features),
                self.temporal_norm(temporal_logits),
            ),
            dim=1,
        )
        return self.output(self.hidden(value))
