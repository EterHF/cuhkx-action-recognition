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


@torch.inference_mode()
def available_correction(
    model: ConditionalCorrector,
    visual: torch.Tensor,
    temporal: torch.Tensor,
    baseline: torch.Tensor,
    both_valid: torch.Tensor,
) -> torch.Tensor:
    """Keep unavailable rows byte-identical and never send them through the head."""
    if both_valid.dtype != torch.bool or both_valid.shape != (len(baseline),):
        raise ValueError("both_valid must be a boolean mask with one entry per row")
    if visual.shape[0] != len(baseline) or temporal.shape != baseline.shape:
        raise ValueError("corrector input rows/classes must match baseline")
    result = baseline.clone()
    if bool(both_valid.any()):
        result[both_valid] += model(visual[both_valid], temporal[both_valid])
    return result
