"""Ordered low-capacity architecture candidates for the strictV3 Visual branch."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from yolo_r2plus1d.strict_v3.models.r2plus1d34 import r2plus1d_34

VARIANTS = ("control", "multiscale", "highres", "modality_gate")


class ModalityGate(nn.Module):
    """Bounded clip-level sensor scaling that is exactly identity at initialization."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 16), nn.GELU(), nn.Linear(16, 2), nn.Tanh()
        )
        nn.init.zeros_(self.network[-2].weight)
        nn.init.zeros_(self.network[-2].bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 5 or inputs.shape[1] != 4:
            raise ValueError("modality gate expects [B,4,T,H,W]")
        depth = inputs[:, :3]
        infrared = inputs[:, 3:]
        statistics = torch.stack(
            (
                depth.mean(dim=(1, 2, 3, 4)),
                depth.std(dim=(1, 2, 3, 4), unbiased=False),
                infrared.mean(dim=(1, 2, 3, 4)),
                infrared.std(dim=(1, 2, 3, 4), unbiased=False),
            ),
            dim=1,
        )
        scale = 1.0 + 0.25 * self.network(statistics)
        return torch.cat(
            (
                depth * scale[:, :1, None, None, None],
                infrared * scale[:, 1:, None, None, None],
            ),
            dim=1,
        )


def expand_stem_to_four_channels(network: nn.Module) -> None:
    original = network.stem[0]
    replacement = type(original)(
        4,
        original.out_channels,
        original.kernel_size,
        original.stride,
        original.padding,
        bias=original.bias is not None,
    )
    with torch.no_grad():
        replacement.weight[:, :3].copy_(original.weight)
        replacement.weight[:, 3:].copy_(original.weight.mean(dim=1, keepdim=True))
        if original.bias is not None:
            replacement.bias.copy_(original.bias)
    network.stem[0] = replacement


def preserve_layer4_temporal_resolution(network: nn.Module) -> None:
    block = network.layer4[0]
    block.conv1[0][3].stride = (1, 1, 1)
    block.downsample[0].stride = (1, 2, 2)


class MultiScaleTemporalHead(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_classes: int = 40) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            nn.Conv1d(channels, hidden_dim, 1) for channels in (128, 256, 512)
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
            nn.GELU(),
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                3,
                padding=2,
                dilation=2,
                groups=hidden_dim,
            ),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
        )
        self.attention = nn.Conv1d(hidden_dim, 1, 1)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, stages: tuple[torch.Tensor, ...]) -> torch.Tensor:
        length = stages[0].shape[2]
        tokens = []
        for projection, stage in zip(self.projections, stages, strict=True):
            projected = projection(stage.mean(dim=(-1, -2)))
            if projected.shape[-1] != length:
                projected = F.interpolate(
                    projected, size=length, mode="linear", align_corners=False
                )
            tokens.append(projected)
        sequence = self.temporal(sum(tokens) / len(tokens))
        weights = self.attention(sequence).softmax(dim=-1)
        return self.classifier((sequence * weights).sum(dim=-1))


class VisualPriorityClassifier(nn.Module):
    def __init__(self, variant: str, num_classes: int = 40) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown Visual priority variant: {variant}")
        self.variant = variant
        self.encoder = r2plus1d_34(num_classes=400)
        expand_stem_to_four_channels(self.encoder)
        if variant in {"highres", "modality_gate"}:
            preserve_layer4_temporal_resolution(self.encoder)
        self.encoder.fc = nn.Identity()
        self.base_classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(512, num_classes)
        )
        self.temporal_head = (
            MultiScaleTemporalHead(num_classes=num_classes)
            if variant != "control"
            else None
        )
        # Keep every shared parameter at the same seeded initialization as highres.
        self.modality_gate = ModalityGate() if variant == "modality_gate" else nn.Identity()

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5 or frames.shape[2] != 4:
            raise ValueError("Visual input must have shape [B,T,4,H,W]")
        value = frames.permute(0, 2, 1, 3, 4).contiguous()
        value = self.modality_gate(value)
        value = self.encoder.stem(value)
        value = self.encoder.layer1(value)
        layer2 = self.encoder.layer2(value)
        layer3 = self.encoder.layer3(layer2)
        layer4 = self.encoder.layer4(layer3)
        base = self.base_classifier(layer4.mean(dim=(2, 3, 4)))
        if self.temporal_head is None:
            return base
        return base + self.temporal_head((layer2, layer3, layer4))
