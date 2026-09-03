"""Temporal fusion of R(2+1)D Depth+IR features and ST-GCN++ skeleton features."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18

H36M_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (0, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (8, 11),
    (11, 12),
    (12, 13),
    (8, 14),
    (14, 15),
    (15, 16),
)


def deterministic_linear_resample_1d(
    inputs: torch.Tensor, output_size: int
) -> torch.Tensor:
    """Resample ``B,C,T`` like linear ``F.interpolate`` with align_corners=False.

    ``aten::upsample_linear1d`` has no deterministic CUDA backward on the
    target stack.  The equivalent interpolation kernel is a fixed sparse
    matrix, so expressing it as ``matmul`` keeps the operation differentiable
    while routing both forward and backward through deterministic GEMM.
    """

    if inputs.ndim != 3:
        raise ValueError(f"expected B,C,T input, got shape {tuple(inputs.shape)}")
    if output_size < 1:
        raise ValueError(f"output_size must be positive, got {output_size}")
    input_size = inputs.shape[-1]
    if input_size < 1:
        raise ValueError("input temporal size must be positive")
    if input_size == output_size:
        return inputs

    # align_corners=False maps output pixel centres to source coordinates
    # (j + 0.5) * input_size / output_size - 0.5, clamped at the boundary.
    positions = (
        torch.arange(output_size, device=inputs.device, dtype=torch.float64) + 0.5
    ) * (float(input_size) / float(output_size)) - 0.5
    positions = positions.clamp(0.0, float(input_size - 1))
    left = positions.floor().to(torch.long)
    right = (left + 1).clamp(max=input_size - 1)
    weight = (positions - left.to(torch.float64)).to(dtype=inputs.dtype)

    # Indexing an identity basis avoids duplicate-index scatter-add, whose
    # CUDA implementation may be nondeterministic.  Each output row has at
    # most two nonzero coefficients, exactly matching linear interpolation.
    basis = torch.eye(input_size, device=inputs.device, dtype=inputs.dtype)
    matrix = basis.index_select(0, left) * (1.0 - weight).unsqueeze(
        1
    ) + basis.index_select(0, right) * weight.unsqueeze(1)
    flattened = inputs.reshape(-1, input_size)
    return flattened.matmul(matrix.transpose(0, 1)).reshape(
        *inputs.shape[:-1], output_size
    )


def normalize_adjacency(matrix: torch.Tensor) -> torch.Tensor:
    return matrix / matrix.sum(dim=0, keepdim=True).clamp_min(1.0)


def h36m_adjacency() -> torch.Tensor:
    inward = torch.zeros(17, 17)
    for source, target in H36M_EDGES:
        inward[source, target] = 1.0
    return torch.stack(
        (torch.eye(17), normalize_adjacency(inward), normalize_adjacency(inward.T))
    )


class AdaptiveGraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.register_buffer("base_adjacency", h36m_adjacency())
        self.adjacency_residual = nn.Parameter(torch.zeros(3, 17, 17))
        self.projection = nn.Conv2d(in_channels, out_channels * 3, 1, bias=False)
        self.out_channels = out_channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, _, time, joints = inputs.shape
        features = self.projection(inputs).view(
            batch, 3, self.out_channels, time, joints
        )
        adjacency = self.base_adjacency + self.adjacency_residual
        return torch.einsum("bkctv,kvw->bctw", features, adjacency)


class MultiScaleTemporalConv(nn.Module):
    def __init__(self, channels: int, stride: int) -> None:
        super().__init__()
        branch_channels = channels // 4
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        branch_channels,
                        (3, 1),
                        (stride, 1),
                        (1, 0),
                        bias=False,
                    ),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        branch_channels,
                        (5, 1),
                        (stride, 1),
                        (2, 0),
                        bias=False,
                    ),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.MaxPool2d((3, 1), (stride, 1), (1, 0)),
                    nn.Conv2d(channels, branch_channels, 1, bias=False),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(channels, branch_channels, 1, (stride, 1), bias=False),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True),
                ),
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.cat([branch(inputs) for branch in self.branches], dim=1)


class STGCNPlusPlusBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.graph = AdaptiveGraphConv(in_channels, out_channels)
        self.graph_norm = nn.BatchNorm2d(out_channels)
        self.temporal = MultiScaleTemporalConv(out_channels, stride)
        if in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, (stride, 1), bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        graph = self.activation(self.graph_norm(self.graph(inputs)))
        return self.activation(self.temporal(graph) + self.residual(inputs))


class STGCNPlusPlus(nn.Module):
    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.input_norm = nn.BatchNorm1d(17 * in_channels)
        self.blocks = nn.Sequential(
            STGCNPlusPlusBlock(in_channels, 64),
            STGCNPlusPlusBlock(64, 64),
            STGCNPlusPlusBlock(64, 128, stride=2),
            STGCNPlusPlusBlock(128, 128),
            STGCNPlusPlusBlock(128, 256, stride=2),
            STGCNPlusPlusBlock(256, 256),
        )

    def forward_features(self, skeleton: torch.Tensor) -> torch.Tensor:
        # B,T,V,C -> B,C,T,V; normalize channels/joints before graph processing.
        batch, time, joints, channels = skeleton.shape
        normalized = self.input_norm(
            skeleton.permute(0, 2, 3, 1).reshape(batch, joints * channels, time)
        )
        return self.blocks(
            normalized.view(batch, joints, channels, time).permute(0, 2, 3, 1)
        )

    def forward(self, skeleton: torch.Tensor) -> torch.Tensor:
        return self.forward_features(skeleton).mean(dim=-1)


class DepthIRTemporalBackbone(nn.Module):
    def __init__(self, pretrained: bool, input_channels: int = 4) -> None:
        super().__init__()
        weights = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
        network = r2plus1d_18(weights=weights)
        original = network.stem[0]
        expanded = nn.Conv3d(
            input_channels,
            original.out_channels,
            original.kernel_size,
            original.stride,
            original.padding,
            bias=False,
        )
        with torch.no_grad():
            expanded.weight[:, :3].copy_(original.weight)
            expanded.weight[:, 3:].copy_(
                original.weight.mean(dim=1, keepdim=True).expand(
                    -1, input_channels - 3, -1, -1, -1
                )
            )
        network.stem[0] = expanded
        self.stem = network.stem
        self.layer1, self.layer2 = network.layer1, network.layer2
        self.layer3, self.layer4 = network.layer3, network.layer4
        self.project3 = nn.Conv1d(256, 256, 1)
        self.project4 = nn.Conv1d(512, 256, 1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        features = frames.permute(0, 2, 1, 3, 4).contiguous()
        features = self.layer2(self.layer1(self.stem(features)))
        layer3 = self.layer3(features)
        layer4 = self.layer4(layer3)
        temporal3 = self.project3(layer3.mean(dim=(-1, -2)))
        temporal4 = deterministic_linear_resample_1d(
            self.project4(layer4.mean(dim=(-1, -2))),
            temporal3.shape[-1],
        )
        return temporal3 + temporal4


class TemporalFusionClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 40,
        skeleton_dropout: float = 0.1,
        pretrained_visual: bool = True,
        skeleton_channels: int = 3,
        visual_channels: int = 4,
    ) -> None:
        super().__init__()
        self.visual_channels = visual_channels
        self.visual = DepthIRTemporalBackbone(pretrained_visual, visual_channels)
        self.skeleton = STGCNPlusPlus(skeleton_channels)
        self.skeleton_dropout = skeleton_dropout
        self.fusion = nn.Sequential(
            nn.Conv1d(512, 256, 3, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Conv1d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=4,
            dim_feedforward=512,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.attention = nn.Conv1d(256, 1, 1)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, num_classes))
        self.skeleton_classifier = nn.Linear(256, num_classes)

    def load_visual_checkpoint(self, state: dict[str, torch.Tensor]) -> None:
        visual_state = {}
        for name, value in state.items():
            if name.startswith("network.") and not name.startswith("network.fc"):
                visual_state[name.removeprefix("network.")] = value
        stem_key = "stem.0.weight"
        if (
            stem_key in visual_state
            and visual_state[stem_key].shape != self.visual.stem[0].weight.shape
        ):
            source = visual_state[stem_key]
            target = self.visual.stem[0].weight.detach().clone()
            target[:, : source.shape[1]].copy_(source)
            if target.shape[1] > source.shape[1]:
                target[:, source.shape[1] :].copy_(
                    source.mean(dim=1, keepdim=True).expand(
                        -1, target.shape[1] - source.shape[1], -1, -1, -1
                    )
                )
            visual_state[stem_key] = target
        missing, unexpected = self.visual.load_state_dict(visual_state, strict=False)
        allowed_missing = {
            "project3.weight",
            "project3.bias",
            "project4.weight",
            "project4.bias",
        }
        if set(missing) != allowed_missing or unexpected:
            raise RuntimeError(
                f"Unexpected visual checkpoint keys: missing={missing}, unexpected={unexpected}"
            )

    def forward(
        self,
        frames: torch.Tensor,
        skeleton: torch.Tensor,
        skeleton_mask: torch.Tensor,
        return_aux: bool = False,
    ):
        visual_features = self.visual(frames)
        skeleton_features = self.skeleton(skeleton)
        skeleton_mask = skeleton_mask[:, None, None].to(skeleton_features.dtype)
        skeleton_aux = skeleton_features * skeleton_mask
        if self.training and self.skeleton_dropout > 0:
            keep = (
                torch.rand(len(skeleton), 1, 1, device=skeleton.device)
                >= self.skeleton_dropout
            ).to(skeleton_features.dtype)
            skeleton_mask = skeleton_mask * keep
        skeleton_features = skeleton_features * skeleton_mask
        if skeleton_features.shape[-1] != visual_features.shape[-1]:
            skeleton_features = deterministic_linear_resample_1d(
                skeleton_features,
                visual_features.shape[-1],
            )
        fused = self.fusion(torch.cat((visual_features, skeleton_features), dim=1))
        fused = self.temporal_encoder(fused.transpose(1, 2)).transpose(1, 2)
        weights = self.attention(fused).softmax(dim=-1)
        logits = self.classifier((fused * weights).sum(dim=-1))
        if return_aux:
            skeleton_logits = self.skeleton_classifier(skeleton_aux.mean(dim=-1))
            return logits, skeleton_logits
        return logits
