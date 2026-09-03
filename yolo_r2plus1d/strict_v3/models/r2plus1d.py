"""Four-channel Depth+IR R(2+1)D classifier."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18


class MixStyle3D(nn.Module):
    """Mix feature statistics across clips during training only.

    The operation is deliberately parameter-free: it perturbs the style
    statistics of layer-1 activations while preserving the action label.  It
    is disabled in ``eval`` mode, so checkpoints remain compatible with the
    original inference path.
    """

    def __init__(
        self, probability: float = 0.5, alpha: float = 0.1, eps: float = 1e-6
    ) -> None:
        super().__init__()
        if not 0.0 <= probability <= 1.0:
            raise ValueError("MixStyle probability must be in [0, 1]")
        if alpha <= 0.0:
            raise ValueError("MixStyle alpha must be positive")
        self.probability = float(probability)
        self.alpha = float(alpha)
        self.eps = float(eps)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if (not self.training) or value.size(0) < 2:
            return value
        if torch.rand((), device=value.device) >= self.probability:
            return value

        # Compute statistics in FP32.  The surrounding model may be under
        # bfloat16 autocast, while feature variance is sensitive to rounding.
        original_dtype = value.dtype
        stats = value.float()
        mean = stats.mean(dim=(2, 3, 4), keepdim=True)
        std = (
            stats.var(dim=(2, 3, 4), keepdim=True, unbiased=False).add(self.eps).sqrt()
        )
        permutation = torch.randperm(value.size(0), device=value.device)
        beta = torch.distributions.Beta(self.alpha, self.alpha)
        lam = beta.sample((value.size(0), 1, 1, 1, 1)).to(value.device)
        mixed_mean = lam * mean + (1.0 - lam) * mean[permutation]
        mixed_std = lam * std + (1.0 - lam) * std[permutation]
        mixed = (stats - mean) / std * mixed_std + mixed_mean
        return mixed.to(original_dtype)


class DepthIRR2Plus1D(nn.Module):
    def __init__(
        self,
        num_classes: int = 40,
        dropout: float = 0.3,
        pretrained: bool = True,
        mixstyle: bool = False,
        mixstyle_p: float = 0.5,
        mixstyle_alpha: float = 0.1,
    ) -> None:
        super().__init__()
        weights = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
        self.network = r2plus1d_18(weights=weights)
        original = self.network.stem[0]
        expanded = nn.Conv3d(
            4,
            original.out_channels,
            original.kernel_size,
            original.stride,
            original.padding,
            bias=False,
        )
        with torch.no_grad():
            expanded.weight[:, :3].copy_(original.weight)
            expanded.weight[:, 3:4].copy_(original.weight.mean(dim=1, keepdim=True))
        self.network.stem[0] = expanded
        self.network.fc = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(512, num_classes)
        )
        self.mixstyle = MixStyle3D(mixstyle_p, mixstyle_alpha) if mixstyle else None

    def _forward_features(self, frames: torch.Tensor) -> torch.Tensor:
        value = frames.permute(0, 2, 1, 3, 4).contiguous()
        value = self.network.stem(value)
        value = self.network.layer1(value)
        if self.mixstyle is not None:
            value = self.mixstyle(value)
        value = self.network.layer2(value)
        value = self.network.layer3(value)
        value = self.network.layer4(value)
        value = self.network.avgpool(value)
        return torch.flatten(value, 1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # Cached layout is B,T,C,H,W; torchvision video models expect B,C,T,H,W.
        return self.network.fc(self._forward_features(frames))

    def forward_features(self, frames: torch.Tensor) -> torch.Tensor:
        """Return the pooled 512-D representation for a cached clip.

        Keeping this path alongside ``forward`` avoids reimplementing the
        torchvision stem/layer ordering in source-only temporal-gradient
        experiments.  It has no effect on the release model or checkpoint
        format and uses exactly the same input permutation as ``forward``.
        """
        return self._forward_features(frames)
