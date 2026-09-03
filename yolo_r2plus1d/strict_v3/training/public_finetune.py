#!/usr/bin/env python3
"""Fine-tune the released packed public R(2+1)D-34 model on train clips.

The public model is stored as signed per-output-channel int5/int6 weights.
This script dequantizes one member for optimization, keeps the split
subject-wise, and writes fp16 logits/checkpoints plus an optional packed
candidate.  Test labels are never loaded or inferred.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from yolo_r2plus1d.strict_v3.models.r2plus1d34 import r2plus1d_34
from yolo_r2plus1d.strict_v3.paths import CHECKPOINT_DIR, REPO_ROOT, RESULT_DIR

N_CLASSES = 40
N_FRAMES = 16
CHANNELS = 4
IMAGE_SIZE = 128
KINETICS_MEAN = torch.tensor(
    (0.43216, 0.394666, 0.37645, (0.43216 + 0.394666 + 0.37645) / 3.0),
    dtype=torch.float32,
).view(1, CHANNELS, 1, 1)
KINETICS_STD = torch.tensor(
    (0.22803, 0.22145, 0.216989, (0.22803 + 0.22145 + 0.216989) / 3.0),
    dtype=torch.float32,
).view(1, CHANNELS, 1, 1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class VideoDataset(Dataset):
    def __init__(
        self,
        cache: Path,
        indices: np.ndarray,
        labels: np.ndarray | None,
        augment: bool,
        horizontal_flip: bool = True,
        sensor_mode: str = "all",
        source_mean: torch.Tensor | None = None,
        source_std: torch.Tensor | None = None,
        normalize_after_affine: bool = False,
        style_augment: bool = False,
        temporal_jitter: bool = False,
        modality_dropout: float = 0.0,
        temporal_difference: bool = False,
        secondary_cache: Path | None = None,
    ):
        self.cache = np.load(cache, mmap_mode="r")
        self.secondary_cache = (
            None if secondary_cache is None else np.load(secondary_cache, mmap_mode="r")
        )
        if (
            self.secondary_cache is not None
            and self.secondary_cache.shape != self.cache.shape
        ):
            raise ValueError("primary and secondary caches must have identical shapes")
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)
        self.augment = bool(augment)
        self.horizontal_flip = bool(horizontal_flip)
        if sensor_mode not in {"all", "depth", "ir"}:
            raise ValueError(f"unknown sensor_mode={sensor_mode}")
        self.sensor_mode = sensor_mode
        self.source_mean = source_mean
        self.source_std = source_std
        self.normalize_after_affine = bool(normalize_after_affine)
        # These augmentations are deliberately restricted to the training
        # stream.  They model sensor/rendering variation (rather than using
        # user ids or target-domain statistics) and leave validation/test
        # preprocessing byte-for-byte unchanged.
        self.style_augment = bool(style_augment)
        self.temporal_jitter = bool(temporal_jitter)
        if not 0.0 <= float(modality_dropout) <= 1.0:
            raise ValueError("modality_dropout must be in [0,1]")
        self.modality_dropout = float(modality_dropout)
        self.temporal_difference = bool(temporal_difference)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        cache = (
            self.secondary_cache
            if self.augment
            and self.secondary_cache is not None
            and torch.rand(()) < 0.5
            else self.cache
        )
        frames = torch.from_numpy(np.array(cache[index], copy=True)).float().div_(255.0)
        if self.augment and self.temporal_jitter:
            # The 16-frame cache is uniformly sampled.  A small monotonic
            # index perturbation emulates a different frame-rate/trigger
            # without reversing actions or consulting timestamps.  Endpoints
            # stay fixed, so the temporal support remains the same.
            n_frames = int(frames.shape[0])
            if n_frames >= 4:
                offsets = torch.empty(n_frames).uniform_(-0.35, 0.35)
                base = torch.arange(n_frames, dtype=torch.float32)
                warped = (base + offsets).clamp_(0.0, float(n_frames - 1))
                warped[0], warped[-1] = 0.0, float(n_frames - 1)
                left = warped.floor().long()
                right = warped.ceil().long()
                alpha = (warped - left).view(-1, 1, 1, 1)
                frames = frames[left] * (1.0 - alpha) + frames[right] * alpha
        flipped = bool(self.augment and self.horizontal_flip and torch.rand(()) < 0.5)
        if flipped:
            frames = frames.flip(-1)
        if self.augment:
            # Mild independent sensor gain avoids fitting cache-specific
            # intensity while preserving the public preprocessing contract.
            depth_gain = 0.95 + 0.10 * torch.rand(1)
            ir_gain = 0.95 + 0.10 * torch.rand(1)
            frames[:, :3].mul_(depth_gain).clamp_(0.0, 1.0)
            frames[:, 3:].mul_(ir_gain).clamp_(0.0, 1.0)
            if self.modality_dropout > 0.0:
                # Drop each complete sensor stream to a train-only mean.  The
                # event is independent of user/session and applies only to
                # outer-train clips; validation/test remain unchanged.
                fill = (
                    self.source_mean if self.source_mean is not None else KINETICS_MEAN
                )
                if torch.rand(()) < self.modality_dropout:
                    frames[:, :3] = fill[:, :3]
                if torch.rand(()) < self.modality_dropout:
                    frames[:, 3:] = fill[:, 3:]
        if self.augment and self.style_augment:
            # Pseudo-colour depth is produced by a fixed LUT, while the
            # camera exposure/IR response is not fixed.  Apply a conservative
            # per-clip affine+gamma perturbation and sparse quantisation.  A
            # single factor is shared by all frames to preserve motion.
            depth = frames[:, :3]
            ir = frames[:, 3:]
            depth_mean = depth.mean(dim=(0, 2, 3), keepdim=True)
            ir_mean = ir.mean(dim=(0, 2, 3), keepdim=True)
            depth_contrast = 0.88 + 0.24 * torch.rand(1)
            ir_contrast = 0.85 + 0.30 * torch.rand(1)
            depth_bias = (torch.rand(1) - 0.5) * 0.08
            ir_bias = (torch.rand(1) - 0.5) * 0.10
            depth = (
                (depth - depth_mean) * depth_contrast + depth_mean + depth_bias
            ).clamp_(0.0, 1.0)
            ir = ((ir - ir_mean) * ir_contrast + ir_mean + ir_bias).clamp_(0.0, 1.0)
            depth_gamma = 0.88 + 0.24 * torch.rand(1)
            ir_gamma = 0.85 + 0.30 * torch.rand(1)
            frames[:, :3] = depth.pow(depth_gamma)
            frames[:, 3:] = ir.pow(ir_gamma)
            if torch.rand(()) < 0.35:
                levels = int(torch.randint(96, 193, ()).item())
                frames = torch.round(frames * levels) / float(levels)
            if torch.rand(()) < 0.25:
                # Sensor holes are local, but the mask is shared across
                # frames so the model cannot identify a particular user.
                h, w = int(frames.shape[-2]), int(frames.shape[-1])
                hole_h = max(2, int(round(h * float(0.02 + 0.04 * torch.rand(1)))))
                hole_w = max(2, int(round(w * float(0.02 + 0.04 * torch.rand(1)))))
                y0 = int(torch.randint(0, max(1, h - hole_h + 1), ()).item())
                x0 = int(torch.randint(0, max(1, w - hole_w + 1), ()).item())
                frames[:, :, y0 : y0 + hole_h, x0 : x0 + hole_w] *= 0.0
        # Keep the four-channel architecture and pretrained stem, but allow
        # controlled single-sensor ablations.  Replacing a disabled sensor by
        # its training mean makes its normalized input exactly zero.
        if self.sensor_mode == "depth":
            frames[:, 3:] = 0.400
        elif self.sensor_mode == "ir":
            frames[:, :3] = torch.tensor((0.43216, 0.394666, 0.37645)).view(3, 1, 1)
        if self.source_mean is None or self.source_std is None:
            frames = (frames - KINETICS_MEAN) / KINETICS_STD
        else:
            frames = (
                (frames - self.source_mean) / self.source_std
            ) * KINETICS_STD + KINETICS_MEAN
            if self.normalize_after_affine:
                frames = (frames - KINETICS_MEAN) / KINETICS_STD
        if self.labels is None:
            return frames, index
        if self.temporal_difference:
            # Auxiliary motion view used only during source-only training.
            # Depth is deliberately excluded because the pseudo-colour LUT can
            # turn quantisation into a false temporal gradient; the IR channel
            # is differenced with exactly the same crop/normalization as the
            # student.  The first frame has no predecessor and is zero.
            difference = torch.zeros_like(frames)
            difference[1:, 3:] = frames[1:, 3:] - frames[:-1, 3:]
            return frames, difference, int(self.labels[index])
        return frames, int(self.labels[index])


class InputAdapter(nn.Module):
    """Small residual spatial adapter from the four cached channels to RGB."""

    def __init__(self) -> None:
        super().__init__()
        self.skip = nn.Conv3d(CHANNELS, 3, kernel_size=1, bias=False)
        self.residual = nn.Sequential(
            nn.Conv3d(
                CHANNELS, 16, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False
            ),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv3d(16, 3, kernel_size=1, bias=True),
        )
        with torch.no_grad():
            self.skip.weight.zero_()
            for channel in range(3):
                self.skip.weight[channel, channel, 0, 0, 0] = 1.0
                self.skip.weight[channel, 3, 0, 0, 0] = 1.0 / 3.0
            nn.init.normal_(self.residual[-1].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.residual[-1].bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.skip(inputs) + self.residual(inputs)


def make_model(input_adapter: bool = False) -> nn.Module:
    class R2Plus1D34(nn.Module):
        def __init__(self):
            super().__init__()
            network = r2plus1d_34(num_classes=400)
            original = network.stem[0]
            if not input_adapter:
                replacement = type(original)(
                    CHANNELS,
                    original.out_channels,
                    original.kernel_size,
                    original.stride,
                    original.padding,
                    bias=original.bias is not None,
                )
                with torch.no_grad():
                    replacement.weight[:, :3].copy_(original.weight)
                    replacement.weight[:, 3:].copy_(
                        original.weight.mean(dim=1, keepdim=True)
                    )
                    if original.bias is not None:
                        replacement.bias.copy_(original.bias)
                network.stem[0] = replacement
            features = network.fc.in_features
            network.fc = nn.Identity()
            self.encoder = network
            self.input_adapter = InputAdapter() if input_adapter else nn.Identity()
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(features, N_CLASSES))
            self.freeze_encoder_bn = bool(input_adapter)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.head(self.forward_features(inputs))

        def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
            inputs = inputs.permute(0, 2, 1, 3, 4).contiguous()
            inputs = self.input_adapter(inputs)
            return self.encoder(inputs)

    return R2Plus1D34()


def unpack_signed(
    packed: torch.Tensor, shape: tuple[int, ...], bits: int
) -> torch.Tensor:
    count = math.prod(shape)
    starts = torch.arange(count, dtype=torch.int64) * bits
    codes = torch.zeros(count, dtype=torch.int16)
    source = packed.to(torch.int16)
    for bit in range(bits):
        positions = starts + bit
        byte_indices = positions >> 3
        shifts = positions & 7
        codes |= torch.bitwise_and(source[byte_indices] >> shifts, 1) << bit
    sign, modulus = 1 << (bits - 1), 1 << bits
    signed = torch.where(codes >= sign, codes - modulus, codes)
    return signed.to(torch.int8).reshape(shape)


def dequantize_state(state: Mapping[str, object]) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if isinstance(value, Mapping):
            shape = tuple(int(item) for item in value["shape"])
            quantized = unpack_signed(value["packed"], shape, int(value["bits"]))
            output[key] = quantized.float() * value["scale"].float()
        else:
            output[key] = value.float() if value.is_floating_point() else value
    return output


def quantize_signed(tensor: torch.Tensor, bits: int) -> dict[str, object]:
    """Quantize a weight exactly in the release's per-output-channel format."""
    tensor = tensor.detach().float().cpu()
    qmax = (1 << (bits - 1)) - 1
    scale = tensor.abs().amax(dim=tuple(range(1, tensor.ndim)), keepdim=True) / qmax
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    values = torch.round(tensor / safe_scale).clamp(-qmax - 1, qmax).to(torch.int16)
    values = values.reshape(-1)
    # A value may straddle two bytes, so direct vector assignment would lose
    # colliding writes.  NumPy's indexed bitwise-or accumulates those writes
    # without a Python loop over the (tens of millions of) weights.
    unsigned = (
        torch.where(values < 0, values + (1 << bits), values).to(torch.int64).numpy()
    )
    positions = np.arange(values.numel(), dtype=np.int64) * bits
    packed_np = np.zeros((int((values.numel() * bits + 7) // 8),), dtype=np.uint8)
    for bit in range(bits):
        pos = positions + bit
        byte_index, shift = pos >> 3, pos & 7
        np.bitwise_or.at(
            packed_np, byte_index, ((unsigned >> bit) & 1).astype(np.uint8) << shift
        )
    packed = torch.from_numpy(packed_np)
    # The release stores fp16 scales.  Preserve zero scales for all-zero rows.
    scale = torch.where(scale > 0, scale, torch.zeros_like(scale)).half()
    return {"packed": packed, "scale": scale, "shape": list(tensor.shape), "bits": bits}


def packed_model(state: Mapping[str, torch.Tensor], bits: int) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in state.items():
        if torch.is_tensor(value) and value.is_floating_point() and value.ndim >= 2:
            output[key] = quantize_signed(value, bits)
        else:
            output[key] = (
                value.detach().cpu().half()
                if torch.is_tensor(value) and value.is_floating_point()
                else value.detach().cpu()
            )
    return output


def train_epoch(
    model,
    loader,
    optimizer,
    scaler,
    criterion,
    device,
    train_mode: str,
    teacher: nn.Module | None = None,
    distill_weight: float = 0.0,
    distill_temperature: float = 2.0,
    mixup_alpha: float = 0.0,
    temporal_difference: bool = False,
    difference_loss_weight: float = 0.25,
):
    model.train()
    if getattr(model, "freeze_encoder_bn", False):
        # Keep frozen pretrained BN statistics fixed.  Otherwise model.train()
        # would silently perform train-domain BN adaptation in this ablation.
        for module in model.encoder.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
    total_loss = correct = count = 0
    for batch in loader:
        if temporal_difference:
            frames, difference, labels = batch
            difference = difference.to(device, non_blocking=True)
        else:
            frames, labels = batch
        frames, labels = (
            frames.to(device, non_blocking=True),
            labels.to(device, non_blocking=True),
        )
        optimizer.zero_grad(set_to_none=True)
        mixup_lambda = None
        labels_a = labels_b = labels
        if mixup_alpha > 0.0 and len(labels) > 1:
            mixup_lambda = float(np.random.beta(mixup_alpha, mixup_alpha))
            permutation = torch.randperm(len(labels), device=labels.device)
            frames = mixup_lambda * frames + (1.0 - mixup_lambda) * frames[permutation]
            labels_a, labels_b = labels, labels[permutation]
        with torch.autocast("cuda", dtype=torch.float16):
            if temporal_difference:
                student_features = model.forward_features(frames)
                difference_features = model.forward_features(difference)
                logits = model.head(student_features)
                difference_logits = model.head(difference_features)
            else:
                logits = model(frames)
            if mixup_lambda is None:
                loss = criterion(logits, labels)
            else:
                loss = mixup_lambda * criterion(logits, labels_a) + (
                    1.0 - mixup_lambda
                ) * criterion(logits, labels_b)
            if temporal_difference:
                # Fixed source-only temporal-gradient auxiliary objective.  A
                # stop-gradient on the difference view makes it a teacher for
                # the raw-frame representation; inference never evaluates it.
                loss = loss + float(difference_loss_weight) * criterion(
                    difference_logits, labels
                )
                cosine = F.cosine_similarity(
                    student_features, difference_features.detach(), dim=1
                )
                loss = loss + float(difference_loss_weight) * (1.0 - cosine.mean())
            if teacher is not None and distill_weight > 0:
                with torch.no_grad():
                    teacher_logits = teacher(frames)
                temperature = float(distill_temperature)
                soft_target = F.softmax(teacher_logits / temperature, dim=1)
                soft_loss = F.kl_div(
                    F.log_softmax(logits / temperature, dim=1),
                    soft_target,
                    reduction="batchmean",
                ) * (temperature**2)
                loss = (1.0 - distill_weight) * loss + distill_weight * soft_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item()) * len(labels)
        correct += int((logits.argmax(1) == labels).sum().item())
        count += len(labels)
    return total_loss / count, correct / count


def make_criterion(
    labels: np.ndarray,
    loss_name: str,
    focal_gamma: float,
    class_balance_beta: float,
    label_smoothing: float = 0.02,
) -> nn.Module:
    """Construct a conservative classification loss for head ablations.

    The default remains the historical label-smoothed CE.  Focal and
    effective-number weighting are intentionally opt-in and use only the
    training split labels, never validation/test labels.
    """
    if loss_name == "ce":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    counts = np.bincount(
        np.asarray(labels, dtype=np.int64), minlength=N_CLASSES
    ).astype(np.float32)
    if loss_name == "focal":
        return FocalCrossEntropy(gamma=focal_gamma, smoothing=label_smoothing)
    if loss_name == "classbalanced":
        beta = float(class_balance_beta)
        present = counts > 0
        effective = 1.0 - np.power(beta, counts[present])
        weights = np.zeros_like(counts)
        weights[present] = (1.0 - beta) / np.maximum(effective, 1e-6)
        weights[present] *= float(present.sum()) / float(weights[present].sum())
        return nn.CrossEntropyLoss(
            weight=torch.from_numpy(weights), label_smoothing=label_smoothing
        )
    raise ValueError(f"unknown loss={loss_name}")


class FocalCrossEntropy(nn.Module):
    def __init__(self, gamma: float = 1.5, smoothing: float = 0.02):
        super().__init__()
        self.gamma = float(gamma)
        self.smoothing = float(smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        nclass = logits.shape[1]
        with torch.no_grad():
            smooth_target = torch.full_like(logp, self.smoothing / max(nclass - 1, 1))
            smooth_target.scatter_(1, target[:, None], 1.0 - self.smoothing)
        focal = (1.0 - p).clamp_min(1e-6).pow(self.gamma)
        return -(smooth_target * focal * logp).sum(dim=1).mean()


@torch.inference_mode()
def evaluate(model, loader, criterion, device, use_tta: bool = False):
    model.eval()
    total_loss = correct = count = 0
    logits_all, labels_all = [], []
    for frames, labels in loader:
        frames, labels = (
            frames.to(device, non_blocking=True),
            labels.to(device, non_blocking=True),
        )
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(frames)
            if use_tta:
                logits = (logits + model(frames.flip(-1))) * 0.5
            loss = criterion(logits, labels)
        total_loss += float(loss.item()) * len(labels)
        correct += int((logits.argmax(1) == labels).sum().item())
        count += len(labels)
        logits_all.append(logits.float().cpu().numpy())
        labels_all.append(labels.cpu().numpy())
    return (
        total_loss / count,
        correct / count,
        np.concatenate(logits_all),
        np.concatenate(labels_all),
    )


@torch.inference_mode()
def infer_test(model, loader, device, use_tta: bool = True) -> np.ndarray:
    model.eval()
    result = np.zeros((len(loader.dataset), N_CLASSES), dtype=np.float32)
    for frames, indices in loader:
        frames = frames.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(frames)
            if use_tta:
                logits = (logits + model(frames.flip(-1))) * 0.5
        result[indices.numpy()] = logits.float().cpu().numpy()
    return result


def set_trainable(model: nn.Module, mode: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(mode == "full")
    if mode == "head":
        for parameter in model.head.parameters():
            parameter.requires_grad_(True)
    elif mode == "adapter":
        if isinstance(model.input_adapter, nn.Identity):
            raise ValueError("--mode adapter requires --input-adapter")
        for parameter in model.input_adapter.parameters():
            parameter.requires_grad_(True)
        for parameter in model.head.parameters():
            parameter.requires_grad_(True)
    elif mode == "last":
        for parameter in model.encoder.layer4.parameters():
            parameter.requires_grad_(True)
        for parameter in model.head.parameters():
            parameter.requires_grad_(True)
    elif mode != "full":
        raise ValueError(f"unknown mode {mode!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache", type=Path, default=REPO_ROOT / ".cache/strict_v3/train_depth_ir.npy"
    )
    parser.add_argument(
        "--test-cache",
        type=Path,
        default=REPO_ROOT / ".cache/strict_v3/test_depth_ir.npy",
    )
    parser.add_argument(
        "--secondary-cache",
        type=Path,
        help="train-only alternate view sampled with fixed probability 0.5",
    )
    parser.add_argument("--metadata", type=Path, default=RESULT_DIR / "metadata.npz")
    parser.add_argument("--packed", type=Path, default=CHECKPOINT_DIR / "model.pt")
    parser.add_argument(
        "--source-encoder",
        type=Path,
        default=None,
        help="optional PKU-MMD source-pretrained checkpoint; loads encoder_state only",
    )
    parser.add_argument(
        "--reset-head",
        action="store_true",
        help=(
            "reinitialize the 40-class target head after loading public/source weights; "
            "required for strict held-user evaluation when the packed 40-class head "
            "may have seen all target users"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-index", type=int, default=0, choices=[0, 1])
    parser.add_argument("--bits", type=int, default=5, choices=[5, 6])
    parser.add_argument(
        "--val-users",
        type=int,
        nargs="*",
        default=[3, 8, 19, 24],
        help="subject IDs for validation; pass no IDs for full-data fine-tuning",
    )
    parser.add_argument(
        "--train-users",
        type=int,
        nargs="*",
        default=None,
        help="optional subject pool for fitting (useful for cohort-specific adaptation)",
    )
    parser.add_argument(
        "--mode", choices=["head", "adapter", "last", "full"], default="last"
    )
    parser.add_argument(
        "--input-adapter",
        action="store_true",
        help="keep the pretrained 3-channel stem and train a 4->3 adapter",
    )
    parser.add_argument(
        "--freeze-encoder-bn",
        action="store_true",
        help="freeze target-fold encoder BN running statistics during fine-tuning",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument(
        "--select-last",
        action="store_true",
        help="use the predeclared final epoch instead of held-user selection",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument(
        "--head-lr",
        type=float,
        default=None,
        help="optional separate learning rate for the freshly initialized target head",
    )
    parser.add_argument(
        "--head-warmup-epochs",
        type=int,
        default=0,
        help="linearly warm the freshly initialized target head for this many epochs",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-flip-augment", action="store_true")
    parser.add_argument(
        "--style-augment",
        action="store_true",
        help="apply train-only sensor rendering/style randomization",
    )
    parser.add_argument(
        "--temporal-jitter",
        action="store_true",
        help="apply train-only monotonic temporal index jitter",
    )
    parser.add_argument(
        "--temporal-difference",
        action="store_true",
        help=(
            "train with an auxiliary IR temporal-difference view; "
            "the difference branch is discarded at inference"
        ),
    )
    parser.add_argument(
        "--modality-dropout",
        type=float,
        default=0.0,
        help="train-only probability of dropping each depth/IR stream",
    )
    parser.add_argument(
        "--user-balanced",
        action="store_true",
        help=(
            "sample outer-train clips with equal subject probability; "
            "weights use only the requested training users"
        ),
    )
    parser.add_argument(
        "--hard-example-oof",
        type=Path,
        help=(
            "cross-fitted train OOF logits used only to upsample "
            "outer-train mistakes; never reads held/test labels"
        ),
    )
    parser.add_argument(
        "--hard-example-multiplier",
        type=float,
        default=2.0,
        help="sampling multiplier for cross-fitted OOF mistakes",
    )
    parser.add_argument("--sensor-mode", choices=("all", "depth", "ir"), default="all")
    parser.add_argument(
        "--pack",
        action="store_true",
        help="write an int5/int6 replacement packed checkpoint",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="skip test-logit extraction for validation-only runs",
    )
    parser.add_argument(
        "--distill-weight",
        type=float,
        default=0.0,
        help="blend CE with KL to the untouched source member",
    )
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument(
        "--loss", choices=("ce", "focal", "classbalanced"), default="ce"
    )
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--class-balance-beta", type=float, default=0.999)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument(
        "--preprocessing-contract",
        type=Path,
        help="frozen train-only affine statistics; no test statistics are read",
    )
    parser.add_argument(
        "--normalize-after-affine",
        action="store_true",
        help=(
            "after train-only affine matching to Kinetics channel moments, apply the "
            "Kinetics normalization expected by an external-only encoder"
        ),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.head_warmup_epochs < 0:
        raise ValueError("--head-warmup-epochs must be non-negative")
    if args.head_warmup_epochs and args.head_lr is None:
        raise ValueError(
            "--head-warmup-epochs requires --head-lr so only the new head is warmed"
        )
    if args.hard_example_multiplier < 1.0 or not math.isfinite(
        args.hard_example_multiplier
    ):
        raise ValueError("--hard-example-multiplier must be finite and at least one")
    seed_everything(args.seed)
    device = torch.device("cuda:0")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.metadata) as metadata:
        labels = metadata["train_y"].astype(np.int64)
        users = metadata["train_users"].astype(np.int64)
    val_mask = np.isin(users, args.val_users)
    fit_mask = (
        np.ones(len(users), dtype=bool)
        if args.train_users is None
        else np.isin(users, args.train_users)
    )
    train_indices, val_indices = (
        np.flatnonzero(fit_mask & ~val_mask),
        np.flatnonzero(val_mask),
    )
    if not len(train_indices):
        raise RuntimeError("the requested training-user pool is empty")
    if set(users[train_indices]).intersection(set(users[val_indices])):
        raise RuntimeError("subject leakage in split")

    source_mean = source_std = None
    preprocessing_contract_used = False
    if args.preprocessing_contract is not None:
        contract = json.loads(args.preprocessing_contract.read_text())
    else:
        release_package = torch.load(args.packed, map_location="cpu", weights_only=True)
        contract = release_package.get("release_contract", {}).get(
            "preprocessing_contract"
        )
    if contract is not None:
        if contract.get("source_split") != "train" or contract.get(
            "test_statistics_used", True
        ):
            raise RuntimeError("preprocessing contract must be fitted on train only")
        source_mean = torch.tensor(contract["mean"], dtype=torch.float32).view(
            1, CHANNELS, 1, 1
        )
        source_std = torch.tensor(contract["std"], dtype=torch.float32).view(
            1, CHANNELS, 1, 1
        )
        preprocessing_contract_used = True

    model = make_model(args.input_adapter)
    strict_external_init = bool(args.source_encoder is not None and args.reset_head)
    packed = None
    if strict_external_init:
        if args.input_adapter:
            raise ValueError(
                "--input-adapter is incompatible with strict external source initialization"
            )
        print(
            json.dumps({"competition_checkpoint_initialization_skipped": True}),
            flush=True,
        )
    else:
        packed = torch.load(args.packed, map_location="cpu", weights_only=True)
        members = packed.get("visual_members")
        if members:
            state = dequantize_state(members[args.model_index]["model_state_packed"])
        else:
            state = dequantize_state(packed["models_packed"][args.model_index])
        if args.input_adapter:
            # The released member has a 4-channel replacement stem.  The adapter
            # variant restores the original pretrained 3-channel kernels.
            stem = state.get("encoder.stem.0.weight")
            if stem is None or stem.ndim != 5 or stem.shape[1] != CHANNELS:
                raise RuntimeError(
                    "source checkpoint lacks the expected 4-channel stem"
                )
            state["encoder.stem.0.weight"] = stem[:, :3].contiguous()
        missing, unexpected = model.load_state_dict(state, strict=False)
        allowed_missing = (
            {key for key in model.state_dict() if key.startswith("input_adapter.")}
            if args.input_adapter
            else set()
        )
        if set(missing) - allowed_missing or unexpected:
            raise RuntimeError(
                f"state mismatch missing={missing} unexpected={unexpected}"
            )
        del state
    if args.source_encoder is not None:
        source_payload = torch.load(
            args.source_encoder, map_location="cpu", weights_only=True
        )
        source_state = (
            source_payload.get("encoder_state")
            if isinstance(source_payload, Mapping)
            else None
        )
        if not isinstance(source_state, Mapping):
            raise RuntimeError(
                f"{args.source_encoder} does not contain encoder_state; "
                "use a PKU-MMD pretrain_pkummd checkpoint"
            )
        source_state = dict(source_state)
        source_stem = source_state.get("stem.0.weight")
        target_stem = model.encoder.state_dict().get("stem.0.weight")
        source_stem_expanded = False
        if (
            source_stem is not None
            and target_stem is not None
            and source_stem.ndim == target_stem.ndim
            and source_stem.shape[1] == 3
            and target_stem.shape[1] == 4
            and source_stem.shape[0] == target_stem.shape[0]
        ):
            expanded = target_stem.clone()
            expanded[:, :3].copy_(source_stem)
            expanded[:, 3:].copy_(source_stem.mean(dim=1, keepdim=True))
            source_state["stem.0.weight"] = expanded
            source_stem_expanded = True
        source_missing, source_unexpected = model.encoder.load_state_dict(
            source_state, strict=False
        )
        if source_missing or source_unexpected:
            raise RuntimeError(
                f"source encoder mismatch missing={source_missing} unexpected={source_unexpected}"
            )
        print(
            json.dumps(
                {
                    "loaded_source_encoder": str(args.source_encoder.resolve()),
                    "source_epoch": source_payload.get("epoch"),
                    "source_validation_accuracy": source_payload.get(
                        "validation_accuracy"
                    ),
                    "source_stem_3ch_to_4ch_expanded": source_stem_expanded,
                }
            ),
            flush=True,
        )
    if args.reset_head:
        linear = model.head[-1]
        if not isinstance(linear, nn.Linear) or linear.out_features != N_CLASSES:
            raise RuntimeError("unexpected target head; cannot apply --reset-head")
        linear.reset_parameters()
        print(json.dumps({"target_head_reset": True}), flush=True)
    model.freeze_encoder_bn = bool(args.freeze_encoder_bn or args.input_adapter)
    model.to(device)
    teacher = None
    if args.distill_weight > 0:
        teacher = make_model(args.input_adapter)
        teacher.load_state_dict(model.state_dict(), strict=True)
        teacher.to(device).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    set_trainable(model, args.mode)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if args.head_lr is None:
        optimizer_groups = [
            {
                "params": trainable,
                "lr": args.lr,
                "name": "backbone",
                "target_lr": args.lr,
            }
        ]
    else:
        head_ids = {
            id(parameter)
            for parameter in model.head.parameters()
            if parameter.requires_grad
        }
        backbone_parameters = [
            parameter for parameter in trainable if id(parameter) not in head_ids
        ]
        head_parameters = [
            parameter for parameter in trainable if id(parameter) in head_ids
        ]
        if not head_parameters:
            raise RuntimeError(
                "--head-lr requested but no trainable head parameters were found"
            )
        optimizer_groups = []
        if backbone_parameters:
            optimizer_groups.append(
                {
                    "params": backbone_parameters,
                    "lr": args.lr,
                    "name": "backbone",
                    "target_lr": args.lr,
                }
            )
        optimizer_groups.append(
            {
                "params": head_parameters,
                "lr": args.head_lr,
                "name": "head",
                "target_lr": args.head_lr,
            }
        )
    optimizer = torch.optim.AdamW(optimizer_groups, lr=args.lr, weight_decay=0.01)
    scheduler = (
        None
        if args.head_warmup_epochs
        else torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs)
        )
    )
    scaler = torch.amp.GradScaler("cuda")
    criterion = make_criterion(
        labels[train_indices],
        args.loss,
        args.focal_gamma,
        args.class_balance_beta,
        args.label_smoothing,
    ).to(device)
    train_dataset = VideoDataset(
        args.cache,
        train_indices,
        labels,
        True,
        not args.no_flip_augment,
        args.sensor_mode,
        source_mean,
        source_std,
        args.normalize_after_affine,
        args.style_augment,
        args.temporal_jitter,
        args.modality_dropout,
        args.temporal_difference,
        args.secondary_cache,
    )
    train_sampler = None
    sample_weights = np.ones(len(train_indices), dtype=np.float64)
    if args.user_balanced:
        # Equalize subject probability using outer-train users only.  This is
        # a source-only domain-generalization control: no held-user count,
        # class prior, availability bit, or test statistic enters the weight.
        train_user_values = users[train_indices]
        unique_users, inverse = np.unique(train_user_values, return_inverse=True)
        user_counts = np.bincount(inverse, minlength=len(unique_users)).astype(
            np.float64
        )
        sample_weights *= 1.0 / np.maximum(user_counts[inverse], 1.0)
    hard_example_count = 0
    if args.hard_example_oof is not None:
        teacher_oof = np.asarray(np.load(args.hard_example_oof), dtype=np.float32)
        if (
            teacher_oof.shape != (len(labels), N_CLASSES)
            or not np.isfinite(teacher_oof).all()
        ):
            raise ValueError(
                f"--hard-example-oof must have shape {(len(labels), N_CLASSES)} and be finite"
            )
        hard_example = teacher_oof[train_indices].argmax(1) != labels[train_indices]
        hard_example_count = int(hard_example.sum())
        sample_weights[hard_example] *= float(args.hard_example_multiplier)
    if args.user_balanced or args.hard_example_oof is not None:
        train_sampler = WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(train_indices),
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        generator=torch.Generator().manual_seed(args.seed),
        drop_last=False,
    )
    val_loader = (
        DataLoader(
            VideoDataset(
                args.cache,
                val_indices,
                labels,
                False,
                False,
                args.sensor_mode,
                source_mean,
                source_std,
                args.normalize_after_affine,
                False,
                False,
                0.0,
            ),
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
        )
        if len(val_indices)
        else None
    )
    test_loader = (
        None
        if args.skip_test
        else DataLoader(
            VideoDataset(
                args.test_cache,
                np.arange(len(np.load(args.test_cache, mmap_mode="r"))),
                None,
                False,
                False,
                args.sensor_mode,
                source_mean,
                source_std,
                args.normalize_after_affine,
                False,
                False,
                0.0,
            ),
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
        )
    )
    best = (-1.0, None, None)
    history = []
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "model_index": args.model_index,
                "mode": args.mode,
                "train": len(train_indices),
                "val": len(val_indices),
                "trainable_parameters": sum(p.numel() for p in trainable),
                "user_balanced": args.user_balanced,
                "hard_example_oof": (
                    str(args.hard_example_oof.resolve())
                    if args.hard_example_oof is not None
                    else None
                ),
                "hard_example_multiplier": (
                    args.hard_example_multiplier
                    if args.hard_example_oof is not None
                    else 1.0
                ),
                "hard_example_train_count": hard_example_count,
                "backbone_lr": args.lr,
                "head_lr": args.head_lr or args.lr,
                "loss": args.loss,
                "label_smoothing": args.label_smoothing,
            }
        ),
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        if scheduler is None:
            # Keep the pretrained backbone on the normal cosine schedule and
            # warm only the freshly initialized head.  This avoids a large
            # first update from a random 40-way head without changing the
            # matched backbone contract.
            cosine = 0.5 * (
                1.0 + math.cos(math.pi * (epoch - 1) / max(1, args.epochs - 1))
            )
            for group in optimizer.param_groups:
                target = float(group["target_lr"])
                warmup = (
                    min(1.0, epoch / max(1, args.head_warmup_epochs))
                    if group.get("name") == "head"
                    else 1.0
                )
                group["lr"] = target * warmup * cosine
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            criterion,
            device,
            args.mode,
            teacher=teacher,
            distill_weight=args.distill_weight,
            distill_temperature=args.distill_temperature,
            mixup_alpha=args.mixup_alpha,
            temporal_difference=args.temporal_difference,
        )
        if val_loader is None:
            val_loss, val_acc = float("nan"), float("nan")
            val_logits = np.zeros((0, N_CLASSES), dtype=np.float32)
        else:
            val_loss, val_acc, val_logits, val_labels = evaluate(
                model, val_loader, criterion, device, use_tta=True
            )
        if scheduler is not None:
            scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "lr": (
                scheduler.get_last_lr()[0]
                if scheduler is not None
                else optimizer.param_groups[0]["lr"]
            ),
            "learning_rates": (
                scheduler.get_last_lr()
                if scheduler is not None
                else [group["lr"] for group in optimizer.param_groups]
            ),
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        selected = val_loader is None or (
            epoch == args.epochs if args.select_last else val_acc > best[0]
        )
        if selected:
            best = (val_acc, val_logits, epoch)
            torch.save(
                {
                    "model_state": {
                        k: v.detach().cpu().half()
                        if v.is_floating_point()
                        else v.detach().cpu()
                        for k, v in model.state_dict().items()
                    },
                    "epoch": epoch,
                    "validation_accuracy": val_acc,
                    "val_users": args.val_users,
                    "source_model_index": args.model_index,
                    "mode": args.mode,
                    "input_adapter": args.input_adapter,
                },
                args.output_dir / "best_fp16.pt",
            )
            np.save(args.output_dir / "best_val_logits.npy", val_logits)
            np.save(args.output_dir / "best_val_predictions.npy", val_logits.argmax(1))
    checkpoint = torch.load(
        args.output_dir / "best_fp16.pt", map_location="cpu", weights_only=True
    )
    best_model = make_model(args.input_adapter)
    best_model.load_state_dict(checkpoint["model_state"])
    best_model.to(device).eval()
    # The test cache is used only for feature extraction; no labels are loaded.
    if test_loader is not None:
        test_logits = infer_test(best_model, test_loader, device, use_tta=True)
        np.save(args.output_dir / "test_logits.npy", test_logits)
        np.save(args.output_dir / "test_predictions.npy", test_logits.argmax(1))
    if args.pack:
        if packed is None:
            packed = torch.load(args.packed, map_location="cpu", weights_only=True)
        replacement = packed_model(checkpoint["model_state"], args.bits)
        candidate = dict(packed)
        models = list(candidate["models_packed"])
        models[args.model_index] = replacement
        candidate["models_packed"] = models
        candidate["bits"] = list(candidate["bits"])
        candidate["bits"][args.model_index] = args.bits
        packed_path = args.output_dir / "ensemble_packed.pt"
        torch.save(candidate, packed_path)
    else:
        packed_path = None
    metrics = {
        "model_index": args.model_index,
        "bits": args.bits,
        "mode": args.mode,
        "input_adapter": args.input_adapter,
        "source_encoder": str(args.source_encoder.resolve())
        if args.source_encoder is not None
        else None,
        "target_head_reset": bool(args.reset_head),
        "competition_checkpoint_initialization_skipped": strict_external_init,
        "freeze_encoder_bn": model.freeze_encoder_bn,
        "sensor_mode": args.sensor_mode,
        "loss": args.loss,
        "hard_example_oof": (
            str(args.hard_example_oof.resolve())
            if args.hard_example_oof is not None
            else None
        ),
        "hard_example_multiplier": (
            args.hard_example_multiplier if args.hard_example_oof is not None else 1.0
        ),
        "hard_example_train_count": hard_example_count,
        "backbone_lr": args.lr,
        "head_lr": args.head_lr or args.lr,
        "head_warmup_epochs": args.head_warmup_epochs,
        "label_smoothing": args.label_smoothing,
        "mixup_alpha": args.mixup_alpha,
        "style_augment": bool(args.style_augment),
        "temporal_jitter": bool(args.temporal_jitter),
        "temporal_difference": bool(args.temporal_difference),
        "difference_loss_weight": 0.25 if args.temporal_difference else 0.0,
        "modality_dropout": float(args.modality_dropout),
        "secondary_cache": None
        if args.secondary_cache is None
        else str(args.secondary_cache.resolve()),
        "secondary_cache_probability": 0.5 if args.secondary_cache is not None else 0.0,
        "user_balanced": bool(args.user_balanced),
        "val_users": args.val_users,
        "train_clips": len(train_indices),
        "validation_clips": len(val_indices),
        "best_validation_accuracy": best[0],
        "best_epoch": best[2],
        "history": history,
        "checkpoint_selection": "fixed_final_epoch"
        if args.select_last
        else "best_accuracy",
        "test_labels_opened": False,
        "trainable_parameters": sum(p.numel() for p in trainable),
        "preprocessing_contract": "embedded in checkpoint"
        if args.preprocessing_contract is None
        else str(args.preprocessing_contract),
        "train_only_preprocessing": preprocessing_contract_used,
        "normalize_after_affine": bool(args.normalize_after_affine),
        "fp16_checkpoint_bytes": (args.output_dir / "best_fp16.pt").stat().st_size,
        "yolo_bytes": (CHECKPOINT_DIR / "yolo11n.pt").stat().st_size,
        "packed_candidate": str(packed_path) if packed_path else None,
        "packed_candidate_bytes": packed_path.stat().st_size if packed_path else None,
        "combined_packed_plus_yolo_bytes": (
            packed_path.stat().st_size + (CHECKPOINT_DIR / "yolo11n.pt").stat().st_size
        )
        if packed_path
        else None,
        "under_100mb": bool(
            packed_path
            and packed_path.stat().st_size
            + (CHECKPOINT_DIR / "yolo11n.pt").stat().st_size
            <= 100_000_000
        ),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(
        json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
