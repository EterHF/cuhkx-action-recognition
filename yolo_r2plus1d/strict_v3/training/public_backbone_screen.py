#!/usr/bin/env python3
"""Screen public Depth/IR encoders without loading project-trained checkpoints."""

from __future__ import annotations

import argparse
import builtins
import hashlib
import importlib
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from yolo_r2plus1d.strict_v3.paths import RESULT_DIR

HELD_USERS = (1, 6, 17, 22)
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1, 1)
DEFM_MEAN = torch.tensor((0.248880, 0.495620, 0.492858)).view(1, 3, 1, 1)
DEFM_STD = torch.tensor((0.139357, 0.271314, 0.297177)).view(1, 3, 1, 1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_jet_inverse_lut() -> torch.Tensor:
    """Map quantised RGB pseudo-colour to the closest OpenCV JET index."""
    jet = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)
    jet = jet.reshape(256, 3)[:, ::-1].astype(np.float32)
    axis = np.arange(32, dtype=np.float32) * 8.0 + 3.5
    rgb = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    distance = ((rgb[:, None] - jet[None]) ** 2).sum(axis=-1)
    return torch.from_numpy(distance.argmin(axis=1).astype(np.float32) / 255.0)


JET_INVERSE_LUT = make_jet_inverse_lut()


def depth_color_to_inverse(frames: torch.Tensor) -> torch.Tensor:
    """Approximate the scalar inverse-depth index used to colourise Depth_Color."""
    if frames.shape[-3] != 3:
        raise ValueError("expected RGB Depth_Color frames")
    quantised = frames.to(torch.int64).div(8, rounding_mode="floor").clamp_(0, 31)
    index = quantised[..., 0, :, :] * 1024 + quantised[..., 1, :, :] * 32 + quantised[..., 2, :, :]
    inverse = JET_INVERSE_LUT.to(index.device)[index]
    black = frames.sum(dim=-3) < 8
    return inverse.masked_fill(black, 0.0)


def defm_preprocess(inverse_depth: torch.Tensor, image_size: int = 224) -> torch.Tensor:
    """Apply DeFM's published non-metric-depth conversion in a batch."""
    flat = inverse_depth.flatten(1)
    low = flat.amin(1).view(-1, 1, 1)
    high = flat.amax(1).view(-1, 1, 1)
    normalised = (inverse_depth - low) / (high - low).clamp_min(1e-8)
    metric = 10.0 * torch.exp(-5.0 * normalised)
    log_depth = torch.log1p(metric)
    c1 = log_depth / np.log1p(100.0)
    c2 = (log_depth / np.log1p(9.0)).clamp(0.0, 1.0)
    low = log_depth.flatten(1).amin(1).view(-1, 1, 1)
    high = log_depth.flatten(1).amax(1).view(-1, 1, 1)
    c3 = (log_depth - low) / (high - low).clamp_min(1e-8)
    value = torch.stack((c1, c2, c3), dim=1)
    value = F.interpolate(
        value, size=(image_size, image_size), mode="bilinear", align_corners=False
    )
    return (value - DEFM_MEAN.to(value.device)) / DEFM_STD.to(value.device)


class ClipDataset(Dataset):
    def __init__(self, cache: Path) -> None:
        self.frames = np.load(cache, mmap_mode="r")

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(np.array(self.frames[index], copy=True))


def loader(cache: Path, batch_size: int, workers: int) -> DataLoader:
    return DataLoader(
        ClipDataset(cache),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def load_defm(args: argparse.Namespace) -> nn.Module:
    sys.path.insert(0, str(args.source_dir.resolve()))
    from defm.model_factory import create_defm_model

    model = create_defm_model(args.model, pretrained=True, pretrained_path=args.weights)
    return model


def defm_features(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "forward_no_bifpn"):
        output = model.forward_no_bifpn(inputs)
    else:
        output = model(inputs)
    if isinstance(output, dict):
        return output["global_backbone"]
    if output.ndim == 3:
        return output[:, 0]
    if output.ndim == 2:
        return output
    raise RuntimeError(f"unsupported DeFM output shape: {output.shape}")


@torch.inference_mode()
def extract_defm(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    model = load_defm(args).to(device).eval()
    output = None
    offset = 0
    for batch_index, frames in enumerate(loader(args.cache, args.batch_size, args.workers), 1):
        batch, time = frames.shape[:2]
        depth = frames[:, :, :3].to(device, non_blocking=True)
        inverse = depth_color_to_inverse(depth).reshape(batch * time, *depth.shape[-2:])
        inputs = defm_preprocess(inverse, args.image_size)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            features = defm_features(model, inputs)
        features = features.reshape(batch, time, -1).float().cpu().numpy().astype(np.float16)
        if output is None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            output = np.lib.format.open_memmap(
                args.output,
                mode="w+",
                dtype=np.float16,
                shape=(len(loader(args.cache, 1, 0).dataset), time, features.shape[-1]),
            )
        output[offset : offset + batch] = features
        offset += batch
        if batch_index % 20 == 0:
            print(f"batches={batch_index} rows={offset}", flush=True)
    if output is None:
        raise RuntimeError("empty cache")
    output.flush()
    write_feature_manifest(args, tuple(output.shape), sum(p.numel() for p in model.parameters()))


def load_omnivore(args: argparse.Namespace) -> nn.Module:
    sys.path.insert(0, str(args.source_dir.resolve()))
    from omnivore.models import omnivore_swinT

    model = omnivore_swinT(pretrained=False, load_heads=False)
    payload = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["trunk"], strict=True)
    return model


@torch.inference_mode()
def extract_omnivore(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    model = load_omnivore(args).to(device)
    # Omnivore's archived Swin override of ``train`` does not return ``self``,
    # so chaining ``.eval()`` turns the local reference into ``None``.
    model.eval()
    values = []
    for batch_index, frames in enumerate(loader(args.cache, args.batch_size, args.workers), 1):
        frames = frames.to(device, non_blocking=True)
        if args.modality == "depth":
            inverse = depth_color_to_inverse(frames[:, :, :3])
            inputs = inverse[:, None].expand(-1, 3, -1, -1, -1)
        else:
            inputs = frames[:, :, 3:4].permute(0, 2, 1, 3, 4).expand(-1, 3, -1, -1, -1)
            inputs = inputs.float().div_(255.0)
        if args.modality == "depth":
            inputs = inputs.float()
        inputs = (inputs - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            features = model(inputs)
        values.append(features[:, None].float().cpu().numpy().astype(np.float16))
        if batch_index % 20 == 0:
            print(f"batches={batch_index}", flush=True)
    output = np.concatenate(values)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, output)
    write_feature_manifest(args, tuple(output.shape), sum(p.numel() for p in model.parameters()))


def load_dformer(args: argparse.Namespace) -> nn.Module:
    sys.path.insert(0, str(args.source_dir.resolve()))
    from models.encoders.DFormerv2 import DFormerv2_B, DFormerv2_L, DFormerv2_S

    factories = {
        "DFormerv2_S": DFormerv2_S,
        "DFormerv2_B": DFormerv2_B,
        "DFormerv2_L": DFormerv2_L,
    }
    if args.model not in factories:
        raise ValueError(f"unsupported DFormerv2 model: {args.model}")
    model = factories[args.model]()
    payload = torch.load(args.weights, map_location="cpu", weights_only=True)
    state = payload.get("model", payload.get("state_dict", payload))
    incompatible = model.load_state_dict(state, strict=False)
    expected_missing = {
        f"extra_norms.{index}.{suffix}" for index in range(3) for suffix in ("weight", "bias")
    }
    expected_unexpected = {
        "proj.weight",
        "proj.bias",
        "norm.weight",
        "norm.bias",
        "norm.running_mean",
        "norm.running_var",
        "norm.num_batches_tracked",
        "head.weight",
        "head.bias",
        "aux_head.weight",
        "aux_head.bias",
    }
    mismatch = (set(incompatible.missing_keys), set(incompatible.unexpected_keys))
    if mismatch != (set(), set()) and mismatch != (expected_missing, expected_unexpected):
        raise RuntimeError(f"unexpected DFormerv2 checkpoint mismatch: {incompatible}")
    return model


@torch.inference_mode()
def extract_dformer(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    model = load_dformer(args).to(device)
    model.eval()
    output = None
    offset = 0
    dataset_rows = len(loader(args.cache, 1, 0).dataset)
    for batch_index, frames in enumerate(loader(args.cache, args.batch_size, args.workers), 1):
        batch, time = frames.shape[:2]
        depth_colour = frames[:, :, :3].to(device, non_blocking=True)
        inverse = depth_color_to_inverse(depth_colour)
        rgb = depth_colour.reshape(batch * time, 3, *depth_colour.shape[-2:]).float()
        rgb = F.interpolate(
            rgb.div_(255.0),
            size=(args.image_size, args.image_size),
            mode="bilinear",
            align_corners=False,
        )
        rgb = (rgb - IMAGENET_MEAN[:, :, 0].to(device)) / IMAGENET_STD[:, :, 0].to(device)
        depth = F.interpolate(
            inverse.reshape(batch * time, 1, *inverse.shape[-2:]),
            size=(args.image_size, args.image_size),
            mode="bilinear",
            align_corners=False,
        )
        with torch.autocast(device.type, dtype=torch.bfloat16):
            features = model(rgb, depth)[-1].mean(dim=(-1, -2))
        features = features.reshape(batch, time, -1).float().cpu().numpy().astype(np.float16)
        if output is None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            output = np.lib.format.open_memmap(
                args.output,
                mode="w+",
                dtype=np.float16,
                shape=(dataset_rows, time, features.shape[-1]),
            )
        output[offset : offset + batch] = features
        offset += batch
        if batch_index % 20 == 0:
            print(f"batches={batch_index} rows={offset}", flush=True)
    if output is None:
        raise RuntimeError("empty cache")
    output.flush()
    write_feature_manifest(args, tuple(output.shape), sum(p.numel() for p in model.parameters()))


def load_mspecgene(args: argparse.Namespace) -> nn.Module:
    sys.path.insert(0, str(args.source_dir.resolve()))
    from mmengine.logging.history_buffer import HistoryBuffer
    from mmpretrain.models.selfsup.mae import MAEViT

    model = MAEViT(arch="b", patch_size=16, mask_ratio=0.9)
    multiarray = importlib.import_module("numpy._core.multiarray")
    numpy_dtype_types = {
        type(np.dtype(dtype))
        for dtype in (np.bool_, np.float32, np.float64, np.int32, np.int64, np.uint64)
    }
    safe_globals = [
        HistoryBuffer,
        builtins.getattr,
        (multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
        (multiarray.scalar, "numpy.core.multiarray.scalar"),
        np.ndarray,
        np.dtype,
        *numpy_dtype_types,
    ]
    with torch.serialization.safe_globals(safe_globals):
        payload = torch.load(args.weights, map_location="cpu", weights_only=True)
    state = payload.get("state_dict", payload.get("model", payload))
    state = {key.removeprefix("module."): value for key, value in state.items()}
    if any(key.startswith("backbone.") for key in state):
        state = {
            key.removeprefix("backbone."): value
            for key, value in state.items()
            if key.startswith("backbone.")
        }
    expected = set(model.state_dict())
    if set(state) != expected:
        missing = sorted(expected - set(state))
        unexpected = sorted(set(state) - expected)
        raise RuntimeError(
            f"unexpected M-SpecGene encoder mismatch: missing={missing}, unexpected={unexpected}"
        )
    model.load_state_dict(state, strict=True)
    return model


@torch.inference_mode()
def extract_mspecgene(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    model = load_mspecgene(args).to(device)
    # This archived mmpretrain fork overrides ``train`` without returning self.
    model.eval()
    output = None
    offset = 0
    dataset_rows = len(loader(args.cache, 1, 0).dataset)
    for batch_index, frames in enumerate(loader(args.cache, args.batch_size, args.workers), 1):
        batch, time = frames.shape[:2]
        infrared = frames[:, :, 3:4].reshape(batch * time, 1, *frames.shape[-2:])
        inputs = infrared.to(device, non_blocking=True).float().div_(255.0).expand(-1, 3, -1, -1)
        inputs = F.interpolate(
            inputs,
            size=(args.image_size, args.image_size),
            mode="bicubic",
            align_corners=False,
        )
        inputs = (inputs - IMAGENET_MEAN[:, :, 0].to(device)) / IMAGENET_STD[:, :, 0].to(
            device
        )
        with torch.autocast(device.type, dtype=torch.bfloat16):
            tokens = model(inputs, None, None, None, mask=None)[0]
            features = tokens[:, 1:].mean(dim=1)
        features = features.reshape(batch, time, -1).float().cpu().numpy().astype(np.float16)
        if output is None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            output = np.lib.format.open_memmap(
                args.output,
                mode="w+",
                dtype=np.float16,
                shape=(dataset_rows, time, features.shape[-1]),
            )
        output[offset : offset + batch] = features
        offset += batch
        if batch_index % 20 == 0:
            print(f"batches={batch_index} rows={offset}", flush=True)
    if output is None:
        raise RuntimeError("empty cache")
    output.flush()
    write_feature_manifest(args, tuple(output.shape), sum(p.numel() for p in model.parameters()))


def write_feature_manifest(
    args: argparse.Namespace, shape: tuple[int, ...], parameters: int
) -> None:
    manifest = {
        "model": args.model,
        "modality": args.modality,
        "public_source_commit": args.source_commit,
        "public_weights": str(args.weights),
        "public_weights_sha256": sha256_file(args.weights),
        "project_checkpoint_loaded": False,
        "cache": str(args.cache),
        "cache_sha256": sha256_file(args.cache),
        "shape": shape,
        "encoder_parameters": parameters,
        "anonymous_test_accessed": False,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def combine_features(args: argparse.Namespace) -> None:
    arrays = [np.load(path, mmap_mode="r") for path in args.features]
    if any(array.ndim != 3 or array.shape[0] != arrays[0].shape[0] for array in arrays):
        raise ValueError("feature arrays must have shape [N,T,D] with matching rows")
    time = max(array.shape[1] for array in arrays)
    if any(array.shape[1] not in (1, time) for array in arrays):
        raise ValueError("time dimensions must match or be singleton")
    expanded = [
        np.broadcast_to(array, (len(array), time, array.shape[-1]))
        if array.shape[1] == 1
        else array
        for array in arrays
    ]
    output = np.concatenate(expanded, axis=-1).astype(np.float16)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, output)
    manifests = [
        json.loads(path.with_suffix(".json").read_text(encoding="utf-8")) for path in args.features
    ]
    manifest = {
        "model": "+".join(str(item["model"]) for item in manifests),
        "modality": "+".join(str(item["modality"]) for item in manifests),
        "components": manifests,
        "shape": output.shape,
        "encoder_parameters": sum(int(item["encoder_parameters"]) for item in manifests),
        "project_checkpoint_loaded": False,
        "anonymous_test_accessed": False,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def late_fuse(args: argparse.Namespace) -> None:
    """Fuse independently trained probes without fitting another held-fold model."""
    if not 0.0 <= args.depth_weight <= 1.0:
        raise ValueError("depth weight must be in [0, 1]")
    depth_indices = np.load(args.depth_dir / "held_indices.npy")
    ir_indices = np.load(args.ir_dir / "held_indices.npy")
    if not np.array_equal(depth_indices, ir_indices):
        raise ValueError("probe held indices do not match")
    depth = softmax_numpy(np.load(args.depth_dir / "held_logits.npy"))
    infrared = softmax_numpy(np.load(args.ir_dir / "held_logits.npy"))
    probabilities = args.depth_weight * depth + (1.0 - args.depth_weight) * infrared
    with np.load(args.metadata) as metadata:
        labels = metadata["train_y"][depth_indices]
        users = metadata["train_users"][depth_indices]
    prediction = probabilities.argmax(axis=1)
    user_accuracy = {
        str(int(user)): float(np.mean(prediction[users == user] == labels[users == user]))
        for user in np.unique(users)
    }
    metrics = {
        "protocol": "single-subject-fold-fixed-late-fusion/v1",
        "depth_probe": str(args.depth_dir),
        "ir_probe": str(args.ir_dir),
        "depth_weight": args.depth_weight,
        "accuracy": float(np.mean(prediction == labels)),
        "worst_user_accuracy": min(user_accuracy.values()),
        "user_accuracy": user_accuracy,
        "project_checkpoint_loaded": False,
        "anonymous_test_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


class FeatureDataset(Dataset):
    def __init__(self, features: Path, indices: np.ndarray, labels: np.ndarray) -> None:
        self.features = np.load(features, mmap_mode="r")
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = labels

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        return torch.from_numpy(np.array(self.features[index], copy=True)).float(), int(
            self.labels[index]
        )


class TemporalProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.input = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim))
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=2, dilation=2, groups=hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.GELU(),
        )
        self.pool = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Dropout(0.2), nn.Linear(hidden_dim, 40)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        value = self.input(features)
        value = value + self.temporal(value.transpose(1, 2)).transpose(1, 2)
        weights = self.pool(value).softmax(dim=1)
        return self.classifier((value * weights).sum(dim=1))


def train_probe(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    with np.load(args.metadata) as metadata:
        labels = metadata["train_y"]
        users = metadata["train_users"]
    held = np.isin(users, HELD_USERS)
    train_indices, held_indices = np.flatnonzero(~held), np.flatnonzero(held)
    if np.intersect1d(users[train_indices], users[held_indices]).size:
        raise RuntimeError("subject leakage")
    features = np.load(args.features, mmap_mode="r")
    model = TemporalProbe(int(features.shape[-1])).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    train_loader = DataLoader(
        FeatureDataset(args.features, train_indices, labels),
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    held_loader = DataLoader(
        FeatureDataset(args.features, held_indices, labels), batch_size=128, shuffle=False
    )
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        correct = count = 0
        for value, target in train_loader:
            value, target = value.to(args.device), target.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(value)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            correct += int((logits.argmax(1) == target).sum())
            count += len(target)
        scheduler.step()
        history.append({"epoch": epoch, "train_accuracy": correct / count})
    model.eval()
    logits = []
    with torch.inference_mode():
        for value, _ in held_loader:
            logits.append(model(value.to(args.device)).cpu().numpy())
    logits = np.concatenate(logits)
    prediction = logits.argmax(1)
    user_accuracy = {
        str(int(user)): float(
            np.mean(
                prediction[users[held_indices] == user]
                == labels[held_indices][users[held_indices] == user]
            )
        )
        for user in np.unique(users[held_indices])
    }
    metrics = {
        "protocol": "single-subject-fold-public-encoder-screen/v1",
        "held_users": list(HELD_USERS),
        "held_rows": len(held_indices),
        "accuracy": float(np.mean(prediction == labels[held_indices])),
        "worst_user_accuracy": min(user_accuracy.values()),
        "user_accuracy": user_accuracy,
        "fixed_epoch": args.epochs,
        "seed": args.seed,
        "feature_manifest": json.loads(
            args.features.with_suffix(".json").read_text(encoding="utf-8")
        ),
        "probe_parameters": sum(p.numel() for p in model.parameters()),
        "project_checkpoint_loaded": False,
        "held_labels_used_for_selection": False,
        "anonymous_test_accessed": False,
        "history": history,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "held_indices.npy", held_indices)
    np.save(args.output_dir / "held_logits.npy", logits)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in metrics.items() if key != "history"}, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--cache", type=Path, required=True)
    common.add_argument("--weights", type=Path, required=True)
    common.add_argument("--source-dir", type=Path, required=True)
    common.add_argument("--source-commit", required=True)
    common.add_argument("--model", required=True)
    common.add_argument("--modality", choices=("depth", "ir"), required=True)
    common.add_argument("--output", type=Path, required=True)
    common.add_argument("--device", default="cuda:0")
    common.add_argument("--batch-size", type=int, default=8)
    common.add_argument("--workers", type=int, default=4)
    defm = commands.add_parser("extract-defm", parents=[common])
    defm.add_argument("--image-size", type=int, default=224)
    defm.set_defaults(function=extract_defm)
    omnivore = commands.add_parser("extract-omnivore", parents=[common])
    omnivore.set_defaults(function=extract_omnivore)
    dformer = commands.add_parser("extract-dformer", parents=[common])
    dformer.add_argument("--image-size", type=int, default=224)
    dformer.set_defaults(function=extract_dformer)
    mspecgene = commands.add_parser("extract-mspecgene", parents=[common])
    mspecgene.add_argument("--image-size", type=int, default=224)
    mspecgene.set_defaults(function=extract_mspecgene)
    combine = commands.add_parser("combine-features")
    combine.add_argument("--features", type=Path, nargs="+", required=True)
    combine.add_argument("--output", type=Path, required=True)
    combine.set_defaults(function=combine_features)
    fusion = commands.add_parser("late-fuse")
    fusion.add_argument("--depth-dir", type=Path, required=True)
    fusion.add_argument("--ir-dir", type=Path, required=True)
    fusion.add_argument("--depth-weight", type=float, default=0.25)
    fusion.add_argument("--metadata", type=Path, default=RESULT_DIR / "metadata.npz")
    fusion.add_argument("--output", type=Path, required=True)
    fusion.set_defaults(function=late_fuse)
    probe = commands.add_parser("train-probe")
    probe.add_argument("--features", type=Path, required=True)
    probe.add_argument("--metadata", type=Path, default=RESULT_DIR / "metadata.npz")
    probe.add_argument("--output-dir", type=Path, required=True)
    probe.add_argument("--epochs", type=int, default=25)
    probe.add_argument("--seed", type=int, default=2026)
    probe.add_argument("--device", default="cuda:0")
    probe.set_defaults(function=train_probe)
    return root


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
