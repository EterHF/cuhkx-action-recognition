#!/usr/bin/env python3
"""Train skeleton-only and public-sensor fusion models on one fixed subject fold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from yolo_r2plus1d.strict_v3.models.public_sensor_fusion import PublicSensorFusion
from yolo_r2plus1d.strict_v3.models.research_skeleton import HighRateSkeletonClassifier
from yolo_r2plus1d.strict_v3.paths import RESULT_DIR
from yolo_r2plus1d.strict_v3.training.public_backbone_screen import (
    HELD_USERS,
    seed_everything,
    sha256_file,
)


class FusionDataset(Dataset):
    def __init__(
        self,
        indices: np.ndarray,
        labels: np.ndarray,
        depth_features: Path,
        ir_features: Path,
        skeleton_root: Path,
        skeleton_noise: float,
    ) -> None:
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = labels
        self.depth = np.load(depth_features, mmap_mode="r")
        self.infrared = np.load(ir_features, mmap_mode="r")
        self.skeleton = np.load(skeleton_root / "train_skeleton.npy", mmap_mode="r")
        self.mask = np.load(skeleton_root / "train_skeleton_mask.npy", mmap_mode="r")
        self.positions = np.load(skeleton_root / "train_positions.npy", mmap_mode="r")
        self.skeleton_noise = float(skeleton_noise)
        rows = len(labels)
        if any(len(value) != rows for value in (self.depth, self.infrared, self.skeleton)):
            raise ValueError("sensor and skeleton feature rows must match labels")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        skeleton = torch.from_numpy(np.array(self.skeleton[index], copy=True))
        mask = torch.from_numpy(np.array(self.mask[index], copy=True)).bool()
        if self.skeleton_noise:
            skeleton[mask] += torch.randn_like(skeleton[mask]) * self.skeleton_noise
        return (
            torch.from_numpy(np.array(self.depth[index], copy=True)).float(),
            torch.from_numpy(np.array(self.infrared[index], copy=True)).float(),
            skeleton,
            mask,
            torch.from_numpy(np.array(self.positions[index], copy=True)),
            int(self.labels[index]),
        )


def make_loader(
    dataset: Dataset, batch_size: int, shuffle: bool, workers: int, seed: int
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )


def apply_branch_dropout(
    batch: list[torch.Tensor], sensor_probability: float, skeleton_probability: float
) -> list[torch.Tensor]:
    depth, infrared, skeleton, mask, positions = batch
    if sensor_probability:
        depth[torch.rand(len(depth), device=depth.device) < sensor_probability] = 0
        infrared[torch.rand(len(infrared), device=infrared.device) < sensor_probability] = 0
    if skeleton_probability:
        keep = torch.rand(len(mask), device=mask.device) >= skeleton_probability
        mask = mask & keep[:, None]
    return [depth, infrared, skeleton, mask, positions]


def evaluate(
    model: nn.Module,
    mode: str,
    data_loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.inference_mode():
        for depth, infrared, skeleton, mask, positions, _ in data_loader:
            values = [
                value.to(device, non_blocking=True)
                for value in (depth, infrared, skeleton, mask, positions)
            ]
            with torch.autocast(device.type, dtype=torch.bfloat16):
                if mode == "skeleton":
                    logits = model(values[2], values[3], values[4])
                else:
                    logits = model(*values)
            outputs.append(logits.float().cpu().numpy())
    return np.concatenate(outputs)


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device(args.device)
    with np.load(args.metadata) as metadata:
        labels = metadata["train_y"]
        users = metadata["train_users"]
    held = np.isin(users, HELD_USERS)
    train_indices = np.flatnonzero(~held)
    held_indices = np.flatnonzero(held)
    if np.intersect1d(users[train_indices], users[held_indices]).size:
        raise RuntimeError("subject leakage")
    train_loader = make_loader(
        FusionDataset(
            train_indices,
            labels,
            args.depth_features,
            args.ir_features,
            args.skeleton_root,
            args.skeleton_noise,
        ),
        args.batch_size,
        True,
        args.workers,
        args.seed,
    )
    held_loader = make_loader(
        FusionDataset(
            held_indices,
            labels,
            args.depth_features,
            args.ir_features,
            args.skeleton_root,
            0.0,
        ),
        args.batch_size * 2,
        False,
        args.workers,
        args.seed,
    )
    depth_dim = int(np.load(args.depth_features, mmap_mode="r").shape[-1])
    ir_dim = int(np.load(args.ir_features, mmap_mode="r").shape[-1])
    if args.mode == "skeleton":
        model: nn.Module = HighRateSkeletonClassifier()
    else:
        model = PublicSensorFusion(depth_dim=depth_dim, ir_dim=ir_dim)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = correct = count = 0
        for depth, infrared, skeleton, mask, positions, target in train_loader:
            inputs = [
                value.to(device, non_blocking=True)
                for value in (depth, infrared, skeleton, mask, positions)
            ]
            inputs = apply_branch_dropout(
                inputs, args.sensor_dropout, args.skeleton_dropout
            )
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                if args.mode == "skeleton":
                    logits = model(inputs[2], inputs[3], inputs[4])
                    loss = criterion(logits, target)
                else:
                    logits, *auxiliary = model(*inputs, return_aux=True)
                    loss = criterion(logits, target)
                    loss += args.auxiliary_weight * sum(
                        criterion(auxiliary_logits, target)
                        for auxiliary_logits in auxiliary
                    )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(target)
            correct += int((logits.argmax(1) == target).sum())
            count += len(target)
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": total_loss / count,
            "train_accuracy": correct / count,
        }
        history.append(record)
        print(json.dumps(record), flush=True)

    logits = evaluate(model, args.mode, held_loader, device)
    prediction = logits.argmax(1)
    held_labels = labels[held_indices]
    held_users = users[held_indices]
    user_accuracy = {
        str(int(user)): float(np.mean(prediction[held_users == user] == held_labels[held_users == user]))
        for user in np.unique(held_users)
    }
    feature_manifests = {
        "depth": json.loads(args.depth_features.with_suffix(".json").read_text()),
        "ir": json.loads(args.ir_features.with_suffix(".json").read_text()),
    }
    metrics = {
        "protocol": "single-subject-fold-public-sensor-fusion/v1",
        "mode": args.mode,
        "held_users": list(HELD_USERS),
        "held_rows": len(held_indices),
        "accuracy": float(np.mean(prediction == held_labels)),
        "worst_user_accuracy": min(user_accuracy.values()),
        "user_accuracy": user_accuracy,
        "fixed_epoch": args.epochs,
        "seed": args.seed,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "feature_manifests": feature_manifests,
        "sensor_features_consumed": args.mode == "fusion",
        "skeleton_input_sha256": {
            name: sha256_file(args.skeleton_root / name)
            for name in ("train_skeleton.npy", "train_skeleton_mask.npy", "train_positions.npy")
        },
        "old_project_checkpoint_loaded": False,
        "current_experiment_checkpoint_used_for_sensor_features": args.mode == "fusion",
        "held_labels_used_for_epoch_selection": False,
        "anonymous_test_accessed": False,
        "history": history,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "held_indices.npy", held_indices)
    np.save(args.output_dir / "held_logits.npy", logits)
    torch.save(model.state_dict(), args.output_dir / "model.pt")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in metrics.items() if key != "history"}, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--mode", choices=("skeleton", "fusion"), required=True)
    result.add_argument("--depth-features", type=Path, required=True)
    result.add_argument("--ir-features", type=Path, required=True)
    result.add_argument("--skeleton-root", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--metadata", type=Path, default=RESULT_DIR / "metadata.npz")
    result.add_argument("--epochs", type=int, default=30)
    result.add_argument("--batch-size", type=int, default=32)
    result.add_argument("--workers", type=int, default=4)
    result.add_argument("--learning-rate", type=float, default=3e-4)
    result.add_argument("--weight-decay", type=float, default=0.02)
    result.add_argument("--label-smoothing", type=float, default=0.05)
    result.add_argument("--auxiliary-weight", type=float, default=0.15)
    result.add_argument("--sensor-dropout", type=float, default=0.1)
    result.add_argument("--skeleton-dropout", type=float, default=0.05)
    result.add_argument("--skeleton-noise", type=float, default=0.003)
    result.add_argument("--seed", type=int, default=2026)
    result.add_argument("--device", default="cuda:0")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
