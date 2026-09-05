#!/usr/bin/env python3
"""Fine-tune one Omnivore trunk on aligned IR appearance and pseudo-depth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from yolo_r2plus1d.strict_v3.models.omnivore_rgbd import OmnivoreRGBDClassifier
from yolo_r2plus1d.strict_v3.paths import RESULT_DIR
from yolo_r2plus1d.strict_v3.training.finetune_public_omnivore import RawSubset
from yolo_r2plus1d.strict_v3.training.public_backbone_screen import (
    HELD_USERS,
    depth_color_to_inverse,
    load_omnivore,
    seed_everything,
    sha256_file,
)


def make_loader(
    cache: Path,
    indices: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    return DataLoader(
        RawSubset(cache, indices, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )


def raw_rgbd(frames: torch.Tensor) -> torch.Tensor:
    """Build [IR, IR, IR, inverse-depth] without changing temporal alignment."""
    infrared = frames[:, :, 3:4].permute(0, 2, 1, 3, 4).float().div_(255.0)
    depth = depth_color_to_inverse(frames[:, :, :3])[:, None]
    return torch.cat((infrared.expand(-1, 3, -1, -1, -1), depth), dim=1)


@torch.inference_mode()
def compute_stats(loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    total = torch.zeros(2, dtype=torch.float64, device=device)
    squared = torch.zeros_like(total)
    count = 0
    for frames, _ in loader:
        values = raw_rgbd(frames.to(device, non_blocking=True))[:, (0, 3)].double()
        total += values.sum(dim=(0, 2, 3, 4))
        squared += values.square().sum(dim=(0, 2, 3, 4))
        count += values.shape[0] * values.shape[2] * values.shape[3] * values.shape[4]
    mean = total / count
    variance = squared / count - mean.square()
    std = variance.clamp_min(1e-8).sqrt()
    return mean.float(), std.float()


def prepare_rgbd(
    frames: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, augment: bool
) -> torch.Tensor:
    values = raw_rgbd(frames)
    if augment:
        if torch.rand((), device=values.device) < 0.5:
            values = values.flip(-1)
        gain = torch.empty((len(values), 2, 1, 1, 1), device=values.device).uniform_(0.9, 1.1)
        offset = torch.empty((len(values), 2, 1, 1, 1), device=values.device).uniform_(
            -0.03, 0.03
        )
        values[:, :3] = (values[:, :3] * gain[:, :1] + offset[:, :1]).clamp_(0.0, 1.0)
        values[:, 3:] = (values[:, 3:] * gain[:, 1:] + offset[:, 1:]).clamp_(0.0, 1.0)
    channel_mean = mean[[0, 0, 0, 1]].view(1, 4, 1, 1, 1).to(values.device)
    channel_std = std[[0, 0, 0, 1]].view(1, 4, 1, 1, 1).to(values.device)
    return (values - channel_mean) / channel_std


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> np.ndarray:
    model.eval()
    outputs = []
    for frames, _ in loader:
        inputs = prepare_rgbd(frames.to(device, non_blocking=True), mean, std, False)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            outputs.append(model(inputs).float().cpu().numpy())
    return np.concatenate(outputs)


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device(args.device)
    with np.load(args.metadata) as metadata:
        labels = metadata["train_y"]
        users = metadata["train_users"]
    held_users = tuple(args.held_users)
    held = np.isin(users, held_users)
    train_indices = np.flatnonzero(~held)
    held_indices = np.flatnonzero(held)
    if np.intersect1d(users[train_indices], users[held_indices]).size:
        raise RuntimeError("subject leakage")

    stats_loader = make_loader(
        args.cache, train_indices, labels, args.batch_size, args.workers, False, args.seed
    )
    mean, std = compute_stats(stats_loader, device)
    train_loader = make_loader(
        args.cache, train_indices, labels, args.batch_size, args.workers, True, args.seed
    )
    held_loader = make_loader(
        args.cache, held_indices, labels, args.batch_size, args.workers, False, args.seed
    )
    model = OmnivoreRGBDClassifier(load_omnivore(args)).to(device)
    optimizer = torch.optim.AdamW(
        (
            {"params": model.trunk.parameters(), "lr": args.encoder_lr},
            {"params": model.classifier.parameters(), "lr": args.head_lr},
        ),
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        correct = count = 0
        for frames, target in train_loader:
            frames = frames.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            inputs = prepare_rgbd(frames, mean, std, True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                logits = model(inputs)
                loss = criterion(logits, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            correct += int((logits.argmax(1) == target).sum())
            count += len(target)
        scheduler.step()
        record = {"epoch": epoch, "train_accuracy": correct / count}
        history.append(record)
        print(json.dumps(record), flush=True)

    logits = evaluate(model, held_loader, device, mean, std)
    prediction = logits.argmax(1)
    held_labels = labels[held_indices]
    held_user_values = users[held_indices]
    user_accuracy = {
        str(int(user)): float(
            np.mean(prediction[held_user_values == user] == held_labels[held_user_values == user])
        )
        for user in np.unique(held_user_values)
    }
    metrics = {
        "protocol": "subject-held-single-omnivore-native-rgbd/v1",
        "input_channels": ["ir", "ir", "ir", "inverse_depth"],
        "normalization_fit": "outer-train-only",
        "sensor_mean": mean.tolist(),
        "sensor_std": std.tolist(),
        "held_users": list(held_users),
        "train_rows": len(train_indices),
        "held_rows": len(held_indices),
        "accuracy": float(np.mean(prediction == held_labels)),
        "worst_user_accuracy": min(user_accuracy.values()),
        "user_accuracy": user_accuracy,
        "fixed_epoch": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "public_source_commit": args.source_commit,
        "public_weights_sha256": sha256_file(args.weights),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "project_checkpoint_loaded": False,
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
    result.add_argument("--cache", type=Path, required=True)
    result.add_argument("--weights", type=Path, required=True)
    result.add_argument("--source-dir", type=Path, required=True)
    result.add_argument("--source-commit", required=True)
    result.add_argument("--model", default="omnivore_swinT")
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--metadata", type=Path, default=RESULT_DIR / "metadata.npz")
    result.add_argument("--held-users", type=int, nargs="+", default=list(HELD_USERS))
    result.add_argument("--epochs", type=int, default=15)
    result.add_argument("--batch-size", type=int, default=16)
    result.add_argument("--workers", type=int, default=4)
    result.add_argument("--encoder-lr", type=float, default=1e-5)
    result.add_argument("--head-lr", type=float, default=3e-4)
    result.add_argument("--weight-decay", type=float, default=0.05)
    result.add_argument("--label-smoothing", type=float, default=0.05)
    result.add_argument("--seed", type=int, default=2026)
    result.add_argument("--device", default="cuda:0")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
