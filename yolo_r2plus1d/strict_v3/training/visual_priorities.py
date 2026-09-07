#!/usr/bin/env python3
"""Run fixed Fold-A screens for the ordered Visual architecture priorities."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from yolo_r2plus1d.strict_v3.models.visual_priorities import (
    VARIANTS,
    VisualPriorityClassifier,
)
from yolo_r2plus1d.strict_v3.paths import RESULT_DIR
from yolo_r2plus1d.strict_v3.training.base import seed_everything
from yolo_r2plus1d.strict_v3.training.public_finetune import VideoDataset

HELD_USERS = (1, 6, 17, 22)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_public_encoder(model: VisualPriorityClassifier, path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("target_data_used", True):
        raise RuntimeError("source encoder must not use target data")
    state = payload.get("encoder_state")
    if not isinstance(state, Mapping):
        raise RuntimeError("source checkpoint does not contain encoder_state")
    state = dict(state)
    stem = state["stem.0.weight"]
    if stem.shape[1] == 3:
        expanded = model.encoder.stem[0].weight.detach().clone()
        expanded[:, :3].copy_(stem)
        expanded[:, 3:].copy_(stem.mean(dim=1, keepdim=True))
        state["stem.0.weight"] = expanded
    elif stem.shape[1] != 4:
        raise RuntimeError(f"unsupported source stem channels: {stem.shape[1]}")
    missing, unexpected = model.encoder.load_state_dict(state, strict=False)
    if set(missing) != set() or unexpected:
        raise RuntimeError(f"encoder state mismatch: missing={missing}, unexpected={unexpected}")
    return {
        "source_dataset": payload.get("source_dataset"),
        "initialization": payload.get("initialization"),
        "sha256": sha256(path),
    }


def set_trainable(model: VisualPriorityClassifier) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.encoder.layer4.parameters():
        parameter.requires_grad_(True)
    head_modules = [model.base_classifier, model.modality_gate]
    if model.temporal_head is not None:
        head_modules.append(model.temporal_head)
    for module in head_modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    encoder = [p for p in model.encoder.layer4.parameters() if p.requires_grad]
    heads = [
        p
        for module in head_modules
        for p in module.parameters()
        if p.requires_grad
    ]
    return encoder, heads


def make_loader(
    dataset: VideoDataset, batch_size: int, workers: int, shuffle: bool, seed: int
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


@torch.inference_mode()
def evaluate(
    model: VisualPriorityClassifier, loader: DataLoader, device: torch.device
) -> np.ndarray:
    model.eval()
    outputs = []
    for frames, _ in loader:
        with torch.autocast(device.type, dtype=torch.bfloat16):
            outputs.append(model(frames.to(device, non_blocking=True)).float().cpu().numpy())
    return np.concatenate(outputs)


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device(args.device)
    with np.load(args.metadata) as metadata:
        labels = metadata["train_y"].astype(np.int64)
        users = metadata["train_users"].astype(np.int64)
    held = np.isin(users, args.held_users)
    train_indices = np.flatnonzero(~held)
    held_indices = np.flatnonzero(held)
    if set(users[train_indices]).intersection(set(users[held_indices])):
        raise RuntimeError("subject leakage")
    contract = json.loads(args.preprocessing_contract.read_text(encoding="utf-8"))
    if contract.get("source_split") != "train" or contract.get("test_statistics_used", True):
        raise RuntimeError("preprocessing contract is not train-only")
    source_mean = torch.tensor(contract["mean"], dtype=torch.float32).view(1, 4, 1, 1)
    source_std = torch.tensor(contract["std"], dtype=torch.float32).view(1, 4, 1, 1)
    train_dataset = VideoDataset(
        args.cache,
        train_indices,
        labels,
        True,
        True,
        "all",
        source_mean,
        source_std,
        True,
    )
    held_dataset = VideoDataset(
        args.cache,
        held_indices,
        labels,
        False,
        False,
        "all",
        source_mean,
        source_std,
        True,
    )
    train_loader = make_loader(
        train_dataset, args.batch_size, args.workers, True, args.seed
    )
    held_loader = make_loader(
        held_dataset, args.batch_size * 2, args.workers, False, args.seed
    )
    model = VisualPriorityClassifier(args.variant)
    source = load_public_encoder(model, args.source_encoder)
    encoder_parameters, head_parameters = set_trainable(model)
    model.to(device)
    optimizer = torch.optim.AdamW(
        (
            {"params": encoder_parameters, "lr": args.encoder_lr},
            {"params": head_parameters, "lr": args.head_lr},
        ),
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for module in model.encoder.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        correct = count = 0
        for frames, target in train_loader:
            frames = frames.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                logits = model(frames)
                loss = criterion(logits, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            correct += int((logits.argmax(1) == target).sum())
            count += len(target)
        scheduler.step()
        record = {"epoch": epoch, "train_accuracy": correct / count}
        history.append(record)
        print(json.dumps(record), flush=True)

    held_logits = evaluate(model, held_loader, device)
    prediction = held_logits.argmax(1)
    held_labels = labels[held_indices]
    held_users = users[held_indices]
    user_accuracy = {
        str(int(user)): float(np.mean(prediction[held_users == user] == held_labels[held_users == user]))
        for user in np.unique(held_users)
    }
    metrics = {
        "protocol": "visual-architecture-priorities-foldA/v1",
        "variant": args.variant,
        "held_users": list(args.held_users),
        "train_rows": len(train_indices),
        "held_rows": len(held_indices),
        "accuracy": float(np.mean(prediction == held_labels)),
        "worst_user_accuracy": min(user_accuracy.values()),
        "user_accuracy": user_accuracy,
        "fixed_epoch": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "public_encoder": source,
        "project_checkpoint_loaded": False,
        "held_labels_used_for_checkpoint_selection": False,
        "anonymous_test_accessed": False,
        "history": history,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "held_indices.npy", held_indices)
    np.save(args.output_dir / "held_logits.npy", held_logits)
    torch.save(model.state_dict(), args.output_dir / "model.pt")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in metrics.items() if key != "history"}, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--variant", choices=VARIANTS, required=True)
    result.add_argument("--cache", type=Path, required=True)
    result.add_argument("--source-encoder", type=Path, required=True)
    result.add_argument("--preprocessing-contract", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--metadata", type=Path, default=RESULT_DIR / "metadata.npz")
    result.add_argument("--held-users", type=int, nargs="+", default=list(HELD_USERS))
    result.add_argument("--epochs", type=int, default=15)
    result.add_argument("--batch-size", type=int, default=4)
    result.add_argument("--workers", type=int, default=4)
    result.add_argument("--encoder-lr", type=float, default=2e-5)
    result.add_argument("--head-lr", type=float, default=1e-4)
    result.add_argument("--weight-decay", type=float, default=0.01)
    result.add_argument("--label-smoothing", type=float, default=0.02)
    result.add_argument("--seed", type=int, default=2026)
    result.add_argument("--device", default="cuda:0")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
