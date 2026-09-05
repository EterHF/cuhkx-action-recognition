#!/usr/bin/env python3
"""Train and aggregate a fail-closed, leakage-free skeleton research OOF."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from yolo_r2plus1d.strict_v3.models.research_skeleton import HighRateSkeletonClassifier
from yolo_r2plus1d.strict_v3.paths import REPO_ROOT, RESULT_DIR

FOLDS = {
    "A": (1, 6, 17, 22),
    "B": (2, 7, 18, 23),
    "C": (3, 8, 19, 24),
    "D": (4, 9, 20),
    "E": (5, 16, 21),
}
LEFT_RIGHT = torch.tensor((0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SkeletonDataset(Dataset):
    def __init__(self, root: Path, indices: np.ndarray, labels: np.ndarray, augment: bool) -> None:
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = labels
        self.augment = augment
        self.skeleton = np.load(root / "train_skeleton.npy", mmap_mode="r")
        self.mask = np.load(root / "train_skeleton_mask.npy", mmap_mode="r")
        self.positions = np.load(root / "train_positions.npy", mmap_mode="r")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        skeleton = torch.from_numpy(np.array(self.skeleton[index], copy=True))
        mask = torch.from_numpy(np.array(self.mask[index], copy=True))
        positions = torch.from_numpy(np.array(self.positions[index], copy=True))
        if self.augment:
            if torch.rand(()) < 0.5:
                skeleton = skeleton[:, LEFT_RIGHT]
                skeleton[..., 0].neg_()
            angle = (torch.rand(()) - 0.5) * (torch.pi / 9.0)
            cosine, sine = angle.cos(), angle.sin()
            x, y = skeleton[..., 0].clone(), skeleton[..., 1].clone()
            skeleton[..., 0] = cosine * x - sine * y
            skeleton[..., 1] = sine * x + cosine * y
            skeleton[mask] += torch.randn_like(skeleton[mask]) * 0.004
        return skeleton, mask, positions, int(self.labels[index])


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


def split(metadata: Path, fold: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(metadata) as values:
        labels = values["train_y"]
        users = values["train_users"]
    held = np.isin(users, FOLDS[fold])
    train_indices, held_indices = np.flatnonzero(~held), np.flatnonzero(held)
    if np.intersect1d(users[train_indices], users[held_indices]).size:
        raise RuntimeError("subject leakage detected")
    if len(train_indices) + len(held_indices) != len(labels):
        raise RuntimeError("incomplete outer split")
    return labels, users, train_indices, held_indices


def train_fold(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    labels, users, train_indices, held_indices = split(args.metadata, args.fold)
    train_loader = make_loader(
        SkeletonDataset(args.skeleton_root, train_indices, labels, True),
        args.batch_size,
        True,
        args.workers,
        args.seed,
    )
    held_loader = make_loader(
        SkeletonDataset(args.skeleton_root, held_indices, labels, False),
        args.batch_size * 2,
        False,
        args.workers,
        args.seed,
    )
    device = torch.device(args.device)
    model = HighRateSkeletonClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = correct = count = 0
        for skeleton, mask, positions, target in train_loader:
            skeleton = skeleton.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            positions = positions.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                logits = model(skeleton, mask, positions)
                loss = criterion(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(target)
            correct += int((logits.argmax(1) == target).sum().item())
            count += len(target)
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": total_loss / count,
            "train_accuracy": correct / count,
        }
        history.append(record)
        print(json.dumps(record), flush=True)

    model.eval()
    outputs = []
    with torch.inference_mode():
        for skeleton, mask, positions, _ in held_loader:
            with torch.autocast(device.type, dtype=torch.bfloat16):
                logits = model(
                    skeleton.to(device, non_blocking=True),
                    mask.to(device, non_blocking=True),
                    positions.to(device, non_blocking=True),
                )
            outputs.append(logits.float().cpu().numpy())
    held_logits = np.concatenate(outputs)
    output = args.output_dir / f"fold{args.fold}"
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "held_indices.npy", held_indices)
    np.save(output / "held_logits.npy", held_logits)
    torch.save(
        {
            "model_state": {
                name: value.detach().cpu().half() if value.is_floating_point() else value.cpu()
                for name, value in model.state_dict().items()
            },
            "fold": args.fold,
            "seed": args.seed,
            "fixed_epoch": args.epochs,
        },
        output / "model_fp16.pt",
    )
    metrics = {
        "protocol": "fully-leakage-free-subject-oof/v1",
        "fold": args.fold,
        "held_users": list(FOLDS[args.fold]),
        "train_rows": len(train_indices),
        "held_rows": len(held_indices),
        "train_subjects": sorted(map(int, np.unique(users[train_indices]))),
        "held_subjects": sorted(map(int, np.unique(users[held_indices]))),
        "seed": args.seed,
        "fixed_epoch": args.epochs,
        "held_accuracy": float(np.mean(held_logits.argmax(1) == labels[held_indices])),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "initialization": "random; no checkpoint loaded",
        "held_labels_used_for_selection": False,
        "anonymous_test_accessed": False,
        "history": history,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in metrics.items() if key != "history"}, indent=2))


def aggregate(args: argparse.Namespace) -> None:
    with np.load(args.metadata) as values:
        labels = values["train_y"]
        users = values["train_users"]
    logits = np.full((len(labels), 40), np.nan, dtype=np.float32)
    fold_metrics = []
    for fold in FOLDS:
        root = args.output_dir / f"fold{fold}"
        indices = np.load(root / "held_indices.npy")
        values = np.load(root / "held_logits.npy")
        if np.isfinite(logits[indices]).any():
            raise RuntimeError(f"duplicate OOF rows in fold {fold}")
        logits[indices] = values
        fold_metrics.append(json.loads((root / "metrics.json").read_text(encoding="utf-8")))
    if not np.isfinite(logits).all():
        raise RuntimeError("OOF contains missing or non-finite rows")
    predictions = logits.argmax(1)
    user_accuracy = {
        str(int(user)): float(np.mean(predictions[users == user] == labels[users == user]))
        for user in np.unique(users)
    }
    class_accuracy = [
        float(np.mean(predictions[labels == target] == target)) for target in range(40)
    ]
    skeleton_files = [
        args.skeleton_root / name
        for name in (
            "train_skeleton.npy",
            "train_skeleton_mask.npy",
            "train_positions.npy",
            "train_lengths.npy",
        )
    ]
    summary = {
        "protocol": "fully-leakage-free-subject-oof/v1",
        "rows": len(labels),
        "correct": int(np.sum(predictions == labels)),
        "accuracy": float(np.mean(predictions == labels)),
        "macro_accuracy": float(np.mean(class_accuracy)),
        "subject_macro_accuracy": float(np.mean(list(user_accuracy.values()))),
        "worst_user_accuracy": float(min(user_accuracy.values())),
        "folds": [
            {key: value for key, value in fold.items() if key != "history"} for fold in fold_metrics
        ],
        "user_accuracy": user_accuracy,
        "input_sha256": {str(path): sha256_file(path) for path in [args.metadata, *skeleton_files]},
        "initialization": "random; no target or external pretrained checkpoint",
        "selection": "fixed recipe and final epoch; held labels evaluated once per fold",
        "anonymous_test_accessed": False,
    }
    np.save(args.output_dir / "oof_logits.npy", logits)
    np.save(args.output_dir / "oof_predictions.npy", predictions)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    subparsers = root.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--metadata", type=Path, default=RESULT_DIR / "metadata.npz")
    common.add_argument(
        "--skeleton-root", type=Path, default=REPO_ROOT / ".cache/highrate_cross_attention/skeleton"
    )
    common.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runs/research_oof_v1")
    train = subparsers.add_parser("train-fold", parents=[common])
    train.add_argument("--fold", choices=FOLDS, required=True)
    train.add_argument("--epochs", type=int, default=40)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--workers", type=int, default=4)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--seed", type=int, default=2026)
    train.add_argument("--device", default="cuda:0")
    train.set_defaults(function=train_fold)
    aggregate_parser = subparsers.add_parser("aggregate", parents=[common])
    aggregate_parser.set_defaults(function=aggregate)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "train-fold" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training")
    args.function(args)


if __name__ == "__main__":
    main()
