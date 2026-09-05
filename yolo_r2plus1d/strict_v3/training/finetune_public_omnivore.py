#!/usr/bin/env python3
"""Fixed-budget fine-tuning of the public Omnivore encoder on one subject fold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from yolo_r2plus1d.strict_v3.paths import RESULT_DIR
from yolo_r2plus1d.strict_v3.training.public_backbone_screen import (
    HELD_USERS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    depth_color_to_inverse,
    load_omnivore,
    seed_everything,
    sha256_file,
)


class RawSubset(Dataset):
    def __init__(self, cache: Path, indices: np.ndarray, labels: np.ndarray) -> None:
        self.frames = np.load(cache, mmap_mode="r")
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = labels

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        return torch.from_numpy(np.array(self.frames[index], copy=True)), int(self.labels[index])


def prepare_inputs(frames: torch.Tensor, modality: str, augment: bool) -> torch.Tensor:
    if modality == "depth":
        inverse = depth_color_to_inverse(frames[:, :, :3])
        inputs = inverse[:, None].expand(-1, 3, -1, -1, -1).float()
    elif modality == "ir":
        inputs = frames[:, :, 3:4].permute(0, 2, 1, 3, 4).expand(-1, 3, -1, -1, -1)
        inputs = inputs.float().div_(255.0)
    else:
        raise ValueError(f"unsupported modality: {modality}")
    if augment:
        scale = torch.empty((len(inputs), 1, 1, 1, 1), device=inputs.device).uniform_(0.9, 1.1)
        offset = torch.empty((len(inputs), 1, 1, 1, 1), device=inputs.device).uniform_(-0.03, 0.03)
        inputs = (inputs * scale + offset).clamp_(0.0, 1.0)
    return (inputs - IMAGENET_MEAN.to(inputs.device)) / IMAGENET_STD.to(inputs.device)


class OmnivoreClassifier(nn.Module):
    def __init__(self, trunk: nn.Module) -> None:
        super().__init__()
        self.trunk = trunk
        self.head = nn.Sequential(nn.LayerNorm(768), nn.Dropout(0.2), nn.Linear(768, 40))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(inputs))


def validate_checkpoint_lineage(
    metrics: dict, modality: str, public_weights_sha256: str
) -> None:
    expected = {
        "model": "omnivore_swinT",
        "modality": modality,
        "public_weights_sha256": public_weights_sha256,
        "project_checkpoint_loaded": False,
    }
    observed = {key: metrics.get(key) for key in expected}
    if observed != expected:
        raise RuntimeError(f"invalid fine-tune checkpoint lineage: {observed}")


@torch.inference_mode()
def extract_features(args: argparse.Namespace) -> None:
    public_hash = sha256_file(args.weights)
    metrics_path = args.checkpoint.parent / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    validate_checkpoint_lineage(metrics, args.modality, public_hash)
    device = torch.device(args.device)
    model = OmnivoreClassifier(load_omnivore(args))
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    with np.load(args.metadata) as metadata:
        labels = metadata["train_y"]
    data_loader = DataLoader(
        RawSubset(args.cache, np.arange(len(labels)), labels),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    values = []
    for batch_index, (frames, _) in enumerate(data_loader, 1):
        inputs = prepare_inputs(frames.to(device, non_blocking=True), args.modality, False)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            if args.feature_layout == "temporal":
                features = model.trunk(inputs, out_feat_keys=["stage3"])[0]
                features = features.mean(dim=(-2, -1)).permute(0, 2, 1)
            else:
                features = model.trunk(inputs)[:, None]
        values.append(features.float().cpu().numpy().astype(np.float16))
        if batch_index % 20 == 0:
            print(f"batches={batch_index}", flush=True)
    output = np.concatenate(values)
    args.feature_output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.feature_output, output)
    manifest = {
        "model": args.model,
        "modality": args.modality,
        "shape": output.shape,
        "feature_layout": args.feature_layout,
        "public_source_commit": args.source_commit,
        "public_weights_sha256": public_hash,
        "fine_tune_checkpoint_sha256": sha256_file(args.checkpoint),
        "fine_tune_protocol": metrics["protocol"],
        "old_project_checkpoint_loaded": False,
        "anonymous_test_accessed": False,
    }
    args.feature_output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device(args.device)
    with np.load(args.metadata) as metadata:
        labels = metadata["train_y"]
        users = metadata["train_users"]
    held_users = tuple(args.held_users)
    held_mask = np.isin(users, held_users)
    held_indices = np.flatnonzero(held_mask)
    train_indices = np.arange(len(labels)) if args.full_fit else np.flatnonzero(~held_mask)
    evaluation_indices = train_indices if args.full_fit else held_indices
    if not args.full_fit and np.intersect1d(users[train_indices], users[held_indices]).size:
        raise RuntimeError("subject leakage")

    trunk = load_omnivore(args)
    model = OmnivoreClassifier(trunk).to(device)
    optimizer = torch.optim.AdamW(
        (
            {"params": model.trunk.parameters(), "lr": args.encoder_lr},
            {"params": model.head.parameters(), "lr": args.head_lr},
        ),
        weight_decay=0.05,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    train_loader = DataLoader(
        RawSubset(args.cache, train_indices, labels),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    evaluation_loader = DataLoader(
        RawSubset(args.cache, evaluation_indices, labels),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        correct = count = 0
        for frames, target in train_loader:
            frames = frames.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            inputs = prepare_inputs(frames, args.modality, augment=True)
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
        accuracy = correct / count
        history.append({"epoch": epoch, "train_accuracy": accuracy})
        print(f"epoch={epoch} train_accuracy={accuracy:.6f}", flush=True)

    model.eval()
    evaluation_logits = []
    with torch.inference_mode():
        for frames, _ in evaluation_loader:
            inputs = prepare_inputs(frames.to(device, non_blocking=True), args.modality, False)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                evaluation_logits.append(model(inputs).float().cpu().numpy())
    evaluation_logits = np.concatenate(evaluation_logits)
    prediction = evaluation_logits.argmax(1)
    evaluation_labels = labels[evaluation_indices]
    evaluation_users = users[evaluation_indices]
    user_accuracy = {
        str(int(user)): float(
            np.mean(
                prediction[evaluation_users == user]
                == evaluation_labels[evaluation_users == user]
            )
        )
        for user in np.unique(evaluation_users)
    }
    metrics = {
        "protocol": (
            "full-fit-public-encoder/v1"
            if args.full_fit
            else "single-subject-fold-public-encoder-finetune/v1"
        ),
        "model": args.model,
        "modality": args.modality,
        "evaluation_split": "train-in-sample" if args.full_fit else "held-subjects",
        "held_users": [] if args.full_fit else list(held_users),
        "train_rows": len(train_indices),
        "held_rows": 0 if args.full_fit else len(held_indices),
        "evaluated_rows": len(evaluation_indices),
        "accuracy": float(np.mean(prediction == evaluation_labels)),
        "worst_user_accuracy": min(user_accuracy.values()),
        "user_accuracy": user_accuracy,
        "fixed_epoch": args.epochs,
        "seed": args.seed,
        "public_source_commit": args.source_commit,
        "public_weights_sha256": sha256_file(args.weights),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "project_checkpoint_loaded": False,
        "held_labels_used_for_selection": False,
        "anonymous_test_accessed": False,
        "history": history,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "train" if args.full_fit else "held"
    np.save(args.output_dir / f"{prefix}_indices.npy", evaluation_indices)
    np.save(args.output_dir / f"{prefix}_logits.npy", evaluation_logits)
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
    result.add_argument("--modality", choices=("depth", "ir"), required=True)
    result.add_argument("--metadata", type=Path, default=RESULT_DIR / "metadata.npz")
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--epochs", type=int, default=15)
    result.add_argument("--encoder-lr", type=float, default=1e-5)
    result.add_argument("--head-lr", type=float, default=3e-4)
    result.add_argument("--batch-size", type=int, default=8)
    result.add_argument("--workers", type=int, default=4)
    result.add_argument("--seed", type=int, default=2026)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--full-fit", action="store_true")
    result.add_argument("--held-users", type=int, nargs="+", default=list(HELD_USERS))
    result.add_argument("--checkpoint", type=Path)
    result.add_argument("--feature-output", type=Path)
    result.add_argument("--feature-layout", choices=("clip", "temporal"), default="clip")
    return result


if __name__ == "__main__":
    parsed = parser().parse_args()
    if (parsed.checkpoint is None) != (parsed.feature_output is None):
        raise ValueError("--checkpoint and --feature-output must be provided together")
    if parsed.checkpoint is None:
        run(parsed)
    else:
        extract_features(parsed)
