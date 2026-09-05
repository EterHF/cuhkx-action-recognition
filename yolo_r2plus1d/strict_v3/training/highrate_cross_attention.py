#!/usr/bin/env python3
"""Extract strict fold features and train the high-rate cross-attention head."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from yolo_r2plus1d.strict_v3.models.highrate_cross_attention import (
    HighRateCrossAttention,
)
from yolo_r2plus1d.strict_v3.paths import RESULT_DIR
from yolo_r2plus1d.strict_v3.release.blend import apply_gate
from yolo_r2plus1d.strict_v3.training.public_finetune import VideoDataset, make_model
from yolo_r2plus1d.strict_v3.training.temporal import Residual

FOLDS = {
    "A": (1, 6, 17, 22),
    "B": (2, 7, 18, 23),
    "C": (3, 8, 19, 24),
    "D": (4, 9, 20),
    "E": (5, 16, 21),
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def extract(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    source = args.source_root
    checkpoint = torch.load(
        source / f"runs/fold_contract_head{args.fold}/best_fp16.pt",
        map_location="cpu",
        weights_only=True,
    )
    contract = json.loads(
        (
            source
            / "best_release/legal_strict_v3/preproc_affine"
            / f"contract_fold{args.fold}.json"
        ).read_text(encoding="utf-8")
    )
    mean = torch.tensor(contract["mean"], dtype=torch.float32).view(1, 4, 1, 1)
    std = torch.tensor(contract["std"], dtype=torch.float32).view(1, 4, 1, 1)
    count = len(np.load(args.visual_cache, mmap_mode="r"))
    dataset = VideoDataset(
        args.visual_cache,
        np.arange(count),
        None,
        augment=False,
        horizontal_flip=False,
        source_mean=mean,
        source_std=std,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    visual = make_model(input_adapter=False)
    visual.load_state_dict(checkpoint["model_state"], strict=True)
    visual.to(device).eval()
    values: list[np.ndarray] = []
    for batch_index, (frames, _) in enumerate(loader, 1):
        frames = frames.permute(0, 2, 1, 3, 4).contiguous().to(
            device, non_blocking=True
        )
        with torch.autocast(device.type, dtype=torch.float16):
            features = visual.encoder.layer2(
                visual.encoder.layer1(visual.encoder.stem(frames))
            )
            features = features.mean(dim=(-1, -2)).transpose(1, 2)
        values.append(features.float().cpu().numpy().astype(np.float16))
        if batch_index % 25 == 0 or batch_index == len(loader):
            print(f"visual_batches={batch_index}/{len(loader)}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / f"fold{args.fold}_visual.npy", np.concatenate(values))
    del visual
    torch.cuda.empty_cache()

    frame_logits = np.load(args.frame_logits, mmap_mode="r")
    temporal_checkpoint = torch.load(
        source
        / f"runs/public_cuhkx_dstformer/logit_residual_tcn_cv_{args.fold}/best_fp16.pt",
        map_location="cpu",
        weights_only=True,
    )
    temporal = Residual(kind="tcn")
    temporal.load_state_dict(temporal_checkpoint["model_state"], strict=True)
    temporal.to(device).eval()
    output = np.empty((len(frame_logits), 40), dtype=np.float32)
    for start in range(0, len(frame_logits), args.temporal_batch_size):
        stop = min(start + args.temporal_batch_size, len(frame_logits))
        batch = torch.from_numpy(np.array(frame_logits[start:stop], copy=True)).to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            output[start:stop] = temporal(batch).float().cpu().numpy()
    np.save(args.output_dir / f"fold{args.fold}_temporal_logits.npy", output)
    print(f"saved_fold={args.fold} rows={count}", flush=True)


def build_temporal_visual_baseline(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    contract = next(
        item for item in manifest["fold_contracts"] if item["fold"] == args.fold
    )
    visual = np.load(
        args.source_root
        / "best_release/legal_strict_v3/preproc_affine"
        / f"visual_fold{args.fold}_train.npy"
    )
    temporal = np.load(args.features / f"fold{args.fold}_temporal_logits.npy")
    weights = dict(contract["weights"])
    weights["fusion"] = 0.0
    logits, _ = apply_gate(
        {"fusion": np.zeros_like(visual), "visual": visual, "temporal": temporal},
        contract["temperatures"],
        weights,
    )
    np.save(args.features / f"fold{args.fold}_temporal_visual_logits.npy", logits)
    print(f"saved_temporal_visual_fold={args.fold} rows={len(logits)}")


class CachedDataset(Dataset):
    def __init__(
        self,
        indices: np.ndarray,
        labels: np.ndarray,
        visual: Path,
        temporal: Path,
        skeleton_root: Path,
    ) -> None:
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = labels
        self.visual = np.load(visual, mmap_mode="r")
        self.temporal = np.load(temporal, mmap_mode="r")
        self.skeleton = np.load(
            skeleton_root / "train_skeleton.npy", mmap_mode="r"
        )
        self.mask = np.load(
            skeleton_root / "train_skeleton_mask.npy", mmap_mode="r"
        )
        self.positions = np.load(
            skeleton_root / "train_positions.npy", mmap_mode="r"
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        return (
            torch.from_numpy(np.array(self.visual[index], copy=True)).float(),
            torch.from_numpy(np.array(self.skeleton[index], copy=True)),
            torch.from_numpy(np.array(self.mask[index], copy=True)),
            torch.from_numpy(np.array(self.positions[index], copy=True)),
            torch.from_numpy(np.array(self.temporal[index], copy=True)),
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


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device(args.device)
    with np.load(args.metadata) as metadata:
        labels = metadata["train_y"]
        users = metadata["train_users"]
    validation = np.isin(users, FOLDS[args.fold])
    train_indices = np.flatnonzero(~validation)
    validation_indices = np.flatnonzero(validation)
    visual = args.features / f"fold{args.fold}_visual.npy"
    baseline_path = args.features / f"fold{args.fold}_{args.baseline_suffix}.npy"
    train_loader = make_loader(
        CachedDataset(
            train_indices, labels, visual, baseline_path, args.skeleton_root
        ),
        args.batch_size,
        True,
        args.workers,
        args.seed,
    )
    validation_loader = make_loader(
        CachedDataset(
            validation_indices, labels, visual, baseline_path, args.skeleton_root
        ),
        args.batch_size * 2,
        False,
        args.workers,
        args.seed,
    )
    visual_dim = int(np.load(visual, mmap_mode="r").shape[-1])
    model = HighRateCrossAttention(
        visual_dim=visual_dim, residual_scale=args.residual_scale
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.02)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = correct = count = 0
        for batch in train_loader:
            *inputs, target = batch
            inputs = [value.to(device, non_blocking=True) for value in inputs]
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                logits = model(*inputs)
                loss = criterion(logits, target)
                if args.distill_weight:
                    teacher = inputs[-1].float()
                    loss = loss + args.distill_weight * F.kl_div(
                        F.log_softmax(logits.float(), dim=1),
                        F.softmax(teacher, dim=1),
                        reduction="batchmean",
                    )
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
        for batch in validation_loader:
            *inputs, _ = batch
            inputs = [value.to(device, non_blocking=True) for value in inputs]
            with torch.autocast(device.type, dtype=torch.bfloat16):
                outputs.append(model(*inputs).float().cpu().numpy())
    logits = np.concatenate(outputs)
    target = labels[validation_indices]
    baseline = np.load(baseline_path, mmap_mode="r")[validation_indices]
    metrics = {
        "fold": args.fold,
        "held_users": list(FOLDS[args.fold]),
        "rows": len(validation_indices),
        "seed": args.seed,
        "fixed_epoch": args.epochs,
        "baseline_suffix": args.baseline_suffix,
        "residual_scale": args.residual_scale,
        "distill_weight": args.distill_weight,
        "baseline_accuracy": float(np.mean(baseline.argmax(1) == target)),
        "candidate_accuracy": float(np.mean(logits.argmax(1) == target)),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "history": history,
        "held_labels_evaluated_once": True,
        "test_accessed": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "val_indices.npy", validation_indices)
    np.save(args.output_dir / "val_logits.npy", logits)
    torch.save(
        {
            "model_state": {
                name: value.detach().cpu().half()
                if value.is_floating_point()
                else value.cpu()
                for name, value in model.state_dict().items()
            },
            "fold": args.fold,
            "fixed_epoch": args.epochs,
            "baseline_suffix": args.baseline_suffix,
            "residual_scale": args.residual_scale,
            "distill_weight": args.distill_weight,
        },
        args.output_dir / "model_fp16.pt",
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in metrics.items() if key != "history"}, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    subparsers = root.add_subparsers(dest="command", required=True)
    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--fold", choices=FOLDS, required=True)
    extract_parser.add_argument("--source-root", type=Path, required=True)
    extract_parser.add_argument("--visual-cache", type=Path, required=True)
    extract_parser.add_argument("--frame-logits", type=Path, required=True)
    extract_parser.add_argument("--output-dir", type=Path, required=True)
    extract_parser.add_argument("--batch-size", type=int, default=16)
    extract_parser.add_argument("--temporal-batch-size", type=int, default=256)
    extract_parser.add_argument("--workers", type=int, default=8)
    extract_parser.add_argument("--device", default="cuda:0")
    extract_parser.set_defaults(function=extract)

    blend_parser = subparsers.add_parser("build-temporal-visual-baseline")
    blend_parser.add_argument("--fold", choices=FOLDS, required=True)
    blend_parser.add_argument("--source-root", type=Path, required=True)
    blend_parser.add_argument("--features", type=Path, required=True)
    blend_parser.add_argument(
        "--manifest", type=Path, default=RESULT_DIR / "release_manifest.json"
    )
    blend_parser.set_defaults(function=build_temporal_visual_baseline)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--fold", choices=FOLDS, required=True)
    train_parser.add_argument("--features", type=Path, required=True)
    train_parser.add_argument("--skeleton-root", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--metadata", type=Path, default=RESULT_DIR / "metadata.npz")
    train_parser.add_argument("--epochs", type=int, default=12)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--workers", type=int, default=4)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument("--baseline-suffix", default="temporal_logits")
    train_parser.add_argument("--residual-scale", type=float, default=1.0)
    train_parser.add_argument("--distill-weight", type=float, default=0.0)
    train_parser.add_argument("--seed", type=int, default=2026)
    train_parser.add_argument("--device", default="cuda:0")
    train_parser.set_defaults(function=train)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command in {"extract", "train"} and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.function(args)


if __name__ == "__main__":
    main()
