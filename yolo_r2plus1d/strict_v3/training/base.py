#!/usr/bin/env python3
"""Subject-wise training for the YOLO-cropped Depth+IR experiment."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from yolo_r2plus1d.strict_v3.data.preprocessing import KINETICS_MEAN, KINETICS_STD
from yolo_r2plus1d.strict_v3.models.muon import NDimMuon
from yolo_r2plus1d.strict_v3.models.r2plus1d import DepthIRR2Plus1D
from yolo_r2plus1d.strict_v3.paths import CHECKPOINT_DIR, REPO_ROOT, RESULT_DIR

MEAN = torch.tensor(KINETICS_MEAN, dtype=torch.float32).view(1, 4, 1, 1)
STD = torch.tensor(KINETICS_STD, dtype=torch.float32).view(1, 4, 1, 1)


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
        labels: np.ndarray,
        augment: bool,
        source_mean: torch.Tensor | None = None,
        source_std: torch.Tensor | None = None,
        normalize_after_affine: bool = False,
    ) -> None:
        self.cache_path = cache
        self.indices = indices.astype(np.int64)
        self.labels = labels
        self.augment = augment
        self.source_mean = source_mean
        self.source_std = source_std
        self.normalize_after_affine = bool(normalize_after_affine)
        self._frames: np.ndarray | None = None

    @property
    def frames(self) -> np.ndarray:
        if self._frames is None:
            self._frames = np.load(self.cache_path, mmap_mode="r")
        return self._frames

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int]:
        index = int(self.indices[item])
        frames = (
            torch.from_numpy(np.array(self.frames[index], copy=True))
            .float()
            .div_(255.0)
        )
        if self.augment and torch.rand(()) < 0.5:
            frames = frames.flip(-1)
        if self.augment:
            depth_gain = 0.9 + 0.2 * torch.rand(1)
            ir_gain = 0.9 + 0.2 * torch.rand(1)
            frames[:, :3].mul_(depth_gain).clamp_(0.0, 1.0)
            frames[:, 3:].mul_(ir_gain).clamp_(0.0, 1.0)
        if self.source_mean is None or self.source_std is None:
            frames = (frames - MEAN) / STD
        else:
            # Fit source_mean/source_std on training clips only.  The target
            # Kinetics affine keeps the public pretrained stem's scale while
            # applying the identical transform to validation and test clips.
            frames = ((frames - self.source_mean) / self.source_std) * STD + MEAN
            if self.normalize_after_affine:
                frames = (frames - MEAN) / STD
        return frames, int(self.labels[index])


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
        drop_last=shuffle,
    )


def parameter_groups(model: nn.Module, muon_lr: float, adamw_lr: float):
    muon_backbone, muon_head, adamw_backbone, adamw_head = [], [], [], []
    for name, parameter in model.named_parameters():
        head = name.startswith("network.fc")
        if parameter.ndim >= 2:
            (muon_head if head else muon_backbone).append(parameter)
        else:
            (adamw_head if head else adamw_backbone).append(parameter)
    muon = NDimMuon(
        [
            {"params": muon_backbone, "lr": muon_lr},
            {"params": muon_head, "lr": muon_lr * 5},
        ],
        lr=muon_lr,
        momentum=0.95,
        weight_decay=0.02,
    )
    adamw = torch.optim.AdamW(
        [
            {"params": adamw_backbone, "lr": adamw_lr},
            {"params": adamw_head, "lr": adamw_lr * 5},
        ],
        lr=adamw_lr,
        weight_decay=0.01,
    )
    return muon, adamw


def train_epoch(
    model, loader, optimizers, schedulers, criterion, device
) -> tuple[float, float]:
    model.train()
    total_loss = correct = count = 0
    for frames, labels in loader:
        frames, labels = (
            frames.to(device, non_blocking=True),
            labels.to(device, non_blocking=True),
        )
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(frames)
            loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        for optimizer, scheduler in zip(optimizers, schedulers, strict=True):
            optimizer.step()
            scheduler.step()
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        count += len(labels)
    return total_loss / count, correct / count


@torch.inference_mode()
def evaluate(model, loader, criterion, device) -> tuple[float, float, np.ndarray]:
    model.eval()
    total_loss = correct = count = 0
    predictions = []
    for frames, labels in loader:
        frames, labels = (
            frames.to(device, non_blocking=True),
            labels.to(device, non_blocking=True),
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(frames)
            loss = criterion(logits, labels)
        predictions.append(logits.float().cpu().numpy())
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        count += len(labels)
    return total_loss / count, correct / count, np.concatenate(predictions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache", type=Path, default=REPO_ROOT / ".cache/strict_v3/train_depth_ir.npy"
    )
    parser.add_argument("--metadata", type=Path, default=RESULT_DIR / "metadata.npz")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runs/visual")
    parser.add_argument(
        "--yolo-weights", type=Path, default=CHECKPOINT_DIR / "yolo11n.pt"
    )
    parser.add_argument("--val-users", type=int, nargs="+", default=[3, 8, 20, 24])
    parser.add_argument(
        "--full-train",
        action="store_true",
        help="fit every labelled clip and select the best training-epoch checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--muon-lr", type=float, default=2e-4)
    parser.add_argument("--adamw-lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--source-encoder",
        type=Path,
        help="external-only source checkpoint containing encoder_state",
    )
    parser.add_argument(
        "--select-last",
        action="store_true",
        help="use the predeclared final epoch instead of held-user checkpoint selection",
    )
    parser.add_argument(
        "--normalize-after-affine",
        action="store_true",
        help="apply Kinetics normalization after train-only moment matching",
    )
    parser.add_argument(
        "--mixstyle",
        action="store_true",
        help="mix layer-1 feature statistics during training only",
    )
    parser.add_argument(
        "--mixstyle-p",
        type=float,
        default=0.5,
        help="probability of applying MixStyle to a training batch",
    )
    parser.add_argument(
        "--mixstyle-alpha",
        type=float,
        default=0.1,
        help="Beta distribution parameter for MixStyle",
    )
    parser.add_argument(
        "--tail-average",
        type=int,
        default=0,
        help="average the final N fixed-epoch checkpoints after training",
    )
    parser.add_argument(
        "--preprocessing-contract",
        type=Path,
        default=REPO_ROOT / ".cache/strict_v3/preprocessing.json",
        help="frozen train-only affine statistics; no test statistics are read",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; launch with CUDA_VISIBLE_DEVICES=0")
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")
    with np.load(args.metadata) as data:
        labels, users = data["train_y"], data["train_users"]
    val_mask = (
        np.zeros_like(users, dtype=bool)
        if args.full_train
        else np.isin(users, args.val_users)
    )
    train_mask = ~val_mask
    if not args.full_train and set(users[train_mask]).intersection(users[val_mask]):
        raise RuntimeError("Subject leakage detected")
    train_indices, val_indices = np.flatnonzero(train_mask), np.flatnonzero(val_mask)
    val_classes = np.unique(labels[val_mask])
    if not args.full_train and len(val_classes) != 40:
        print(
            f"warning: validation covers only {len(val_classes)}/40 classes", flush=True
        )
    print(
        f"device={torch.cuda.get_device_name(0)} train_users={sorted(set(users[train_mask]))} "
        f"val_users={sorted(set(users[val_mask])) if not args.full_train else []} train={len(train_indices)} val={len(val_indices)} "
        f"val_classes={len(val_classes)}/40",
        flush=True,
    )
    source_mean = source_std = None
    contract_used = False
    if args.preprocessing_contract.exists():
        contract = json.loads(args.preprocessing_contract.read_text())
        if contract.get("source_split") != "train" or contract.get(
            "test_statistics_used", True
        ):
            raise RuntimeError("preprocessing contract must be fitted on train only")
        source_mean = torch.tensor(contract["mean"], dtype=torch.float32).view(
            1, 4, 1, 1
        )
        source_std = torch.tensor(contract["std"], dtype=torch.float32).view(1, 4, 1, 1)
        contract_used = True
    train_loader = make_loader(
        VideoDataset(
            args.cache,
            train_indices,
            labels,
            True,
            source_mean,
            source_std,
            args.normalize_after_affine,
        ),
        args.batch_size,
        True,
        args.workers,
        args.seed,
    )
    val_loader = (
        None
        if args.full_train
        else make_loader(
            VideoDataset(
                args.cache,
                val_indices,
                labels,
                False,
                source_mean,
                source_std,
                args.normalize_after_affine,
            ),
            args.batch_size * 2,
            False,
            args.workers,
            args.seed,
        )
    )
    model = DepthIRR2Plus1D(
        pretrained=True,
        mixstyle=args.mixstyle,
        mixstyle_p=args.mixstyle_p,
        mixstyle_alpha=args.mixstyle_alpha,
    ).to(device)
    if args.source_encoder is not None:
        payload = torch.load(args.source_encoder, map_location="cpu", weights_only=True)
        source = dict(payload["encoder_state"])
        stem = source["stem.0.weight"]
        if stem.shape[1] == 3:
            expanded = model.network.stem[0].weight.detach().cpu().clone()
            expanded[:, :3].copy_(stem)
            expanded[:, 3:4].copy_(stem.mean(dim=1, keepdim=True))
            source["stem.0.weight"] = expanded
        elif stem.shape[1] != 4:
            raise RuntimeError(
                f"unsupported source stem input channels: {stem.shape[1]}"
            )
        missing, unexpected = model.network.load_state_dict(source, strict=False)
        if set(missing) != {"fc.1.weight", "fc.1.bias"} or unexpected:
            raise RuntimeError(
                f"source mismatch missing={missing} unexpected={unexpected}"
            )
        print(
            json.dumps(
                {
                    "loaded_external_source": str(args.source_encoder.resolve()),
                    "source_epoch": payload.get("epoch"),
                    "target_head_reset": True,
                }
            ),
            flush=True,
        )
    optimizers = parameter_groups(model, args.muon_lr, args.adamw_lr)
    schedulers = tuple(
        torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[group["lr"] for group in optimizer.param_groups],
            epochs=args.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
        )
        for optimizer in optimizers
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.02)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_fp16.pt"
    best_accuracy, best_epoch = -1.0, 0
    history = []
    tail_states: list[dict[str, torch.Tensor]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_epoch(
            model, train_loader, optimizers, schedulers, criterion, device
        )
        if args.full_train:
            val_loss, val_accuracy, predictions = (
                float("nan"),
                train_accuracy,
                np.empty((0, 40), dtype=np.float32),
            )
        else:
            val_loss, val_accuracy, predictions = evaluate(
                model, val_loader, criterion, device
            )
        record = dict(
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
        )
        history.append(record)
        print(json.dumps(record), flush=True)
        if args.tail_average:
            if args.tail_average > args.epochs:
                raise ValueError("tail-average cannot exceed epochs")
            if epoch > args.epochs - args.tail_average:
                tail_states.append(
                    {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    }
                )
        selection_accuracy = train_accuracy if args.full_train else val_accuracy
        selected = (
            epoch == args.epochs
            if args.select_last
            else selection_accuracy > best_accuracy
        )
        if selected:
            best_accuracy = selection_accuracy
            best_epoch = epoch
            torch.save(
                {
                    "model_state": {
                        name: value.detach().cpu().half()
                        if value.is_floating_point()
                        else value.cpu()
                        for name, value in model.state_dict().items()
                    },
                    "epoch": epoch,
                    "validation_accuracy": None if args.full_train else val_accuracy,
                    "val_users": [] if args.full_train else args.val_users,
                },
                checkpoint_path,
            )
            np.save(args.output_dir / "best_val_logits.npy", predictions)
            np.save(args.output_dir / "best_val_predictions.npy", predictions.argmax(1))

    if args.tail_average:
        if len(tail_states) != args.tail_average:
            raise RuntimeError(
                f"captured {len(tail_states)} tail states, expected {args.tail_average}"
            )
        averaged_state: dict[str, torch.Tensor] = {}
        for name in tail_states[0]:
            values = [state[name] for state in tail_states]
            if values[0].is_floating_point():
                averaged_state[name] = torch.stack(values, dim=0).mean(dim=0)
            else:
                # Integer buffers such as num_batches_tracked have no useful
                # arithmetic mean; retain the last fixed-epoch value.
                averaged_state[name] = values[-1]
        model.load_state_dict(averaged_state, strict=True)
        if args.full_train:
            averaged_loss, averaged_accuracy = float("nan"), train_accuracy
            averaged_predictions = np.empty((0, 40), dtype=np.float32)
        else:
            averaged_loss, averaged_accuracy, averaged_predictions = evaluate(
                model, val_loader, criterion, device
            )
        best_accuracy = averaged_accuracy
        best_epoch = args.epochs - args.tail_average + 1
        torch.save(
            {
                "model_state": {
                    name: value.detach().cpu().half()
                    if value.is_floating_point()
                    else value.cpu()
                    for name, value in model.state_dict().items()
                },
                "epoch": best_epoch,
                "validation_accuracy": None if args.full_train else averaged_accuracy,
                "val_users": [] if args.full_train else args.val_users,
                "checkpoint_selection": f"fixed_tail_average_{args.tail_average}",
            },
            checkpoint_path,
        )
        np.save(args.output_dir / "best_val_logits.npy", averaged_predictions)
        np.save(
            args.output_dir / "best_val_predictions.npy", averaged_predictions.argmax(1)
        )

    yolo_bytes = args.yolo_weights.stat().st_size
    checkpoint_bytes = checkpoint_path.stat().st_size
    metrics = {
        "best_validation_accuracy": None if args.full_train else best_accuracy,
        "best_training_accuracy": best_accuracy if args.full_train else None,
        "best_epoch": best_epoch,
        "validation_users": [] if args.full_train else args.val_users,
        "train_clips": len(train_indices),
        "validation_clips": len(val_indices),
        "validation_class_coverage": len(val_classes),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "classifier_checkpoint_bytes": checkpoint_bytes,
        "yolo_checkpoint_bytes": yolo_bytes,
        "combined_checkpoint_bytes": checkpoint_bytes + yolo_bytes,
        "under_100mb": checkpoint_bytes + yolo_bytes <= 100_000_000,
        "preprocessing_contract": str(args.preprocessing_contract),
        "train_only_preprocessing": contract_used,
        "normalize_after_affine": args.normalize_after_affine,
        "source_encoder": None
        if args.source_encoder is None
        else str(args.source_encoder),
        "target_head_reset": True,
        "checkpoint_selection": "fixed_final_epoch"
        if args.select_last
        else "best_accuracy",
        "mixstyle": args.mixstyle,
        "mixstyle_p": args.mixstyle_p,
        "mixstyle_alpha": args.mixstyle_alpha,
        "tail_average": args.tail_average,
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in metrics.items() if key != "history"}, indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
