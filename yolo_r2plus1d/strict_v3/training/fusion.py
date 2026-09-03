#!/usr/bin/env python3
"""Train R(2+1)D + ST-GCN++ temporal fusion with subject-wise validation."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from yolo_r2plus1d.strict_v3.models.fusion import TemporalFusionClassifier
from yolo_r2plus1d.strict_v3.models.muon import NDimMuon
from yolo_r2plus1d.strict_v3.paths import CHECKPOINT_DIR, REPO_ROOT, RESULT_DIR
from yolo_r2plus1d.strict_v3.training.base import MEAN, STD

THERMAL_MEAN = 0.39306015
THERMAL_STD = 0.21199141


LEFT_RIGHT = torch.tensor((0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FusionDataset(Dataset):
    def __init__(
        self,
        visual_cache: Path,
        skeleton_cache: Path,
        skeleton_mask: Path,
        indices: np.ndarray,
        labels: np.ndarray,
        augment: bool,
        temporal_crop: int | None = None,
        skeleton_motion: bool = False,
        thermal_cache: Path | None = None,
        gain_low: float = 0.9,
        gain_high: float = 1.1,
        return_index: bool = False,
        temporal_stride: int = 1,
        skeleton_acceleration: bool = False,
        skeleton_noise: float = 0.005,
        source_mean: torch.Tensor | None = None,
        source_std: torch.Tensor | None = None,
    ) -> None:
        self.visual_path, self.skeleton_path = visual_cache, skeleton_cache
        self.mask = np.load(skeleton_mask, mmap_mode="r")
        self.indices, self.labels, self.augment = (
            indices.astype(np.int64),
            labels,
            augment,
        )
        self.temporal_crop = temporal_crop
        self.skeleton_motion = skeleton_motion
        self.thermal_path = thermal_cache
        self.gain_low, self.gain_high = float(gain_low), float(gain_high)
        self.return_index = bool(return_index)
        self.temporal_stride = max(1, int(temporal_stride))
        self.skeleton_acceleration = bool(skeleton_acceleration)
        self.skeleton_noise = float(skeleton_noise)
        self.source_mean = source_mean
        self.source_std = source_std
        self._visual: np.ndarray | None = None
        self._skeleton: np.ndarray | None = None
        self._thermal: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        if self._visual is None:
            self._visual = np.load(self.visual_path, mmap_mode="r")
            self._skeleton = np.load(self.skeleton_path, mmap_mode="r")
            if self.thermal_path is not None:
                self._thermal = np.load(self.thermal_path, mmap_mode="r")
        frames = (
            torch.from_numpy(np.array(self._visual[index], copy=True))
            .float()
            .div_(255.0)
        )
        skeleton = torch.from_numpy(np.array(self._skeleton[index], copy=True))
        if (
            self.temporal_crop is not None
            and frames.shape[0] >= self.temporal_crop * self.temporal_stride
        ):
            limit = frames.shape[0] - self.temporal_crop * self.temporal_stride
            start = (
                int(torch.randint(limit + 1, ()).item()) if self.augment else limit // 2
            )
            frames = frames[
                start : start
                + self.temporal_crop * self.temporal_stride : self.temporal_stride
            ]
        if self.thermal_path is not None:
            thermal = (
                torch.from_numpy(np.array(self._thermal[index], copy=True))
                .float()
                .div_(255.0)
            )
            if (
                self.temporal_crop is not None
                and thermal.shape[0] >= self.temporal_crop * self.temporal_stride
            ):
                thermal = thermal[
                    start : start
                    + self.temporal_crop * self.temporal_stride : self.temporal_stride
                ]
            if thermal.shape[-2:] != frames.shape[-2:]:
                thermal = F.interpolate(
                    thermal,
                    size=frames.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            frames = torch.cat((frames, thermal), dim=1)
        if self.augment and torch.rand(()) < 0.5:
            frames = frames.flip(-1)
            skeleton = skeleton[:, LEFT_RIGHT]
            skeleton[..., 0].neg_()
        if self.augment:
            gain_depth = self.gain_low + (self.gain_high - self.gain_low) * torch.rand(
                1
            )
            gain_ir = self.gain_low + (self.gain_high - self.gain_low) * torch.rand(1)
            frames[:, :3].mul_(gain_depth).clamp_(0.0, 1.0)
            frames[:, 3:4].mul_(gain_ir).clamp_(0.0, 1.0)
            if self.thermal_path is not None:
                frames[:, 4:].mul_(
                    self.gain_low + (self.gain_high - self.gain_low) * torch.rand(1)
                ).clamp_(0.0, 1.0)
            if self.mask[index] and self.skeleton_noise > 0:
                skeleton.add_(torch.randn_like(skeleton) * self.skeleton_noise)
        if self.skeleton_motion:
            velocity = torch.zeros_like(skeleton)
            velocity[1:] = skeleton[1:] - skeleton[:-1]
            features = [skeleton, velocity]
            if self.skeleton_acceleration:
                acceleration = torch.zeros_like(velocity)
                acceleration[1:] = velocity[1:] - velocity[:-1]
                features.append(acceleration)
            skeleton = torch.cat(features, dim=-1)
        if self.thermal_path is not None:
            target_mean = torch.cat((MEAN, torch.tensor([[[[THERMAL_MEAN]]]])), dim=1)
            target_std = torch.cat((STD, torch.tensor([[[[THERMAL_STD]]]])), dim=1)
        else:
            target_mean, target_std = MEAN, STD
        if self.source_mean is not None and self.source_std is not None:
            # The train-only statistics describe the first four channels.  A
            # Thermal channel, when explicitly requested, keeps its frozen
            # training contract above and is never fitted from test clips.
            source_mean = self.source_mean.to(dtype=frames.dtype, device=frames.device)
            source_std = self.source_std.to(dtype=frames.dtype, device=frames.device)
            frames[:, :4] = ((frames[:, :4] - source_mean) / source_std) * STD + MEAN
            if self.thermal_path is not None:
                frames[:, 4:] = (frames[:, 4:] - target_mean[:, 4:]) / target_std[:, 4:]
        else:
            frames = (frames - target_mean) / target_std
        result = (
            (frames - MEAN) / STD,
            skeleton,
            bool(self.mask[index]),
            int(self.labels[index]),
        )
        if self.return_index:
            result = result + (index,)
        return result


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
        drop_last=shuffle,
        generator=torch.Generator().manual_seed(seed),
    )


def optimizers_for(model: nn.Module, visual_lr: float, new_lr: float):
    matrix_visual, matrix_new, vector_visual, vector_new = [], [], [], []
    for name, parameter in model.named_parameters():
        # Frozen parameters must not be handed to Muon/AdamW.  Besides avoiding
        # needless optimizer state, this makes --freeze-visual robust if a
        # parameter happens to retain a stale gradient from a resumed run.
        if not parameter.requires_grad:
            continue
        pretrained_visual = name.startswith("visual.") and ".project" not in name
        if parameter.ndim >= 2:
            (matrix_visual if pretrained_visual else matrix_new).append(parameter)
        else:
            (vector_visual if pretrained_visual else vector_new).append(parameter)
    muon = NDimMuon(
        [
            {"params": matrix_visual, "lr": visual_lr},
            {"params": matrix_new, "lr": new_lr},
        ],
        lr=new_lr,
        momentum=0.95,
        weight_decay=0.02,
    )
    adamw = torch.optim.AdamW(
        [
            {"params": vector_visual, "lr": visual_lr * 0.5},
            {"params": vector_new, "lr": new_lr * 0.5},
        ],
        weight_decay=0.01,
    )
    return muon, adamw


def train_epoch(
    model,
    loader,
    optimizers,
    schedulers,
    criterion,
    device,
    auxiliary_weight: float,
    freeze_visual: bool = False,
    teacher_logits: torch.Tensor | None = None,
    kd_weight: float = 0.0,
    kd_temperature: float = 2.0,
):
    model.train()
    # A frozen visual trunk should also keep BatchNorm statistics fixed; simply
    # disabling gradients is insufficient because model.train() updates them.
    if freeze_visual:
        model.visual.eval()
    total_loss = correct = count = 0
    for batch in loader:
        if len(batch) == 5:
            frames, skeleton, skeleton_mask, labels, sample_indices = batch
        else:
            frames, skeleton, skeleton_mask, labels = batch
            sample_indices = None
        frames = frames.to(device, non_blocking=True)
        skeleton = skeleton.to(device, non_blocking=True)
        skeleton_mask = skeleton_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, skeleton_logits = model(
                frames, skeleton, skeleton_mask, return_aux=True
            )
            loss = criterion(logits, labels)
            if skeleton_mask.any():
                loss = loss + auxiliary_weight * criterion(
                    skeleton_logits[skeleton_mask], labels[skeleton_mask]
                )
            if teacher_logits is not None and kd_weight > 0.0:
                if sample_indices is None:
                    raise RuntimeError("KD requires dataset indices")
                target = teacher_logits[sample_indices].to(
                    device=device, dtype=logits.dtype
                )
                temperature = float(kd_temperature)
                kd = F.kl_div(
                    F.log_softmax(logits / temperature, dim=1),
                    F.softmax(target / temperature, dim=1),
                    reduction="batchmean",
                ) * (temperature * temperature)
                loss = (1.0 - float(kd_weight)) * loss + float(kd_weight) * kd
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
def evaluate(model, loader, criterion, device, freeze_visual: bool = False):
    model.eval()
    if freeze_visual:
        model.visual.eval()
    total_loss = correct = count = 0
    predictions = []
    all_logits = []
    for frames, skeleton, skeleton_mask, labels in loader:
        frames, skeleton = (
            frames.to(device, non_blocking=True),
            skeleton.to(device, non_blocking=True),
        )
        skeleton_mask, labels = (
            skeleton_mask.to(device, non_blocking=True),
            labels.to(device, non_blocking=True),
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(frames, skeleton, skeleton_mask)
            loss = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        count += len(labels)
        predictions.append(logits.argmax(1).cpu().numpy())
        all_logits.append(logits.float().cpu().numpy())
    return (
        total_loss / count,
        correct / count,
        np.concatenate(predictions),
        np.concatenate(all_logits),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--visual-cache",
        type=Path,
        default=REPO_ROOT / ".cache/strict_v3/train_depth_ir.npy",
    )
    parser.add_argument(
        "--skeleton-cache",
        type=Path,
        default=REPO_ROOT / ".cache/strict_v3/train_skeleton.npy",
    )
    parser.add_argument(
        "--skeleton-mask",
        type=Path,
        default=REPO_ROOT / ".cache/strict_v3/train_skeleton_mask.npy",
    )
    parser.add_argument("--metadata", type=Path, default=RESULT_DIR / "metadata.npz")
    parser.add_argument(
        "--visual-checkpoint", type=Path, default=REPO_ROOT / "runs/visual/best_fp16.pt"
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="optional complete Fusion checkpoint to continue under the selected preprocessing contract",
    )
    parser.add_argument("--skeleton-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runs/fusion")
    parser.add_argument(
        "--yolo-weights", type=Path, default=CHECKPOINT_DIR / "yolo11n.pt"
    )
    parser.add_argument("--val-users", type=int, nargs="+", default=[3, 8, 20, 24])
    parser.add_argument(
        "--full-train",
        action="store_true",
        help="fit every labelled clip and save the best training-epoch checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--visual-lr", type=float, default=5e-5)
    parser.add_argument("--new-lr", type=float, default=2e-4)
    parser.add_argument("--auxiliary-weight", type=float, default=0.25)
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.02,
        help="cross-entropy label smoothing; 0 disables it",
    )
    parser.add_argument(
        "--freeze-visual",
        action="store_true",
        help="freeze the loaded R(2+1)D visual trunk, including BatchNorm statistics",
    )
    parser.add_argument(
        "--class-balanced",
        action="store_true",
        help="use mild inverse-frequency CE weights (mean_count/count)^0.25",
    )
    parser.add_argument(
        "--class-balance-power",
        type=float,
        default=0.25,
        help="power for --class-balanced inverse-frequency weights",
    )
    parser.add_argument(
        "--temporal-crop",
        type=int,
        default=0,
        help="random contiguous crop length for cached videos; 0 keeps all frames",
    )
    parser.add_argument(
        "--temporal-stride",
        type=int,
        default=1,
        help="stride inside the temporal crop; e.g. 2 samples 16 frames across a 32-frame cache",
    )
    parser.add_argument(
        "--skeleton-motion",
        action="store_true",
        help="append per-frame H36M joint velocity (6 skeleton channels)",
    )
    parser.add_argument(
        "--skeleton-acceleration",
        action="store_true",
        help="append per-frame acceleration after velocity (9 skeleton channels)",
    )
    parser.add_argument(
        "--skeleton-channels",
        type=int,
        default=0,
        help="explicit input channel count for a custom skeleton cache; 0 derives 3/6/9",
    )
    parser.add_argument(
        "--skeleton-noise",
        type=float,
        default=0.005,
        help="training Gaussian noise on valid skeletons; 0 disables it",
    )
    parser.add_argument(
        "--thermal-cache",
        type=Path,
        help="optional aligned thermal cache; creates a 5-channel visual stream",
    )
    parser.add_argument("--gain-low", type=float, default=0.9)
    parser.add_argument("--gain-high", type=float, default=1.1)
    parser.add_argument(
        "--teacher-logits",
        type=Path,
        help="optional full-train teacher logits [N,40] for soft-label KD",
    )
    parser.add_argument("--kd-weight", type=float, default=0.0)
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument(
        "--preprocessing-contract",
        type=Path,
        default=REPO_ROOT / ".cache/strict_v3/preprocessing.json",
        help="frozen train-only affine statistics; no test statistics are read",
    )
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; launch with CUDA_VISIBLE_DEVICES=0")
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    with np.load(args.metadata) as data:
        labels, users = data["train_y"], data["train_users"]
    val_mask = (
        np.zeros_like(users, dtype=bool)
        if args.full_train
        else np.isin(users, args.val_users)
    )
    train_indices, val_indices = np.flatnonzero(~val_mask), np.flatnonzero(val_mask)
    if set(users[~val_mask]).intersection(users[val_mask]):
        raise RuntimeError("Subject leakage detected")
    temporal_crop = args.temporal_crop if args.temporal_crop > 0 else None
    source_mean = source_std = None
    preprocessing_contract_used = False
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
        preprocessing_contract_used = True
    teacher_logits = None
    if args.teacher_logits is not None and args.kd_weight > 0.0:
        teacher_logits = torch.from_numpy(
            np.load(args.teacher_logits).astype(np.float32)
        )
        if teacher_logits.shape != (len(labels), 40):
            raise ValueError(
                f"teacher logits shape {tuple(teacher_logits.shape)} != {(len(labels), 40)}"
            )
        print(
            f"kd_teacher={args.teacher_logits} weight={args.kd_weight} temperature={args.kd_temperature}",
            flush=True,
        )
    train_loader = make_loader(
        FusionDataset(
            args.visual_cache,
            args.skeleton_cache,
            args.skeleton_mask,
            train_indices,
            labels,
            True,
            temporal_crop,
            args.skeleton_motion,
            args.thermal_cache,
            args.gain_low,
            args.gain_high,
            return_index=teacher_logits is not None,
            temporal_stride=args.temporal_stride,
            skeleton_acceleration=args.skeleton_acceleration,
            skeleton_noise=args.skeleton_noise,
            source_mean=source_mean,
            source_std=source_std,
        ),
        args.batch_size,
        True,
        args.workers,
        args.seed,
    )
    val_loader = make_loader(
        FusionDataset(
            args.visual_cache,
            args.skeleton_cache,
            args.skeleton_mask,
            val_indices,
            labels,
            False,
            temporal_crop,
            args.skeleton_motion,
            args.thermal_cache,
            args.gain_low,
            args.gain_high,
            temporal_stride=args.temporal_stride,
            skeleton_acceleration=args.skeleton_acceleration,
            skeleton_noise=args.skeleton_noise,
            source_mean=source_mean,
            source_std=source_std,
        ),
        args.batch_size * 2,
        False,
        args.workers,
        args.seed,
    )
    derived_skeleton_channels = (
        (9 if args.skeleton_acceleration else 6) if args.skeleton_motion else 3
    )
    skeleton_channels = args.skeleton_channels or derived_skeleton_channels
    if skeleton_channels < 1:
        raise ValueError(f"invalid skeleton channel count: {skeleton_channels}")
    model = TemporalFusionClassifier(
        pretrained_visual=False,
        skeleton_channels=skeleton_channels,
        visual_channels=5 if args.thermal_cache is not None else 4,
    )
    checkpoint = torch.load(
        args.visual_checkpoint, map_location="cpu", weights_only=True
    )
    model.load_visual_checkpoint(checkpoint["model_state"])
    if args.resume is not None:
        resumed = torch.load(args.resume, map_location="cpu", weights_only=True)
        state = resumed.get("model_state", resumed)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"resume state mismatch missing={missing} unexpected={unexpected}"
            )
        print(f"resumed_fusion_checkpoint={args.resume}", flush=True)
    if args.skeleton_checkpoint is not None:
        skeleton_checkpoint = torch.load(
            args.skeleton_checkpoint, map_location="cpu", weights_only=True
        )
        model.skeleton.load_state_dict(skeleton_checkpoint["model_state"], strict=True)
        print(f"loaded_skeleton_checkpoint={args.skeleton_checkpoint}", flush=True)
    if args.freeze_visual:
        for parameter in model.visual.parameters():
            parameter.requires_grad_(False)
        print("visual_trunk=frozen", flush=True)
    device = torch.device("cuda:0")
    model.to(device)
    optimizers = optimizers_for(model, args.visual_lr, args.new_lr)
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
    class_weights = None
    if args.class_balanced:
        counts = np.bincount(labels[train_indices], minlength=40).astype(np.float64)
        if np.any(counts <= 0):
            raise RuntimeError(
                f"class-balanced CE requires every class in training split; counts={counts}"
            )
        # The fourth-root weighting is deliberately mild: rare classes receive
        # some protection without allowing a tiny class to dominate Muon.
        class_weights = torch.as_tensor(
            np.power(counts.mean() / counts, args.class_balance_power),
            dtype=torch.float32,
            device=device,
        )
        print(
            f"class_balanced=True power={args.class_balance_power} "
            f"weight_range=({class_weights.min().item():.4f},{class_weights.max().item():.4f})",
            flush=True,
        )
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=args.label_smoothing
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_fp16.pt"
    best_accuracy, best_epoch, history = -1.0, 0, []
    skeleton_masks = np.load(args.skeleton_mask)
    print(
        f"device={torch.cuda.get_device_name(0)} train={len(train_indices)} val={len(val_indices)} "
        f"val_users={args.val_users} skeleton_train_present={skeleton_masks[train_indices].mean():.4f} "
        f"skeleton_val_present={skeleton_masks[val_indices].mean():.4f}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_epoch(
            model,
            train_loader,
            optimizers,
            schedulers,
            criterion,
            device,
            args.auxiliary_weight,
            args.freeze_visual,
            teacher_logits,
            args.kd_weight,
            args.kd_temperature,
        )
        if args.full_train:
            val_loss, val_accuracy, predictions, val_logits = (
                float("nan"),
                train_accuracy,
                np.empty(0, dtype=np.int64),
                np.empty((0, 40), dtype=np.float32),
            )
        else:
            val_loss, val_accuracy, predictions, val_logits = evaluate(
                model, val_loader, criterion, device, args.freeze_visual
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
        selection_accuracy = train_accuracy if args.full_train else val_accuracy
        if selection_accuracy > best_accuracy:
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
                    "validation_accuracy": val_accuracy,
                    "val_users": args.val_users,
                },
                checkpoint_path,
            )
            np.save(args.output_dir / "best_val_predictions.npy", predictions)
            np.save(args.output_dir / "best_val_logits.npy", val_logits)
            if args.full_train:
                torch.save(
                    {
                        "model_state": {
                            name: value.detach().cpu().half()
                            if value.is_floating_point()
                            else value.cpu()
                            for name, value in model.state_dict().items()
                        },
                        "epoch": epoch,
                        "validation_accuracy": None,
                        "val_users": [],
                    },
                    args.output_dir / f"epoch{epoch}_fp16.pt",
                )

    checkpoint_bytes = checkpoint_path.stat().st_size
    yolo_bytes = args.yolo_weights.stat().st_size
    metrics = {
        "best_validation_accuracy": None if args.full_train else best_accuracy,
        "best_training_accuracy": best_accuracy if args.full_train else None,
        "visual_baseline_accuracy": 0.6323076923076923,
        "improvement": best_accuracy - 0.6323076923076923,
        "best_epoch": best_epoch,
        "validation_users": args.val_users,
        "freeze_visual": args.freeze_visual,
        "class_balanced": args.class_balanced,
        "class_balance_power": args.class_balance_power,
        "label_smoothing": args.label_smoothing,
        "skeleton_acceleration": args.skeleton_acceleration,
        "skeleton_noise": args.skeleton_noise,
        "preprocessing_contract": str(args.preprocessing_contract),
        "train_only_preprocessing": preprocessing_contract_used,
        "class_weights": class_weights.detach().cpu().tolist()
        if class_weights is not None
        else None,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "fusion_checkpoint_bytes": checkpoint_bytes,
        "yolo_checkpoint_bytes": yolo_bytes,
        "combined_checkpoint_bytes": checkpoint_bytes + yolo_bytes,
        "under_100mb": checkpoint_bytes + yolo_bytes <= 100_000_000,
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
