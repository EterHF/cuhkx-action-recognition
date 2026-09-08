#!/usr/bin/env python3
"""Learn a small residual temporal decoder on public DSTFormer frame logits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from yolo_r2plus1d.strict_v3.data.validity import (
    load_validity_mask,
    masked_classification_loss,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Set(Dataset):
    def __init__(
        self,
        path,
        idx,
        y,
        train=False,
        seed=2026,
        reverse_prob=0.25,
        noise_std=0.01,
        loss_mask=None,
        secondary_path=None,
        secondary_probability=0.5,
        users=None,
    ):
        self.x = np.load(path, mmap_mode="r")
        self.idx = np.asarray(idx)
        self.y = y
        self.train = train
        self.seed = seed
        self.reverse_prob = float(reverse_prob)
        self.noise_std = float(noise_std)
        self.loss_mask = loss_mask
        self.secondary = (
            None if secondary_path is None else np.load(secondary_path, mmap_mode="r")
        )
        self.secondary_probability = float(secondary_probability)
        self.users = users
        if self.secondary is not None and self.secondary.shape != self.x.shape:
            raise ValueError("primary and secondary temporal logits must align")

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = int(self.idx[k])
        source = (
            self.secondary
            if self.train
            and self.secondary is not None
            and np.random.default_rng(self.seed + k * 3571).random()
            < self.secondary_probability
            else self.x
        )
        x = np.array(source[i], copy=True).astype(np.float32)
        if self.train:
            rng = np.random.default_rng(self.seed + k * 7919)
            if rng.random() < self.reverse_prob:
                x = x[::-1].copy()
            if self.noise_std:
                x += rng.normal(0, self.noise_std, x.shape).astype(np.float32)
        result = (torch.from_numpy(x), int(self.y[i]) if self.train else i)
        if self.train and self.loss_mask is not None:
            result += (bool(self.loss_mask[i]),)
        if self.train and self.users is not None:
            result += (int(self.users[i]),)
        return result


class Residual(nn.Module):
    def __init__(self, kind="gru"):
        super().__init__()
        self.kind = kind
        self.norm = nn.LayerNorm(40)
        if kind == "gru":
            self.temporal = nn.GRU(
                40, 64, 2, batch_first=True, bidirectional=True, dropout=0.15
            )
            dim = 128
        elif kind == "tcn":
            self.temporal = nn.Sequential(
                nn.Conv1d(40, 96, 3, padding=1),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Conv1d(96, 96, 3, padding=2, dilation=2),
                nn.GELU(),
                nn.Conv1d(96, 40, 1),
            )
            dim = 40
        else:
            layer = nn.TransformerEncoderLayer(
                40, 4, 160, 0.15, batch_first=True, norm_first=True
            )
            self.temporal = nn.TransformerEncoder(layer, 2)
            self.pos = nn.Parameter(torch.zeros(1, 16, 40))
            nn.init.trunc_normal_(self.pos, std=0.02)
            dim = 40
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 40))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward_features(self, x):
        base = x.mean(1)
        z = self.norm(x)
        if self.kind == "gru":
            z, _ = self.temporal(z)
        elif self.kind == "tcn":
            z = self.temporal(z.transpose(1, 2)).transpose(1, 2)
        else:
            z = self.temporal(z + self.pos)
        return base, z.mean(1)

    def forward(self, x):
        base, features = self.forward_features(x)
        return base + self.out(features)


def cross_user_supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    users: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, int]:
    """Use same-class/different-user positives without false same-class negatives."""
    normalized = F.normalize(features.float(), dim=1)
    logits = normalized @ normalized.T / float(temperature)
    identity = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    same_class = labels[:, None] == labels[None, :]
    positive = same_class & (users[:, None] != users[None, :])
    denominator = (~same_class | positive) & ~identity
    anchors = positive.any(dim=1)
    anchor_count = int(anchors.sum().item())
    if anchor_count == 0:
        return features.sum() * 0.0, 0
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = logits.exp() * denominator
    log_probability = logits - exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12).log()
    positive_mean = (log_probability * positive).sum(dim=1) / positive.sum(dim=1).clamp_min(1)
    return -positive_mean[anchors].mean(), anchor_count


@torch.inference_mode()
def pred(model, path, idx, y, dev, batch, workers):
    ds = Set(path, idx, y, False)
    ld = DataLoader(
        ds,
        batch,
        False,
        num_workers=workers,
        pin_memory=dev.type == "cuda",
    )
    out = np.zeros((len(idx), 40), np.float32)
    loc = {int(v): i for i, v in enumerate(idx)}
    model.eval()
    for x, ii in ld:
        with torch.autocast(
            device_type=dev.type,
            dtype=torch.bfloat16,
            enabled=dev.type == "cuda",
        ):
            z = model(x.to(dev, non_blocking=True))
        for r, v in enumerate(ii.numpy()):
            out[loc[int(v)]] = z[r].float().cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--logits", type=Path, default=Path(".cache/strict_v3/train_frame_logits.npy")
    )
    ap.add_argument("--test-logits", type=Path)
    ap.add_argument(
        "--metadata", type=Path, default=Path("results/strict_v3/metadata.npz")
    )
    ap.add_argument(
        "--validity-mask",
        type=Path,
        help="optional NPZ; exclude invalid branch inputs from training CE only",
    )
    ap.add_argument("--validity-key", default="skeleton")
    ap.add_argument("--secondary-logits", type=Path)
    ap.add_argument("--secondary-probability", type=float, default=0.5)
    ap.add_argument("--cross-user-contrast-weight", type=float, default=0.0)
    ap.add_argument("--contrast-temperature", type=float, default=0.1)
    ap.add_argument("--val-users", nargs="*", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--kind", choices=["gru", "tcn", "transformer"], default="gru")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument(
        "--scheduler-epochs",
        type=int,
        help="cosine schedule period; defaults to --epochs",
    )
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--reverse-prob", type=float, default=0.25)
    ap.add_argument("--noise-std", type=float, default=0.01)
    ap.add_argument("--label-smoothing", type=float, default=0.01)
    ap.add_argument(
        "--balance-power",
        type=float,
        default=0.0,
        help="sqrt-style class-balanced augmentation sampler; zero disables",
    )
    ap.add_argument("--save-all-val", action="store_true")
    ap.add_argument(
        "--select-last",
        action="store_true",
        help="use the predeclared final epoch instead of held-user selection",
    )
    ap.add_argument(
        "--defer-val-metrics",
        action="store_true",
        help="do not read/evaluate held labels until the fixed final checkpoint is written",
    )
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)
    if args.defer_val_metrics and not args.select_last:
        raise ValueError("--defer-val-metrics requires --select-last")
    if not 0.0 <= args.secondary_probability <= 1.0:
        raise ValueError("--secondary-probability must be in [0,1]")
    if args.cross_user_contrast_weight < 0.0:
        raise ValueError("--cross-user-contrast-weight must be non-negative")
    if args.contrast_temperature <= 0.0:
        raise ValueError("--contrast-temperature must be positive")
    if args.cross_user_contrast_weight > 0.0 and args.validity_mask is not None:
        raise ValueError("contrast and validity masking are separate experiments")
    scheduler_epochs = (
        args.epochs if args.scheduler_epochs is None else args.scheduler_epochs
    )
    if scheduler_epochs < args.epochs:
        raise ValueError("--scheduler-epochs must be at least --epochs")
    m = np.load(args.metadata)
    y = m["train_y"].astype(np.int64)
    u = m["train_users"].astype(np.int64)
    val_users = [] if args.val_users is None else args.val_users
    vm = np.isin(u, np.asarray(val_users))
    tr = np.flatnonzero(~vm)
    va = np.flatnonzero(vm)
    unmasked_train_rows = len(tr)
    validity = None
    if args.validity_mask is not None:
        validity = load_validity_mask(args.validity_mask, args.validity_key, len(y))
        if not validity[tr].any():
            raise RuntimeError("validity masking removed every temporal training target")
    model = Residual(args.kind).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, scheduler_epochs)
    lossfn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    sampler = None
    if args.balance_power > 0:
        counts = np.bincount(y[tr], minlength=40)
        sample_weight = np.maximum(counts[y[tr]], 1).astype(np.float64) ** (
            -args.balance_power
        )
        sampler = WeightedRandomSampler(
            torch.from_numpy(sample_weight),
            len(tr),
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
    ld = DataLoader(
        Set(
            args.logits,
            tr,
            y,
            True,
            seed=args.seed,
            reverse_prob=args.reverse_prob,
            noise_std=args.noise_std,
            loss_mask=validity,
            secondary_path=args.secondary_logits,
            secondary_probability=args.secondary_probability,
            users=u if args.cross_user_contrast_weight > 0.0 else None,
        ),
        args.batch,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=dev.type == "cuda",
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    best = -1.0
    hist = []
    for ep in range(1, args.epochs + 1):
        model.train()
        n = correct = 0
        total = 0.0
        contrast_total = contrast_anchors = 0
        for batch_values in ld:
            if args.cross_user_contrast_weight > 0.0:
                x, t, batch_users = batch_values
                batch_users = batch_users.to(dev, non_blocking=True)
                loss_mask = None
            elif len(batch_values) == 3:
                x, t, loss_mask = batch_values
                loss_mask = loss_mask.to(dev, non_blocking=True)
            else:
                x, t = batch_values
                loss_mask = None
            opt.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=dev.type,
                dtype=torch.bfloat16,
                enabled=dev.type == "cuda",
            ):
                inputs = x.to(dev, non_blocking=True)
                if args.cross_user_contrast_weight > 0.0:
                    base, features = model.forward_features(inputs)
                    z = base + model.out(features)
                else:
                    z = model(inputs)
                target = t.to(dev, non_blocking=True)
                loss = masked_classification_loss(lossfn, z, target, loss_mask)
                if args.cross_user_contrast_weight > 0.0:
                    contrast, anchors = cross_user_supervised_contrastive_loss(
                        features,
                        target,
                        batch_users,
                        args.contrast_temperature,
                    )
                    loss = loss + args.cross_user_contrast_weight * contrast
                    contrast_total += float(contrast.detach()) * anchors
                    contrast_anchors += anchors
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach()) * len(t)
            correct += int((z.argmax(1) == target).sum())
            n += len(t)
        sch.step()
        if len(va) and not args.defer_val_metrics:
            q = pred(model, args.logits, va, y, dev, args.batch, args.workers)
            acc = float((q.argmax(1) == y[va]).mean())
        elif len(va):
            # In the strict fixed-final protocol, held labels are not opened
            # during training.  The single final evaluation is performed
            # after the checkpoint has been written below.
            q = None
            acc = None
        else:
            acc = float((correct / max(1, n)))
        row = {
            "epoch": ep,
            "train_acc": correct / max(1, n),
            "loss": total / max(1, n),
            "val_acc": acc,
            "contrast_loss": contrast_total / max(1, contrast_anchors),
            "contrast_anchors": contrast_anchors,
        }
        hist.append(row)
        print(json.dumps(row), flush=True)
        if args.save_all_val and len(va) and q is not None:
            np.save(args.out / f"val_logits_ep{ep:02d}.npy", q)
        selected = ep == args.epochs if args.select_last else acc > best
        if selected:
            if acc is not None:
                best = acc
            if len(va) and q is not None:
                np.save(args.out / "val_logits.npy", q)
            torch.save(
                {
                    "model_state": {
                        k: v.detach().cpu().half()
                        if v.is_floating_point()
                        else v.detach().cpu()
                        for k, v in model.state_dict().items()
                    },
                    "kind": args.kind,
                    "val_users": val_users,
                    "val_acc": best,
                    "epoch": ep,
                    "scheduler_epochs": scheduler_epochs,
                    "checkpoint_selection": "fixed_final_epoch"
                    if args.select_last
                    else "best_accuracy",
                },
                args.out / "best_fp16.pt",
            )
    checkpoint = torch.load(
        args.out / "best_fp16.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(dev).eval()
    if len(va) and args.defer_val_metrics:
        # This is the only point at which held-user labels are read.  It is a
        # post-training OOF measurement, never a checkpoint/schedule choice.
        q = pred(model, args.logits, va, y, dev, args.batch, args.workers)
        np.save(args.out / "val_logits.npy", q)
        best = float((q.argmax(1) == y[va]).mean())
    if args.test_logits is not None:
        test_idx = np.arange(len(np.load(args.test_logits, mmap_mode="r")))
        np.save(
            args.out / "test_logits.npy",
            pred(
                model,
                args.test_logits,
                test_idx,
                np.zeros(len(test_idx), np.int64),
                dev,
                args.batch,
                args.workers,
            ),
        )
    (args.out / "metrics.json").write_text(
        json.dumps(
            {
                "kind": args.kind,
                "device": str(dev),
                "epochs": args.epochs,
                "batch": args.batch,
                "workers": args.workers,
                "learning_rate": args.lr,
                "seed": args.seed,
                "reverse_probability": args.reverse_prob,
                "noise_standard_deviation": args.noise_std,
                "label_smoothing": args.label_smoothing,
                "val_users": val_users,
                "best_val_acc": best,
                "checkpoint_epoch": checkpoint["epoch"],
                "checkpoint_selection": checkpoint["checkpoint_selection"],
                "scheduler_epochs": scheduler_epochs,
                "balance_power": args.balance_power,
                "defer_val_metrics": args.defer_val_metrics,
                "train_rows": len(tr),
                "unmasked_train_rows": unmasked_train_rows,
                "masked_training_rows": int((~validity[tr]).sum())
                if validity is not None
                else 0,
                "validity_mask": str(args.validity_mask.resolve())
                if args.validity_mask is not None
                else None,
                "validity_key": args.validity_key
                if args.validity_mask is not None
                else None,
                "secondary_logits": str(args.secondary_logits.resolve())
                if args.secondary_logits is not None
                else None,
                "secondary_probability": args.secondary_probability
                if args.secondary_logits is not None
                else 0.0,
                "cross_user_contrast_weight": args.cross_user_contrast_weight,
                "contrast_temperature": args.contrast_temperature,
                "inputs": {
                    "frame_logits": str(args.logits.resolve()),
                    "frame_logits_sha256": sha256(args.logits),
                    "test_frame_logits": (
                        None
                        if args.test_logits is None
                        else str(args.test_logits.resolve())
                    ),
                    "test_frame_logits_sha256": (
                        None if args.test_logits is None else sha256(args.test_logits)
                    ),
                    "metadata": str(args.metadata.resolve()),
                    "metadata_sha256": sha256(args.metadata),
                },
                "history": hist,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
