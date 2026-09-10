"""Fixed-epoch subject-held-out thermal training; no anonymous-data interface."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from yolo_r2plus1d.strict_v3.evaluation.visual_quantization import FOLDS
from yolo_r2plus1d.strict_v3.models.thermal import ThermalTSN


class ThermalDataset(Dataset):
    def __init__(self, cache: Path, rows: np.ndarray, labels: np.ndarray, train: bool):
        self.frames = np.load(cache, mmap_mode="r")
        self.rows, self.labels, self.train = rows, labels, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        # Four independently jittered segments during training; eight uniform
        # frames at evaluation. Every spatial transform is shared across time.
        positions = (
            np.arange(4) * 4 + np.random.randint(0, 4, 4)
            if self.train else np.linspace(0, 15, 8).round().astype(int)
        )
        frames = torch.from_numpy(np.array(self.frames[row, positions], copy=True))
        if self.train:
            size = np.random.randint(136, 161)
            top, left = np.random.randint(0, 161 - size, 2)
            frames = TF.resized_crop(frames, int(top), int(left), int(size), int(size), [160, 160])
            if np.random.rand() < 0.5:
                frames = frames.flip(-1)
        return frames, int(self.labels[row])


def split_rows(users: np.ndarray, valid: np.ndarray, fold: str):
    held = np.zeros(len(users), dtype=bool) if fold == "full" else np.isin(users, FOLDS[fold])
    return np.flatnonzero(~held & valid), np.flatnonzero(held)


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    output = []
    for frames, _ in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output.append(model(frames.to(device)).float().cpu().numpy())
    return np.concatenate(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", choices=[*FOLDS, "full"], required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--temporal-shift", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    with np.load(args.metadata, allow_pickle=False) as meta:
        labels, users = meta["train_y"], meta["train_users"]
    valid = np.load(args.valid, allow_pickle=False)
    train, held = split_rows(users, valid, args.fold)
    model = ThermalTSN()
    state = torch.load(args.pretrained, map_location="cpu", weights_only=True)
    state = {k: v for k, v in state.items() if not k.startswith("fc.")}
    model.encoder.load_state_dict(state, strict=True)
    if args.temporal_shift:
        model.enable_shift()
    model.to(args.device)
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": 1e-4},
        {"params": model.head.parameters(), "lr": 1e-3},
    ], weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    loader = DataLoader(ThermalDataset(args.cache, train, labels, True),
                        batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        pin_memory=True, persistent_workers=args.workers > 0)
    started = time.monotonic()
    for epoch in range(args.epochs):
        model.train()
        # Fixed ImageNet running statistics prevent the four correlated frames
        # per clip from defining a subject-dependent normalization at inference.
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        loss_sum = correct = count = 0
        for frames, y in loader:
            frames, y = frames.to(args.device), y.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(frames)
                loss = F.cross_entropy(logits, y, label_smoothing=0.05)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(y)
            correct += int((logits.argmax(1) == y).sum())
            count += len(y)
        scheduler.step()
        print(json.dumps({"epoch": epoch + 1, "loss": loss_sum / count,
                          "train_accuracy": correct / count,
                          "seconds": round(time.monotonic() - started, 1)}), flush=True)
    # Evaluate the exact fp16-stored state, with the same fp32/bfloat16 compute
    # recipe used by the future package. No held metrics select an epoch.
    saved = {k: v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu()
             for k, v in model.state_dict().items()}
    torch.save({"model_state": saved, "seed": args.seed, "fold": args.fold,
                "epochs": args.epochs, "temporal_shift": args.temporal_shift,
                "train_users": np.unique(users[train]).tolist()},
               args.output / "model.pt")
    model.load_state_dict(saved)
    if len(held):
        evaluation = DataLoader(ThermalDataset(args.cache, held, labels, False),
                                batch_size=args.batch_size, num_workers=args.workers,
                                pin_memory=True)
        logits = predict(model, evaluation, args.device)
        np.savez_compressed(args.output / "held.npz", rows=held, logits=logits)
        report = {"held_accuracy": float(np.mean(logits.argmax(1) == labels[held])),
                  "held_valid_accuracy": float(np.mean(
                      logits[valid[held]].argmax(1) == labels[held][valid[held]]))}
    else:
        report = {}
    report.update({"train_rows": len(train), "held_rows": len(held),
                   "seconds": time.monotonic() - started, "test_data_loaded": False,
                   "checkpoint_bytes": (args.output / "model.pt").stat().st_size})
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
