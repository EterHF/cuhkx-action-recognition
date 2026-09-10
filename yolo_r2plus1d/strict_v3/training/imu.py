"""Train fixed-epoch IMU models with a paired coordinate-view ablation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from yolo_r2plus1d.strict_v3.evaluation.visual_quantization import FOLDS
from yolo_r2plus1d.strict_v3.models.imu import IMUClassifier, select_view
from yolo_r2plus1d.strict_v3.training.base import seed_everything


@torch.inference_mode()
def predict(model, inputs, device):
    model.eval()
    logits = []
    for batch in inputs.split(128):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits.append(model(batch.to(device)).float().cpu())
    return torch.cat(logits).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=Path("results/strict_v3/metadata.npz"))
    parser.add_argument("--fold", choices=[*FOLDS, "full"], required=True)
    parser.add_argument("--view", choices=["local", "global"], required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    with np.load(args.metadata, allow_pickle=False) as meta:
        labels, users = meta["train_y"], meta["train_users"]
    signals = torch.from_numpy(np.load(args.inputs / "signals.npy", allow_pickle=False))
    valid = np.load(args.inputs / "sensor_valid.npy", allow_pickle=False).sum(1) >= 3
    x = select_view(signals, args.view)
    held = np.zeros(len(users), dtype=bool) if args.fold == "full" else np.isin(users, FOLDS[args.fold])
    train_rows, held_rows = np.flatnonzero(~held & valid), np.flatnonzero(held)
    # All normalization is learned by batch normalization on outer-train
    # minibatches. Physical input scaling is fixed in the parser.
    model = IMUClassifier().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 80)
    train_x = x[train_rows].to(args.device)
    train_y = torch.from_numpy(labels[train_rows]).to(args.device)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=64, shuffle=True)
    start = time.monotonic()
    for epoch in range(80):
        model.train()
        correct = count = 0
        total_loss = 0
        for batch, y in loader:
            # Shared temporal crop in both coordinate views and all locations.
            length = int(torch.randint(205, 257, ()).item())
            offset = int(torch.randint(0, 257 - length, ()).item())
            batch = F.interpolate(batch[:, :, offset:offset + length], size=256,
                                  mode="linear", align_corners=False)
            batch = batch + torch.randn_like(batch) * 0.01
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(batch)
                loss = F.cross_entropy(logits, y, label_smoothing=0.05)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            correct += int((logits.argmax(1) == y).sum())
            count += len(y)
            total_loss += float(loss.detach()) * len(y)
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(json.dumps({"epoch": epoch + 1, "train_acc": correct / count,
                              "loss": total_loss / count,
                              "seconds": time.monotonic() - start}), flush=True)
    state = {k: v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu()
             for k, v in model.state_dict().items()}
    torch.save({"model_state": state, "view": args.view, "seed": args.seed,
                "epochs": 80, "fold": args.fold, "train_users": np.unique(users[train_rows]).tolist()},
               args.output / "model.pt")
    model.load_state_dict(state)
    metrics = {"seconds": time.monotonic() - start, "parameters": sum(p.numel() for p in model.parameters()),
               "train_rows": len(train_rows), "test_loaded": False,
               "checkpoint_bytes": (args.output / "model.pt").stat().st_size}
    if len(held_rows):
        output = predict(model, x[held_rows], args.device)
        np.savez_compressed(args.output / "held.npz", rows=held_rows, logits=output)
        metrics["held_valid_accuracy"] = float(np.mean(output[valid[held_rows]].argmax(1)
                                                      == labels[held_rows][valid[held_rows]]))
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
