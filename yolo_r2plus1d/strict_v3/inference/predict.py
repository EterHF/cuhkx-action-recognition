#!/usr/bin/env python3
"""Run the best Depth+IR checkpoint on the competition test set."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from yolo_r2plus1d.strict_v3.models.r2plus1d import DepthIRR2Plus1D
from yolo_r2plus1d.strict_v3.paths import REPO_ROOT, RESULT_DIR
from yolo_r2plus1d.strict_v3.training.base import MEAN, STD


class TestVideoDataset(Dataset):
    def __init__(self, cache: Path) -> None:
        self.frames = np.load(cache, mmap_mode="r")

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> torch.Tensor:
        frames = (
            torch.from_numpy(np.array(self.frames[index], copy=True))
            .float()
            .div_(255.0)
        )
        return (frames - MEAN) / STD


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache", type=Path, default=REPO_ROOT / ".cache/strict_v3/test_depth_ir.npy"
    )
    parser.add_argument("--metadata", type=Path, default=RESULT_DIR / "metadata.npz")
    parser.add_argument(
        "--checkpoint", type=Path, default=REPO_ROOT / "runs/visual/best_fp16.pt"
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runs/visual")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; launch with CUDA_VISIBLE_DEVICES=0")
    with np.load(args.metadata) as data:
        test_ids = data["test_ids"]
    dataset = TestVideoDataset(args.cache)
    if len(dataset) != len(test_ids):
        raise RuntimeError(
            f"Cache/ID mismatch: {len(dataset)} frames vs {len(test_ids)} IDs"
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = DepthIRR2Plus1D(pretrained=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = torch.device("cuda:0")
    model.to(device).eval()
    probabilities = []
    for index, frames in enumerate(loader, 1):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            probabilities.append(
                model(frames.to(device, non_blocking=True))
                .softmax(1)
                .float()
                .cpu()
                .numpy()
            )
        if index % 5 == 0 or index == len(loader):
            print(f"predicted_batches={index}/{len(loader)}", flush=True)
    probabilities_array = np.concatenate(probabilities)
    predictions = probabilities_array.argmax(axis=1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "test_probabilities.npy", probabilities_array)
    submission = args.output_dir / "submission.csv"
    with submission.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "prediction"))
        writer.writeheader()
        writer.writerows(
            {
                "path": f"small_model_track_test/{sample_id}/",
                "prediction": int(prediction),
            }
            for sample_id, prediction in zip(test_ids, predictions, strict=True)
        )
    print(
        f"saved={submission} rows={len(predictions)} prediction_range="
        f"[{predictions.min()},{predictions.max()}] unique_classes={len(np.unique(predictions))}",
        flush=True,
    )


if __name__ == "__main__":
    main()
