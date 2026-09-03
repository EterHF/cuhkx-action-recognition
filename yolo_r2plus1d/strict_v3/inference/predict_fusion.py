#!/usr/bin/env python3
"""Generate a competition submission from the temporal visual+skeleton fusion model.

This intentionally mirrors ``FusionDataset`` preprocessing used during training:
the visual cache is scaled to [0, 1] and transformed by the frozen train-only
affine contract; the skeleton cache is already pelvis-centered H36M-17
coordinates.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from yolo_r2plus1d.strict_v3.models.fusion import TemporalFusionClassifier
from yolo_r2plus1d.strict_v3.paths import REPO_ROOT, RESULT_DIR
from yolo_r2plus1d.strict_v3.training.base import MEAN, STD


class FusionTestDataset(Dataset):
    def __init__(
        self,
        visual_cache: Path,
        skeleton_cache: Path,
        mask_cache: Path,
        temporal_crop: int | None = None,
        source_mean: torch.Tensor | None = None,
        source_std: torch.Tensor | None = None,
    ) -> None:
        self.visual = np.load(visual_cache, mmap_mode="r")
        self.skeleton = np.load(skeleton_cache, mmap_mode="r")
        self.mask = np.load(mask_cache, mmap_mode="r")
        self.temporal_crop = temporal_crop
        self.source_mean = source_mean
        self.source_std = source_std
        if len(self.visual) != len(self.skeleton) or len(self.visual) != len(self.mask):
            raise ValueError("visual/skeleton/mask caches have different lengths")

    def __len__(self) -> int:
        return len(self.visual)

    def __getitem__(self, index: int):
        frames = (
            torch.from_numpy(np.array(self.visual[index], copy=True))
            .float()
            .div_(255.0)
        )
        skeleton = torch.from_numpy(np.array(self.skeleton[index], copy=True)).float()
        if self.temporal_crop is not None and frames.shape[0] >= self.temporal_crop:
            start = (frames.shape[0] - self.temporal_crop) // 2
            frames = frames[start : start + self.temporal_crop]
            skeleton = skeleton[start : start + self.temporal_crop]
        if self.source_mean is not None and self.source_std is not None:
            frames = ((frames - self.source_mean) / self.source_std) * STD + MEAN
        else:
            frames = (frames - MEAN) / STD
        return frames, skeleton, bool(self.mask[index])


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--visual-cache",
        type=Path,
        default=REPO_ROOT / ".cache/strict_v3/test_depth_ir.npy",
    )
    parser.add_argument(
        "--skeleton-cache",
        type=Path,
        default=REPO_ROOT / ".cache/strict_v3/test_skeleton.npy",
    )
    parser.add_argument(
        "--skeleton-mask",
        type=Path,
        default=REPO_ROOT / ".cache/strict_v3/test_skeleton_mask.npy",
    )
    parser.add_argument("--metadata", type=Path, default=RESULT_DIR / "metadata.npz")
    parser.add_argument(
        "--checkpoint", type=Path, default=REPO_ROOT / "runs/fusion/best_fp16.pt"
    )
    parser.add_argument(
        "--output", type=Path, default=REPO_ROOT / "runs/fusion/submission.csv"
    )
    parser.add_argument("--probabilities", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--temporal-crop",
        type=int,
        default=0,
        help="center crop this many cached frames at inference",
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
    with np.load(args.metadata) as data:
        test_ids = data["test_ids"]
    source_mean = source_std = None
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
    dataset = FusionTestDataset(
        args.visual_cache,
        args.skeleton_cache,
        args.skeleton_mask,
        args.temporal_crop if args.temporal_crop > 0 else None,
        source_mean,
        source_std,
    )
    if len(dataset) != len(test_ids):
        raise ValueError(f"cache/metadata mismatch: {len(dataset)} vs {len(test_ids)}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = TemporalFusionClassifier(pretrained_visual=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = torch.device("cuda:0")
    model.to(device).eval()
    probabilities = []
    for batch_index, (frames, skeleton, skeleton_mask) in enumerate(loader, 1):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(
                frames.to(device, non_blocking=True),
                skeleton.to(device, non_blocking=True),
                skeleton_mask.to(device, non_blocking=True),
            )
            probabilities.append(logits.float().softmax(1).cpu().numpy())
        if batch_index % 5 == 0 or batch_index == len(loader):
            print(f"predicted_batches={batch_index}/{len(loader)}", flush=True)

    probabilities_array = np.concatenate(probabilities)
    predictions = probabilities_array.argmax(axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "prediction"))
        writer.writeheader()
        writer.writerows(
            {
                "path": f"small_model_track_test/{sample_id}/",
                "prediction": int(prediction),
            }
            for sample_id, prediction in zip(test_ids, predictions, strict=True)
        )
    probabilities_path = args.probabilities or args.output.with_suffix(".npy")
    np.save(probabilities_path, probabilities_array)
    print(
        f"saved={args.output} probabilities={probabilities_path} rows={len(predictions)} "
        f"skeleton_present={np.asarray(dataset.mask).mean():.4f} unique_classes={len(np.unique(predictions))}",
        flush=True,
    )


if __name__ == "__main__":
    main()
