#!/usr/bin/env python3
"""Run the self-contained equal-logit public sensor bundle on test caches."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from yolo_r2plus1d.strict_v3.export.int8_state import unpack_state
from yolo_r2plus1d.strict_v3.models.research_skeleton import HighRateSkeletonClassifier
from yolo_r2plus1d.strict_v3.training.finetune_public_omnivore import (
    OmnivoreClassifier,
    prepare_inputs,
)
from yolo_r2plus1d.strict_v3.training.public_backbone_screen import sha256_file


class FrameDataset(Dataset):
    def __init__(self, path: Path) -> None:
        self.frames = np.load(path, mmap_mode="r")

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(np.array(self.frames[index], copy=True))


def build_omnivore(source_dir: Path) -> OmnivoreClassifier:
    sys.path.insert(0, str(source_dir.resolve()))
    from omnivore.models import omnivore_swinT

    return OmnivoreClassifier(omnivore_swinT(pretrained=False, load_heads=False))


@torch.inference_mode()
def sensor_logits(
    package: dict,
    modality: str,
    source_dir: Path,
    cache: Path,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> np.ndarray:
    model = build_omnivore(source_dir)
    model.load_state_dict(unpack_state(package, model.state_dict()), strict=True)
    model.to(device).eval()
    outputs = []
    loader = DataLoader(
        FrameDataset(cache),
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    for frames in loader:
        inputs = prepare_inputs(frames.to(device, non_blocking=True), modality, False)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            outputs.append(model(inputs).float().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(outputs)


@torch.inference_mode()
def skeleton_logits(
    state: dict[str, torch.Tensor],
    root: Path,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> np.ndarray:
    values = torch.from_numpy(np.array(np.load(root / "test_skeleton.npy"), copy=True))
    mask = torch.from_numpy(np.array(np.load(root / "test_skeleton_mask.npy"), copy=True)).bool()
    positions = torch.from_numpy(np.array(np.load(root / "test_positions.npy"), copy=True))
    model = HighRateSkeletonClassifier().to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    outputs = []
    loader = DataLoader(
        TensorDataset(values, mask, positions),
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    for skeleton, valid, location in loader:
        with torch.autocast(device.type, dtype=torch.bfloat16):
            outputs.append(
                model(
                    skeleton.to(device, non_blocking=True),
                    valid.to(device, non_blocking=True),
                    location.to(device, non_blocking=True),
                ).float().cpu().numpy()
            )
    return np.concatenate(outputs)


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    bundle = torch.load(args.bundle, map_location="cpu", weights_only=True)
    if bundle.get("format") != "public-sensor-equal-logit-mixed-int8/v1":
        raise RuntimeError("unsupported public sensor bundle")
    depth = sensor_logits(
        bundle["depth"], "depth", args.source_dir, args.visual_cache,
        device, args.sensor_batch_size, args.workers,
    )
    infrared = sensor_logits(
        bundle["ir"], "ir", args.source_dir, args.visual_cache,
        device, args.sensor_batch_size, args.workers,
    )
    skeleton = skeleton_logits(
        bundle["skeleton"], args.skeleton_root, device,
        args.skeleton_batch_size, args.workers,
    )
    if not (depth.shape == infrared.shape == skeleton.shape):
        raise RuntimeError("branch logit shapes differ")
    logits = (depth + infrared + skeleton) / 3
    prediction = logits.argmax(1)
    with args.sample_submission.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(prediction) or set(rows[0]) != {"path", "prediction"}:
        raise RuntimeError("sample submission does not match predictions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "prediction"))
        writer.writeheader()
        writer.writerows(
            {"path": row["path"], "prediction": int(label)}
            for row, label in zip(rows, prediction, strict=True)
        )
    classes, counts = np.unique(prediction, return_counts=True)
    metrics = {
        "rows": len(prediction),
        "class_coverage": len(classes),
        "class_histogram": {str(int(key)): int(value) for key, value in zip(classes, counts)},
        "bundle_bytes": args.bundle.stat().st_size,
        "bundle_sha256": sha256_file(args.bundle),
        "submission_sha256": sha256_file(args.output),
        "anonymous_test_labels_accessed": False,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--bundle", type=Path, required=True)
    result.add_argument("--source-dir", type=Path, required=True)
    result.add_argument("--visual-cache", type=Path, required=True)
    result.add_argument("--skeleton-root", type=Path, required=True)
    result.add_argument("--sample-submission", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--sensor-batch-size", type=int, default=8)
    result.add_argument("--skeleton-batch-size", type=int, default=64)
    result.add_argument("--workers", type=int, default=4)
    result.add_argument("--device", default="cuda:0")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
