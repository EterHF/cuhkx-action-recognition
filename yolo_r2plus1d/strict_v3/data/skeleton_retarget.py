#!/usr/bin/env python3
"""Build original and mildly bone-retargeted DSTFormer frame-logit views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from yolo_r2plus1d.strict_v3.data.indexing import discover_train
from yolo_r2plus1d.strict_v3.data.skeleton_cache import load_person
from yolo_r2plus1d.strict_v3.models.public_dstformer import Net as PublicDSTNet
from yolo_r2plus1d.strict_v3.paths import DATA_DIR, RESULT_DIR
from yolo_r2plus1d.strict_v3.release.replay import RawFrames, crop_scale

PARENTS = (0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15)
BILATERAL_GROUPS = ((1, 4), (2, 5), (3, 6), (11, 14), (12, 15), (13, 16))
CENTRAL_BONES = (7, 8, 9, 10)


def mild_scales(seed: int, clip_index: int, magnitude: float) -> np.ndarray:
    rng = np.random.default_rng(seed + clip_index * 7919)
    scales = np.ones(17, dtype=np.float32)
    for group in BILATERAL_GROUPS:
        value = rng.uniform(1.0 - magnitude, 1.0 + magnitude)
        scales[list(group)] = value
    for joint in CENTRAL_BONES:
        scales[joint] = rng.uniform(1.0 - magnitude, 1.0 + magnitude)
    return scales


def retarget_bone_lengths(sequence: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Apply clip-fixed bone scales while retaining projected motion and directions."""
    source = np.asarray(sequence, dtype=np.float32)
    if source.shape[-2:] != (17, 3) or scales.shape != (17,):
        raise ValueError("expected sequence [...,17,3] and scales [17]")
    output = source.copy()
    for child, parent in enumerate(PARENTS):
        if child == 0:
            continue
        vector = source[:, child, :2] - source[:, parent, :2]
        length = np.linalg.norm(vector, axis=1)
        valid = (
            (source[:, child, 2] != 0.0)
            & (source[:, parent, 2] != 0.0)
            & (length > 1e-6)
        )
        if not valid.any():
            continue
        rebuilt = output[:, parent, :2] + vector * float(scales[child])
        output[valid, child, :2] = rebuilt[valid]
    return output


def load_clip(directory: Path, frames: int) -> np.ndarray:
    paths = sorted((directory / "predictions").glob("*.json"))
    if not paths:
        return np.zeros((frames, 17, 3), dtype=np.float32)
    indices = np.linspace(0, len(paths) - 1, frames).round().astype(int)
    values = []
    for index in indices:
        points = load_person(paths[int(index)])
        if points is None:
            points = np.zeros((17, 3), dtype=np.float32)
        values.append(crop_scale(points))
    return np.stack(values)


@torch.inference_mode()
def frame_logits(
    values: np.ndarray,
    package: dict,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model = PublicDSTNet()
    model.load_state_dict(package["dstformer"]["state_dict"], strict=True)
    model.to(device).eval()
    output = np.zeros((len(values), values.shape[1], 40), dtype=np.float32)
    loader = DataLoader(
        RawFrames(values),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    for inputs, indices in loader:
        batch, steps = inputs.shape[:2]
        flattened = inputs.reshape(batch * steps, 1, 17, 3).to(
            device, non_blocking=True
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits = model(flattened)
        output[indices.numpy()] = logits.reshape(batch, steps, 40).float().cpu().numpy()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-root", type=Path, default=DATA_DIR / "processed/train/HAR/data"
    )
    parser.add_argument("--metadata", type=Path, default=RESULT_DIR / "metadata.npz")
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--magnitude", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not 0.0 < args.magnitude <= 0.1:
        parser.error("--magnitude must be in (0,0.1]")
    samples = discover_train(args.train_root)
    with np.load(args.metadata, allow_pickle=False) as metadata:
        expected_keys = np.asarray(metadata["train_keys"])
    if not np.array_equal(np.asarray([sample[0] for sample in samples]), expected_keys):
        raise RuntimeError("training discovery order differs from frozen metadata")
    original = np.stack(
        [load_clip(paths["Skeleton"], args.frames) for _, _, _, paths in samples]
    )
    retargeted = np.stack(
        [
            retarget_bone_lengths(
                clip, mild_scales(args.seed, index, args.magnitude)
            )
            for index, clip in enumerate(original)
        ]
    )
    package = torch.load(args.package, map_location="cpu", weights_only=True)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    original_logits = frame_logits(original, package, device, args.batch_size)
    retargeted_logits = frame_logits(retargeted, package, device, args.batch_size)
    np.save(args.output_dir / "original_frame_logits.npy", original_logits)
    np.save(args.output_dir / "retargeted_frame_logits.npy", retargeted_logits)
    valid = np.any(original[..., 2] != 0.0, axis=(1, 2))
    displacement = np.linalg.norm(
        retargeted[valid, ..., :2] - original[valid, ..., :2], axis=-1
    )
    report = {
        "schema_version": "cuhkx-mild-bone-retarget/v2",
        "rows": len(original),
        "valid_rows": int(valid.sum()),
        "magnitude": args.magnitude,
        "bilateral_scales_shared": True,
        "clip_fixed_scales": True,
        "per_frame_projected_length_change_preserved": True,
        "root_trajectory_unchanged": bool(
            np.array_equal(retargeted[:, :, 0], original[:, :, 0])
        ),
        "confidence_unchanged": bool(
            np.array_equal(retargeted[..., 2], original[..., 2])
        ),
        "mean_joint_displacement": float(displacement.mean()),
        "max_joint_displacement": float(displacement.max()),
        "mean_frame_logit_abs_change": float(
            np.mean(np.abs(retargeted_logits[valid] - original_logits[valid]))
        ),
        "anonymous_test_accessed": False,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
