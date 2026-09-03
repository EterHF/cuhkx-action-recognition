#!/usr/bin/env python3
"""Train-only preprocessing contract for Depth/IR and Thermal caches.

The old loaders used a mixture of Kinetics constants, per-clip transforms and
historical test-time probes.  This module provides one auditable alternative:
statistics are fitted from labelled training clips only, then frozen and
reused for validation and test.  It deliberately has no function that reads
test statistics while constructing the transform.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from yolo_r2plus1d.strict_v3.paths import REPO_ROOT, RESULT_DIR

KINETICS_MEAN = np.asarray([0.43216, 0.394666, 0.37645, 0.400], dtype=np.float64)
KINETICS_STD = np.asarray([0.22803, 0.22145, 0.216989, 0.222], dtype=np.float64)


def _channel_stats(cache: Path, indices: np.ndarray, stride: int = 8) -> tuple[np.ndarray, np.ndarray]:
    values = np.load(cache, mmap_mode="r")
    if values.ndim != 5 or values.shape[2] != 4:
        raise ValueError(f"{cache}: expected [N,T,4,H,W], got {values.shape}")
    # Subsample spatial pixels deterministically; this is sufficient for the
    # affine contract and avoids loading a multi-gigabyte cache into RAM.
    sample = np.asarray(values[indices, ::2, :, ::stride, ::stride], dtype=np.float64) / 255.0
    flat = sample.transpose(0, 1, 3, 4, 2).reshape(-1, 4)
    return flat.mean(0), np.maximum(flat.std(0), 1e-4)


def fit_train_contract(cache: Path, metadata: Path, output: Path,
                       indices: np.ndarray | None = None) -> dict[str, object]:
    meta = np.load(metadata, allow_pickle=True)
    if indices is None:
        indices = np.arange(len(meta["train_y"]), dtype=np.int64)
    else:
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or not len(indices):
            raise ValueError("train-only preprocessing needs at least one training row")
        if indices.min() < 0 or indices.max() >= len(meta["train_y"]):
            raise ValueError("preprocessing indices outside the training cache")
    mean, std = _channel_stats(cache, indices)
    fit_users = np.unique(meta["train_users"].astype(np.int64)[indices]).astype(int).tolist()
    contract = {
        "schema_version": "cuhkx-train-only-affine/v1",
        "source_split": "train",
        "cache": str(cache),
        "channels": ["depth_r", "depth_g", "depth_b", "ir"],
        "mean": mean.tolist(),
        "std": std.tolist(),
        "target_mean": KINETICS_MEAN.tolist(),
        "target_std": KINETICS_STD.tolist(),
        "formula": "((x - train_mean) / train_std) * target_std + target_mean",
        "fit_rows": int(len(indices)),
        "fit_users": fit_users,
        "test_statistics_used": False,
        "timestamp_metadata_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(contract, indent=2) + "\n")
    return contract


def transform(frames: np.ndarray, contract: dict[str, object]) -> np.ndarray:
    """Apply a frozen train-only affine map to uint8/float frames."""
    source = np.asarray(frames)
    if source.ndim < 4 or source.shape[-3] != 4:
        raise ValueError(f"expected [..,4,H,W] frames, got {source.shape}")
    array = source.astype(np.float32, copy=False)
    if source.dtype.kind in "ui":
        array = array / 255.0
    mean = np.asarray(contract["mean"], dtype=np.float32)
    std = np.asarray(contract["std"], dtype=np.float32)
    target_mean = np.asarray(contract["target_mean"], dtype=np.float32)
    target_std = np.asarray(contract["target_std"], dtype=np.float32)
    shape = (1,) * (array.ndim - 3) + (4, 1, 1)
    return ((array - mean.reshape(shape)) / std.reshape(shape)) * target_std.reshape(shape) + target_mean.reshape(shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=REPO_ROOT / ".cache/strict_v3/train_depth_ir.npy")
    parser.add_argument("--metadata", type=Path, default=RESULT_DIR / "metadata.npz")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / ".cache/strict_v3/preprocessing.json")
    parser.add_argument("--train-users", type=int, nargs="*", default=None,
                        help="optional subject IDs; fit statistics only on these training users")
    args = parser.parse_args()
    indices = None
    if args.train_users is not None:
        metadata = np.load(args.metadata, allow_pickle=True)
        users = metadata["train_users"].astype(np.int64)
        indices = np.flatnonzero(np.isin(users, np.asarray(args.train_users, dtype=np.int64)))
    print(json.dumps(fit_train_contract(args.cache, args.metadata, args.output, indices), indent=2))


if __name__ == "__main__":
    main()
