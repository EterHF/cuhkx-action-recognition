#!/usr/bin/env python3
"""Audit branch input validity from decodable source files and skeleton cache masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from yolo_r2plus1d.strict_v3.data.indexing import (
    IMAGE_SUFFIXES,
    discover_train,
)
from yolo_r2plus1d.strict_v3.paths import DATA_DIR, RESULT_DIR


def has_decodable_image(directory: Path) -> bool:
    """Return true only after an image from the recorded modality decodes."""
    if not directory.is_dir():
        return False
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            return True
        except (OSError, UnidentifiedImageError, ValueError):
            continue
    return False


def load_validity_mask(path: Path, key: str, rows: int) -> np.ndarray:
    """Load a named, one-dimensional boolean mask and validate its contract."""
    with np.load(path, allow_pickle=False) as values:
        if key not in values.files:
            raise KeyError(f"validity key {key!r} is absent from {path}")
        mask = np.asarray(values[key])
    if mask.shape != (rows,) or mask.dtype != np.bool_:
        raise ValueError(
            f"validity mask {key!r} must be bool[{rows}], got {mask.dtype}{mask.shape}"
        )
    return mask


def masked_classification_loss(
    criterion: torch.nn.Module,
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid: torch.Tensor | None,
) -> torch.Tensor:
    """Mask supervision while retaining every row in the forward pass."""
    if valid is None:
        return criterion(logits, labels)
    if valid.any():
        return criterion(logits[valid], labels[valid])
    return logits.sum() * 0.0


def audit(args: argparse.Namespace) -> None:
    samples = discover_train(args.train_root)
    with np.load(args.metadata, allow_pickle=False) as metadata:
        keys = np.asarray(metadata["train_keys"])
        labels = np.asarray(metadata["train_y"], dtype=np.int64)
        users = np.asarray(metadata["train_users"], dtype=np.int64)
    discovered = np.asarray([sample[0] for sample in samples])
    if not np.array_equal(discovered, keys):
        raise RuntimeError("training discovery order differs from frozen metadata")

    depth = np.asarray(
        [has_decodable_image(paths["Depth_Color"]) for _, _, _, paths in samples]
    )
    infrared = np.asarray(
        [has_decodable_image(paths["IR"]) for _, _, _, paths in samples]
    )
    frame_mask = np.load(args.skeleton_frame_mask, allow_pickle=False)
    if frame_mask.ndim != 2 or frame_mask.shape[0] != len(samples):
        raise ValueError(
            "skeleton frame mask must have shape [training rows, temporal frames]"
        )
    skeleton = np.asarray(frame_mask, dtype=bool).any(axis=1)
    visual = depth | infrared
    no_primary = ~visual & ~skeleton
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        depth=depth,
        infrared=infrared,
        visual=visual,
        skeleton=skeleton,
        all_primary=visual & skeleton,
        no_primary=no_primary,
    )

    def grouped(values: np.ndarray) -> dict[str, int]:
        unique, counts = np.unique(values[no_primary], return_counts=True)
        return {str(int(item)): int(count) for item, count in zip(unique, counts, strict=True)}

    report = {
        "schema_version": "cuhkx-branch-validity/v1",
        "rows": len(samples),
        "valid": {
            "depth": int(depth.sum()),
            "infrared": int(infrared.sum()),
            "visual_depth_or_ir": int(visual.sum()),
            "skeleton": int(skeleton.sum()),
            "all_primary": int((visual & skeleton).sum()),
        },
        "missing": {
            "depth": int((~depth).sum()),
            "infrared": int((~infrared).sum()),
            "visual_depth_and_ir": int((~visual).sum()),
            "skeleton": int((~skeleton).sum()),
            "all_primary": int(no_primary.sum()),
        },
        "no_primary_by_class": grouped(labels),
        "no_primary_by_user": grouped(users),
        "validity_source": {
            "visual": "at least one source image passes PIL verification",
            "skeleton": "at least one frame passed the existing load_person cache parser",
        },
        "anonymous_test_accessed": False,
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--train-root", type=Path, default=DATA_DIR / "processed/train/HAR/data"
    )
    result.add_argument("--metadata", type=Path, default=RESULT_DIR / "metadata.npz")
    result.add_argument("--skeleton-frame-mask", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


if __name__ == "__main__":
    audit(parser().parse_args())
