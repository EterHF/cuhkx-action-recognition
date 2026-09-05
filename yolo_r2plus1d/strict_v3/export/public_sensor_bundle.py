#!/usr/bin/env python3
"""Build one size-auditable mixed-int8 bundle for the three public branches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from yolo_r2plus1d.strict_v3.export.int8_state import pack_state
from yolo_r2plus1d.strict_v3.training.public_backbone_screen import sha256_file


def validated_state(directory: Path, expected: str) -> dict[str, torch.Tensor]:
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    if metrics.get("protocol") not in {"full-fit-public-encoder/v1", "full-fit-public-sensor/v1"}:
        raise RuntimeError(f"not a full-fit checkpoint: {directory}")
    if metrics.get("project_checkpoint_loaded", metrics.get("old_project_checkpoint_loaded")) is not False:
        raise RuntimeError(f"project checkpoint lineage is not clean: {directory}")
    observed = metrics.get("modality", metrics.get("mode"))
    if observed != expected:
        raise RuntimeError(f"expected {expected}, found {observed}: {directory}")
    return torch.load(directory / "model.pt", map_location="cpu", weights_only=True)


def run(args: argparse.Namespace) -> None:
    depth = validated_state(args.depth_dir, "depth")
    infrared = validated_state(args.ir_dir, "ir")
    skeleton = validated_state(args.skeleton_dir, "skeleton")
    skeleton_fp16 = {
        key: value.half() if value.is_floating_point() else value
        for key, value in skeleton.items()
    }
    bundle = {
        "format": "public-sensor-equal-logit-mixed-int8/v1",
        "weights": [1 / 3, 1 / 3, 1 / 3],
        "depth": pack_state(depth),
        "ir": pack_state(infrared),
        "skeleton": skeleton_fp16,
        "training": {
            "rows": 3036,
            "sensor_epochs": 15,
            "skeleton_epochs": 30,
            "seed": 2026,
            "historical_project_checkpoint_loaded": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, args.output)
    size = args.output.stat().st_size
    if size >= args.limit_bytes:
        raise RuntimeError(f"bundle exceeds size limit: {size} >= {args.limit_bytes}")
    print(json.dumps({"path": str(args.output), "bytes": size, "sha256": sha256_file(args.output)}))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--depth-dir", type=Path, required=True)
    result.add_argument("--ir-dir", type=Path, required=True)
    result.add_argument("--skeleton-dir", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--limit-bytes", type=int, default=100_000_000)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
