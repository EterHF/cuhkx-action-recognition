#!/usr/bin/env python3
"""Build and inspect the single-checkpoint strictV3 inference bundle."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import torch

from yolo_r2plus1d.strict_v3.paths import CHECKPOINT_DIR

BUNDLE_FORMAT = "cuhkx-single-checkpoint-v1"
MODEL_LIMIT_BYTES = 100_000_000
DEFAULT_BUNDLE = CHECKPOINT_DIR / "submission_bundle.pt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_bundle(path: Path) -> dict[str, Any]:
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(bundle, dict) or bundle.get("format") != BUNDLE_FORMAT:
        raise RuntimeError(f"unsupported inference bundle: {path}")
    if not isinstance(bundle.get("model"), dict):
        raise RuntimeError("inference bundle has no model package")
    detector = bundle.get("detector_bytes")
    if not isinstance(detector, torch.Tensor) or detector.dtype != torch.uint8:
        raise RuntimeError("inference bundle detector is not a uint8 tensor")
    if detector.ndim != 1 or detector.numel() == 0:
        raise RuntimeError("inference bundle detector is empty or malformed")
    metadata = bundle.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError("inference bundle has no metadata")
    actual_detector_hash = hashlib.sha256(detector.numpy().tobytes()).hexdigest()
    if actual_detector_hash != metadata.get("detector_sha256"):
        raise RuntimeError("embedded detector hash mismatch")
    return bundle


def write_detector(bundle: dict[str, Any], output: Path) -> None:
    output.write_bytes(bundle["detector_bytes"].numpy().tobytes())


def build_bundle(model_path: Path, detector_path: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing bundle: {output}")
    model = torch.load(model_path, map_location="cpu", weights_only=True)
    detector_bytes = detector_path.read_bytes()
    detector = torch.frombuffer(bytearray(detector_bytes), dtype=torch.uint8).clone()
    payload = {
        "format": BUNDLE_FORMAT,
        "model": model,
        "detector_bytes": detector,
        "metadata": {
            "model_sha256": sha256(model_path),
            "detector_sha256": hashlib.sha256(detector_bytes).hexdigest(),
            "model_size_rule": "all inference weights in one file under 100 MB",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    load_bundle(output)
    if output.stat().st_size >= MODEL_LIMIT_BYTES:
        raise RuntimeError(
            f"bundle is {output.stat().st_size} bytes; limit is strictly below "
            f"{MODEL_LIMIT_BYTES} bytes"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=CHECKPOINT_DIR / "model.pt")
    parser.add_argument("--detector", type=Path, default=CHECKPOINT_DIR / "yolo11n.pt")
    parser.add_argument("--output", type=Path, default=DEFAULT_BUNDLE)
    args = parser.parse_args()
    build_bundle(args.model, args.detector, args.output)
    print(
        f"wrote {args.output} ({args.output.stat().st_size} bytes, "
        f"{MODEL_LIMIT_BYTES - args.output.stat().st_size} bytes below limit)"
    )


if __name__ == "__main__":
    main()
