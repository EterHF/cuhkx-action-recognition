#!/usr/bin/env python3
"""Build a strictV3 Temporal+Visual package without inactive branch weights."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch

from yolo_r2plus1d.strict_v3.paths import CHECKPOINT_DIR
from yolo_r2plus1d.strict_v3.release.bundle import MODEL_LIMIT_BYTES, build_bundle


def temporal_visual_package(source: dict[str, Any]) -> dict[str, Any]:
    """Remove Fusion/dead thermal weights while preserving the T+V contract."""
    required = {"fusion_4bit", "visual_member0", "dstformer", "release_contract"}
    missing = required.difference(source)
    if missing:
        raise RuntimeError(f"source package is missing required keys: {sorted(missing)}")
    result = copy.deepcopy(source)
    release = result["release_contract"]
    blend = result.get("blend")
    if not isinstance(release, dict) or not isinstance(blend, dict):
        raise RuntimeError("source package lacks a release/blend contract")
    if float(release.get("thermal_weight", float("nan"))) != 0.0:
        raise RuntimeError("refusing to remove an active thermal branch")
    if float(release["weights"].get("fusion", float("nan"))) <= 0.0:
        raise RuntimeError("source Fusion weight is already inactive")

    result.pop("fusion_4bit")
    removed = ["fusion_4bit"]
    if "thermal_mobilenet" in result:
        result.pop("thermal_mobilenet")
        removed.append("thermal_mobilenet")
    release["weights"]["fusion"] = 0.0
    blend["fusion_weight"] = 0.0
    if isinstance(release.get("presence"), dict):
        release["presence"]["fusion"] = "removed; fixed zero weight"
    if isinstance(blend.get("presence"), dict):
        blend["presence"]["fusion"] = "removed; fixed zero weight"
    release["deployment_branches"] = ["temporal", "visual"]
    release["removed_inactive_weights"] = removed
    return result


def run(args: argparse.Namespace) -> None:
    if args.output_model.exists() or args.output_bundle.exists():
        raise FileExistsError("refusing to overwrite a Temporal+Visual artifact")
    source = torch.load(args.source_model, map_location="cpu", weights_only=True)
    if not isinstance(source, dict):
        raise RuntimeError("source model package must be a mapping")
    package = temporal_visual_package(source)
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(package, args.output_model)
    build_bundle(args.output_model, args.detector, args.output_bundle)
    report = {
        "source_model_bytes": args.source_model.stat().st_size,
        "temporal_visual_model_bytes": args.output_model.stat().st_size,
        "reclaimed_model_bytes": (
            args.source_model.stat().st_size - args.output_model.stat().st_size
        ),
        "single_checkpoint_bytes": args.output_bundle.stat().st_size,
        "single_checkpoint_margin_bytes": (
            MODEL_LIMIT_BYTES - args.output_bundle.stat().st_size
        ),
        "removed": package["release_contract"]["removed_inactive_weights"],
        "visual_package_output_scale": package["release_contract"].get(
            "visual_package_output_scale"
        ),
        "temperatures": package["release_contract"]["temperatures"],
        "weights": package["release_contract"]["weights"],
    }
    args.output_model.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--source-model", type=Path, default=CHECKPOINT_DIR / "model.pt"
    )
    result.add_argument("--detector", type=Path, default=CHECKPOINT_DIR / "yolo11n.pt")
    result.add_argument("--output-model", type=Path, required=True)
    result.add_argument("--output-bundle", type=Path, required=True)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
