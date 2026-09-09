#!/usr/bin/env python3
"""Materialize the explicitly authorized gate-off anonymous-test submission."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from yolo_r2plus1d.strict_v3.data.validity import has_decodable_image
from yolo_r2plus1d.strict_v3.evaluation.quality_gate_ablation import (
    file_record,
    fixed_quality_logits,
)
from yolo_r2plus1d.strict_v3.paths import DATA_DIR, RESULT_DIR
from yolo_r2plus1d.strict_v3.release.blend import apply_gate


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gate_off_predictions(
    visual: np.ndarray,
    temporal: np.ndarray,
    both_valid: np.ndarray,
    release: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep inherited-gate output except on rows with both primary inputs."""
    weights = dict(release["weights"])
    weights["fusion"] = 0.0
    control_logits, _ = apply_gate(
        {"fusion": np.zeros_like(visual), "visual": visual, "temporal": temporal},
        release["temperatures"],
        weights,
    )
    candidate_logits = np.array(control_logits, copy=True)
    candidate_logits[both_valid] = fixed_quality_logits(
        visual, temporal, release["temperatures"], weights
    )[both_valid]
    return control_logits.argmax(1), candidate_logits.argmax(1)


def run(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected = manifest["test_source_sha256"]
    if sha256(args.visual_logits) != expected["visual"]:
        raise RuntimeError("Visual test logits do not match the frozen release source")
    if sha256(args.temporal_logits) != expected["temporal"]:
        raise RuntimeError("Temporal test logits do not match the frozen release source")

    with np.load(args.metadata, allow_pickle=False) as metadata:
        test_ids = np.asarray(metadata["test_ids"])
    visual = np.asarray(np.load(args.visual_logits, allow_pickle=False), dtype=np.float32)
    temporal = np.asarray(np.load(args.temporal_logits, allow_pickle=False), dtype=np.float32)
    skeleton_valid = np.asarray(
        np.load(args.skeleton_validity, allow_pickle=False), dtype=bool
    )
    if visual.shape != (len(test_ids), 40) or temporal.shape != visual.shape:
        raise RuntimeError("test branch logits are not aligned 40-class arrays")
    if skeleton_valid.shape != (len(test_ids),):
        raise RuntimeError("skeleton validity is not aligned with test rows")
    visual_valid = np.asarray(
        [
            has_decodable_image(args.test_root / test_id / "Depth_Color")
            or has_decodable_image(args.test_root / test_id / "IR")
            for test_id in test_ids
        ],
        dtype=bool,
    )
    both_valid = visual_valid & skeleton_valid

    package = torch.load(args.package, map_location="cpu", weights_only=True)
    release = package["release_contract"]
    control, candidate = gate_off_predictions(visual, temporal, both_valid, release)
    sample = pd.read_csv(args.sample_submission)
    expected_paths = [f"small_model_track_test/{test_id}/" for test_id in test_ids]
    if sample.columns.tolist() != ["path", "prediction"] or sample["path"].tolist() != expected_paths:
        raise RuntimeError("sample submission is not aligned with frozen test IDs")
    sample["prediction"] = candidate

    args.output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = args.output_dir / "submission.csv"
    predictions_path = args.output_dir / "predictions.npz"
    sample.to_csv(submission_path, index=False)
    np.savez_compressed(
        predictions_path,
        test_id=test_ids,
        visual_valid=visual_valid,
        skeleton_valid=skeleton_valid,
        both_primary_valid=both_valid,
        control_top1=control,
        candidate_top1=candidate,
    )
    report = {
        "schema_version": "cuhkx-quality-gate-off-submission/v1",
        "authorization": "explicit user override after the preregistered promotion gate failed",
        "promotion_gate_passed": False,
        "candidate": "fixed T/V base weights with confidence quality factors disabled",
        "rows": len(candidate),
        "classes": int(np.unique(candidate).size),
        "both_primary_valid": int(both_valid.sum()),
        "partial_primary": int((visual_valid ^ skeleton_valid).sum()),
        "no_primary": int((~visual_valid & ~skeleton_valid).sum()),
        "changed_vs_inherited_gate": int(np.sum(candidate != control)),
        "anonymous_test_labels_used": False,
        "decision_scope": (
            "This submission does not retroactively pass the OOF promotion gate and "
            "must not be described as an updated release baseline."
        ),
        "inputs": {
            "manifest": file_record(args.manifest),
            "package": file_record(args.package),
            "visual_logits": file_record(args.visual_logits),
            "temporal_logits": file_record(args.temporal_logits),
            "skeleton_validity": file_record(args.skeleton_validity),
            "sample_submission": file_record(args.sample_submission),
        },
        "outputs": {
            "submission": file_record(submission_path),
            "predictions": file_record(predictions_path),
        },
    }
    (args.output_dir / "deployment.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--visual-logits", type=Path, required=True)
    result.add_argument("--temporal-logits", type=Path, required=True)
    result.add_argument("--skeleton-validity", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--manifest", type=Path, default=RESULT_DIR / "release_manifest.json")
    result.add_argument("--metadata", type=Path, default=RESULT_DIR / "metadata.npz")
    result.add_argument("--package", type=Path, default=Path("checkpoints/strict_v3/model.pt"))
    result.add_argument(
        "--test-root", type=Path, default=DATA_DIR / "processed/test/small_model_track_test"
    )
    result.add_argument(
        "--sample-submission",
        type=Path,
        default=DATA_DIR / "Small-Model-Track/Testing/test_file/sample_submission.csv",
    )
    return result


if __name__ == "__main__":
    run(parser().parse_args())
