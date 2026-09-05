#!/usr/bin/env python3
"""Evaluate one frozen Temporal-anchored public Depth/IR probability blend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from yolo_r2plus1d.strict_v3.paths import RESULT_DIR
from yolo_r2plus1d.strict_v3.training.highrate_cross_attention import FOLDS
from yolo_r2plus1d.strict_v3.training.public_backbone_screen import softmax_numpy


def scores(logits: np.ndarray, labels: np.ndarray, users: np.ndarray) -> dict:
    prediction = logits.argmax(1)
    per_user = {
        str(int(user)): float(np.mean(prediction[users == user] == labels[users == user]))
        for user in np.unique(users)
    }
    return {
        "correct": int(np.sum(prediction == labels)),
        "accuracy": float(np.mean(prediction == labels)),
        "worst_user_accuracy": min(per_user.values()),
        "user_accuracy": per_user,
    }


def component_directory(args: argparse.Namespace, fold: str, modality: str) -> Path:
    if fold == "A":
        return args.fold_a_depth if modality == "depth" else args.fold_a_ir
    return args.fold_root / f"fold{fold}" / modality


def run(args: argparse.Namespace) -> None:
    weights = np.asarray((args.temporal_weight, args.depth_weight, args.ir_weight))
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError("non-negative branch weights must sum to one")
    with np.load(args.metadata) as metadata:
        labels = metadata["train_y"]
        users = metadata["train_users"]
    temporal = np.load(args.temporal_oof)
    if temporal.shape != (len(labels), 40):
        raise RuntimeError("Temporal OOF shape mismatch")
    sensor = {
        "depth": np.full_like(temporal, np.nan),
        "ir": np.full_like(temporal, np.nan),
    }
    fold_rows = {}
    for fold, held_users in FOLDS.items():
        expected = np.flatnonzero(np.isin(users, held_users))
        fold_rows[fold] = expected
        for modality in sensor:
            directory = component_directory(args, fold, modality)
            indices = np.load(directory / "held_indices.npy")
            if not np.array_equal(indices, expected):
                raise RuntimeError(f"fold {fold} {modality} indices mismatch")
            metrics = json.loads((directory / "metrics.json").read_text())
            if metrics.get("project_checkpoint_loaded") is not False:
                raise RuntimeError(f"fold {fold} {modality} has invalid lineage")
            sensor[modality][indices] = np.load(directory / "held_logits.npy")
    if any(np.isnan(value).any() for value in sensor.values()):
        raise RuntimeError("sensor OOF is incomplete")
    temporal_probability = softmax_numpy(temporal)
    depth_probability = softmax_numpy(sensor["depth"])
    ir_probability = softmax_numpy(sensor["ir"])
    candidate = (
        weights[0] * temporal_probability
        + weights[1] * depth_probability
        + weights[2] * ir_probability
    )
    baseline_metrics = scores(temporal_probability, labels, users)
    candidate_metrics = scores(candidate, labels, users)
    folds = {}
    for fold, indices in fold_rows.items():
        baseline = scores(temporal_probability[indices], labels[indices], users[indices])
        treatment = scores(candidate[indices], labels[indices], users[indices])
        folds[fold] = {
            "baseline": baseline,
            "candidate": treatment,
            "correct_delta": treatment["correct"] - baseline["correct"],
        }
    passed = (
        candidate_metrics["correct"] >= baseline_metrics["correct"]
        and candidate_metrics["worst_user_accuracy"]
        >= baseline_metrics["worst_user_accuracy"]
        and all(value["correct_delta"] >= 0 for value in folds.values())
    )
    receipt = {
        "protocol": "fixed-temporal-public-sensor-probability-residual/v1",
        "weights": {
            "temporal": args.temporal_weight,
            "depth": args.depth_weight,
            "ir": args.ir_weight,
        },
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "folds": folds,
        "fold_a_reused_after_inspection": True,
        "folds_b_to_e_blind_confirmation": True,
        "weight_scan_performed": False,
        "anonymous_test_accessed": False,
        "promotion_gate_passed": passed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "depth_oof_logits.npy", sensor["depth"])
    np.save(args.output_dir / "ir_oof_logits.npy", sensor["ir"])
    np.save(args.output_dir / "candidate_oof_probabilities.npy", candidate)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--temporal-oof", type=Path, required=True)
    result.add_argument("--fold-a-depth", type=Path, required=True)
    result.add_argument("--fold-a-ir", type=Path, required=True)
    result.add_argument("--fold-root", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--metadata", type=Path, default=RESULT_DIR / "metadata.npz")
    result.add_argument("--temporal-weight", type=float, default=0.9)
    result.add_argument("--depth-weight", type=float, default=0.05)
    result.add_argument("--ir-weight", type=float, default=0.05)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
