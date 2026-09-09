#!/usr/bin/env python3
"""Ablate confidence quality factors while freezing the T/V base contract."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from yolo_r2plus1d.strict_v3.evaluation.input_validity_masking import assemble
from yolo_r2plus1d.strict_v3.evaluation.visual_quantization import FOLDS
from yolo_r2plus1d.strict_v3.release.blend import apply_gate
from yolo_r2plus1d.strict_v3.training.conditional_corrector import (
    classification_metrics,
    prediction_delta,
)


def fixed_quality_logits(
    visual: np.ndarray,
    temporal: np.ndarray,
    temperatures: dict[str, float],
    base_weights: dict[str, float],
) -> np.ndarray:
    """Fuse with unchanged base weights and equal, cancelling quality factors."""
    visual_weight = float(base_weights["visual"])
    temporal_weight = float(base_weights["temporal"])
    total = visual_weight + temporal_weight
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Temporal and Visual base weights must have a positive sum")
    return (
        visual_weight * visual / max(float(temperatures["visual"]), 1e-4)
        + temporal_weight * temporal / max(float(temperatures["temporal"]), 1e-4)
    ) / total


def exact_mcnemar_p(corrected: int, broken: int) -> float:
    changes = corrected + broken
    if changes == 0:
        return 1.0
    tail = sum(math.comb(changes, index) for index in range(min(corrected, broken) + 1))
    return min(1.0, 2.0 * tail / (2**changes))


def clustered_bootstrap(
    control: np.ndarray,
    candidate: np.ndarray,
    labels: np.ndarray,
    users: np.ndarray,
    seed: int,
    replicates: int,
) -> dict[str, float | int | list[float]]:
    unique_users = np.unique(users)
    indices = {user: np.flatnonzero(users == user) for user in unique_users}
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = rng.choice(unique_users, len(unique_users), replace=True)
        rows = np.concatenate([indices[user] for user in sampled])
        deltas[replicate] = np.mean(candidate[rows] == labels[rows]) - np.mean(
            control[rows] == labels[rows]
        )
    interval = np.quantile(deltas, (0.025, 0.975))
    return {
        "seed": seed,
        "replicates": replicates,
        "accuracy_delta_95_percentile_interval": [
            float(interval[0]),
            float(interval[1]),
        ],
        "probability_delta_non_positive": float(np.mean(deltas <= 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--validity", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--visual-root", type=Path, required=True)
    parser.add_argument("--temporal-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--changes-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()

    with np.load(args.metadata, allow_pickle=False) as metadata:
        labels = np.asarray(metadata["train_y"], dtype=np.int64)
        users = np.asarray(metadata["train_users"], dtype=np.int64)
        keys = np.asarray(metadata["train_keys"])
    with np.load(args.validity, allow_pickle=False) as validity:
        both_valid = np.asarray(validity["all_primary"], dtype=bool)
        no_primary = np.asarray(validity["no_primary"], dtype=bool)
    release = torch.load(
        args.package, map_location="cpu", weights_only=True
    )["release_contract"]
    visual = assemble(args.visual_root, "best_val_logits.npy", users)
    temporal = assemble(args.temporal_root, "val_logits.npy", users)
    scaled_visual = visual * float(release["visual_package_output_scale"])
    base_weights = dict(release["weights"])
    base_weights["fusion"] = 0.0
    branches = {
        "fusion": np.zeros_like(visual),
        "visual": scaled_visual,
        "temporal": temporal,
    }
    control_logits, effective_weights = apply_gate(
        branches, release["temperatures"], base_weights
    )
    fixed_logits = fixed_quality_logits(
        scaled_visual, temporal, release["temperatures"], base_weights
    )
    candidate_logits = np.array(control_logits, copy=True)
    candidate_logits[both_valid] = fixed_logits[both_valid]
    control = control_logits.argmax(1)
    candidate = candidate_logits.argmax(1)
    if int(np.sum(control == labels)) != 2889:
        raise RuntimeError("control is not the locked 2,889/3,036 paired OOF")
    if not np.array_equal(candidate[~both_valid], control[~both_valid]):
        raise RuntimeError("missing or partial-primary predictions changed")

    visual_prediction = visual.argmax(1)
    temporal_prediction = temporal.argmax(1)
    visual_margin = np.sort(visual, axis=1)[:, -1] - np.sort(visual, axis=1)[:, -2]
    temporal_margin = np.sort(temporal, axis=1)[:, -1] - np.sort(temporal, axis=1)[:, -2]
    same_unique_top1 = (
        both_valid
        & (visual_prediction == temporal_prediction)
        & (visual_margin > 0.0)
        & (temporal_margin > 0.0)
    )
    changed = candidate != control
    same_top1_changes = int(np.sum(changed & same_unique_top1))
    if same_top1_changes:
        raise RuntimeError("gate ablation changed a same-unique-top1 row")

    baseline_error = control != labels
    error_types = {
        "no_primary_input": baseline_error & no_primary,
        "temporal_top1_only": (
            baseline_error
            & ~no_primary
            & (temporal_prediction == labels)
            & (visual_prediction != labels)
        ),
        "visual_top1_only": (
            baseline_error
            & ~no_primary
            & (visual_prediction == labels)
            & (temporal_prediction != labels)
        ),
        "both_top1_wrong": (
            baseline_error
            & ~no_primary
            & (visual_prediction != labels)
            & (temporal_prediction != labels)
        ),
    }
    delta = prediction_delta(control, candidate, labels)
    fold_delta = {
        fold: prediction_delta(
            control[np.isin(users, held)],
            candidate[np.isin(users, held)],
            labels[np.isin(users, held)],
        )
        for fold, held in FOLDS.items()
    }
    user_delta = {
        str(int(user)): prediction_delta(
            control[users == user], candidate[users == user], labels[users == user]
        )
        for user in np.unique(users)
    }
    control_metrics = classification_metrics(control, labels, users)
    candidate_metrics = classification_metrics(candidate, labels, users)
    bootstrap = clustered_bootstrap(
        control, candidate, labels, users, args.seed, args.bootstrap_replicates
    )
    fixed_visual_weight = float(base_weights["visual"]) / (
        float(base_weights["visual"]) + float(base_weights["temporal"])
    )
    corrected_mask = baseline_error & (candidate == labels)
    broken_mask = (control == labels) & (candidate != labels)
    report = {
        "schema_version": "cuhkx-quality-gate-ablation/v1",
        "exploratory_reused_oof": True,
        "contract": {
            "base_weights": base_weights,
            "temperatures": release["temperatures"],
            "visual_package_output_scale": release["visual_package_output_scale"],
            "candidate_fixed_effective_visual_weight": fixed_visual_weight,
            "control_effective_visual_weight_both_valid": {
                "mean": float(effective_weights[both_valid, 1].mean()),
                "minimum": float(effective_weights[both_valid, 1].min()),
                "maximum": float(effective_weights[both_valid, 1].max()),
            },
            "changed_rows": "both-primary-valid only",
            "fitted_parameters": 0,
        },
        "control": control_metrics,
        "candidate": candidate_metrics,
        "paired_delta": delta,
        "exact_two_sided_mcnemar_p": exact_mcnemar_p(
            delta["corrected"], delta["broken"]
        ),
        "fold_delta": fold_delta,
        "user_delta": user_delta,
        "baseline_error_breakdown": {
            name: {
                "rows": int(mask.sum()),
                "corrected": int(np.sum(mask & corrected_mask)),
            }
            for name, mask in error_types.items()
        },
        "cost_on_originally_correct": {
            "all_originally_correct_rows": int(np.sum(control == labels)),
            "both_primary_valid_originally_correct_rows": int(
                np.sum((control == labels) & both_valid)
            ),
            "broken": int(broken_mask.sum()),
        },
        "consistency": {
            "changed_predictions": int(changed.sum()),
            "changed_on_branch_disagreement": int(
                np.sum(changed & (visual_prediction != temporal_prediction))
            ),
            "same_unique_branch_top1_rows": int(same_unique_top1.sum()),
            "same_unique_branch_top1_changes": same_top1_changes,
            "non_both_valid_changes": int(np.sum(changed & ~both_valid)),
        },
        "user_cluster_bootstrap": bootstrap,
        "promotion_gate_passed": bool(
            delta["net"] > 0
            and sum(item["net"] >= 0 for item in fold_delta.values()) >= 4
            and candidate_metrics["subject_macro_accuracy"]
            >= control_metrics["subject_macro_accuracy"]
            and candidate_metrics["worst_user_accuracy"]
            >= control_metrics["worst_user_accuracy"]
            and np.mean(candidate[both_valid] == labels[both_valid])
            >= np.mean(control[both_valid] == labels[both_valid])
            and bootstrap["accuracy_delta_95_percentile_interval"][0] > 0.0
            and same_top1_changes == 0
        ),
        "anonymous_test_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.changes_output.parent.mkdir(parents=True, exist_ok=True)
    with args.changes_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "row",
                "key",
                "user",
                "label",
                "control_top1",
                "candidate_top1",
                "temporal_top1",
                "visual_top1",
                "outcome",
                "control_effective_visual_weight",
                "candidate_effective_visual_weight",
            ],
        )
        writer.writeheader()
        for row in np.flatnonzero(changed):
            writer.writerow(
                {
                    "row": int(row),
                    "key": str(keys[row]),
                    "user": int(users[row]),
                    "label": int(labels[row]),
                    "control_top1": int(control[row]),
                    "candidate_top1": int(candidate[row]),
                    "temporal_top1": int(temporal_prediction[row]),
                    "visual_top1": int(visual_prediction[row]),
                    "outcome": "corrected" if corrected_mask[row] else "broken",
                    "control_effective_visual_weight": float(
                        effective_weights[row, 1]
                    ),
                    "candidate_effective_visual_weight": fixed_visual_weight,
                }
            )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
