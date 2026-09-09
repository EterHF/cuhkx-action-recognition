#!/usr/bin/env python3
"""Describe current 2,889-row T+V OOF errors without fitting a new rule."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from yolo_r2plus1d.strict_v3.evaluation.input_validity_masking import assemble, fuse


def true_class_rank(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return one-based competition rank of each row's true-class logit."""
    truth = logits[np.arange(len(labels)), labels]
    return 1 + np.sum(logits > truth[:, None], axis=1)


def delta_counts(
    baseline: np.ndarray, candidate: np.ndarray, labels: np.ndarray
) -> dict[str, int]:
    return {
        "corrected": int(np.sum((baseline != labels) & (candidate == labels))),
        "broken": int(np.sum((baseline == labels) & (candidate != labels))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--validity", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--visual-root", type=Path, required=True)
    parser.add_argument("--current-root", type=Path, required=True)
    parser.add_argument("--validity-root", type=Path, required=True)
    parser.add_argument("--retarget-root", type=Path, required=True)
    parser.add_argument("--contrast-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.metadata, allow_pickle=False) as metadata:
        labels = np.asarray(metadata["train_y"], dtype=np.int64)
        users = np.asarray(metadata["train_users"], dtype=np.int64)
        keys = np.asarray(metadata["train_keys"])
    with np.load(args.validity, allow_pickle=False) as validity:
        both_valid = np.asarray(validity["all_primary"], dtype=bool)
        no_primary = np.asarray(validity["no_primary"], dtype=bool)
        visual_valid = np.asarray(validity["visual"], dtype=bool)
        skeleton_valid = np.asarray(validity["skeleton"], dtype=bool)
    release = torch.load(
        args.package, map_location="cpu", weights_only=True
    )["release_contract"]

    visual = assemble(args.visual_root / "control", "best_val_logits.npy", users)
    temporal = assemble(args.current_root / "control", "val_logits.npy", users)
    baseline_logits = fuse(visual, temporal, release)
    baseline = baseline_logits.argmax(1)
    if int(np.sum(baseline == labels)) != 2889:
        raise RuntimeError("current baseline is not the locked 2,889/3,036 OOF")

    validity_control_visual = assemble(
        args.validity_root / "visual" / "control", "best_val_logits.npy", users
    )
    validity_control_temporal = assemble(
        args.validity_root / "temporal" / "control", "val_logits.npy", users
    )
    if not np.array_equal(
        fuse(validity_control_visual, validity_control_temporal, release).argmax(1),
        baseline,
    ):
        raise RuntimeError("historical validity control predictions differ from baseline")
    validity_visual = assemble(
        args.validity_root / "visual" / "masked", "best_val_logits.npy", users
    )
    validity_temporal = assemble(
        args.validity_root / "temporal" / "masked", "val_logits.npy", users
    )
    candidate_logits = {
        "visual_masked_only": fuse(
            validity_visual, validity_control_temporal, release
        ),
        "temporal_masked_only": fuse(
            validity_control_visual, validity_temporal, release
        ),
        "both_masked": fuse(validity_visual, validity_temporal, release),
        "skeleton_retargeting": fuse(
            visual,
            assemble(args.retarget_root, "val_logits.npy", users),
            release,
        ),
        "cross_user_contrast": fuse(
            visual,
            assemble(args.contrast_root, "val_logits.npy", users),
            release,
        ),
        "prelogit_feature_tcn": fuse(
            visual,
            assemble(args.feature_root, "val_logits.npy", users),
            release,
        ),
    }
    candidates = {name: values.argmax(1) for name, values in candidate_logits.items()}

    temporal_prediction = temporal.argmax(1)
    visual_prediction = visual.argmax(1)
    temporal_rank = true_class_rank(temporal, labels)
    visual_rank = true_class_rank(visual, labels)
    errors = baseline != labels
    nonmissing_errors = errors & ~no_primary
    both_wrong = nonmissing_errors & (temporal_prediction != labels) & (
        visual_prediction != labels
    )
    error_type = np.full(len(labels), "baseline_correct", dtype=object)
    error_type[errors & no_primary] = "no_primary_input"
    error_type[
        nonmissing_errors
        & (temporal_prediction == labels)
        & (visual_prediction != labels)
    ] = "temporal_top1_only"
    error_type[
        nonmissing_errors
        & (visual_prediction == labels)
        & (temporal_prediction != labels)
    ] = "visual_top1_only"
    error_type[both_wrong] = "both_top1_wrong"
    unclassified = errors & (error_type == "baseline_correct")
    if unclassified.any():
        raise RuntimeError(f"unclassified baseline errors: {int(unclassified.sum())}")

    corrected_by = {
        name: errors & (prediction == labels) for name, prediction in candidates.items()
    }
    ever_corrected = np.logical_or.reduce(list(corrected_by.values())) & errors
    persistent = errors & ~ever_corrected

    def rank_summary(mask: np.ndarray) -> dict[str, int]:
        def histogram(values: np.ndarray) -> dict[str, int]:
            unique, counts = np.unique(values[mask], return_counts=True)
            return {
                str(int(rank)): int(count)
                for rank, count in zip(unique, counts, strict=True)
            }

        return {
            "rows": int(mask.sum()),
            "temporal_true_rank_histogram": histogram(temporal_rank),
            "visual_true_rank_histogram": histogram(visual_rank),
            "either_branch_true_rank_le_2": int(
                np.sum(mask & ((temporal_rank <= 2) | (visual_rank <= 2)))
            ),
            "either_branch_true_rank_le_3": int(
                np.sum(mask & ((temporal_rank <= 3) | (visual_rank <= 3)))
            ),
            "either_branch_true_rank_le_5": int(
                np.sum(mask & ((temporal_rank <= 5) | (visual_rank <= 5)))
            ),
            "both_branches_true_rank_gt_5": int(
                np.sum(mask & (temporal_rank > 5) & (visual_rank > 5))
            ),
            "both_branches_true_rank_gt_10": int(
                np.sum(mask & (temporal_rank > 10) & (visual_rank > 10))
            ),
        }

    report = {
        "schema_version": "cuhkx-current-tv-error-audit/v1",
        "baseline": {
            "correct": int(np.sum(~errors)),
            "errors": int(np.sum(errors)),
            "rows": len(labels),
        },
        "input_validity": {
            "no_primary_rows": int(no_primary.sum()),
            "no_primary_errors": int(np.sum(errors & no_primary)),
            "remaining_errors": int(np.sum(nonmissing_errors)),
            "both_primary_valid_errors": int(np.sum(errors & both_valid)),
            "partial_primary_errors": int(
                np.sum(errors & ~no_primary & ~(visual_valid & skeleton_valid))
            ),
        },
        "remaining_error_top1_coverage": {
            "temporal_top1_only": int(np.sum(error_type == "temporal_top1_only")),
            "visual_top1_only": int(np.sum(error_type == "visual_top1_only")),
            "both_top1_wrong": int(np.sum(error_type == "both_top1_wrong")),
        },
        "both_top1_wrong_true_rank": rank_summary(both_wrong),
        "historical_candidate_priority": {
            "candidate_names": list(candidates),
            "ever_corrected": int(ever_corrected.sum()),
            "persistent": int(persistent.sum()),
            "per_candidate_all_rows": {
                name: delta_counts(baseline, prediction, labels)
                for name, prediction in candidates.items()
            },
            "per_candidate_baseline_error_corrections": {
                name: int(mask.sum()) for name, mask in corrected_by.items()
            },
            "ever_corrected_by_error_type": {
                name: int(np.sum(ever_corrected & (error_type == name)))
                for name in (
                    "no_primary_input",
                    "temporal_top1_only",
                    "visual_top1_only",
                    "both_top1_wrong",
                )
            },
            "persistent_by_error_type": {
                name: int(np.sum(persistent & (error_type == name)))
                for name in (
                    "no_primary_input",
                    "temporal_top1_only",
                    "visual_top1_only",
                    "both_top1_wrong",
                )
            },
            "not_an_ensemble_or_training_target": True,
        },
        "notes": {
            "top1_coverage_is_not_a_soft_fusion_upper_bound": True,
            "descriptive_only_no_threshold_or_weight_scan": True,
            "anonymous_test_accessed": False,
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "row",
                "key",
                "label",
                "user",
                "error_type",
                "visual_top1",
                "visual_true_rank",
                "temporal_top1",
                "temporal_true_rank",
                "fused_top1",
                "visual_valid",
                "skeleton_valid",
                "ever_corrected",
                "corrected_by",
            ],
        )
        writer.writeheader()
        for row in np.flatnonzero(errors):
            writer.writerow(
                {
                    "row": int(row),
                    "key": str(keys[row]),
                    "label": int(labels[row]),
                    "user": int(users[row]),
                    "error_type": error_type[row],
                    "visual_top1": int(visual_prediction[row]),
                    "visual_true_rank": int(visual_rank[row]),
                    "temporal_top1": int(temporal_prediction[row]),
                    "temporal_true_rank": int(temporal_rank[row]),
                    "fused_top1": int(baseline[row]),
                    "visual_valid": bool(visual_valid[row]),
                    "skeleton_valid": bool(skeleton_valid[row]),
                    "ever_corrected": bool(ever_corrected[row]),
                    "corrected_by": ";".join(
                        name for name, mask in corrected_by.items() if mask[row]
                    ),
                }
            )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
