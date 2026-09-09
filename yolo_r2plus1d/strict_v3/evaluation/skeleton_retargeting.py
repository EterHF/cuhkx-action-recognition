#!/usr/bin/env python3
"""Evaluate the preregistered mild skeleton-retargeting OOF pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from yolo_r2plus1d.strict_v3.evaluation.input_validity_masking import (
    assemble,
    fuse,
    subset_metrics,
)
from yolo_r2plus1d.strict_v3.evaluation.visual_quantization import FOLDS
from yolo_r2plus1d.strict_v3.training.conditional_corrector import (
    classification_metrics,
    prediction_delta,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--candidate-name", default="retargeted")
    parser.add_argument("--visual-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--validity", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.metadata, allow_pickle=False) as metadata:
        labels = np.asarray(metadata["train_y"], dtype=np.int64)
        users = np.asarray(metadata["train_users"], dtype=np.int64)
    with np.load(args.validity, allow_pickle=False) as validity:
        masks = {
            "all": np.ones(len(labels), dtype=bool),
            "both_primary_valid": np.asarray(validity["all_primary"], dtype=bool),
            "no_primary_input": np.asarray(validity["no_primary"], dtype=bool),
        }
    package = torch.load(args.package, map_location="cpu", weights_only=True)
    release = package["release_contract"]
    visual = assemble(args.visual_root, "best_val_logits.npy", users)
    candidate_root = (
        args.root / "temporal" / args.candidate_name
        if args.candidate_root is None
        else args.candidate_root
    )
    temporal = {
        "control": assemble(
            args.root / "temporal" / "control", "val_logits.npy", users
        ),
        args.candidate_name: assemble(candidate_root, "val_logits.npy", users),
    }
    fused = {arm: fuse(visual, values, release) for arm, values in temporal.items()}
    control = fused["control"].argmax(1)
    temporal_control = temporal["control"].argmax(1)
    arms = {}
    for arm, logits in fused.items():
        prediction = logits.argmax(1)
        temporal_prediction = temporal[arm].argmax(1)
        arms[arm] = {
            "temporal_correct": int(np.sum(temporal_prediction == labels)),
            "temporal_overall": classification_metrics(
                temporal_prediction, labels, users
            ),
            "temporal_delta_vs_control": prediction_delta(
                temporal_control, temporal_prediction, labels
            ),
            "overall": classification_metrics(prediction, labels, users),
            "subsets": subset_metrics(prediction, labels, masks),
            "delta_vs_control": prediction_delta(control, prediction, labels),
            "subset_delta_vs_control": {
                name: prediction_delta(
                    control[mask], prediction[mask], labels[mask]
                )
                for name, mask in masks.items()
            },
            "user_delta_vs_control": {
                str(user): prediction_delta(
                    control[users == user],
                    prediction[users == user],
                    labels[users == user],
                )
                for user in np.unique(users)
            },
            "fold_delta_vs_control": {
                fold: prediction_delta(
                    control[np.isin(users, held)],
                    prediction[np.isin(users, held)],
                    labels[np.isin(users, held)],
                )
                for fold, held in FOLDS.items()
            },
        }
    candidate, baseline = arms[args.candidate_name], arms["control"]
    gate = bool(
        candidate["delta_vs_control"]["net"] > 0
        and sum(
            delta["net"] >= 0
            for delta in candidate["fold_delta_vs_control"].values()
        )
        >= 4
        and candidate["overall"]["subject_macro_accuracy"]
        >= baseline["overall"]["subject_macro_accuracy"]
        and candidate["subsets"]["both_primary_valid"]["accuracy"]
        >= baseline["subsets"]["both_primary_valid"]["accuracy"]
    )
    output = {
        "schema_version": "cuhkx-temporal-paired-oof/v1",
        "candidate_name": args.candidate_name,
        "arms": arms,
        "promotion_gate_passed": gate,
        "held_labels_used_for_checkpoint_selection": False,
        "anonymous_test_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
