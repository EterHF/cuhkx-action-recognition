#!/usr/bin/env python3
"""Assemble and compare the preregistered input-validity masking OOF arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from yolo_r2plus1d.strict_v3.evaluation.visual_quantization import FOLDS
from yolo_r2plus1d.strict_v3.release.blend import apply_gate
from yolo_r2plus1d.strict_v3.training.conditional_corrector import (
    classification_metrics,
    prediction_delta,
)


def assemble(root: Path, filename: str, users: np.ndarray) -> np.ndarray:
    output = np.full((len(users), 40), np.nan, dtype=np.float32)
    for fold, held_users in FOLDS.items():
        rows = np.flatnonzero(np.isin(users, held_users))
        values = np.asarray(np.load(root / f"fold{fold}" / filename), dtype=np.float32)
        if values.shape != (len(rows), 40):
            raise ValueError(f"fold {fold} has misaligned logits {values.shape}")
        output[rows] = values
    if not np.isfinite(output).all():
        raise RuntimeError("assembled OOF contains missing or non-finite values")
    return output


def fuse(visual: np.ndarray, temporal: np.ndarray, release: dict) -> np.ndarray:
    weights = dict(release["weights"])
    weights["fusion"] = 0.0
    logits, _ = apply_gate(
        {
            "fusion": np.zeros_like(visual),
            "visual": visual * float(release["visual_package_output_scale"]),
            "temporal": temporal,
        },
        release["temperatures"],
        weights,
    )
    return np.asarray(logits, dtype=np.float32)


def subset_metrics(
    prediction: np.ndarray, labels: np.ndarray, masks: dict[str, np.ndarray]
) -> dict[str, dict[str, float | int]]:
    result = {}
    for name, mask in masks.items():
        correct = int(np.sum(prediction[mask] == labels[mask]))
        result[name] = {
            "rows": int(mask.sum()),
            "correct": correct,
            "accuracy": correct / int(mask.sum()),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--validity", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.metadata, allow_pickle=False) as metadata:
        labels = np.asarray(metadata["train_y"], dtype=np.int64)
        users = np.asarray(metadata["train_users"], dtype=np.int64)
    with np.load(args.validity, allow_pickle=False) as validity:
        visual_valid = np.asarray(validity["visual"], dtype=bool)
        skeleton_valid = np.asarray(validity["skeleton"], dtype=bool)
        all_primary = np.asarray(validity["all_primary"], dtype=bool)
        no_primary = np.asarray(validity["no_primary"], dtype=bool)
    package = torch.load(args.package, map_location="cpu", weights_only=True)
    release = package["release_contract"]
    arms = {}
    for arm in ("control", "masked"):
        arms[f"visual_{arm}"] = assemble(
            args.root / "visual" / arm, "best_val_logits.npy", users
        )
        arms[f"temporal_{arm}"] = assemble(
            args.root / "temporal" / arm, "val_logits.npy", users
        )
    candidates = {
        "control": fuse(arms["visual_control"], arms["temporal_control"], release),
        "visual_masked_only": fuse(
            arms["visual_masked"], arms["temporal_control"], release
        ),
        "temporal_masked_only": fuse(
            arms["visual_control"], arms["temporal_masked"], release
        ),
        "both_masked": fuse(arms["visual_masked"], arms["temporal_masked"], release),
    }
    masks = {
        "all": np.ones(len(labels), dtype=bool),
        "both_primary_valid": all_primary,
        "no_primary_input": no_primary,
        "visual_valid": visual_valid,
        "visual_invalid": ~visual_valid,
        "skeleton_valid": skeleton_valid,
        "skeleton_invalid": ~skeleton_valid,
    }
    predictions = {name: value.argmax(1) for name, value in candidates.items()}
    control = predictions["control"]
    output = {
        "schema_version": "cuhkx-input-validity-masking-oof/v1",
        "arms": {
            name: {
                "overall": classification_metrics(prediction, labels, users),
                "subsets": subset_metrics(prediction, labels, masks),
                "delta_vs_control": prediction_delta(control, prediction, labels),
                "fold_delta_vs_control": {
                    fold: prediction_delta(
                        control[np.isin(users, held)],
                        prediction[np.isin(users, held)],
                        labels[np.isin(users, held)],
                    )
                    for fold, held in FOLDS.items()
                },
            }
            for name, prediction in predictions.items()
        },
        "promotion_gate_passed": False,
        "held_labels_used_for_checkpoint_selection": False,
        "anonymous_test_accessed": False,
    }
    treatment = output["arms"]["both_masked"]
    baseline = output["arms"]["control"]
    output["promotion_gate_passed"] = bool(
        treatment["delta_vs_control"]["net"] > 0
        and sum(
            delta["net"] >= 0
            for delta in treatment["fold_delta_vs_control"].values()
        )
        >= 4
        and treatment["overall"]["subject_macro_accuracy"]
        >= baseline["overall"]["subject_macro_accuracy"]
        and treatment["subsets"]["both_primary_valid"]["accuracy"]
        >= baseline["subsets"]["both_primary_valid"]["accuracy"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
