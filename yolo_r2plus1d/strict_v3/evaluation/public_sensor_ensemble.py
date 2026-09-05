#!/usr/bin/env python3
"""Evaluate a fixed equal-logit ensemble of independently trained branches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from yolo_r2plus1d.strict_v3.paths import RESULT_DIR


def run(args: argparse.Namespace) -> None:
    component_dirs = tuple(args.component_dir)
    if len(component_dirs) < 2:
        raise ValueError("at least two independent components are required")
    indices = np.load(component_dirs[0] / "held_indices.npy")
    logits = []
    components = []
    for directory in component_dirs:
        candidate_indices = np.load(directory / "held_indices.npy")
        if not np.array_equal(indices, candidate_indices):
            raise RuntimeError("component held indices differ")
        component_metrics = json.loads((directory / "metrics.json").read_text())
        if component_metrics.get("anonymous_test_accessed") is not False:
            raise RuntimeError(f"invalid component lineage: {directory}")
        logits.append(np.load(directory / "held_logits.npy"))
        components.append(
            {
                "path": str(directory),
                "accuracy": component_metrics["accuracy"],
                "worst_user_accuracy": component_metrics["worst_user_accuracy"],
            }
        )
    ensemble_logits = np.mean(logits, axis=0)
    with np.load(args.metadata) as metadata:
        labels = metadata["train_y"][indices]
        users = metadata["train_users"][indices]
    prediction = ensemble_logits.argmax(1)
    user_accuracy = {
        str(int(user)): float(np.mean(prediction[users == user] == labels[users == user]))
        for user in np.unique(users)
    }
    metrics = {
        "protocol": "equal-logit-independent-public-sensor-ensemble/v1",
        "components": components,
        "weights": [1 / len(components)] * len(components),
        "held_rows": len(indices),
        "accuracy": float(np.mean(prediction == labels)),
        "worst_user_accuracy": min(user_accuracy.values()),
        "user_accuracy": user_accuracy,
        "held_fold_reused_after_component_inspection": True,
        "selection_unbiased": False,
        "anonymous_test_accessed": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "held_indices.npy", indices)
    np.save(args.output_dir / "held_logits.npy", ensemble_logits)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--component-dir", type=Path, action="append", required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--metadata", type=Path, default=RESULT_DIR / "metadata.npz")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
