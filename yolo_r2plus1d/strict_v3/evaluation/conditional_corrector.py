#!/usr/bin/env python3
"""Summarize strict nested-OOF conditional-corrector artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from yolo_r2plus1d.strict_v3.paths import RESULT_DIR
from yolo_r2plus1d.strict_v3.training.conditional_corrector import (
    classification_metrics,
    paired_exact_pvalue,
    prediction_delta,
    subject_bootstrap_interval,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def comparison(
    reference: np.ndarray,
    candidate: np.ndarray,
    labels: np.ndarray,
    users: np.ndarray,
    seed: int,
) -> dict:
    delta = prediction_delta(reference, candidate, labels)
    return {
        "delta": delta,
        "paired_exact_pvalue": paired_exact_pvalue(delta["corrected"], delta["broken"]),
        "subject_bootstrap_accuracy_delta_95ci": subject_bootstrap_interval(
            reference, candidate, labels, users, seed
        ),
    }


def run(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    corrector = json.loads(args.corrector_metrics.read_text(encoding="utf-8"))
    with np.load(args.metadata) as metadata:
        labels = metadata["train_y"].astype(np.int64)
        users = metadata["train_users"].astype(np.int64)
    corrected_logits = np.asarray(np.load(args.corrected_oof), dtype=np.float32)
    baseline_logits = np.asarray(np.load(args.baseline_oof), dtype=np.float32)
    visual_logits = np.empty_like(baseline_logits)
    temporal_logits = np.empty_like(baseline_logits)
    assigned = np.zeros(len(labels), dtype=bool)
    for fold in manifest["folds"]:
        record = fold["held"]
        indices = np.asarray(np.load(record["indices"]), dtype=np.int64)
        if bool(assigned[indices].any()):
            raise RuntimeError("outer-held rows overlap")
        assigned[indices] = True
        visual_logits[indices] = np.load(record["visual_logits"])
        temporal_logits[indices] = np.load(record["temporal_logits"])
    if not bool(assigned.all()):
        raise RuntimeError("outer-held rows do not cover the dataset")
    predictions = {
        "visual": visual_logits.argmax(1),
        "temporal": temporal_logits.argmax(1),
        "baseline": baseline_logits.argmax(1),
        "candidate": corrected_logits.argmax(1),
    }
    per_user_delta = {
        str(int(user)): int(
            np.sum(predictions["candidate"][users == user] == labels[users == user])
            - np.sum(predictions["baseline"][users == user] == labels[users == user])
        )
        for user in np.unique(users)
    }
    output = {
        "protocol": "conditional-corrector-nested-oof/v1",
        "interpretation": "strict external-only research proxy; not a strictV3 deployment score",
        "rows": len(labels),
        "branches": {
            name: classification_metrics(value, labels, users)
            for name, value in predictions.items()
        },
        "candidate_vs_baseline": comparison(
            predictions["baseline"], predictions["candidate"], labels, users, args.seed
        ),
        "candidate_vs_visual": comparison(
            predictions["visual"], predictions["candidate"], labels, users, args.seed
        ),
        "candidate_vs_temporal": comparison(
            predictions["temporal"], predictions["candidate"], labels, users, args.seed
        ),
        "folds": corrector["folds"],
        "all_folds_positive": all(item["delta"]["net"] > 0 for item in corrector["folds"]),
        "per_user_net_correct": per_user_delta,
        "all_users_positive": all(value > 0 for value in per_user_delta.values()),
        "protected_train_rows_per_fold": [
            item["protected_train_rows"] for item in corrector["folds"]
        ],
        "limitations": [
            "External-only proxy branch accuracies are much lower than deployed strictV3.",
            "Frozen strictV3 Temporal-dominant fusion weights are mismatched to this proxy, where Visual is stronger.",
            "Few inner-training baseline rows exceed the preregistered 0.8 KL-protection threshold.",
            "The result validates the correction mechanism under strict isolation but does not estimate Kaggle gain.",
        ],
        "reproducibility": {
            "manifest_sha256": sha256(args.manifest),
            "corrected_oof_sha256": sha256(args.corrected_oof),
            "baseline_oof_sha256": sha256(args.baseline_oof),
            "fixed_epochs": corrector["fixed_epochs"],
            "seed": corrector["seed"],
        },
        "held_labels_used_for_checkpoint_selection": False,
        "anonymous_test_accessed": False,
        "kaggle_submitted": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--corrector-metrics", type=Path, required=True)
    result.add_argument("--corrected-oof", type=Path, required=True)
    result.add_argument("--baseline-oof", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--metadata", type=Path, default=RESULT_DIR / "metadata.npz")
    result.add_argument("--seed", type=int, default=2026)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
