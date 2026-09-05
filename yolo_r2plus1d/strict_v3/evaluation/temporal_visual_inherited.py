#!/usr/bin/env python3
"""Remove strictV3 Fusion while inheriting its frozen Temporal/Visual contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from yolo_r2plus1d.strict_v3.evaluation.temporal_visual_equal import checked_load, sha256
from yolo_r2plus1d.strict_v3.paths import RESULT_DIR
from yolo_r2plus1d.strict_v3.release.blend import apply_gate


def inherited_logits(
    visual: np.ndarray,
    temporal: np.ndarray,
    temperatures: dict[str, float],
    base_weights: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    weights = dict(base_weights)
    weights["fusion"] = 0.0
    return apply_gate(
        {"fusion": np.zeros_like(visual), "visual": visual, "temporal": temporal},
        temperatures,
        weights,
    )


def classification_metrics(
    prediction: np.ndarray, labels: np.ndarray, users: np.ndarray
) -> dict:
    class_accuracy = [
        float(np.mean(prediction[labels == label] == label)) for label in np.unique(labels)
    ]
    user_accuracy = {
        str(int(user)): float(np.mean(prediction[users == user] == labels[users == user]))
        for user in np.unique(users)
    }
    return {
        "correct": int(np.sum(prediction == labels)),
        "accuracy": float(np.mean(prediction == labels)),
        "macro_recall": float(np.mean(class_accuracy)),
        "subject_macro_accuracy": float(np.mean(list(user_accuracy.values()))),
        "worst_user_accuracy": min(user_accuracy.values()),
    }


def run(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    with np.load(args.metadata) as metadata:
        labels = metadata["train_y"]
        users = metadata["train_users"]
    candidate_oof = np.empty((len(labels), 40), dtype=np.float32)
    strict_oof = np.empty_like(candidate_oof)
    fold_metrics = []
    for contract in manifest["fold_contracts"]:
        fold = contract["fold"]
        source = next(item for item in manifest["oof_sources"] if item["fold"] == fold)
        indices = np.flatnonzero(np.isin(users, contract["held_users"]))
        visual = checked_load(
            args.source_root
            / "best_release/legal_strict_v3/preproc_affine"
            / f"visual_fold{fold}_train.npy",
            source["source_sha256"]["visual_oof"],
        )[indices]
        temporal = checked_load(
            args.source_root
            / "runs/public_cuhkx_dstformer"
            / f"logit_residual_tcn_cv_{fold}/val_logits.npy",
            source["source_sha256"]["temporal_oof"],
        )
        fusion = checked_load(
            args.source_root
            / "best_release/legal_strict_v3/preproc_affine"
            / f"fusion_fold{fold}_train.npy",
            source["source_sha256"]["fusion_oof"],
        )[indices]
        candidate, _ = inherited_logits(
            visual, temporal, contract["temperatures"], contract["weights"]
        )
        strict, _ = apply_gate(
            {"fusion": fusion, "visual": visual, "temporal": temporal},
            contract["temperatures"],
            contract["weights"],
        )
        candidate_oof[indices] = candidate
        strict_oof[indices] = strict
        candidate_correct = int(np.sum(candidate.argmax(1) == labels[indices]))
        strict_correct = int(np.sum(strict.argmax(1) == labels[indices]))
        fold_metrics.append(
            {
                "fold": fold,
                "rows": len(indices),
                "candidate_correct": candidate_correct,
                "strictv3_correct": strict_correct,
                "delta": candidate_correct - strict_correct,
                "candidate_accuracy": candidate_correct / len(indices),
            }
        )

    test_root = args.source_root / "best_release/legal_strict_v3/selfcontained_baseline_replay_work"
    visual_test = checked_load(
        test_root / "visual_logits.npy", manifest["test_source_sha256"]["visual"]
    )
    temporal_test = checked_load(
        test_root / "temporal_logits.npy", manifest["test_source_sha256"]["temporal"]
    )
    test_logits, test_weights = inherited_logits(
        visual_test,
        temporal_test,
        manifest["blend"]["temperatures"],
        manifest["blend"]["weights"],
    )
    prediction = test_logits.argmax(1)
    sample = pd.read_csv(args.sample_submission)
    if sample.columns.tolist() != ["path", "prediction"] or len(sample) != len(prediction):
        raise RuntimeError("sample submission schema mismatch")
    sample["prediction"] = prediction
    strict_prediction = pd.read_csv(RESULT_DIR / "submission.csv")["prediction"].to_numpy()
    metrics = {
        "protocol": "strictv3-temporal-visual-inherited-gate/v1",
        "change": "set Fusion base weight to zero; inherit all other release settings",
        "candidate_oof": classification_metrics(candidate_oof.argmax(1), labels, users),
        "strictv3_reconstructed_oof": classification_metrics(
            strict_oof.argmax(1), labels, users
        ),
        "folds": fold_metrics,
        "all_folds_non_degrading": all(item["delta"] >= 0 for item in fold_metrics),
        "test": {
            "rows": len(prediction),
            "classes": int(np.unique(prediction).size),
            "changed_vs_strictv3": int(np.sum(prediction != strict_prediction)),
            "mean_effective_visual_weight": float(test_weights[:, 1].mean()),
            "effective_visual_weight_range": [
                float(test_weights[:, 1].min()),
                float(test_weights[:, 1].max()),
            ],
        },
        "bundle": {
            "path": str(args.bundle),
            "bytes": args.bundle.stat().st_size,
            "sha256": sha256(args.bundle),
        },
        "anonymous_test_labels_used": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.output_dir / "submission.csv", index=False)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--source-root", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--manifest", type=Path, default=RESULT_DIR / "release_manifest.json")
    result.add_argument("--metadata", type=Path, default=RESULT_DIR / "metadata.npz")
    result.add_argument(
        "--sample-submission",
        type=Path,
        default=Path("data/Small-Model-Track/Testing/test_file/sample_submission.csv"),
    )
    result.add_argument(
        "--bundle", type=Path, default=Path("checkpoints/strict_v3/submission_bundle.pt")
    )
    return result


if __name__ == "__main__":
    run(parser().parse_args())
