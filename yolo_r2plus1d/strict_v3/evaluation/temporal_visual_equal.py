#!/usr/bin/env python3
"""Build the fixed equal-weight strictV3 Temporal+Visual ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from yolo_r2plus1d.strict_v3.paths import RESULT_DIR


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def equal_tempered_logits(
    visual: np.ndarray,
    temporal: np.ndarray,
    visual_temperature: float,
    temporal_temperature: float,
) -> np.ndarray:
    return 0.5 * visual / visual_temperature + 0.5 * temporal / temporal_temperature


def checked_load(path: Path, expected_sha256: str) -> np.ndarray:
    observed = sha256(path)
    if observed != expected_sha256:
        raise RuntimeError(f"source hash mismatch for {path}: {observed}")
    return np.load(path)


def run(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    with np.load(args.metadata) as metadata:
        labels = metadata["train_y"]
        users = metadata["train_users"]
    oof = np.empty((len(labels), 40), dtype=np.float32)
    fold_metrics = []
    for contract in manifest["fold_contracts"]:
        fold = contract["fold"]
        source = next(item for item in manifest["oof_sources"] if item["fold"] == fold)
        indices = np.flatnonzero(np.isin(users, contract["held_users"]))
        visual_path = (
            args.source_root
            / "best_release/legal_strict_v3/preproc_affine"
            / f"visual_fold{fold}_train.npy"
        )
        temporal_path = (
            args.source_root
            / "runs/public_cuhkx_dstformer"
            / f"logit_residual_tcn_cv_{fold}/val_logits.npy"
        )
        visual = checked_load(visual_path, source["source_sha256"]["visual_oof"])[indices]
        temporal = checked_load(temporal_path, source["source_sha256"]["temporal_oof"])
        logits = equal_tempered_logits(
            visual,
            temporal,
            contract["temperatures"]["visual"],
            contract["temperatures"]["temporal"],
        )
        if len(logits) != len(indices):
            raise RuntimeError(f"fold {fold} row mismatch")
        oof[indices] = logits
        correct = int(np.sum(logits.argmax(1) == labels[indices]))
        fold_metrics.append(
            {
                "fold": fold,
                "rows": len(indices),
                "correct": correct,
                "accuracy": correct / len(indices),
            }
        )

    test_root = args.source_root / "best_release/legal_strict_v3/selfcontained_baseline_replay_work"
    visual_test = checked_load(
        test_root / "visual_logits.npy", manifest["test_source_sha256"]["visual"]
    )
    temporal_test = checked_load(
        test_root / "temporal_logits.npy", manifest["test_source_sha256"]["temporal"]
    )
    temperatures = manifest["blend"]["temperatures"]
    test_logits = equal_tempered_logits(
        visual_test, temporal_test, temperatures["visual"], temperatures["temporal"]
    )
    sample = pd.read_csv(args.sample_submission)
    if sample.columns.tolist() != ["path", "prediction"] or len(sample) != len(test_logits):
        raise RuntimeError("sample submission schema mismatch")
    sample["prediction"] = test_logits.argmax(1)
    strict_prediction = pd.read_csv(RESULT_DIR / "submission.csv")["prediction"].to_numpy()
    prediction = test_logits.argmax(1)
    correct = int(np.sum(oof.argmax(1) == labels))
    metrics = {
        "protocol": "strictv3-temporal-visual-equal-tempered-logits/v1",
        "weights": {"visual": 0.5, "temporal": 0.5, "fusion": 0.0},
        "quality_gate": False,
        "temperature_source": "frozen strictV3 fold/full contracts",
        "oof": {
            "rows": len(labels),
            "correct": correct,
            "accuracy": correct / len(labels),
            "folds": fold_metrics,
        },
        "test": {
            "rows": len(prediction),
            "classes": int(np.unique(prediction).size),
            "changed_vs_strictv3": int(np.sum(prediction != strict_prediction)),
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
