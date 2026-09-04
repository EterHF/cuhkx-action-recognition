"""Fail-closed audit for the two preregistered strictV3 consensus candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from yolo_r2plus1d.strict_v3.paths import CHECKPOINT_DIR, REPO_ROOT, RESULT_DIR
from yolo_r2plus1d.strict_v3.release.bundle import MODEL_LIMIT_BYTES, build_bundle

MANIFEST = REPO_ROOT / "results/experiments/manifest.json"
FORBIDDEN_TRUE_KEYS = {
    "class_repair_used",
    "leaderboard_used_for_coefficient",
    "leaderboard_used_for_selection",
    "test_labels_used",
    "test_statistics_used",
    "timestamp_metadata_used",
    "timestamp_or_cohort_metadata_used",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def forbidden_flags(value: Any, prefix: str = "") -> list[str]:
    failures: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            if key in FORBIDDEN_TRUE_KEYS and item is not False:
                failures.append(name)
            failures.extend(forbidden_flags(item, name))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            failures.extend(forbidden_flags(item, f"{prefix}[{index}]"))
    return failures


def audit_candidate(name: str, replayed_csv: Path | None = None) -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    spec = manifest["candidates"][name]
    paths = {
        key: REPO_ROOT / spec[key]
        for key in ("package", "metrics", "oof", "raw_replay", "submission")
    }
    for key, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"missing {name} {key}: {path}")
        expected = spec[f"{key}_sha256"]
        if digest(path) != expected:
            raise RuntimeError(f"{name} {key} hash mismatch")

    with np.load(RESULT_DIR / "metadata.npz", allow_pickle=False) as metadata:
        labels = np.asarray(metadata["train_y"], dtype=np.int64)
        users = np.asarray(metadata["train_users"], dtype=np.int64)
    baseline = np.asarray(
        np.load(RESULT_DIR / "train_oof_logits.npy"), dtype=np.float64
    ).argmax(1)
    candidate_oof = np.load(paths["oof"])
    if spec["oof_kind"] == "probabilities":
        if candidate_oof.shape != (len(labels), 40):
            raise RuntimeError(f"{name} OOF probability shape mismatch")
        predictions = candidate_oof.argmax(1)
    else:
        predictions = np.asarray(candidate_oof, dtype=np.int64)
        if predictions.shape != labels.shape:
            raise RuntimeError(f"{name} OOF prediction shape mismatch")

    baseline_accuracy = float(np.mean(baseline == labels))
    candidate_accuracy = float(np.mean(predictions == labels))
    if not np.isclose(baseline_accuracy, manifest["baseline"]["oof_accuracy"]):
        raise RuntimeError("baseline OOF accuracy mismatch")
    if not np.isclose(candidate_accuracy, spec["oof_accuracy"]):
        raise RuntimeError(f"{name} OOF accuracy mismatch")

    release_manifest = json.loads(
        (RESULT_DIR / "release_manifest.json").read_text(encoding="utf-8")
    )
    fold_deltas: dict[str, float] = {}
    for fold in release_manifest["oof_sources"]:
        mask = np.isin(users, fold["held_users"])
        delta = float(np.mean(predictions[mask] == labels[mask])) - float(
            np.mean(baseline[mask] == labels[mask])
        )
        fold_deltas[str(fold["fold"])] = delta
    if any(delta < -1e-12 for delta in fold_deltas.values()):
        raise RuntimeError(f"{name} degrades an outer fold: {fold_deltas}")

    package = torch.load(paths["package"], map_location="cpu", weights_only=True)
    unsafe = forbidden_flags(package)
    if unsafe:
        raise RuntimeError(f"{name} has unsafe contract flags: {unsafe}")
    detector = CHECKPOINT_DIR / "yolo11n.pt"
    component_sum_bytes = paths["package"].stat().st_size + detector.stat().st_size
    if paths["package"].stat().st_size != spec["package_bytes"]:
        raise RuntimeError(f"{name} package size mismatch")
    with tempfile.TemporaryDirectory(prefix=f"cuhkx_{name}_bundle_") as directory:
        bundle_path = Path(directory) / "inference_bundle.pt"
        build_bundle(paths["package"], detector, bundle_path)
        single_checkpoint_bytes = bundle_path.stat().st_size
    if single_checkpoint_bytes >= MODEL_LIMIT_BYTES:
        raise RuntimeError(f"{name} single checkpoint exceeds the 100 MB track limit")

    submission = pd.read_csv(paths["submission"])
    if submission.columns.tolist() != ["path", "prediction"] or len(submission) != 405:
        raise RuntimeError(f"{name} submission schema mismatch")
    if submission["prediction"].nunique() != 40:
        raise RuntimeError(f"{name} does not naturally cover all 40 classes")
    if replayed_csv is not None and digest(replayed_csv) != spec["submission_sha256"]:
        raise RuntimeError(f"{name} raw replay hash mismatch")

    return {
        "candidate": name,
        "ok": True,
        "baseline_oof_accuracy": baseline_accuracy,
        "candidate_oof_accuracy": candidate_accuracy,
        "oof_delta": candidate_accuracy - baseline_accuracy,
        "fold_deltas": fold_deltas,
        "package_bytes": paths["package"].stat().st_size,
        "component_sum_bytes": component_sum_bytes,
        "single_checkpoint_bytes": single_checkpoint_bytes,
        "single_checkpoint_margin_bytes": MODEL_LIMIT_BYTES - single_checkpoint_bytes,
        "submission_sha256": digest(paths["submission"]),
        "changed_test_predictions": int(
            (
                submission["prediction"].to_numpy()
                != pd.read_csv(RESULT_DIR / "submission.csv")["prediction"].to_numpy()
            ).sum()
        ),
        "test_labels_used": False,
        "kaggle_submitted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "candidate",
        nargs="?",
        choices=("sched30_consensus", "temporal_pool_consensus"),
    )
    parser.add_argument("--replayed-csv", type=Path)
    args = parser.parse_args()
    names = (
        [args.candidate]
        if args.candidate
        else ["sched30_consensus", "temporal_pool_consensus"]
    )
    if args.replayed_csv is not None and len(names) != 1:
        parser.error("--replayed-csv requires one candidate")
    results = [audit_candidate(name, args.replayed_csv) for name in names]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
