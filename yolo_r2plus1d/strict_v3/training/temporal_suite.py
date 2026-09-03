#!/usr/bin/env python3
"""Run and audit the complete strictV3 temporal training suite."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from yolo_r2plus1d.strict_v3.training.temporal import sha256

FOLDS = {
    "A": [1, 6, 17, 22],
    "B": [2, 7, 18, 23],
    "C": [3, 8, 19, 24],
    "D": [4, 9, 20],
    "E": [5, 16, 21],
}


def run_job(command: list[str], output: Path, resume: bool) -> None:
    metrics = output / "metrics.json"
    if resume and metrics.is_file():
        return
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty run directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "train.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)


def command(
    args: argparse.Namespace,
    output: Path,
    seed: int,
    users: list[int] | None,
    sched30: bool,
    full: bool,
) -> list[str]:
    result = [
        sys.executable,
        "-m",
        "yolo_r2plus1d.strict_v3.training.temporal",
        "--logits",
        str(args.frame_logits),
        "--metadata",
        str(args.metadata),
        "--out",
        str(output),
        "--kind",
        "tcn",
        "--epochs",
        "5" if sched30 or full else "30",
        "--batch",
        str(args.batch),
        "--workers",
        str(args.workers),
        "--seed",
        str(seed),
        "--device",
        args.device,
    ]
    if users is not None:
        result.extend(["--val-users", *(str(user) for user in users)])
    if full:
        if args.test_frame_logits is None:
            raise ValueError("--test-frame-logits is required for full-data jobs")
        result.extend(["--test-logits", str(args.test_frame_logits)])
    if sched30:
        result.extend(
            ["--scheduler-epochs", "30", "--select-last", "--defer-val-metrics"]
        )
    return result


def assemble_oof(root: Path, seeds: list[int], metadata: Path) -> tuple[np.ndarray, dict]:
    with np.load(metadata, allow_pickle=False) as values:
        labels = np.asarray(values["train_y"], dtype=np.int64)
        users = np.asarray(values["train_users"], dtype=np.int64)
    seed_oof = []
    fold_metrics: dict[str, dict[str, float]] = {}
    for seed in seeds:
        oof = np.zeros((len(labels), 40), dtype=np.float32)
        for fold, held_users in FOLDS.items():
            rows = np.flatnonzero(np.isin(users, held_users))
            logits = np.load(root / f"seed{seed}" / f"fold{fold}" / "val_logits.npy")
            if logits.shape != (len(rows), 40):
                raise RuntimeError(f"fold {fold} logits are not aligned: {logits.shape}")
            oof[rows] = logits
            fold_metrics.setdefault(fold, {})[str(seed)] = float(
                np.mean(logits.argmax(1) == labels[rows])
            )
        seed_oof.append(oof)
    mean_oof = np.mean(seed_oof, axis=0, dtype=np.float32)
    return mean_oof, {
        "seed_accuracy": {
            str(seed): float(np.mean(oof.argmax(1) == labels))
            for seed, oof in zip(seeds, seed_oof, strict=True)
        },
        "seed_mean_accuracy": float(np.mean(mean_oof.argmax(1) == labels)),
        "fold_seed_accuracy": fold_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-logits", type=Path, required=True)
    parser.add_argument("--test-frame-logits", type=Path)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recipe", choices=("strict", "sched30"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    for path in (args.frame_logits, args.metadata):
        if not path.is_file():
            parser.error(f"input does not exist: {path}")
    if args.test_frame_logits is None:
        parser.error("--test-frame-logits is required because every suite includes a full job")
    if not args.test_frame_logits.is_file():
        parser.error(f"input does not exist: {args.test_frame_logits}")

    seeds = [2026] if args.recipe == "strict" else [2026, 2027, 2028]
    sched30 = args.recipe == "sched30"
    args.output.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        for fold, held_users in FOLDS.items():
            output = args.output / f"seed{seed}" / f"fold{fold}"
            run_job(
                command(args, output, seed, held_users, sched30, full=False),
                output,
                args.resume,
            )
        output = args.output / f"full_seed{seed}"
        run_job(
            command(args, output, seed, None, sched30, full=True),
            output,
            args.resume,
        )

    oof, metrics = assemble_oof(args.output, seeds, args.metadata)
    np.save(args.output / "temporal_oof_logits.npy", oof)
    tests = [
        np.load(args.output / f"full_seed{seed}" / "test_logits.npy") for seed in seeds
    ]
    test_logits = np.mean(tests, axis=0, dtype=np.float32)
    np.save(args.output / "temporal_test_logits.npy", test_logits)
    receipt = {
        "schema_version": "cuhkx-temporal-retrain/v1",
        "recipe": args.recipe,
        "device": args.device,
        "seeds": seeds,
        "folds": FOLDS,
        "inputs": {
            "frame_logits": str(args.frame_logits.resolve()),
            "frame_logits_sha256": sha256(args.frame_logits),
            "test_frame_logits": str(args.test_frame_logits.resolve()),
            "test_frame_logits_sha256": sha256(args.test_frame_logits),
            "metadata": str(args.metadata.resolve()),
            "metadata_sha256": sha256(args.metadata),
        },
        **metrics,
        "outputs": {
            "oof_sha256": sha256(args.output / "temporal_oof_logits.npy"),
            "test_sha256": sha256(args.output / "temporal_test_logits.npy"),
        },
    }
    (args.output / "receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
