#!/usr/bin/env python3
"""Fail-closed integrity and score check for the published strict-v3 release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from yolo_r2plus1d.strict_v3.paths import CHECKPOINT_DIR, RESULT_DIR
from yolo_r2plus1d.strict_v3.release.bundle import (
    DEFAULT_BUNDLE,
    MODEL_LIMIT_BYTES,
    load_bundle,
)

EXPECTED = {
    "model.pt": "c1feb649267aaa41ef490a93660735a76af4eae4629a687a90a102ac4e63bd75",
    "yolo11n.pt": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
    "release_manifest.json": "32bdaa22b2bd9abd9f1e1029c2bcd818d251e8a28da2ce4df8496058937ebcd6",
    "metrics.json": "af8446407d6c38a004f9b07f271c8e4bb85bb5466601118c08c11648f341f04a",
    "metadata.npz": "bf2e93e558b4ae148b14835c1f65c943828bff97dfaaf224a39c249c7599f099",
    "submission.csv": "e25094918e57f357e23346f479050bd9bf09dc907c849d4c8938a66812003343",
    "train_oof_logits.npy": "4dde2be7d0754f895f76b46c4ed3be291ecfd7b4fd9e4aa77d6000cf1712ee31",
    "test_logits.npy": "d700e4cf680f9958c23294d8e468855579289e37a468b481a8a1a5abe974fe3e",
}
EXPECTED_RAW_REPLAY = "e25094918e57f357e23346f479050bd9bf09dc907c849d4c8938a66812003343"
EXPECTED_BUNDLE = "4856eacc9f10043e873a6ac8c4c1359e9cf7d2a13685d8d07c6b2a7594800bf3"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--replayed-csv", type=Path)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    args = parser.parse_args()

    paths = {
        "model.pt": args.checkpoint_dir / "model.pt",
        "yolo11n.pt": args.checkpoint_dir / "yolo11n.pt",
        **{
            name: args.result_dir / name
            for name in EXPECTED
            if name not in {"model.pt", "yolo11n.pt"}
        },
    }
    mismatches = {}
    for name, expected in EXPECTED.items():
        actual = digest(paths[name]) if paths[name].is_file() else None
        if actual != expected:
            mismatches[name] = {"expected": expected, "actual": actual}
    if mismatches:
        raise SystemExit(f"release hash mismatch: {json.dumps(mismatches, indent=2)}")

    if not args.bundle.is_file():
        raise SystemExit(f"single-checkpoint inference bundle not found: {args.bundle}")
    bundle_hash = digest(args.bundle)
    if bundle_hash != EXPECTED_BUNDLE:
        raise SystemExit(
            f"single-checkpoint bundle hash mismatch: expected={EXPECTED_BUNDLE} "
            f"actual={bundle_hash}"
        )
    bundle = load_bundle(args.bundle)
    bundle_bytes = args.bundle.stat().st_size
    if bundle_bytes >= MODEL_LIMIT_BYTES:
        raise SystemExit("single-checkpoint release is not strictly below 100 MB")

    metrics = json.loads((args.result_dir / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads(paths["release_manifest.json"].read_text(encoding="utf-8"))
    with np.load(args.result_dir / "metadata.npz", allow_pickle=False) as metadata:
        labels = np.asarray(metadata["train_y"], dtype=np.int64)
    oof = np.asarray(np.load(paths["train_oof_logits.npy"]), dtype=np.float64)
    test_logits = np.asarray(np.load(paths["test_logits.npy"]), dtype=np.float64)
    submission = pd.read_csv(paths["submission.csv"])
    if oof.shape != (len(labels), 40) or test_logits.shape != (405, 40):
        raise SystemExit(
            f"unexpected logit shapes: oof={oof.shape}, test={test_logits.shape}"
        )
    if submission.columns.tolist() != ["path", "prediction"] or len(submission) != 405:
        raise SystemExit("submission schema mismatch")
    if not np.array_equal(submission["prediction"].to_numpy(), test_logits.argmax(1)):
        raise SystemExit("submission does not match the saved test logits")
    oof_accuracy = float(np.mean(oof.argmax(1) == labels))
    if not np.isclose(
        oof_accuracy, float(metrics["aggregate_oof_accuracy"]), atol=1e-15
    ):
        raise SystemExit("OOF score differs from metrics.json")
    if manifest["sha256"]["submission.csv"] != digest(paths["submission.csv"]):
        raise SystemExit("manifest submission hash mismatch")
    if manifest["test_replay"]["package_sha256"] != digest(paths["model.pt"]):
        raise SystemExit("manifest replay package hash mismatch")

    package = bundle["model"]
    contract = package.get("release_contract", {})
    preprocessing = contract.get("preprocessing_contract", {})
    raw_pipeline = contract.get("raw_pipeline", {})
    required = {
        "per_sample_zscore": False,
        "thermal_weight": 0.0,
        "skeleton_mask_used_in_blend": False,
        "test_labels_used": False,
        "test_statistics_used": False,
        "timestamp_metadata_used": False,
    }
    if any(contract.get(key) != value for key, value in required.items()):
        raise SystemExit("unsafe release contract")
    if preprocessing.get("source_split") != "train" or preprocessing.get(
        "test_statistics_used", True
    ):
        raise SystemExit("preprocessing contract is not train-only")
    if raw_pipeline.get("frames") != 16 or raw_pipeline.get("image_size") != 128:
        raise SystemExit("raw inference contract mismatch")
    if not np.isclose(float(contract.get("visual_package_output_scale", 0.0)), 0.5):
        raise SystemExit("visual package output scale mismatch")
    combined_bytes = (
        paths["model.pt"].stat().st_size + paths["yolo11n.pt"].stat().st_size
    )
    if combined_bytes > 100_000_000:
        raise SystemExit("release exceeds the 100 MB track limit")

    replay_matches = None
    replay_differences = None
    if args.replayed_csv is not None:
        replay = pd.read_csv(args.replayed_csv)
        replay_hash = digest(args.replayed_csv)
        if replay_hash != EXPECTED_RAW_REPLAY:
            raise SystemExit(f"unexpected raw replay hash: {replay_hash}")
        replay_matches = replay.equals(submission)
        if not replay_matches:
            replay_differences = int(
                (
                    replay["prediction"].to_numpy()
                    != submission["prediction"].to_numpy()
                ).sum()
            )
            raise SystemExit(f"raw replay differs in {replay_differences} predictions")

    print(
        json.dumps(
            {
                "ok": True,
                "oof_accuracy": oof_accuracy,
                "csv_rows": int(len(submission)),
                "test_classes": int(submission["prediction"].nunique()),
                "model_bytes": int(paths["model.pt"].stat().st_size),
                "detector_bytes": int(paths["yolo11n.pt"].stat().st_size),
                "combined_bytes": int(combined_bytes),
                "single_checkpoint_bytes": int(bundle_bytes),
                "single_checkpoint_margin_bytes": int(MODEL_LIMIT_BYTES - bundle_bytes),
                "single_checkpoint_sha256": bundle_hash,
                "manifest_hash": digest(paths["release_manifest.json"]),
                "raw_replay_matches": replay_matches,
                "raw_replay_differences": replay_differences,
                "test_labels_used": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
