#!/usr/bin/env python3
"""Export released DSTFormer pre-fc2 features and fold-train PCA projections."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from yolo_r2plus1d.strict_v3.data.indexing import discover_train
from yolo_r2plus1d.strict_v3.data.skeleton_retarget import load_clip
from yolo_r2plus1d.strict_v3.evaluation.visual_quantization import FOLDS
from yolo_r2plus1d.strict_v3.models.public_dstformer import Net as PublicDSTNet
from yolo_r2plus1d.strict_v3.paths import DATA_DIR, RESULT_DIR
from yolo_r2plus1d.strict_v3.release.replay import RawFrames


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def export_representations(
    raw: np.ndarray,
    package: dict,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model = PublicDSTNet()
    model.load_state_dict(package["dstformer"]["state_dict"], strict=True)
    model.to(device).eval()
    features = np.empty((len(raw), raw.shape[1], 2048), dtype=np.float32)
    logits = np.empty((len(raw), raw.shape[1], 40), dtype=np.float32)
    loader = DataLoader(
        RawFrames(raw),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    for inputs, indices in loader:
        batch, steps = inputs.shape[:2]
        flattened = inputs.reshape(batch * steps, 1, 17, 3).to(
            device, non_blocking=True
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            representation = model.forward_features(flattened)
            frame_logits = model.head["fc2"](representation)
        indices = indices.numpy()
        features[indices] = (
            representation.reshape(batch, steps, 2048).float().cpu().numpy()
        )
        logits[indices] = frame_logits.reshape(batch, steps, 40).float().cpu().numpy()
    return features, logits


def fit_fold_pca(
    features: np.ndarray,
    users: np.ndarray,
    output_dir: Path,
    seed: int,
) -> dict[str, dict]:
    reports = {}
    for fold, held_users in FOLDS.items():
        train_rows = np.flatnonzero(~np.isin(users, held_users))
        held_rows = np.flatnonzero(np.isin(users, held_users))
        train_frames = np.asarray(features[train_rows], dtype=np.float32).reshape(
            -1, features.shape[-1]
        )
        pca = PCA(
            n_components=40,
            whiten=False,
            svd_solver="randomized",
            random_state=seed,
        )
        pca.fit(train_frames)
        projected = np.empty((len(features), features.shape[1], 40), np.float32)
        for start in range(0, len(features), 256):
            stop = min(start + 256, len(features))
            values = np.asarray(features[start:stop], dtype=np.float32).reshape(
                -1, features.shape[-1]
            )
            projected[start:stop] = pca.transform(values).reshape(
                stop - start, features.shape[1], 40
            )
        projection_path = output_dir / f"fold{fold}_pca40.npy"
        state_path = output_dir / f"fold{fold}_pca40.npz"
        np.save(projection_path, projected)
        np.savez_compressed(
            state_path,
            mean=pca.mean_.astype(np.float32),
            components=pca.components_.astype(np.float32),
            explained_variance=pca.explained_variance_.astype(np.float32),
            explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
            train_rows=train_rows,
            held_rows=held_rows,
        )
        reports[fold] = {
            "held_users": list(held_users),
            "train_clips": int(len(train_rows)),
            "train_frames": int(len(train_frames)),
            "held_clips": int(len(held_rows)),
            "held_frames_used_for_fit": 0,
            "explained_variance_ratio_sum": float(
                pca.explained_variance_ratio_.sum()
            ),
            "projection_sha256": sha256(projection_path),
            "pca_state_sha256": sha256(state_path),
        }
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-root", type=Path, default=DATA_DIR / "processed/train/HAR/data"
    )
    parser.add_argument("--metadata", type=Path, default=RESULT_DIR / "metadata.npz")
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-logits", type=Path)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    samples = discover_train(args.train_root)
    with np.load(args.metadata, allow_pickle=False) as metadata:
        expected_keys = np.asarray(metadata["train_keys"])
        users = np.asarray(metadata["train_users"], dtype=np.int64)
    if not np.array_equal(np.asarray([sample[0] for sample in samples]), expected_keys):
        raise RuntimeError("training discovery order differs from frozen metadata")
    raw = np.stack(
        [load_clip(paths["Skeleton"], args.frames) for _, _, _, paths in samples]
    )
    package = torch.load(args.package, map_location="cpu", weights_only=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    features, logits = export_representations(
        raw, package, torch.device(args.device), args.batch_size
    )
    features_path = args.output_dir / "pre_fc2_features.npy"
    logits_path = args.output_dir / "frame_logits.npy"
    np.save(features_path, features)
    np.save(logits_path, logits)
    reference = None
    if args.reference_logits is not None:
        expected = np.asarray(np.load(args.reference_logits), dtype=np.float32)
        if expected.shape != logits.shape:
            raise RuntimeError("reference logits have a different shape")
        reference = {
            "path": str(args.reference_logits.resolve()),
            "sha256": sha256(args.reference_logits),
            "max_abs_difference": float(np.max(np.abs(expected - logits))),
            "frame_argmax_agreement": float(
                np.mean(expected.argmax(-1) == logits.argmax(-1))
            ),
            "clip_argmax_agreement": float(
                np.mean(expected.mean(1).argmax(-1) == logits.mean(1).argmax(-1))
            ),
        }
    folds = fit_fold_pca(features, users, args.output_dir, args.seed)
    report = {
        "schema_version": "cuhkx-prelogit-pca40/v1",
        "rows": len(features),
        "frames": features.shape[1],
        "source_dimensions": features.shape[2],
        "projected_dimensions": 40,
        "feature_definition": "post-ReLU 2048-D representation immediately before fc2",
        "features_sha256": sha256(features_path),
        "frame_logits_sha256": sha256(logits_path),
        "same_forward_pass": True,
        "reference_comparison": reference,
        "folds": folds,
        "held_labels_accessed": False,
        "anonymous_test_accessed": False,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
