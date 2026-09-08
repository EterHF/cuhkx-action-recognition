#!/usr/bin/env python3
"""Train leakage-checked nested-OOF conditional correction heads."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from yolo_r2plus1d.strict_v3.models.conditional_corrector import ConditionalCorrector
from yolo_r2plus1d.strict_v3.paths import RESULT_DIR
from yolo_r2plus1d.strict_v3.training.base import seed_everything


def correction_loss(
    baseline_logits: torch.Tensor,
    corrected_logits: torch.Tensor,
    labels: torch.Tensor,
    confidence_threshold: float,
    kl_weight: float,
) -> tuple[torch.Tensor, int]:
    ce = F.cross_entropy(corrected_logits, labels)
    baseline_probability = baseline_logits.softmax(dim=1)
    confidence, prediction = baseline_probability.max(dim=1)
    protect = (prediction == labels) & (confidence >= confidence_threshold)
    if bool(protect.any()):
        log_corrected = corrected_logits.log_softmax(dim=1)
        kl = F.kl_div(log_corrected[protect], baseline_probability[protect], reduction="batchmean")
    else:
        kl = corrected_logits.sum() * 0.0
    return ce + kl_weight * kl, int(protect.sum())


def prediction_delta(
    baseline: np.ndarray, candidate: np.ndarray, labels: np.ndarray
) -> dict[str, int]:
    baseline_correct = baseline == labels
    candidate_correct = candidate == labels
    corrected = int(np.sum(~baseline_correct & candidate_correct))
    broken = int(np.sum(baseline_correct & ~candidate_correct))
    return {"corrected": corrected, "broken": broken, "net": corrected - broken}


def classification_metrics(prediction: np.ndarray, labels: np.ndarray, users: np.ndarray) -> dict:
    class_accuracy = [
        float(np.mean(prediction[labels == target] == target)) for target in np.unique(labels)
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
        "worst_user_accuracy": float(min(user_accuracy.values())),
        "user_accuracy": user_accuracy,
    }


def paired_exact_pvalue(corrected: int, broken: int) -> float:
    discordant = corrected + broken
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, value) for value in range(min(corrected, broken) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def subject_bootstrap_interval(
    baseline: np.ndarray,
    candidate: np.ndarray,
    labels: np.ndarray,
    users: np.ndarray,
    seed: int,
    replicates: int = 10_000,
) -> list[float]:
    unique_users = np.unique(users)
    contributions = np.asarray(
        [
            np.mean(candidate[users == user] == labels[users == user])
            - np.mean(baseline[users == user] == labels[users == user])
            for user in unique_users
        ]
    )
    generator = np.random.default_rng(seed)
    samples = generator.choice(contributions, (replicates, len(unique_users)), replace=True)
    return [float(value) for value in np.quantile(samples.mean(axis=1), (0.025, 0.975))]


def checked_array(path: Path, rows: int, columns: int) -> np.ndarray:
    value = np.asarray(np.load(path), dtype=np.float32)
    if value.shape != (rows, columns) or not np.isfinite(value).all():
        raise RuntimeError(f"invalid nested array {path}: {value.shape}")
    return value


def load_split(record: dict, expected_indices: np.ndarray) -> tuple[np.ndarray, ...]:
    indices = np.asarray(np.load(record["indices"]), dtype=np.int64)
    if not np.array_equal(indices, expected_indices):
        raise RuntimeError("nested manifest indices do not match metadata split")
    rows = len(indices)
    return (
        checked_array(Path(record["visual_features"]), rows, 512),
        checked_array(Path(record["temporal_logits"]), rows, 40),
        checked_array(Path(record["baseline_logits"]), rows, 40),
    )


def validate_provenance(record: dict, held_users: set[int], training: bool) -> None:
    excluded = set(int(value) for value in record.get("upstream_excluded_users", []))
    if not held_users.issubset(excluded):
        raise RuntimeError("upstream models do not exclude every outer-held user")
    if training and record.get("cross_fitted_within_outer_train") is not True:
        raise RuntimeError("corrector training inputs are not inner cross-fitted")
    if record.get("target_labels_used_for_upstream_selection") is not False:
        raise RuntimeError("unsafe upstream checkpoint selection provenance")


@torch.inference_mode()
def predict(
    model: ConditionalCorrector,
    features: np.ndarray,
    temporal: np.ndarray,
    baseline: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    residual = model(
        torch.from_numpy(features).to(device),
        torch.from_numpy(temporal).to(device),
    )
    return baseline + residual.float().cpu().numpy()


def run(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "conditional-corrector-nested-oof/v1":
        raise RuntimeError("unsupported or missing nested OOF manifest")
    with np.load(args.metadata) as metadata:
        labels = metadata["train_y"].astype(np.int64)
        users = metadata["train_users"].astype(np.int64)
    corrected_oof = np.empty((len(labels), 40), dtype=np.float32)
    baseline_oof = np.empty_like(corrected_oof)
    assigned = np.zeros(len(labels), dtype=bool)
    fold_metrics = []
    device = torch.device(args.device)
    for fold_record in manifest["folds"]:
        fold = str(fold_record["fold"])
        held_users = set(int(value) for value in fold_record["held_users"])
        held_indices = np.flatnonzero(np.isin(users, list(held_users)))
        train_indices = np.flatnonzero(~np.isin(users, list(held_users)))
        if not len(held_indices) or bool(assigned[held_indices].any()):
            raise RuntimeError("outer folds are empty or overlap")
        assigned[held_indices] = True
        train_record, held_record = fold_record["train"], fold_record["held"]
        validate_provenance(train_record, held_users, training=True)
        validate_provenance(held_record, held_users, training=False)
        train_features, train_temporal, train_baseline = load_split(train_record, train_indices)
        held_features, held_temporal, held_baseline = load_split(held_record, held_indices)

        seed_everything(args.seed)
        model = ConditionalCorrector(hidden_dim=args.hidden_dim).to(device)
        # Only the final layer is zero initialized; hidden projections retain
        # their seeded random initialization while the initial residual is zero.
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        dataset = TensorDataset(
            torch.from_numpy(train_features),
            torch.from_numpy(train_temporal),
            torch.from_numpy(train_baseline),
            torch.from_numpy(labels[train_indices]),
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
        train_probability = torch.from_numpy(train_baseline).softmax(dim=1)
        train_confidence, train_prediction = train_probability.max(dim=1)
        protected_train_rows = int(
            np.sum(
                (train_prediction.numpy() == labels[train_indices])
                & (train_confidence.numpy() >= args.confidence_threshold)
            )
        )
        for _ in range(args.epochs):
            model.train()
            for features, temporal, baseline, target in loader:
                features = features.to(device)
                temporal = temporal.to(device)
                baseline = baseline.to(device)
                target = target.to(device)
                optimizer.zero_grad(set_to_none=True)
                candidate = baseline + model(features, temporal)
                loss, _ = correction_loss(
                    baseline,
                    candidate,
                    target,
                    args.confidence_threshold,
                    args.kl_weight,
                )
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

        held_corrected = predict(model, held_features, held_temporal, held_baseline, device)
        corrected_oof[held_indices] = held_corrected
        baseline_oof[held_indices] = held_baseline
        baseline_prediction = held_baseline.argmax(1)
        corrected_prediction = held_corrected.argmax(1)
        fold_metrics.append(
            {
                "fold": fold,
                "rows": len(held_indices),
                "baseline_correct": int(np.sum(baseline_prediction == labels[held_indices])),
                "candidate_correct": int(np.sum(corrected_prediction == labels[held_indices])),
                "delta": prediction_delta(
                    baseline_prediction, corrected_prediction, labels[held_indices]
                ),
                "protected_train_rows": protected_train_rows,
            }
        )
        torch.save(model.state_dict(), args.output_dir / f"fold{fold}.pt")
        print(json.dumps(fold_metrics[-1]), flush=True)

    if not bool(assigned.all()):
        raise RuntimeError("outer folds do not cover every metadata row exactly once")
    baseline_prediction = baseline_oof.argmax(1)
    corrected_prediction = corrected_oof.argmax(1)
    metrics = {
        "protocol": "conditional-corrector-nested-oof/v1",
        "fixed_epochs": args.epochs,
        "seed": args.seed,
        "hidden_dim": args.hidden_dim,
        "kl_weight": args.kl_weight,
        "confidence_threshold": args.confidence_threshold,
        "baseline_correct": int(np.sum(baseline_prediction == labels)),
        "candidate_correct": int(np.sum(corrected_prediction == labels)),
        "delta": prediction_delta(baseline_prediction, corrected_prediction, labels),
        "baseline_metrics": classification_metrics(baseline_prediction, labels, users),
        "candidate_metrics": classification_metrics(corrected_prediction, labels, users),
        "folds": fold_metrics,
        "all_folds_non_degrading": all(item["delta"]["net"] >= 0 for item in fold_metrics),
        "held_labels_used_for_checkpoint_selection": False,
        "anonymous_test_accessed": False,
    }
    metrics["paired_exact_pvalue"] = paired_exact_pvalue(
        metrics["delta"]["corrected"], metrics["delta"]["broken"]
    )
    metrics["subject_bootstrap_accuracy_delta_95ci"] = subject_bootstrap_interval(
        baseline_prediction, corrected_prediction, labels, users, args.seed
    )
    np.save(args.output_dir / "baseline_oof.npy", baseline_oof)
    np.save(args.output_dir / "corrected_oof.npy", corrected_oof)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--metadata", type=Path, default=RESULT_DIR / "metadata.npz")
    result.add_argument("--epochs", type=int, default=60)
    result.add_argument("--batch-size", type=int, default=128)
    result.add_argument("--hidden-dim", type=int, default=128)
    result.add_argument("--lr", type=float, default=3e-4)
    result.add_argument("--weight-decay", type=float, default=0.01)
    result.add_argument("--kl-weight", type=float, default=1.0)
    result.add_argument("--confidence-threshold", type=float, default=0.8)
    result.add_argument("--seed", type=int, default=2026)
    result.add_argument("--device", default="cuda:0")
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=False)
    run(arguments)
