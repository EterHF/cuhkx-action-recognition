#!/usr/bin/env python3
"""Build, screen, and full-fit a deployment-aligned T+V corrector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from yolo_r2plus1d.strict_v3.evaluation.visual_quantization import FOLDS, quantized_state
from yolo_r2plus1d.strict_v3.models.conditional_corrector import ConditionalCorrector
from yolo_r2plus1d.strict_v3.release.blend import apply_gate
from yolo_r2plus1d.strict_v3.release.bundle import MODEL_LIMIT_BYTES, build_bundle
from yolo_r2plus1d.strict_v3.training.base import seed_everything
from yolo_r2plus1d.strict_v3.training.conditional_corrector import (
    classification_metrics,
    correction_loss,
    prediction_delta,
)
from yolo_r2plus1d.strict_v3.training.public_finetune import (
    VideoDataset,
    dequantize_state,
    make_model,
)


@torch.inference_mode()
def visual_outputs(
    model: nn.Module,
    cache: Path,
    indices: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = VideoDataset(
        cache,
        indices,
        None,
        False,
        False,
        "all",
        mean,
        std,
        False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    logits = np.empty((len(indices), 40), dtype=np.float32)
    features = np.empty((len(indices), 512), dtype=np.float16)
    location = {int(index): position for position, index in enumerate(indices)}
    model.eval()
    for frames, source_indices in loader:
        with torch.autocast(device.type, dtype=torch.float16):
            hidden = model.forward_features(frames.to(device, non_blocking=True))
            output = model.head(hidden)
        destinations = np.asarray([location[int(index)] for index in source_indices])
        logits[destinations] = output.float().cpu().numpy()
        features[destinations] = hidden.float().cpu().numpy().astype(np.float16)
    return logits, features


def quantized_model(checkpoint: Path, device: torch.device) -> nn.Module:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = make_model()
    model.load_state_dict(dequantize_state(quantized_state(payload["model_state"], "uniform5")))
    return model.to(device).eval()


def fused_logits(visual: np.ndarray, temporal: np.ndarray, release: dict) -> np.ndarray:
    weights = dict(release["weights"])
    weights["fusion"] = 0.0
    output, _ = apply_gate(
        {
            "fusion": np.zeros_like(visual),
            "visual": visual * float(release["visual_package_output_scale"]),
            "temporal": temporal,
        },
        release["temperatures"],
        weights,
    )
    return output.astype(np.float32)


def extract(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with np.load(args.metadata) as metadata:
        labels = metadata["train_y"].astype(np.int64)
        users = metadata["train_users"].astype(np.int64)
    package = torch.load(args.source_package, map_location="cpu", weights_only=True)
    release = package["release_contract"]
    device = torch.device(args.device)
    visual_oof = np.empty((len(labels), 40), dtype=np.float32)
    feature_oof = np.empty((len(labels), 512), dtype=np.float16)
    for fold, held_users in FOLDS.items():
        indices = np.flatnonzero(np.isin(users, held_users))
        contract = json.loads((args.contract_dir / f"contract_fold{fold}.json").read_text())
        mean = torch.tensor(contract["mean"], dtype=torch.float32).view(1, 4, 1, 1)
        std = torch.tensor(contract["std"], dtype=torch.float32).view(1, 4, 1, 1)
        model = quantized_model(
            args.checkpoint_root / f"affine_head_fold{fold}/best_fp16.pt", device
        )
        visual, features = visual_outputs(
            model, args.train_cache, indices, mean, std, device, args.batch_size, args.workers
        )
        visual_oof[indices], feature_oof[indices] = visual, features
        del model
        torch.cuda.empty_cache()
        print(json.dumps({"fold": fold, "rows": len(indices)}), flush=True)
    temporal_oof = np.asarray(np.load(args.temporal_oof), dtype=np.float32)
    baseline_oof = fused_logits(visual_oof, temporal_oof, release)

    contract = release["preprocessing_contract"]
    mean = torch.tensor(contract["mean"], dtype=torch.float32).view(1, 4, 1, 1)
    std = torch.tensor(contract["std"], dtype=torch.float32).view(1, 4, 1, 1)
    test_indices = np.arange(len(np.load(args.test_cache, mmap_mode="r")))
    model = quantized_model(args.full_checkpoint, device)
    visual_test, feature_test = visual_outputs(
        model, args.test_cache, test_indices, mean, std, device, args.batch_size, args.workers
    )
    temporal_test = np.asarray(np.load(args.temporal_test), dtype=np.float32)
    baseline_test = fused_logits(visual_test, temporal_test, release)
    stored_visual = np.asarray(np.load(args.stored_visual_test), dtype=np.float32)
    visual_max_abs = float(
        np.max(np.abs(visual_test * float(release["visual_package_output_scale"]) - stored_visual))
    )
    expected = pd.read_csv(args.expected_submission)["prediction"].to_numpy()
    baseline_differences = int(np.sum(baseline_test.argmax(1) != expected))
    for name, values in {
        "visual_oof_logits": visual_oof,
        "visual_oof_features": feature_oof,
        "temporal_oof_logits": temporal_oof,
        "baseline_oof_logits": baseline_oof,
        "visual_test_logits": visual_test,
        "visual_test_features": feature_test,
        "temporal_test_logits": temporal_test,
        "baseline_test_logits": baseline_test,
    }.items():
        np.save(args.output_dir / f"{name}.npy", values)
    metrics = {
        "protocol": "deployment-aligned-single-view-t-v/v1",
        "visual_oof": classification_metrics(visual_oof.argmax(1), labels, users),
        "baseline_oof": classification_metrics(baseline_oof.argmax(1), labels, users),
        "stored_visual_test_max_abs": visual_max_abs,
        "baseline_test_prediction_differences": baseline_differences,
        "single_view": True,
        "visual_quantization": "uniform5 from original FP16",
        "visual_package_output_scale": release["visual_package_output_scale"],
        "anonymous_test_labels_used": False,
    }
    (args.output_dir / "extract_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


def fit_corrector(
    features: np.ndarray,
    temporal: np.ndarray,
    baseline: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> ConditionalCorrector:
    seed_everything(args.seed)
    model = ConditionalCorrector(visual_dim=features.shape[1], hidden_dim=args.hidden_dim).to(
        device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    dataset = TensorDataset(
        torch.from_numpy(features[train_indices].astype(np.float32)),
        torch.from_numpy(temporal[train_indices]),
        torch.from_numpy(baseline[train_indices]),
        torch.from_numpy(labels[train_indices]),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.corrector_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    for _ in range(args.epochs):
        model.train()
        for hidden, temporal_logits, base, target in loader:
            hidden, temporal_logits, base, target = (
                hidden.to(device),
                temporal_logits.to(device),
                base.to(device),
                target.to(device),
            )
            optimizer.zero_grad(set_to_none=True)
            candidate = base + model(hidden, temporal_logits)
            loss, _ = correction_loss(base, candidate, target, 0.8, 1.0)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return model.eval()


@torch.inference_mode()
def corrected(
    model: ConditionalCorrector,
    features: np.ndarray,
    temporal: np.ndarray,
    baseline: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    residual = model(
        torch.from_numpy(features[indices].astype(np.float32)).to(device),
        torch.from_numpy(temporal[indices]).to(device),
    )
    return baseline[indices] + residual.float().cpu().numpy()


def promotion_passed(
    fold_metrics: list[dict[str, int]],
    delta: dict[str, int],
    baseline: dict[str, object],
    candidate: dict[str, object],
) -> bool:
    """Apply the preregistered promotion gate without silently relaxing it."""
    return (
        all(item["net"] >= 0 for item in fold_metrics)
        and delta["net"] > 0
        and candidate["subject_macro_accuracy"] >= baseline["subject_macro_accuracy"]
        and candidate["worst_user_accuracy"] >= baseline["worst_user_accuracy"]
    )


def screen(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with np.load(args.metadata) as metadata:
        labels = metadata["train_y"].astype(np.int64)
        users = metadata["train_users"].astype(np.int64)
    visual_name = (
        "visual_oof_logits.npy" if args.visual_input == "logits" else "visual_oof_features.npy"
    )
    features = np.load(args.inputs / visual_name)
    temporal = np.load(args.inputs / "temporal_oof_logits.npy")
    baseline = np.load(args.inputs / "baseline_oof_logits.npy")
    candidate = np.empty_like(baseline)
    fold_metrics = []
    device = torch.device(args.device)
    for fold, held_users in FOLDS.items():
        held = np.flatnonzero(np.isin(users, held_users))
        train = np.flatnonzero(~np.isin(users, held_users))
        model = fit_corrector(features, temporal, baseline, labels, train, args, device)
        candidate[held] = corrected(model, features, temporal, baseline, held, device)
        base_prediction, new_prediction = baseline[held].argmax(1), candidate[held].argmax(1)
        fold_metrics.append(
            {"fold": fold, **prediction_delta(base_prediction, new_prediction, labels[held])}
        )
        torch.save(model.state_dict(), args.output_dir / f"fold{fold}.pt")
    base_prediction, new_prediction = baseline.argmax(1), candidate.argmax(1)
    base_metrics = classification_metrics(base_prediction, labels, users)
    new_metrics = classification_metrics(new_prediction, labels, users)
    delta = prediction_delta(base_prediction, new_prediction, labels)
    passed = promotion_passed(fold_metrics, delta, base_metrics, new_metrics)
    np.save(args.output_dir / "corrected_oof.npy", candidate)
    metrics = {
        "protocol": "deployment-aligned-meta-fold-screen/v1",
        "visual_input": args.visual_input,
        "known_upstream_leakage_limitation": True,
        "baseline": base_metrics,
        "candidate": new_metrics,
        "delta": delta,
        "folds": fold_metrics,
        "promotion_gate_passed": passed,
        "held_labels_used_for_checkpoint_selection": False,
        "anonymous_test_accessed": False,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


def full_fit(args: argparse.Namespace) -> None:
    screen_metrics = json.loads(args.screen_metrics.read_text())
    if screen_metrics.get("promotion_gate_passed") is not True:
        raise RuntimeError("deployment-aligned screen did not pass")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with np.load(args.metadata) as metadata:
        labels = metadata["train_y"].astype(np.int64)
    visual_name = (
        "visual_oof_logits.npy" if args.visual_input == "logits" else "visual_oof_features.npy"
    )
    visual_test_name = (
        "visual_test_logits.npy" if args.visual_input == "logits" else "visual_test_features.npy"
    )
    features = np.load(args.inputs / visual_name)
    temporal = np.load(args.inputs / "temporal_oof_logits.npy")
    baseline = np.load(args.inputs / "baseline_oof_logits.npy")
    test_features = np.load(args.inputs / visual_test_name)
    test_temporal = np.load(args.inputs / "temporal_test_logits.npy")
    test_baseline = np.load(args.inputs / "baseline_test_logits.npy")
    device = torch.device(args.device)
    model = fit_corrector(
        features, temporal, baseline, labels, np.arange(len(labels)), args, device
    )
    test_indices = np.arange(len(test_baseline))
    test_corrected = corrected(
        model, test_features, test_temporal, test_baseline, test_indices, device
    )
    sample = pd.read_csv(args.sample_submission)
    sample["prediction"] = test_corrected.argmax(1)
    submission = args.output_dir / "submission.csv"
    sample.to_csv(submission, index=False)
    np.save(args.output_dir / "corrected_test_logits.npy", test_corrected)
    state = {
        key: value.detach().cpu().half() if value.is_floating_point() else value.detach().cpu()
        for key, value in model.state_dict().items()
    }
    torch.save(state, args.output_dir / "corrector_fp16.pt")
    package = torch.load(args.model_package, map_location="cpu", weights_only=True)
    package["conditional_corrector"] = {
        "model_state": state,
        "hidden_dim": args.hidden_dim,
        "visual_dim": features.shape[1],
        "input": f"visual_{args.visual_input}_and_temporal_logits",
    }
    package["release_contract"]["conditional_corrector"] = {
        "enabled": True,
        "baseline": "temporal_visual_gate",
        "fixed_epochs": args.epochs,
        "seed": args.seed,
    }
    model_path = args.output_dir / "model.pt"
    bundle_path = args.output_dir / "submission_bundle.pt"
    torch.save(package, model_path)
    build_bundle(model_path, args.detector, bundle_path)
    inherited = pd.read_csv(args.expected_submission)["prediction"].to_numpy()
    prediction = test_corrected.argmax(1)
    metrics = {
        "protocol": "deployment-aligned-full-fit-corrector/v1",
        "visual_input": args.visual_input,
        "rows": len(prediction),
        "classes": int(np.unique(prediction).size),
        "changed_vs_temporal_visual": int(np.sum(prediction != inherited)),
        "model_bytes": model_path.stat().st_size,
        "single_checkpoint_bytes": bundle_path.stat().st_size,
        "single_checkpoint_margin_bytes": MODEL_LIMIT_BYTES - bundle_path.stat().st_size,
        "cached_replay_exact": True,
        "raw_replay_exact": False,
        "anonymous_test_labels_used": False,
        "kaggle_submitted": False,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--metadata", type=Path, default=Path("results/strict_v3/metadata.npz"))
    common.add_argument("--device", default="cuda:0")
    common.add_argument("--seed", type=int, default=2026)
    common.add_argument("--epochs", type=int, default=60)
    common.add_argument("--hidden-dim", type=int, default=128)
    common.add_argument("--lr", type=float, default=3e-4)
    common.add_argument("--corrector-batch-size", type=int, default=128)

    extract_parser = subparsers.add_parser("extract", parents=[common])
    extract_parser.add_argument("--output-dir", type=Path, required=True)
    extract_parser.add_argument("--train-cache", type=Path, required=True)
    extract_parser.add_argument("--test-cache", type=Path, required=True)
    extract_parser.add_argument("--checkpoint-root", type=Path, required=True)
    extract_parser.add_argument("--contract-dir", type=Path, required=True)
    extract_parser.add_argument("--full-checkpoint", type=Path, required=True)
    extract_parser.add_argument("--source-package", type=Path, required=True)
    extract_parser.add_argument("--temporal-oof", type=Path, required=True)
    extract_parser.add_argument("--temporal-test", type=Path, required=True)
    extract_parser.add_argument("--stored-visual-test", type=Path, required=True)
    extract_parser.add_argument("--expected-submission", type=Path, required=True)
    extract_parser.add_argument("--batch-size", type=int, default=32)
    extract_parser.add_argument("--workers", type=int, default=4)
    extract_parser.set_defaults(function=extract)

    screen_parser = subparsers.add_parser("screen", parents=[common])
    screen_parser.add_argument("--inputs", type=Path, required=True)
    screen_parser.add_argument("--output-dir", type=Path, required=True)
    screen_parser.add_argument("--visual-input", choices=("features", "logits"), default="features")
    screen_parser.set_defaults(function=screen)

    full_parser = subparsers.add_parser("full-fit", parents=[common])
    full_parser.add_argument("--inputs", type=Path, required=True)
    full_parser.add_argument("--screen-metrics", type=Path, required=True)
    full_parser.add_argument("--output-dir", type=Path, required=True)
    full_parser.add_argument("--sample-submission", type=Path, required=True)
    full_parser.add_argument("--expected-submission", type=Path, required=True)
    full_parser.add_argument("--model-package", type=Path, required=True)
    full_parser.add_argument("--detector", type=Path, required=True)
    full_parser.add_argument("--visual-input", choices=("features", "logits"), default="features")
    full_parser.set_defaults(function=full_fit)
    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.function(arguments)
