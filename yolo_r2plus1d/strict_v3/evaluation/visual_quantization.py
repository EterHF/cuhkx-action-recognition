#!/usr/bin/env python3
"""Evaluate Visual quantization policies under the frozen T+V fusion contract."""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from yolo_r2plus1d.strict_v3.paths import RESULT_DIR
from yolo_r2plus1d.strict_v3.release.blend import apply_gate
from yolo_r2plus1d.strict_v3.release.bundle import MODEL_LIMIT_BYTES, build_bundle
from yolo_r2plus1d.strict_v3.release.prune_fusion import temporal_visual_package
from yolo_r2plus1d.strict_v3.training.public_finetune import (
    VideoDataset,
    dequantize_state,
    evaluate,
    make_model,
    quantize_signed,
)

FOLDS = {
    "A": (1, 6, 17, 22),
    "B": (2, 7, 18, 23),
    "C": (3, 8, 19, 24),
    "D": (4, 9, 20),
    "E": (5, 16, 21),
}
POLICIES = (
    "fp16",
    "uniform4",
    "uniform5",
    "uniform6",
    "uniform8",
    "late8",
    "mixed4_late8",
    "mixed6_late8",
)


def bits_for(policy: str, name: str) -> int | None:
    if policy == "fp16":
        return None
    if policy.startswith("uniform"):
        return int(policy.removeprefix("uniform"))
    late = name.startswith(("encoder.layer4.", "head."))
    if policy == "late8":
        return 8 if late else 5
    if policy == "mixed4_late8":
        return 8 if late else 4
    if policy == "mixed6_late8":
        return 8 if late else 6
    raise ValueError(f"unknown quantization policy: {policy}")


def quantized_state(
    state: Mapping[str, torch.Tensor], policy: str
) -> dict[str, object]:
    output: dict[str, object] = {}
    for name, value in state.items():
        bits = bits_for(policy, name)
        if bits is not None and value.is_floating_point() and value.ndim >= 2:
            output[name] = quantize_signed(value, bits)
        else:
            output[name] = (
                value.detach().cpu().half()
                if value.is_floating_point()
                else value.detach().cpu()
            )
    return output


def prediction_delta(
    baseline: np.ndarray, candidate: np.ndarray, labels: np.ndarray
) -> dict[str, int]:
    base_correct = baseline == labels
    candidate_correct = candidate == labels
    corrected = int(np.sum(~base_correct & candidate_correct))
    broken = int(np.sum(base_correct & ~candidate_correct))
    return {"corrected": corrected, "broken": broken, "net": corrected - broken}


def build_policy_bundle(
    policy: str,
    source_package: Path,
    full_checkpoint: Path,
    detector: Path,
    output_dir: Path,
) -> dict:
    source = torch.load(source_package, map_location="cpu", weights_only=True)
    checkpoint = torch.load(full_checkpoint, map_location="cpu", weights_only=True)
    package = temporal_visual_package(source)
    packed = quantized_state(checkpoint["model_state"], policy)
    package["visual_member0"]["model_state_packed"] = packed
    package["visual_member0"]["bits"] = policy
    package["release_contract"]["visual_quantization_policy"] = policy
    model_path = output_dir / f"temporal_visual_{policy}.pt"
    bundle_path = output_dir / f"temporal_visual_{policy}_bundle.pt"
    torch.save(package, model_path)
    build_bundle(model_path, detector, bundle_path)
    return {
        "model_bytes": model_path.stat().st_size,
        "single_checkpoint_bytes": bundle_path.stat().st_size,
        "single_checkpoint_margin_bytes": MODEL_LIMIT_BYTES - bundle_path.stat().st_size,
    }


def run(args: argparse.Namespace) -> None:
    if len(set(args.policies)) != len(args.policies):
        raise ValueError("quantization policies must be unique")
    if "fp16" not in args.policies or "uniform5" not in args.policies:
        raise ValueError("evaluation requires fp16 and uniform5 anchors")
    with np.load(args.metadata) as metadata:
        labels = metadata["train_y"].astype(np.int64)
        users = metadata["train_users"].astype(np.int64)
    logits = {
        policy: np.empty((len(labels), 40), dtype=np.float32)
        for policy in args.policies
    }
    fused = {policy: np.empty_like(value) for policy, value in logits.items()}
    fold_metrics = []
    device = torch.device(args.device)
    criterion = nn.CrossEntropyLoss().to(device)
    for fold, held_users in FOLDS.items():
        indices = np.flatnonzero(np.isin(users, held_users))
        contract = json.loads(
            (args.contract_dir / f"contract_fold{fold}.json").read_text()
        )
        mean = torch.tensor(contract["mean"], dtype=torch.float32).view(1, 4, 1, 1)
        std = torch.tensor(contract["std"], dtype=torch.float32).view(1, 4, 1, 1)
        loader = DataLoader(
            VideoDataset(
                args.cache,
                indices,
                labels,
                False,
                False,
                "all",
                mean,
                std,
                False,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
        )
        checkpoint = torch.load(
            args.checkpoint_root / f"affine_head_fold{fold}/best_fp16.pt",
            map_location="cpu",
            weights_only=True,
        )
        state = checkpoint["model_state"]
        temporal = np.load(
            args.temporal_root / f"logit_residual_tcn_cv_{fold}/val_logits.npy"
        )
        contract_weights = dict(args.release_contract["weights"])
        contract_weights["fusion"] = 0.0
        for policy in args.policies:
            model = make_model()
            packed = quantized_state(state, policy)
            model.load_state_dict(dequantize_state(packed), strict=True)
            model.to(device)
            _, _, held_logits, _ = evaluate(
                model, loader, criterion, device, use_tta=True
            )
            logits[policy][indices] = held_logits
            visual = held_logits * float(
                args.release_contract["visual_package_output_scale"]
            )
            fused_logits, _ = apply_gate(
                {
                    "fusion": np.zeros_like(visual),
                    "visual": visual,
                    "temporal": temporal,
                },
                args.release_contract["temperatures"],
                contract_weights,
            )
            fused[policy][indices] = fused_logits
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        anchor_prediction = fused["uniform5"][indices].argmax(1)
        fold_metrics.append(
            {
                "fold": fold,
                "rows": len(indices),
                "policies": {
                    policy: {
                        "visual_accuracy": float(
                            np.mean(logits[policy][indices].argmax(1) == labels[indices])
                        ),
                        "fused_accuracy": float(
                            np.mean(fused[policy][indices].argmax(1) == labels[indices])
                        ),
                        "fused_delta_vs_uniform5": prediction_delta(
                            anchor_prediction,
                            fused[policy][indices].argmax(1),
                            labels[indices],
                        ),
                    }
                    for policy in args.policies
                },
            }
        )
        print(json.dumps(fold_metrics[-1]), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    package_sizes = {
        policy: build_policy_bundle(
            policy,
            args.source_package,
            args.full_checkpoint,
            args.detector,
            args.output_dir,
        )
        for policy in args.policies
        if policy != "fp16"
    }
    anchor_visual = logits["uniform5"].argmax(1)
    anchor_fused = fused["uniform5"].argmax(1)
    aggregate = {
        policy: {
            "visual_correct": int(np.sum(logits[policy].argmax(1) == labels)),
            "visual_accuracy": float(np.mean(logits[policy].argmax(1) == labels)),
            "visual_delta_vs_uniform5": prediction_delta(
                anchor_visual, logits[policy].argmax(1), labels
            ),
            "fused_correct": int(np.sum(fused[policy].argmax(1) == labels)),
            "fused_accuracy": float(np.mean(fused[policy].argmax(1) == labels)),
            "fused_delta_vs_uniform5": prediction_delta(
                anchor_fused, fused[policy].argmax(1), labels
            ),
            "package": package_sizes.get(policy),
        }
        for policy in args.policies
    }
    for policy in args.policies:
        np.save(args.output_dir / f"visual_oof_{policy}.npy", logits[policy])
        np.save(args.output_dir / f"fused_oof_{policy}.npy", fused[policy])
    metrics = {
        "protocol": "strictv3-visual-quantization-budget/v1",
        "policies": list(args.policies),
        "aggregate": aggregate,
        "folds": fold_metrics,
        "visual_package_output_scale": args.release_contract[
            "visual_package_output_scale"
        ],
        "temperatures": args.release_contract["temperatures"],
        "weights": {**args.release_contract["weights"], "fusion": 0.0},
        "source_fp16_only": True,
        "anonymous_test_accessed": False,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"aggregate": aggregate}, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--cache", type=Path, required=True)
    result.add_argument("--checkpoint-root", type=Path, required=True)
    result.add_argument("--contract-dir", type=Path, required=True)
    result.add_argument("--temporal-root", type=Path, required=True)
    result.add_argument("--full-checkpoint", type=Path, required=True)
    result.add_argument("--source-package", type=Path, required=True)
    result.add_argument("--detector", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--metadata", type=Path, default=RESULT_DIR / "metadata.npz")
    result.add_argument("--policies", nargs="+", choices=POLICIES, default=POLICIES)
    result.add_argument("--batch-size", type=int, default=32)
    result.add_argument("--workers", type=int, default=4)
    result.add_argument("--device", default="cuda:0")
    args = result.parse_args()
    package = torch.load(args.source_package, map_location="cpu", weights_only=True)
    args.release_contract = copy.deepcopy(package["release_contract"])
    return args


if __name__ == "__main__":
    run(parser())
