#!/usr/bin/env python3
"""Quantization sensitivity proxy for traceable external Visual checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from yolo_r2plus1d.strict_v3.evaluation.visual_quantization import (
    FOLDS,
    prediction_delta,
    quantized_state,
)
from yolo_r2plus1d.strict_v3.paths import RESULT_DIR
from yolo_r2plus1d.strict_v3.training.public_finetune import (
    VideoDataset,
    dequantize_state,
    evaluate,
    make_model,
)

POLICIES = ("fp16", "uniform4", "uniform5", "uniform6", "uniform8", "mixed4_late8")


def run(args: argparse.Namespace) -> None:
    with np.load(args.metadata) as metadata:
        labels = metadata["train_y"].astype(np.int64)
        users = metadata["train_users"].astype(np.int64)
    logits = {
        policy: np.empty((len(labels), 40), dtype=np.float32)
        for policy in POLICIES
    }
    device = torch.device(args.device)
    criterion = nn.CrossEntropyLoss().to(device)
    folds = []
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
                args.normalize_after_affine,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
        )
        checkpoint_path = args.checkpoint_root / f"fold{fold}/best_fp16.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = checkpoint["model_state"]
        if not all(value.is_floating_point() for value in state.values() if value.ndim >= 2):
            raise RuntimeError(f"{checkpoint_path} is not an original float checkpoint")
        for policy in POLICIES:
            model = make_model()
            model.load_state_dict(
                dequantize_state(quantized_state(state, policy)), strict=True
            )
            model.to(device)
            _, _, held_logits, _ = evaluate(
                model, loader, criterion, device, use_tta=True
            )
            logits[policy][indices] = held_logits
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        fold_result = {
            "fold": fold,
            "rows": len(indices),
            "correct": {
                policy: int(
                    np.sum(logits[policy][indices].argmax(1) == labels[indices])
                )
                for policy in POLICIES
            },
        }
        folds.append(fold_result)
        print(json.dumps(fold_result), flush=True)

    fp16_prediction = logits["fp16"].argmax(1)
    int4_prediction = logits["uniform4"].argmax(1)
    aggregate = {}
    for policy in POLICIES:
        prediction = logits[policy].argmax(1)
        aggregate[policy] = {
            "correct": int(np.sum(prediction == labels)),
            "accuracy": float(np.mean(prediction == labels)),
            "delta_vs_fp16": prediction_delta(fp16_prediction, prediction, labels),
            "delta_vs_int4": prediction_delta(int4_prediction, prediction, labels),
        }
        np.save(args.output_dir / f"oof_{policy}.npy", logits[policy])
    metrics = {
        "protocol": "external-visual-quantization-proxy/v1",
        "source": str(args.checkpoint_root),
        "source_is_attachment_1998": False,
        "original_float_checkpoints_used": True,
        "normalize_after_affine": args.normalize_after_affine,
        "aggregate": aggregate,
        "folds": folds,
        "final_fusion_evaluated": False,
        "reason_final_fusion_not_evaluated": (
            "this traceable proxy has no preregistered frozen external-branch fusion rule"
        ),
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
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--metadata", type=Path, default=RESULT_DIR / "metadata.npz")
    result.add_argument("--normalize-after-affine", action="store_true")
    result.add_argument("--batch-size", type=int, default=32)
    result.add_argument("--workers", type=int, default=4)
    result.add_argument("--device", default="cuda:0")
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=False)
    run(arguments)
