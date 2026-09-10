"""Fixed-recipe deployment corrector trained/routed only on available inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from yolo_r2plus1d.strict_v3.evaluation.quality_gate_ablation import (
    clustered_bootstrap,
    file_record,
)
from yolo_r2plus1d.strict_v3.evaluation.visual_quantization import FOLDS
from yolo_r2plus1d.strict_v3.models.conditional_corrector import available_correction
from yolo_r2plus1d.strict_v3.training.conditional_corrector import (
    classification_metrics,
    prediction_delta,
)
from yolo_r2plus1d.strict_v3.training.deployment_corrector import fit_corrector, promotion_passed


def available_split(users, both_valid, held_users):
    held = np.isin(users, held_users)
    return np.flatnonzero(~held & both_valid), np.flatnonzero(held)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.epochs, args.hidden_dim, args.lr, args.corrector_batch_size = 60, 128, 3e-4, 128
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    root = Path("runs/experiments/conditional_corrector_deployment_inputs_v1")
    meta_path = Path("results/strict_v3/metadata.npz")
    validity_path = Path("runs/experiments/input_validity_masking_v1/validity.npz")
    with np.load(meta_path, allow_pickle=False) as data:
        labels, users = data["train_y"], data["train_users"]
    with np.load(validity_path, allow_pickle=False) as data:
        valid = data["all_primary"]
    visual = np.load(root / "visual_oof_logits.npy", allow_pickle=False)
    temporal = np.load(root / "temporal_oof_logits.npy", allow_pickle=False)
    baseline = np.load(root / "baseline_oof_logits.npy", allow_pickle=False)
    if any(x.shape != (len(users), 40) or not np.isfinite(x).all()
           for x in (visual, temporal, baseline)):
        raise ValueError("invalid logit arrays")
    candidate = baseline.copy()
    seen = np.zeros(len(users), dtype=bool)
    folds = []
    device = torch.device(args.device)
    for fold, held_users in FOLDS.items():
        train, held = available_split(users, valid, held_users)
        if seen[held].any() or set(users[train]) & set(held_users):
            raise ValueError("duplicate held rows or subject leakage")
        model = fit_corrector(visual, temporal, baseline, labels, train, args, device)
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        path = args.output / f"fold{fold}.pt"
        torch.save({"model_state": state, "seed": args.seed, "fold": fold,
                    "train_users": np.unique(users[train]).tolist(),
                    "train_rows": train.tolist(), "epochs": args.epochs}, path)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model_state"])
        candidate[held] = available_correction(
            model, torch.from_numpy(visual[held]).to(device),
            torch.from_numpy(temporal[held]).to(device),
            torch.from_numpy(baseline[held]).to(device),
            torch.from_numpy(valid[held]).to(device),
        ).cpu().numpy()
        seen[held] = True
        folds.append({"fold": fold, "train_rows": len(train), "held_rows": len(held),
                      **prediction_delta(baseline[held].argmax(1), candidate[held].argmax(1), labels[held])})
        print(json.dumps(folds[-1]), flush=True)
    if not seen.all() or not np.array_equal(candidate[~valid], baseline[~valid]):
        raise ValueError("incomplete coverage or changes on unavailable rows")
    before = classification_metrics(baseline.argmax(1), labels, users)
    after = classification_metrics(candidate.argmax(1), labels, users)
    delta = prediction_delta(baseline.argmax(1), candidate.argmax(1), labels)
    bootstrap = clustered_bootstrap(baseline.argmax(1), candidate.argmax(1), labels, users, 2026, 10000)
    passed = promotion_passed(folds, delta, before, after)
    if args.seed == 2026:
        passed &= bootstrap["accuracy_delta_95_percentile_interval"][0] > 0
    metrics = {"seed": args.seed, "baseline": before, "candidate": after,
               "delta": delta, "folds": folds, "bootstrap": bootstrap,
               "seed_gate_passed": bool(passed), "unavailable_rows_unchanged": True,
               "test_data_loaded": False, "train_only_both_valid": True,
               "inputs": [file_record(p) for p in (meta_path, validity_path,
                   root / "visual_oof_logits.npy", root / "temporal_oof_logits.npy",
                   root / "baseline_oof_logits.npy")],
               "checkpoints": [file_record(args.output / f"fold{f}.pt") for f in FOLDS]}
    np.save(args.output / "corrected_oof.npy", candidate)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({k: metrics[k] for k in ("seed", "delta", "bootstrap", "seed_gate_passed")}), flush=True)


if __name__ == "__main__":
    main()
