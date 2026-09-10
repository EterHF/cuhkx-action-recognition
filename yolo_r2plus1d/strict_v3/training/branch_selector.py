"""Fixed seed/subject-fold screen for the confidence-bounded binary selector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from yolo_r2plus1d.strict_v3.evaluation.quality_gate_ablation import (
    clustered_bootstrap,
    file_record,
)
from yolo_r2plus1d.strict_v3.evaluation.visual_quantization import FOLDS
from yolo_r2plus1d.strict_v3.models.branch_selector import (
    BranchSelector,
    select_branches,
    selector_inputs,
)
from yolo_r2plus1d.strict_v3.training.base import seed_everything
from yolo_r2plus1d.strict_v3.training.conditional_corrector import (
    classification_metrics,
    prediction_delta,
)
from yolo_r2plus1d.strict_v3.training.deployment_corrector import promotion_passed


def fit_selector(visual, temporal, baseline, labels, valid, train, seed):
    seed_everything(seed)
    features, ti, vi = selector_inputs(visual, temporal, baseline)
    selected = train & valid & (ti != vi) & ((labels == ti) | (labels == vi))
    target = (labels[selected] == vi[selected]).long()
    if target.unique().numel() != 2:
        raise ValueError("selector training needs both branch outcomes")
    model = BranchSelector()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.01)
    loader = DataLoader(TensorDataset(features[selected], target), batch_size=64, shuffle=True)
    for _ in range(100):
        model.train()
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(model(x), y).backward()
            optimizer.step()
    return model.eval(), selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    root = Path("runs/experiments/conditional_corrector_deployment_inputs_v1")
    meta_path = Path("results/strict_v3/metadata.npz")
    mask_path = Path("runs/experiments/input_validity_masking_v1/validity.npz")
    with np.load(meta_path, allow_pickle=False) as data:
        y, users = data["train_y"], data["train_users"]
    with np.load(mask_path, allow_pickle=False) as data:
        valid = torch.from_numpy(data["all_primary"])
    visual, temporal, base = [torch.from_numpy(np.load(root / f"{name}_oof_logits.npy", allow_pickle=False))
                              for name in ("visual", "temporal", "baseline")]
    labels = torch.from_numpy(y)
    candidate = base.clone()
    folds = []
    for fold, held_users in FOLDS.items():
        held = torch.from_numpy(np.isin(users, held_users))
        model, selected = fit_selector(visual, temporal, base, labels, valid, ~held, args.seed)
        if bool((selected & held).any()):
            raise ValueError("held subjects in selector training")
        path = args.output / f"fold{fold}.pt"
        torch.save({"model_state": model.state_dict(), "seed": args.seed, "fold": fold,
                    "epochs": 100, "train_rows": selected.nonzero().flatten().tolist(),
                    "train_users": np.unique(users[selected.numpy()]).tolist()}, path)
        model.load_state_dict(torch.load(path, weights_only=True)["model_state"])
        candidate[held] = select_branches(model, visual[held], temporal[held], base[held], valid[held])
        folds.append({"fold": fold, "train_rows": int(selected.sum()),
                      **prediction_delta(base[held].argmax(1).numpy(), candidate[held].argmax(1).numpy(), y[held.numpy()])})
        print(json.dumps(folds[-1]), flush=True)
    bp, cp = base.argmax(1).numpy(), candidate.argmax(1).numpy()
    before, after = [classification_metrics(p, y, users) for p in (bp, cp)]
    delta = prediction_delta(bp, cp, y)
    bootstrap = clustered_bootstrap(bp, cp, y, users, 2026, 10000)
    passed = promotion_passed(folds, delta, before, after)
    if args.seed == 2026:
        passed &= bootstrap["accuracy_delta_95_percentile_interval"][0] > 0
    features, ti, vi = selector_inputs(visual, temporal, base)
    eligible = valid & (ti != vi) & (features[:, 5] < .8)
    if not torch.equal(candidate[~eligible], base[~eligible]):
        raise ValueError("selector changed ineligible rows")
    metrics = {"seed": args.seed, "baseline": before, "candidate": after, "delta": delta,
               "folds": folds, "bootstrap": bootstrap, "eligible_rows": int(eligible.sum()),
               "seed_gate_passed": bool(passed), "ineligible_logits_unchanged": True,
               "test_data_loaded": False, "inputs": [file_record(p) for p in (meta_path, mask_path,
                   root / "visual_oof_logits.npy", root / "temporal_oof_logits.npy", root / "baseline_oof_logits.npy")],
               "checkpoints": [file_record(args.output / f"fold{fold}.pt") for fold in FOLDS]}
    np.save(args.output / "selected_oof.npy", candidate.numpy())
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({k: metrics[k] for k in ("seed", "delta", "bootstrap", "seed_gate_passed")}), flush=True)


if __name__ == "__main__":
    main()
