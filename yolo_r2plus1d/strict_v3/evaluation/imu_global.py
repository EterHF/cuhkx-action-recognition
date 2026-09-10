"""Audit fixed local/world IMU ablation against historical and paired baselines."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from yolo_r2plus1d.strict_v3.evaluation.input_validity_masking import assemble, fuse
from yolo_r2plus1d.strict_v3.evaluation.quality_gate_ablation import (
    clustered_bootstrap,
    file_record,
)
from yolo_r2plus1d.strict_v3.evaluation.visual_quantization import FOLDS
from yolo_r2plus1d.strict_v3.models.imu import imu_blend
from yolo_r2plus1d.strict_v3.training.conditional_corrector import (
    classification_metrics,
    prediction_delta,
)
from yolo_r2plus1d.strict_v3.training.deployment_corrector import promotion_passed


def load_oof(root, users, valid, seed, view):
    output = np.full((len(users), 40), np.nan, dtype=np.float32)
    checkpoints = []
    for fold, held_users in FOLDS.items():
        folder = root / view / f"seed{seed}" / f"fold{fold}"
        rows = np.flatnonzero(np.isin(users, held_users))
        with np.load(folder / "held.npz", allow_pickle=False) as data:
            if not np.array_equal(rows, data["rows"]) or data["logits"].shape != (len(rows), 40):
                raise ValueError("IMU held row mismatch")
            output[rows] = data["logits"]
        checkpoint = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
        expected = set(users[~np.isin(users, held_users) & valid])
        if set(checkpoint["train_users"]) != expected or expected & set(held_users):
            raise ValueError("IMU train subject exclusion failed")
        if any(checkpoint[k] != v for k, v in
               {"view": view, "seed": seed, "epochs": 80, "fold": fold}.items()):
            raise ValueError("IMU checkpoint recipe differs from preregistration")
        checkpoints.append(file_record(folder / "model.pt"))
    if not np.isfinite(output).all():
        raise ValueError("Incomplete/nonfinite IMU OOF")
    return output, checkpoints


def audit_pair(base, imu, labels, users, eligible, seed):
    candidate = imu_blend(torch.from_numpy(base), torch.from_numpy(imu),
                          torch.from_numpy(eligible)).numpy()
    bp, cp = base.argmax(1), candidate.argmax(1)
    before, after = [classification_metrics(p, labels, users) for p in (bp, cp)]
    delta = prediction_delta(bp, cp, labels)
    folds = [{"fold": fold, **prediction_delta(bp[np.isin(users, held)], cp[np.isin(users, held)],
                                              labels[np.isin(users, held)])}
             for fold, held in FOLDS.items()]
    bootstrap = clustered_bootstrap(bp, cp, labels, users, 2026, 10000)
    unchanged = np.array_equal(candidate[~eligible], base[~eligible])
    passed = promotion_passed(folds, delta, before, after) and unchanged
    if seed == 2026:
        passed &= bootstrap["accuracy_delta_95_percentile_interval"][0] > 0
    return {"baseline": before, "candidate": after, "delta": delta, "folds": folds,
            "bootstrap": bootstrap, "unchanged_outside_mask": unchanged,
            "seed_gate_passed": bool(passed)}, candidate


def main():
    torch.set_num_threads(2)
    root = Path("runs/experiments/imu_global_v1")
    output = Path("results/experiments/imu_global")
    meta_path = Path("results/strict_v3/metadata.npz")
    validity_path = Path("runs/experiments/input_validity_masking_v1/validity.npz")
    with np.load(meta_path, allow_pickle=False) as data:
        labels, users, keys = data["train_y"], data["train_users"], data["train_keys"]
    with np.load(validity_path, allow_pickle=False) as data:
        primary = data["all_primary"]
    sensor_path = root / "inputs/sensor_valid.npy"
    valid = np.load(sensor_path, allow_pickle=False).sum(1) >= 3
    eligible = primary & valid
    branch_root = Path("runs/experiments/input_validity_masking_v1")
    visual = assemble(branch_root / "visual/control", "best_val_logits.npy", users)
    temporal = assemble(branch_root / "temporal/control", "val_logits.npy", users)
    release = torch.load("checkpoints/strict_v3/model.pt", map_location="cpu", weights_only=True)["release_contract"]
    baselines = {"canonical": np.load("results/strict_v3/train_oof_logits.npy", allow_pickle=False),
                 "current_paired_tv": fuse(visual, temporal, release)}
    report = {"anonymous_data_loaded": False, "seeds": {}, "eligible_rows": int(eligible.sum()),
              "inputs": [file_record(p) for p in (meta_path, validity_path, sensor_path,
                                                  root / "inputs/signals.npy")],
              "known_limitation": "Baseline upstream encoders saw target labels; sensor models exclude held users."}
    saved = {"labels": labels, "users": users, "keys": keys, "eligible": eligible, **baselines}
    standalone = {view: [] for view in ("local", "global")}
    passed = True
    for seed in (2026, 2027, 2028):
        report["seeds"][str(seed)] = entry = {}
        for view in ("local", "global"):
            logits, checkpoints = load_oof(root, users, valid, seed, view)
            standalone[view].append(float(np.mean(logits[valid].argmax(1) == labels[valid])))
            entry[view] = {"standalone": classification_metrics(logits[valid].argmax(1), labels[valid], users[valid]),
                           "checkpoints": checkpoints, "comparisons": {}}
            saved[f"{view}_seed{seed}"] = logits
            for name, baseline in baselines.items():
                metrics, candidate = audit_pair(baseline, logits, labels, users, eligible, seed)
                entry[view]["comparisons"][name] = metrics
                saved[f"{name}_{view}_seed{seed}"] = candidate
                if view == "global":
                    passed &= metrics["seed_gate_passed"]
    report["standalone_mean_accuracy"] = {k: float(np.mean(v)) for k, v in standalone.items()}
    report["global_beats_local"] = report["standalone_mean_accuracy"]["global"] > report["standalone_mean_accuracy"]["local"]
    passed &= report["global_beats_local"]
    report["oof_gate_passed"] = bool(passed)
    report["decision"] = "proceed to full-fit/replay" if passed else "reject; no test inference or submission"
    np.savez_compressed(output / "predictions.npz", **saved)
    report["prediction_artifact"] = file_record(output / "predictions.npz")
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"decision": report["decision"], "standalone_mean": report["standalone_mean_accuracy"],
                      "global_net_canonical": [report["seeds"][str(s)]["global"]["comparisons"]["canonical"]["delta"]
                                               for s in (2026, 2027, 2028)]}, indent=2))


if __name__ == "__main__":
    main()
