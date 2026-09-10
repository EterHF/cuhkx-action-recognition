"""Audit the preregistered missing-primary thermal fallback across three seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from yolo_r2plus1d.strict_v3.evaluation.input_validity_masking import (
    assemble as assemble_branch,
)
from yolo_r2plus1d.strict_v3.evaluation.input_validity_masking import fuse
from yolo_r2plus1d.strict_v3.evaluation.quality_gate_ablation import (
    clustered_bootstrap,
    exact_mcnemar_p,
    file_record,
)
from yolo_r2plus1d.strict_v3.evaluation.visual_quantization import FOLDS
from yolo_r2plus1d.strict_v3.models.thermal import thermal_fallback
from yolo_r2plus1d.strict_v3.training.conditional_corrector import (
    classification_metrics,
    prediction_delta,
)


def assemble(root: Path, users: np.ndarray, seed: int, temporal_shift: bool) -> np.ndarray:
    logits = np.full((len(users), 40), np.nan, dtype=np.float32)
    for fold, held_users in FOLDS.items():
        with np.load(root / f"fold{fold}/held.npz", allow_pickle=False) as data:
            rows = np.flatnonzero(np.isin(users, held_users))
            if not np.array_equal(rows, data["rows"]):
                raise ValueError(f"held rows mismatch in {root}/{fold}")
            if data["logits"].shape != (len(rows), 40):
                raise ValueError("invalid thermal held logit shape")
            logits[rows] = data["logits"]
        checkpoint = torch.load(root / f"fold{fold}/model.pt", map_location="cpu", weights_only=True)
        if set(checkpoint["train_users"]) & set(held_users):
            raise ValueError("thermal checkpoint trained on a held subject")
        if checkpoint["epochs"] != 15 or checkpoint["fold"] != fold:
            raise ValueError("checkpoint differs from the registered fold/epoch contract")
        if checkpoint["seed"] != seed or bool(checkpoint.get("temporal_shift", False)) != temporal_shift:
            raise ValueError("checkpoint differs from the registered seed/architecture")
    if not np.isfinite(logits).all():
        raise ValueError("OOF coverage incomplete or nonfinite")
    return logits


def audit_pair(baseline, thermal, labels, users, missing, valid):
    result = thermal_fallback(torch.from_numpy(baseline), torch.from_numpy(thermal),
                              torch.from_numpy(missing), torch.from_numpy(valid)).numpy()
    control, candidate = baseline.argmax(1), result.argmax(1)
    before = classification_metrics(control, labels, users)
    after = classification_metrics(candidate, labels, users)
    delta = prediction_delta(control, candidate, labels)
    folds = {
        fold: prediction_delta(control[np.isin(users, held)], candidate[np.isin(users, held)],
                               labels[np.isin(users, held)])
        for fold, held in FOLDS.items()
    }
    bootstrap = clustered_bootstrap(control, candidate, labels, users, 2026, 10000)
    gates = {
        "net_positive": delta["net"] > 0,
        "five_folds_non_degrading": all(value["net"] >= 0 for value in folds.values()),
        "subject_macro_positive": after["subject_macro_accuracy"] > before["subject_macro_accuracy"],
        "worst_user_improves": after["worst_user_accuracy"] > before["worst_user_accuracy"],
        "primary_present_logits_unchanged": np.array_equal(result[~missing], baseline[~missing]),
    }
    return {
        "baseline": before, "candidate": after, "delta": delta, "folds": folds,
        "missing_primary_correct": int(np.sum(candidate[missing] == labels[missing])),
        "baseline_missing_primary_correct": int(np.sum(control[missing] == labels[missing])),
        "missing_primary_rows": int(missing.sum()),
        "mcnemar_p": exact_mcnemar_p(delta["corrected"], delta["broken"]),
        "bootstrap": bootstrap, "gates": gates,
        "user_net": {
            str(int(u)): int(np.sum(candidate[users == u] == labels[users == u])
                            - np.sum(control[users == u] == labels[users == u]))
            for u in np.unique(users)
        },
    }, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temporal-shift", action="store_true")
    parser.add_argument("--control-output", type=Path)
    args = parser.parse_args()
    meta_path = Path("results/strict_v3/metadata.npz")
    validity_path = Path("runs/experiments/input_validity_masking_v1/validity.npz")
    thermal_valid_path = args.run_root / "inputs/valid.npy"
    baseline_path = Path("results/strict_v3/train_oof_logits.npy")
    paired_path = Path("results/experiments/quality_gate_ablation/predictions.npz")
    with np.load(meta_path, allow_pickle=False) as meta:
        labels, users, keys = meta["train_y"], meta["train_users"], meta["train_keys"]
    with np.load(validity_path, allow_pickle=False) as validity:
        missing = validity["no_primary"]
    valid = np.load(thermal_valid_path, allow_pickle=False)
    baseline = np.load(baseline_path, allow_pickle=False)
    branch_root = Path("runs/experiments/input_validity_masking_v1")
    visual = assemble_branch(branch_root / "visual/control", "best_val_logits.npy", users)
    temporal = assemble_branch(branch_root / "temporal/control", "val_logits.npy", users)
    package = torch.load("checkpoints/strict_v3/model.pt", map_location="cpu", weights_only=True)
    paired_baseline = fuse(visual, temporal, package["release_contract"])
    with np.load(paired_path, allow_pickle=False) as paired:
        if not np.array_equal(paired["key"], keys) or not np.array_equal(
            paired["control_top1"], paired_baseline.argmax(1)
        ):
            raise ValueError("paired baseline does not reproduce archived predictions")
    report = {"anonymous_data_loaded": False, "seeds": {},
              "known_limitation": "Baseline upstream encoders saw target labels; thermal is subject-held-out ImageNet-only.",
              "inputs": {str(p): file_record(p) for p in
                         (meta_path, validity_path, thermal_valid_path, baseline_path, paired_path)}}
    saved = {"labels": labels, "users": users, "keys": keys, "no_primary": missing,
             "thermal_valid": valid, "baseline_logits": baseline}
    passed = True
    control = None
    if args.control_output is not None:
        with np.load(args.control_output / "predictions.npz", allow_pickle=False) as data:
            if not np.array_equal(keys, data["keys"]):
                raise ValueError("architecture control row order differs")
            control = {seed: data[f"candidate_seed{seed}"] for seed in (2026, 2027, 2028)}
    extra_net = 0
    for seed in (2026, 2027, 2028):
        thermal = assemble(args.run_root / f"seed{seed}", users, seed, args.temporal_shift)
        primary, result = audit_pair(baseline, thermal, labels, users, missing, valid)
        paired, _ = audit_pair(paired_baseline, thermal, labels, users, missing, valid)
        passed &= all(primary["gates"].values()) and all(paired["gates"].values())
        if seed == 2026:
            passed &= primary["bootstrap"]["accuracy_delta_95_percentile_interval"][0] > 0
            passed &= paired["bootstrap"]["accuracy_delta_95_percentile_interval"][0] > 0
        report["seeds"][str(seed)] = {
            "canonical_release_comparison": primary, "current_paired_tv_comparison": paired,
            "thermal_valid_metrics": classification_metrics(thermal[valid].argmax(1),
                                                              labels[valid], users[valid]),
        }
        saved[f"thermal_seed{seed}"] = thermal
        saved[f"candidate_seed{seed}"] = result
        if control is not None:
            delta = prediction_delta(control[seed].argmax(1), result.argmax(1), labels)
            report["seeds"][str(seed)]["delta_vs_tsn_control"] = delta
            extra_net += delta["net"]
    if control is not None:
        passed &= extra_net > 0
        report["extra_gate_positive_net_vs_tsn"] = extra_net > 0
        report["sum_net_vs_tsn"] = extra_net
    report["oof_gate_passed"] = bool(passed)
    report["decision"] = "proceed to seed2026 full-fit and package replay" if passed else "reject; no test inference or Kaggle submission"
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "predictions.npz", **saved)
    report["prediction_artifact"] = file_record(args.output / "predictions.npz")
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"decision": report["decision"], "seeds": {
        seed: {"delta": value["canonical_release_comparison"]["delta"],
               "gates": value["canonical_release_comparison"]["gates"],
               "missing_correct": value["canonical_release_comparison"]["missing_primary_correct"]}
        for seed, value in report["seeds"].items()}}, indent=2))


if __name__ == "__main__":
    main()
