"""Evaluate two fixed triplets under one preregistered unanimity rule."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from yolo_r2plus1d.strict_v3.evaluation.quality_gate_ablation import (
    clustered_bootstrap,
    file_record,
)
from yolo_r2plus1d.strict_v3.evaluation.visual_quantization import FOLDS
from yolo_r2plus1d.strict_v3.models.branch_selector import selector_inputs, unanimous_selection
from yolo_r2plus1d.strict_v3.training.conditional_corrector import (
    classification_metrics,
    prediction_delta,
)
from yolo_r2plus1d.strict_v3.training.deployment_corrector import promotion_passed


def main():
    torch.set_num_threads(2)
    root = Path("runs/experiments/branch_selector_v1")
    inputs = Path("runs/experiments/conditional_corrector_deployment_inputs_v1")
    output = Path("results/experiments/selector_consensus")
    with np.load("results/strict_v3/metadata.npz", allow_pickle=False) as m:
        y, users, keys = m["train_y"], m["train_users"], m["train_keys"]
    with np.load("runs/experiments/input_validity_masking_v1/validity.npz", allow_pickle=False) as m:
        valid = torch.from_numpy(m["all_primary"])
    visual, temporal, baseline = [torch.from_numpy(np.load(inputs / f"{name}_oof_logits.npy", allow_pickle=False))
                                  for name in ("visual", "temporal", "baseline")]
    features, ti, vi = selector_inputs(visual, temporal, baseline)
    eligible = valid & (ti != vi) & (features[:, 5] < .8)
    base_pred = baseline.argmax(1).numpy()
    before = classification_metrics(base_pred, y, users)
    report = {"anonymous_data_loaded": False, "triplets": {}, "baseline": before,
              "eligible_rows": int(eligible.sum()), "oof_gate_passed": True}
    saved = {"baseline": baseline.numpy(), "labels": y, "users": users, "keys": keys,
             "eligible": eligible.numpy()}
    for name, seeds in {"deployment": (2026, 2027, 2028), "replication": (2029, 2030, 2031)}.items():
        members = []
        records = []
        for seed in seeds:
            path = root / f"seed{seed}/selected_oof.npy"
            members.append(torch.from_numpy(np.load(path, allow_pickle=False)))
            records.append(file_record(path))
            for fold, held_users in FOLDS.items():
                checkpoint_path = root / f"seed{seed}/fold{fold}.pt"
                ck = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
                rows = np.asarray(ck["train_rows"])
                expected = (~np.isin(users, held_users) & valid.numpy() & (ti != vi).numpy()
                            & ((y == ti.numpy()) | (y == vi.numpy())))
                if not np.array_equal(rows, np.flatnonzero(expected)):
                    raise ValueError("selector training rows violate frozen eligibility or held-user exclusion")
                if ck["seed"] != seed or ck["epochs"] != 100 or ck["fold"] != fold:
                    raise ValueError("selector checkpoint recipe mismatch")
                records.append(file_record(checkpoint_path))
        candidate = unanimous_selection(baseline, members)
        if not torch.equal(candidate[~eligible], baseline[~eligible]):
            raise ValueError("consensus changes protected rows")
        cp = candidate.argmax(1).numpy()
        after = classification_metrics(cp, y, users)
        delta = prediction_delta(base_pred, cp, y)
        folds = [{"fold": f, **prediction_delta(base_pred[np.isin(users, held)], cp[np.isin(users, held)],
                                               y[np.isin(users, held)])} for f, held in FOLDS.items()]
        bootstrap = clustered_bootstrap(base_pred, cp, y, users, 2026, 10000)
        passed = (promotion_passed(folds, delta, before, after)
                  and bootstrap["accuracy_delta_95_percentile_interval"][0] > 0)
        report["oof_gate_passed"] &= bool(passed)
        report["triplets"][name] = {"seeds": seeds, "candidate": after, "delta": delta, "folds": folds,
                                    "bootstrap": bootstrap, "gate_passed": bool(passed), "inputs": records}
        saved[name] = candidate.numpy()
    report["decision"] = "proceed to fullfit, bundle and raw replay" if report["oof_gate_passed"] else "reject; no fullfit, test inference or submission"
    np.savez_compressed(output / "predictions.npz", **saved)
    report["prediction_artifact"] = file_record(output / "predictions.npz")
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"decision": report["decision"], "triplets": {
        k: {n: v[n] for n in ("delta", "folds", "bootstrap", "gate_passed")}
        for k, v in report["triplets"].items()}}, indent=2))


if __name__ == "__main__":
    main()
