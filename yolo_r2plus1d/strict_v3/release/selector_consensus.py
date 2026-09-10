"""Full-fit and raw-input inference for the frozen three-selector consensus."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from yolo_r2plus1d.strict_v3.data.highrate_skeleton_cache import build_clip
from yolo_r2plus1d.strict_v3.data.indexing import discover_train
from yolo_r2plus1d.strict_v3.data.validity import has_decodable_image
from yolo_r2plus1d.strict_v3.evaluation.quality_gate_ablation import file_record
from yolo_r2plus1d.strict_v3.models.branch_selector import (
    BranchSelector,
    select_branches,
    unanimous_selection,
)
from yolo_r2plus1d.strict_v3.release.bundle import build_bundle
from yolo_r2plus1d.strict_v3.release.prune_fusion import temporal_visual_package
from yolo_r2plus1d.strict_v3.training.branch_selector import fit_selector

SELECTOR_CONTRACT = {
    "format": "bounded-selector-unanimity/v1", "seeds": [2026, 2027, 2028],
    "baseline_confidence_threshold": .8, "visual_input_scale": .5,
    "both_primary_required": True, "skeleton_validity_native_center_frames": 256,
    "head_compute_dtype": "float32", "head_device": "cpu",
}


def primary_available(paths):
    depth, infrared, skeleton = paths
    visual = has_decodable_image(depth) or has_decodable_image(infrared)
    return visual and bool(build_clip((0, depth, skeleton, 256))[2].any())


def infer_consensus(package, visual, temporal, baseline, test_root, test_ids, workers):
    if package["release_contract"].get("selector_consensus") != SELECTOR_CONTRACT:
        raise ValueError("unknown or modified selector inference contract")
    states = package["selector_consensus"]
    if len(states) != 3:
        raise ValueError("selector package requires exactly three model states")
    tasks = [(test_root / str(s) / "Depth_Color", test_root / str(s) / "IR",
              test_root / str(s) / "Skeleton") for s in test_ids]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        valid = torch.tensor(list(pool.map(primary_available, tasks, chunksize=8)))
    scale = package["release_contract"]["visual_package_output_scale"]
    if scale != .5:
        raise ValueError("selector expects the registered visual package output scale")
    visual, temporal, baseline = [torch.from_numpy(np.asarray(a, dtype=np.float32))
                                  for a in (visual / scale, temporal, baseline)]
    members = []
    for state in states:
        model = BranchSelector().eval()
        model.load_state_dict(state)
        members.append(select_branches(model, visual, temporal, baseline, valid))
    result = unanimous_selection(baseline, members)
    return result.numpy(), valid.numpy()


def main():
    torch.set_num_threads(2)
    evidence = Path("results/experiments/selector_consensus")
    metrics = json.loads((evidence / "metrics.json").read_text())
    if metrics["oof_gate_passed"] is not True:
        raise RuntimeError("consensus did not pass its locked OOF gates")
    root = Path("runs/experiments/conditional_corrector_deployment_inputs_v1")
    output = Path("runs/experiments/selector_consensus_v1")
    output.mkdir(parents=True, exist_ok=False)
    with np.load("results/strict_v3/metadata.npz", allow_pickle=False) as meta:
        y, keys = meta["train_y"], meta["train_keys"]
    with np.load("runs/experiments/input_validity_masking_v1/validity.npz", allow_pickle=False) as data:
        valid = torch.from_numpy(data["all_primary"])
    samples = discover_train(Path("data/processed/train/HAR/data"))
    if not np.array_equal(keys, [sample[0] for sample in samples]):
        raise ValueError("raw training rows differ from selector input order")
    tasks = [(p["Depth_Color"], p["IR"], p["Skeleton"]) for _, _, _, p in samples]
    with ProcessPoolExecutor(max_workers=16) as pool:
        rebuilt = torch.tensor(list(pool.map(primary_available, tasks, chunksize=8)))
    if not torch.equal(rebuilt, valid):
        raise ValueError("deployment availability parser does not reproduce training mask")
    visual, temporal, baseline = [torch.from_numpy(np.load(root / f"{name}_oof_logits.npy", allow_pickle=False))
                                  for name in ("visual", "temporal", "baseline")]
    states = []
    for seed in SELECTOR_CONTRACT["seeds"]:
        model, selected = fit_selector(visual, temporal, baseline, torch.from_numpy(y), valid,
                                      torch.ones(len(y), dtype=torch.bool), seed)
        states.append(model.state_dict())
        print(json.dumps({"seed": seed, "train_rows": int(selected.sum())}), flush=True)
    source = Path("checkpoints/strict_v3/model.pt")
    package = temporal_visual_package(torch.load(source, map_location="cpu", weights_only=True))
    package["selector_consensus"] = states
    package["release_contract"]["selector_consensus"] = SELECTOR_CONTRACT
    model_path, bundle_path = output / "model.pt", output / "submission_bundle.pt"
    torch.save(package, model_path)
    build_bundle(model_path, Path("checkpoints/strict_v3/yolo11n.pt"), bundle_path)
    receipt = {"training_availability_reproduced": True, "test_data_loaded": False,
               "source": file_record(source), "bundle": file_record(bundle_path),
               "oof_evidence": file_record(evidence / "metrics.json"), "contract": SELECTOR_CONTRACT}
    (evidence / "full_fit.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
