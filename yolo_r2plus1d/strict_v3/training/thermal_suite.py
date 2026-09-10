"""Run the fixed three-seed thermal subject-OOF suite across available GPUs."""

from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from yolo_r2plus1d.strict_v3.evaluation.quality_gate_ablation import file_record
from yolo_r2plus1d.strict_v3.evaluation.visual_quantization import FOLDS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--temporal-shift", action="store_true")
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus):
        raise ValueError("GPU slots must be distinct")
    inputs = args.root / "inputs"
    pretrained = Path(".cache/torch/hub/checkpoints/resnet18-f37072fd.pth")
    metadata = Path("results/strict_v3/metadata.npz")
    jobs = []
    for seed in (2026, 2027, 2028):
        for fold in FOLDS:
            output = args.root / f"seed{seed}/fold{fold}"
            if output.exists():
                raise FileExistsError(f"refusing to reuse an existing training fold: {output}")
            command = [sys.executable, "-m", "yolo_r2plus1d.strict_v3.training.thermal",
                       "--cache", str(inputs / "frames.npy"), "--valid", str(inputs / "valid.npy"),
                       "--metadata", str(metadata), "--pretrained", str(pretrained),
                       "--output", str(output), "--seed", str(seed), "--fold", fold]
            if args.temporal_shift:
                command.append("--temporal-shift")
            jobs.append((output, command))
    receipt = {"inputs": [file_record(path) for path in
                          (inputs / "frames.npy", inputs / "valid.npy", pretrained, metadata)],
               "test_data_loaded": False, "jobs": []}
    devices = queue.Queue()
    for device in args.gpus:
        devices.put(device)

    def run(job):
        output, command = job
        device = devices.get()
        command = [*command, "--device", f"cuda:{device}"]
        output.parent.mkdir(parents=True, exist_ok=True)
        log_path = output.with_suffix(".log")
        print(f"START {output} GPU {device}", flush=True)
        try:
            with log_path.open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            print(f"DONE {output} returncode {result.returncode}", flush=True)
            return {"command": command, "log": str(log_path), "returncode": result.returncode}
        finally:
            devices.put(device)

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        receipt["jobs"] = list(pool.map(run, jobs))
    receipt["completed"] = all(job["returncode"] == 0 for job in receipt["jobs"])
    (args.root / "suite_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if not receipt["completed"]:
        raise RuntimeError("thermal suite failed; see the failed jobs' logs in suite_receipt.json")


if __name__ == "__main__":
    main()
