#!/usr/bin/env python3
"""Build Depth-aligned, pelvis-centered H36M/MotionBERT-17 skeleton tensors."""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from yolo_r2plus1d.strict_v3.data.indexing import IMAGE_SUFFIXES, discover_train
from yolo_r2plus1d.strict_v3.paths import DATA_DIR, REPO_ROOT, RESULT_DIR

FRAME_ID = re.compile(r"_(\d+)(?:_Color)?\.(?:png|json)$")


def frame_id(path: Path) -> int | None:
    match = FRAME_ID.search(path.name)
    return int(match.group(1)) if match else None


def depth_paths(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def sampled_ids(directory: Path, frames: int) -> list[int | None]:
    paths = depth_paths(directory)
    if not paths:
        return [None] * frames
    indices = np.linspace(0, len(paths) - 1, frames).round().astype(int)
    return [frame_id(paths[index]) for index in indices]


def load_person(path: Path) -> np.ndarray | None:
    try:
        people = json.loads(path.read_text(encoding="utf-8"))
        if not people:
            return None
        person = max(people, key=lambda item: float(np.mean(item.get("keypoint_scores", [0.0]))))
        points = np.asarray(person["keypoints"], dtype=np.float32)
        if points.shape != (17, 3) or not np.isfinite(points).all():
            return None
        return points
    except (OSError, ValueError, KeyError, TypeError):
        return None


def build_clip(task: tuple[int, Path, Path, int]) -> tuple[int, np.ndarray, bool]:
    index, depth_dir, skeleton_dir, frames = task
    prediction_dir = skeleton_dir / "predictions"
    skeleton_paths = prediction_dir.glob("*.json") if prediction_dir.is_dir() else []
    mapping = {identifier: path for path in skeleton_paths if (identifier := frame_id(path)) is not None}
    sequence = []
    valid = []
    for identifier in sampled_ids(depth_dir, frames):
        points = load_person(mapping[identifier]) if identifier in mapping else None
        sequence.append(np.zeros((17, 3), dtype=np.float32) if points is None else points)
        valid.append(points is not None)
    points = np.stack(sequence)
    valid_array = np.asarray(valid)
    if not valid_array.any():
        return index, points, False

    # MotionBERT/H36M-17 order: joint 0 is the pelvis/root.
    roots = points[:, 0:1]
    centered = points - roots
    valid_points = points[valid_array]
    hip_width = np.linalg.norm(valid_points[:, 1] - valid_points[:, 4], axis=1)
    shoulder_width = np.linalg.norm(valid_points[:, 11] - valid_points[:, 14], axis=1)
    torso_length = np.linalg.norm(valid_points[:, 8] - valid_points[:, 0], axis=1)
    scale = float(np.median(np.maximum.reduce((shoulder_width, hip_width, torso_length))))
    centered /= max(scale, 1e-3)
    centered[~valid_array] = 0.0
    return index, centered.clip(-5.0, 5.0).astype(np.float32), True


def write_cache(tasks: list[tuple[int, Path, Path, int]], output: Path, mask_output: Path, workers: int) -> None:
    frames = tasks[0][3]
    array = np.lib.format.open_memmap(output, mode="w+", dtype=np.float32, shape=(len(tasks), frames, 17, 3))
    masks = np.zeros(len(tasks), dtype=np.bool_)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for completed, (index, sequence, present) in enumerate(executor.map(build_clip, tasks, chunksize=8), 1):
            array[index], masks[index] = sequence, present
            if completed % 300 == 0 or completed == len(tasks):
                print(f"cached_skeleton={completed}/{len(tasks)}", flush=True)
    array.flush()
    np.save(mask_output, masks)
    print(f"saved={output} present_rate={masks.mean():.4f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, default=DATA_DIR / "processed/train/HAR/data")
    parser.add_argument("--test-root", type=Path, default=DATA_DIR / "processed/test/small_model_track_test")
    parser.add_argument("--metadata", type=Path, default=RESULT_DIR / "metadata.npz")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / ".cache/strict_v3")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.metadata) as data:
        expected_keys, test_ids = data["train_keys"], data["test_ids"]
    if not args.test_only:
        samples = discover_train(args.train_root)
        if not np.array_equal(np.asarray([sample[0] for sample in samples]), expected_keys):
            raise RuntimeError("Training discovery order differs from metadata")
        train_tasks = [
            (index, paths.get("Depth_Color", Path("__missing__")), paths.get("Skeleton", Path("__missing__")), args.frames)
            for index, (_, _, _, paths) in enumerate(samples)
        ]
        write_cache(train_tasks, args.output_dir / "train_skeleton.npy", args.output_dir / "train_skeleton_mask.npy", args.workers)
    if args.include_test or args.test_only:
        test_tasks = [
            (index, args.test_root / sample_id / "Depth_Color", args.test_root / sample_id / "Skeleton", args.frames)
            for index, sample_id in enumerate(test_ids)
        ]
        write_cache(test_tasks, args.output_dir / "test_skeleton.npy", args.output_dir / "test_skeleton_mask.npy", args.workers)


if __name__ == "__main__":
    main()
