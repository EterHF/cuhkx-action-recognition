#!/usr/bin/env python3
"""Cache ordered native skeleton frames without temporal interpolation."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from yolo_r2plus1d.strict_v3.data.indexing import IMAGE_SUFFIXES, discover_train
from yolo_r2plus1d.strict_v3.data.skeleton_cache import frame_id, load_person
from yolo_r2plus1d.strict_v3.paths import DATA_DIR, REPO_ROOT, RESULT_DIR


def build_clip(
    task: tuple[int, Path, Path, int],
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, int]:
    index, depth_dir, skeleton_dir, max_frames = task
    depth = sorted(
        path
        for path in depth_dir.iterdir()
        if depth_dir.is_dir() and path.suffix.lower() in IMAGE_SUFFIXES
    ) if depth_dir.is_dir() else []
    prediction_dir = skeleton_dir / "predictions"
    skeleton_paths = prediction_dir.glob("*.json") if prediction_dir.is_dir() else []
    mapping = {
        identifier: path
        for path in skeleton_paths
        if (identifier := frame_id(path)) is not None
    }
    if len(depth) > max_frames:
        start = (len(depth) - max_frames) // 2
        depth = depth[start : start + max_frames]

    output = np.zeros((max_frames, 17, 3), dtype=np.float32)
    valid = np.zeros(max_frames, dtype=np.bool_)
    positions = np.zeros(max_frames, dtype=np.float32)
    for offset, path in enumerate(depth):
        identifier = frame_id(path)
        points = load_person(mapping[identifier]) if identifier in mapping else None
        if points is not None:
            output[offset] = points
            valid[offset] = True
    length = len(depth)
    if length > 1:
        positions[:length] = np.linspace(0.0, 1.0, length, dtype=np.float32)
    if valid.any():
        raw = output[valid]
        root = raw[:, 0].copy()
        hip = np.linalg.norm(raw[:, 1] - raw[:, 4], axis=1)
        shoulder = np.linalg.norm(raw[:, 11] - raw[:, 14], axis=1)
        torso = np.linalg.norm(raw[:, 8] - raw[:, 0], axis=1)
        scale = max(
            float(np.median(np.maximum.reduce((hip, shoulder, torso)))), 1e-3
        )
        raw = (raw - root[:, None]) / scale
        raw[:, 0] = (root - root[:1]) / scale
        output[valid] = raw.clip(-5.0, 5.0)
    return index, output, valid, positions, length


def write_cache(
    tasks: list[tuple[int, Path, Path, int]],
    output_dir: Path,
    prefix: str,
    workers: int,
) -> None:
    if not tasks:
        return
    max_frames = tasks[0][3]
    output_dir.mkdir(parents=True, exist_ok=True)
    skeleton = np.lib.format.open_memmap(
        output_dir / f"{prefix}_skeleton.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(tasks), max_frames, 17, 3),
    )
    mask = np.zeros((len(tasks), max_frames), dtype=np.bool_)
    positions = np.zeros((len(tasks), max_frames), dtype=np.float32)
    lengths = np.zeros(len(tasks), dtype=np.int16)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for done, (index, value, valid, location, length) in enumerate(
            executor.map(build_clip, tasks, chunksize=8), 1
        ):
            skeleton[index] = value
            mask[index] = valid
            positions[index] = location
            lengths[index] = length
            if done % 300 == 0 or done == len(tasks):
                print(f"cached_highrate_skeleton={done}/{len(tasks)}", flush=True)
    skeleton.flush()
    np.save(output_dir / f"{prefix}_skeleton_mask.npy", mask)
    np.save(output_dir / f"{prefix}_positions.npy", positions)
    np.save(output_dir / f"{prefix}_lengths.npy", lengths)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-root", type=Path, default=DATA_DIR / "processed/train/HAR/data"
    )
    parser.add_argument(
        "--test-root",
        type=Path,
        default=DATA_DIR / "processed/test/small_model_track_test",
    )
    parser.add_argument("--metadata", type=Path, default=RESULT_DIR / "metadata.npz")
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / ".cache/highrate_skeleton"
    )
    parser.add_argument("--max-frames", type=int, default=256)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--include-test", action="store_true")
    args = parser.parse_args()
    if args.max_frames <= 0:
        parser.error("--max-frames must be positive")

    with np.load(args.metadata) as metadata:
        train_keys = metadata["train_keys"]
        test_ids = metadata["test_ids"]
    samples = discover_train(args.train_root)
    if not np.array_equal(np.asarray([sample[0] for sample in samples]), train_keys):
        raise RuntimeError("Training discovery order differs from metadata")
    train_tasks = [
        (
            index,
            paths.get("Depth_Color", Path("__missing__")),
            paths.get("Skeleton", Path("__missing__")),
            args.max_frames,
        )
        for index, (_, _, _, paths) in enumerate(samples)
    ]
    write_cache(train_tasks, args.output_dir, "train", args.workers)
    if args.include_test:
        test_tasks = [
            (
                index,
                args.test_root / sample_id / "Depth_Color",
                args.test_root / sample_id / "Skeleton",
                args.max_frames,
            )
            for index, sample_id in enumerate(test_ids)
        ]
        write_cache(test_tasks, args.output_dir, "test", args.workers)


if __name__ == "__main__":
    main()
