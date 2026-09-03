#!/usr/bin/env python3
"""Create compact uint8 Depth-RGB + IR temporal tensors from fixed YOLO crops."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from yolo_r2plus1d.strict_v3.data.indexing import IMAGE_SUFFIXES, discover_train
from yolo_r2plus1d.strict_v3.paths import DATA_DIR, REPO_ROOT


def paths_in(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def sample(paths: list[Path], count: int) -> list[Path | None]:
    if not paths:
        return [None] * count
    indices = np.linspace(0, len(paths) - 1, count).round().astype(int)
    return [paths[index] for index in indices]


def load_frame(path: Path | None, mode: str, window: np.ndarray, image_size: int) -> np.ndarray:
    channels = 3 if mode == "RGB" else 1
    if path is None:
        return np.zeros((image_size, image_size, channels), dtype=np.uint8)
    try:
        with Image.open(path) as source:
            image = source.convert(mode)
            if np.isfinite(window).all():
                width, height = image.size
                x0, y0, x1, y1 = window
                bounds = (
                    int(round(x0 * width)), int(round(y0 * height)),
                    int(round(x1 * width)), int(round(y1 * height)),
                )
                image = image.crop(bounds)
            image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.uint8)
            return array[..., None] if channels == 1 else array
    except (OSError, ValueError):
        return np.zeros((image_size, image_size, channels), dtype=np.uint8)


def build_clip(task: tuple[int, Path, Path, np.ndarray, int, int]) -> tuple[int, np.ndarray]:
    index, depth_dir, ir_dir, window, frames, image_size = task
    depth_paths, ir_paths = sample(paths_in(depth_dir), frames), sample(paths_in(ir_dir), frames)
    output = []
    for depth_path, ir_path in zip(depth_paths, ir_paths, strict=True):
        depth = load_frame(depth_path, "RGB", window, image_size)
        infrared = load_frame(ir_path, "L", window, image_size)
        output.append(np.concatenate((depth, infrared), axis=-1).transpose(2, 0, 1))
    return index, np.stack(output)


def write_cache(tasks: list[tuple[int, Path, Path, np.ndarray, int, int]], output: Path, workers: int) -> None:
    if not tasks:
        return
    _, _, _, _, frames, image_size = tasks[0]
    output.parent.mkdir(parents=True, exist_ok=True)
    array = np.lib.format.open_memmap(output, mode="w+", dtype=np.uint8, shape=(len(tasks), frames, 4, image_size, image_size))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for completed, (index, clip) in enumerate(executor.map(build_clip, tasks, chunksize=4), 1):
            array[index] = clip
            if completed % 200 == 0 or completed == len(tasks):
                print(f"cached={completed}/{len(tasks)}", flush=True)
    array.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=Path, default=REPO_ROOT / ".cache/strict_v3/windows.npz")
    parser.add_argument("--train-root", type=Path, default=DATA_DIR / "processed/train/HAR/data")
    parser.add_argument("--test-root", type=Path, default=DATA_DIR / "processed/test/small_model_track_test")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / ".cache/strict_v3")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    args = parser.parse_args()

    with np.load(args.windows) as data:
        metadata = {name: data[name] for name in data.files}
    if not args.test_only:
        train_samples = discover_train(args.train_root)
        discovered_keys = np.asarray([sample[0] for sample in train_samples])
        if not np.array_equal(discovered_keys, metadata["train_keys"]):
            raise RuntimeError("Training discovery order differs from windows metadata")
        train_tasks = [
            (index, paths.get("Depth_Color", Path("__missing__")), paths.get("IR", Path("__missing__")),
             metadata["train_windows"][index], args.frames, args.image_size)
            for index, (_, _, _, paths) in enumerate(train_samples)
        ]
        write_cache(train_tasks, args.output_dir / "train_depth_ir.npy", args.workers)
    if args.include_test or args.test_only:
        test_tasks = [
            (index, args.test_root / sample_id / "Depth_Color", args.test_root / sample_id / "IR",
             metadata["test_windows"][index], args.frames, args.image_size)
            for index, sample_id in enumerate(metadata["test_ids"])
        ]
        write_cache(test_tasks, args.output_dir / "test_depth_ir.npy", args.workers)
    np.savez_compressed(
        args.output_dir / "metadata.npz",
        train_keys=metadata["train_keys"], train_y=metadata["train_y"], train_users=metadata["train_users"],
        train_has_crop=metadata["train_has_crop"], test_ids=metadata["test_ids"], test_has_crop=metadata["test_has_crop"],
        frames=np.asarray(args.frames), image_size=np.asarray(args.image_size),
    )
    print(f"saved_cache={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
