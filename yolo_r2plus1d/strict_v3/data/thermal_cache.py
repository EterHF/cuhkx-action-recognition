"""Build a train-only, full-field thermal video cache in the canonical row order."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from yolo_r2plus1d.strict_v3.data.cache import paths_in, sample
from yolo_r2plus1d.strict_v3.data.indexing import discover_train


def thermal_clip(directory: Path, frames: int = 16, size: int = 160,
                 window: np.ndarray | None = None) -> tuple[np.ndarray, bool]:
    paths = paths_in(directory)
    if not paths:
        return np.zeros((frames, 3, size, size), dtype=np.uint8), False
    images = []
    for path in sample(paths, frames):
        # An existing but corrupt image must fail the cache build, not become a
        # silently valid zero image. Missing directories have an explicit mask.
        with Image.open(path) as image:
            if window is not None and np.isfinite(window).all():
                width, height = image.size
                x0, y0, x1, y1 = window
                image = image.crop((round(x0 * width), round(y0 * height),
                                    round(x1 * width), round(y1 * height)))
            image = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
            images.append(np.asarray(image).transpose(2, 0, 1))
    return np.stack(images), True


def build_clip(task: tuple[Path, np.ndarray | None]) -> tuple[np.ndarray, bool]:
    return thermal_clip(task[0], window=task[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--person-crop", action="store_true")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    samples = discover_train(args.train_root)
    with np.load(args.metadata, allow_pickle=False) as meta:
        if not np.array_equal([row[0] for row in samples], meta["train_keys"]):
            raise ValueError("thermal row order differs from canonical training metadata")
    args.output.mkdir(parents=True, exist_ok=False)
    windows = [None] * len(samples)
    if args.person_crop:
        import torch

        from yolo_r2plus1d.strict_v3.data.windows import YOLO, detect_windows
        from yolo_r2plus1d.strict_v3.paths import CHECKPOINT_DIR

        torch.set_num_threads(2)
        windows, found = detect_windows(
            YOLO(CHECKPOINT_DIR / "yolo11n.pt"),
            [paths_in(row[3]["Thermal"]) for row in samples],
            None, None, 8, 64, args.device, 0.25, 1.4, 0.35,
        )
        np.savez_compressed(args.output / "windows.npz", windows=windows, found=found)
    cache = np.lib.format.open_memmap(
        args.output / "frames.npy", mode="w+", dtype=np.uint8,
        shape=(len(samples), 16, 3, 160, 160),
    )
    valid = np.zeros(len(samples), dtype=bool)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, (frames, present) in enumerate(
            pool.map(build_clip, [(row[3]["Thermal"], window)
                                  for row, window in zip(samples, windows, strict=True)], chunksize=4)
        ):
            cache[index] = frames
            valid[index] = present
            if (index + 1) % 400 == 0:
                print(f"cached={index + 1}/{len(samples)}", flush=True)
    cache.flush()
    np.save(args.output / "valid.npy", valid)
    print(f"valid={valid.sum()}/{len(valid)}", flush=True)


if __name__ == "__main__":
    main()
