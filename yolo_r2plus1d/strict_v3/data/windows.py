#!/usr/bin/env python3
"""Detect people in sampled IR frames and save one fixed crop per clip."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".cache/ultralytics").resolve()))
from ultralytics import YOLO

from yolo_r2plus1d.strict_v3.data.indexing import IMAGE_SUFFIXES, discover_train, read_test_ids
from yolo_r2plus1d.strict_v3.paths import CHECKPOINT_DIR, DATA_DIR, REPO_ROOT


def image_paths(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def sample_paths(paths: list[Path], count: int) -> list[Path]:
    if not paths:
        return []
    indices = np.linspace(0, len(paths) - 1, min(count, len(paths))).round().astype(int)
    sampled = []
    for index in indices:
        path = paths[index]
        try:
            with Image.open(path) as image:
                image.verify()
            sampled.append(path)
        except (OSError, UnidentifiedImageError):
            print(f"skipping_corrupt_image={path}", flush=True)
    return sampled


def union_window(boxes: list[np.ndarray], width: int, height: int, margin: float, min_side: float) -> np.ndarray:
    """Return normalized xyxy for a square-ish, enlarged union crop."""
    if not boxes:
        return np.full(4, np.nan, dtype=np.float32)
    boxes_array = np.stack(boxes)
    x0, y0 = boxes_array[:, :2].min(axis=0)
    x1, y1 = boxes_array[:, 2:].max(axis=0)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    side = max((x1 - x0) * width, (y1 - y0) * height) * margin
    side = max(side, min_side * max(width, height))
    half_x, half_y = side / (2.0 * width), side / (2.0 * height)
    return np.asarray(
        [max(0.0, cx - half_x), max(0.0, cy - half_y), min(1.0, cx + half_x), min(1.0, cy + half_y)],
        dtype=np.float32,
    )


def median_center_window(
    boxes: list[np.ndarray], width: int, height: int, margin: float, min_side: float,
) -> np.ndarray:
    """Center the crop on the median detection while retaining person scale."""
    if not boxes:
        return np.full(4, np.nan, dtype=np.float32)
    boxes_array = np.stack(boxes)
    centers = 0.5 * (boxes_array[:, :2] + boxes_array[:, 2:])
    cx, cy = np.median(centers, axis=0)
    sizes = boxes_array[:, 2:] - boxes_array[:, :2]
    side = max(float(sizes[:, 0].max()) * width,
               float(sizes[:, 1].max()) * height) * margin
    side = max(side, min_side * max(width, height))
    half_x, half_y = side / (2.0 * width), side / (2.0 * height)
    return np.asarray(
        [max(0.0, cx - half_x), max(0.0, cy - half_y),
         min(1.0, cx + half_x), min(1.0, cy + half_y)],
        dtype=np.float32,
    )


def hybrid_window(
    boxes: list[np.ndarray], width: int, height: int, margin: float, min_side: float,
) -> np.ndarray:
    """Fixed midpoint between contextual union and robust median-center crops."""
    union = union_window(boxes, width, height, margin, min_side)
    median = median_center_window(boxes, width, height, margin, min_side)
    return np.asarray(0.5 * (union + median), dtype=np.float32)


def detect_windows(
    model: YOLO,
    clips: list[list[Path]],
    fallback_clips: list[list[Path]] | None,
    fallback_device: str | None,
    probe_frames: int,
    batch_size: int,
    device: str,
    confidence: float,
    margin: float,
    min_side: float,
    window_mode: str = "union",
) -> tuple[np.ndarray, np.ndarray]:
    probes: list[str] = []
    owners: list[int] = []
    for clip_index, paths in enumerate(clips):
        sampled = sample_paths(paths, probe_frames)
        probes.extend(str(path) for path in sampled)
        owners.extend([clip_index] * len(sampled))

    clip_boxes: list[list[np.ndarray]] = [[] for _ in clips]
    dimensions: list[tuple[int, int] | None] = [None] * len(clips)
    for offset in range(0, len(probes), batch_size):
        batch = probes[offset : offset + batch_size]
        results = model.predict(batch, classes=[0], conf=confidence, imgsz=640, device=device, verbose=False)
        for local_index, result in enumerate(results):
            owner = owners[offset + local_index]
            height, width = result.orig_shape
            dimensions[owner] = width, height
            if result.boxes is None or len(result.boxes) == 0:
                continue
            best = int(result.boxes.conf.argmax().item())
            clip_boxes[owner].append(result.boxes.xyxyn[best].detach().cpu().numpy().astype(np.float32))
        done = min(offset + batch_size, len(probes))
        if done % (batch_size * 10) == 0 or done == len(probes):
            print(f"detected_frames={done}/{len(probes)}", flush=True)

    if fallback_clips is not None:
        retry = [index for index, boxes in enumerate(clip_boxes) if not boxes]
        fallback_probes: list[str] = []
        fallback_owners: list[int] = []
        for clip_index in retry:
            sampled = sample_paths(fallback_clips[clip_index], probe_frames)
            fallback_probes.extend(str(path) for path in sampled)
            fallback_owners.extend([clip_index] * len(sampled))
        for offset in range(0, len(fallback_probes), batch_size):
            batch = fallback_probes[offset : offset + batch_size]
            results = model.predict(batch, classes=[0], conf=confidence, imgsz=640,
                                    device=fallback_device or device, verbose=False)
            for local_index, result in enumerate(results):
                owner = fallback_owners[offset + local_index]
                height, width = result.orig_shape
                dimensions[owner] = width, height
                if result.boxes is None or len(result.boxes) == 0:
                    continue
                best = int(result.boxes.conf.argmax().item())
                clip_boxes[owner].append(
                    result.boxes.xyxyn[best].detach().cpu().numpy().astype(np.float32)
                )
        print(f"depth_fallback_retry={len(retry)} rescued={sum(bool(clip_boxes[i]) for i in retry)}", flush=True)

    window_fn = {
        "union": union_window,
        "median-center": median_center_window,
        "hybrid-half": hybrid_window,
    }[window_mode]
    windows = []
    for boxes, shape in zip(clip_boxes, dimensions, strict=True):
        width, height = shape if shape is not None else (640, 480)
        windows.append(window_fn(boxes, width, height, margin, min_side))
    windows_array = np.stack(windows)
    return windows_array, np.isfinite(windows_array).all(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, default=DATA_DIR / "processed/train/HAR/data")
    parser.add_argument("--test-root", type=Path, default=DATA_DIR / "processed/test/small_model_track_test")
    parser.add_argument("--test-csv", type=Path, default=DATA_DIR / "Small-Model-Track/Testing/test_file/test.csv")
    parser.add_argument("--weights", type=Path, default=CHECKPOINT_DIR / "yolo11n.pt")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / ".cache/strict_v3/windows.npz")
    parser.add_argument("--probe-frames", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="0")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--margin", type=float, default=1.4)
    parser.add_argument("--min-side", type=float, default=0.35)
    parser.add_argument(
        "--window-mode", choices=("union", "median-center", "hybrid-half"), default="union",
    )
    parser.add_argument("--depth-fallback", action="store_true",
                        help="retry test clips without IR detections using Depth_Color probes")
    parser.add_argument("--symmetric-fallback", action="store_true",
                        help="use the same IR->Depth fallback for train and test clips")
    parser.add_argument("--test-only", action="store_true",
                        help="index and detect only the anonymous test stream")
    parser.add_argument("--fallback-device", default="cpu",
                        help="device for the optional Depth_Color fallback pass")
    args = parser.parse_args()

    train_samples = [] if args.test_only else discover_train(args.train_root)
    test_ids = read_test_ids(args.test_csv)
    train_clips = [image_paths(paths.get("IR", Path("__missing__"))) for _, _, _, paths in train_samples]
    test_clips = [image_paths(args.test_root / sample_id / "IR") for sample_id in test_ids]
    test_fallback = [image_paths(args.test_root / sample_id / "Depth_Color") for sample_id in test_ids]
    print(f"clips: train={len(train_clips)} test={len(test_clips)}", flush=True)
    model = YOLO(args.weights)
    train_fallback = [image_paths(paths.get("Depth_Color", Path("__missing__")))
                      for _, _, _, paths in train_samples]
    if args.test_only:
        train_windows = np.empty((0, 4), dtype=np.float32)
        train_has_crop = np.empty((0,), dtype=bool)
    else:
        train_windows, train_has_crop = detect_windows(
            model, train_clips, train_fallback if args.symmetric_fallback else None,
            args.fallback_device if args.symmetric_fallback else None,
            args.probe_frames, args.batch_size, args.device,
            args.confidence, args.margin, args.min_side, args.window_mode,
        )
    test_windows, test_has_crop = detect_windows(
        model, test_clips,
        test_fallback if (args.depth_fallback or args.symmetric_fallback) else None,
        args.fallback_device,
        args.probe_frames, args.batch_size, args.device,
        args.confidence, args.margin, args.min_side, args.window_mode,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        train_windows=train_windows,
        train_has_crop=train_has_crop,
        train_keys=np.asarray([sample[0] for sample in train_samples]),
        train_y=np.asarray([sample[1] for sample in train_samples], dtype=np.int64),
        train_users=np.asarray([sample[2] for sample in train_samples], dtype=np.int64),
        test_windows=test_windows,
        test_has_crop=test_has_crop,
        test_ids=np.asarray(test_ids),
    )
    print(
        f"saved={args.output} train_detection_rate={(train_has_crop.mean() if len(train_has_crop) else 0.0):.4f} "
        f"test_detection_rate={test_has_crop.mean():.4f}", flush=True,
    )


if __name__ == "__main__":
    main()
