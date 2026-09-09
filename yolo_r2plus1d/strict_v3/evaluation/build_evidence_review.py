#!/usr/bin/env python3
"""Build blinded raw/selected/cache/skeleton contact sheets for error review."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

from yolo_r2plus1d.strict_v3.data.cache import build_clip, paths_in, sample
from yolo_r2plus1d.strict_v3.data.indexing import discover_train
from yolo_r2plus1d.strict_v3.data.skeleton_retarget import PARENTS, load_clip
from yolo_r2plus1d.strict_v3.data.windows import detect_windows, image_paths
from yolo_r2plus1d.strict_v3.paths import DATA_DIR

TILE = 96
HEADER = 22


def thumbnail(path: Path | None) -> Image.Image:
    if path is None:
        return Image.new("RGB", (TILE, TILE))
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail((TILE, TILE), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (TILE, TILE))
    canvas.paste(image, ((TILE - image.width) // 2, (TILE - image.height) // 2))
    return canvas


def array_tile(value: np.ndarray) -> Image.Image:
    image = Image.fromarray(np.asarray(value, dtype=np.uint8), mode="RGB")
    return image.resize((TILE, TILE), Image.Resampling.NEAREST)


def skeleton_tile(points: np.ndarray) -> Image.Image:
    image = Image.new("RGB", (TILE, TILE), "white")
    draw = ImageDraw.Draw(image)
    xy = (np.asarray(points[:, :2]) + 1.0) * (TILE - 1) / 2.0
    valid = np.asarray(points[:, 2]) != 0.0
    for child, parent in enumerate(PARENTS):
        if child and valid[child] and valid[parent]:
            draw.line((*xy[parent], *xy[child]), fill=(40, 90, 180), width=2)
    for point, present in zip(xy, valid, strict=True):
        if present:
            x, y = point
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(190, 40, 40))
    return image


def grid(label: str, tiles: list[Image.Image], columns: int = 16) -> Image.Image:
    grid_rows = max(1, math.ceil(len(tiles) / columns))
    output = Image.new("RGB", (columns * TILE, HEADER + grid_rows * TILE), "white")
    ImageDraw.Draw(output).text((4, 3), label, fill="black")
    for index, tile in enumerate(tiles):
        output.paste(
            tile,
            ((index % columns) * TILE, HEADER + (index // columns) * TILE),
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument(
        "--train-root", type=Path, default=DATA_DIR / "processed/train/HAR/data"
    )
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    with args.selection.open(newline="", encoding="utf-8") as handle:
        selection = list(csv.DictReader(handle))
    samples = discover_train(args.train_root)
    rows = [int(item["row"]) for item in selection]
    chosen = [samples[index] for index in rows]
    infrared = [image_paths(item[3]["IR"]) for item in chosen]
    depth = [image_paths(item[3]["Depth_Color"]) for item in chosen]
    model = YOLO(args.detector)
    windows, has_crop = detect_windows(
        model,
        infrared,
        depth,
        "cpu",
        8,
        128,
        args.device,
        0.25,
        1.4,
        0.35,
        "union",
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = []
    for item, sample_item, window, detected in zip(
        selection, chosen, windows, has_crop, strict=True
    ):
        blind_id = item["blind_id"]
        paths = sample_item[3]
        depth_paths = paths_in(paths["Depth_Color"])
        full_tiles = [thumbnail(path) for path in depth_paths]
        selected_paths = sample(depth_paths, 16)
        selected_tiles = [thumbnail(path) for path in selected_paths]
        _, cache = build_clip((0, paths["Depth_Color"], paths["IR"], window, 16, 128))
        cached_depth = [array_tile(frame[:3].transpose(1, 2, 0)) for frame in cache]
        cached_ir = [
            array_tile(np.repeat(frame[3:4].transpose(1, 2, 0), 3, axis=2))
            for frame in cache
        ]
        skeleton = load_clip(paths["Skeleton"], 16)
        skeleton_tiles = [skeleton_tile(points) for points in skeleton]
        columns = 16
        sheet_rows = [
            grid(
                f"{blind_id} | complete raw timeline ({len(depth_paths)} frames)",
                full_tiles,
                columns,
            ),
            grid("selected 16 | uncropped Depth_Color", selected_tiles, columns),
            grid(
                "model cache | cropped/resized Depth_Color 128x128",
                cached_depth,
                columns,
            ),
            grid("model cache | cropped/resized IR 128x128", cached_ir, columns),
            grid("model skeleton | selected 16 H36M-17", skeleton_tiles, columns),
        ]
        sheet = Image.new(
            "RGB", (columns * TILE, sum(value.height for value in sheet_rows)), "white"
        )
        offset = 0
        for value in sheet_rows:
            sheet.paste(value, (0, offset))
            offset += value.height
        output_path = args.output_dir / f"{blind_id}.png"
        sheet.save(output_path)
        manifest.append(
            {
                "blind_id": blind_id,
                "source_frames": len(depth_paths),
                "full_timeline_frames_shown": len(full_tiles),
                "selected_frames": 16,
                "has_person_crop": bool(detected),
                "window": [float(value) for value in window],
                "sheet": output_path.name,
            }
        )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "cuhkx-blinded-input-evidence/v1",
                "prediction_and_label_hidden_in_sheets": True,
                "detector_protocol": "IR->Depth fallback, 8 probes, conf 0.25, union margin 1.4, min side 0.35",
                "items": manifest,
                "anonymous_test_accessed": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
