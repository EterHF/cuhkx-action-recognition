"""Parse five body IMUs without exposing hardware IDs or absolute time to models."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from yolo_r2plus1d.strict_v3.data.indexing import discover_train

LOCATIONS = ("WTLA", "WTRA", "WTC", "WTLL", "WTRL")
MOTION_COLUMNS = ["加速度X(g)", "加速度Y(g)", "加速度Z(g)",
                  "角速度X(°/s)", "角速度Y(°/s)", "角速度Z(°/s)"]
QUAT_COLUMNS = ["四元数0()", "四元数1()", "四元数2()", "四元数3()"]


def rotate_vectors(vector: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    """Apply a normalized w,x,y,z quaternion, invariant to quaternion sign."""
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm < 0.5) or not np.isfinite(quaternion).all():
        raise ValueError("invalid orientation quaternion")
    q = quaternion / norm
    xyz = q[..., 1:]
    return vector + 2 * np.cross(xyz, np.cross(xyz, vector) + q[..., :1] * vector)


def read_clip(directory: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    arrays = []
    for path in sorted(directory.glob("*.csv")):
        data = pd.read_csv(path, encoding="utf-8-sig")
        required = ["时间", "设备名称", *MOTION_COLUMNS, *QUAT_COLUMNS]
        if not set(required).issubset(data.columns):
            raise ValueError(f"unexpected IMU schema: {path}")
        if len(data):
            arrays.append(data[required])
    output = np.zeros((2, 5, 6, 256), dtype=np.float32)
    present = np.zeros(5, dtype=bool)
    report = {"rows": 0, "invalid_rows": 0, "duplicate_times": 0, "short_sensors": 0}
    if not arrays:
        return output, present, report
    data = pd.concat(arrays, ignore_index=True)
    # The location prefix is a sensor-layout lookup only. MACs/device identity,
    # wall-clock values, firmware, temperature, and battery are never features.
    location = data["设备名称"].astype(str).str.split("(", regex=False).str[0].str.strip()
    unknown = set(location) - set(LOCATIONS)
    if unknown:
        raise ValueError(f"unknown IMU body-location prefixes: {sorted(unknown)}")
    motion = data[MOTION_COLUMNS + QUAT_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    times = pd.to_datetime(data["时间"], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
    good = (np.isfinite(motion).all(1) & times.notna().to_numpy()
            & (np.linalg.norm(motion[:, 6:], axis=1) >= 0.5))
    report["rows"] = len(data)
    report["invalid_rows"] = int((~good).sum())
    if not good.any():
        return output, present, report
    time_ns = times.astype("int64").to_numpy()
    # Align the body sensors on a common within-clip grid; absolute start time
    # is discarded before interpolation and is not returned in any array.
    origin = time_ns[good].min()
    relative = (time_ns - origin) / 1e9
    grid = np.linspace(0, relative[good].max(), 256)
    for index, name in enumerate(LOCATIONS):
        rows = np.flatnonzero(good & (location.to_numpy() == name))
        rows = rows[np.argsort(relative[rows], kind="stable")]
        unique, first = np.unique(relative[rows], return_index=True)
        report["duplicate_times"] += len(rows) - len(first)
        rows = rows[first]
        if len(rows) < 4:
            report["short_sensors"] += 1
            continue
        raw = motion[rows, :6]
        orientation = motion[rows, 6:]
        world = np.concatenate((rotate_vectors(raw[:, :3], orientation),
                                rotate_vectors(raw[:, 3:], orientation)), axis=1)
        for view, signals in enumerate((raw, world)):
            for channel in range(6):
                output[view, index, channel] = np.interp(grid, unique, signals[:, channel])
        output[:, index, 3:] /= 180.0
        present[index] = True
    return output, present, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    rows = discover_train(args.train_root)
    with np.load(args.metadata, allow_pickle=False) as metadata:
        if not np.array_equal([row[0] for row in rows], metadata["train_keys"]):
            raise ValueError("IMU cache row order differs from canonical metadata")
    args.output.mkdir(parents=True, exist_ok=False)
    values = np.empty((len(rows), 2, 5, 6, 256), dtype=np.float32)
    mask = np.zeros((len(rows), 5), dtype=bool)
    reports = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, (signal, present, report) in enumerate(
            pool.map(read_clip, [row[3]["IMU"] for row in rows], chunksize=8)
        ):
            values[index], mask[index] = signal, present
            reports.append(report)
    if not np.isfinite(values).all():
        raise ValueError("IMU cache contains nonfinite values")
    np.save(args.output / "signals.npy", values)
    np.save(args.output / "sensor_valid.npy", mask)
    audit = {"train_rows": len(rows), "at_least_three_sensors": int((mask.sum(1) >= 3).sum()),
             "all_five_sensors": int(mask.all(1).sum()), "no_sensor": int((~mask.any(1)).sum()),
             "sensor_coverage": dict(zip(LOCATIONS, mask.sum(0).tolist(), strict=True)),
             "parser_totals": {key: int(sum(r[key] for r in reports)) for key in reports[0]},
             "features": "local/global acceleration in g, angular velocity /180 deg/s",
             "identity_absolute_time_environment_metadata_features": False,
             "anonymous_test_loaded": False}
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
