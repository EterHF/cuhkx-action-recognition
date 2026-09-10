from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from yolo_r2plus1d.strict_v3.data.imu_cache import (
    MOTION_COLUMNS,
    QUAT_COLUMNS,
    read_clip,
    rotate_vectors,
)
from yolo_r2plus1d.strict_v3.models.imu import imu_blend, select_view


def test_quaternion_rotation_uses_correct_direction_and_ignores_sign():
    q = np.array([[np.sqrt(.5), 0, 0, np.sqrt(.5)]])
    v = np.array([[1., 0, 0]])
    np.testing.assert_allclose(rotate_vectors(v, q), [[0, 1, 0]], atol=1e-12)
    np.testing.assert_allclose(rotate_vectors(v, -q), [[0, 1, 0]], atol=1e-12)
    with pytest.raises(ValueError, match="orientation"):
        rotate_vectors(v, np.zeros_like(q))


def records():
    output = []
    for i in range(4):
        item = {"时间": f"2025-01-01 10:00:0{i}.000", "设备名称": "WTLA(original-device)"}
        item.update(dict(zip(MOTION_COLUMNS, [0, 0, 1, i * 180, 0, 0], strict=True)))
        item.update(dict(zip(QUAT_COLUMNS, [1, 0, 0, 0], strict=True)))
        item.update({"温度(°C)": 10, "电量(%)": 20})
        output.append(item)
    return pd.DataFrame(output)


def test_cache_is_invariant_to_absolute_time_identity_metadata_and_row_order(tmp_path: Path):
    first, second = tmp_path / "one", tmp_path / "two"
    first.mkdir()
    second.mkdir()
    data = records()
    data.to_csv(first / "up.csv", index=False)
    modified = data.iloc[::-1].copy()
    modified["时间"] = modified["时间"].str.replace("2025-01-01 10", "2026-08-03 15")
    modified["设备名称"] = "WTLA(different-hardware)"
    modified["温度(°C)"], modified["电量(%)"] = 99, 1
    modified.to_csv(second / "arbitrary-filename.csv", index=False)
    a, valid_a, _ = read_clip(first)
    b, valid_b, _ = read_clip(second)
    assert np.array_equal(a, b) and np.array_equal(valid_a, valid_b)
    assert valid_a.tolist() == [True, False, False, False, False]
    np.testing.assert_allclose(a[0, 0, 3], np.linspace(0, 3, 256), atol=1e-6)
    assert np.array_equal(a[0], a[1])  # Identity orientation.


def test_missing_data_is_explicit_and_unknown_locations_fail(tmp_path: Path):
    x, mask, report = read_clip(tmp_path)
    assert not x.any() and not mask.any() and report["rows"] == 0
    data = records()
    data["设备名称"] = "unknown(sensor)"
    data.to_csv(tmp_path / "data.csv", index=False)
    with pytest.raises(ValueError, match="body-location"):
        read_clip(tmp_path)


def test_view_ablation_keeps_channel_budget_and_changes_only_second_view():
    values = torch.stack([torch.ones(2, 5, 6, 256), torch.full((2, 5, 6, 256), 2.)], dim=1)
    local, global_view = select_view(values, "local"), select_view(values, "global")
    assert local.shape == global_view.shape == (2, 60, 256)
    assert torch.equal(local[:, :30], global_view[:, :30])
    assert (local[:, 30:] == 1).all() and (global_view[:, 30:] == 2).all()


def test_imu_blend_preserves_unavailable_rows_and_exact_probability_weight():
    base = torch.tensor([[2., 1.], [2., 1.]])
    imu = torch.tensor([[float("nan"), float("nan")], [0., 3.]])
    actual = imu_blend(base, imu, torch.tensor([False, True]))
    assert torch.equal(actual[0], base[0])
    torch.testing.assert_close(actual[1].exp(), .9 * base[1].softmax(0) + .1 * imu[1].softmax(0))
