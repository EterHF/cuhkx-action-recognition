from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, UnidentifiedImageError

from yolo_r2plus1d.strict_v3.data.thermal_cache import thermal_clip
from yolo_r2plus1d.strict_v3.evaluation.thermal_fallback import assemble
from yolo_r2plus1d.strict_v3.models.thermal import shift_frames, thermal_fallback
from yolo_r2plus1d.strict_v3.training.thermal import split_rows


def test_fallback_preserves_every_row_with_primary_evidence_or_no_thermal():
    baseline = torch.tensor([[8., 0.], [8., 0.], [8., 0.], [8., 0.]])
    sensor = baseline.flip(1)
    actual = thermal_fallback(baseline, sensor, torch.tensor([False, True, True, False]),
                              torch.tensor([True, True, False, False]))
    assert actual.argmax(1).tolist() == [0, 1, 0, 0]
    assert torch.equal(actual[[0, 2, 3]], baseline[[0, 2, 3]])


def test_split_excludes_held_subjects_and_missing_thermal_from_training():
    users = np.array([1, 2, 6, 7, 17, 18, 22, 23])
    train, held = split_rows(users, np.array([True] * 7 + [False]), "A")
    assert train.tolist() == [1, 3, 5]
    assert held.tolist() == [0, 2, 4, 6]


def test_cache_missing_is_explicit_but_corrupt_is_an_error(tmp_path: Path):
    frames, valid = thermal_clip(tmp_path, frames=2, size=8)
    assert not valid and not frames.any()
    (tmp_path / "broken.jpg").write_bytes(b"not an image")
    with pytest.raises(UnidentifiedImageError):
        thermal_clip(tmp_path, frames=2, size=8)


def test_cache_orders_frames_and_keeps_the_full_field(tmp_path: Path):
    for index, value in ((2, 200), (1, 10)):
        Image.fromarray(np.full((8, 8, 3), value, dtype=np.uint8)).save(
            tmp_path / f"frame_{index:06}.png")
    frames, valid = thermal_clip(tmp_path, frames=2, size=8)
    assert valid and frames[:, 0, 0, 0].tolist() == [10, 200]


def test_temporal_shift_never_mixes_clips_or_wraps_endpoints():
    x = torch.arange(1., 7.).view(6, 1, 1, 1).expand(-1, 8, -1, -1).clone()
    x.requires_grad_()
    y = shift_frames(x, 3)
    assert y[:, 0, 0, 0].tolist() == [2, 3, 0, 5, 6, 0]
    assert y[:, 1, 0, 0].tolist() == [0, 1, 2, 0, 4, 5]
    assert torch.equal(y[:, 2:], x[:, 2:])
    y.sum().backward()
    assert x.grad[:, 0, 0, 0].tolist() == [0, 1, 1, 0, 1, 1]


def test_thermal_crop_uses_its_own_coordinates_and_no_detection_falls_back(tmp_path: Path):
    image = np.zeros((8, 16, 3), dtype=np.uint8)
    image[:, 8:] = 200
    Image.fromarray(image).save(tmp_path / "frame_000001.png")
    full, _ = thermal_clip(tmp_path, frames=2, size=8)
    cropped, _ = thermal_clip(tmp_path, frames=2, size=8, window=np.array([.5, 0, 1, 1]))
    fallback, _ = thermal_clip(tmp_path, frames=2, size=8, window=np.full(4, np.nan))
    assert (cropped == 200).all()
    assert np.array_equal(full, fallback)


@pytest.mark.parametrize("rows,train_users,error", [
    ([1], [2], "held rows mismatch"),
    ([0], [1, 2], "trained on a held subject"),
])
def test_oof_audit_rejects_misalignment_and_target_user_leakage(
    tmp_path: Path, rows, train_users, error,
):
    fold = tmp_path / "foldA"
    fold.mkdir()
    np.savez_compressed(fold / "held.npz", rows=np.array(rows), logits=np.zeros((1, 40)))
    torch.save({"train_users": train_users, "epochs": 15, "fold": "A", "seed": 2026},
               fold / "model.pt")
    with pytest.raises(ValueError, match=error):
        assemble(tmp_path, np.array([1, 2]), 2026, False)
