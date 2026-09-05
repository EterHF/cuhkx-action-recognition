from pathlib import Path

import numpy as np
import torch

from yolo_r2plus1d.strict_v3.data.highrate_skeleton_cache import build_clip
from yolo_r2plus1d.strict_v3.models.highrate_cross_attention import (
    HighRateCrossAttention,
)


def test_cross_attention_starts_as_exact_temporal_baseline() -> None:
    model = HighRateCrossAttention(visual_dim=128).eval()
    baseline = torch.randn(2, 40)
    output = model(
        torch.randn(2, 8, 128),
        torch.randn(2, 32, 17, 3),
        torch.ones(2, 32, dtype=torch.bool),
        torch.linspace(0, 1, 32).repeat(2, 1),
        baseline,
    )
    torch.testing.assert_close(output, baseline)


def test_cross_attention_missing_skeleton_is_exact_baseline() -> None:
    model = HighRateCrossAttention(visual_dim=128)
    with torch.no_grad():
        model.classifier[-1].weight.fill_(0.1)
    baseline = torch.randn(2, 40)
    output = model(
        torch.randn(2, 8, 128),
        torch.zeros(2, 32, 17, 3),
        torch.zeros(2, 32, dtype=torch.bool),
        torch.zeros(2, 32),
        baseline,
    )
    torch.testing.assert_close(output, baseline)


def test_native_cache_does_not_interpolate_or_repeat_frames(tmp_path: Path) -> None:
    depth = tmp_path / "Depth_Color"
    skeleton = tmp_path / "Skeleton" / "predictions"
    depth.mkdir(parents=True)
    skeleton.mkdir(parents=True)
    for frame in (1, 2, 3):
        (depth / f"frame_{frame}_Color.png").touch()
        points = np.zeros((17, 3), dtype=np.float32)
        points[:, 0] = np.arange(17) + frame
        scores = np.ones(17, dtype=np.float32)
        (skeleton / f"frame_{frame}.json").write_text(
            '[{"keypoints": '
            + repr(points.tolist()).replace("'", '"')
            + ', "keypoint_scores": '
            + repr(scores.tolist())
            + "}]",
            encoding="utf-8",
        )
    _, value, mask, positions, length = build_clip(
        (0, depth, skeleton.parent, 8)
    )
    assert length == 3
    assert mask.tolist() == [True, True, True, False, False, False, False, False]
    np.testing.assert_allclose(positions[:3], [0.0, 0.5, 1.0])
    assert not np.any(value[3:])
