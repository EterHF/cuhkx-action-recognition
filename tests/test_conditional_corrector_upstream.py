import numpy as np

from yolo_r2plus1d.strict_v3.training.conditional_corrector_upstream import (
    FOLDS,
    baseline_logits,
    visual_output_scale,
)


def test_nested_fold_groups_are_disjoint_and_cover_all_subjects() -> None:
    groups = [set(users) for users in FOLDS.values()]
    assert set.union(*groups) == set(range(1, 10)) | set(range(16, 25))
    assert all(
        not left.intersection(right) for i, left in enumerate(groups) for right in groups[i + 1 :]
    )


def test_baseline_applies_visual_release_scale() -> None:
    visual = np.zeros((1, 40), dtype=np.float32)
    visual[0, 3] = 8.0
    temporal = np.zeros_like(visual)
    temperatures = {"fusion": 1.0, "visual": 1.0, "temporal": 1.0}
    weights = {"fusion": 0.0, "visual": 1.0, "temporal": 0.0}
    scaled = baseline_logits(visual, temporal, temperatures, weights, 0.5)
    unscaled = baseline_logits(visual, temporal, temperatures, weights, 1.0)
    assert scaled.argmax(1).item() == 3
    assert np.max(scaled) < np.max(unscaled)


def test_visual_scale_uses_release_manifest_blend_contract() -> None:
    assert visual_output_scale({"blend": {"visual_package_output_scale": 0.5}}) == 0.5
