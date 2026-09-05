import numpy as np

from yolo_r2plus1d.strict_v3.evaluation.temporal_visual_equal import (
    equal_tempered_logits,
)
from yolo_r2plus1d.strict_v3.release.blend import apply_gate


def test_gate_is_finite_and_normalized() -> None:
    logits = np.arange(80, dtype=np.float64).reshape(2, 40)
    branches = {
        name: logits + offset
        for offset, name in enumerate(("fusion", "visual", "temporal"))
    }
    fused, weights = apply_gate(
        branches,
        {"fusion": 1.1, "visual": 1.5, "temporal": 1.2},
        {"fusion": 0.11, "visual": 0.22, "temporal": 0.67},
    )
    assert fused.shape == (2, 40)
    assert np.isfinite(fused).all()
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)


def test_gate_rejects_invalid_quality_contract() -> None:
    branches = {name: np.zeros((1, 40)) for name in ("fusion", "visual", "temporal")}
    temperatures = {name: 1.0 for name in branches}
    weights = {name: 1 / 3 for name in branches}
    for floor, gamma in ((-0.1, 1.0), (1.1, 1.0), (0.25, 0.0)):
        try:
            apply_gate(
                branches,
                temperatures,
                weights,
                quality_floor=floor,
                quality_gamma=gamma,
            )
        except ValueError:
            continue
        raise AssertionError("invalid quality contract was accepted")


def test_equal_tempered_logits_uses_fixed_half_weights() -> None:
    visual = np.array([[2.0, 4.0]], dtype=np.float32)
    temporal = np.array([[3.0, 9.0]], dtype=np.float32)
    observed = equal_tempered_logits(visual, temporal, 2.0, 3.0)
    np.testing.assert_allclose(observed, [[1.0, 2.5]])
